import time
from typing import Dict, Any

import numpy as np
import os
import pandas as pd

from src.questioner import ask_question
from src.CBT import run_cbt
from src.utils.config_loader import (
    ITEM_N_STATES,
    NUMBER_QUESTIONS,
    GAMMA,
    ALPHA,
    QUESTION_LIB_FILENAME,
    SUBJECT_ID,
    DATA_DIR,
)
from src.utils.io_question_lib import load_question_lib, save_question_lib, generate_results
from src.utils.io_record import init_record
from src.utils.rl_qtables import (
    initialize_q_table,
    choose_action,
    get_env_feedback,
    initialize_question_mask,
    initialize_question_table,
)
# Set up logger for this module
from src.utils.log_util import get_logger
logger = get_logger("HandlerRL")

class HandlerRL:
    """
    Top-level RL workflow coordinator.
    Handles the main reinforcement learning loop for question selection and evaluation.
    All file I/O is performed via utility modules.
    """

    def __init__(self):
        # Stores the last question asked to the user
        self.last_question: str = " "
        # Stores all user responses for later result generation
        self.new_response: list = []
        # The main question library loaded from file
        self.question_lib: Dict[str, Any] = {}
        # Masks for available questions per item (to avoid repeats)
        self.all_question_mask: Dict[int, Any] = {}
        # Q-tables for each question set (per item)
        self.all_question_q_table: Dict[int, Any] = {}
        # Q-table for item selection (top-level RL)
        self.item_q_table = None

    def setup(self):
        """
        Initialize records, load question library, and set up Q-tables and masks.
        """
        logger.info("Initializing RL handler setup: loading records and question library.")
        init_record()
        self.question_lib = load_question_lib(QUESTION_LIB_FILENAME)
        # Define possible actions for item selection (as string indices)
        item_actions = ['{0}'.format(e) for e in np.arange(0, ITEM_N_STATES)]
        # Initialize masks and Q-tables for all questions and items
        self.all_question_mask = initialize_question_mask(NUMBER_QUESTIONS)
        self.all_question_q_table = initialize_question_table(NUMBER_QUESTIONS)
        self.item_q_table = initialize_q_table(ITEM_N_STATES, item_actions)
        self.item_actions = item_actions
  
        # Load persistent Q tables (if exist)
        qdir = os.path.join(DATA_DIR, "q_tables")
        if os.path.exists(os.path.join(qdir, f"item_qtable_{SUBJECT_ID}.csv")):
            self.item_q_table = pd.read_csv(os.path.join(qdir, f"item_qtable_{SUBJECT_ID}.csv"), index_col=0)
        logger.info(f"Loaded item Q table for subject {SUBJECT_ID}.")
        
        logger.info("RL handler setup complete.")

    def run(self):
        """
        Main RL loop for the entire screening process.
        Iteratively selects items and asks questions using RL, updating Q-tables and saving results.
        """
        logger.info("Starting main RL screening process.")
        self.setup()
        new_q_table = self.item_q_table.copy()
        S = 0  # Start state for item RL
        is_terminated = False
        # Mask for available items (first item is always available)
        item_mask = [0] + [1] * (ITEM_N_STATES - 1)
        while not is_terminated:
            # Select an item to ask about using RL policy
            A = choose_action(S, self.item_q_table, item_mask, ITEM_N_STATES, self.item_actions)
            # Mark this item as used
            item_mask[int(A)] = 0
            # Ask questions for the selected item
            openai_res, DLA_terminate, last_question_updated = ask_question(self.question_lib, int(A))
            self.last_question = last_question_updated
            # Get next state and reward for item RL
            S_, R = get_env_feedback(S, A, openai_res, DLA_terminate, item_mask)
            # Q-learning update for item Q-table
            q_predict = self.item_q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * self.item_q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            new_q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            # If the DLA process signals termination, end the loop and save results
            if DLA_terminate == 1:
                logger.info("DLA process signaled termination. Saving question library and ending session.")
                is_terminated = True
                save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
                save_question_lib(save_filename, self.question_lib)
                # log_question("Goodbye. We will do the screening in another time. 886")
                logger.info("Goodbye. We will do the screening in another time. 886")        # Save results if terminated
        if is_terminated:
            logger.info("Session terminated. Saving question library and generating results.")
            save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
            save_question_lib(save_filename, self.question_lib)
            
            # Save Q tables (in parallel with existing results)
            logger.info("Saving Q tables.")
            qdir = os.path.join(DATA_DIR, "q_tables")
            self.item_q_table = new_q_table
            if not os.path.exists(qdir):
                os.makedirs(qdir, exist_ok=True)
            self.item_q_table.to_csv(os.path.join(qdir, f"item_qtable_{SUBJECT_ID}.csv"))

        # Run CBT after the screening loop concludes
        run_cbt(self.question_lib)
        # Persist question_lib again to capture CBT notes
        save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
        save_question_lib(save_filename, self.question_lib)

        # Generate final results for this session
        generate_results(self.question_lib, self.new_response)
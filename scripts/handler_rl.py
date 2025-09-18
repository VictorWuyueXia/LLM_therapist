import time
import logging
from typing import Tuple, Dict, Any

import numpy as np

from scripts.config_loader import (
    ITEM_N_STATES,
    NUMBER_QUESTIONS,
    GAMMA,
    ALPHA,
    QUESTION_LIB_FILENAME,
)
from scripts.utils.io_question_lib import load_question_lib, save_question_lib, generate_results
from scripts.utils.io_record import log_question, get_answer, get_resp_log, init_record
from scripts.utils.rl_qtables import (
    initialize_q_table,
    choose_action,
    get_env_feedback,
    initialize_question_mask,
    initialize_question_table,
)
from scripts.utils.response_bridge import get_openai_resp
from scripts.text_generators import (
    generate_change,
    generate_change_positive,
    generate_change_negative,
    generate_synonymous_sentences,
    generate_therapist_chat,
)
from scripts.utils.log_util import get_logger
from scripts.questioner import classify_segments, evaluate_result_core

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
        logger.info("RL handler setup complete.")

    def evaluate_result(self, DLA_result, S, question_A, user_input, original_question_asked):
        """
        Evaluate the result of a user's response to a question.
        Updates the question library and last question as needed.
        Optionally generates a therapist response and logs notes.
        """
        # Call the core evaluation logic
        valid, terminate, last_q, updated = evaluate_result_core(
            [(lbl, sc) for lbl, sc in DLA_result], S, question_A, user_input, original_question_asked, self.question_lib
        )
        self.question_lib = updated
        # Update last_question if a new one is provided
        self.last_question = last_q or self.last_question
        if last_q:
            # Log the last question and get a follow-up response
            log_question(last_q)
            follow_up = get_resp_log()
            # Optionally generate a therapist response based on last question and follow-up
            therapist_resp = generate_therapist_chat((last_q + " " + follow_up).strip())
            self.last_question = therapist_resp
            # Record notes for this question/response interaction
            note_resp = [
                "original_question: " + original_question_asked,
                "original_resp: " + (user_input[0] if user_input else ""),
                "followup_resp: " + follow_up,
                "therapist_resp: " + therapist_resp,
            ]
            self.question_lib[str(S)][str(question_A)]["notes"].append(note_resp)
        return valid, terminate, self.last_question, self.question_lib

    def ask_question(self, S: int) -> Tuple[float, int, str]:
        """
        Handles the RL loop for asking questions within a given item (S).
        Returns the total reward, termination flag, and the last question asked.
        """
        logger.info(f"Starting question RL loop for item S={S}.")
        question_reward = []
        DLA_terminate = 0
        # If there are multiple questions for this item, use RL to select among them
        if NUMBER_QUESTIONS[S] > 1:
            # Prepare actions, Q-table, and mask for this item
            question_actions = [str(element) for element in np.arange(NUMBER_QUESTIONS[S] + 1)]
            question_q_table = self.all_question_q_table[S].copy()
            question_mask = self.all_question_mask[S]
            new_question_q_table = question_q_table.copy()
            is_terminated = False
            number_of_states = NUMBER_QUESTIONS[S] + 1
            question_S = 0  # Start state for question RL
            while not is_terminated:
                # Select an action (question) using the RL policy
                question_A = choose_action(question_S, question_q_table, question_mask, number_of_states, question_actions)
                # Mark this question as used
                question_mask[int(question_A)] = 0
                # If this question has not been answered yet
                if len(self.question_lib[str(S)][str(question_A)]["score"]) == 0:
                    # Randomly select a question variant if multiple are available
                    number_of_questions = len(self.question_lib[str(S)][str(question_A)]["question"])
                    choice_of_question = np.random.randint(number_of_questions)
                    question_text = self.question_lib[str(S)][str(question_A)]["question"][choice_of_question]
                    # With small probability, generate a synonymous version of the question
                    if np.random.uniform() > 0.95:
                        question_text = generate_synonymous_sentences(question_text)
                    # Prepare the full question to ask (including context)
                    question_text_ask = self.last_question + "  " + question_text
                    log_question(question_text_ask)
                    # Get user input for this question
                    _ , user_input = get_answer()
                    # Classify the user response
                    DLA_result = [[label, score] for (label, score) in classify_segments(user_input)]
                    # Evaluate the result and update state
                    valid, DLA_terminate, self.last_question, self.question_lib = self.evaluate_result(
                        DLA_result, S, question_A, user_input, question_text
                    )
                    # If the answer is not valid, keep asking until a valid answer is given or terminated
                    if valid == 0 and DLA_terminate == 0:
                        valid_loop = 0
                        while valid_loop < 1:
                            log_question(question_text_ask)
                            _ , user_input = get_answer()
                            DLA_result = [[label, score] for (label, score) in classify_segments(user_input)]
                            valid_loop, DLA_terminate, self.last_question, self.question_lib = self.evaluate_result(
                                DLA_result, S, question_A, user_input, question_text
                            )
                # Calculate the reward for this question based on scores
                all_score = self.question_lib[str(S)][str(question_A)]["score"]
                if not all_score:
                    question_openai_res = 0.0
                    # Debug log: if score is still empty after answer, print S, question_A, DLA_result for troubleshooting
                    logging.debug(f"[RL DEBUG] Empty score after answer: S={S}, question_A={question_A}, DLA_result={DLA_result}")
                else:
                    question_openai_res = np.mean(all_score)
                # Get the next state and reward from the environment
                question_S_, question_R = get_env_feedback(
                    question_S, question_A, question_openai_res, DLA_terminate, question_mask
                )
                question_reward.append(question_R)
                # Q-learning update for this question's Q-table
                q_predict = question_q_table.loc[question_S, question_A]
                if question_S_ != 'terminal':
                    q_target = question_R + GAMMA * question_q_table.iloc[question_S_, :].max()
                else:
                    q_target = question_R
                    is_terminated = True
                new_question_q_table.loc[question_S, question_A] += ALPHA * (q_target - q_predict)
                question_S = question_S_
                # If the DLA process signals termination, end the loop
                if DLA_terminate == 1:
                    logger.info(f"Question RL loop for item S={S} terminated by DLA signal.")
                    is_terminated = True
            # Save updated Q-table and mask for this item
            self.all_question_q_table[S] = new_question_q_table.copy()
            self.all_question_mask[S] = question_mask
        else:
            # If only one question for this item, ask it directly
            question_A = "1"
            if len(self.question_lib[str(S)][str(question_A)]["score"]) == 0:
                number_of_questions = len(self.question_lib[str(S)][str(question_A)]["question"])
                choice_of_question = np.random.randint(number_of_questions)
                question_text = self.question_lib[str(S)][str(question_A)]["question"][choice_of_question]
                if np.random.uniform() > 0.95:
                    question_text = generate_synonymous_sentences(question_text)
                question_text_ask = self.last_question + "  " + question_text
                log_question(question_text_ask)
                _ , user_input = get_answer()
                DLA_result = [[label, score] for (label, score) in classify_segments(user_input)]
                valid, DLA_terminate, self.last_question, self.question_lib = self.evaluate_result(
                    DLA_result, S, question_A, user_input, question_text
                )
                if valid == 0 and DLA_terminate == 0:
                    valid_loop = 0
                    while valid_loop < 1:
                        log_question(question_text_ask)
                        _ , user_input = get_answer()
                        DLA_result = [[label, score] for (label, score) in classify_segments(user_input)]
                        valid_loop, DLA_terminate, self.last_question, self.question_lib = self.evaluate_result(
                            DLA_result, S, question_A, user_input, question_text
                        )
            all_score = self.question_lib[str(S)][str(question_A)]["score"]
            question_openai_res = np.mean(all_score) if all_score else 0.0
            question_reward.append(question_openai_res)

        # Return the total reward, termination flag, and last question
        logger.info(f"Finished question RL loop for item S={S}. Total reward: {float(sum(question_reward))}, DLA_terminate: {int(DLA_terminate)}")
        return float(sum(question_reward)), int(DLA_terminate), self.last_question

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
            openai_res, DLA_terminate, last_question_updated = self.ask_question(int(A))
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
                log_question("Goodbye. We will do the screening in another time. 886")
        # Save results if terminated
        if is_terminated:
            logger.info("Session terminated. Saving question library and generating results.")
            save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
            save_question_lib(save_filename, self.question_lib)
        # Generate final results for this session
        generate_results(self.question_lib, self.new_response)

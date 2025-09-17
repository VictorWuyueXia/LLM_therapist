# scripts/rl_session.py
# Main RL session module for the LLM therapist application
# This module handles the reinforcement learning session logic for therapeutic conversations

# Standard library imports
import time
import numpy as np
import pandas as pd

# Configuration imports - hyperparameters and constants
from scripts.config import ITEM_N_STATES, NUMBER_QUESTIONS, GAMMA, ALPHA, QUESTION_LIB_FILENAME

# I/O module imports for question library and session recording
from scripts.io_question_lib import load_question_lib, save_question_lib, generate_results
from scripts.io_record import log_question, get_answer, get_resp_log, init_record

# Reinforcement learning components - Q-tables and action selection
from scripts.rl_qtables import initialize_q_table, choose_action, get_env_feedback, initialize_question_mask

# Response processing bridge to OpenAI API
from scripts.response_bridge import get_openai_resp

# Text generation utilities for therapeutic responses
from scripts.text_generators import (
    generate_change, 
    generate_change_positive, 
    generate_change_negative, 
    generate_synonymous_sentences, 
    generate_therapist_chat
)

# initialize question q table
from scripts.rl_qtables import initialize_q_table, choose_action, get_env_feedback, initialize_question_mask, initialize_question_table

# for MI/CBT
import json, random
from word2number import w2n

# Logger
from scripts.log_util import get_logger
logger = get_logger("RLSession")

# Global session state variables
last_question = " "  # Stores the last question asked to maintain conversation context
new_response = []    # Accumulates user responses during the session

def remove_duplicate_items(DLA_result, user_input):
    dla_res, user_res = [], []
    for i in range(0, len(DLA_result)):
        if DLA_result[i] not in dla_res:
            dla_res.append(DLA_result[i]); user_res.append(user_input[i])
        else:
            idx = dla_res.index(DLA_result[i]); user_res[idx] += " "+user_input[i]
    return dla_res, user_res

def evaluate_result(DLA_result, S, question_A, user_input, original_question_asked, question_lib):
    """
    Evaluate the result of a user's answer to a question.
    - Use log_question/get_resp_log for interaction.
    - Use get_openai_resp and text generation modules.
    - Update last_question/new_response/question_lib in memory, not via file I/O.
    - Return (valid, DLA_terminate, last_question, question_lib).
    """
    global last_question
    last_question = " "
    therapist_resp = ""

    # If the DLA_result indicates the user did not understand the question, repeat the question
    if DLA_result[0][1] == "Question":
        logger.info("evaluate question check")
        question = question_lib[str(S)][str(question_A)]["question"][0]
        Question_text = "Sounds like you did not understand my question. Let me ask it again. " + question
        log_question(Question_text)
        # Get new answer from user (in-memory, not file)
        _ , user_input = get_answer()

    # Remove duplicate items from DLA_result and user_input
    DLA_result, user_input = remove_duplicate_items(DLA_result, user_input)
    valid = 0
    DLA_terminate = 0
    logger.info("check")

    logger.info("3")
    # Check if the first DLA_result is a direct string answer (yes/no/stop)
    if isinstance(DLA_result[0][1], str):
        logger.info("1 + There is direct yes/no/stop to the question")
        if DLA_result[0][1] == "Stop":
            DLA_terminate = 1
        else:
            logger.info("2")
            score = question_lib[str(S)][str(question_A)][DLA_result[0][1]]
            question_lib[str(S)][str(question_A)]["score"].append(score)
            valid = 1
            if score > 1:
                logger.info("If score > 1 for direct yes/no/stop")
                text = question_lib[str(S)][str(question_A)]["question"][0]
                if DLA_result[0][1] == "Yes":
                    text = generate_change_positive(text)
                else:
                    text = generate_change_negative(text)
                # Ask a follow-up question
                text_temp = generate_synonymous_sentences(" Can you tell me more about it?")
                question_text = "It seems that " + text + " " + text_temp
                log_question(question_text)
                logger.info("4")
                user_input_followup = get_resp_log()
                logger.info(("5", user_input_followup))
                therapist_resp = generate_therapist_chat(text + " " + user_input_followup)
                logger.info(("therapist_resp in evaluate Y/N", therapist_resp))
                logger.info("6")
                last_question = therapist_resp

                original_resp = "original_resp: " + user_input[0]
                followup_resp = "followup_resp: " + user_input_followup
                original_question_asked_record = "original_question: " + original_question_asked
                note_resp = [original_question_asked_record, original_resp, followup_resp, "therapist_resp: " + therapist_resp]
                question_lib[str(S)][str(question_A)]["notes"].append(note_resp)
            else:
                original_resp = "original_resp: " + user_input[0]
                original_question_asked_record = "original_question: " + original_question_asked
                note_resp = [original_question_asked_record, original_resp]
                question_lib[str(S)][str(question_A)]["notes"].append(note_resp)

    # Continue to process all DLA_result items for more complex answers
    question_label = question_lib[str(S)][str(question_A)]["label"]
    error_count = 0
    rephrase_count = 0
    for i in range(0, len(DLA_result)):
        logger.info(("Evaluate DLA: ", DLA_result[i]))

        # If the answer is a string and is "Stop", terminate
        if isinstance(DLA_result[i][1], str):
            if DLA_result[i][1] == "Stop":
                DLA_terminate = 1
                valid = 1
            return valid, DLA_terminate, last_question, question_lib
        else:
            label = DLA_result[i][0]
            logger.info(("label", label))
            # If the answer is an integer and not 99, process the score
            if isinstance(DLA_result[i][1], int) and DLA_result[i][1] != 99:
                if DLA_result[i][0].lower() == question_label.lower():
                    valid = 1
                logger.info(valid)
                score = DLA_result[i][1]
                logger.info(("Score: ", score))
                if score > 1:
                    text = user_input[i]
                    text = generate_change(text).lower()
                    # Ask a follow-up question
                    question_text = "You mentioned that " + text + " Can you tell me more?"
                    log_question(question_text)
                    logger.info("7")
                    user_input_followup = get_resp_log()
                    logger.info(("8", user_input_followup))
                    therapist_resp = generate_therapist_chat(text + " " + user_input_followup)
                    logger.info(("therapist_resp in evaluate", therapist_resp))
                    logger.info("9")

                # Update the question_lib with the score and notes (in-memory)
                logger.info("check1")
                item_number = DLA_result[i][0].split("_")[1]
                logger.info(item_number)
                if int(item_number) == 21:
                    item_number = 19
                logger.info(len(question_lib[str(item_number)]))
                if len(question_lib[str(item_number)]) == 1:
                    logger.info("Only one question in this item.")
                    if question_lib[str(item_number)]["1"]["label"].lower() == label.lower():
                        question_number = 1
                else:
                    for ind in range(1, len(question_lib[str(item_number)]) + 1):
                        if question_lib[str(item_number)][str(ind)]["label"].lower() == label.lower():
                            question_number = ind

                logger.info(("item_number, question_number, valid:", item_number, question_number, valid))
                question_lib[str(item_number)][str(question_number)]["score"].append(score)
                if score > 1:
                    original_resp = "original_resp: " + user_input[i]
                    followup_resp = "followup_resp: " + user_input_followup
                    original_question_asked_record = "original_question: " + original_question_asked
                    note_resp = [original_question_asked_record, original_resp, followup_resp, "therapist_resp: " + therapist_resp]
                    question_lib[str(item_number)][str(question_number)]["notes"].append(note_resp)
                else:
                    original_resp = "original_resp: " + user_input[i]
                    original_question_asked_record = "original_question: " + original_question_asked
                    note_resp = [original_question_asked_record, original_resp, original_question_asked_record]
                    question_lib[str(item_number)][str(question_number)]["notes"].append(note_resp)

                    # Special case: if label is DLA_21_sports and score is 0, also update hobbies
                    if label == "DLA_21_sports" and score == 0:
                        question_lib[str(11)][str(1)]["notes"].append(note_resp)
                        question_lib[str(11)][str(1)]["score"].append(score)
                        if int(S) == 11 and int(question_A) == 1:
                            valid = 1

    # Update last_question with the therapist response
    last_question = therapist_resp
    logger.info(('last_question in evaluate_result: ', last_question))

    return valid, DLA_terminate, last_question, question_lib
            

def ask_question(S, all_question_mask, all_question_q_table, question_lib):
    """
    Ask questions for a given item S using RL logic.
    Handles both single-question and multi-question items.
    Returns the sum of rewards (openai_res_sum), DLA_terminate flag, last_question, and updated question_lib.
    """
    global last_question
    logger.info(("Item number: ", S))

    question_S = 0  # State for question-level RL
    question_A_order = []  # Track the order of actions taken
    question_reward = []   # Track rewards for each question
    DLA_terminate = 0      # Flag for early termination

    # Multi-question item
    if NUMBER_QUESTIONS[S] > 1:
        question_actions = [str(element) for element in np.arange(NUMBER_QUESTIONS[S] + 1)]
        question_q_table = all_question_q_table[S].copy()
        question_mask = all_question_mask[S]
        new_question_q_table = question_q_table.copy()
        is_terminated = False
        number_of_states = NUMBER_QUESTIONS[S] + 1

        while not is_terminated:
            logger.info("*" * 99)
            # Choose next question to ask using RL policy
            question_A = choose_action(question_S, question_q_table, question_mask, number_of_states, question_actions)
            logger.info(("Question Action", question_A))
            question_mask[int(question_A)] = 0  # Mark this question as asked
            logger.info(("question_mask", question_mask))
            logger.info(("score in log", question_lib[str(S)][str(question_A)]["score"]))

            # Only ask if this question has not been answered before
            if len(question_lib[str(S)][str(question_A)]["score"]) == 0:
                number_of_questions = len(question_lib[str(S)][str(question_A)]["question"])
                choice_of_question = np.random.randint(number_of_questions)
                question_text = question_lib[str(S)][str(question_A)]["question"][choice_of_question]
                question_label = question_lib[str(S)][str(question_A)]["label"]

                # Occasionally generate a synonymous sentence
                if np.random.uniform() > 0.95:
                    question_text = generate_synonymous_sentences(question_text)

                logger.info(question_text)
                question_text_ask = last_question + "  " + question_text
                logger.info((last_question, question_text))

                # Log the question to the record
                log_question(question_text_ask)

                # Get the user's answer
                _ , user_input = get_answer()
                logger.info((DLA_result, user_input))
                
                # classify and score each segment, fill DLA_result
                DLA_result = []
                for seg in user_input:
                    if not seg:
                        continue
                    category, score = get_openai_resp(seg)
                    DLA_result.append([category, score])

                # Evaluate the answer
                valid, DLA_terminate, last_question, question_lib = evaluate_result(
                    DLA_result, S, question_A, user_input, question_text, question_lib
                )
                logger.info(last_question)
                logger.info(("DLA_result:", DLA_result))
                logger.info(("valid:", valid))
                logger.info(("DLA_terminate:", DLA_terminate))

                # If answer is not valid and not terminated, keep asking for clarification
                if valid == 0 and DLA_terminate == 0:
                    valid_loop = 0
                    while valid_loop < 1:
                        question_to_ask = (
                            last_question +
                            "Sorry. Do you mind rephrasing your answer in a different way. Please make sure you answer the question I ask. "
                        )
                        if np.random.uniform() > 0.5:
                            question_to_ask += (
                                "And try to answer my question in a complete sentence and in a concise and deterministic way if you can."
                            )
                        log_question(question_text_ask)
                        _ , user_input = get_answer()
                        
                        # classify and score each segment, fill DLA_result
                        DLA_result = []
                        for seg in user_input:
                            if not seg:
                                continue
                            category, score = get_openai_resp(seg)
                            DLA_result.append([category, score])
                            
                        valid_loop, DLA_terminate, last_question, question_lib = evaluate_result(
                            DLA_result, S, question_A, user_input, question_text, question_lib
                        )
                logger.info(("last question", last_question))
            else:
                # If already answered, just use the historical reward
                logger.info(("Already answered, get reward from history"))

            # Calculate reward for this question
            all_score = question_lib[str(S)][str(question_A)]["score"]
            question_openai_res = np.mean(all_score)

            # RL environment feedback: get next state and reward
            question_S_, question_R = get_env_feedback(
                question_S, question_A, question_openai_res, DLA_terminate, question_mask
            )
            question_A_order.append(question_A)
            question_reward.append(question_R)
            q_predict = question_q_table.loc[question_S, question_A]

            # Q-learning update
            if question_S_ != 'terminal':
                q_target = question_R + GAMMA * question_q_table.iloc[question_S_, :].max()
            else:
                q_target = question_R
                is_terminated = True  # End of episode

            logger.info(("q_target", q_target, "q_predict", q_predict))
            new_question_q_table.loc[question_S, question_A] += ALPHA * (q_target - q_predict)
            logger.info(("new_question_q_table after update", new_question_q_table))
            question_S = question_S_  # Move to next state

            if DLA_terminate == 1:
                is_terminated = True

        # Update the Q-table and mask for this item
        question_q_table = new_question_q_table.copy()
        logger.info(("question_q_table after update", question_q_table))
        all_question_q_table[S] = question_q_table.copy()
        logger.info(("all_question_q_table[S]", all_question_q_table[S]))
        all_question_mask[S] = question_mask
        logger.info("_" * 99)

    # Single-question item
    else:
        question_A = "1"  # Only one question for this item

        # Only ask if not answered before
        if len(question_lib[str(S)][str(question_A)]["score"]) == 0:
            number_of_questions = len(question_lib[str(S)][str(question_A)]["question"])
            choice_of_question = np.random.randint(number_of_questions)
            question_text = question_lib[str(S)][str(question_A)]["question"][choice_of_question]
            question_label = question_lib[str(S)][str(question_A)]["label"]

            # Occasionally generate a synonymous sentence
            if np.random.uniform() > 0.95:
                question_text = generate_synonymous_sentences(question_text)

            logger.info(question_text)
            question_text_ask = last_question + "  " + question_text

            # Log the question to the record
            log_question(question_text_ask)

            # Get the user's answer
            _ , user_input = get_answer()
            
            # classify and score each segment, fill DLA_result
            DLA_result = []
            for seg in user_input:
                if not seg:
                    continue
                category, score = get_openai_resp(seg)
                DLA_result.append([category, score])
                
            valid, DLA_terminate, last_question, question_lib = evaluate_result(
                DLA_result, S, question_A, user_input, question_text, question_lib
            )

            logger.info(("DLA_result:", DLA_result))
            logger.info(("valid:", valid))
            logger.info(("DLA_terminate:", DLA_terminate))

            # If answer is not valid and not terminated, keep asking for clarification
            if valid == 0 and DLA_terminate == 0:
                valid_loop = 0
                while valid_loop < 1:
                    question_to_ask = (
                        last_question +
                        "Sorry. Do you mind rephrasing your answer in a different way. Please make sure you answer the question I ask. "
                    )
                    if np.random.uniform() > 0.5:
                        question_to_ask += (
                            "And try to answer my question in a complete sentence and in a concise and deterministic way if you can."
                        )
                    log_question(question_text_ask)
                    _ , user_input = get_answer()
                    
                    # classify and score each segment, fill DLA_result
                    DLA_result = []
                    for seg in user_input:
                        if not seg:
                            continue
                        category, score = get_openai_resp(seg)
                        DLA_result.append([category, score])
                    
                    valid_loop, DLA_terminate, last_question, question_lib = evaluate_result(
                        DLA_result, S, question_A, user_input, question_text, question_lib
                    )
            logger.info(("last question", last_question))
        else:
            # If already answered, just use the historical reward
            logger.info(("Already answered, get reward from history"))

        # Calculate reward for this question
        all_score = question_lib[str(S)][str(question_A)]["score"]
        question_openai_res = np.mean(all_score)
        question_reward.append(question_openai_res)

    logger.info(("question_reward: ", question_reward))
    openai_res_sum = sum(question_reward)
    logger.info(("DLA_terminate", DLA_terminate))

    return openai_res_sum, DLA_terminate, last_question, question_lib, all_question_q_table


def proceed_MI_CBT(save_filename, question_lib):
    """
    Proceed with Motivational Interviewing and Cognitive Behavioral Therapy (MI-CBT) session.
    This function analyzes the user's responses and conducts a therapeutic conversation
    based on identified issues and strengths.
    
    Args:
        save_filename (str): Path to the saved question library results file
        question_lib (dict): Current question library with user responses
    """
    # Load the question library results from file
    f = open(save_filename)
    question_lib_result = json.load(f)
    
    # Prepare dimensions analysis - identify issues and strengths
    issue_dimension, good_dimension = prepare_for_MI_CBT(save_filename)
    
    # Initialize category and score variables for OpenAI response processing
    category = "category"
    score = "score"
    
    # Case 1: No issues identified - provide positive feedback and closure
    if len(issue_dimension) == 0:
        Q5 = "It seems like you are doing pretty well. You work well in dimensions including: " 
        # Randomly select 3 good dimensions to highlight
        sample_good_dimension = random.sample(range(len(good_dimension)), 3)
        Q5 += good_dimension[sample_good_dimension[0]][2] + ", " + good_dimension[sample_good_dimension[1]][2] + ", and " + good_dimension[sample_good_dimension[2]][2]
        Q5 += ". Please reach out to your primary care or your therapist if you have further problems or emergencies. "
        Q5 += "Goodbye. We will followup later. 886"
        logger.info(("Q5:", Q5))
        log_question(Q5)
    
    # Case 2: Issues identified - conduct CBT intervention
    else:
        # Q1: Present identified issues and ask user to choose which to work on
        Q1 = "Thank you for answering all the questions. "
        # Occasionally use synonymous phrasing for variety
        if (np.random.uniform() > 0.95):
            Q1 = generate_synonymous_sentences(Q1)
        Q1 += "According to your previous responses, you have issues in: "
        # List all identified issue dimensions
        for i in range(0, len(issue_dimension)):
            Q1 += str(i+1) + ": " + issue_dimension[i][2] + ", "
        Q1 += "Which dimension do you want to work on today? Please speak out the dimension number, for example, 1."
        logger.info(("Q1:", Q1))
        
        log_question(Q1)
        
        # Get user's choice of dimension to work on
        user_dimension = str(get_resp_log())
        user_dimension = user_dimension.replace(".0", "").replace(".", "")
        
        logger.info(("user_dimension1:", user_dimension))
        
        # Process user response using OpenAI to understand intent
        try:
            category, score = get_openai_resp(user_dimension)
        except:
            pass
        
        Q2 = " "
        
        # Handle case where user didn't understand the question
        if score == "Question":
            logger.info("check")
            Q1 = "It seems like you don't understand me well. Let me repeat my question. According to your previous responses, you have issues in: "
            for i in range(0, len(issue_dimension)):
                Q1 += str(i+1) + ": " + issue_dimension[i][2] + ", "
            Q1 += "Which dimension do you want to work on today? Please speak out the dimension number, for example, 1."
            
            log_question(Q1)
            user_dimension = str(get_resp_log())
            logger.info(("user_dimension2:", user_dimension))
            user_dimension = user_dimension.replace(".0", "").replace(".", "")
            category, score = get_openai_resp(user_dimension)
            
            # If still unclear, randomly select a dimension
            if score == "Question":
                logger.info("Question again.")
                user_dimension = random.choice(range(len(issue_dimension)))
                issue_dimension_number = int(user_dimension)
                logger.info(("user_dimension2:", user_dimension))
                Q2 = "I will pick a dimension to work on today."
        
        # Convert user input to dimension number
        try:
            issue_dimension_number = w2n.word_to_num(user_dimension)
            issue_dimension_number = int(user_dimension)
        except:
            try:
                issue_dimension_number = w2n.word_to_num(user_dimension)
            except:
                logger.info("fail to get issue_dimension_number")
                user_dimension = random.choice(range(len(issue_dimension)))
                issue_dimension_number = int(user_dimension)
                Q2 = "I have problem getting the dimension you want to work on. So I pick a dimension to work on today. "
        
        # Validate dimension number is within range
        if issue_dimension_number > len(issue_dimension) + 1:
            user_dimension = random.choice(range(len(issue_dimension)))
            issue_dimension_number = int(user_dimension)
            Q2 = "Looks like you are doing OK with the dimension you chose. So I pick an issue dimension to work on today. "
        
        # Q2: Get details about the chosen dimension and ask about unhelpful thoughts
        item_number, question_number, summary_original_response, summary_followup_response = get_dimension_to_work(save_filename, issue_dimension, issue_dimension_number)
        
        Q2 += "Let's work on Dimension " + str(issue_dimension_number) + ". From my record, your responses to my question in this dimension are: "
        
        # Include original responses in the summary
        for i in range(0, len(summary_original_response)):
            Q2 += summary_original_response[i] + " and "
        
        # Include follow-up responses in the summary
        for i in range(0, len(summary_followup_response)):
            Q2 += summary_followup_response[i] + " and "
        
        Q2 = Q2[:-5]  # Remove trailing " and "
        Q2 += " Can you try to identify any unhelpful thoughts you have that contribute to this situation?"
        logger.info(("Q2:", Q2))
        log_question(Q2)
        
        # Get user's identification of unhelpful thoughts
        user_thought = str(get_resp_log())
        logger.info(("user_thought:", user_thought))
        category, score = get_openai_resp(user_thought)
        
        # Handle case where user didn't understand the thought identification question
        if score == "Question":
            Q2 = "It seems like you can not get my question. Let me repeat my question. We are working on Dimension " + str(issue_dimension_number) + ". From my record, you mentioned: "
            
            for i in range(0, len(summary_original_response)):
                Q2 += summary_original_response[i] + " and "
            
            for i in range(0, len(summary_followup_response)):
                Q2 += summary_followup_response[i] + " and "
            
            Q2 = Q2[:-5]  # Remove trailing " and "
            Q2 += " Can you try to identify any unhelpful thoughts you have that contribute to this situation?"
            logger.info(("Q2:", Q2))
            log_question(Q2)
            user_thought = get_resp_log()
        
        # Q3: Ask user to challenge their identified unhelpful thoughts
        Q3 = "Can you challenge your thought?"
        log_question(Q3)
        logger.info(("Q3:", Q3))
        user_challenge = str(get_resp_log())
        logger.info(("user_challenge:", user_challenge))
        
        # Q4: Help user develop alternative thinking patterns
        rephrase_challenge = generate_change(user_challenge).lower()
        Q4 = "You mentioned that: " + rephrase_challenge + " to challenge your thought. Now, what is another way of thinking about this situation?"
        logger.info(("Q4:", Q4))
        log_question(Q4)
        user_new_way = get_resp_log()
        logger.info(("user_new_way:", user_new_way))
        
        # Q5: Provide positive closure and highlight strengths
        Q5 = "Congratulation, you figure out a way for yourself. You also work well in dimensions including: "
        # Randomly select 3 good dimensions to highlight
        sample_good_dimension = random.sample(range(len(good_dimension)), 3)
        Q5 += good_dimension[sample_good_dimension[0]][2] + ", " + good_dimension[sample_good_dimension[1]][2] + ", and " + good_dimension[sample_good_dimension[2]][2]
        Q5 += ". Please reach out to your primary care or your therapist if you have further problems or emergencies. "
        Q5 += "Goodbye. We will followup later. 886"
        logger.info(("Q5:", Q5))
        log_question(Q5)
        
        # Record CBT session notes for future reference
        CBT_notes = [
            'CBT_Dimension: ' + str(user_dimension),
            "CBT_unhelpful_thought: " + str(user_thought),
            "CBT_challenge: " + str(user_challenge),
            "CBT_new_way:" + str(user_new_way)
        ]
        
        # Update question library with CBT notes
        f = open(save_filename)
        question_lib_result = json.load(f)
        question_lib_result[str(item_number)][str(question_number)]["notes"].append(CBT_notes)
        
        # Save updated question library back to file
        with open(save_filename, 'w') as f:
            json.dump(question_lib_result, f)
        

def run_session():
    init_record()
    question_lib = load_question_lib(QUESTION_LIB_FILENAME)
    item_actions = ['{0}'.format(e) for e in np.arange(0, ITEM_N_STATES)]
    all_question_mask = initialize_question_mask(NUMBER_QUESTIONS)
    all_question_q_table = initialize_question_table(NUMBER_QUESTIONS)
    item_q_table = initialize_q_table(ITEM_N_STATES, item_actions)

    global last_question, new_response
    last_question = " "; new_response = []

    new_q_table = item_q_table.copy()
    S = 0; is_terminated = False
    item_mask = [0] + [1] * (ITEM_N_STATES-1)

    while not is_terminated:
        A = choose_action(S, item_q_table, item_mask, ITEM_N_STATES, item_actions)
        item_mask[int(A)] = 0
        openai_res, DLA_terminate, last_question_updated, question_lib, all_question_q_table = \
            ask_question(int(A), all_question_mask, all_question_q_table, question_lib)
        last_question = last_question_updated
        S_, R = get_env_feedback(S, A, openai_res, DLA_terminate, item_mask)
        q_predict = item_q_table.loc[S, A]
        if S_ != 'terminal':
            q_target = R + GAMMA * item_q_table.iloc[S_, :].max()
        else:
            q_target = R; is_terminated = True
        new_q_table.loc[S, A] += ALPHA * (q_target - q_predict)
        S = S_
        if DLA_terminate == 1:
            is_terminated = True
            save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
            save_question_lib(save_filename, question_lib)
            log_question("Goodbye. We will do the screening in another time. 886")

    if is_terminated:
        save_filename = QUESTION_LIB_FILENAME.replace(".json", f"_{int(time.time())}.json")
        save_question_lib(save_filename, question_lib)
        if DLA_terminate != 1:
            proceed_MI_CBT(save_filename, question_lib)

    generate_results(question_lib, new_response)
from typing import List, Tuple, Dict, Any

import numpy as np

from src.utils.response_bridge import get_openai_resp
from src.utils.text_generators import (
    generate_change,
    generate_change_positive,
    generate_change_negative,
    generate_synonymous_sentences,
    generate_therapist_chat,
)

# Set up logger for this module
from src.utils.log_util import get_logger, log_question
from src.utils.io_record import get_answer, get_resp_log
logger = get_logger("Questioner")

from src.reflection_validation import rv_reasoner, rv_guide, rv_validation

def classify_segments(user_segments: List[str]) -> List[Tuple[str, int]]:
    """
    Classifies each user segment using the OpenAI response bridge.
    Returns a list of (label, score) tuples for each non-empty segment.
    """
    logger.info("Classifying user segments. Total segments: %d", len(user_segments))
    result = []
    for seg in user_segments:
        if not seg:
            # Skip empty segments
            continue
        label, score = get_openai_resp(seg)
        logger.debug("Segment classified: '%s' -> (label: %s, score: %s)", seg, label, str(score))
        result.append((label, score))
    logger.info("Classification complete. Results: %s", str(result))
    return result

def _if_valid_response(
    dla_result: List[Tuple[str, Any]],
    item_index: int,
    question_index: str,
    user_segments: List[str],
    original_question: str,
    question_lib: Dict[str, Any],
    ) -> Tuple[int, int, str, Dict[str, Any]]:
    """
    Unified logic: iterate over all labels in dla_result, 
    return as soon as an identifiable valid or command-like label is found.
    """
    last_question = ""
    if not dla_result:
        logger.info("No DLA result provided. Returning default values.")
        return 0, 0, last_question, question_lib

    question_label = question_lib[str(item_index)][str(question_index)]["label"]

    for i, (label, score_val) in enumerate(dla_result):
        # Normalize label for robust match
        label_norm = str(label).strip()
        score_norm = score_val
        logger.info(f"Processing dla_result entry: {label_norm}, {score_norm}")

        # Yes/No/Stop
        if label_norm in ["Yes", "No", "Stop"]:
            logger.info(f"Match special token: {label_norm}")
            if label_norm == "Stop":
                logger.info("Received 'Stop' label. Terminating evaluation.")
                return 1, 1, last_question, question_lib

            score = question_lib[str(item_index)][str(question_index)].get(label_norm, 99)
            question_lib[str(item_index)][str(question_index)]["score"].append(score)
            logger.info("Appended score %s for label %s to question_lib[%s][%s].", str(score), label_norm, str(item_index), str(question_index))

            if score > 1:
                text = question_lib[str(item_index)][str(question_index)]["question"][0]
                if label_norm == "Yes":
                    text = generate_change_positive(text)
                else:
                    text = generate_change_negative(text)
                text_temp = generate_synonymous_sentences(" Can you tell me more about it?")
                last_question = "It seems that " + text + " " + text_temp
            else:
                last_question = ""
            # Prepare note for follow-up, to be appended by caller after collecting follow-up
            original_resp = "original_resp: " + (user_segments[i] if i < len(user_segments) else user_segments[0])
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
            logger.debug("Appended note to question_lib[%s][%s]['notes'].", str(item_index), str(question_index))
            return 1, 0, last_question, question_lib

        # Valid response: Label matches question label & score in [0,1,2]
        if label_norm.lower() == str(question_label).lower() and score_norm in [0, 1, 2]:
            logger.info("Valid response: label matches and score is in [0,1,2]")
            question_lib[str(item_index)][str(question_index)]["score"].append(score_norm)
            if score_norm > 1:
                text = generate_change(user_segments[i]).lower() if i < len(user_segments) else ''
                last_question = "You mentioned that " + text + " Can you tell me more?"
            # Prepare note
            original_resp = "original_resp: " + (user_segments[i] if i < len(user_segments) else user_segments[0])
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
            logger.debug("Appended note to question_lib[%s][%s]['notes'].", str(item_index), str(question_index))
            return 1, 0, last_question, question_lib

        # Skip Maybe or Question, follow-up will be collected by caller
        if label_norm in ["Maybe", "Question"]:
            logger.info("Processing 'Maybe' or 'Question' token.")
            # return 0, 0, last_question, question_lib
            continue

    # If nothing matched, fallback: invalid response
    logger.info("No valid, yes, no, or stop label found in results. Marking as invalid response.")
    return 0, 0, last_question, question_lib

def evaluate_result(question_lib, DLA_result, S, question_A, user_input, original_question_asked):
    """
    Evaluate the result of a user's response to a question.
    Updates the question library and last question as needed.
    ReflectionValidation three steps（topic = the dimension label of the current question）
    """
    logger.info(f"Evaluating result for item {S}, question {question_A}.")
    # If valid response, update the question library and last question
    valid, terminate, last_q, updated = _if_valid_response(
        [(lbl, sc) for lbl, sc in DLA_result], S, question_A, user_input, original_question_asked, question_lib
    )
    question_lib = updated
    # Update last_question if a new one is provided
    last_question = last_q or last_question
    if last_q:
        # If valid response, log the last question and collect follow-up
        logger.info(f"Logging last question and collecting follow-up for item {S}, question {question_A}.")
        # Log the last question and get a follow-up response
        log_question(last_q)
        follow_up = get_resp_log()

        # ReflectionValidation three steps（topic = the dimension label of the current question）
        topic = question_lib[str(S)][str(question_A)]["label"]
        original_resp = user_input[0] if user_input else ""

        logger.info(f"Running ReflectionValidation reasoner for topic '{topic}'.")
        rv_decision_raw = rv_reasoner(topic, original_question_asked, original_resp, follow_up)
        # Simple parsing: extract '0/1'（no try/except, error directly thrown）
        rv_decision_token = "0" if "0" in rv_decision_raw else "1"
        logger.info(f"ReflectionValidation decision: {rv_decision_token}")

        # If not related (1), give guidance, recollect follow-up
        rv_guide_text = ""
        follow_up_0 = ""
        if rv_decision_token == "1":
            logger.info("Follow-up not related, generating guidance and recollecting follow-up.")
            follow_up_0 = follow_up
            rv_guide_text = rv_guide(topic, original_question_asked, original_resp, follow_up)
            log_question(rv_guide_text)
            follow_up = get_resp_log()

        # Empathic validation
        logger.info("Running ReflectionValidation empathic validation.")
        rv_validation_text = rv_validation(topic, original_question_asked, original_resp, follow_up)
        
        # Generate therapist response (store as context, do not prompt user)
        logger.info("Generating therapist response.")
        therapist_resp = generate_therapist_chat((last_q + " " + follow_up).strip())
        last_question = ""
        
        # Record notes (expand RV fields)
        logger.info("Recording notes for this question/response.")
        note_resp = [
            "original_question: " + original_question_asked,
            "original_resp: " + (user_input[0] if user_input else ""),
            "followup_resp: " + follow_up_0 if follow_up_0 else "followup_resp: " + follow_up,
            "rv_decision: " + rv_decision_token,
            ("rv_guide: " + rv_guide_text) if rv_guide_text else "rv_guide: ",
            "followup_resp_1: " + follow_up if follow_up_0 else "followup_resp_1: "
            "rv_validation: " + rv_validation_text,
            "therapist_resp: " + therapist_resp
        ]
        question_lib[str(S)][str(question_A)]["notes"].append(note_resp)
        
    return valid, terminate, last_question, question_lib

def ask_question(question_lib, S: int) -> Tuple[float, int, str]:
        """
        Handles the RL loop for asking questions within a given item (S).
        Returns the total reward, termination flag, and the last question asked.
        """
        logger.info(f"Starting question RL loop for item S={S}.")
        question_reward = []
        DLA_terminate = 0
        
        # If there is only one question for this item, ask it directly
        question_A = "1"
        # Check if the score list for this question is empty (i.e., not answered yet)
        if len(question_lib[str(S)][str(question_A)]["score"]) == 0:
            # Get the number of available question variants for this item
            number_of_questions = len(question_lib[str(S)][str(question_A)]["question"])
            # Randomly select one question variant to ask
            choice_of_question = np.random.randint(number_of_questions)
            question_text = question_lib[str(S)][str(question_A)]["question"][choice_of_question]
            # With probability, generate a synonymous version of the question
            if np.random.uniform() < 0.95:
                question_text = generate_synonymous_sentences(question_text)
            # Concatenate the last question (context) with the current question
            question_text_ask = question_text
            # Log the question being asked
            log_question(question_text_ask)
            # Get user input for the question
            _ , user_input = get_answer()
            # Classify the user response into DLA result segments
            DLA_result = [[label, score] for (label, score) in classify_segments(user_input)]
            # Evaluate the result and update state
            valid, DLA_terminate, last_question, question_lib = evaluate_result(
                DLA_result, S, question_A, user_input, question_text
            )
            # If the answer is invalid (valid == 0) and the process has not been terminated (DLA_terminate == 0), 
            # we may want to give the user a chance to clarify their response.
            # Only retry if DLA_result is empty or every (label, score) pair suggests NA or an unclassified response (score==99 or label=="NA").
            if valid == 0 and DLA_terminate == 0:
                # Retry asking the same question and get a new user response, regardless of label and score.
                log_question(question_text_ask)
                _ , user_input = get_answer()
                # Classify the new user response
                DLA_result = [[label, score] for (label, score) in classify_segments(user_input)]
                # Re-evaluate the new answer and update state accordingly
                valid, DLA_terminate, last_question, question_lib = evaluate_result(
                    DLA_result, S, question_A, user_input, question_text
                )
        # Retrieve all scores for this question after answering
        all_score = question_lib[str(S)][str(question_A)]["score"]
        # Calculate the mean score if available, otherwise set to 0.0
        question_openai_res = np.mean(all_score) if all_score else 0.0
        # Append the result to the question reward list
        question_reward.append(question_openai_res)

        # Return the total reward, termination flag, and last question
        logger.info(f"Finished question RL loop for item S={S}. Total reward: {float(sum(question_reward))}, DLA_terminate: {int(DLA_terminate)}")
        return float(sum(question_reward)), int(DLA_terminate), last_question

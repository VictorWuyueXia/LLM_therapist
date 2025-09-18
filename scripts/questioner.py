from typing import List, Tuple, Dict, Any

from scripts.utils.response_bridge import get_openai_resp
from scripts.text_generators import (
    generate_change,
    generate_change_positive,
    generate_change_negative,
    generate_synonymous_sentences,
    generate_therapist_chat,
)


def classify_segments(user_segments: List[str]) -> List[Tuple[str, int]]:
    """
    Pure function: take user segments, call classifier, return [(label, score)]
    - label: 'DLA_xxx' or 'DLA' or 'NA'
    - score: integer in {0,1,2} or special values like 99
    """
    result = []
    for seg in user_segments:
        if not seg:
            continue
        label, score = get_openai_resp(seg)
        result.append((label, score))
    return result


def evaluate_result_core(
    dla_result: List[Tuple[str, Any]],
    item_index: int,
    question_index: str,
    user_segments: List[str],
    original_question: str,
    question_lib: Dict[str, Any],
) -> Tuple[int, int, str, Dict[str, Any]]:
    """
    Core evaluation logic without I/O lock decisions.
    Returns: (valid, terminate_flag, last_question_text, updated_question_lib)
    """
    last_question = ""
    if not dla_result:
        return 0, 0, last_question, question_lib

    # direct label tokens path
    if isinstance(dla_result[0][1], str):
        if dla_result[0][1] == "Stop":
            return 1, 1, last_question, question_lib
        score = question_lib[str(item_index)][str(question_index)][dla_result[0][1]]
        question_lib[str(item_index)][str(question_index)]["score"].append(score)
        if score > 1:
            text = question_lib[str(item_index)][str(question_index)]["question"][0]
            text = generate_change_positive(text) if dla_result[0][1] == "Yes" else generate_change_negative(text)
            text_temp = generate_synonymous_sentences(" Can you tell me more about it?")
            last_question = "It seems that " + text + " " + text_temp
            # followup note to be appended by caller after collecting follow-up
            original_resp = "original_resp: " + user_segments[0]
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
        else:
            original_resp = "original_resp: " + user_segments[0]
            note_resp = [
                "original_question: " + original_question,
                original_resp,
            ]
            question_lib[str(item_index)][str(question_index)]["notes"].append(note_resp)
        return 1, 0, last_question, question_lib

    # numeric scores path (multi labels)
    # For simplicity, mark valid if any scored label matches the question label
    question_label = question_lib[str(item_index)][str(question_index)]["label"]
    valid = 0
    therapist_resp = ""
    for i, (label, score_val) in enumerate(dla_result):
        if isinstance(score_val, str) and score_val == "Stop":
            return 1, 1, last_question, question_lib
        if isinstance(score_val, int) and score_val != 99:
            if label.lower() == question_label.lower():
                valid = 1
            question_lib[str(item_index)][str(question_index)]["score"].append(score_val)
            if score_val > 1:
                text = generate_change(user_segments[i]).lower()
                last_question = "You mentioned that " + text + " Can you tell me more?"
                # followup is to be acquired by caller and saved as notes
    return valid, 0, therapist_resp or last_question, question_lib




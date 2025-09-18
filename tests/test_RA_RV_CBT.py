#!/usr/bin/env python

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.response_analyzer import classify_dimension_and_score
from src.utils.response_bridge import get_openai_resp
from src.questioner import classify_segments, evaluate_result_core
from src.utils.io_question_lib import load_question_lib
from src.utils.config_loader import QUESTION_LIB_FILENAME
from src.CBT import stage1_guide
from src.utils.io_record import init_record, log_question, get_answer, get_resp_log


def main():
    result = classify_dimension_and_score("My weight increased a lot recently.")
    print("Test 1 - classify_dimension_and_score:", result)

    resp = get_openai_resp("My weight increased a lot recently.")
    print("Test 2 - get_openai_resp:", resp)

    ql = load_question_lib(QUESTION_LIB_FILENAME)
    segs = ["I gained a lot of weight."]
    dla = classify_segments(segs)
    print("Test 3 - classify_segments:", dla)
    eval_result = evaluate_result_core(dla, 1, "1", segs, "Do you have weight change?", ql)
    print("Test 3 - evaluate_result_core (valid, terminate, last_question):", eval_result[0:3])

    guide = stage1_guide("I avoid speaking in meetings.")
    print("Test 4 - stage1_guide:", guide)

    init_record()
    log_question("Test question?")
    print("Test 5 - Please manually write a response to 'record.csv' and set Resp_Lock=0 before continuing.")
    answer = get_answer()
    print("Test 5 - get_answer:", answer)
    log_question("Follow up?")
    followup = get_resp_log()
    print("Test 5 - get_resp_log:", followup)


if __name__ == "__main__":
    main()



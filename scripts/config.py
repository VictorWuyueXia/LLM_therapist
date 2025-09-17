# scripts/config.py
import os
SUBJECT_ID = os.environ.get("SUBJECT_ID", "8901")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Subject_8899_Mobicom_SenSys")
RESULT_DIR = os.path.join(DATA_DIR, "data_collection_results")

QUESTION_LIB_FILENAME = os.path.join(DATA_DIR, f"question_lib_v3_{SUBJECT_ID}.json")
REPORT_FILE = os.path.join(RESULT_DIR, f"Report_{SUBJECT_ID}.csv")
NOTES_FILE  = os.path.join(RESULT_DIR, f"Notes_{SUBJECT_ID}.csv")
RECORD_CSV  = os.path.join(DATA_DIR, "record.csv")

ITEM_N_STATES = 20
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9

ITEM_IMPORTANCE = [0, 5, 4, 4, 2, 5, 2, 2, 1, 3, 4, 3, 1, 4, 2, 4, 3, 1, 4, 4]
NUMBER_QUESTIONS = [0, 4, 1, 2, 2, 5, 1, 1, 1, 2, 3, 2, 1, 1, 3, 2, 1, 3, 1, 1]
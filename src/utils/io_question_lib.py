import json
import csv
import os
from src.utils.config_loader import REPORT_FILE, NOTES_FILE

def load_question_lib(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_question_lib(path: str, question_lib: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(question_lib, f)

def generate_results(
    question_lib: dict,
    new_response: list,
    report_file: str = REPORT_FILE,
    notes_file: str = NOTES_FILE
):
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    rows = []
    for i in range(1, len(question_lib) + 1):
        for ind in range(1, len(question_lib[str(i)]) + 1):
            rows.append([
                question_lib[str(i)][str(ind)]["label"],
                question_lib[str(i)][str(ind)]["score"],
                question_lib[str(i)][str(ind)]["notes"]
            ])
    with open(report_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(['Item Label', 'Score', 'Notes'])
        w.writerows(rows)
    rows_new = []
    for rec in new_response:
        try:
            rows_new.append([
                rec["item"],
                rec["question"],
                rec["DLA_result"],
                rec["User_input"],
                rec["User_comment"]
            ])
        except:
            rows_new.append([
                rec["item"],
                rec["question"],
                rec["DLA_result"],
                rec["User_input"]
            ])
    with open(notes_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(['Item', "question", "Original_question", "DLA_result", "User_input", "User_comment"])
        w.writerows(rows_new)



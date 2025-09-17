# scripts/io_question_lib.py

import json
import csv
import os
from scripts.config import REPORT_FILE, NOTES_FILE

def load_question_lib(path: str):
    """
    Load the question library from a JSON file.
    Args:
        path (str): Path to the JSON file.
    Returns:
        dict: The loaded question library.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_question_lib(path: str, question_lib: dict):
    """
    Save the question library to a JSON file.
    Args:
        path (str): Path to the JSON file.
        question_lib (dict): The question library to save.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(question_lib, f)

def generate_results(
    question_lib: dict,
    new_response: list,
    report_file: str = REPORT_FILE,
    notes_file: str = NOTES_FILE
):
    """
    Generate and save results and notes from the question library and new responses.
    Writes two CSV files: one for the report and one for notes.

    Args:
        question_lib (dict): The question library containing items and their details.
        new_response (list): List of new response records (dicts).
        report_file (str): Path to the report CSV file.
        notes_file (str): Path to the notes CSV file.
    """
    # Ensure the directory for the report file exists
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    # Prepare rows for the report file
    rows = []
    for i in range(1, len(question_lib) + 1):
        # For each item in the question library
        for ind in range(1, len(question_lib[str(i)]) + 1):
            # For each question under the item
            rows.append([
                question_lib[str(i)][str(ind)]["label"],
                question_lib[str(i)][str(ind)]["score"],
                question_lib[str(i)][str(ind)]["notes"]
            ])
    # Write the report CSV file with item label, score, and notes
    with open(report_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(['Item Label', 'Score', 'Notes'])
        w.writerows(rows)

    # Prepare rows for the notes file
    rows_new = []
    for rec in new_response:
        # Try to extract all expected fields; if "User_comment" is missing, skip it
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
    # Write the notes CSV file with detailed response information
    with open(notes_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(['Item', "question", "Original_question", "DLA_result", "User_input", "User_comment"])
        w.writerows(rows_new)
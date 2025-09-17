# scripts/io_record.py

import pandas as pd
from scripts.config import RECORD_CSV

# Define the columns to be used in the record CSV file
HEADER = ["Question", "Question_Lock", "Resp", "Resp_Lock"]

def _read():
    """
    Read the record CSV file and return it as a pandas DataFrame.
    """
    return pd.read_csv(RECORD_CSV)

def _write(df):
    """
    Write the DataFrame to the record CSV file, keeping only the specified columns.
    """
    df.to_csv(RECORD_CSV, columns=HEADER, index=False)

def log_question(text: str):
    """
    Log a new question to the record file.
    Waits until the Question_Lock is released (set to 0), then writes the question,
    sets the lock to 1 (locked), and saves the file.
    """
    while True:
        data = _read()
        # Only write the question if the lock is released
        if data.loc[0, "Question_Lock"] == 0:
            data.loc[0, "Question"] = text
            data.loc[0, "Question_Lock"] = 1  # Lock after writing
            _write(data)
            break

def get_answer():
    """
    Wait for a new answer to be available (Resp_Lock == 0), then read and lock it.
    Splits the answer into segments by replacing certain conjunctions with periods,
    and returns the segments. DLA_result is left empty for the caller to fill.
    """
    while True:
        data = _read()
        # Only read the answer if the lock is released
        if data.loc[0, "Resp_Lock"] == 0:
            user_input = data.loc[0, "Resp"]
            data.loc[0, "Resp_Lock"] = 1  # Lock after reading
            _write(data)
            break
    user_input = str(user_input)
    # Replace certain conjunctions with periods to segment the answer
    user_input = user_input.replace(", and", ".").replace("but", ".")
    user_input = user_input.split(".")
    DLA_result, segments = [], []
    for seg in user_input:
        if not seg:
            continue
        # Remove leading space if present
        if seg[0] == " ":
            seg = seg[1:]
        segments.append(seg)
    return DLA_result, segments  # DLA_result is to be filled by the caller

def get_resp_log():
    """
    Wait for a new response to be available (Resp_Lock == 0), then read and lock it.
    Returns the raw response string.
    """
    while True:
        data = _read()
        # Only read the response if the lock is released
        if data.loc[0, "Resp_Lock"] == 0:
            follow = data.loc[0, "Resp"]
            data.loc[0, "Resp_Lock"] = 1  # Lock after reading
            _write(data)
            break
    return follow

def init_record():
    """
    Initialize the record file by releasing the question lock and locking the response.
    This prepares the system for a new question/answer cycle.
    """
    data = _read()
    data.loc[0, 'Question_Lock'] = 0  # Release question lock
    data.loc[0, 'Resp_Lock'] = 1      # Lock response
    _write(data)
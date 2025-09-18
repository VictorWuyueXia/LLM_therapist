import pandas as pd
from scripts.config import RECORD_CSV

HEADER = ["Question", "Question_Lock", "Resp", "Resp_Lock"]

def _read():
    return pd.read_csv(RECORD_CSV)

def _write(df):
    df.to_csv(RECORD_CSV, columns=HEADER, index=False)

def log_question(text: str):
    while True:
        data = _read()
        if data.loc[0, "Question_Lock"] == 0:
            data.loc[0, "Question"] = text
            data.loc[0, "Question_Lock"] = 1
            _write(data)
            break

def get_answer():
    while True:
        data = _read()
        if data.loc[0, "Resp_Lock"] == 0:
            user_input = data.loc[0, "Resp"]
            data.loc[0, "Resp_Lock"] = 1
            _write(data)
            break
    user_input = str(user_input)
    user_input = user_input.replace(", and", ".").replace("but", ".")
    user_input = user_input.split(".")
    DLA_result, segments = [], []
    for seg in user_input:
        if not seg:
            continue
        if seg[0] == " ":
            seg = seg[1:]
        segments.append(seg)
    return DLA_result, segments

def get_resp_log():
    while True:
        data = _read()
        if data.loc[0, "Resp_Lock"] == 0:
            follow = data.loc[0, "Resp"]
            data.loc[0, "Resp_Lock"] = 1
            _write(data)
            break
    return follow

def init_record():
    data = _read()
    data.loc[0, 'Question_Lock'] = 0
    data.loc[0, 'Resp_Lock'] = 1
    _write(data)



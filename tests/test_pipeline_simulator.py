#!/usr/bin/env python
# tests/test_pipeline_simulator.py

import os
import sys
import time
import random
import subprocess
from typing import Tuple

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import RECORD_CSV

HEADER = ["Question", "Question_Lock", "Resp", "Resp_Lock"]


def ensure_record_csv() -> None:
    folder = os.path.dirname(RECORD_CSV)
    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(RECORD_CSV):
        df = pd.DataFrame([["", 0, "", 1]], columns=HEADER)
        df.to_csv(RECORD_CSV, index=False)
    else:
        df = pd.read_csv(RECORD_CSV)
        for col in HEADER:
            if col not in df.columns:
                df[col] = "" if col in ("Question", "Resp") else 0
        if len(df) == 0:
            df = pd.DataFrame([["", 0, "", 1]], columns=HEADER)
        df = df[HEADER]
        df.to_csv(RECORD_CSV, index=False)


def read_state() -> Tuple[str, int, str, int]:
    df = pd.read_csv(RECORD_CSV)
    row = df.iloc[0]
    return (
        str(row["Question"]),
        int(row["Question_Lock"]),
        str(row["Resp"]),
        int(row["Resp_Lock"]),
    )


def write_answer(answer_text: str) -> None:
    df = pd.read_csv(RECORD_CSV)
    df.loc[0, "Resp"] = answer_text
    df.loc[0, "Resp_Lock"] = 0
    df.loc[0, "Question_Lock"] = 0
    df.to_csv(RECORD_CSV, index=False)


def choose_answer(question_text: str, step_idx: int, max_steps: int) -> str:
    if step_idx >= max_steps - 1:
        return "Stop"
    return random.choice(["Yes", "No"])


def run_app_background() -> subprocess.Popen:
    py = sys.executable
    cmd = [py, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLM_therapist_Application.py")]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def simulate(rounds: int = 8, poll_interval_s: float = 0.3, overall_timeout_s: float = 180.0) -> int:
    ensure_record_csv()
    proc = run_app_background()
    start = time.time()
    step = 0
    try:
        while True:
            if time.time() - start > overall_timeout_s:
                return 2
            code = proc.poll()
            if code is not None:
                return 0
            q_text, q_lock, _, _ = read_state()
            if q_lock == 1 and q_text.strip():
                ans = choose_answer(q_text, step, rounds)
                write_answer(ans)
                step += 1
                if step >= rounds:
                    write_answer("Stop")
            time.sleep(poll_interval_s)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def main():
    code = simulate()
    if code == 0:
        print("Pipeline simulation finished (app exited).")
        sys.exit(0)
    elif code == 2:
        print("Pipeline simulation timeout.")
        sys.exit(2)
    else:
        print("Pipeline simulation ended with unknown status.")
        sys.exit(code)


if __name__ == "__main__":
    main()



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

from scripts.utils.config_loader import RECORD_CSV

HEADER = ["Question", "Question_Lock", "Resp", "Resp_Lock"]

def _atomic_write(df: pd.DataFrame) -> None:
    time.sleep(0.03)
    folder = os.path.dirname(RECORD_CSV)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    tmp_path = RECORD_CSV + ".test.tmp"
    df.to_csv(tmp_path, columns=HEADER, index=False)
    time.sleep(0.1)
    os.replace(tmp_path, RECORD_CSV)
    time.sleep(0.03)

def read_state() -> Tuple[str, int, str, int]:
    # Light retry to avoid partial reads
    for _ in range(5):
        try:
            time.sleep(0.03)
            df = pd.read_csv(RECORD_CSV)
            break
        except Exception:
            time.sleep(0.05)
    row = df.iloc[0]
    return (
        str(row["Question"]),
        int(row["Question_Lock"]),
        str(row["Resp"]),
        int(row["Resp_Lock"]),
    )

def write_answer(answer_text: str) -> None:
    time.sleep(0.03)
    df = pd.read_csv(RECORD_CSV)
    # Normalize answer to avoid triple quotes in CSV
    normalized = str(answer_text).strip().strip('"')
    df.loc[0, "Resp"] = normalized
    df.loc[0, "Resp_Lock"] = 0
    df.loc[0, "Question_Lock"] = 0
    _atomic_write(df)

def choose_answer(question_text: str, step_idx: int, max_steps: int) -> str:
    if step_idx >= max_steps - 1:
        return "Stop"
    return random.choice(['Yes', 'No', 'I think so', 'I don\'t think so'])

def run_app_background() -> subprocess.Popen:
    py = sys.executable
    cmd = [py, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LLM_therapist_Application.py")]
    
    # # running in the background
    # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # running in the foreground
    proc = subprocess.Popen(cmd)
    
    return proc

def simulate(rounds: int = 8, poll_interval_s: float = 0.3, overall_timeout_s: float = 180.0) -> int:
    # App (init_record) is responsible for initializing record.csv
    proc = run_app_background()
    start = time.time()
    step = 0
    try:
        while True:
            if time.time() - start > overall_timeout_s:
                return 2
            code = proc.poll()
            if code is not None:
                try:
                    out = proc.stdout.read().decode('utf-8', errors='ignore') if proc.stdout else ''
                    if out:
                        print(out)
                except Exception:
                    pass
                return code
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

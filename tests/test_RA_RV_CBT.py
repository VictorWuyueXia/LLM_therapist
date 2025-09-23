#!/usr/bin/env python

import sys
import os

# Add project root to sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.handler_rl import HandlerRL


def test_cbt_success_flow(monkeypatch):
    """
    Verify: CBT stages 0-3 run successfully and results are written to the selected dimension's notes.
    - stage0: select a candidate dimension with score=2
    - stage1: identify unhelpful thoughts
    - stage2: challenge unhelpful thoughts
    - stage3: reframe unhelpful thoughts
    """
    # Construct a minimal question_lib (1 dimension, already has score=2)
    handler = HandlerRL()
    handler.question_lib = {
        "1": {
            "1": {
                "label": "1_weight",
                "score": [2],
                "question": ["dummy question"],
                "notes": [],
            }
        }
    }

    # Store asked questions (avoid real I/O)
    asked = []

    def _fake_log_question(text: str):
        asked.append(text)

    # Predefine user response sequence: select dimension -> statement -> unhelpful thought -> challenge -> reframe
    responses = iter([
        "1",
        "I want to perform better in meetings.",
        "People will judge me if I speak.",
        "I can look for evidence and alternative explanations.",
        "I can contribute even if it's not perfect.",
    ])

    def _fake_get_resp_log():
        return next(responses)

    # Patch CBT LLM stages: always return valid (DECISION: 0), avoid external API calls
    monkeypatch.setattr("src.handler_rl.stage0_prompter", lambda history: "QUESTION: Please choose a dimension to work on.")
    monkeypatch.setattr("src.handler_rl.stage1_reasoner", lambda statement, unhelpful: "DECISION: 0")
    monkeypatch.setattr("src.handler_rl.stage2_reasoner", lambda statement, unhelpful, challenge: "DECISION: 0")
    monkeypatch.setattr("src.handler_rl.stage3_reasoner", lambda statement, unhelpful, challenge, reframe: "DECISION: 0")
    # Guides are not used in the success path, but still stubbed to avoid accidental calls
    monkeypatch.setattr("src.handler_rl.stage1_guide", lambda statement: "UNHELPFUL_THOUGHTS: you think others will judge you.")
    monkeypatch.setattr("src.handler_rl.stage2_guide", lambda statement, unhelpful: "CHALLENGE: consider evidence and alternatives.")
    monkeypatch.setattr("src.handler_rl.stage3_guide", lambda statement, unhelpful, challenge: "REFRAME: a more balanced view.")

    # Patch I/O functions to avoid real CSV read/write
    monkeypatch.setattr("src.handler_rl.log_question", _fake_log_question)
    monkeypatch.setattr("src.handler_rl.get_resp_log", _fake_get_resp_log)

    # Run CBT
    handler.run_cbt()

    # Assert that notes record the successful CBT process
    notes = handler.question_lib["1"]["1"]["notes"]
    assert len(notes) >= 1
    last = notes[-1]
    # last is a list of strings, should contain key fields
    assert any("CBT_stage: success" in s for s in last)
    assert any("CBT_statement:" in s for s in last)
    assert any("CBT_unhelpful_thoughts:" in s for s in last)
    assert any("CBT_challenge:" in s for s in last)
    assert any("CBT_reframe:" in s for s in last)

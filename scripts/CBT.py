import os
import logging
from openai import OpenAI
from scripts.config_loader import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")
client = OpenAI(api_key=_api_key)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.DEBUG)
    logger.addHandler(_ch)

REASONER_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.
You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The user's answer towards the CBT question "Can you try to identify any unhelpful thoughts you have that contribute to this situation?".
Your goal is: Justify if the user identifies the unhelpful thoughts properly (0: identified properly, 1: not properly identified).
Response format:
DECISION: 0/1
Provide response with [DECISION] only.
'''

REASONER_CBT_STAGE2_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to CBT questions.
You will be provided with:
1. STATEMENT
2. UNHELPFUL_THOUGHTS
3. CHALLENGE
Your goal is: Justify if the patient challenges the unhelpful thoughts (0: properly challenge, 1: not challenge).
Response format:
DECISION: 0/1
Provide response with [DECISION] only.
'''

REASONER_CBT_STAGE3_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You will be provided with:
1. STATEMENT
2. UNHELPFUL_THOUGHTS
3. CHALLENGE
4. REFRAME
Your goal is: Justify if the patient reframes the unhelpful thoughts properly (0: properly reframe, 1: fail to reframe).
Response format:
DECISION: 0/1
Provide response with [DECISION] only.
'''

GUIDE_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
Try to recognize negative thoughts based on the patient's statement (use second person voice).
Response format:
UNHELPFUL_THOUGHTS: xxxx
'''

GUIDE_CBT_STAGE2_PROMPT = '''You will be provided with:
1. STATEMENT
2. UNHELPFUL_THOUGHTS
Your goal is: Help the patient challenge the unhelpful thoughts properly.
Response format:
CHALLENGE: xxxx
'''

GUIDE_CBT_STAGE3_PROMPT = '''You will be provided with:
1. STATEMENT
2. UNHELPFUL_THOUGHTS
3. CHALLENGE
Your goal is: Reframe the unhelpful thoughts for the patient.
Response format:
REFRAME: xxxx
'''

def _chat_complete(system_content: str, user_content: str):
    logger.debug({"model": OPENAI_MODEL, "user": user_content[:200]})
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_tokens=OPENAI_MAX_TOKENS,
        temperature=OPENAI_TEMPERATURE,
    )
    return resp.choices[0].message.content

def stage1_reasoner(statement: str, unhelpful_thoughts: str) -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts};"'
    return _chat_complete(REASONER_CBT_STAGE1_PROMPT, payload)

def stage2_reasoner(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge};"'
    return _chat_complete(REASONER_CBT_STAGE2_PROMPT, payload)

def stage3_reasoner(statement: str, unhelpful_thoughts: str, challenge: str, reframe: str) -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge}; REFRAME: {reframe};"'
    return _chat_complete(REASONER_CBT_STAGE3_PROMPT, payload)

def stage1_guide(statement: str) -> str:
    payload = f"STATEMENT: {statement}"
    return _chat_complete(GUIDE_CBT_STAGE1_PROMPT, payload)

def stage2_guide(statement: str, unhelpful_thoughts: str) -> str:
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}"
    return _chat_complete(GUIDE_CBT_STAGE2_PROMPT, payload)

def stage3_guide(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}. CHALLENGE: {challenge}"
    return _chat_complete(GUIDE_CBT_STAGE3_PROMPT, payload)

__all__ = [
    "stage1_reasoner",
    "stage2_reasoner",
    "stage3_reasoner",
    "stage1_guide",
    "stage2_guide",
    "stage3_guide",
]



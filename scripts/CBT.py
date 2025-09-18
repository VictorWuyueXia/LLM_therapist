import os
import logging
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

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

def _chat_complete(system_content: str, user_content: str, gpt_model: str = "gpt-4"):
    logger.debug({"model": gpt_model, "user": user_content[:200]})
    resp = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        max_tokens=300,
        temperature=0.7,
        top_p=1
    )
    return resp['choices'][0]['message']['content']

def stage1_reasoner(statement: str, unhelpful_thoughts: str, gpt_model: str = "gpt-4") -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts};"'
    return _chat_complete(REASONER_CBT_STAGE1_PROMPT, payload, gpt_model=gpt_model)

def stage2_reasoner(statement: str, unhelpful_thoughts: str, challenge: str, gpt_model: str = "gpt-4") -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge};"'
    return _chat_complete(REASONER_CBT_STAGE2_PROMPT, payload, gpt_model=gpt_model)

def stage3_reasoner(statement: str, unhelpful_thoughts: str, challenge: str, reframe: str, gpt_model: str = "gpt-4") -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge}; REFRAME: {reframe};"'
    return _chat_complete(REASONER_CBT_STAGE3_PROMPT, payload, gpt_model=gpt_model)

def stage1_guide(statement: str, gpt_model: str = "gpt-4") -> str:
    payload = f"STATEMENT: {statement}"
    return _chat_complete(GUIDE_CBT_STAGE1_PROMPT, payload, gpt_model=gpt_model)

def stage2_guide(statement: str, unhelpful_thoughts: str, gpt_model: str = "gpt-4") -> str:
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}"
    return _chat_complete(GUIDE_CBT_STAGE2_PROMPT, payload, gpt_model=gpt_model)

def stage3_guide(statement: str, unhelpful_thoughts: str, challenge: str, gpt_model: str = "gpt-4") -> str:
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}. CHALLENGE: {challenge}"
    return _chat_complete(GUIDE_CBT_STAGE3_PROMPT, payload, gpt_model=gpt_model)

__all__ = [
    "stage1_reasoner",
    "stage2_reasoner",
    "stage3_reasoner",
    "stage1_guide",
    "stage2_guide",
    "stage3_guide",
]



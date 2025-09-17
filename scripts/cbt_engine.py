# scripts/cbt_engine.py
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

# === Reasoner Prompt（from CaiRE_CBT.ipynb）===

REASONER_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.
You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The user's answer towards the CBT question "Can you try to identify any unhelpful thoughts you have that contribute to this situation?". This is the step that the patient tries to recognize negative thoughts. These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
Usually the patient's statement and responses contain situation that is not valid or useful. As an AI assistant, you need to examine the validaity and utility of the patient's response.
There are 13 possible common cognitive distortions that the patient might encounter.
Your goal is:
Justify if the user is identify the unhelpful thoughts properly in the statement(0: identified properly, 1: not properly identified).
Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.
'''

REASONER_CBT_STAGE2_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.
You will be provided with:
1. STATEMENT
2. UNHELPFUL_THOUGHTS
3. CHALLENGE
Your goal is:
Justify if the patient challenges the unhelpful thoughts (0: properly challenge, 1: not challenge).
Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.
'''

REASONER_CBT_STAGE3_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You will be provided with:
1. STATEMENT
2. UNHELPFUL_THOUGHTS
3. CHALLENGE
4. REFRAME
Your goal is:
Justify if the patient reframes the unhelpful thoughts properly (0: properly reframe, 1: fail to reframe).
Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.
'''

# === Guide Prompt（from CaiRE_CBT.ipynb）===

GUIDE_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to answer the cognitive behavioural therapy (CBT) questions based-on patient's statement provided.
Your goal is:
Try to recognize negative thoughts based on the statement (use second person voice).
Response format:
UNHELPFUL_THOUGHTS: xxxx
'''

GUIDE_CBT_STAGE2_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You will be provided with:
1. STATEMENT
2. UNHELPFUL_THOUGHTS
Your goal is:
Help the patient challenge the unhelpful thoughts properly.
Response format:
CHALLENGE: xxxx
'''

GUIDE_CBT_STAGE3_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You will be provided with:
1. STATEMENT
2. UNHELPFUL_THOUGHTS
3. CHALLENGE
Your goal is:
Reframe the unhelpful thoughts for the patient.
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

# === Reasoner  ===

def stage1_reasoner(statement: str, unhelpful_thoughts: str, gpt_model: str = "gpt-4") -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts};"'
    return _chat_complete(REASONER_CBT_STAGE1_PROMPT, payload, gpt_model=gpt_model)

def stage2_reasoner(statement: str, unhelpful_thoughts: str, challenge: str, gpt_model: str = "gpt-4") -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge};"'
    return _chat_complete(REASONER_CBT_STAGE2_PROMPT, payload, gpt_model=gpt_model)

def stage3_reasoner(statement: str, unhelpful_thoughts: str, challenge: str, reframe: str, gpt_model: str = "gpt-4") -> str:
    payload = f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; CHALLENGE: {challenge}; REFRAME: {reframe};"'
    return _chat_complete(REASONER_CBT_STAGE3_PROMPT, payload, gpt_model=gpt_model)

# === Guide  ===

def stage1_guide(statement: str, gpt_model: str = "gpt-4") -> str:
    payload = f"STATEMENT: {statement}"
    return _chat_complete(GUIDE_CBT_STAGE1_PROMPT, payload, gpt_model=gpt_model)

def stage2_guide(statement: str, unhelpful_thoughts: str, gpt_model: str = "gpt-4") -> str:
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}"
    return _chat_complete(GUIDE_CBT_STAGE2_PROMPT, payload, gpt_model=gpt_model)

def stage3_guide(statement: str, unhelpful_thoughts: str, challenge: str, gpt_model: str = "gpt-4") -> str:
    payload = f"STATEMENT: {statement}. UNHELPFUL_THOUGHTS: {unhelpful_thoughts}. CHALLENGE: {challenge}"
    return _chat_complete(GUIDE_CBT_STAGE3_PROMPT, payload, gpt_model=gpt_model)
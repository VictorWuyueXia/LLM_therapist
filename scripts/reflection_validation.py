# scripts/reflection_validation.py
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

RV_FOLLOW_UP_SYSTEM_REASONER_PROMPT = '''You are an intelligent agent who have strong reasoning capability and psychology and mental health commonsense knowledge. 
You are in a conversation session with a user. You need to evaluate if the user provide a follow-up response that's related to the original respone or the conversation topic. 
You will be provided with:
The conversattion topic, the original response, and the followup response in the format of '{"Topic": XXXX, "Original Response": XXXX, "Follow Up Response": XXXX}'
DECISION = 0 if the follow-up response is related to the "Original Response" or the "Topic". Otherwise, DECISION = 1.
Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.
'''

RV_FOLLOW_UP_GUIDE_SYSTEM_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are in the conversation with a client.
You will be provided with:
1. The conversattion topic.
2. The original response from the client.
3. The follow-up response from the client to the question 'Can you tell me more about it?'. This response is unrelated to the topic or the original response. Or sometimes, it is hard to tell if it is related to the topic or the original response. And thus needs more information or clarification from the client.
These infromation will be provided in the format of '{"Topic": XXXX, "Original Response": XXXX, "Follow-up Response": XXXX}'
Goal:
You need to guide the user to comeup with the valid follow-up response, which should give more details to your original response or the topic.
You need to first express the understanding to the client's follow-up response, and then try to lead the client to the right direction.
Don't read into the clients' mind and make too much assumptions. Try to use the phrases used by the client in your response, instead of rephrasing too much.
Response format:
Guide: xxxx
'''

RV_FOLLOW_UP_VALIDATION_SYSTEM_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are in the conversation with a client. You need to provide empathic validation and support to the client.
You will be provided with:
1. The conversattion topic.
2. The original response from the client.
3. The follow-up response from the client to the question 'Can you tell me more about it?'.
These infromation will be provided in the format of '{"Topic": XXXX, "Original Response": XXXX, "Follow-up Response": XXXX}'
Goal:
You need to provide empathic validation and support to the client based on the conversation topic, origianl response, and followup response.
You need to first express the understanding to the client's follow-up response, and then try to lead the client to the right direction.
Don't read into the clients' mind and make too much assumptions. Try to use the phrases used by the client in your response, instead of rephrasing too much.
Response format:
VALIDATION: xxxx
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

def rv_reasoner(topic: str, original_response: str, follow_up_response: str) -> str:
    payload = f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow Up Response": {follow_up_response!r}}}'
    return _chat_complete(RV_FOLLOW_UP_SYSTEM_REASONER_PROMPT, payload)

def rv_guide(topic: str, original_response: str, follow_up_response: str) -> str:
    payload = f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow-up Response": {follow_up_response!r}}}'
    return _chat_complete(RV_FOLLOW_UP_GUIDE_SYSTEM_PROMPT, payload)

def rv_validation(topic: str, original_response: str, follow_up_response: str) -> str:
    payload = f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow-up Response": {follow_up_response!r}}}'
    return _chat_complete(RV_FOLLOW_UP_VALIDATION_SYSTEM_PROMPT, payload)
# scripts/reflection_validation.py

import os
import logging
from openai import OpenAI
from scripts.utils.config_loader import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS

# Retrieve OpenAI API key from environment variable
_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")
# Initialize OpenAI client with API key
client = OpenAI(api_key=_api_key)

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.DEBUG)
    logger.addHandler(_ch)

# Prompt for the reasoner: checks if follow-up is related to topic or original response
RV_FOLLOW_UP_SYSTEM_REASONER_PROMPT = '''You are an intelligent agent who have strong reasoning capability and psychology and mental health commonsense knowledge. 
You are in a conversation session with a user. You need to evaluate if the user provide a follow-up response that's related to the original respone or the conversation topic. 
You will be provided with:
The conversattion topic, the original response, and the followup response in the format of '{"Topic": XXXX, "Original Response": XXXX, "Follow Up Response": XXXX}'
DECISION = 0 if the follow-up response is related to the "Original Response" or the "Topic". Otherwise, DECISION = 1.
Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.
'''

# Prompt for the guide: helps user provide a more relevant follow-up
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

# Prompt for validation: provides empathic validation and support
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
    """
    Send a chat completion request to OpenAI API with the given system and user content.
    Returns the content of the first message in the response.
    """
    logger.info("Sending request to OpenAI API for chat completion.")
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
    logger.info("Received response from OpenAI API.")
    return resp.choices[0].message.content

def rv_reasoner(topic: str, original_response: str, follow_up_response: str) -> str:
    """
    Use the reasoner prompt to determine if the follow-up is related to the topic or original response.
    Returns the decision as a string.
    """
    logger.info("Running reflection validation reasoner.")
    payload = f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow Up Response": {follow_up_response!r}}}'
    return _chat_complete(RV_FOLLOW_UP_SYSTEM_REASONER_PROMPT, payload)

def rv_guide(topic: str, original_response: str, follow_up_response: str) -> str:
    """
    Use the guide prompt to help the user provide a more relevant follow-up response.
    Returns the guide as a string.
    """
    logger.info("Running reflection validation guide.")
    payload = f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow-up Response": {follow_up_response!r}}}'
    return _chat_complete(RV_FOLLOW_UP_GUIDE_SYSTEM_PROMPT, payload)

def rv_validation(topic: str, original_response: str, follow_up_response: str) -> str:
    """
    Use the validation prompt to provide empathic validation and support to the user.
    Returns the validation as a string.
    """
    logger.info("Running reflection validation support/validation.")
    payload = f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, "Follow-up Response": {follow_up_response!r}}}'
    return _chat_complete(RV_FOLLOW_UP_VALIDATION_SYSTEM_PROMPT, payload)
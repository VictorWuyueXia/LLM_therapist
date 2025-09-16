#!/usr/bin/env python
# coding: utf-8

# # RL workflow for personalized question generation

# ### There will be a q table for 20 items and q tables for each item that has more than 1 question

# ## 1. The way to ask questions

# ## 2. Segmentation

# model names:
# 03-22-2022: davinci:ft-personal-2022-03-22-01-51-13
# 03-29-2022: curie:ft-personal-2022-03-29-00-33-49
# 04-07-2022: davinci:ft-personal-2022-04-07-05-39-11
# 04-11-2022: davinci:ft-personal-2022-04-11-23-57-59

# In[3]:


import pandas as pd
import random
#import matplotlib.pyplot as plt
# data = pd.read_csv('record.csv')
import warnings
warnings.filterwarnings('ignore')
import time
import random
from word2number import w2n


# In[4]:


subjectID = str(8901)#str(input("please enter subject number"))
filename =  "question_lib_v3_"+subjectID+".json"
print(filename)


# In[5]:


# Make sure that the locks are ready to go
data = pd.read_csv('record.csv')
data['Question_Lock'][0]=0
data['Resp_Lock'][0]=1
header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
data.to_csv('record.csv', columns = header)
data


# lower fedality sensor

# In[6]:


openai_model = "davinci:ft-personal-2022-05-30-20-14-19" #""davinci:ft-personal-2022-05-21-22-14-23"# "davinci:ft-personal-2022-04-11-23-57-59"
report_file_name = "data_collection_results/Report_"+subjectID+'_'+str(time.time())+".csv"
notes_file_name = "data_collection_results/Notes_"+subjectID+'_'+str(time.time())+".csv"


# In[7]:


import numpy as np
import pandas as pd
import time

# import sounddevice as sd
# from scipy.io.wavfile import write
# import wavio as wv

# from google.cloud import speech
import os
# import google.cloud.texttospeech as tts
#from playsound import playsound

import openai
import json

import csv


np.random.seed(2)  # reproducible

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "platinum-scout-key.json"
ITEM_N_STATES = 20   # initial state + DLA_1 to DLA_18 + additional question
ITEM_ACTIONS = ['{0}'.format(element) for element in np.arange(0, ITEM_N_STATES)]    # available actions
ITEM_IMPORTANCE = [0, 5, 4, 4, 2, 5, 2, 2, 1, 3, 4, 3, 1, 4, 2, 4, 3, 1, 4, 4]  # importance rated by Vera
NUMBER_QUESTIONS = [0, 4, 1, 2, 2, 5, 1, 1, 1, 2, 3, 2, 1, 1, 3, 2, 1, 3, 1, 1] # number of questions in each item
QUESTIONS_IN_ITEM = []
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move
US_speakers = ['en-US-Standard-A',
 'en-US-Standard-B',
 'en-US-Standard-C',
 'en-US-Standard-D',
 'en-US-Standard-E',
 'en-US-Standard-F',
 'en-US-Standard-G',
 'en-US-Standard-H',
 'en-US-Standard-I',
 'en-US-Standard-J',
 'en-US-Wavenet-A',
 'en-US-Wavenet-B',
 'en-US-Wavenet-C',
 'en-US-Wavenet-D',
 'en-US-Wavenet-E',
 'en-US-Wavenet-F',
 'en-US-Wavenet-G',
 'en-US-Wavenet-H',
 'en-US-Wavenet-I',
 'en-US-Wavenet-J']

# Opening JSON file
f = open(filename)
 
# returns JSON object as
# a dictionary
question_lib = json.load(f)


# In[8]:


def generate_prompt_synonymous_sentences(user_input):
    return """Generate synonymous sentences.

    User: I am sad.
    Answer: I feel sad.
    User: I really enjoy my work recently.
    Answer: I like my job a lot those days.
    User: I have problem hearing you well.
    Answer: I have problem understand you well.
    User:{}
    Answer:""".format(
            user_input.capitalize()
        )

def generate_synonymous_sentences(question_text):
    openai.api_key = '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'
    user_input = question_text
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=generate_prompt_synonymous_sentences(user_input),
        temperature=0.8,
        max_tokens = 1000,
    )
    results = response.choices[0].text
    print(response.choices[0].text)
    return results

def generate_prompt_therapist(user_input):
    return """Chat with people as a therapist.

    User: I feel so depressed daily.
    Answer: I am so sorry to hear that. It's OK to feel a little bit depressed but you need to figure out a way to makes you feel better. You can talk to a friend or family member. Or you can reach out to a therapist. And I am always here to support you.
    User: I don't want to talk.
    Answer: I get that you don’t want to have this conversation. But it's important to share your feelings with others and find out ways to make you feel better. 
    User: My partner wants to check my messages everyday.
    Answer: When you having a controlling partner, you might want to know the following items. Understand Controlling Personality Types. Recognize the Part of You That Accepts Another's Control. Take Back Responsibility for Your Life. Decide Whether You Need or Want Controlling Men in Your Life. Know What You Want Out of Life. Learn and Practice Assertiveness. Set Healthy Boundaries.
    User: I don't know what's going on with me.
    Answer: It's fine not to know the reason why you don't feel well now. Doing medication might help you understand yourself better. Or you can reach out to your family members, friends, or therapist to help you out.
    User:{}
    Answer:""".format(
            user_input.capitalize()
        )

def generate_therapist_chat(user_input):
    openai.api_key = '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'
    user_input = user_input
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=generate_prompt_therapist(user_input),
        temperature=0.6,
        max_tokens = 1000,
    )
    print(response.choices[0].text)
    result = response.choices[0].text
    return result

def generate_prompt_change(user_input):
    return """　Change from first-person sentence to second-person.

    User: I feel so depressed daily.
    Answer: You feel so depressed daily.
    User: I am so happy.
    Answer: You are so happy.
    User: I am under a lot of pressure.
    Answer: You are under a lot of pressure.
    User:{}
    Answer:""".format(
            user_input.capitalize()
        )

def generate_change(user_input): 
    openai.api_key = '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'
    user_input = user_input
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=generate_prompt_change(user_input),
        temperature=0.6,
        max_tokens = 1000,
    )
    print(response.choices[0].text)
    resp = response.choices[0].text
    return resp


def generate_prompt_change_positive(user_input):
    return """　Change from question to positive declarative sentence.

    User: Do you have coping skills to help you calm down.
    Answer: You have coping skills to help you calm down.
    User: Do you have self-harming behaviours?
    Answer: You have self-harming behaviours.
    User: Are you involved in any legal issues recently?
    Answer: You are involved in some legal issues recently.
    User:{}
    Answer:""".format(
            user_input.capitalize()
        )

def generate_change_positive(user_input): 
    openai.api_key = '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'
    user_input = user_input
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=generate_prompt_change_positive(user_input),
        temperature=0.6,
        max_tokens = 1000,
    )
    print(response.choices[0].text)
    resp = response.choices[0].text
    return resp

def generate_prompt_change_negative(user_input):
    return """　Change from question to negative declarative sentence.

    User: Do you have coping skills to help you calm down.
    Answer: You don't have coping skills to help you calm down.
    User: Do you feel productive?
    Answer: You don't feel productive.
    User: Have you done anything creative recently?
    Answer: You haven't done anything creative recently.
    User:{}
    Answer:""".format(
            user_input.capitalize()
        )

def generate_change_negative(user_input): 
    openai.api_key = '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'
    user_input = user_input
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=generate_prompt_change_negative(user_input),
        temperature=0.6,
        max_tokens = 1000,
    )
    print(response.choices[0].text)
    resp = response.choices[0].text
    return resp


# In[9]:


generate_therapist_chat("I am worried about COVID-19.")


# In[10]:


def generate_results():

    fields = ['Item Label', 'Score', 'Notes'] 

#     # data rows of csv file 
#     rows = [ ['Nikhil', 'COE', '2', '9.0'], 
#              ['Sanchit', 'COE', '2', '9.1'], 
#              ['Aditya', 'IT', '2', '9.3'], 
#              ['Sagar', 'SE', '1', '9.5'], 
#              ['Prateek', 'MCE', '3', '7.8'], 
#              ['Sahil', 'EP', '2', '9.1']] 

    rows = []
    cnt = 0
    for i in range(1, len(question_lib)+1):
        print("----")
        print(len(question_lib[str(i)]))
        for ind in range(1, len(question_lib[str(i)])+1):
            print(question_lib[str(i)][str(ind)]["label"], question_lib[str(i)][str(ind)]["score"])
            items = [question_lib[str(i)][str(ind)]["label"], question_lib[str(i)][str(ind)]["score"], question_lib[str(i)][str(ind)]["notes"] ]
            rows.append(items)
            cnt += 1
    print(rows)



    with open(report_file_name, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(rows)
        
    rows_new=[]
    for i in range(0, len(new_response)):
        try:
            #items = [new_response[i]["item"], new_response[i]["question"], new_response[i]["original question"], new_response[i]["DLA_result"], new_response[i]["User_input"], new_response[i]["User_comment"]]
            items = [new_response[i]["item"], new_response[i]["question"], new_response[i]["DLA_result"], new_response[i]["User_input"], new_response[i]["User_comment"]]

        except:
            items = [new_response[i]["item"], new_response[i]["question"], new_response[i]["DLA_result"], new_response[i]["User_input"]]
        rows_new.append(items)
    
    fields = ['Item', "question", "Original_question", "DLA_result", "User_input", "User_comment"] 
        
    with open(notes_file_name, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        write.writerows(rows_new)
        


# In[11]:


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
#     print(table)    # show table
    return table


# In[12]:


def initialize_question_q_table(n_states, actions):
    question_q_table = build_q_table(n_states, actions)
    for i in range(1, n_states):
        question_q_table[str(i)] = question_q_table[str(i)].apply(lambda x: x+1)
    return question_q_table


# In[13]:


# Initialize Q table for the 20 items
def initialize_q_table(ITEM_N_STATES, ITEM_ACTIONS):
    item_q_table = build_q_table(ITEM_N_STATES, ITEM_ACTIONS)
    for i in range(0, ITEM_N_STATES):
        item_q_table[str(i)] = item_q_table[str(i)].apply(lambda x: x+ITEM_IMPORTANCE[i])
    return item_q_table


# In[14]:


def choose_action(state, q_table, mask, number_states, actions):
    # This is how to choose an action
    print("state in choose action: ", state)
    state_action = q_table.iloc[state, :]
    #print(state_action)
    # update q_table with the mask
    for i in range(1, number_states):
        q_table[str(i)] = q_table[str(i)].apply(lambda x: x*mask[i])
    #print("q_table in choose_action():", q_table)

    if (np.random.uniform() > EPSILON) or ((state_action == 0).all()):  # act non-greedy or state-action have no value
#         print("random: ")
        action_name = np.random.choice(actions[1:number_states])
    else:   # act greedy
        # some actions may have the same value, randomly choose on in these actions
#         print("max: ")
        action_name = np.random.choice(state_action[state_action == np.max(state_action)].index)
#    action_name = np.random.choice(state_action[state_action == np.max(state_action)].index) #with no greedy

    return action_name


# In[15]:


def get_env_feedback(S, A, openai_res, DLA_terminate, item_mask):
    # This is how agent will interact with the environment
    if sum(item_mask) == 0: # If all the questions has been asked or the user is really agitated
        S_ = 'terminal'
        R = 10
    elif DLA_terminate == 1:
        S_ = 'terminal'
        R = 0
    else:   # if in the state of asking questions
        S_ = int(A)
        R = openai_res
#     print(S_, R)

    return S_, R


# ## Functions for Google APIs
# 

# In[16]:


# def speech_to_text():
#     user_input = ""
    
#     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "platinum-scout-key.json"

#     # Instantiates a client
#     client = speech.SpeechClient()

#     # Load media files
#     media_file_name_wav = "demo.wav"
#     with open(media_file_name_wav, 'rb') as f1:
#         byte_data_wav = f1.read()
#     audio_wav = speech.RecognitionAudio(content = byte_data_wav)

#     config_wav = speech.RecognitionConfig(
#         sample_rate_hertz=44100,
#         enable_automatic_punctuation = True,
#         language_code="en-US",
#         audio_channel_count = 1
#     )

#     # Transcribing the recognition audio object
#     response_wav = client.recognize(
#         config = config_wav,
#         audio = audio_wav
#     )
#     for result in response_wav.results:
#         print("Transcript: {}".format(result.alternatives[0].transcript))
#         user_input += result.alternatives[0].transcript
#     return user_input


# In[17]:


# def text_to_wav(voice_name: str, text: str):
#     language_code = "-".join(voice_name.split("-")[:2])
#     text_input = tts.SynthesisInput(text=text)
#     voice_params = tts.VoiceSelectionParams(
#         language_code=language_code, name=voice_name
#     )
#     audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

#     client = tts.TextToSpeechClient()
#     response = client.synthesize_speech(
#         input=text_input, voice=voice_params, audio_config=audio_config
#     )

#     filename = f"{language_code}.wav"
#     with open(filename, "wb") as out:
#         out.write(response.audio_content)
#         print(f'Generated speech saved to "{filename}"')
        
#     playsound(filename)


# In[18]:


def get_openai_resp(user_input):

    openai.api_key = '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'
    res = openai.Completion.create(
     # model='davinci:ft-columbia-university-2022-02-28-15-50-17',
     model = openai_model,
        #'davinci:ft-personal-2022-03-20-19-44-33',
     prompt = user_input+"->",
     max_tokens = 17,)

    cmd = res['choices'][0]['text']

#     cmd = get_openai_results(user_input)

#     print(cmd)
#     print(type(cmd))
    cmd = cmd.replace("->","")
    cmd = cmd.replace(";",",")
    cmd = cmd.replace(".",",")
    
    response = cmd.split(",")[:2]
    try:
        category = response[0].replace(" ","")
        score = 99
        if category == "DLA":
            if "No" in response[1]:
                score = "No"
            elif "Yes" in response[1]:
                score = "Yes"
            elif "Stop" in response[1]:
                score = "Stop"
            elif "Question" in response[1]:
                score = "Question"
            elif "Maybe" in response[1]:
                score = 1
        else:
            try:
                if float(response[1]) >=2:
                    score = 2        
                elif float(response[1]) < 1:
                    score = 0
                else:
                    score = 1   
            except:
                pass
        
    except:
        pass
#     print(score)
    if score == 99:
        print("1")
        try:
            all_cmd = cmd.split(",")
            print(all_cmd)
            if "DLA" in all_cmd:
                category = "DLA"
                print("category",category)
                if "Stop" in all_cmd:
                    score ="Stop"
                elif "Yes" in all_cmd:
                    score ="Yes"
                elif "No" in all_cmd:
                    score = "No"
                else:
                    score = 99
            else:
                category = "NA"
                score = 99
        except:
            pass
        
    ### Check if there is directly yes or no in the user_input sentence
    user_correction = user_input.replace(".", " ")
    user_correction = user_correction.replace(",", " ")
    user_correction = user_correction.replace("?", " ")
    user_correction = user_correction.split(" ")
    user_correction =[ i.lower() for i in user_correction]
    if "yes" in user_correction:
        category = "DLA"
        score ="Yes"
    if "no" in user_correction:
        category = "DLA"
        score ="No"
        
    if "stop" in user_correction[0]:
        category = "DLA"
        score ="Stop"

    
    return category, score


# In[19]:


get_openai_resp("What about you tell me?")


# In[20]:


def get_answer():
    while True:
        try:
            data = pd.read_csv('record.csv')
        except:
            pass
        if data["Resp_Lock"][0] == 0:
            user_input = data["Resp"][0] 
            data["Resp_Lock"][0] = 1
            header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
            data.to_csv('record.csv', columns = header)
            break
#     while user_input == "":
#         text_to_wav('en-US-Standard-A', "Sorry, I have problem hearing you, please try to speak louder.")
#         detect_and_record()
#         user_input = speech_to_text()
    user_input = str(user_input)
    user_input.replace(", and", ".")
    user_input = user_input.split(".")
    print(user_input)
    DLA_result = []
    for i in range(0, len(user_input)):
        sentence = user_input[i]
        if user_input[i] == "":
            pass
        else:
            if user_input[i][0] == " ":
                user_input[i] = user_input[i][1:]
            category, score = get_openai_resp(user_input[i])
            openai_res = [category, score]
            DLA_result.append(openai_res)
    print(DLA_result)
    return DLA_result, user_input


# In[21]:


def remove_duplicate_items(DLA_result, user_input):
    dla_res = []
    user_res = []
    for i in range(0, len(DLA_result)):
        if DLA_result[i] not in dla_res:
            dla_res.append(DLA_result[i])
            user_res.append(user_input[i])
        else:
            index = dla_res.index(DLA_result[i])
            user_res[index] += " "+user_input[i]
    return dla_res, user_res


# In[22]:


def log_question(text):
    while True:
        try:
            data = pd.read_csv('record.csv')
        except:
            pass
        if data["Question_Lock"][0] == 0:
#                             question = question_lib[str(S)][str(question_A)]["question"][0]
#                             Question_text = "Sounds like you did not understand my question. Let me ask it again. "+question
            data["Question"][0] = text
            print(text)
            data["Question_Lock"][0] = 1
            header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
            data.to_csv('record.csv', columns = header)
            break


# In[23]:


def get_resp_log():
    while True:
        try:
            data = pd.read_csv('record.csv')
        except:
            pass
        if data["Resp_Lock"][0] == 0:
            user_input_followup = data["Resp"][0]                            
            data["Resp_Lock"][0] = 1
            header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
            data.to_csv('record.csv', columns = header)
            break
    return user_input_followup


# In[24]:


# evaluate_result([['DLA_1_height', 0]], "1", "1", ["My weight changes a lot", ''], "Have you lost or gained a significant amount of weight?")


# In[25]:


###Mobile Version

def evaluate_result(DLA_result, S, question_A, user_input, original_question_asked):
    global last_question
    last_question = " "
    # have valid answer
    therapist_resp = ""
    try:
        if DLA_result[0][1] == "Question":
            print("evaluate question check")
            question = question_lib[str(S)][str(question_A)]["question"][0]
            Question_text = "Sounds like you did not understand my question. Let me ask it again. "+question
            log_question(Question_text)
#             while True:
#                 try:
#                     data = pd.read_csv('record.csv')
#                 except:
#                     pass
#                 if data["Question_Lock"][0] == 0:
#                     question = question_lib[str(S)][str(question_A)]["question"][0]
#                     Question_text = "Sounds like you did not understand my question. Let me ask it again. "+question
#                     data["Question"][0] = Question_text 
#                     print(Question_text)
#                     data["Question_Lock"][0] = 1
#                     header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
#                     data.to_csv('record.csv', columns = header)
#                     break
            DLA_result, user_input = get_answer()
    except:
        pass
    
    DLA_result, user_input = remove_duplicate_items(DLA_result, user_input)
    valid = 0
    DLA_terminate = 0
    print("check")
    
    try:
        print("3")
        if type(DLA_result[0][1]) == str: # Check if there is direct yes or no ansewer to the question
            print("1 + There is direct yes/no/stop to the question")
            if DLA_result[0][1] == "Stop":
                DLA_terminate = 1             
                
            else:
                print("2")
                score = question_lib[str(S)][str(question_A)][DLA_result[0][1]]
                question_lib[str(S)][str(question_A)]["score"].append(score)
                valid = 1
                if score > 1:  
                    print("If score > 1 for direct yes/no/stop")
                    text = question_lib[str(S)][str(question_A)]["question"][0]
                    if DLA_result[0][1] == "Yes":
                        text = generate_change_positive(text)
                    else:
                        text = generate_change_negative(text)
                # Repeat the question
                    text_temp = generate_synonymous_sentences(" Can you tell me more about it?")
                    question_text = "It seems that "+text+" "+text_temp
                    
                    log_question(question_text)
                    print("4")
                    
                    user_input_followup = get_resp_log()
                    
                    print("5", user_input_followup)
                                
                    therapist_resp = generate_therapist_chat(text+" "+user_input_followup)
                    #log_question(therapist_resp)
                    print("therapist_resp in evaluate Y/N", therapist_resp)
                    print("6")
                    last_question = therapist_resp
        
                    original_resp = "original_resp: " + user_input[0]
                    followup_resp = "followup_resp: " + user_input_followup
                    #therapist_resp =  "therapist_resp: " + therapist_resp
                    original_question_asked_record = "original_question: "+original_question_asked
                    note_resp = [original_question_asked_record, original_resp, followup_resp, "therapist_resp: " + therapist_resp]
                    question_lib[str(S)][str(question_A)]["notes"].append(note_resp)
                else:
                    original_resp = "original_resp: " + user_input[0]
                    original_question_asked_record = "original_question: "+original_question_asked
                    note_resp = [original_question_asked_record, original_resp]
                    question_lib[str(S)][str(question_A)]["notes"].append(note_resp)
        question_label = question_lib[str(S)][str(question_A)]["label"]
        error_count = 0
        rephrase_count = 0
        for i in range(0, len(DLA_result)): 
            # check if the user have valid answer for the question that been asked
            print("Evaluate DLA: ", DLA_result[i])

            if type(DLA_result[i][1]) == str: # Check if there is a stop indication to the question
                if DLA_result[i][1] == "Stop":
                    DLA_terminate = 1
                    valid = 1
                return valid, DLA_terminate, last_question
            else:            
                label = DLA_result[i][0]
                print("label", label)
                if type(DLA_result[i][1]) == int and DLA_result[i][1] != 99:
                    if DLA_result[i][0].lower() == question_label.lower():
                        valid = 1
                    print(valid)
                    score = DLA_result[i][1]
                    print("Score: ", score)
                    if score > 1:
                        text = user_input[i]
                        text = generate_change(text)
                    # Repeat the question
                        question_text = "You mentioned that "+text+" Can you tell me more?"
                        
#                         text_to_wav(speaker, question_text)
#                         print(question_text)
#                         detect_and_record()
                        
                        
                        log_question(question_text)
                        print("7")

#                         user_input_followup = speech_to_text()
                        user_input_followup = get_resp_log()

                        print("8", user_input_followup)

                        therapist_resp = generate_therapist_chat(text+" "+user_input_followup)
                        print("therapist_resp in evaluate", therapist_resp)
                        #log_question(therapist_resp)
                        print("9")
                        
#                         therapist_resp = generate_therapist_chat(text+" "+user_input_followup)
#                         text_to_wav(speaker, therapist_resp)

                    try:
                        print("check1")
                        item_number = DLA_result[i][0].split("_")[1]
                        print(item_number)
                        if int(item_number) == 21:
                            item_number = 19
                        print(len(question_lib[str(item_number)]))
                        if len(question_lib[str(item_number)]) == 1:
                            print("Only one question in this item.")
    #                         print(question_lib[str(item_number)]["1"]["label"].lower(), label.lower())
                            if question_lib[str(item_number)]["1"]["label"].lower() == label.lower():
                                question_number = 1
                            
                        else: 
                            for ind in range(1, len(question_lib[str(item_number)])+1):
                                if question_lib[str(item_number)][str(ind)]["label"].lower() == label.lower():
                                    question_number = ind
                       
                        try:
                            print("item_number, question_number, valid:", item_number, question_number, valid)
                            question_lib[str(item_number)][str(question_number)]["score"].append(score)
                            if score>1:
                                original_resp = "original_resp: " + user_input[i]
                                followup_resp = "followup_resp: " + user_input_followup
#                                 therapist_resp =  "therapist_resp: " + therapist_resp
                                original_question_asked_record = "original_question: "+original_question_asked
                                note_resp = [original_question_asked_record, original_resp, followup_resp, "therapist_resp: " + therapist_resp]
                                question_lib[str(item_number)][str(question_number)]["notes"].append(note_resp)
                            else:
                                original_resp = "original_resp: " + user_input[i]
                                original_question_asked_record = "original_question: "+original_question_asked
                                note_resp = [original_question_asked_record, original_resp, original_question_asked_record]
                                question_lib[str(item_number)][str(question_number)]["notes"].append(note_resp)
                                
                                if label == "DLA_21_sports" and score == 0:
                                    question_lib[str(11)][str(1)]["notes"].append(note_resp) ## if people do sports, that means they have hobbies
                                    question_lib[str(11)][str(1)]["score"].append(score)
                                    if int(S) == 11 and int(question_A) == 1:
                                        valid = 1
                                    
                        except:
                            print("Have problem processing the response.")
                            print(S, question_A, DLA_result[i], user_input[i])
                            correction = {"item": S, "question": question_A, "DLA_result":DLA_result[i], "User_input":user_input[i]}
    #                         new_response.append({"item": S, "question": question_A, "DLA_result":DLA_result[i], "User_input":user_input[i]})
                            if len(DLA_result) == 1:
                                question_text = "Sorry, our system currently cannot process your response in a correct way. We need your help to improve the system. "
                                question_text += "You are trying to answer: "+ original_question_asked+". And your response is: "+user_input[i]+". Is that right? "
                                question_text += "If that's right, please say YES. If we didn't get it right, please say No."
                                print("10", question_text)
                        
                                log_question(question_text)
                                print("11")

                                user_correction = get_resp_log()
                                print("12", user_correction)
                                

                            else:
                                if error_count == 0:
                                    question_text = "Sorry, our system currently cannot process part of your response in a correct way. We need your help to improve the system. "
                                    question_text += "You are trying to answer: "+ original_question_asked+"that the system asked. And a part of your response is: "+user_input[i]+". Is that right? "
                                    question_text += "If that's right, please say YES. If we didn't get it right, please say No."
                                    print("10", question_text)
                        
                                    log_question(question_text)
                                    print("11")

                                    error_count += 1
                                else:
                                    question_text = "Sorry again, our system currently cannot process another part of your response in a correct way. We need your help again."
                                    question_text += "You are trying to answer: "+ original_question_asked+"that the system asked. And a part of your response is: "+user_input[i]+". Is that right? "
                                    question_text += "If that's right, please say YES. If we didn't get it right, please say No."
                                    print("12", question_text)
                        
                                    log_question(question_text)
                                    print("13")
                                
                                    error_count += 1

                                user_correction = get_resp_log()
                                print("15", user_correction)
                                
                            user_correction = user_correction.replace(".", " ")
                            user_correction = user_correction.replace(",", " ")
                            user_correction = user_correction.replace("?", " ")
                            user_correction = user_correction.split(" ")
                            user_correction =[ i.lower() for i in user_correction]
                            if "no" in user_correction:
                                question_text = "It seems like we did not get it right. Please tell us what is not right for you."
                                log_question(question_text)
                                user_correction = get_resp_log()
                                correction = {"item": S, "question": question_A, "original question": original_question_asked, "DLA_result":DLA_result[i], "User_input":user_input[i], "User_comment": user_correction}
                            elif "yes" in user_correction:
                                print("in evaluate: check if need to rephrase the answer.")
                                if rephrase_count == 0:
                                    print("Please rephrase.")
                                    question_text = "Could you please rephrase your answer to our question: "+original_question_asked
                                    log_question(question_text)
                                    user_correction = get_resp_log()
                                    correction = {"item": S, "question": question_A, "original question": original_question_asked, "DLA_result":DLA_result[i], "User_input":user_input[i], "User_comment": user_correction}
                                    rephrase_count += 1
                                else:
                                    pass
                                    
                            therapist_resp = "Thank you for your feedback. We will improve our system and provide a better user experience for you."
                            print("16", therapist_resp)
                            #log_question(question_text)
                            new_response.append(correction)
                            valid = 1

                    except:
                        pass
    except:
        pass
    
    last_question = therapist_resp
    print('last_question in evaluate_result: ', last_question)
    
    return valid, DLA_terminate, last_question
            


# In[26]:


# initialize all question q tables
def initialize_question_table():
    all_question_q_table = {}
    ITEM_ACTIONS = ['{0}'.format(element) for element in np.arange(0, ITEM_N_STATES)]   
    for i in range(0, ITEM_N_STATES):
        if NUMBER_QUESTIONS[i]>1:
            question_actions = ['{0}'.format(element) for element in np.arange(NUMBER_QUESTIONS[i]+1)]
            question_q_table = initialize_question_q_table(NUMBER_QUESTIONS[i]+1, question_actions)
            #print(question_q_table)
            all_question_q_table[i] = question_q_table
    return all_question_q_table


# In[27]:


def initialize_question_mask():
    all_question_mask = {}
    for i in range(0, ITEM_N_STATES):
        if NUMBER_QUESTIONS[i]>1:
            all_question_mask[i] = [0]+[1] * NUMBER_QUESTIONS[i]
    return all_question_mask


# In[28]:


def ask_question(S, all_question_mask): # the sequence to ask question in each item
    global last_question
    print("Item number: ", S)
#     speaker = US_speakers[np.random.randint(20)]
    question_S = 0
    question_A_order = []
    question_reward = []
    DLA_terminate = 0
    if NUMBER_QUESTIONS[S]>1: # if there's more than one question
        question_actions = ['{0}'.format(element) for element in np.arange(NUMBER_QUESTIONS[S]+1)]
        question_q_table = all_question_q_table[S].copy()
        question_mask = all_question_mask[S]
        new_question_q_table = question_q_table.copy()
#         print("question_q_table", question_q_table)
#         print("new_question_q_table", new_question_q_table)
        is_terminated = False
        number_of_states = NUMBER_QUESTIONS[S]+1
        while not is_terminated:
            print("*"*99)
            question_A = choose_action(question_S, question_q_table, question_mask, number_of_states, question_actions)
            print("Question Action", question_A)
            question_mask[int(question_A)] = 0
            print("question_mask", question_mask)
            print("score in log", question_lib[str(S)][str(question_A)]["score"])
            #### interface to OpenAI model & Alexa here:
            #############################################
            
            ## Only ask the question and get the answer from the user when the item is not been answered before.
            if len(question_lib[str(S)][str(question_A)]["score"]) == 0:
                number_of_questions = len(question_lib[str(S)][str(question_A)]["question"])
                choice_of_question = np.random.randint(number_of_questions)
                question_text = question_lib[str(S)][str(question_A)]["question"][choice_of_question]
                question_label = question_lib[str(S)][str(question_A)]["label"]
                #generate synonymous sentence under certain probability
                if (np.random.uniform() > 0.95):
                    question_text = generate_synonymous_sentences(question_text)

                print(question_text)
                question_text_ask = last_question +"  "+ question_text
                print(last_question, question_text)
                # Either ask question from ALEXA or from a standalone speaker
                #text_to_wav(speaker, question_text)
                
                log_question(question_text_ask)
                

                # Get the response from the user:
                DLA_result, user_input = get_answer()
                print(DLA_result, user_input)
                valid, DLA_terminate, last_question = evaluate_result(DLA_result, S, question_A, user_input, question_text)
                print(last_question)
                
                print("DLA_result:", DLA_result)
                print("valid:", valid)
                print("DLA_terminate:", DLA_terminate)

                #ask until valid
                if valid == 0 and DLA_terminate == 0:
                    valid_loop = 0
                    while valid_loop < 1:
                        question_to_ask = last_question+"Sorry. Do you mind rephrasing your answer in a different way. Please make sure you answer the question I ask. "
                        if (np.random.uniform() > 0.5):
                            question_to_ask += "And try to answer my question in a complete sentence and in a concise and deterministic way if you can."
                        log_question(question_text_ask)
                        DLA_result, user_input = get_answer()
                        valid_loop, DLA_terminate, last_question = evaluate_result(DLA_result, S, question_A, user_input,question_text)
                print("last question", last_question)
            else:
                print("Already answered, get reward from history")

            all_score = question_lib[str(S)][str(question_A)]["score"]            
            question_openai_res = np.mean(all_score)
            #DLA_terminate = 0
            
            
            ###
            question_S_, question_R = get_env_feedback(question_S, question_A, question_openai_res, DLA_terminate, question_mask)  # take action & get next state and reward
            question_A_order.append(question_A)
            question_reward.append(question_R)
            q_predict = question_q_table.loc[question_S, question_A]
            if question_S_ != 'terminal':
                q_target = question_R + GAMMA * question_q_table.iloc[question_S_, :].max()   # next state is not terminal
            else:
                q_target = question_R     # next state is terminal
                is_terminated = True    # terminate this episode
            print("q_target", q_target, "q_predict", q_predict)
            new_question_q_table.loc[question_S, question_A] += ALPHA * (q_target - q_predict)  # update
            print("new_question_q_table after update", new_question_q_table)
            question_S = question_S_  # move to next state  
            if DLA_terminate == 1:
                is_terminated = True

        question_q_table = new_question_q_table.copy()
        print("question_q_table after update", question_q_table)
        all_question_q_table[S] = question_q_table.copy()
        print("all_question_q_table[S]", all_question_q_table[S])
        all_question_mask[S] = question_mask
        print("_"*99)
    else:
        question_A = "1" ## There will be only 1 question
        
        ## Only ask the question and get the answer from the user when the item is not been answered before.
        if len(question_lib[str(S)][str(question_A)]["score"]) == 0:
            number_of_questions = len(question_lib[str(S)][str(question_A)]["question"])
            choice_of_question = np.random.randint(number_of_questions)
            question_text = question_lib[str(S)][str(question_A)]["question"][choice_of_question]
            question_label = question_lib[str(S)][str(question_A)]["label"]
            #generate synonymous sentence under certain probability
            if (np.random.uniform() > 0.95):
                question_text = generate_synonymous_sentences(question_text)

            print(question_text)

#             # Either ask question from ALEXA or from a standalone speaker
#             text_to_wav(speaker, question_text)

#             # Get the response from the user:
#             DLA_result, user_input = get_answer()
#             valid, DLA_terminate = evaluate_result(DLA_result, S, question_A, user_input, question_text)
#             print(DLA_result)
#             print(valid)
#             print("DLA_terminate:", DLA_terminate)
            

            question_text_ask = last_question + "  "+question_text

            # Either ask question from ALEXA or from a standalone speaker
            #text_to_wav(speaker, question_text)
            log_question(question_text_ask)

            # Get the response from the user:
            DLA_result, user_input = get_answer()
            valid, DLA_terminate, last_question = evaluate_result(DLA_result, S, question_A, user_input, question_text)
 
            print("DLA_result:", DLA_result)
            print("valid:", valid)
            print("DLA_terminate:", DLA_terminate)

            #ask until valid
            if valid == 0 and DLA_terminate == 0:
                valid_loop = 0
                while valid_loop < 1:
                    question_to_ask = last_question+"Sorry. Do you mind rephrasing your answer in a different way. Please make sure you answer the question I ask. "
                    if (np.random.uniform() > 0.5):
                        question_to_ask += "And try to answer my question in a complete sentence and in a concise and deterministic way if you can."
                    log_question(question_text_ask)
                    DLA_result, user_input = get_answer()
                    valid_loop, DLA_terminate, last_question = evaluate_result(DLA_result, S, question_A, user_input,question_text)
            print("last question", last_question)
            

#             #ask until valid
#             if valid == 0 and DLA_terminate == 0:
#                 valid_loop = 0
#                 while valid_loop < 1:
#                     text_to_wav(speaker, "Sorry. Do you mind rephrasing your answer in a different way. Please make sure you answer the question I ask.")
#                     if (np.random.uniform() > 0.5):
#                         text_to_wav(speaker, "And try to answer my question in a complete sentence if you can.")
#                     text_to_wav(speaker, question_text)
#                     DLA_result, user_input = get_answer()
#                     valid_loop, DLA_terminate = evaluate_result(DLA_result, S, question_A, user_input, question_text)
        else:
            print("Already answered, get reward from history")

        all_score = question_lib[str(S)][str(question_A)]["score"]            
        question_openai_res = np.mean(all_score)
#         ### interface to OpenAI model & Alexa here:
#         question_text = question_lib[str(S)]["1"]["question"][0]
            
#         #generate synonymous sentence under certain probability
#         if (np.random.uniform() > 0.5):
#             question_text = generate_synonymous_sentences(question_text)

#         print(question_text)

#         # Either ask question from ALEXA or from a standalone speaker
#         text_to_wav("en-US-Wavenet-F", question_text)
        
#         # Get the response from the user:
#         DLA_result = get_answer()
#         print(DLA_result)
        
#         question_openai_res = np.random.randint(5)
# #         DLA_terminate = 0
        ###
        question_reward.append(question_openai_res)
    print("question_reward: ", question_reward)
    openai_res = sum(question_reward) 
    print("DLA_terminate", DLA_terminate)
    
    return openai_res, DLA_terminate


# ## CBT AND MI functions:

# In[51]:


def prepare_for_MI_CBT(save_filename):
    #global question_lib_result
    f = open(save_filename)
    question_lib_result = json.load(f)
    issue_dimension = []
    good_dimension = []
    for i in range(1, len(question_lib_result)+1):
        for ind in range(1, len(question_lib_result[str(i)])+1):
            if sum(question_lib_result[str(i)][str(ind)]["score"]) > 1:
                name = question_lib_result[str(i)][str(ind)]["name"]
                issue_dimension.append([i, ind, name])

            elif sum(question_lib_result[str(i)][str(ind)]["score"]) < 1:
                name = question_lib_result[str(i)][str(ind)]["name"]
                good_dimension.append([i, ind, name])
                
    return issue_dimension, good_dimension

def get_dimension_to_work(save_filename, issue_dimension, issue_dimension_number):
    #global question_lib_result
    f = open(save_filename)
    question_lib_result = json.load(f)
#     for i in range(0, len(issue_dimension)):
#         stop = issue_dimension[i][2].index(":") 
#         dimension_name_number = issue_dimension[i][2][0:stop]
#         if "Dimension "+str(issue_dimension_number) == dimension_name_number:
    item_number = issue_dimension[issue_dimension_number-1][0]
    question_number = issue_dimension[issue_dimension_number-1][1]
    dimension_name = issue_dimension[issue_dimension_number-1][2]
    print(item_number, question_number, dimension_name)
            
    summary_original_response = []
    summary_followup_response = []

    for i in range(len(question_lib_result[str(item_number)][str(question_number)]["notes"])):
        for ind in range(len(question_lib_result[str(item_number)][str(question_number)]["notes"][i])):
            if "original_resp" in question_lib_result[str(item_number)][str(question_number)]["notes"][i][ind]:
                temp = question_lib_result[str(item_number)][str(question_number)]["notes"][i][ind].index(":")
                resp = question_lib_result[str(item_number)][str(question_number)]["notes"][i][ind][temp+2:]
                category, score = get_openai_resp(resp)
                if score == "Yes":
                    question = question_lib_result[str(item_number)][str(question_number)]["question"][0]
                    resp = generate_change_positive(question)
                elif score == "No":
                    question = question_lib_result[str(item_number)][str(question_number)]["question"][0]
                    resp = generate_change_negative(question)
                    
                summary_original_response.append(resp)

            if "followup_resp" in question_lib_result[str(item_number)][str(question_number)]["notes"][i][ind]:
                temp = question_lib_result[str(item_number)][str(question_number)]["notes"][i][ind].index(":")
                resp = question_lib_result[str(item_number)][str(question_number)]["notes"][i][ind][temp+2:]
                summary_followup_response.append(resp)
    
    if len(summary_original_response)+len(summary_followup_response) == 0:
        question = question_lib_result[str(item_number)][str(question_number)]["question"][0]
        summary_followup_response.append(generate_change_positive(question))
        
    return item_number, question_number, summary_original_response, summary_followup_response

def proceed_MI_CBT(save_filename):
    f = open(save_filename)
    question_lib_result = json.load(f)
    issue_dimension, good_dimension = prepare_for_MI_CBT(save_filename)
    ## if no issue with the subject:
    category = "category"
    score = "score"
    if len(issue_dimension)==0:
        Q5 = "It seems like you are doing pretty well. You work well in dimensions including: " 
        sample_good_dimension = random.sample(range(len(good_dimension)), 3)
        Q5 += good_dimension[sample_good_dimension[0]][2] + ", "+good_dimension[sample_good_dimension[1]][2]+", and "+ good_dimension[sample_good_dimension[2]][2]
        Q5 += ". Please reach out to your primary care or your therapist if you have further problems or emergencies. "
        Q5 += "Goodbye. We will followup later. 886"
        print("Q5:", Q5)
        log_question(Q5)
    
    else:
        #Q1:#####################################################
        Q1 = "Thank you for answering all the questions. "
        if (np.random.uniform() > 0.95):
            Q1 = generate_synonymous_sentences(Q1)
        Q1 += "According to your previous responses, you have issues in: "
        for i in range(0, len(issue_dimension)):
            Q1 += str(i+1)+": "+issue_dimension[i][2]+", "
        Q1 += "Which dimension do you want to work on today? Please speak out the dimension number, for example, 1."
        print("Q1:", Q1)
        
        log_question(Q1)
        
        user_dimension = str(get_resp_log())
        user_dimension = user_dimension.replace(".0","").replace(".", "")
        
        print("user_dimension1:", user_dimension)
        try:
            category, score = get_openai_resp(user_dimension)
        except:
            pass
        Q2 = " "
        if score == "Question":
            print("check")
            Q1 = "It seems like you don't understand me well. Let me repeat my question. According to your previous responses, you have issues in: "
            for i in range(0, len(issue_dimension)):
                Q1 += str(i+1)+": "+issue_dimension[i][2]+", "
            Q1 += "Which dimension do you want to work on today? Please speak out the dimesion number, for example, 1."
#             stop = issue_dimension[i][2].index(":") 
#             Q1 += issue_dimension[0][2][stop-2:stop-1] + "."
            log_question(Q1)
            user_dimension = str(get_resp_log())
            print("user_dimension2:", user_dimension)
            user_dimension = user_dimension.replace(".0","").replace(".", "")
            category, score = get_openai_resp(user_dimension)
            if score == "Question":
                print("Question again.")
                user_dimension = random.choice(range(len(issue_dimension)))
                issue_dimension_number = int(user_dimension)
                print("user_dimension2:", user_dimension)
                Q2 = "I will pick a dimension to work on today."
        try:
            issue_dimension_number = w2n.word_to_num(user_dimension)
            issue_dimension_number = int(user_dimension)
        except:
            try:
                issue_dimension_number = w2n.word_to_num(user_dimension)
            except:
                print("fail to get issue_dimension_number")
                user_dimension = random.choice(range(len(issue_dimension)))
                issue_dimension_number = int(user_dimension)
                Q2 = "I have problem getting the dimension you want to work on. So I pick a dimension to work on today. "
        
#         cnt = 0
#         for i in range(0, len(issue_dimension)):
#             dim = "Dimension "+str(issue_dimension_number)
#             if dim in issue_dimension[i][2]:
#                 cnt+=1
        if issue_dimension_number > len(issue_dimension)+1:
            user_dimension = random.choice(range(len(issue_dimension)))
            issue_dimension_number = int(user_dimension)
            Q2 = "Looks like you are doing OK with the dimension you chose. So I pick an issue dimension to work on today. "
        #Q2:#####################################################
        item_number, question_number, summary_original_response, summary_followup_response = get_dimension_to_work(save_filename, issue_dimension, issue_dimension_number)
        
        Q2 += "Let's work on Dimension " + str(issue_dimension_number)+". From my record, your responses to my question in this dimension are: "
            
        for i in range(0, len(summary_original_response)):
            Q2 += summary_original_response[i]+" and "

        for i in range(0, len(summary_followup_response)):
            Q2 += summary_followup_response[i]+" and "

        Q2 = Q2[:-5]
        Q2 += " Can you try to identify any unhelpful thoughts you have that contribute to this situation?"
        print("Q2:", Q2)
        log_question(Q2)        
        user_thought = str(get_resp_log())
        print("user_thought:", user_thought)
        category, score = get_openai_resp(user_thought)
        
        if score == "Question":
            Q2 = "It seems like you can not get my question. Let me repeat my question. We are working on Dimension " + str(issue_dimension_number)+". From my record, you mentioned: "
                
            for i in range(0, len(summary_original_response)):
                Q2 += summary_original_response[i]+" and "

            for i in range(0, len(summary_followup_response)):
                Q2 += summary_followup_response[i]+" and "

            Q2 = Q2[:-5]
            Q2 += " Can you try to identify any unhelpful thoughts you have that contribute to this situation?"
            print("Q2:", Q2)
            log_question(Q2)        
            user_thought = get_resp_log()
       
        ##### Q3
        
        Q3 = "Can you challenge your thought?"
        log_question(Q3)   
        print("Q3:", Q3)
        user_challenge = str(get_resp_log())
        print("user_challenge:", user_challenge)
        ####### Q4
        
        rephrase_challenge = generate_change(user_challenge)
        Q4 = "You mentioned that: "+rephrase_challenge + " to challenge your thought. Now, what is another way of thinking about this situation?"
        print("Q4:", Q4)
        log_question(Q4)
        user_new_way = get_resp_log()
        print("user_new_way:", user_new_way)
        
        ####### Q5
        
        Q5 = "Congratulation, you figure out a way for yourself. You also work well in dimensions including: " 
        sample_good_dimension = random.sample(range(len(good_dimension)), 3)
        Q5 += good_dimension[sample_good_dimension[0]][2] + ", "+good_dimension[sample_good_dimension[1]][2]+", and "+ good_dimension[sample_good_dimension[2]][2]
        Q5 += ". Please reach out to your primary care or your therapist if you have further problems or emergencies. "
        Q5 += "Goodbye. We will followup later. 886"
        print("Q5:", Q5)
        log_question(Q5)
        
        CBT_notes = ['CBT_Dimension: '+str(user_dimension), "CBT_unhelpful_thought: "+str(user_thought), "CBT_challenge: "+str(user_challenge),
                    "CBT_new_way:"+str(user_new_way)]
        
        f = open(save_filename)
        question_lib_result = json.load(f)
        question_lib_result[str(item_number)][str(question_number)]["notes"].append(CBT_notes)
        
        with open(save_filename, 'w') as f:
            json.dump(question_lib_result, f)
        
        


# In[28]:


def save_all_question_q_table():
    elements = [1, 3, 4, 5, 9, 10, 11, 14, 15, 17]
    for i in range(len(all_question_q_table)):
        filename = str(elements[i])+".pkl"
        all_question_q_table[elements[i]].to_pickle(filename)



def load_all_question_q_table():
    test = {}
    elements = [1, 3, 4, 5, 9, 10, 11, 14, 15, 17]
    for i in range(10):
        filename = str(elements[i])+".pkl"
        test[elements[i]] = pd.read_pickle(filename)
    return test


# In[29]:


# new_subject = int(input("New Subject? Yes:1, No:0"))
# if new_subject:
#     all_question_q_table = initialize_question_table()   ### Only change if change to a new person;
#     save_all_question_q_table()
#     print('initialize the q tables')


# In[ ]:


all_question_mask = initialize_question_mask()
all_question_q_table = load_all_question_q_table()    ### Only change if change to a new person;

f = open(filename)
question_lib = json.load(f)

item_q_table = initialize_q_table(ITEM_N_STATES, ITEM_ACTIONS)
new_response = []
# # speaker = US_speakers[np.random.randint(len(US_speakers))]
last_question = " "

for episode in range(1):
    cnt = 0 
    S = 0
    is_terminated = False
    item_mask = [0] + [1] * (ITEM_N_STATES-1)
    new_q_table = item_q_table.copy()
    while not is_terminated:
#         print("item_mask: ", item_mask)
        A = choose_action(S, item_q_table, item_mask, ITEM_N_STATES, ITEM_ACTIONS)
        print("ITEM A", A)
        item_mask[int(A)] = 0
        openai_res, DLA_terminate = ask_question(int(A), all_question_mask)
#         openai_res = np.random.randint(5)
#         DLA_terminate = 0
        print("ITEM openai_res", openai_res)
        print("ITEM DLA_terminate", DLA_terminate)
        S_, R = get_env_feedback(S, A, openai_res, DLA_terminate, item_mask)  # take action & get next state and reward
        q_predict = item_q_table.loc[S, A]
        if S_ != 'terminal':
            q_target = R + GAMMA * item_q_table.iloc[S_, :].max()   # next state is not terminal
        else:
            q_target = R     # next state is terminal
            is_terminated = True    # terminate this episode

        new_q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
        S = S_  # move to next state  
        cnt += 1
        if DLA_terminate == 1:
            is_terminated = True
            save_filename = "question_lib_v3_" + str(subjectID) + "_"+ str(int(time.time()))+".json"
            with open(save_filename, 'w') as f:
                json.dump(question_lib, f)
            log_question("Goodbye. We will do the screening in another time. 886")
#             is_terminated = True
#             log_question("Sounds like it's not a good time to talk. Do you want to continue? Say yes if you want to continue. Say no if you want to quit the current session.")
#             user_correction = get_resp_log()
#             print(type(user_correction))
#             user_correction = user_correction.replace(".", " ")
#             user_correction = user_correction.replace(",", " ")
#             user_correction = user_correction.replace("?", " ")
#             user_correction = user_correction.split(" ")
#             user_correction =[ i.lower() for i in user_correction]
#             if "no" in user_correction:
#                 log_question("Goodbye. We will do the screening in another time. 886")
#                 is_terminated = True
    if is_terminated == True:
        save_filename = "question_lib_v3_" + str(subjectID) + "_"+ str(int(time.time()))+".json"
        with open(save_filename, 'w') as f:
            json.dump(question_lib, f)
        if DLA_terminate == 1:
            pass
        else:
            proceed_MI_CBT(save_filename) ## MI and CBT as the summary
    print("total", cnt)
    item_q_table = new_q_table.copy()


#clear out the original one for next use
for i in range(1, len(question_lib)+1):
    for ind in range(1, len(question_lib[str(i)])+1):
        question_lib[str(i)][str(ind)]["notes"] = []
        question_lib[str(i)][str(ind)]["score"] = []

with open(filename, 'w') as f:
    json.dump(question_lib, f)

#save q table:
    
save_all_question_q_table()
generate_results()


# In[31]:


#proceed_MI_CBT(save_filename) 


# In[32]:


# save_all_question_q_table()
# generate_results()


# In[ ]:





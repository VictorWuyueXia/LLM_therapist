# coding:utf8
 
 
from flask import Flask, render_template, Response
#from flask_sqlalchemy import SQLAlchemy
from flask import jsonify
from flask import request,send_from_directory
from flask_cors import CORS
import pandas as pd

#import pymysql
import json
#import pickle
import requests
import csv
import os
import time
#import pycreate2
import openai



from requests.packages import urllib3
urllib3.disable_warnings()
app = Flask(__name__)
app.debug = True
#def after_request(resp):
#    resp.headers['Access-Control-Allow-Origin'] = '*'
#    return resp

#app.after_request(after_request) 

CORS(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root: @127.0.0.1:3306/longmax"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
_host = "0.0.0.0"
port_number = 8899
_port = port_number
file_name = 'question_lib_v3_8901.json'
last_intervention_time = time.time()
print(last_intervention_time)

f= open(file_name)
 
# returns JSON object as
# a dictionary
question_lib = json.load(f)

activity_dimension = {"weight_change":["DLA_1_weight", "1" , "1"], 
"smoke":["DLA_10_ciga", "10", "2"],
"drink":["DLA_10_alcohol", "10", "1"],
"shower":["DLA_18_hygiene", "18", "1"],
"wash_clothes":["DLA_18_hygiene", "18", "1"],
"exercise":["DLA_21_sports", "19", "1"],
"sudden_fall":["DLA_4_safe", "4", "1"],
"nap":["DLA_5_sleep", "5", "1"],
"cook":["DLA_7_nutrition", "7", "1"]
}
 

#db = SQLAlchemy(app)
data = pd.read_csv('record.csv')

openai_model = "davinci:ft-personal-2022-04-11-23-57-59"

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

    print(cmd)
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
    print(score)
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
    user_correction =user_correction[0:3]
    if "yes" in user_correction:
        category = "DLA"
        score ="Yes"
    if "no" in user_correction:
        category = "DLA"
        score ="No"
    # if "maybe" in user_correction:
    #     category = "DLA"
    #     score ="Maybe"
    
    return category, score


def runPatrol():
    # Create a Create2 Bot
    #port = '/dev/tty.usbserial-DA01NX3Z'  # this is the serial port on my iMac
    port = '/dev/ttyUSB0'  # this is the serial port on my raspberry pi
    baud = {
        'default': 115200,
        'alt': 19200  # shouldn't need this unless you accidentally set it to this
    }

    bot = pycreate2.Create2(port=port, baud=baud['default'])

    # define a movement path
    path = [
        [ 200, 200, 3, 'forward 3'],
        [-200,-200, 3, 'back 3'],
        # [   0,   0, 1, 'stop 1'],
        # [ 100,   0, 2, 'rightt 2'],
        # [   0, 100, 4, 'left 4'],
        # [ 100,   0, 2, 'right 2'],
        # [ 200, 200, 5, 'forward 5'],
        [   0,   0, 1, 'stop 1']
    ]

    bot.start()
    bot.safe()

    # path to move
    for lft, rht, dt, s in path:
        print(s)
        bot.digit_led_ascii(s)
        bot.drive_direct(lft, rht)
        time.sleep(dt)

    print('shutting down ... bye')
    bot.drive_stop()
    time.sleep(0.1)
    
#def generate_prompt(animal):
#    return """Suggest three names for an animal that is a superhero.
#
#Animal: Cat
#Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
#Animal: Dog
#Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
#Animal: {}
#Names:""".format(
#        animal.capitalize()
#    )

def generate_prompt(user_input):
    return """Tell people why do we need mental health care.

    User: What the benefits of having the system
    Answer: With professional help from mental health services, most of these people enjoy improved life quality
    User: Why do we need this system
    Answer: Mental health issues can make life unbearable for the people who have them. At the same time, these problems can have a wider effect on society as a whole, especially when they go untreated or treatment is delayed.
    User: What can the system do
    Answer: The system can provide continous mental health care support by monitoring the daily activities of the patients and provide appropraite intervention
    User:{}
    Names:""".format(
            user_input.capitalize()
        )

def generate_prompt_complete_sentences(user_input):
    return """Complete the sentence if it misses the subject.

    User: am sad
    Answer: I am sad.
    User: really enjoy my work recently.
    Answer: I really enjoy my work recently.
    User: have problem hearing you well.
    Answer: I have problem hearing you well.
    User: I am so depressed.
    Answer: I am so depressed.
    User:{}
    Answer:""".format(
            user_input.capitalize()
        )

def generate_complete_sentences(question_text):
    openai.api_key = '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'
    user_input = question_text
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=generate_prompt_complete_sentences(user_input),
        temperature=0.8,
        max_tokens = 1000,
    )
    results = response.choices[0].text
    # logger.info(response.choices[0].text)
    return results

def get_question():
    while True:
        try:
            data = pd.read_csv('record.csv')
        except:
            pass
        if data["Question_Lock"][0] == 1:
#                             question = question_lib[str(S)][str(question_A)]["question"][0]
#                             Question_text = "Sounds like you did not understand my question. Let me ask it again. "+question
            text = data["Question"][0]
            print(text)
            text = "Congratulation, you figure out a way for yourself. You also work well in dimensions: Support from Social Network, Managing Mood, and Expressing Feelings to Other People. Please contact your primary care or therapist if you have further problems or emergencies. Goodbye. I will follow up later."
            data["Question_Lock"][0] = 0
            print(data["Question_Lock"][0])
            header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
            data.to_csv('record.csv', columns = header)
            break
    return text
                    
def log_resp(user_input):
    while True:
        try:
            data = pd.read_csv('record.csv')
        except:
            pass
        if data["Resp_Lock"][0] == 1:
            data["Resp"][0] = user_input                          
            data["Resp_Lock"][0] = 0
            print(data["Resp_Lock"][0])
            header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
            data.to_csv('record.csv', columns = header)
            break


@app.route('/ControlPath', methods=['POST', 'GET'])
def ControlPath():
    result = None
    if request.method == 'POST':
        data = request.get_data()
        data_str=data.decode('UTF-8')
        #post_data=eval(data_str)
        print(data_str)

        runPatrol()

        res = "Successfully receive the request"
        rsp = Response(json.dumps(res), status=200, content_type="application/json")
        return rsp
    else:
        result = "Invalid request."
        return result, 400, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/GPT3Alexa', methods=['POST', 'GET'])
def GPT3Alexa():
    import re
    result = None
    if request.method == 'POST':
        data = request.get_data()
        #import pdb; pdb.set_trace()
        data_str=data.decode('UTF-8')
        print(data_str)
        P = r'Yes or No: (.+)\. Dimension: (.+)'
        data = re.findall(P, data_str)
        # data_str.replace("=",":")
        # print(data_str)
        # data_json = eval(data_str)
        # user_input = data_json["user_input"]
        # subject_ID = data_json["subject_ID"]
        user_input = data[0][0]
        dimension = data[0][1]
        print(user_input)
        if dimension == "start":
            print("-"*99)

            initial_data = pd.read_csv('record.csv')
            initial_data["Question_Lock"][0] = 1
            #initial_data["Resp_Lock"][0] = 0  
            header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
            initial_data.to_csv('record.csv', columns = header)
           
            print("Check, start")
            next_question = get_question()
            print("question", next_question)

        else:
            if user_input == "None":
                dimension = generate_complete_sentences(dimension)
                log_resp(dimension)
                print(dimension)

                next_question = get_question()
                print(next_question)
            else:
                log_resp(user_input)
                print(user_input)

                next_question = get_question()
                print(next_question)

        end_question = str(next_question).split(".")

        # if " 886" in end_question:
        #     print('check')
        #     initial_data = pd.read_csv('record.csv')
        #     initial_data["Question_Lock"][0] = 0
        #     header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
        #     initial_data.to_csv('record.csv', columns = header)

        #category, score = get_openai_resp(data_str)
        #print("category, score: ", category, score)
        # openai.api_key = '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'
        # res = openai.Completion.create(
        #  model = "davinci:ft-personal-2022-04-11-23-57-59",
        #  #model = 'davinci:ft-columbia-university-2022-02-28-23-00-46',
        #  prompt=data_str+"->",
        #  max_tokens = 17)

        # cmd = res['choices'][0]['text']
        # cmd = cmd.split("}")[0].split(":")[1][2:-1].split(",")
        # response = cmd[0]+" "+cmd[1]
        next_question = next_question.replace("\n\n", "")
        next_question = next_question.replace("\n", "")

        cmd = " "+next_question
        rsp = Response(cmd, status=200, content_type="application/json")
        return rsp
    else:
        result = "Invalid request."
        return result, 400, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/GPT3', methods=['POST', 'GET'])
def GPT3():
    result = None
    if request.method == 'POST':
        data = request.get_data()
        #import pdb; pdb.set_trace()
        data_str=data.decode('UTF-8')
        print(data_str)
        # data_str.replace("=",":")
        # print(data_str)
        data_json = eval(data_str)
        user_input = data_json["user_input"]
        subject_ID = data_json["subject_ID"]

        if user_input == "start":

            initial_data = pd.read_csv('record.csv')
            initial_data["Question_Lock"][0] = 1
            #initial_data["Resp_Lock"][0] = 0  
            header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
            initial_data.to_csv('record.csv', columns = header)
           
            print("Check, start")
            next_question = get_question()
            print("question", next_question)

        else:
            log_resp(user_input)
            print(user_input)

            next_question = get_question()
            print(next_question)

        end_question = str(next_question).split(".")

        if " 886" in end_question:
            print('check')
            initial_data = pd.read_csv('record.csv')
            initial_data["Question_Lock"][0] = 0
            header = ["Question", "Question_Lock", "Resp", "Resp_Lock"]
            initial_data.to_csv('record.csv', columns = header)

        #category, score = get_openai_resp(data_str)
        #print("category, score: ", category, score)
        # openai.api_key = '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'
        # res = openai.Completion.create(
        #  model = "davinci:ft-personal-2022-04-11-23-57-59",
        #  #model = 'davinci:ft-columbia-university-2022-02-28-23-00-46',
        #  prompt=data_str+"->",
        #  max_tokens = 17)

        # cmd = res['choices'][0]['text']
        # cmd = cmd.split("}")[0].split(":")[1][2:-1].split(",")
        # response = cmd[0]+" "+cmd[1]

        cmd = {"subject_ID": subject_ID, "question": next_question}
        rsp = Response(json.dumps(cmd), status=200, content_type="application/json")
        return rsp
    else:
        result = "Invalid request."
        return result, 400, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/ratings', methods=['POST', 'GET'])
def ratings():
    result = None
    if request.method == 'POST':
        data = request.get_data()
        #import pdb; pdb.set_trace()
        data_str=data.decode('UTF-8')
        print(data_str)
        data_json = eval(data_str)
        ratings_0 = data_json["Q0"]
        ratings_1 = data_json["Q1"]
        ratings_2 = data_json["Q2"]
        ratings_3 = data_json["Q3"]
        ratings_4 = data_json["Q4"]
        ratings_5 = data_json["Q5"]
        subject_ID = data_json["subject_ID"]

        
        print(ratings_0, ratings_1, ratings_2, ratings_3, ratings_4, ratings_5, subject_ID)
        msg = str(subject_ID)+","+str(time.time())+str(ratings_0)+","+str(ratings_1)+","+str(ratings_2)+","+str(ratings_3)+","+str(ratings_4)+","+str(ratings_5)+"\n"
        file1 = open("ratings.csv", 'a')
        file1.write(msg)
        file1.close()
        #category, score = get_openai_resp(data_str)
        #print("category, score: ", category, score)
        # openai.api_key = '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'
        # res = openai.Completion.create(
        #  model = "davinci:ft-personal-2022-04-11-23-57-59",
        #  #model = 'davinci:ft-columbia-university-2022-02-28-23-00-46',
        #  prompt=data_str+"->",
        #  max_tokens = 17)

        # cmd = res['choices'][0]['text']
        # cmd = cmd.split("}")[0].split(":")[1][2:-1].split(",")
        # response = cmd[0]+" "+cmd[1]

        cmd = {"success": "success"}
        rsp = Response(json.dumps(cmd), status=200, content_type="application/json")
        return rsp
    else:
        result = "Invalid request."
        return result, 400, {'Content-Type': 'text/plain; charset=utf-8'}



@app.route('/check', methods=['POST', 'GET'])
def check():
    global last_intervention_time
    result = None
    if request.method == 'POST':

        data = request.get_data()
        #import pdb; pdb.set_trace()
        data_str=data.decode('UTF-8')
        print(data_str)

        data_json = eval(data_str)
        activity = data_json["activity"]
        item = activity_dimension[str(activity)][1]
        question = activity_dimension[str(activity)][2]

        score = question_lib[item][question]["Yes"]
        question_lib[item][question]["score"].append(score)

        #subject_ID = data_json["subject_ID"]

        with open(file_name, 'w') as f:
            json.dump(question_lib, f)

        problem_activities = ["drink", "sudden_fall"]

        if activity in problem_activities:
            print("Problem activity: ", activity)
            current_time = time.time()
            if (current_time - last_intervention_time) > 10:
                print("last_intervention_time", last_intervention_time, "current_time", current_time)
                url = 'http://caiti_ros.p.icsl.cc:20000/InterventionRobot'
                flag = 1 
                while flag == 1:
                    x = requests.post(url, json = {"Intervention": "Yes"})
                    print(x.status_code)
                    print("Sent Intervention Request", x)
                    if x.status_code == 200:
                        flag = 0
                        break
                last_intervention_time = time.time()
                
        
        if activity == "weight_change":
            current_time = time.time()
            if (current_time - last_intervention_time) > 30:
                print("last_intervention_time", last_intervention_time, "current_time", current_time)
                message = str(data_json["message"])+"\n"
                file0 = open("weight.csv", 'a')
                file0.write(message)
                file0.close()
                print("Save weight.")
                
                data = pd.read_csv("weight.csv")
                if len(data)>0:
                    previous_weight = data.to_numpy()[-2][0]
                    print(type(previous_weight))
                    previous_weight = int(float(previous_weight))
                    weight_change = previous_weight-int(float(message))
                    print("previous weight", previous_weight, "message", int(float(message)))
                    if abs(weight_change)>60000:
                        print("Problem activity: ", activity)
                        url = 'http://caiti_ros.p.icsl.cc:20000/InterventionRobot'
                        x = requests.post(url, json = {"Intervention": "Yes"})
                        print("Sent Intervention Request", x)
                        last_intervention_time = time.time()

            else:
                previous_weight = "NA"

            cmd = {"success": str(previous_weight)}

        else:

            cmd = {"success": "success"}
        
        
        rsp = Response(json.dumps(cmd), status=200, content_type="application/json")
        return rsp
    else:
        result = "Invalid request."
        return result, 400, {'Content-Type': 'text/plain; charset=utf-8'}

# {“activity”: “weight_change”, “message”: “yes”}
# {“activity”: “smoke”, “message”: “yes”}
# {“activity”: “drink”, “message”: “yes”}
# {“activity”: “shower”, “message”: “yes”}
# {“activity”: “wash_clothes”, “message”: “yes”}
# {“activity”: “exercise”, “message”: “yes”}
# {“activity”: “sudden fall”, “message”: “yes”}
# {“activity”: “nap”, “message”: “yes”}
# {“activity”: “cook”, “message”: “yes”}
@app.route('/HomeCheck', methods=['POST', 'GET'])
def HomeCheck():
    result = None
    if request.method == 'POST':
        data = request.get_data()
        #import pdb; pdb.set_trace()
        data_str=data.decode('UTF-8')
        print(data_str)

        data_json = eval(data_str)
        activity = data_json["activity"]
        start_time = data_json["start_time"]
        end_time = data_json["end_time"]
        msg = str(activity)+","+str(start_time)+","+str(end_time)+" \n"
        
        # log event for question_lib
        item = activity_dimension[str(activity)][1]
        question = activity_dimension[str(activity)][2]

        score = question_lib[item][question]["Yes"]
        question_lib[item][question]["score"].append(score)
        print(question_lib[item][question])

        #subject_ID = data_json["subject_ID"]
                                                                                                                                                                                                                                  
        with open(file_name, 'w') as f:
            json.dump(question_lib, f)
            print("saved event")
        
                                                                                                                                                                                                                                  
        file1 = open("events.csv", 'a')
        file1.write(msg)
        file1.close()

        cmd = {"success": "success to record event"}
        rsp = Response(json.dumps(cmd), status=200, content_type="application/json")
        return rsp
    else:
        result = "Invalid request."
        return result, 400, {'Content-Type': 'text/plain; charset=utf-8'}

# {“activity”: “weight_change”, “start_time”: “1656127564”, "end_time":"1656127664"}
# {“activity”: “smoke”, “start_time”: “1656127564”, "end_time":"1656127664"}
# {“activity”: “drink”, “start_time”: “1656127564”, "end_time":"1656127664"}
# {“activity”: “shower”, “start_time”: “1656127564”, "end_time":"1656127664"}
# {“activity”: “wash_clothes”, “start_time”: “1656127564”, "end_time":"1656127664"}
# {“activity”: “exercise”, “start_time”: “1656127564”, "end_time":"1656127664"}
# {“activity”: “sudden fall”, “start_time”: “1656127564”, "end_time":"1656127664"}
# {“activity”: “nap”, “start_time”: “1656127564”, "end_time":"1656127664"}
# {“activity”: “cook”, “start_time”: “1656127564”, "end_time":"1656127664"}

@app.route('/GPT3Intro', methods=['POST', 'GET'])
def GPT3Intro():
    result = None
    if request.method == 'POST':
        data = request.get_data()
        data_str=data.decode('UTF-8')
        print(data_str)

        # data_json=eval(data_str)
        # print(data_json)
        openai.api_key = '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'
        user_input = " "
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=generate_prompt(user_input),
            temperature=0.6,
            max_tokens = 100,
        )
        
        

        res = response.choices[0].text
        print(res)

        rsp = Response(json.dumps(res), status=200, content_type="application/json")
        return rsp
    else:
        result = "Invalid request."
        return result, 400, {'Content-Type': 'text/plain; charset=utf-8'}


if __name__== '__main__':
    app.run(debug=True, host=_host, port=_port)

import requests
import pandas as pd
import traceback
import time


# def post_text_update(keyword, response):
#     requests.request("POST", 'http://caiti_demo_frontend.p.icsl.cc:20000/update_frontend',
#                      json={'field': 'text', 'data': {'kw': keyword, 'resp': response}}, timeout=1)

def post_text_update(keyword, response):
    requests.request("POST", 'http://caiti_demo_frontend.p.icsl.cc:20000/update_frontend', headers={'Field': 'text'}, json={'kw': keyword, 'resp': response})

def read_resp_question():
    while True:
        try:
            data = pd.read_csv('record.csv')
            keyword = data["Resp"][0]                        
            response = data["Question"][0]
            print(keyword, response)
            post_text_update(keyword, response)
            time.sleep(1)
            # break
        except:
            traceback.print_exc()
read_resp_question()
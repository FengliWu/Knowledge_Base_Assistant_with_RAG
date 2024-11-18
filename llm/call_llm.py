import openai
import json
import requests
import _thread as thread
import base64
import datetime
from dotenv import load_dotenv, find_dotenv
import hashlib
import hmac
import os
import queue
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import zhipuai
from langchain.utils import get_from_dict_or_env

import websocket  # Use websocket_client

def get_completion(prompt :str, model :str, temperature=0.1,api_key=None, secret_key=None, access_token=None, appid=None, api_secret=None, max_tokens=2048):
    '''
    Call the large model to get the reply, support the above three models + gpt
    arguments:
    prompt: input prompt
    model: model name
    temperature: temperature coefficient
    api_key: as name
    secret_key, access_token: required to call Wenxin series models
    appid, api_secret: required to call the Spark series model
    max_tokens: Return the longest sequence
    return: model return, string
    '''

    if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
        return get_completion_gpt(prompt, model, temperature, api_key, max_tokens)
    elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
        return get_completion_wenxin(prompt, model, temperature, api_key, secret_key)
    elif model in ["Spark-1.5", "Spark-2.0"]:
        return get_completion_spark(prompt, model, temperature, api_key, appid, api_secret, max_tokens)
    elif model in ["chatglm_pro", "chatglm_std", "chatglm_lite"]:
        return get_completion_glm(prompt, model, temperature, api_key, max_tokens)
    else:
        return "Incorrect model"
    
def get_completion_gpt(prompt : str, model : str, temperature : float, api_key:str, max_tokens:int):
    
    # Encapsulate OpenAI native interface
    if api_key == None:
        api_key = parse_llm_api_key("openai")
    openai.api_key = api_key
    
    # Specific call
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # Temperature coefficient of model output, controlling the degree of randomness of the output
        max_tokens = max_tokens, # Maximum length of reply
    )
    # Call OpenAI’s ChatCompletion interface to get the response
    return response.choices[0].message.content

def get_access_token(api_key, secret_key):
    """
    Use API Key, Secret Key to obtain access_token, replace the application API Key, application Secret Key in the following examples
    """
    # Specify URL
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    # Set up POST access
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # Obtain the access_token corresponding to the account through POST access
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

def get_completion_wenxin(prompt: str, model: str, temperature: float, api_key:str, secret_key: str):
    # Encapsulate Baidu Wenxin native interface
    if api_key == None or secret_key == None:
        api_key, secret_key = parse_llm_api_key("wenxin")
    # Get access_token
    access_token = get_access_token(api_key, secret_key)
    # Call interface
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={access_token}"
    # Configure POST parameters
    payload = json.dumps({
        "messages": [
            {
                "role": "user",# user prompt
                "content": "{}".format(prompt)# Input prompt
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    # Initiate a request
    response = requests.request("POST", url, headers=headers, data=payload)
    # Returns a Json string
    js = json.loads(response.text)
    return js["result"]

def get_completion_spark(prompt: str, model: str, temperature: float, api_key:str, appid: str, api_secret: str, max_tokens: int):
    if api_key == None or appid == None and api_secret == None:
        api_key, appid, api_secret = parse_llm_api_key("spark")
    
    # Configure different environments for 1.5 and 2
    if model == "Spark-1.5":
        domain = "general"
        Spark_url = "ws://spark-api.xf-yun.com/v1.1/chat" # The address of the v1.5 environment
    else:
        domain = "generalv2" # v2.0 version
        Spark_url = "ws://spark-api.xf-yun.com/v2.1/chat" # The address of the v2.0 environment

    question = [{"role":"user", "content":prompt}]
    response = spark_main(appid,api_key,api_secret,Spark_url,domain,question,temperature, max_tokens)
    return response
def get_completion_glm(prompt : str, model : str, temperature : float, api_key:str, max_tokens : int):
    # 获取GLM回答
    if api_key == None:
        api_key = parse_llm_api_key("zhipuai")
    zhipuai.api_key = api_key
    client = zhipuai.ZhipuAI(api_key=os.environ.get("ZHIPUAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user", "content":prompt}],
        temperature = temperature,
        max_tokens=max_tokens
        )
    return response.choices[0].message.content.strip('"').strip(" ")

# def getText(role, content, text = []):
# # role is the specified role, content is the prompt content
# jsoncon = {}
# jsoncon["role"] = role
# jsoncon["content"] = content
# text.append(jsoncon)
# return text

# Spark API call usage
answer = ""

class Ws_Param(object):
    # initialization
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url
        #Customize
        self.temperature = 0
        self.max_tokens = 2048

    # Generate url
    def create_url(self):
        # Generate timestamp in RFC1123 format
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # Splice strings
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # Combine the requested authentication parameters into a dictionary
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # Splice authentication parameters and generate url
        url = self.Spark_url + '?' + urlencode(v)
        # Here print out the url when establishing the connection. When referring to this demo, you can cancel the comment printed above and compare whether the url generated when using the same parameters is consistent with the url generated by your own code.
        return url


#Handling of websocket errors received
def on_error(ws, error):
    print("### error:", error)


# Receive the processing of websocket closing
def on_close(ws,one,two):
    print(" ")


# Receive the processing of websocket connection establishment
def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, domain= ws.domain,question=ws.question, temperature = ws.temperature, max_tokens = ws.max_tokens))
    ws.send(data)


# Processing of received websocket messages
def on_message(ws, message):
    # print(message)
    data = json.loads(message)
    code = data.header.code
    if code != 0:
        print(f'Request error: {code}, {data}')
        ws.close()
    else:
        choices = data.payload.choices
        print('choices', choices)
        status = choices.status
        content = choices[0].message.content
        print(content,end="")
        global answer
        answer += content
        # print(1)
        if status == 2:
            ws.close()


def gen_params(appid, domain,question, temperature, max_tokens):
    """
    Generate request parameters through appid and user's questions
    """
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "random_threshold": 0.5,
                "max_tokens": max_tokens,
                "temperature" : temperature,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": question
            }
        }
    }
    return data


def spark_main(appid, api_key, api_secret, Spark_url,domain, question, temperature, max_tokens):
    output_queue = queue.Queue()
    def on_message(ws, message):
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            print(f'Request error: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            # print(content, end='')
            output_queue.put(content)
            if status == 2:
                ws.close()

    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.question = question
    ws.domain = domain
    ws.temperature = temperature
    ws.max_tokens = max_tokens
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return ''.join([output_queue.get() for _ in range(output_queue.qsize())])

def parse_llm_api_key(model:str, env_file=None):
    """
    Parse platform parameters through model and env_file
    """
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "openai":
        return env_file["OPENAI_API_KEY"]
    elif model == "wenxin":
        return env_file["wenxin_api_key"], env_file["wenxin_secret_key"]
    elif model == "spark":
        return env_file["spark_api_key"], env_file["spark_appid"], env_file["spark_api_secret"]
    elif model == "zhipuai":
        return get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")
        # return env_file["ZHIPUAI_API_KEY"]
    else:
        raise ValueError(f"model{model} not support!!!")

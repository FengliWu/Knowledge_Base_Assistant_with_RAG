from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
from llm.self_llm import Self_LLM
import json
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket
import queue

class Spark_LLM(Self_LLM):
    # URL
    url : str = "ws://spark-api.xf-yun.com/v1.1/chat"
    # APPID
    appid : str = None
    # APISecret
    api_secret : str = None
    # Domain
    domain :str = "general"
    # max_token
    max_tokens : int = 4096

    def getText(self, role, content, text = []):
        # role is the specified role, content is the prompt content
        jsoncon = {}
        jsoncon["role"] = role
        jsoncon["content"] = content
        text.append(jsoncon)
        return text

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        if self.api_key == None or self.appid == None or self.api_secret == None:
            # Only when all three keys exist can it be called normally
            print("Please fill in Key")
            raise ValueError("Key does not exist")
        # Fill Prompt to Spark format
        question = self.getText("user", prompt)
        # Initiate a request
        try:
            response = spark_main(self.appid,self.api_key,self.api_secret,self.url,self.domain,question, self.temperature, self.max_tokens)
            return response
        except Exception as e:
            print(e)
            print("Request failed")
            return "Request failed"
        
    @property
    def _llm_type(self) -> str:
        return "Spark"

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

        # Perform hmac-sha256 encryption
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


# 收到websocket消息的处理
def on_message(ws, message):
    # print(message)
    data = json.loads(message)
    code = data['header']['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        print(content,end ="")
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
    # print("星火:")
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





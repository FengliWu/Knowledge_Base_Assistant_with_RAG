import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import re

class Chat_QA_chain_self:
    """
    Q&A chain with history
    -model: the name of the called model
    -temperature: temperature coefficient, controlling the randomness of generation
    -top_k: Returns the top k similar documents retrieved
    -chat_history: history, enter a list, the default is an empty list
    -history_len: Controls the most recent history_len conversations retained
    -file_path: the path where the library creation file is located
    -persist_path: vector database persistence path
    -appid: Spark
    -api_key: Parameters that need to be passed for Spark, Baidu Wenxin, OpenAI, and Zhipu
    -Spark_api_secret: Spark secret key
    -Wenxin_secret_key: Wenxin secret key
    -embeddings: embedding model used
    -embedding_key: the secret key of the embedding model used (Zhipu or OpenAI)
    """
    def __init__(self,model:str, temperature:float=0.0, top_k:int=4, chat_history:list=[], file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None,Wenxin_secret_key:str=None, embedding = "openai",embedding_key:str=None):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        #self.history_len = history_len
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key


        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
        
    
    def clear_history(self):
        return self.chat_history.clear()
    
    def change_history_length(self,history_len:int=1):
        """
        Save the history of a specified conversation turn
        Input parameters:
        -history_len: Controls the most recent history_len conversations retained
        -chat_history: current historical conversation record
        Output: Return the most recent history_len conversations
        """
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

 
    def answer(self, question:str=None, temperature = None, top_k = 4):
        """"
        Core method, calling the question and answer chain
        arguments: 
        -question: User question
        """
        
        if len(question) == 0:
            return "", self.chat_history
        
        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature
        llm = model_to_llm(self.model, temperature, self.appid, self.api_key, self.Spark_api_secret,self.Wenxin_secret_key)
        # self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': top_k})  # default similarity，k=4

        qa = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = retriever,
            return_source_documents = True,
            get_chat_history=lambda h : h  # This works!
        )
        
        #print(self.llm)
        
        #  Convert Gradio's chat history format to LangChain's expected format
        langchain_history = [(msg[1], self.chat_history[i+1][1] if i+1 < len(self.chat_history) else "") for i, msg in enumerate(self.chat_history) if i % 2 == 0]
        # Get response from QA chain
        result = qa({"question": question, "chat_history": langchain_history})
        
        # result = qa({"question": question, "chat_history": self.chat_history}) #result contains question, chat_history, answer
        answer = result['answer']
        answer = re.sub(r"\\n", '<br/>', answer)
        self.chat_history.append((question,answer)) # Update history

        return self.chat_history # Return this answer and updated history
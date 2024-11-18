import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_community.embeddings import OpenAIEmbeddings    # 调用 OpenAI 的 Embeddings 模型
from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from database.create_db import create_db,load_knowledge_db
from embedding.call_embedding import get_embedding

def get_vectordb(file_path:str=None, persist_path:str=None, embedding = "m3e", embedding_key:str=None):
    """
    Returns a vector database object
    Input parameters:
    question: the question to be answered
    llm: language model (required parameters), an object
    vectordb: vector database (required parameters), an object
    template: Prompt template (optional parameter) You can design a prompt template yourself, and some are used by default.
    Embedding: You can use embedding such as zhipuai. If you do not enter this parameter, openai embedding will be used by default. Be careful not to enter the wrong api_key at this time.
    """
    embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)
    if os.path.exists(persist_path):
        contents = os.listdir(persist_path)
        if len(contents) == 0:
            print("Directory is empty")
            vectordb = create_db(file_path, persist_path, embedding)
            # #presit_knowledge_db(vectordb)
            # vectordb = load_knowledge_db(persist_path, embedding)
        else:
            print("Directory is not empty")
            vectordb = load_knowledge_db(persist_path, embedding)
    else: # The directory does not exist, create the vector database from scratch
        vectordb = create_db(file_path, persist_path, embedding)
        #presit_knowledge_db(vectordb)
        vectordb = load_knowledge_db(persist_path, embedding)

    return vectordb
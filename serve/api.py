from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
#Import function module directory
sys.path.append("../")
from qa_chain.QA_chain_self import QA_chain_self

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

app = FastAPI() # Create api object

template = """
Use the following context to answer the final question. If you don't know the answer, just say you don't know and don't try to make up the answer. Use a maximum of three sentences. Try to keep your answers concise and to the point. Use the same language as the user's prompts in your response.
    {context}
    Question: {question}
    Useful answer:
    """


# Define a data model for receiving data in POST requests
class Item(BaseModel):
    prompt : str # user prompt
    model : str = "gpt-3.5-turbo"# The model used
    temperature : float = 0.1# Temperature coefficient
    if_history : bool = False # Whether to use the historical dialogue function
    #API_Key
    api_key: str = None
    #Secret_Key
    secret_key : str = None
    # access_token
    access_token: str = None
    #APPID
    appid : str = None
    #APISecret
    Spark_api_secret : str = None
    #Secret_key
    Wenxin_secret_key : str = None
    # Database path
    db_path : str = "/Users/lta/Desktop/llm-universe/data_base/vector_db/chroma"
    # Source file path
    file_path : str = "/Users/lta/Desktop/llm-universe/data_base/knowledge_db"
    # prompt template
    prompt_template : str = template
    # Template variable
    input_variables : list = ["context","question"]
    #Embdding
    embedding : str = "m3e"
    #TopK
    top_k : int = 5
    # embedding_key
    embedding_key : str = None

@app.post("/")
async def get_response(item: Item):

    # First determine the chain that needs to be called
    if not item.if_history:
        # Call Chat chain
    # return item.embedding_key
        if item.embedding_key == None:
            item.embedding_key = item.api_key
        chain = QA_chain_self(model=item.model, temperature=item.temperature, top_k=item.top_k, file_path=item.file_path, persist_path=item.db_path,
    appid=item.appid, api_key=item.api_key, embedding=item.embedding, template=template, Spark_api_secret=item.Spark_api_secret, Wenxin_secret_key=item.Wenxin_secret_key, embedding_key=item.embedding_key)

        response = chain.answer(question = item.prompt)
    
        return response
    
    # Due to immediacy issues in the API, historical chains cannot be supported.
    else:
        return "API does not support history chain"
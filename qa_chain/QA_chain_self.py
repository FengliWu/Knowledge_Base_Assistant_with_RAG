import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import re

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge_db')
DEFAULT_PERSIST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vector_db')


class QA_chain_self():
    """"
    Q&A chain without history
    -model: the name of the called model
    -temperature: temperature coefficient, controlling the randomness of generation
    -top_k: Returns the top k similar documents retrieved
    -file_path: the path where the library creation file is located
    -persist_path: vector database persistence path
    -appid: Spark needs to be entered
    -api_key: required by all models
    -Spark_api_secret: Spark secret key
    -Wenxin_secret_key: Wenxin secret key
    -embeddings: embedding model used
    -embedding_key: the secret key of the embedding model used (Zhipu or OpenAI)
    -template: You can customize the prompt template. If there is no input, the default prompt template default_template_rq will be used.
    """

    #The default prompt template used by the prompt built based on the recall results and query
    default_template_rq = """
        Use the following context to answer the final question. If you don't know the answer, just say you don't know and don't try to make up the answer. Use a maximum of three sentences. Try to keep your answers concise and to the point. Use the same language as the user's prompts in your response.
        {context}
        Question: {question}
        Useful answer:
        """

    def __init__(self, model:str, temperature:float=0.0, top_k:int=4,  file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None,Wenxin_secret_key:str=None, embedding = "openai",  embedding_key = None, template=default_template_rq):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key
        self.template = template
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
        self.llm = model_to_llm(self.model, self.temperature, self.appid, self.api_key, self.Spark_api_secret,self.Wenxin_secret_key)

        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=self.template)
        self.retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': self.top_k})  #default similarity，k=4
        print('----------------RETRIEVER', self.retriever)

        # Customized QA chain
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                        retriever=self.retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":self.QA_CHAIN_PROMPT})
        print(self.qa_chain)

    #Default prompt template used by large model-based Q&A prompt
    #default_template_llm = """Please answer the following questions: {question}"""
           
    def answer(self, question:str=None, temperature = None, top_k = 4):
        """
        Core method, calling the question and answer chain
        arguments:
        -question: user questions
        """

        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature
            
        if top_k == None:
            top_k = self.top_k

        result = self.qa_chain({"query": question, "temperature": temperature, "top_k": top_k})
        answer = result["result"]
        answer = re.sub(r"\\n", '<br/>', answer)
        return answer   


if __name__ == "__main__":
    print('---------------------------QA_CHAIN_SELF')
    qa_chain = QA_chain_self(model="chatglm_std", temperature=0.0, top_k=4, file_path=DEFAULT_DB_PATH, persist_path=DEFAULT_PERSIST_PATH, appid=None, api_key=None, Spark_api_secret=None, Wenxin_secret_key=None, embedding="m3e", embedding_key=None, template=QA_chain_self.default_template_rq)
    print('---------------------------QACHAIN: ', qa_chain)
    question = "How to apply for an account for Shanghai University's ML platform?"
    answer = qa_chain.answer(question)
    print('---------------------------ANSWER: ', answer)
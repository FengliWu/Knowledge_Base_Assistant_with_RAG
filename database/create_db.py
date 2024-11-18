import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import re
import tempfile
import numpy as np
from dotenv import load_dotenv, find_dotenv
from embedding.call_embedding import get_embedding
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'knowledge_db')
DEFAULT_PERSIST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vector_db')


def get_files(dir_path):
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        print(len(filenames))
        for filename in filenames:
            file_list.append(os.path.join(filepath, filename))
    return file_list


def file_loader(file, loaders):
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name
    if not os.path.isfile(file):
        [file_loader(os.path.join(file, f), loaders) for f in  os.listdir(file)]
        return
    file_type = file.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file))
    elif file_type == 'txt':
        loaders.append(UnstructuredFileLoader(file))
    return


def create_db_info(files=DEFAULT_DB_PATH, embeddings="m3e", persist_directory=DEFAULT_PERSIST_PATH):
    if embeddings == 'openai' or embeddings == 'm3e' or embeddings =='zhipuai':
        vectordb = create_db(files, persist_directory, embeddings)
    return ""


def create_db(files=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings="m3e"):
    """
    This function is used to load PDF files, split the document, generate the embedded vector of the document, and create a vector database.

    parameter:
    file: The path to store the file.
    embeddings: the model used to produce Embeddings

    return:
    vectordb: created database.
    """
    if files == None:
        return "can't load empty file"
    if type(files) != list:
        files = [files]
    loaders = []
    [file_loader(file, loaders) for file in files]
    docs = []
    for loader in loaders:
        if loader is not None:
            docs.extend(loader.load())
    
    # Split document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    if type(embeddings) == str:
        embeddings = get_embedding(embedding=embeddings)

    # Load database
    vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory # Allows us to save the persist_directory directory to disk
    )

    # vectordb.persist()
    return vectordb


# def presit_knowledge_db(vectordb):
#     """
#     This function is used to persist the vector database.

#     parameter:
#     vectordb: The vector database to be persisted.
#     """
#     vectordb.persist()


def load_knowledge_db(path, embeddings):
    """
    This function is used to load the vector database.

    parameter:
    path: The path to the vector database to load.
    embeddings: embedding model used by vector database.

    return:
    vectordb: loaded database.
    """
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb


if __name__ == "__main__":
    create_db(embeddings="m3e")
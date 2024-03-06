#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chromadb
import numpy as np
from chromadb.config import Settings
from numpy import dot
from numpy.linalg import norm

import gpt
import pdf_common
from gpt import ModelGPT


def cos_sim(a, b):
    """ 余弦距离计算，越大越相近
    :param a:
    :param b:
    :return:
    """
    return dot(a, b) / (norm(a) * norm(b))


def l2(a, b):
    """ 欧式距离，越小越相似
    :param a:
    :param b:
    :return:
    """
    x = np.asarray(a) - np.asarray(b)
    return norm(x)


class VectorDBEngine:
    def __init__(self, collection_name, embedding_fn):
        # 基于内存的向量数据库
        # chroma_client = chromadb.Client(Settings(allow_reset=True))
        # 基于http的向量数据库
        # chroma_client = chromadb.HttpClient(host='localhost', port=8000)

        # 本地持久化的向量数据库
        self.chroma_client = chromadb.PersistentClient(settings=Settings(allow_reset=True), path="./pdf.db")
        self.chroma_client.reset()

        # create collection
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name,
                                                                      metadata={"hnsw:space": "l2"})  # 默认l2
        self.embedding_fn = embedding_fn
        self.collection_name = collection_name
        self.id = 0

    def data_load(self, para_graphs):
        """ load data to collection
        :param para_graphs:
        :return:
        """
        self.__private_add_documents(para_graphs)

    def __private_add_documents(self, documents):
        """ add documents into vectordb(chromadb)
        :param documents:
        :return:
        """
        print("add documents count is {}".format(len(documents)))
        chunk_size = 20
        sub_documents = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
        for sub_document in sub_documents:
            print("documents add. id={}, len={}".format(self.id, len(sub_document)))
            self.collection.add(embeddings=self.embedding_fn(sub_document),  # vector
                                documents=sub_document,  # text
                                ids=[f"id{i + self.id}" for i in range(len(sub_document))]  # id
                                )
            self.id += len(sub_document)

        print("documents add finish")

    def search(self, query, top_n):
        """ search from chromadb
        :param query:
        :param top_n:
        :return:
        """
        results = self.collection.query(query_embeddings=self.embedding_fn([query]), n_results=top_n)
        return results


def vector_rag():
    collection_name = "muhui_test"
    llm = ModelGPT()
    # 1. 创建一个向量数据库对象，文档的切分使用交叠的方式
    vector_db = VectorDBEngine(collection_name, llm.get_embeddings)
    print("connect vectordb success")

    # 2. extract text from pdf
    pdf_utils = pdf_common.PdfUtils("llama2.pdf")
    para_graphs = pdf_utils.extract_text_from_pdf(select_page_numbers=[1, 4], min_line_length=10)
    chunks = pdf_common.split_text(para_graphs, 300, 100)
    print(f"extract_text_from_pdf success. len={len(para_graphs)}, split_chunk={len(chunks)}")

    # 3. load data into vector_db
    vector_db.data_load(chunks)
    print("load data success")

    # 4. search
    # user_query = "Llama 2有多少参数"
    user_query = "Llama 2有对话版吗"
    # user_query = "llama 2可以商用吗？"
    search_results = vector_db.search(user_query, 2)
    for para in search_results['documents'][0]:
        print("已知信息： " + para + "\n")

    # 5. llm
    prompt = gpt.generate_gpt_prompt(info=search_results, query=user_query)
    print("===Prompt===\n" + prompt)
    response = llm.get_completion(prompt)
    print("===Response===\n" + response)


vector_rag()

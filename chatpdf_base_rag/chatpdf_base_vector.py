#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    chatpdf base rag, use VectorDB(`chromadb`) to search
"""
import numpy as np
from chromadb.config import Settings
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import CrossEncoder

import chromadb
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
    def __init__(self, collection_name, embedding_fn, recreate_collection_if_have=True):
        # 基于内存的向量数据库
        # chroma_client = chromadb.Client(Settings(allow_reset=True))
        # 基于http的向量数据库
        # chroma_client = chromadb.HttpClient(host='localhost', port=8000)

        # 本地持久化的向量数据库
        self.chroma_client = chromadb.PersistentClient(settings=Settings(allow_reset=True), path="chromadb")
        # 如果有数据，则重新创建
        if recreate_collection_if_have:
            self.chroma_client.reset()

        # create collection
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name,
                                                                      metadata={"hnsw:space": "l2"})  # 默认l2
        self.embedding_fn = embedding_fn
        self.collection_name = collection_name
        self.id = 0

    def load_data(self, para_graphs):
        """ add documents into vectordb(chromadb)
        :param para_graphs:
        :return:
        """
        # print("add documents count is {}".format(len(para_graphs)))
        chunk_size = 20
        sub_documents = [para_graphs[i:i + chunk_size] for i in range(0, len(para_graphs), chunk_size)]
        for sub_document in sub_documents:
            # print("documents add. id={}, len={}".format(self.id, len(sub_document)))
            self.collection.add(embeddings=self.embedding_fn(sub_document),  # vector
                                documents=sub_document,  # text
                                ids=[f"id{i + self.id}" for i in range(len(sub_document))]  # id
                                )
            self.id += len(sub_document)

    def search(self, query, top_n):
        """ search from chromadb
        :param query:
        :param top_n:
        :return:
        """
        results_ = self.collection.query(query_embeddings=self.embedding_fn([query]), n_results=top_n)
        results = list(zip(results_["documents"][0], results_["ids"][0]))
        return [{"text": result[0], "id": result[1], "rank": i, } for i, result in enumerate(results)]


def chatpdf_base_vector():
    rank_model = CrossEncoder('/Users/haizhi/workspace/huggingface/ms-marco-MiniLM-L-6-v2', max_length=512)

    collection_name = "demo"
    llm = ModelGPT()
    # 1. 创建一个向量数据库对象，文档的切分使用交叠的方式
    vector_db = VectorDBEngine(collection_name, llm.get_embeddings)
    print("connect vectordb success")

    # 2. extract text from pdf
    pdf_utils = pdf_common.PdfUtils("llama2.pdf")
    para_graphs = pdf_utils.extract_text_from_pdf(select_page_numbers=[1, 3], min_line_length=10)
    # 优化1: 对段落按照指定 chunk 进行 split，前向扩展100个词的上下文
    chunks = pdf_common.split_text(para_graphs, 300, 100)
    print(f"extract_text_from_pdf success. len={len(para_graphs)}, split_chunk={len(chunks)}")

    # 3. load data into vector_db
    vector_db.load_data(chunks)
    print("load data success")

    # 4. search
    top_n = 3
    # user_query = "Llama 2有多少参数"
    # user_query = "Llama 2有对话版吗"
    # user_query = "llama 2可以商用吗？"
    user_query = "how safe is llama 2"
    search_results = vector_db.search(user_query, top_n * 2)  # 召回2N条记录
    search_results = [result["text"] for result in search_results]
    print("已知信息: {}".format(search_results))

    # 优化2: 过召回一部分数据，使用排序模型(`ms-marco-MiniLM-L-6-v2`)对结果集进行二次排序，取排序后的 top_n 交给llm
    print("start rank")
    scores = rank_model.predict([(user_query, doc) for doc in search_results])
    sorted_search_results = sorted(zip(scores, search_results), key=lambda x: x[0], reverse=True)
    sorted_search_results = [tup[1] for tup in sorted_search_results]
    sorted_search_results = sorted_search_results[0:max(top_n, len(search_results))]
    print("sorted_search_results: {}".format(search_results))

    # 5. llm
    prompt = gpt.generate_gpt_prompt(info=sorted_search_results, query=user_query)
    print("===Prompt===\n" + prompt)
    response = llm.chat(prompt)
    print("===Response===\n" + response)

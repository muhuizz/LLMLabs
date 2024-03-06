#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from time import sleep

from elasticsearch7 import Elasticsearch, helpers

import gpt
import pdf_common
from gpt import ModelGPT

warnings.simplefilter("ignore")  # Suppress some ES (Elasticsearch) warnings


es_endpoint = ''
es_user = ''
es_passwd = ''


class ESEngine:
    def __init__(self, es_endpoint, user, passwd):
        self.hosts = es_endpoint
        self.user = user
        self.passwd = passwd
        self.engine = Elasticsearch(self.hosts, http_auth=(self.user, self.passwd))

    def search(self, index_name, query_string, top_n=3):
        """ 从 es 中查询文本
        :param index_name:
        :param query_string:
        :param top_n:
        :return:
        """
        search_query = {"match": {"keywords": pdf_common.to_keywords(query_string)}}
        print("query={}, keyword={}".format(query_string, search_query))
        res = self.engine.search(index=index_name, query=search_query, size=top_n)
        return [hit["_source"]["text"] for hit in res["hits"]["hits"]]

    def load_data(self, index_name, para_graphs, recreate_index_if_have=True):
        """ 数据导入，索引初始化
        :param index_name:
        :param para_graphs:
        :param recreate_index_if_have:
        :return:
        """
        # 1. drop old index if you need
        if self.engine.indices.exists(index=index_name) and recreate_index_if_have:
            self.engine.indices.delete(index=index_name)
            print(f"index {index_name} deleted success, will recreate")
        if not self.engine.indices.exists(index=index_name):
            self.engine.indices.create(index=index_name)
            print(f"index {index_name} created success")

        # 2. load data into es
        actions = [{"_index": index_name, "_source": {"keywords": pdf_common.to_keywords(para), "text": para}} for para
                   in para_graphs]
        helpers.bulk(self.engine, actions)

        # 3. wait es refresh interval. default 1s
        sleep(1)


def rag_base_es():
    # 1. connect es
    es = ESEngine(es_endpoint, es_user, es_passwd)
    print("connect es success")

    # 2. extract text from pdf
    # pdf_utils = pdf_common.PdfUtils("llama2.pdf")
    # para_graphs = pdf_utils.extract_text_from_pdf(min_line_length=10)
    # print(f"extract_text_from_pdf success. len={len(para_graphs)}")

    # 3. load data into es
    index_name = "muhui_rag_index"
    # es.load_data(index_name, para_graphs)
    # print("load data success")

    # 4. search from es
    query = "how many parameters does llama 2 have?"
    # query = "llama2 有多少参数"  # 文档没有中文，所以这里匹配不到中文
    search_results = es.search(index_name, query, 2)
    print("search finished. len =", len(search_results))
    for r in search_results:
        print("search_results: {}".format(r) + "\n")

    # 5. get response from llm
    prompt = gpt.generate_gpt_prompt(info=search_results, query=query)
    print("===Prompt===\n" + prompt)
    llm = ModelGPT()
    response = llm.get_completion(prompt)
    print("===Response===\n" + response)


rag_base_es()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import warnings
from time import sleep

import nltk
from elasticsearch7 import Elasticsearch, helpers
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from openai import OpenAI
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer
from dotenv import load_dotenv, find_dotenv

warnings.simplefilter("ignore")  # Suppress some ES (Elasticsearch) warnings
nltk.download('punkt')  # 英文切词、词根、切句等方法
nltk.download('stopwords')  # 英文停用词库
_ = load_dotenv(find_dotenv())
gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
es_endpoint = ''
es_user = ''
es_passwd = ''

def extract_text_from_pdf(filename, skip_page_numbers=None, min_line_length=1):
    """ 从 PDF 文件中（按指定页码）提取文字
    :param filename:
    :param skip_page_numbers: 要跳过的页
    :param min_line_length: 一行不超过 min_line_length 个单词，认为该段结束
    :return:
    """
    para_graphs = []
    buffer = ''
    full_text = ''
    # 按页提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if skip_page_numbers is not None and i not in skip_page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
            # else:
            #     print(f'ignore {element}')

    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            para_graphs.append(buffer)
            buffer = ''
    if buffer:
        para_graphs.append(buffer)
    return para_graphs


def get_gpt_completion(prompt, model="gpt-3.5-turbo"):
    """ gpt prompt 接口封装
    :param prompt:
    :param model:
    :return:
    """
    messages = [{"role": "user", "content": prompt}]
    response = gpt_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content


def to_keywords(input_string):
    """ Given an input text string, return a list of keywords (supports English only)
    :param input_string:
    :return:
    """
    # Replace all non-alphanumeric characters with spaces using regular expressions
    no_symbols = re.sub(r'[^a-zA-Z0-9\s]', ' ', input_string)
    word_tokens = word_tokenize(no_symbols)
    # Load the stop word list
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    # Stem and filter out stop words
    filtered_sentence = [ps.stem(w)
                         for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)


def es_search(es, index_name, query_string, top_n=3):
    """ 从 es 中查询文本
    :param es:
    :param index_name:
    :param query_string:
    :param top_n:
    :return:
    """
    search_query = {
        "match": {
            "keywords": to_keywords(query_string)
        }
    }
    print("keyword: ", to_keywords(query_string))
    res = es.search(index=index_name, query=search_query, size=top_n)
    return [hit["_source"]["text"] for hit in res["hits"]["hits"]]


def es_init(es, index_name):
    """ 数据导入，索引初始化
    :param es:
    :param index_name:
    :return:
    """
    # 1. drop old index if you need
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # 2. create
    es.indices.create(index=index_name)
    print(f"index {index_name} created success")

    # 3. load data into es
    para_graphs = extract_text_from_pdf("llama2.pdf", min_line_length=10)
    print(f"extract text success. len={len(para_graphs)}")
    actions = [
        {
            "_index": index_name,
            "_source": {
                "keywords": to_keywords(para),
                "text": para
            }
        }
        for para in para_graphs
    ]
    helpers.bulk(es, actions)
    print(f"bulk success")

    # 5. wait es refresh interval. default 1s
    sleep(1)


prompt_template = """
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
__INFO__

用户问：
__QUERY__

请用中文回答用户问题。
"""


def generate_gpt_prompt(**kwargs):
    """ Generate prompt based on the template
    :param kwargs:
    :return:
    """
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        # replace info & query
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt


def es_rag():
    # 1. connect es
    es = Elasticsearch(
        hosts=[es_endpoint],
        http_auth=(es_user, es_passwd),
    )

    # 2. init es
    index_name = "mh_demo_index20240229"
    # es_init(es, index_name)

    # 3. search from es
    query = "how many parameters does llama 2 have?"
    search_results = es_search(es, index_name, query, 2)
    print("search finished. len=", len(search_results))
    for r in search_results:
        print(r + "\n")

    # 4. llm
    prompt = generate_gpt_prompt(info=search_results, query=query)
    print("===Prompt===")
    print(prompt)

    # 5. get response from llm
    response = get_gpt_completion(prompt)
    print("===Response===")
    print(response)


es_rag()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from llm_model import LLMModel

_ = load_dotenv(find_dotenv())

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


class ModelGPT(LLMModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        :param model: support "gpt-3.5-turbo", "gpt-4"
        """
        self.model = model
        self.gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

    def get_completion(self, prompt):
        """ gpt prompt 接口封装
        :param prompt:
        :return:
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.gpt_client.chat.completions.create(model=self.model, messages=messages, temperature=0)
        return response.choices[0].message.content

    def get_embeddings(self, texts, model="text-embedding-ada-002", dimensions=None):
        """ 文本向量化，openai 度embedding接口
        :param texts:
        :param model:
        :param dimensions: 向量维度
        :return:
        """
        if model == "text-embedding-ada-002":
            dimensions = None
        if dimensions:
            data = self.gpt_client.embeddings.create(input=texts, model=model, dimensions=dimensions).data
        else:
            data = self.gpt_client.embeddings.create(input=texts, model=model).data
        return [x.embedding for x in data]

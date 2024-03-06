# LLMLabs

一些我在学习大模型过程中的代码示例


# Introduction

* rag_base_es.py
    一个基于 es 的 pdf 助手，可以回答基于 pdf 内容的问题，没有提供向量索引仅适用于英文文档和英文问题
* rag_base_vector.py
    一个基于 vector 的 pdf 助手，可以回答基于 pdf 内容的问题，提供基于 gpt text-embedding-ada-002 的向量模型提供向量检索的功能;
    文本经过二次split，对上下文进行了拼接，提高了检索的准确率，但是对于长文本的检索效果有改善

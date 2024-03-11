# LLMLabs

一些我在学习大模型过程中的代码示例

# Introduction

## chatpdf_base_rag

提供了基于 es、vectordb、以及混合检索模式的不同 chatpdf 助手的实现，可以回答基于 pdf 内容的问题，提供基于 gpt
text-embedding-ada-002 的向量模型提供向量检索的功能

```shell
python3 chatpdf.py
```

#### chatpdf_base_es

* 基于 es 的 chatpdf 助手，可以回答基于 pdf 内容的问题
* 没有提供向量索引，需要关键词完全匹配，不支持跨语言提问

#### chatpdf_base_vector

* 一个基于 vector 的 pdf 助手，可以回答基于 pdf 内容的问题
* 提供基于 gpt text-embedding-ada-002 的向量模型提供向量检索的功能 
* 文本经过二次split，对上下文进行了拼接，提高了检索的准确率，对于长文本的检索效果有改善
* 提供基于 ms-marco-MiniLM-L-6-v2 的二次排序功能

#### chatpdf_base_vector
* 基于 Reciprocal Rank Fusion（RRF）的混合检索模型，针对一些专有场景（专有名词较多）效果较好
  

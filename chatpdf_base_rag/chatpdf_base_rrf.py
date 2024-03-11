import json

from chatpdf_base_es import *
from chatpdf_base_vector import *

documents = ["李某患有肺癌，癌细胞已转移", "刘某肺癌I期", "张某经诊断为非小细胞肺癌III期", "小细胞肺癌是肺癌的一种"]


def rrf(ranks, k=1):
    """ 计算 RRF 得分，可以考虑为不同的结果集增加权重
    :param ranks:
    :param k:
    :return:
    """
    ret = {}
    # 遍历每次的排序结果
    for rank in ranks:
        # 遍历排序中每个元素
        for id, val in rank.items():
            if id not in ret:
                ret[id] = {"score": 0, "text": val["text"]}
            # 计算 RRF 得分
            ret[id]["score"] += 1.0 / (k + val["rank"])
    # 按 RRF 得分排序，并返回
    return dict(sorted(ret.items(), key=lambda item: item[1]["score"], reverse=True))


def chatpdf_base_rrf():
    # 1. connect es & vectorDB
    es_index_name = "es_rrf"
    llm = ModelGPT()
    es = ESEngine(es_endpoint, es_user, es_passwd, pdf_common.to_keywords_chinese)
    vector_db = VectorDBEngine(collection_name="demo", embedding_fn=llm.get_embeddings)
    print("connect es&vectordb success")

    # 2. data load
    es.load_data(es_index_name, para_graphs=documents)
    vector_db.load_data(documents)

    # 3. search
    query = "非小细胞肺癌的患者"
    es_search_result = es.search(es_index_name, query, 3)
    vectordb_search_result = vector_db.search(query, 3)

    # 4. preprocess
    es_search_result = {res["id"]: {"text": res["text"], "rank": res["rank"]} for res in es_search_result}
    vectordb_search_result = {res["id"]: {"text": res["text"], "rank": res["rank"]} for res in vectordb_search_result}

    print("es: {}".format(es_search_result))
    print("vector: {}".format(vectordb_search_result))

    # 5. rerank
    reranked = rrf([es_search_result, vectordb_search_result])
    print(json.dumps(reranked, indent=2, ensure_ascii=False))

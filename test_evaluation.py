import json
import numpy as np
from pipeline.elastic_pipeline import ElasticPipeline
from pipeline.qdrant_pipeline import QdrantPipeline
# ---------- Dummy Vector DB (contoh implementasi) ----------


def evaluate(db, dataset):
    """
    db: instance of AbstractDB
    dataset: list of dict seperti yang kamu kasih
    embedding_func: fungsi untuk convert text menjadi vector
    """

    # 2. Hitung skor
    total_score = 0
    for data in dataset:
        for query in data["query"]:
            _, sparse_result = db.search_data(query, top_k=5)
            # cek posisi gold doc
            for idx, result in enumerate(sparse_result):
                if result['id'] == int(data["gold_doc_ids"][0]):
                    total_score += (idx + 1)  # rank 1 = +1, rank2 = +2, dst
                    print(f"Query: {query} | Found gold doc at rank {idx+1}")
                    break
    return total_score


# ---------- Contoh penggunaan ----------
if __name__ == "__main__":
    print("Evaluating Elastic DB...")
    dataset = json.load(
        open("data/clean/eval_cookpad.json", "r", encoding="utf-8"))
    print("Total queries to evaluate:", len(dataset)*5)

    qdrant = QdrantPipeline(collection_name="test-bang", mode="sparse")
    elastic = ElasticPipeline(index_name="test-bang", mode="sparse")

    for db in [qdrant, elastic]:
        print(f"Evaluating {db.__class__.__name__}...")
        score = evaluate(elastic, dataset[:10])  # batasi 10 data dulu
        print("Total evaluation score:", score)
        print("-" * 30)

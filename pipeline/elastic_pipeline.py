import csv
from typing import List, Dict, Any
from elasticsearch import Elasticsearch, helpers
from settings import Settings as env

from pipeline.elastic_generateVector import GenerateDenseVector, GenerateSparseVectors


class ElasticPipeline:
    def __init__(self, client_url: str = "http://localhost:9200", index_name: str = "default", mode: str = "hybrid"):
        self.es = Elasticsearch(client_url, api_key=env.ES_LOCAL_API_KEY)
        self.index = index_name
        self.mode = mode.lower()
        self.sparse_generator = GenerateSparseVectors(model_type="opensearch")
        self.dense_generator = GenerateDenseVector()

    # --- Utility ---
    def check_index_existence(self) -> bool:
        return self.es.indices.exists(index=self.index)

    # --- Create Index ---
    def create_new_index(self) -> Dict[str, Any]:
        if self.check_index_existence():
            return {"status": 400, "message": "Index already exists"}

        try:
            # Konfigurasi mapping berdasarkan mode
            properties = {
                "id": {"type": "integer"},
                "category": {"type": "keyword"},
                "url": {"type": "keyword"},
                "title": {"type": "text"},
                "image": {"type": "keyword"},
                "ingredients": {"type": "text"},
                "steps": {"type": "text"}
            }

            if self.mode in ["dense", "hybrid"]:
                properties["dense_vector_ingre"] = {
                    "type": "dense_vector", "dims": 1024, "similarity": "cosine"}
                properties["dense_vector_all"] = {
                    "type": "dense_vector", "dims": 1024, "similarity": "cosine"}

            if self.mode in ["sparse", "hybrid"]:
                properties["sparse_vector_ingre"] = {"type": "sparse_vector"}

            self.es.indices.create(
                index=self.index,
                mappings={"properties": properties}
            )
            return {"status": 200, "message": f"Index '{self.index}' created successfully."}
        except Exception as e:
            return {"status": 500, "message": f"Failed to create index: {e}"}

    # --- Generate Embeddings ---
    def generate_embeddings_from_file(self, path_files: str, column: str = "text"):
        dense_generator, sparse_generator = None, None
        dense_vectors, sparse_vectors, dense_vectors_all = [], [], []

        try:
            if self.mode in ["dense", "hybrid"]:
                dense_generator = GenerateDenseVector()
            if self.mode in ["sparse", "hybrid"]:
                sparse_generator = GenerateSparseVectors(
                    model_type="opensearch")

            with open(path_files, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)

            print(
                f"Processing {len(data)} rows from {path_files} in mode '{self.mode}'")

            for item in data:
                text_value = item.get(column, "").strip()
                text_all = item.get("all_text", "").strip()
                if not text_value and not text_all:
                    continue

                payload = {
                    "id": int(item["id"]),
                    "category": item.get("category"),
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "image": item.get("image"),
                    "ingredients": item.get("ingredients"),
                    "steps": item.get("steps"),
                }

                # Dense embeddings
                if self.mode in ["dense", "hybrid"]:
                    dense_vec = dense_generator.get_dense_embedding(text_value)
                    dense_all = dense_generator.get_dense_embedding(text_all)
                    if dense_vec:
                        dense_vectors.append(
                            {"payload": payload, "dense_vector_ingre": dense_vec})
                    if dense_all:
                        dense_vectors_all.append(
                            {"payload": payload, "dense_vector_all": dense_all})

                # Sparse embeddings
                if self.mode in ["sparse", "hybrid"]:
                    sparse_vals = sparse_generator.get_sparse_vector(text_value)[
                        0]
                    if sparse_vals.get("indices") and sparse_vals.get("values"):
                        sparse_vectors.append({
                            "payload": payload,
                            "sparse_vector_ingre": {
                                str(i): float(v) for i, v in zip(sparse_vals["indices"], sparse_vals["values"])
                            }
                        })

            print(
                f"Generated {len(dense_vectors)} dense | {len(sparse_vectors)} sparse vectors")
            return dense_vectors, sparse_vectors, dense_vectors_all

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [], [], []

    # --- Upsert Documents ---
    def upsert_data(self, dense_vectors=None, sparse_vectors=None, dense_vectors_all=None):
        if not self.check_index_existence():
            raise RuntimeError("Index does not exist. Please create it first.")

        actions = []

        if self.mode == "dense":
            for d, da in zip(dense_vectors, dense_vectors_all):
                doc = d["payload"]
                doc["dense_vector_ingre"] = d["dense_vector_ingre"]
                doc["dense_vector_all"] = da["dense_vector_all"]
                actions.append({"_index": self.index, "_source": doc})

        elif self.mode == "sparse":
            for s in sparse_vectors:
                doc = s["payload"]
                doc["sparse_vector_ingre"] = s["sparse_vector_ingre"]
                actions.append({"_index": self.index, "_source": doc})

        elif self.mode == "hybrid":
            for d, s, da in zip(dense_vectors, sparse_vectors, dense_vectors_all):
                doc = d["payload"]
                doc["dense_vector_ingre"] = d["dense_vector_ingre"]
                doc["dense_vector_all"] = da["dense_vector_all"]
                doc["sparse_vector_ingre"] = s["sparse_vector_ingre"]
                actions.append({"_index": self.index, "_source": doc})

        try:
            helpers.bulk(self.es, actions)
            print(f"Indexed {len(actions)} documents into '{self.index}'")
        except Exception as e:
            print(f"Error during bulk insert: {e}")

    # --- Search Query ---
    def search_data(self, query: str, top_k: int = 3):
        """
        Melakukan pencarian ke Elasticsearch dan memisahkan hasilnya menjadi dua:
        - dense_results: hasil dari dense vector (script_score)
        - sparse_results: hasil dari sparse vector (native sparse query)
        """
        if not self.check_index_existence():
            raise RuntimeError("Index does not exist. Please create it first.")

        mode = self.mode.lower()
        dense_results, sparse_results = [], []

        # === Dense Search ===
        if mode in ["dense", "hybrid"]:
            dense_vec = self.dense_generator.get_dense_embedding(query)

            if dense_vec:
                dense_query = {
                    "size": top_k,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'dense_vector_ingre') + 1.0",
                                "params": {"query_vector": dense_vec},
                            },
                        }
                    },
                }

                resp_dense = self.client.search(
                    index=self.collection_name, body=dense_query
                )

                dense_results = [
                    {
                        "id": hit["_source"].get("id"),
                        "score": hit["_score"],
                        "title": hit["_source"].get("title"),
                        "category": hit["_source"].get("category"),
                        "image": hit["_source"].get("image"),
                        "ingredients": hit["_source"].get("ingredients"),
                        "steps": hit["_source"].get("steps"),
                    }
                    for hit in resp_dense["hits"]["hits"]
                ]

        # === Sparse Search ===
        if mode in ["sparse", "hybrid"]:
            sparse_vec = self.sparse_generator.get_sparse_vector(query)[0]

            if sparse_vec and sparse_vec.get("indices") and sparse_vec.get("values"):
                query_vector = {
                    str(i): float(v)
                    for i, v in zip(sparse_vec["indices"], sparse_vec["values"])
                }

                sparse_query = {
                    "size": top_k,
                    "query": {
                        "sparse_vector": {
                            "field": "sparse_vector_ingre",
                            "query_vector": query_vector,
                        }
                    },
                }

                resp_sparse = self.es.search(
                    index=self.index, body=sparse_query
                )

                sparse_results = [
                    {
                        "id": hit["_source"].get("id"),
                        "score": hit["_score"],
                        "title": hit["_source"].get("title"),
                        "category": hit["_source"].get("category"),
                        "image": hit["_source"].get("image"),
                        "ingredients": hit["_source"].get("ingredients"),
                        "steps": hit["_source"].get("steps"),
                    }
                    for hit in resp_sparse["hits"]["hits"]
                ]

        return dense_results, sparse_results

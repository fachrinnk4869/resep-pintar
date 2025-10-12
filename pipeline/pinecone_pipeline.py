import os
import json
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import numpy as np

from pipeline.pinecone_generateVector import PineconeVectorGenerator

# === ENV ===
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NAME_PINECONE_DENSE = os.getenv("NAME_PINECONE_DENSE")
NAME_PINECONE_SPARSE = os.getenv("NAME_PINECONE_SPARSE")
NAMESPACE = os.getenv("NAMESPACE")              # ingredient
NAMESPACE2 = os.getenv("NAMESPACE2")            # all-text
EMBED_DIM = int(os.getenv("EMBED_DIM") or 1024)

RECIPES_FOLDER = "data/clean"

pc = Pinecone(api_key=PINECONE_API_KEY)

# === Helper to build lookup ===
@lru_cache(maxsize=1)
def _build_recipe_lookup(folder_path: str):
    lookup = {}
    if not folder_path or not os.path.isdir(folder_path):
        return lookup

    try:
        for entry in os.scandir(folder_path):
            if not entry.is_file() or not entry.name.endswith(".json"):
                continue
            with open(entry.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for obj in items:
                rid = obj.get("id")
                if not rid:
                    continue
                lookup[rid] = {
                    "url": obj.get("url"),
                    "title": obj.get("title"),
                    "image": obj.get("image"),
                    "ingredients": obj.get("ingredients"),
                    "steps": obj.get("steps"),
                    "category": obj.get("category"),
                }
    except Exception as e:
        print(f"Failed to build lookup: {e}")

    return lookup


# === Pinecone Pipeline ===
class PineconePipeline:
    """Hybrid RAG pipeline for Pinecone (dense + sparse + metadata lookup)."""

    def __init__(self, bm25_model_path: str = "pipeline/model/bm25_params.json"):
        self.vector_gen = PineconeVectorGenerator(bm25_model_path)
        self.index_dense = pc.Index(name=NAME_PINECONE_DENSE)
        self.index_sparse = pc.Index(name=NAME_PINECONE_SPARSE)

    # === CREATE INDEX ===
    def create_indexes(self):
        if not pc.has_index(NAME_PINECONE_DENSE):
            print("Creating dense index...")
            pc.create_index(
                name=NAME_PINECONE_DENSE,
                vector_type="dense",
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        if not pc.has_index(NAME_PINECONE_SPARSE):
            print("Creating sparse index...")
            pc.create_index(
                name=NAME_PINECONE_SPARSE,
                vector_type="sparse",
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        print("Pinecone indexes ready.")

    # === UPSERT ===
    def upsert_data(self, dense_vectors, sparse_vectors, namespace: str = NAMESPACE):
        if not dense_vectors and not sparse_vectors:
            print("⚠️ No vectors to upsert.")
            return

        print(f"📤 Upserting {len(dense_vectors)} dense and {len(sparse_vectors)} sparse to '{namespace}'...")

        for i in range(0, len(dense_vectors), 100):
            self.index_dense.upsert(vectors=dense_vectors[i:i+100], namespace=namespace)
        for i in range(0, len(sparse_vectors), 100):
            self.index_sparse.upsert(vectors=sparse_vectors[i:i+100], namespace=namespace)

        print("Upsert completed.")

    # === SEARCH ===
    def search_dense(self, text: str, top_k: int = 20):
        vec = self.vector_gen.dense_model.get_embedding(text)
        if not vec:
            return []
        res = self.index_dense.query(
            namespace=NAMESPACE, vector=vec, top_k=top_k, include_metadata=True
        )
        matches = res.get("matches", []) or []
        # Map to simple dicts (id, score, category) to be consistent for fusion
        out = []
        for m in matches:
            out.append({
                "id": m.get("id"),
                "score": float(m.get("score") or 0.0),
                "category": (m.get("metadata") or {}).get("category")
            })
        return out

    def search_sparse(self, text: str, top_k: int = 20):
        sp = self.vector_gen.sparse_model.encode(text, query_type="search")
        if not sp:
            return []
        res = self.index_sparse.query(
            namespace=NAMESPACE, sparse_vector=sp, top_k=top_k, include_metadata=True
        )
        matches = res.get("matches", []) or []
        out = []
        for m in matches:
            out.append({
                "id": m.get("id"),
                "score": float(m.get("score") or 0.0),
                "category": (m.get("metadata") or {}).get("category")
            })
        return out

    # === FETCH ===
    def fetch_dense_by_ids(self, ids: list, query_vec=None):
        if not ids:
            return {}, {}
        fetch_res = self.index_dense.fetch(ids=ids, namespace=NAMESPACE) or {}
        vectors_map = getattr(fetch_res, "vectors", None) or {}
        out, sim = {}, {}
        for _id, rec in vectors_map.items():
            vals = (rec or {}).get("values")
            if vals:
                out[_id] = vals
                # Safe cosine: only compute when both vectors look valid
                if query_vec and np.linalg.norm(vals) > 0 and np.linalg.norm(query_vec) > 0:
                    sim[_id] = self.cosine_similarity(vals, query_vec)
        return out, sim

    def fetch_alltext_by_ids(self, ids: list):
        if not ids or not NAMESPACE2:
            return {}
        fetch_res = self.index_dense.fetch(ids=ids, namespace=NAMESPACE2) or {}
        return getattr(fetch_res, "vectors", None) or {}

    # === RRF ===
    def rrf_fusion(self, dense_results, sparse_results, k=60, top_n=20):
        # If one list is empty, just return the other.
        if not dense_results and not sparse_results:
            return []
        if dense_results and not sparse_results:
            return dense_results[:top_n]
        if sparse_results and not dense_results:
            return sparse_results[:top_n]

        # Standard RRF when both present
        scores = defaultdict(float)
        for rank, res in enumerate(dense_results, 1):
            if res and res.get("id"):
                scores[res["id"]] += 1 / (k + rank)
        for rank, res in enumerate(sparse_results, 1):
            if res and res.get("id"):
                scores[res["id"]] += 1 / (k + rank)

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Build lookup to merge content back
        all_results = {r["id"]: r for r in dense_results + sparse_results if r.get("id")}
        return [all_results[i] for i, _ in fused[:top_n]]

    # === UTIL ===
    @staticmethod
    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # === AUTO ===
    def generate_and_upsert_from_file(self, file_path: str, column: str = "text", namespace: str = NAMESPACE):
        dense_vecs, sparse_vecs = self.vector_gen.generate_embeddings_from_json(file_path, column)
        self.upsert_data(dense_vecs, sparse_vecs, namespace)
        return len(dense_vecs), len(sparse_vecs)

    # === MAIN HYBRID RAG ===
    def search_and_fetch_full(self, query: str, top_k: int = 10, similarity_threshold: float = 0.0):
        """Hybrid search (dense + sparse) + fetch vectors + enrich with metadata.
        NOTE: default threshold = 0.0 to avoid ‘0 results’ surprises.
        """
        dense_res = self.search_dense(query, top_k)
        sparse_res = self.search_sparse(query, top_k)
        fused = self.rrf_fusion(dense_res, sparse_res, top_n=top_k)

        if not fused:
            return []

        ids = [r["id"] for r in fused if r.get("id")]
        if not ids:
            return []

        query_vec = self.vector_gen.dense_model.get_embedding(query)
        id_to_dense, id_to_sim = self.fetch_dense_by_ids(ids, query_vec)
        id_to_all = self.fetch_alltext_by_ids(ids)
        recipe_lookup = _build_recipe_lookup(RECIPES_FOLDER)

        results = []
        for r in fused:
            _id = r["id"]
            meta = recipe_lookup.get(_id, {})  # fill from JSON files
            # use cosine sim if available, else fallback to Pinecone score
            sim = float(id_to_sim.get(_id, r.get("score", 0.0)))

            enriched = {
                "id": _id,
                "similarity": sim,
                "category": meta.get("category"),
                "url": meta.get("url"),
                "title": meta.get("title"),
                "image": meta.get("image"),
                "ingredients": meta.get("ingredients"),
                "steps": meta.get("steps"),
                # include vectors but DO NOT include sparse_values or raw values fields
                "dense_ingre": id_to_dense.get(_id),
                "dense_all": (id_to_all.get(_id) or {}).get("values"),
            }

            # only filter if you really want to; default threshold is 0.0
            if sim >= similarity_threshold:
                results.append(enriched)

        return results
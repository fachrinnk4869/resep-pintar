import csv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct
from typing import List, Dict, Any
from pipeline.qdrant_generateVector import GenerateDenseVector, GenerateSparseVectors


class QdrantPipeline:
    def __init__(self, client_url: str = "http://localhost:6333", collection_name: str = "default", mode: str = "hybrid"):
        self.client = QdrantClient(url=client_url)
        self.collection_name = collection_name
        self.mode = mode
        self.sparse_generator = GenerateSparseVectors(model_type="opensearch")
        self.dense_generator = GenerateDenseVector()

    def check_collection_existence(self) -> bool:
        return self.client.collection_exists(collection_name=self.collection_name)

    def create_new_collection(self) -> Dict[str, Any]:
        """
        Create a Qdrant collection dynamically:
        - hybrid: dense + sparse
        - dense: dense only
        - sparse: sparse only
        """
        try:
            if self.check_collection_existence():
                return {"status": 400, "message": "Collection already exists"}

            # Normalize mode name
            mode = self.mode.lower()
            if mode not in ["hybrid", "dense", "sparse"]:
                return {"status": 400, "message": f"Invalid mode '{mode}'. Must be one of: hybrid, dense, sparse."}

            # Prepare configs based on mode
            vector_config = None
            sparse_config = None

            if mode == "dense":
                vector_config = {
                    "cookpad-ingre-dense": VectorParams(size=1024, distance=Distance.COSINE),
                    "cookpad-all-dense": VectorParams(size=1024, distance=Distance.COSINE)
                }

            elif mode == "sparse":
                sparse_config = {
                    "cookpad-ingre-sparse": SparseVectorParams()
                }

            elif mode == "hybrid":
                vector_config = {
                    "cookpad-ingre-dense": VectorParams(size=1024, distance=Distance.COSINE),
                    "cookpad-all-dense": VectorParams(size=1024, distance=Distance.COSINE)
                }
                sparse_config = {
                    "cookpad-ingre-sparse": SparseVectorParams()
                }

            # Create collection dynamically
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config,
                sparse_vectors_config=sparse_config
            )

            return {
                "status": 200,
                "message": f"Collection '{self.collection_name}' created successfully with mode '{mode}'."
            }

        except Exception as e:
            return {"status": 500, "message": f"Failed to create collection: {e}"}

    def generate_embeddings_from_file(
        self,
        path_files: str,
        column: str = "text"
    ):
        """
        Membaca data CSV dan menghasilkan embedding berdasarkan mode koleksi:
        - dense: hanya dense vector (SiliconFlow Qwen3)
        - sparse: hanya sparse vector (OpenSearch model)
        - hybrid: dense + sparse vector
        """
        try:
            mode = self.mode.lower()
            if mode not in ["dense", "sparse", "hybrid"]:
                raise ValueError(
                    f"Invalid mode '{mode}'. Must be one of: dense, sparse, hybrid")

            dense_generator = None
            sparse_generator = None
            dense_vectors, sparse_vectors, dense_vectors_all = [], [], []

            if mode in ["dense", "hybrid"]:
                dense_generator = GenerateDenseVector()
            if mode in ["sparse", "hybrid"]:
                sparse_generator = GenerateSparseVectors(
                    model_type="opensearch")

            # --- Baca CSV ---
            with open(path_files, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)

            print(
                f"Processing {len(data)} rows from {path_files} in mode '{mode}'")

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

                # --- Dense Embedding ---
                if mode in ["dense", "hybrid"]:
                    dense_values = dense_generator.get_dense_embedding(
                        text_value)
                    if dense_values:
                        dense_vectors.append({
                            "id": int(item["id"]),
                            "vector": dense_values,
                            "payload": payload
                        })

                    # text all
                    dense_values_all = dense_generator.get_dense_embedding(
                        text_all)
                    if dense_values_all:
                        dense_vectors_all.append({
                            "id": int(item["id"]),
                            "vector": dense_values_all
                        })

                # --- Sparse Embedding ---
                if mode in ["sparse", "hybrid"]:
                    sparse_vals = sparse_generator.get_sparse_vector(text_value)[
                        0]
                    if sparse_vals and sparse_vals.get("indices") and sparse_vals.get("values"):
                        sparse_vectors.append({
                            "id": int(item["id"]),
                            "vector": models.SparseVector(
                                indices=sparse_vals["indices"],
                                values=sparse_vals["values"]
                            ),
                            "payload": payload
                        })

            print(
                f"Generated: {len(dense_vectors)} dense | {len(sparse_vectors)} sparse vectors")
            return dense_vectors, sparse_vectors, dense_vectors_all

        except FileNotFoundError:
            print(f"Error: File {path_files} not found.")
        except Exception as e:
            print(f"Unexpected error while generating embeddings: {e}")
        return [], [], []

    def upsert_data(
        self,
        dense_vectors: List[Dict] = None,
        sparse_vectors: List[Dict] = None,
        dense_vectors_all: List[Dict] = None
    ):
        """
        Upsert data ke Qdrant sesuai mode:
        - dense: hanya cookpad-ingre-dense
        - sparse: hanya cookpad-ingre-sparse
        - hybrid: keduanya
        """
        if not self.check_collection_existence():
            raise RuntimeError(
                "Collection does not exist. Please create it first.")

        mode = self.mode.lower()
        if mode not in ["dense", "sparse", "hybrid"]:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: dense, sparse, hybrid")

        points = []

        if mode == "dense" and dense_vectors and dense_vectors_all:
            for d, d_all in zip(dense_vectors, dense_vectors_all):
                points.append(
                    PointStruct(
                        id=d["id"],
                        payload=d["payload"],
                        vector={
                            "cookpad-ingre-dense": d["vector"],
                            "cookpad-all-dense": d_all["vector"]
                        }
                    )
                )

        elif mode == "sparse" and sparse_vectors:
            for s in sparse_vectors:
                points.append(
                    PointStruct(
                        id=s["id"],
                        payload=s["payload"],
                        vector={"cookpad-ingre-sparse": s["vector"]}
                    )
                )

        elif mode == "hybrid" and dense_vectors and sparse_vectors and dense_vectors_all:
            for d, s, d_all in zip(dense_vectors, sparse_vectors, dense_vectors_all):
                points.append(
                    PointStruct(
                        id=d["id"],
                        payload=d["payload"],
                        vector={
                            "cookpad-ingre-dense": d["vector"],
                            "cookpad-ingre-sparse": s["vector"],
                            "cookpad-all-dense": d_all["vector"]
                        },
                    )
                )

        if not points:
            print("No points to upsert. Check input vectors or mode.")
            return

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(
            f"Upserted {len(points)} points in '{self.collection_name}' using mode '{mode}'")

    def search_data(
        self,
        query: str,
        top_k: int = 1
    ):
        if not self.check_collection_existence():
            raise RuntimeError(
                "Collection does not exist. Please create it first.")

        mode = self.mode.lower()
        if mode not in ["dense", "sparse", "hybrid"]:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: dense, sparse, hybrid")

        dense_vectors, sparse_vectors = [], []
        dense_results, sparse_results = [], []
        if mode in ["dense", "hybrid"]:
            dense_vectors = self.dense_generator.get_dense_embedding(query)
            if dense_vectors:
                dense_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=dense_vectors,
                    using="cookpad-ingre-dense",
                    limit=top_k,
                    with_vectors=True
                ).points

                for r in dense_result:
                    dense = r.vector.get("cookpad-ingre-dense")
                    dense_all = r.vector.get("cookpad-all-dense")
                    dense_results.append({
                        "id": r.id,
                        "category": r.payload.get("category"),
                        "title": r.payload.get("title"),
                        "image": r.payload.get("image"),
                        "ingredients": r.payload.get("ingredients"),
                        "steps": r.payload.get("steps"),
                        "cookpad-ingre-dense": dense,
                        "vector_all": dense_all
                    })

        if mode in ["sparse", "hybrid"]:
            sparse_vectors = self.sparse_generator.get_sparse_vector(query)[0]
            if sparse_vectors and sparse_vectors.get("indices") and sparse_vectors.get("values"):
                sparse_result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=models.SparseVector(
                        indices=sparse_vectors.get("indices"),
                        values=sparse_vectors.get("values")
                    ),
                    using="cookpad-ingre-sparse",
                    limit=top_k,
                    with_vectors=True
                ).points

                for r in sparse_result:
                    dense = r.vector.get("cookpad-ingre-dense")
                    dense_all = r.vector.get("cookpad-all-dense")
                    sparse_results.append({
                        "id": r.id,
                        "category": r.payload.get("category"),
                        "title": r.payload.get("title"),
                        "image": r.payload.get("image"),
                        "ingredients": r.payload.get("ingredients"),
                        "steps": r.payload.get("steps"),
                        "cookpad-ingre-dense": dense,
                        "vector_all": dense_all
                    })

        return dense_results, sparse_results

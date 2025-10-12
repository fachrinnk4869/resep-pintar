import os
import requests
from pathlib import Path
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
import json

load_dotenv()

SILICONFLOW_URL_EMBEDDING = os.getenv("SILICONFLOW_URL_EMBEDDING")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
EMBED_DIM = int(os.getenv("EMBED_DIM")) if os.getenv("EMBED_DIM") else 1024

if not SILICONFLOW_URL_EMBEDDING:
    raise RuntimeError("SILICONFLOW_URL_EMBEDDING is not set")
if not SILICONFLOW_API_KEY:
    raise RuntimeError("SILICONFLOW_API_KEY is not set")

HEADERS = {
    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
    "Content-Type": "application/json",
}

# === BM25 Model ===
class BM25Model:
    """Wrapper class untuk manajemen BM25Encoder."""

    def __init__(self, model_path: Optional[str] = None, stem: bool = False):
        self.model = BM25Encoder(stem=stem)
        self.model_path = Path(model_path) if model_path else None

    def load(self, strict: bool = False):
        """Load BM25 parameter file jika tersedia."""
        if not self.model_path or not self.model_path.exists():
            if strict:
                raise FileNotFoundError(f"BM25 params not found: {self.model_path}")
            print("BM25 params not found, using fresh model.")
            return self.model

        try:
            self.model.load(str(self.model_path))
            print(f"BM25 model loaded from {self.model_path}")
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to load BM25 params: {e}")
            print(f"Failed to load BM25 model, using empty model.")
        return self.model

    def train_and_save(self, corpus: list[str]):
        """Train model baru dan simpan ke file path."""
        if not corpus:
            raise ValueError("Corpus is empty. Cannot train BM25 model.")
        self.model.fit(corpus)
        if self.model_path:
            self.model.dump(str(self.model_path))
            print(f"BM25 model saved to {self.model_path}")
        return self.model

    def get_model(self) -> BM25Encoder:
        """Ambil instance encoder."""
        return self.model
# === Dense Embedding Model ===
class DenseEmbeddingModel:
    """Class pembungkus API SiliconFlow untuk menghasilkan dense vector."""
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", dim: int = EMBED_DIM):
        self.model_name = model_name
        self.dim = dim

    def get_embedding(self, text: str) -> Optional[List[float]]:
        if not text:
            return None

        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float",
            "dimensions": self.dim,
        }

        try:
            response = requests.post(
                SILICONFLOW_URL_EMBEDDING,
                json=payload,
                headers=HEADERS,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()

            if "data" not in data or not data["data"]:
                raise ValueError("Invalid response: missing 'data'")
            embedding = data["data"][0].get("embedding")
            if not embedding:
                raise ValueError("Missing 'embedding' in response.")
            return embedding

        except requests.exceptions.RequestException as e:
            print(f"HTTP Error while getting embedding: {e}")
        except ValueError as e:
            print(f"Data format error: {e}")
        except Exception as e:
            print(f"Unexpected error while getting embedding: {e}")

        return None

# === Sparse Embedding Model ===
class SparseEmbeddingModel:
    """Class pembungkus BM25Encoder untuk sparse vector (BM25-based)."""

    def __init__(self, bm25_model: BM25Encoder):
        self.bm25_model = bm25_model

    def encode(self, text: str, query_type: str = "search") -> Dict[str, Any]:
        if not text:
            return {}
        if query_type == "upsert":
            return self.bm25_model.encode_documents([text])[0]
        return self.bm25_model.encode_queries([text])[0]

# === Pinecone Vector Generator ===
class PineconeVectorGenerator:
    """Class untuk menghasilkan dense & sparse vectors dari file JSON."""

    def __init__(self, bm25_model_path: str = "pipeline/model/bm25_params.json"):
        self.bm25 = BM25Model(bm25_model_path).load()
        self.dense_model = DenseEmbeddingModel()
        self.sparse_model = SparseEmbeddingModel(self.bm25)

    def generate_embeddings_from_json(
        self, file_path: str, column: str = "text"
    ) -> Tuple[List[Dict], List[Dict]]:
        dense_vectors = []
        sparse_vectors = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Invalid file format: expected list of dict objects")

            print(f"Processing {len(data)} records from {file_path}")

            for item in data:
                text_value = item.get(column, "")
                if not text_value:
                    continue

                metadata = {
                    "category": item.get("category"),
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "image": item.get("image"),
                    "ingredients": item.get("ingredients"),
                    "steps": item.get("steps"),
                }

                # Dense vector
                dense_vec = self.dense_model.get_embedding(text_value)
                if dense_vec:
                    dense_vectors.append(
                        {"id": item["id"], "values": dense_vec, "metadata": metadata}
                    )

                # Sparse vector
                sparse_vals = self.sparse_model.encode(text_value, query_type="upsert")
                if sparse_vals.get("indices") and sparse_vals.get("values"):
                    sparse_vectors.append(
                        {"id": item["id"], "sparse_values": sparse_vals, "metadata": metadata}
                    )

            print(f"Generated {len(dense_vectors)} dense and {len(sparse_vectors)} sparse vectors.")
            return dense_vectors, sparse_vectors

        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Could not decode JSON from {file_path}")
        except Exception as e:
            print(f"Unexpected error while generating embeddings: {e}")
        return [], []

    def generate_from_texts(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict]]:
        dense_vecs, sparse_vecs = [], []

        for text in texts:
            dense_vec = self.dense_model.get_embedding(text)
            if dense_vec:
                dense_vecs.append(dense_vec)

            sparse_vec = self.sparse_model.encode(text, query_type="upsert")
            sparse_vecs.append(sparse_vec)

        return dense_vecs, sparse_vecs

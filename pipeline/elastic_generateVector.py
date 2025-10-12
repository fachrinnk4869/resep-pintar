import os
import requests
from dotenv import load_dotenv
import torch
import json
from typing import List, Dict, Union
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import hf_hub_download


class GenerateSparseVectors:
    def __init__(self, model_type: str = "splade", device: str = None):
        self.model_type = model_type.lower()
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type == "splade":
            self._init_splade()
        elif self.model_type == "opensearch":
            self._init_opensearch()
        elif self.model_type == "bm25":
            self.vectorizer = None
        else:
            raise ValueError(
                "Unsupported model_type: choose 'splade', 'opensearch', or 'bm25'")

    # -------------------- SPLADE --------------------
    def _init_splade(self):
        self.model_id = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_id).to(self.device)
        self.model.eval()

    def _compute_splade(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt",
                                truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**tokens).logits
            relu_log = torch.log(1 + torch.relu(logits))
            weighted = relu_log * tokens.attention_mask.unsqueeze(-1)
            max_val, _ = torch.max(weighted, dim=1)
        return max_val.squeeze(0).cpu()

    # -------------------- OPENSEARCH --------------------
    def _init_opensearch(self):
        self.model_id = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_id).to(self.device)
        self.model.eval()

        # Load IDF
        local_cached_path = hf_hub_download(
            repo_id=self.model_id, filename="idf.json")
        with open(local_cached_path) as f:
            idf = json.load(f)
        self.idf_vector = torch.zeros(self.tokenizer.vocab_size)
        for token, weight in idf.items():
            token_id = self.tokenizer._convert_token_to_id_with_added_voc(
                token)
            self.idf_vector[token_id] = weight

        self.special_token_ids = [self.tokenizer.vocab[tok]
                                  for tok in self.tokenizer.special_tokens_map.values()]

    def _compute_opensearch(self, text: str) -> torch.Tensor:
        feature = self.tokenizer([text], padding=True, truncation=True,
                                 return_tensors='pt', return_token_type_ids=False).to(self.device)
        with torch.no_grad():
            output = self.model(**feature)[0]
        values, _ = torch.max(
            output * feature["attention_mask"].unsqueeze(-1), dim=1)
        values = torch.log(1 + torch.relu(values))
        values[:, self.special_token_ids] = 0
        max_values = values.max(dim=-1)[0].unsqueeze(1) * 0.1
        sparse_vec = values * (values > max_values)
        return sparse_vec.squeeze(0).cpu()

    # -------------------- BM25 --------------------
    def _init_bm25_vectorizer(self, texts: List[str]):
        self.vectorizer = TfidfVectorizer(
            analyzer="word", stop_words="english", max_features=30000, smooth_idf=True)
        self.vectorizer.fit(texts)

    # -------------------- PUBLIC API --------------------
    def get_sparse_vector(self, texts: Union[str, List[str]]) -> List[Dict[str, List[float]]]:
        """
        Return list of {'indices': [...], 'values': [...]} ready for elastic SparseVector
        """
        if isinstance(texts, str):
            texts = [texts]

        results = []
        if self.model_type == "splade":
            for text in texts:
                vec = self._compute_splade(text)
                indices = torch.nonzero(vec, as_tuple=True)[0].tolist()
                values = vec[indices].tolist()
                results.append({"indices": indices, "values": values})

        elif self.model_type == "opensearch":
            for text in texts:
                vec = self._compute_opensearch(text)
                indices = torch.nonzero(vec, as_tuple=True)[0].tolist()
                values = vec[indices].tolist()
                results.append({"indices": indices, "values": values})

        elif self.model_type == "bm25":
            if self.vectorizer is None:
                self._init_bm25_vectorizer(texts)
            tfidf_matrix = self.vectorizer.transform(texts)
            for row in tfidf_matrix:
                results.append({"indices": row.indices.tolist(),
                               "values": row.data.tolist()})

        return results


class GenerateDenseVector:
    def __init__(self):
        """
        Inisialisasi konfigurasi untuk SiliconFlow Embedding API.
        Pastikan .env berisi:
        SILICONFLOW_URL_EMBEDDING=<url endpoint>
        SILICONFLOW_API_KEY=<api key>
        EMBED_DIM=1024
        """
        load_dotenv()

        self.api_url = os.getenv("SILICONFLOW_URL_EMBEDDING")
        self.api_key = os.getenv("SILICONFLOW_API_KEY")
        self.embed_dim = int(os.getenv("EMBED_DIM") or 1024)

        if not self.api_url:
            raise RuntimeError(
                "SILICONFLOW_URL_EMBEDDING is not set in environment")
        if not self.api_key:
            raise RuntimeError("SILICONFLOW_API_KEY is not set in environment")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_dense_embedding(self, text: str, dim_size: int = None) -> List[float]:
        """
        Mengambil dense embedding dari 1 text string.
        Return: list[float]
        """
        dim = dim_size or self.embed_dim
        payload = {
            "model": "Qwen/Qwen3-Embedding-8B",
            "input": text,
            "encoding_format": "float",
            "dimensions": dim,
        }

        try:
            response = requests.post(
                self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            # Validasi format respons
            if "data" not in data or not data["data"]:
                raise ValueError(
                    "Response JSON tidak memiliki field 'data' atau kosong.")
            if "embedding" not in data["data"][0]:
                raise ValueError(
                    "Field 'embedding' tidak ditemukan di 'data[0]'.")

            return data["data"][0]["embedding"]

        except requests.exceptions.RequestException as e:
            print(f"[HTTP Error] {e}")
        except ValueError as e:
            print(f"[Data Error] {e}")
        except Exception as e:
            print(f"[Unexpected Error] {e}")

        return []

    def get_dense_embeddings(self, texts: Union[str, List[str]], dim_size: int = None) -> List[List[float]]:
        """
        Mendapatkan embedding untuk 1 atau banyak teks sekaligus.
        """
        if isinstance(texts, str):
            texts = [texts]

        return [self.get_dense_embedding(t, dim_size) for t in texts]

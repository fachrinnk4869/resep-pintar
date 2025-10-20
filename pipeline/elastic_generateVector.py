import os
import requests
from dotenv import load_dotenv
import torch
import json
from typing import List, Dict, Union
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer


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
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str = None):
        """
        Generate dense embedding menggunakan model HuggingFace: intfloat/multilingual-e5-small
        """
        self.model_name = model_name
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔹 Loading model: {self.model_name} on {self.device}")
        self.model = SentenceTransformer('intfloat/multilingual-e5-small')

    def get_dense_embedding(self, text: str) -> List[float]:
        """
        Menghasilkan dense embedding dari 1 teks
        """
        # Model e5 butuh prefix 'query:' atau 'passage:' tergantung konteks (search/retrieval)
        # Untuk umum, gunakan 'query: ' agar konsisten
        text = text.strip()
        if not text.startswith("query:") and not text.startswith("passage:"):
            text = "query: " + text

        embeddings = self.model.encode(text, normalize_embeddings=True)
        return embeddings

    def get_dense_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Mendapatkan embedding untuk 1 atau banyak teks sekaligus.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            emb = self.get_dense_embedding(text)
            embeddings.append(emb)

        return embeddings

from typing import List, Optional
import numpy as np
from pipeline.get_embedding import get_dense_embeddings
from mapping import MappingOutput


class AlgorithmClass:
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.reset()

    # ----------------------------------------
    # Reset and initialization
    # ----------------------------------------
    def reset(self):
        """Reset semua variabel state internal."""
        self.selected: List[MappingOutput] = []
        self.user_pref: Optional[np.ndarray] = None
        self.candidates: List[MappingOutput] = []
        self.current_recipe: Optional[dict] = None
        self.current_item_embeddding: Optional[np.ndarray] = None

    # ----------------------------------------
    # Embedding handling
    # ----------------------------------------
    def generate_recipe_embeddings(self, recipes: List[dict]):
        """Generate embedding dari list of recipes."""
        embeddings_all = [r.get('vector_all', []) for r in recipes]
        embeding_ingredients = [
            r.get('cookpad-ingre-dense', []) for r in recipes]
        return embeddings_all, embeding_ingredients

    def generate_input_embedding(self, text_input: str):
        """Generate embedding untuk input user (text query)."""
        return get_dense_embeddings(text_input)

    def mapping_input(self, text_input: str, embedding_input: Optional[np.ndarray] = None):
        """Set user preference berdasarkan input text atau embedding yang sudah ada."""
        self.user_pref = embedding_input or self.generate_input_embedding(
            text_input)
        return text_input, self.user_pref

    # ----------------------------------------
    # Mapping & candidate generation
    # ----------------------------------------
    def mapping_output(self, recipes, embeddings=None, embeding_ingredients=None):
        """Buat daftar MappingOutput kandidat dari daftar resep."""
        if embeddings is None or embeding_ingredients is None:
            embeddings, embeding_ingredients = self.generate_recipe_embeddings(
                recipes)

        selected_ids = {
            item.id for item in self.selected if item.id is not None}

        self.candidates = [
            MappingOutput(
                id=recipe.get('id'),
                title=recipe['title'],
                image=recipe.get('image'),
                ingredients=recipe.get('ingredients'),
                steps=recipe.get('steps'),
                ingredients_vector=emb_ing,
                all_vector=emb_all,
                final_vector=self.rerank_ingredients(
                    emb_all, emb_ing, lambd=0.5)
            )
            for recipe, emb_all, emb_ing in zip(recipes, embeddings, embeding_ingredients)
            if recipe.get('id') not in selected_ids
        ]

    # ----------------------------------------
    # Recipe selection & rating
    # ----------------------------------------
    def get_recipe(self):
        return self.current_recipe

    def first_generate_recipe(self):
        """Ambil rekomendasi awal tanpa rating sebelumnya."""
        return self.rating_recipe(0)

    def rating_recipe(self, rating: float):
        """Update preferensi user berdasarkan rating dan rekomendasikan resep baru."""
        if self.current_item_embeddding is not None:
            self.user_pref = self.update_user_pref(
                self.user_pref, self.current_item_embeddding, rating, lr=0.1
            )

        if not self.candidates and self.selected:
            self.candidates, self.selected = self.selected, []

        if not self.candidates:
            return None

        reranked = self.mmr_rerank(lambd=0.99, top_k=1)[0]
        self.current_item_embeddding = reranked.final_vector
        self.current_recipe = {
            "title": reranked.title,
            "image": reranked.image,
            "ingredients": reranked.ingredients,
            "steps": reranked.steps,
        }
        return self.current_recipe

    # ----------------------------------------
    # Utility functions
    # ----------------------------------------
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Hitung cosine similarity antara dua vektor."""
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if denom == 0 else np.dot(a, b) / denom

    def update_user_pref(
        self, user_pref: np.ndarray, item_embedding: np.ndarray, rating: float, lr: float = 0.5
    ) -> np.ndarray:
        """Update preferensi user menggunakan feedback rating."""
        if rating == 0:
            return user_pref
        return user_pref + lr * rating * (item_embedding - user_pref)

    def mmr_rerank(self, lambd: float = 0.7, top_k: int = 1) -> List[MappingOutput]:
        """Lakukan Maximal Marginal Relevance (MMR) reranking."""
        bests = []
        while len(bests) < top_k and self.candidates:
            sims_user = np.array([
                self.cosine_similarity(self.user_pref, c.final_vector) for c in self.candidates
            ])
            sims_selected = np.array([
                max((self.cosine_similarity(c.final_vector, s.final_vector)
                    for s in self.selected), default=0)
                for c in self.candidates
            ])
            scores = lambd * sims_user - (1 - lambd) * sims_selected
            best_idx = np.argmax(scores)
            best = self.candidates.pop(best_idx)
            self.selected.append(best)
            bests.append(best)
        return bests

    @staticmethod
    def rerank_ingredients(embed_all, embed_ingredients, lambd: float = 0.7) -> np.ndarray:
        """Kombinasikan embedding ingredients dan embedding keseluruhan dengan bobot lambda."""
        if not (0.0 <= lambd <= 1.0):
            raise ValueError("λ (lambd) must be in [0, 1]")

        e1 = np.asarray(embed_ingredients, dtype=np.float32)
        e2 = np.asarray(embed_all, dtype=np.float32)

        if e1.shape != e2.shape:
            raise ValueError(f"Shape mismatch: {e1.shape} vs {e2.shape}")

        return lambd * e1 + (1 - lambd) * e2

    @staticmethod
    def matching_algorithm(list_rag: List[str]) -> List[str]:
        """Contoh dummy matching algorithm."""
        return [rag for rag in list_rag if "match" in rag]

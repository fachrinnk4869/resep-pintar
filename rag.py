import json
from AlgorithmClass import AlgorithmClass
from pipeline.elastic_generateVector import GenerateDenseVector, GenerateSparseVectors
from pipeline.elastic_pipeline import ElasticPipeline
from pipeline.rag_pipeline import RAG_pipeline
from settings import Settings as env
import asyncio


class Datahandle:
    def __init__(self):
        # search Elastic
        self.client = ElasticPipeline(
            index_name=env.INDEX_ELASTIC_NAME, mode="sparse")
        self.sparse_generator = GenerateSparseVectors(model_type="opensearch")
        self.dense_generator = GenerateDenseVector()

    async def get_recipes(self, query_input: str, possible_ingredient: str = None):
        """
        Input:
            query_input (str): teks utama (lebih penting)
            possible_ingredient (str): bahan tambahan (kurang penting)
        Output:
            list of recipes (dict)
        """
        loop = asyncio.get_event_loop()

        # Dapatkan sparse vector
        vec_query_input = self.sparse_generator.get_sparse_vector(query_input)[
            0]
        vec_possible_ingredient = (
            self.sparse_generator.get_sparse_vector(possible_ingredient)[0]
            if possible_ingredient
            else {"indices": [], "values": []}
        )

        # Terapkan bobot
        if possible_ingredient:
            alpha = 0.7
        else:
            alpha = 1.0

        beta = 0.3
        vec_query_input["values"] = [
            v * alpha for v in vec_query_input["values"]]
        vec_possible_ingredient["values"] = [
            v * beta for v in vec_possible_ingredient["values"]]

        # Gabungkan sparse vector berdasarkan index
        combined = {}
        for idx, val in zip(vec_query_input["indices"], vec_query_input["values"]):
            combined[idx] = combined.get(idx, 0) + val
        for idx, val in zip(vec_possible_ingredient["indices"], vec_possible_ingredient["values"]):
            combined[idx] = combined.get(idx, 0) + val

        # Ubah kembali ke format Elasticsearch
        vec_query_result = {
            "indices": list(combined.keys()),
            "values": list(combined.values()),
        }

        # Jalankan pencarian asynchronous
        result = await loop.run_in_executor(None, self.client.search_data_vec, vec_query_result, 10)
        return result[1]

    # ini ganti sama embedding sesuai maneh pake model apa untuk embeddingnya

    async def get_dense_input(self, text_input: str):
        '''
        Input: text input (str)
        Output: embedding_input '''
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self.dense_generator.get_dense_embedding, text_input)
        return embedding

    # ini ganti sama embedding recipe dari maneh
    def get_embeddings_recipe(self, recipes):
        '''
        Input: list of recipes (dict)
        Output: embeddings_all, embeding_ingredients
        '''
        return AlgorithmClass().generate_recipe_embeddings(
            recipes)

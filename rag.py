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

    async def get_recipes(self, query: str):
        # get top 50 recipe based on query from vector db
        ''' 
        Input: text input (str)
        Output: list of recipes (dict) '''
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.client.search_data, query, 10)
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

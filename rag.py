import json
from AlgorithmClass import AlgorithmClass
from pipeline.elastic_generateVector import GenerateDenseVector, GenerateSparseVectors
from pipeline.elastic_pipeline import ElasticPipeline
from pipeline.rag_pipeline import RAG_pipeline


class Datahandle:
    def __init__(self):
        # search Elastic
        self.client = ElasticPipeline(index_name="test-bang", mode="sparse")
        self.sparse_generator = GenerateSparseVectors(model_type="opensearch")
        self.dense_generator = GenerateDenseVector()

    def get_recipes(self, query: str):
        # get top 50 recipe based on query from vector db
        ''' 
        Input: text input (str)
        Output: list of recipes (dict) '''
        output_recipes = self.client.search_data(
            query, top_k=10)[1]  # get sparse embedding
        return output_recipes

    # ini ganti sama embedding sesuai maneh pake model apa untuk embeddingnya
    def get_dense_input(self, text_input: str):
        '''
        Input: text input (str)
        Output: embedding_input '''
        return self.dense_generator.get_dense_embedding(text_input)

    # ini ganti sama embedding recipe dari maneh
    def get_embeddings_recipe(self, recipes):
        '''
        Input: list of recipes (dict)
        Output: embeddings_all, embeding_ingredients
        '''
        return AlgorithmClass().generate_recipe_embeddings(
            recipes)

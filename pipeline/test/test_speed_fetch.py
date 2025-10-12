import chromadb
from pipeline.get_embedding import get_dense_embeddings
from pipeline.rag_pipeline import search_dense_index
from pinecone.grpc import PineconeGRPC as Pinecone
import time
import os
TOP_K = 50
EMBED_DIM = int(os.getenv('EMBED_DIM')) if os.getenv('EMBED_DIM') else 1024
NAMESPACE = os.getenv('NAMESPACE')


class pineconeTest:
    def __init__(self):
        PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
        NAME_PINECONE_DENSE = os.getenv('NAME_PINECONE_DENSE')
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_dense = self.pc.Index(name=NAME_PINECONE_DENSE)

    def speed_fetch(self, query: str, top_k: int = 50):
        vec = get_dense_embeddings(query, EMBED_DIM)
        dense_response = self.index_dense.query(
            namespace=NAMESPACE,
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=True
        )
        return dense_response

    def speed_fetch_vec(self, vec, top_k: int = 50):
        dense_response = self.index_dense.query(
            namespace=NAMESPACE,
            vector=vec,
            top_k=top_k,
            include_metadata=True,
            include_values=True
        )
        return dense_response


class chromaTest:
    def __init__(self):
        NAME_PINECONE_DENSE = os.getenv('NAME_PINECONE_DENSE')
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = chroma_client.get_or_create_collection(
            name=NAME_PINECONE_DENSE)

    def speed_fetch(self, query: str, top_k: int = 50):
        vec = get_dense_embeddings(query, EMBED_DIM)
        dense_response = self.collection.query(
            query_embeddings=vec,
            n_results=top_k,
        )
        return dense_response

    def speed_fetch_vec(self, vec, top_k: int = 50):
        dense_response = self.collection.query(
            query_embeddings=vec,
            n_results=top_k,
        )
        return dense_response


if __name__ == "__main__":
    query = "How to make fried rice? yang lembut dan menyenangkan"
    tok_ks = reversed([1, 5, 10, 50, 100])
    test_pinecone = pineconeTest()
    test_chroma = chromaTest()
    vec = get_dense_embeddings(query, EMBED_DIM)
    for top_k in tok_ks:
        print(f"Top K: {top_k}")
        start_time = time.time()
        test_pinecone.speed_fetch_vec(vec, top_k)
        end_time = time.time()
        print(f"Pinecone Dense search Time")
        print(
            f"Time taken for dense search: {end_time - start_time:.4f} seconds")

        start_time = time.time()
        test_chroma.speed_fetch_vec(vec, top_k)
        end_time = time.time()
        print(f"Chroma Dense search Time")
        print(
            f"Time taken for dense search: {end_time - start_time:.4f} seconds")
        print("-" * 50)

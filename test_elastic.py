from pipeline.elastic_pipeline import ElasticPipeline

"""
curl -fsSL https://elastic.co/start-local | sh

access dashboard : http://localhost:5601
COPY API key dari elastic search ke .env
"""

if __name__ == "__main__":
    # Inisialisasi ElasticSearch
    elastic = ElasticPipeline(index_name="test-bang", mode="hybrid")

    # Buat collection (dense + sparse)
    elastic.create_new_index()

    # Generate embeddings
    dense_vecs, sparse_vecs, dense_vecs_all = elastic.generate_embeddings_from_file(
        path_files="data/clean/new_cookpad.csv",
        column="text"
    )

    # Upsert ke Elastic
    elastic.upsert_data(dense_vecs, sparse_vecs, dense_vecs_all)

    # search Elastic
    elastic = ElasticPipeline(index_name="test-bang", mode="sparse")
    query = "ayam goreng"
    dense_result, sparse_result = elastic.search_data(query)
    print("Sparse Result", sparse_result)

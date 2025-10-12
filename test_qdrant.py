from pipeline.qdrant_pipeline import QdrantPipeline

"""
init qdrant:
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
"""

if __name__ == "__main__":
    # Inisialisasi Qdrant
    qdrant = QdrantPipeline(collection_name="test-bang", mode="hybrid")

    # Buat collection (dense + sparse)
    qdrant.create_new_collection()

    # Generate embeddings
    dense_vecs, sparse_vecs, dense_vecs_all = qdrant.generate_embeddings_from_file(
        path_files="data/clean/new_cookpad.csv", 
        column="text"
    )

    # Upsert ke Qdrant
    qdrant.upsert_data(dense_vecs, sparse_vecs, dense_vecs_all)

    # search Qdrant
    qdrant = QdrantPipeline(collection_name="test-bang", mode="sparse")
    query = "ayam goreng"
    dense_result, sparse_result = qdrant.search_data(query)
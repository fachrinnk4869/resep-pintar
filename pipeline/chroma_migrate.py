import os
import chromadb
import pinecone
from tqdm import tqdm
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
load_dotenv()
# --- 1. CONFIGURE YOUR CREDENTIALS ---

# Pinecone credentials
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# ChromaDB client (configure as needed)
# For a persistent local instance:
# chroma_client = chromadb.PersistentClient(path="/path/to/your/db")
# For an in-memory instance (for testing):
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# For a remote instance:
# chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# --- 2. DEFINE BATCH SIZES ---
# Adjust based on your system's memory and network capacity
FETCH_BATCH_SIZE = 1000  # How many vectors to fetch from Pinecone at a time
ADD_BATCH_SIZE = 2000    # How many items to add to ChromaDB at a time

# --- 3. INITIALIZE CLIENTS ---
print("Initializing clients...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Verify ChromaDB connection
    chroma_client.heartbeat()
    print("✅ Pinecone and ChromaDB clients initialized successfully.")
except Exception as e:
    print(f"❌ Error initializing clients: {e}")
    exit()

# --- 4. MIGRATION LOGIC ---
NAMESPACE2 = os.getenv('NAMESPACE2')
NAME_PINECONE_DENSE = os.getenv('NAME_PINECONE_DENSE')


def migrate():
    print("\nFetching list of Pinecone indexes...")
    pinecone_indexes = [NAME_PINECONE_DENSE]

    if not pinecone_indexes:
        print("No indexes found in your Pinecone project.")
        return

    print(f"Found {len(pinecone_indexes)} indexes: {pinecone_indexes}")

    for index_name in pinecone_indexes:
        print(f"\n{'='*50}")
        print(f"🚀 Starting migration for index: '{index_name}'")
        print(f"{'='*50}")

        # Connect to the Pinecone index
        pinecone_index = pc.Index(index_name)

        # Create a corresponding Chroma collection (or get it if it exists)
        print(f"Creating/getting Chroma collection: '{index_name}'")
        chroma_collection = chroma_client.get_or_create_collection(
            name=index_name)

        # Get all vector IDs from Pinecone.
        # NOTE: This assumes you are using the default "" namespace.
        # If you use namespaces, you will need to loop through them.
        print("Fetching all vector IDs from Pinecone index...")
        try:
            # This line is creating a list of lists. Let's call it nested_ids.
            nested_ids = [
                res_id for res_id in pinecone_index.list(namespace=NAMESPACE2)]

            # ✅ FIX: Flatten the list of lists into a single, flat list of IDs.
            all_ids = [item for sublist in nested_ids for item in sublist]

            # Optional: A check to confirm it worked
            if all_ids:
                print(
                    f"Successfully flattened list. First ID type is now: {type(all_ids[0])}")

        except Exception as e:
            print(
                f"⚠️ Could not fetch IDs for index '{index_name}'. It might be empty or have a different configuration. Skipping. Error: {e}")
            continue

        if not all_ids:
            print(f"Index '{index_name}' contains no vectors. Skipping.")
            continue

        print(f"Found {len(all_ids)} vectors to migrate.")

        # Fetch and add in batches
        with tqdm(total=len(all_ids), desc=f"Migrating '{index_name}'") as pbar:
            for i in range(0, len(all_ids), FETCH_BATCH_SIZE):
                batch_ids = all_ids[i:i + FETCH_BATCH_SIZE]

                # Fetch vector data from Pinecone
                fetch_response = pinecone_index.fetch(
                    ids=batch_ids, namespace=NAMESPACE2)
                print(
                    f"Fetched {type(fetch_response)} vectors from Pinecone.")
                vectors_data = fetch_response.vectors

                # Prepare data for ChromaDB
                chroma_ids = []
                chroma_embeddings = []
                chroma_metadatas = []
                chroma_documents = []

                for vec_id, vec_data in vectors_data.items():
                    metadata = vec_data.get('metadata', {})

                    # --- IMPORTANT: CUSTOMIZE THIS PART ---
                    # Extract the source text ('document') from your Pinecone metadata.
                    # Here, we assume it's stored under the key 'text'.
                    # Change 'text' to whatever key you use.
                    # Safely remove 'text'
                    document = vec_id
                    # print(f"Warning: No 'text' found in metadata for ID {vec_id}. Using a placeholder.")
                    # -----------------------------------------

                    chroma_ids.append(vec_id)
                    chroma_embeddings.append(vec_data['values'])
                    chroma_metadatas.append(metadata)
                    chroma_documents.append(document)

                # Add the batch to ChromaDB
                if chroma_ids:
                    # Using chroma's batching is more efficient than looping
                    chroma_collection.add(
                        ids=chroma_ids,
                        embeddings=chroma_embeddings,
                        metadatas=chroma_metadatas,
                        documents=chroma_documents
                    )

                pbar.update(len(batch_ids))

        # Final verification
        count = chroma_collection.count()
        print(
            f"✅ Migration for '{index_name}' complete. Total items in Chroma collection: {count}")


# --- 5. RUN THE MIGRATION ---
if __name__ == "__main__":
    migrate()
    print("\n🎉 All indexes have been processed.")

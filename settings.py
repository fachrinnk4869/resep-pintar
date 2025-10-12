import os
from dotenv import load_dotenv
load_dotenv()


class Settings:
    # .env import
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    NAME_PINECONE_DENSE: str = os.getenv("NAME_PINECONE_DENSE")
    NAME_PINECONE_SPARSE: str = os.getenv("NAME_PINECONE_SPARSE")
    SILICONFLOW_URL_RERANK: str = os.getenv("SILICONFLOW_URL_RERANK")
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY")
    NAMESPACE: str = os.getenv("NAMESPACE")
    NAMESPACE2: str = os.getenv("NAMESPACE2")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM") or 1024)
    ES_LOCAL_API_KEY: str = os.getenv("ES_LOCAL_API_KEY")

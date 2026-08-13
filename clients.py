import chromadb
import ollama

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

from config import (
    CHROMA_HOST,
    CHROMA_PORT,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_URL,
)


ollama_client = ollama.Client(
    host=OLLAMA_URL,
)


embedding_function = OllamaEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    url=OLLAMA_URL,
)


chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
)


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)
import os
from pathlib import Path


CHROMA_HOST = os.getenv(
    "CHROMA_HOST",
    "localhost",
)

CHROMA_PORT = int(
    os.getenv(
        "CHROMA_PORT",
        "8000",
    )
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:3b",
)

# Judge model used by Ragas validation. Defaults to the generation
# model so evaluation works out of the box; prefer a stronger local
# model such as mistral:7b or llama3.1:8b for more reliable scores.
JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    LLM_MODEL,
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "nexora_hr",
)

API_KEY = os.getenv(
    "API_KEY"
)

ADMIN_API_KEY = os.getenv(
    "ADMIN_API_KEY"
)

UPLOAD_DIR = Path(
    "./pdfs"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
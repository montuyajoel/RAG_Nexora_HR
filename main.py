from fastapi import FastAPI

from routers.ask import router as ask_router
from routers.documents import router as documents_router
from routers.healthcheck import router as health_router
from routers.validate import router as validate_router


app = FastAPI(
    title="Nexora HR RAG API",
    description=(
        "RAG API using FastAPI, "
        "ChromaDB and Ollama"
    ),
    version="1.2.0",
)


app.include_router(
    health_router
)

app.include_router(
    ask_router
)

app.include_router(
    documents_router
)

app.include_router(
    validate_router
)
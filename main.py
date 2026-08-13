import os
import shutil
import uuid
from pathlib import Path

import ollama
import chromadb

from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from knowledge_base import build_knowledge_base


# ============================================================
# CONFIGURATION
# ============================================================

# Local development defaults:
# ChromaDB -> localhost:8000
# Ollama   -> localhost:11434
#
# These can be overridden later using environment variables
# when the application is deployed with Docker / Kubernetes.

CHROMA_HOST = os.getenv(
    "CHROMA_HOST",
    "localhost"
)

CHROMA_PORT = int(
    os.getenv(
        "CHROMA_PORT",
        "8000"
    )
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:3b"
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "nexora_hr"
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_DIR = Path("./pdfs")

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Nexora HR RAG API",
    description="RAG API using FastAPI, ChromaDB and Ollama",
    version="1.0.0",
)


# ============================================================
# OLLAMA
# ============================================================

ollama_client = ollama.Client(
    host=OLLAMA_URL
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_function = OllamaEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    url=OLLAMA_URL,
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "nexora-rag-api"
    }


# ============================================================
# ASK
# ============================================================

@app.get("/ask")
def ask(question: str):

    # --------------------------------------------------------
    # STEP 1: RETRIEVE
    # --------------------------------------------------------

    try:

        results = collection.query(
            query_texts=[question],
            n_results=4,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail=f"ChromaDB retrieval failed: {str(error)}"
        )


    # --------------------------------------------------------
    # CHECK RETRIEVED DOCUMENTS
    # --------------------------------------------------------

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]


    if not documents:

        return {
            "question": question,
            "answer": "No relevant information was found in the knowledge base.",
            "context_used": [],
            "metadata": [],
            "distances": []
        }


    # --------------------------------------------------------
    # STEP 2: AUGMENT
    # --------------------------------------------------------

    context = "\n\n".join(
        documents
    )


    augmented_prompt = f"""
You are an HR policy assistant.

Answer the question using ONLY the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- Do not infer policies that are not explicitly stated.
- Distinguish carefully between annual leave, sick leave,
  parental leave, unpaid leave, and other leave types.
- Do not use information about one leave type to answer a
  question about another leave type.
- If the context does not explicitly contain enough information
  to answer the question, say:
  "The provided context does not contain enough information to
  answer this question."
- Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""


    # --------------------------------------------------------
    # STEP 3: GENERATE
    # --------------------------------------------------------

    try:

        response = ollama_client.chat(
            model=LLM_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": augmented_prompt
                }
            ],
        )

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail=f"Ollama generation failed: {str(error)}"
        )


    # --------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------

    return {
        "question": question,

        "answer":
            response["message"]["content"],

        "context_used":
            documents,

        "metadata":
            metadatas,

        "distances":
            distances
    }


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.post("/documents")
def add_document(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # STEP 1: VALIDATE FILE
    # --------------------------------------------------------

    if file.content_type != "application/pdf":

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )


    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file must have a filename."
        )


    # --------------------------------------------------------
    # STEP 2: CREATE SAFE FILENAME
    # --------------------------------------------------------

    original_filename = Path(
        file.filename
    ).name


    document_id = uuid.uuid4().hex


    stored_filename = (
        f"{document_id}_"
        f"{original_filename}"
    )


    file_path = (
        UPLOAD_DIR /
        stored_filename
    )


    # --------------------------------------------------------
    # STEP 3: SAVE PDF
    # --------------------------------------------------------

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save PDF: {str(error)}"
        )

    finally:

        file.file.close()


    # --------------------------------------------------------
    # STEP 4: BUILD KNOWLEDGE BASE
    # --------------------------------------------------------

    try:

        build_knowledge_base(
            pdf_paths=[
                str(file_path)
            ],

            collection=collection
        )

    except Exception as error:

        # Remove the PDF when ingestion fails
        # so failed uploads do not remain
        # in the upload directory.

        if file_path.exists():

            file_path.unlink()


        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(error)}"
        )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "message":
            "Document uploaded and added to the knowledge base.",

        "document_id":
            document_id,

        "filename":
            original_filename
    }


# ============================================================
# GET ALL CHUNKS
# ============================================================

@app.get("/documents")
def get_documents():

    try:

        results = collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail=f"Failed to retrieve documents: {str(error)}"
        )


    documents = []


    for i in range(
        len(results["ids"])
    ):

        documents.append({

            "id":
                results["ids"][i],

            "metadata":
                results["metadatas"][i],

            "content":
                results["documents"][i]

        })


    return {
        "total_chunks":
            len(documents),

        "documents":
            documents
    }


# ============================================================
# GET UNIQUE UPLOADED DOCUMENTS
# ============================================================

@app.get("/documents/list")
def get_uploaded_documents():

    try:

        results = collection.get(
            include=[
                "metadatas"
            ]
        )

    except Exception as error:

        raise HTTPException(
            status_code=503,
            detail=f"Failed to retrieve document list: {str(error)}"
        )


    unique_documents = {}


    for metadata in results["metadatas"]:

        source = metadata.get(
            "source",
            "unknown"
        )


        if source not in unique_documents:

            unique_documents[source] = {

                "source":
                    source,

                "filename":
                    Path(source).name,

                "chunks":
                    0
            }


        unique_documents[
            source
        ]["chunks"] += 1


    return {
        "total_documents":
            len(unique_documents),

        "documents":
            list(
                unique_documents.values()
            )
    }
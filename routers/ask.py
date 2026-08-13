from fastapi import (
    APIRouter,
    HTTPException,
    Security,
)

from clients import (
    collection,
    ollama_client,
)

from config import LLM_MODEL
from security import verify_api_key

from tools.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Endpoint to handle user questions and provide answers based on the knowledge base.
@router.get("/ask")
def ask(
    question: str,
    api_key: str = Security(
        verify_api_key
    ),
):
    try:
        results = collection.query(
            query_texts=[question],
            n_results=4,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    except Exception as error:
        logger.exception(
            "ChromaDB query failed: "
            f"{str(error)}"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "ChromaDB retrieval failed."
            ),
        )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    if not documents:
        return {
            "question": question,
            "answer": (
                "No relevant information "
                "was found in the knowledge base."
            ),
            "context_used": [],
            "metadata": [],
            "distances": [],
        }

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

    try:
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": augmented_prompt,
                }
            ],
        )

    except Exception as error:
        logger.exception(
            "Ollama generation failed: "
            f"{str(error)}"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Response generation is not available at the moment."
            ),
        )

    return {
        "question": question,
        "answer": response[
            "message"
        ]["content"],
        "context_used": documents,
        "metadata": metadatas,
        "distances": distances,
    }

# Endpoint to retrieve relevant documents from the knowledge base based on a user question.
@router.get("/retrieve")
def retrieve(
    question: str,
    api_key: str = Security(
        verify_api_key
    ),
):
    try:
        results = collection.query(
            query_texts=[question],
            n_results=3,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    except Exception as error:
        logger.exception(
            "ChromaDB retrieval failed: "
            f"{str(error)}"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Database retrieval is not available at the moment."
            ),
        )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    return {
        "question": question,
        "context_used": documents,
        "metadata": metadatas,
        "distances": distances,
    }
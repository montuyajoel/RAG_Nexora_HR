import shutil
import uuid

from pathlib import Path

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Security,
    UploadFile,
)

from clients import collection
from config import UPLOAD_DIR
from tools.file_processor import build_knowledge_base
from security import verify_api_key

# Set up logging
from tools.logger import get_logger
logger = get_logger(__name__)

# Create an APIRouter instance for document-related endpoints.
router = APIRouter()

# Endpoint to upload a PDF document, process it, and add its content to the knowledge base.
@router.post("/documents")
def add_document(
    file: UploadFile = File(...),
    api_key: str = Security(
        verify_api_key
    ),
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file must "
                "have a filename."
            ),
        )

    original_filename = Path(
        file.filename
    ).name

    document_id = uuid.uuid4().hex

    stored_filename = (
        f"{document_id}_"
        f"{original_filename}"
    )

    file_path = (
        UPLOAD_DIR
        / stored_filename
    )

    try:
        with open(
            file_path,
            "wb",
        ) as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

    except Exception as error:
        logger.error(
            "Failed to save PDF: "
            f"{str(error)}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save PDF."
            ),
        )

    finally:
        file.file.close()

    try:
        build_knowledge_base(
            pdf_paths=[
                str(file_path)
            ],
            collection=collection,
        )

    except Exception as error:
        if file_path.exists():
            file_path.unlink()

        logger.error(
            "Failed to process document: "
            f"{str(error)}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process document."
            ),
        )

    return {
        "message": (
            "Document uploaded and added "
            "to the knowledge base."
        ),
        "document_id": document_id,
        "filename": original_filename,
    }


@router.get("/documents")
def get_documents(
    api_key: str = Security(
        verify_api_key
    ),
):
    try:
        results = collection.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

    except Exception as error:
        logger.error(
            "Failed to retrieve documents: "
            f"{str(error)}"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Document service is currently unavailable."
            ),
        )

    documents = []

    for i in range(
        len(results["ids"])
    ):
        documents.append(
            {
                "id":
                    results["ids"][i],

                "metadata":
                    results["metadatas"][i],

                "content":
                    results["documents"][i],
            }
        )

    return {
        "total_chunks": len(
            documents
        ),
        "documents": documents,
    }


@router.get("/documents/list")
def get_uploaded_documents(
    api_key: str = Security(
        verify_api_key
    ),
):
    try:
        results = collection.get(
            include=[
                "metadatas",
            ]
        )

    except Exception as error:
        logger.error(
            "Failed to retrieve document list: "
            f"{str(error)}"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Document service is currently unavailable."
            ),
        )

    unique_documents = {}

    for metadata in results["metadatas"]:
        source = metadata.get(
            "source",
            "unknown",
        )

        if source not in unique_documents:
            unique_documents[source] = {
                "source": source,
                "filename":
                    Path(source).name,
                "chunks": 0,
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
            ),
    }

# Delete all documents and their chunks from the ChromaDB collection.
@router.delete("/documents")
def delete_documents(
    api_key: str = Security(
        verify_api_key
    ),
):
    try:
        collection.delete(
            where={}
        )

    except Exception as error:
        logger.error(
            "Failed to delete documents: "
            f"{str(error)}"
        )
        
        raise HTTPException(
            status_code=503,
            detail=(
                "Document service is currently unavailable."
            ),
        )

    return {
        "message": (
            "All documents and their "
            "chunks have been deleted."
        )
    }

# Delete a specific document and its chunks from the ChromaDB collection based on the document ID.
@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    api_key: str = Security(
        verify_api_key
    ),
):
    try:
        results = collection.get(
            include=[
                "ids",
                "metadatas",
            ]
        )

    except Exception as error:
        logger.error(
            "Failed to retrieve documents: "
            f"{str(error)}"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Document service is currently unavailable."
            ),
        )

    document_ids_to_delete = []

    for i in range(
        len(results["ids"])
    ):
        metadata = results["metadatas"][i]

        source = metadata.get(
            "source",
            "unknown",
        )

        if source.startswith(
            f"{document_id}_"
        ):
            document_ids_to_delete.append(
                results["ids"][i]
            )

    if not document_ids_to_delete:
        raise HTTPException(
            status_code=404,
            detail=(
                "Document not found or "
                "already deleted."
            ),
        )

    try:
        collection.delete(
            where={
                "id": {
                    "$in": document_ids_to_delete
                }
            }
        )

    except Exception as error:
        logger.error(
            "Failed to delete document: "
            f"{str(error)}"
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Document service is currently unavailable."
            ),
        )

    return {
        "message": (
            f"Document {document_id} and its "
            f"chunks have been deleted."
        )
    }      
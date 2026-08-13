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
from security import verify_api_key, verify_admin_api_key

MAX_FILE_SIZE = 10 * 1024 * 1024

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
        verify_admin_api_key
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

    # Read only up to the maximum size + 1 byte.
    content = file.file.read(
        MAX_FILE_SIZE + 1
    )

    if len(content) > MAX_FILE_SIZE:
        file.file.close()

        raise HTTPException(
            status_code=413,
            detail=(
                "PDF exceeds the "
                "10 MB upload limit."
            ),
        )

    # Verify actual PDF signature.
    if not content.startswith(b"%PDF-"):
        file.file.close()

        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file.",
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
            buffer.write(
                content
            )

    except Exception:
        logger.exception(
            "Failed to save PDF"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to save PDF.",
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

    except Exception:
        if file_path.exists():
            file_path.unlink()

        logger.exception(
            "Failed to process document"
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
        verify_admin_api_key
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
        logger.exception(
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
        verify_admin_api_key
    ),
):
    try:
        results = collection.get(
            include=[
                "metadatas",
            ]
        )

    except Exception as error:
        logger.exception(
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
    confirm: str,
    api_key: str = Security(
        verify_admin_api_key
    ),
):
    if confirm != "DELETE_ALL_DOCUMENTS":
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid deletion confirmation."
            ),
        )

    try:
        results = collection.get()

        ids_to_delete = results.get(
            "ids",
            []
        )

        if not ids_to_delete:
            return {
                "message": (
                    "No documents found to delete."
                ),
                "deleted_chunks": 0,
            }

        collection.delete(
            ids=ids_to_delete
        )

    except Exception:
        logger.exception(
            "Failed to delete documents"
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Document service is "
                "currently unavailable."
            ),
        )

    logger.warning(
        "All document chunks deleted: %s",
        len(ids_to_delete),
    )

    return {
        "message": (
            "All documents and their "
            "chunks have been deleted."
        ),
        "deleted_chunks":
            len(ids_to_delete),
    }

# Delete a specific document and its chunks from the ChromaDB collection based on the document ID.
@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    api_key: str = Security(
        verify_admin_api_key
    ),
):
    try:
        results = collection.get(
            include=[
                "metadatas",
            ]
        )

    except Exception as error:
        logger.exception(
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
            ids=document_ids_to_delete
        )

    except Exception as error:
        logger.exception   (
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
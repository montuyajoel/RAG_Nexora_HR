from pathlib import Path
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Builds a knowledge base from a list of PDF file paths and stores the chunks in a ChromaDB collection.
def build_knowledge_base(
    pdf_paths: list[str],
    collection
):
    # 1. Load PDFs
    documents = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(documents)

    # 3. Generate unique IDs
    ids = []

    for chunk in chunks:

        source = Path(
            chunk.metadata.get(
                "source",
                "unknown"
            )
        ).stem

        page = chunk.metadata.get(
            "page",
            -1
        )

        content_hash = hashlib.sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:12]

        chunk_id = (
            f"{source}_"
            f"page_{page}_"
            f"{content_hash}"
        )

        ids.append(chunk_id)


    # 4. Store chunks in ChromaDB
    collection.upsert(
        ids=ids,

        documents=[
            chunk.page_content
            for chunk in chunks
        ],

        metadatas=[
            {
                "source":
                    chunk.metadata.get(
                        "source",
                        "unknown"
                    ),

                "page":
                    chunk.metadata.get(
                        "page",
                        -1
                    ),

                "chunk_index":
                    i,
            }

            for i, chunk in enumerate(chunks)
        ],
    )
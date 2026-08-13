from pathlib import Path
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_knowledge_base(
    pdf_paths: list[str],
    collection
):
    # 1. Load PDFs
    documents = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    print(f"Loaded {len(documents)} pages.")


    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(documents)

    print(f"Split into {len(chunks)} chunks.")


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


    print(
        f"Added {len(chunks)} chunks "
        "to the 'nexora_hr' collection."
    )
# Nexora HR RAG API

A local-first **Retrieval-Augmented Generation (RAG)** application for querying HR policy documents using **FastAPI, ChromaDB, Ollama, LangChain, and local LLMs**.

The application allows PDF documents to be uploaded through an API, converts their contents into vector embeddings, stores them in ChromaDB, retrieves relevant document chunks based on a user's question, and uses an Ollama-hosted language model to generate a grounded response.

The project is deployed on **Google Cloud Platform (GCP)** using **Docker, Google Kubernetes Engine (GKE), Artifact Registry, Kubernetes persistent storage, and GitHub Actions**. The current GKE runtime uses separate deployments for FastAPI, ChromaDB, and Ollama in the `rag` namespace.

---

## Architecture

![Nexora HR RAG - GKE Architecture](Nexora_RAG_ARCH.png)

The architecture image summarizes the RAG data path and GKE deployment. The public LoadBalancer routes requests to replicated FastAPI pods, which use Kubernetes services to reach ChromaDB and Ollama. Persistent volumes retain Chroma vector data and Ollama model files across pod replacement.

```text
                         User
                          |
                          v
                     FastAPI API
                    /     |      \\
                   /      |       \\
            /documents   /ask    /health
                |         |
                v         v
          PDF Ingestion   Query
                |         |
                v         v
          PyPDFLoader   nomic-embed-text
                |         |
                v         v
          Text Splitter  ChromaDB
                |         |
                v         |
          nomic-embed-text|
                |         |
                v         v
             ChromaDB  Relevant Chunks
                          |
                          v
                    Augmented Prompt
                          |
                          v
                       Ollama
                          |
                          v
                    llama3.2:3b
                          |
                          v
                     RAG Answer
```

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application language |
| FastAPI | REST API |
| Uvicorn | ASGI application server |
| LangChain | PDF loading and text processing |
| PyPDFLoader | Extracts text from PDFs |
| RecursiveCharacterTextSplitter | Splits documents into chunks |
| ChromaDB | Vector database |
| `nomic-embed-text` | Embedding model |
| Ollama | Local model runtime |
| `llama3.2:3b` | Response generation |
| Docker | Application containerization |
| GKE | Kubernetes runtime for the deployed application |
| Artifact Registry | Docker image registry |
| GitHub Actions | CI/CD pipeline for build and GKE deployment |

---

# RAG Workflow

The application follows the standard **Retrieve → Augment → Generate** architecture.

## 1. Retrieve

When a question reaches `/ask`, the question is converted into an embedding using:

```text
nomic-embed-text
```

ChromaDB compares the question embedding against the stored document embeddings and retrieves the most semantically relevant chunks.

```text
Question
   |
   v
nomic-embed-text
   |
   v
Query Vector
   |
   v
ChromaDB
   |
   v
Top Relevant Chunks
```

The current configuration retrieves:

```python
n_results=4
```

---

## 2. Augment

The retrieved document chunks are combined with the original question.

The resulting prompt instructs the LLM to answer using only the retrieved HR policy context.

```text
Retrieved Context
       +
User Question
       |
       v
Augmented Prompt
```

The prompt includes grounding rules intended to reduce hallucinations and prevent information from unrelated leave or policy categories from being mixed together.

---

## 3. Generate

The augmented prompt is sent to the language model through Ollama.

Current generation model:

```text
llama3.2:3b
```

The generated response is returned together with:

- retrieved context;
- metadata;
- vector distances.

This makes it possible to inspect which document chunks were used to generate the answer.

---

# Document Ingestion

PDF documents can be uploaded through:

```http
POST /documents
```

The ingestion pipeline is:

```text
PDF Upload
    |
    v
Save PDF
    |
    v
PyPDFLoader
    |
    v
Extract Pages
    |
    v
RecursiveCharacterTextSplitter
    |
    v
Text Chunks
    |
    v
nomic-embed-text
    |
    v
Vector Embeddings
    |
    v
ChromaDB
```

Current chunking configuration:

```python
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)
```

Document chunks contain metadata such as:

```json
{
    "source": "./pdfs/example.pdf",
    "page": 3,
    "chunk_index": 12
}
```

---

# API Endpoints

## Health Check

```http
GET /health
```

Example response:

```json
{
    "status": "healthy",
    "service": "nexora-rag-api"
}
```

The endpoint is designed for Docker/Kubernetes health checks.

---

## Ask a Question

```http
GET /ask
```

Query parameter:

```text
question
```

Example:

```text
/ask?question=Does annual leave carry over to next year?
```

Example response:

```json
{
    "question": "Does annual leave carry over to next year?",
    "answer": "Up to 5 unused days may be carried over...",
    "context_used": [
        "..."
    ],
    "metadata": [
        {
            "source": "./pdfs/leave_policy.pdf",
            "page": 4,
            "chunk_index": 7
        }
    ],
    "distances": [
        0.32
    ]
}
```

---

## Upload a PDF

```http
POST /documents
```

The endpoint accepts:

```text
multipart/form-data
```

Only PDF files are accepted.

Example response:

```json
{
    "message": "Document uploaded and added to the knowledge base.",
    "document_id": "0db30c7d...",
    "filename": "leave_policy.pdf"
}
```

---

## Inspect Stored Chunks

```http
GET /documents
```

Returns all chunks currently stored in the ChromaDB collection.

This endpoint is primarily intended for development and debugging.

Example:

```json
{
    "total_chunks": 120,
    "documents": [
        {
            "id": "leave_policy_page_1_...",
            "metadata": {
                "source": "./pdfs/leave_policy.pdf",
                "page": 1,
                "chunk_index": 2
            },
            "content": "..."
        }
    ]
}
```

---

## List Uploaded Documents

```http
GET /documents/list
```

Returns unique source documents instead of individual chunks.

Example:

```json
{
    "total_documents": 2,
    "documents": [
        {
            "source": "./pdfs/leave_policy.pdf",
            "filename": "leave_policy.pdf",
            "chunks": 38
        }
    ]
}
```

---

# Project Structure

```text
RAG/
├── main.py
├── knowledge_base.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
│
├── pdfs/
│
├── kubernetes/
│   ├── api.yaml
│   ├── chroma.yaml
│   ├── chroma-pvc.yaml
│   ├── ollama.yaml
│   └── ollama-pvc.yaml
│
└── .github/
    └── workflows/
        └── deploy.yml
```

---

# Local Installation

## 1. Clone the repository

```bash
git clone \<YOUR_REPOSITORY_URL>

cd \<YOUR_REPOSITORY>
```

---

## 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include:

```text
fastapi
uvicorn[standard]
chromadb
ollama
langchain-community
langchain-text-splitters
pypdf
python-multipart
```

---

# Ollama Setup

Install and start Ollama.

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Pull the generation model:

```bash
ollama pull llama3.2:3b
```

Verify:

```bash
ollama list
```

Expected models:

```text
nomic-embed-text
llama3.2:3b
```

Ollama normally runs locally on:

```text
http\://localhost:11434
```

---

# ChromaDB

The current architecture uses ChromaDB in **server mode** rather than `PersistentClient`.

This allows multiple API instances to communicate with the same vector database.

Create a Docker network:

```bash
docker network create rag-network
```

Start Chroma:

```bash
docker run -d \\
  --name chroma \\
  --network rag-network \\
  -p 8001:8000 \\
  -v chroma-data:/data \\
  chromadb/chroma\:latest
```

Check:

```bash
docker ps
```

Test the Chroma server:

```bash
curl http\://localhost:8001/api/v2/heartbeat
```

---

# Run FastAPI Locally

When FastAPI runs directly on the host:

```bash
export CHROMA_HOST=localhost
export CHROMA_PORT=8001
export OLLAMA_URL=http\://localhost:11434

uvicorn main\:app --reload
```

Swagger UI:

```text
http\://localhost:8000/docs
```

Health check:

```text
http\://localhost:8000/health
```

---

# Docker

## Build

```bash
docker build -t nexora-rag-api .
```

Verify the image:

```bash
docker images
```

---

## Run

When Chroma runs in Docker and Ollama runs on the host machine:

```bash
docker run \\
  --name nexora-rag-api \\
  --network rag-network \\
  -p 8000:8000 \\
  -e CHROMA_HOST=chroma \\
  -e CHROMA_PORT=8000 \\
  -e OLLAMA_URL=http\://host.docker.internal:11434 \\
  nexora-rag-api
```

The resulting local architecture is:

```text
Host Machine
│
├── Ollama
│   :11434
│
└── Docker Network
    │
    ├── FastAPI
    │   :8000
    │
    └── ChromaDB
        :8000
```

The host exposes:

```text
FastAPI   localhost:8000
ChromaDB  localhost:8001
Ollama    localhost:11434
```

---

# Environment Variables

The application supports configuration through environment variables.

| Variable | Local Default | Kubernetes |
|---|---|---|
| `CHROMA_HOST` | `localhost` | `chroma` |
| `CHROMA_PORT` | `8000` / configured host port | `8000` |
| `OLLAMA_URL` | `http\://localhost:11434` | `http\://ollama:11434` |
| `EMBEDDING_MODEL` | `nomic-embed-text` | `nomic-embed-text` |
| `LLM_MODEL` | `llama3.2:3b` | `llama3.2:3b` |
| `CHROMA_COLLECTION` | `nexora_hr` | `nexora_hr` |

This allows the same Python application to run locally, in Docker, and in Kubernetes without changing the source code.

---

# Google Kubernetes Engine (GKE) Deployment

The deployed cloud architecture is:

```text
GitHub
   |
   | push
   v
GitHub Actions
   |
   +--> Test
   |
   +--> Docker Build
   |
   +--> Artifact Registry
   |
   v
Google Kubernetes Engine
   |
   +-----------------------------+
   |                             |
   v                             v
FastAPI Pod 1              FastAPI Pod 2
   |                             |
   +-------------+---------------+
                 |
        +--------+--------+
        |                 |
        v                 v
     ChromaDB           Ollama
                           |
                   +-------+-------+
                   |               |
                   v               v
          nomic-embed-text   llama3.2:3b
```

---

# GKE Deployment Details

The cloud deployment runs in Google Cloud project `nexora-ai-agent-505409` with the following configuration:

| Setting | Value |
|---|---|
| Platform | Google Kubernetes Engine (GKE) |
| Region | `europe-west1` |
| Zone | `europe-west1-b` |
| Cluster | `rag-cluster` |
| Namespace | `rag` |
| Artifact Registry repository | `rag-containers` |
| API image | `nexora-rag-api` |
| FastAPI replicas | 2 |
| ChromaDB replicas | 1 |
| Ollama replicas | 1 |
| API service | `LoadBalancer` |
| ChromaDB storage | PersistentVolumeClaim (`chroma-data`) |
| Ollama storage | PersistentVolumeClaim (`ollama-data`) |

### Runtime request path

```text
Client
  |
  v
GCP Load Balancer :80
  |
  v
rag-api-service
  |
  +--> rag-api pod 1 :8000
  |
  +--> rag-api pod 2 :8000
          |
          +--> chroma:8000 ------> ChromaDB + persistent vector data
          |
          +--> ollama:11434 -----> nomic-embed-text / llama3.2:3b
```

### Kubernetes resources

```text
namespace/rag
├── deployment/rag-api       (2 replicas)
├── service/rag-api-service  (LoadBalancer)
├── deployment/chroma        (1 replica)
├── service/chroma           (ClusterIP)
├── pvc/chroma-data
├── deployment/ollama        (1 replica)
├── service/ollama           (ClusterIP)
└── pvc/ollama-data
```

### Verify the deployment

```bash
kubectl get pods -n rag
kubectl get deployments -n rag
kubectl get svc -n rag
kubectl get pvc -n rag
```

Expected deployment readiness:

```text
chroma    1/1
ollama    1/1
rag-api   2/2
```

---

# Kubernetes Configuration

Inside Kubernetes, services communicate through Kubernetes DNS.

FastAPI configuration:

```yaml
env:
  - name: CHROMA_HOST
    value: "chroma"

  - name: CHROMA_PORT
    value: "8000"

  - name: OLLAMA_URL
    value: "http\://ollama:11434"

  - name: EMBEDDING_MODEL
    value: "nomic-embed-text"

  - name: LLM_MODEL
    value: "llama3.2:3b"
```

The API can therefore reach:

```text
ChromaDB
http\://chroma:8000

Ollama
http\://ollama:11434
```

---

# GitHub Actions CI/CD

The GitHub Actions deployment pipeline is:

```text
Developer
   |
git push
   |
   v
GitHub
   |
   v
GitHub Actions
   |
   +--> Run tests
   |
   +--> Authenticate to GCP
   |
   +--> Build Docker image
   |
   +--> Push image
   |
   v
Artifact Registry
   |
   v
GKE Deployment
   |
   v
Rolling Update
```

Each commit should produce a versioned Docker image using the Git commit SHA.

Example:

```text
europe-west1-docker.pkg.dev/
PROJECT_ID/
rag-containers/
nexora-rag-api:
COMMIT_SHA
```

---

# Current Development Status

### Implemented

- FastAPI REST API
- PDF upload endpoint
- PDF validation
- LangChain document loading
- Recursive text chunking
- `nomic-embed-text` embeddings
- ChromaDB semantic retrieval
- Ollama LLM generation
- Grounded RAG prompting
- Retrieval metadata inspection
- Similarity-distance inspection
- Document listing
- ChromaDB server architecture
- Dockerized FastAPI application
- Health endpoint
- Environment-based service configuration

### Infrastructure Extension

- Docker networking
- ChromaDB persistent Docker volume
- Google Artifact Registry
- Google Kubernetes Engine
- Multiple FastAPI replicas
- Kubernetes health probes
- GitHub Actions CI/CD
- Workload Identity Federation
- Persistent Kubernetes storage

### Planned Production Improvements

- Google Cloud Storage for uploaded PDFs
- asynchronous document ingestion
- Redis or Pub/Sub job queue
- worker services
- Cloud SQL
- tenant/user isolation
- authentication and authorization
- Secret Manager
- HTTPS
- rate limiting
- structured logging
- Cloud Monitoring
- horizontal pod autoscaling
- load testing
- failure testing
- Terraform

---

# Key Concepts Demonstrated

This project provides practical exposure to:

- Retrieval-Augmented Generation
- semantic search
- vector embeddings
- vector databases
- prompt grounding
- hallucination reduction
- document ingestion pipelines
- REST API development
- containerization
- service networking
- persistent storage
- stateless API design
- distributed application architecture
- Kubernetes
- horizontal scaling
- health and readiness checks
- CI/CD
- cloud deployment
- infrastructure automation

---

# Important Production Considerations

The current implementation is primarily a learning and engineering project.

Before production use, several areas require additional hardening.

Uploaded PDFs should not permanently reside on the FastAPI pod filesystem because Kubernetes pods are ephemeral. Production document storage should use an object store such as Google Cloud Storage.

Document ingestion should also be moved out of the synchronous `/documents` request path.

The target architecture is:

```text
POST /documents
       |
       v
Cloud Storage
       |
       v
Job Queue
       |
       v
Worker
       |
       +--> Parse
       +--> Chunk
       +--> Embed
       |
       v
ChromaDB
```

Authentication and tenant-level authorization are also required before storing documents belonging to multiple users or organizations.

---

# Testing Order

When starting the application locally, test the infrastructure in this order:

```text
1\. Ollama
      |
      v
2\. ChromaDB
      |
      v
3\. FastAPI /health
      |
      v
4\. /documents/list
      |
      v
5\. POST /documents
      |
      v
6\. /documents/list
      |
      v
7\. /ask
```

This isolates infrastructure failures before testing the complete RAG pipeline.

---

# Example RAG Test

Question:

```text
Does the annual leave balance carry over to next year?
```

Expected retrieval behavior:

```text
Question
   |
   v
Retrieve annual-leave policy
   |
   v
Find carryover rule
   |
   v
Provide context to LLM
   |
   v
Generate grounded answer
```

The response should be based only on the retrieved policy context and should not incorrectly substitute sick-leave rules for annual-leave rules.

---

# License

This project is currently intended for educational, portfolio, and development purposes.

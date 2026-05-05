from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import logging
from backend.config import UPLOAD_DIR, LLM_MODEL, LLM_BASE_URL
from backend.models.schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ConversationRequest,
    Source
)

from backend.ingestion.loader import load_document, load_all_documents
from backend.ingestion.chunker import chunk_documents
from backend.retrieval.embedder import get_embedding_model
from backend.retrieval.vectorstore import get_vectorstore
from backend.retrieval.retriever import retrieve_documents, compute_confidence
from backend.generation.chain import generate_answer
from backend.generation.memory import (
    get_conversation_history,
    add_to_conversation,
    clear_conversation
)

# ── App Setup ─────────────────────────────────────────
app = FastAPI(
    title="Universal Knowledge Assistant",
    description="Query any document with AI-powered answers and citations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Startup: Load Models ───────────────────────────────
embedding_model = None
vectorstore = None

@app.on_event("startup")
async def startup_event():
    global embedding_model, vectorstore
    logger.info("Loading embedding model on startup...")
    embedding_model = get_embedding_model()
    vectorstore = get_vectorstore(
        embedding_model=embedding_model
    )
    logger.info("App ready!")

# ── Endpoint 1: Ingest Documents ──────────────────────
@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(file: UploadFile = File(...)):
    global vectorstore
    
    # Save uploaded file to uploads directory
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Saved file: {file.filename}")

    # Load, chunk and store
    documents = load_document(str(file_path))
    chunks = chunk_documents(documents)
    vectorstore = get_vectorstore(
        chunks=chunks,
        embedding_model=embedding_model
    )

    return IngestResponse(
        message=f"Successfully ingested {file.filename}",
        documents_loaded=len(documents),
        chunks_created=len(chunks)
    )

# ── Endpoint 2: Query Documents ───────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Please upload documents first."
        )

    # Retrieve relevant chunks
    retrieved_docs = retrieve_documents(
        query=request.question,
        vectorstore=vectorstore
    )

    # Compute confidence
    confidence = compute_confidence(
        query=request.question,
        retrieved_docs=retrieved_docs,
        vectorstore=vectorstore
    )

    # Generate answer
    from langchain_community.llms import Ollama
    llm = Ollama(model=LLM_MODEL, base_url=LLM_BASE_URL)
    answer = generate_answer(request.question, retrieved_docs, llm)

    # Build sources
    sources = [
        Source(
            document_name=doc.metadata.get("source", "Unknown"),
            page=doc.metadata.get("page"),
            chunk_preview=doc.page_content[:150],
            relevance_score=confidence
        )
        for doc in retrieved_docs
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence
    )

# ── Endpoint 3: Conversation ───────────────────────────
@app.post("/conversation", response_model=QueryResponse)
async def conversation(request: ConversationRequest):
    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Please upload documents first."
        )

    # Get conversation history
    history = get_conversation_history(request.conversation_id)

    # Build question with context from history
    if history:
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history[-4:]
        ])
        enriched_question = f"{history_text}\nUSER: {request.question}"
    else:
        enriched_question = request.question

    # Retrieve, generate and store
    retrieved_docs = retrieve_documents(
        query=enriched_question,
        vectorstore=vectorstore
    )
    confidence = compute_confidence(
        query=enriched_question,
        retrieved_docs=retrieved_docs,
        vectorstore=vectorstore
    )

    from langchain_community.llms import Ollama
    llm = Ollama(model=LLM_MODEL, base_url=LLM_BASE_URL)
    answer = generate_answer(enriched_question, retrieved_docs, llm)

    # Save to memory
    add_to_conversation(request.conversation_id, request.question, answer)

    sources = [
        Source(
            document_name=doc.metadata.get("source", "Unknown"),
            page=doc.metadata.get("page"),
            chunk_preview=doc.page_content[:150],
            relevance_score=confidence
        )
        for doc in retrieved_docs
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence
    )

# ── Health Check ──────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": embedding_model is not None}
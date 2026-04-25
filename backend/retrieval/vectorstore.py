from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
import logging

from backend.config import VECTORSTORE_DIR, CHROMA_COLLECTION

logger = logging.getLogger(__name__)

def create_vectorstore(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings
) -> Chroma:
    """
    Create a ChromaDB vectorstore from document chunks.
    """
    logger.info(f"Creating vectorstore with {len(chunks)} chunks")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(VECTORSTORE_DIR)
    )
    logger.info("Vectorstore created and persisted to disk")
    return vectorstore

def load_vectorstore(
    embedding_model: HuggingFaceEmbeddings
) -> Chroma:
    """
    Load an existing ChromaDB vectorstore from disk.
    """
    logger.info("Loading existing vectorstore from disk")
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embedding_model,
        persist_directory=str(VECTORSTORE_DIR)
    )
    logger.info("Vectorstore loaded successfully")
    return vectorstore

def get_vectorstore(
    chunks: List[Document] = None,
    embedding_model: HuggingFaceEmbeddings = None
) -> Chroma:
    """
    Get vectorstore — load if exists, create if not.
    """
    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()):
        return load_vectorstore(embedding_model)
    else:
        return create_vectorstore(chunks, embedding_model)
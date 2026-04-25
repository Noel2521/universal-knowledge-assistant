from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import logging
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)



def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of documents into smaller chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
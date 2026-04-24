from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader
)
from pathlib import Path
from typing import List
from langchain.schema import Document

import logging

logger = logging.getLogger(__name__)

def load_document(file_path: str) -> List[Document]:
    """
    Load a single document and return a list of LangChain Document objects.
    Supports: PDF, DOCX, CSV
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == '.pdf':
        loader= PyPDFLoader(str(path))
    elif extension == '.docx':
        loader = Docx2txtLoader(str(path))
    elif extension == '.csv':
        loader = CSVLoader(str(path))
    else:
        raise ValueError(f"Unsupported file type:{extension}")
    
    try:
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {path.name}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load {path.name}: {e}")
        raise

def load_all_documents(upload_dir: str) -> List[Document]:
    """
    Load all supported documents from the uploads directory.
    """
    all_documents = []
    supported_extensions = {".pdf", ".docx", ".csv"}
    upload_path = Path(upload_dir)
    for file_path in upload_path.iterdir():
        if file_path.suffix.lower() in supported_extensions:
            try:
                docs = load_document(str(file_path))
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Skipping {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(all_documents)}") 
    return all_documents


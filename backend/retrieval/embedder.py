from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import EMBEDDING_MODEL
import logging

logger = logging.getLogger(__name__)

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace embedding Model.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}
    )
    logger.info("Embedding model loaded successfully")
    return embeddings
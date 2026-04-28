from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Tuple
import logging

from backend.config import RETRIEVAL_K, MMR_DIVERSITY

logger = logging.getLogger(__name__)


def retrieve_documents(
    query: str,
    vectorstore: Chroma,
) -> List[Document]:
    """
    Retrieve most relevant documents using MMR search.
    """
    logger.info(f"Retrieving documents for query: {query[:50]}...")
    results = vectorstore.max_marginal_relevance_search(
        query=query,
        k=RETRIEVAL_K,
        fetch_k=RETRIEVAL_K * 3,
        lambda_mult=MMR_DIVERSITY
    )
    logger.info(f"Retrieved {len(results)} chunks")
    return results

def compute_confidence(
    query: str,
    retrieved_docs: List[Document],
    vectorstore: Chroma
) -> float:
    """
    Compute confidence score based on retrieval quality.
    """
    if not retrieved_docs:
        return 0.0

    results_with_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=RETRIEVAL_K
    )
    if not results_with_scores:
        return 0.0

    # Convert distances to similarity scores (1 - distance)
    similarity_scores = [
        1 - score for _, score in results_with_scores
    ]

    # Average similarity across retrieved chunks
    avg_similarity = sum(similarity_scores) / len(similarity_scores)

    # Count unique source documents
    unique_sources = len(set(
        doc.metadata.get("source", "") for doc in retrieved_docs
    ))

    # Boost confidence if answer spans multiple documents
    source_boost = min(unique_sources * 0.05, 0.15)

    confidence = min(avg_similarity + source_boost, 1.0)
    logger.info(f"Confidence score: {confidence:.2f}")
    return round(confidence, 2)


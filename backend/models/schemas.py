from pydantic import BaseModel
from typing import List, Optional

# ── Ingestion Schemas
class IngestResponse(BaseModel):
    message: str
    documents_loaded: int
    chunks_created:int

# ── Query Schemas
class QueryRequest(BaseModel):
    question:str
    top_k:Optional[int] = 5

# ── Source
class Source(BaseModel):
    document_name:str
    page: Optional[int] = None
    chunk_preview: str
    relevance_score: float

# ── Query Response
class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float

# ── Conversation Schemas
class ConversationRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = 5

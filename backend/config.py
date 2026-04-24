from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


# ── Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR/"data"
UPLOAD_DIR = DATA_DIR/"uploads"
VECTORSTORE_DIR = DATA_DIR/"vectorstore"

# ── Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── LLM
LLM_MODEL = "llama3.2"
LLM_BASE_URL = "http://localhost:11434"

# ── Chunks
CHUNK_SIZE  = 500
CHUNK_OVERLAP = 50


# ── Retrieval 
RETRIEVAL_K = 5
MMR_DIVERSITY = 0.3

# ── ChromaDB
CHROMA_COLLECTION = "novatech_docs"





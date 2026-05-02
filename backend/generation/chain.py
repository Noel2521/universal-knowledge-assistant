from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List, Tuple
import logging

from backend.config import LLM_MODEL, LLM_BASE_URL

logger = logging.getLogger(__name__)
PROMPT_TEMPLATE = """
You are an expert knowledge assistant for NovaTech AI.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Provide a detailed answer with specific citations like 
"According to [document name], ..." for every claim you make.

Answer:
"""

def format_context(retrieved_docs: List[Document]) -> str:
    """
    Format retrieved documents into a context string.
    """
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        page_info = f", Page {page + 1}" if page != "" else ""
        context_parts.append(
            f"--- Document {i+1}: {source}{page_info} ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(context_parts)

def build_chain(llm: Ollama) -> PromptTemplate:
    """
    Build the QA prompt chain.
    """
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    return prompt

def generate_answer(
    query: str,
    retrieved_docs: List[Document],
    llm: Ollama
) -> str:
    """
    Generate an answer from retrieved documents using the LLM.
    """
    logger.info(f"Generating answer for: {query[:50]}...")
    
    context = format_context(retrieved_docs)
    prompt = build_chain(llm)
    
    formatted_prompt = prompt.format(
        context=context,
        question=query
    )
    
    answer = llm(formatted_prompt)
    logger.info("Answer generated successfully")
    return answer
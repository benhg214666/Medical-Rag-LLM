"""Dependency composition for the RAG service."""

from fastapi import Depends

from app.llm.base import LLMProvider
from app.llm.dependencies import get_llm_provider
from app.rag.service import RAGService
from app.retrieval.dependencies import get_vector_retriever
from app.retrieval.pipeline import RetrievalPipeline


def get_rag_service(
    retriever: RetrievalPipeline = Depends(get_vector_retriever),
    llm: LLMProvider = Depends(get_llm_provider),
) -> RAGService:
    return RAGService(retriever, llm)

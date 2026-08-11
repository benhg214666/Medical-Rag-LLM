"""Orchestrate retrieval, prompt construction, and grounded generation."""

from app.llm.base import LLMProvider
from app.prompts.prompt_builder import build_rag_prompt
from app.rag.models import RAGAnswer, RAGSource
from app.retrieval.models import RetrievalResult
from app.retrieval.vector_retriever import VectorRetriever

NO_CONTEXT_ANSWER = (
    "The available documents do not contain enough information to answer this question."
)


class RAGService:
    def __init__(self, retriever: VectorRetriever, llm: LLMProvider) -> None:
        self.retriever = retriever
        self.llm = llm

    def answer(self, query: str, top_k: int | None = None) -> RAGAnswer:
        retrieved = self.retriever.retrieve(query=query, top_k=top_k)
        usable: list[RetrievalResult] = [item for item in retrieved if item.text.strip()]
        if not usable:
            return RAGAnswer(
                answer=NO_CONTEXT_ANSWER, model=self.llm.model_name, sources=[]
            )

        prompt = build_rag_prompt(query.strip(), usable)
        answer = self.llm.generate(
            system_prompt=prompt.system, user_prompt=prompt.user
        )
        sources = [
            RAGSource(source_number=number, **item.model_dump())
            for number, item in enumerate(usable, start=1)
        ]
        return RAGAnswer(answer=answer, model=self.llm.model_name, sources=sources)

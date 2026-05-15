"""
DocImageAgent — retrieves answers from documents and images using RAG.
"""
import logging
from typing import List, Optional, Tuple

from Prompts.rag_prompts import RELEVANCE_THRESHOLD

logger = logging.getLogger(__name__)


class DocImageAgent:
    """Retrieves and answers questions from indexed documents/images via RAG."""

    def __init__(self, rag_engine, vector_search_tool):
        self._rag = rag_engine
        self._vs_tool = vector_search_tool

    def run(
        self,
        question: str,
        memory,
        n_results: int = 8,
        temperature: float = 0.0,
        source_filter: Optional[List[str]] = None,
        table_answer: str = "",
    ) -> Tuple[str, List[str], int, str]:
        """Answer question from docs/images. Returns (answer, sources, chunks_used, context).

        table_answer: SQL result to inject as extra context for hybrid queries.
        context: the raw text context sent to the LLM (used by hallucination checker).
        """
        if not source_filter:
            results = self._vs_tool.search_per_source(question, n_per_source=4)
        else:
            results = self._vs_tool.search(question, n_results=n_results, source_filter=source_filter)

        relevant = [r for r in results if r.get("distance", 2.0) <= RELEVANCE_THRESHOLD]
        chunks_used = len(relevant)

        if relevant:
            best_dist = min(r.get("distance", 2.0) for r in relevant)
            source_chunks = [r for r in relevant if r.get("distance", 2.0) <= best_dist + 0.3]
        else:
            source_chunks = results if not source_filter else []

        sources = list({r["metadata"].get("source", "") for r in source_chunks})

        # Pass only relevance-filtered chunks to the LLM so irrelevant sources
        # (e.g. reference txt files) don't pollute context.
        llm_results = relevant if relevant else results

        context = self._rag._build_context(llm_results)

        parts = []
        for token in self._rag.query(
            question,
            memory,
            n_results=n_results,
            temperature=temperature,
            stream=False,
            source_filter=source_filter,
            pre_fetched_results=llm_results,
            extra_context=table_answer,
        ):
            parts.append(token)
        answer = "".join(parts)
        return answer, sources, chunks_used, context

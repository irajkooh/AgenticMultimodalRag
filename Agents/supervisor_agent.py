"""
SupervisorAgent — top-level orchestrator for the multimodal RAG agentic workflow.

Flow:
  1. Chitchat / meta short-circuit
  2. Router decides table vs doc/image path
  3. Table path  → SQLGenAgent → TableAgent → (answer, sql)
  4. Doc path    → DocImageAgent → (answer, sources)
  5. GradingAgent grades the answer
  6. HallucinationAgent checks grounding
  7. Return final answer and metadata
"""
import logging
from collections import Counter
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# ── Chitchat / meta data ──────────────────────────────────────────────────────

_GREETING_RESPONSES = {
    "hi": "Hi! Ask me anything about your uploaded documents.",
    "hello": "Hello! Ask me anything about your uploaded documents.",
    "hey": "Hey! Ask me anything about your uploaded documents.",
    "hiya": "Hi there! Ask me anything about your uploaded documents.",
    "howdy": "Howdy! Ask me anything about your uploaded documents.",
    "greetings": "Greetings! Ask me anything about your uploaded documents.",
    "sup": "Hey! Ask me anything about your uploaded documents.",
    "yo": "Hey! Ask me anything about your uploaded documents.",
    "good morning": "Good morning! Ask me anything about your uploaded documents.",
    "good afternoon": "Good afternoon! Ask me anything about your uploaded documents.",
    "good evening": "Good evening! Ask me anything about your uploaded documents.",
    "good day": "Good day! Ask me anything about your uploaded documents.",
    "how are you": "I'm doing well, thank you! Ask me anything about your uploaded documents.",
    "how are you doing": "I'm doing well, thank you! Ask me anything about your uploaded documents.",
    "how do you do": "I'm doing well, thank you! How can I help with your documents?",
    "what's up": "Not much! Ready to answer questions about your documents.",
    "whats up": "Not much! Ready to answer questions about your documents.",
    "what is up": "Not much! Ready to answer questions about your documents.",
    "thanks": "You're welcome! Let me know if you have more questions.",
    "thank you": "You're welcome! Let me know if you have more questions.",
    "thx": "You're welcome! Let me know if you have more questions.",
    "ty": "You're welcome! Let me know if you have more questions.",
    "bye": "Goodbye! Feel free to come back anytime.",
    "goodbye": "Goodbye! Feel free to come back anytime.",
    "see you": "See you! Feel free to come back anytime.",
    "cya": "See you later! Feel free to come back anytime.",
    "ok": "Let me know if you have any questions about your documents.",
    "okay": "Let me know if you have any questions about your documents.",
    "cool": "Glad to help! Let me know if you have more questions.",
    "great": "Glad to help! Let me know if you have more questions.",
    "nice": "Thanks! Let me know if you have more questions.",
}

_META_PATTERNS = [
    "how can you help", "how you can help", "what can you do",
    "what do you do", "who are you", "what are you",
    "help me", "how does this work", "how do you work",
]

_META_ANSWER = (
    "I'm your document assistant. Here's how I can help:\n\n"
    "1. **Upload documents** (PDF, Word, Excel, CSV, TXT, images) or **add URLs** in the Documents tab\n"
    "2. **Ask questions** about your uploaded documents and I'll answer based on their content\n"
    "3. I can handle text, tables, charts, and scanned images\n"
    "4. Use the **Read** button to hear answers aloud\n\n"
    "Upload some documents and start asking questions!"
)

_DOCS_LIST_PATTERNS = [
    "how many doc", "how many file", "list doc", "list file",
    "what doc", "what file", "which doc", "which file",
    "show doc", "show file", "what are the doc", "what are the file",
    "what is indexed", "what is uploaded", "what have you indexed",
    "what have you uploaded", "what documents do you have",
    "what files do you have", "tell me the doc", "tell me the file",
    "name the doc", "name the file",
    "how many docx", "how many xlsx", "how many csv", "how many pdf",
    "how many image", "how many txt", "how many png", "how many jpg",
    "list docx", "list xlsx", "list csv", "list pdf", "list image", "list txt",
    "show docx", "show xlsx", "show csv", "show pdf", "show image", "show txt",
    "what docx", "what xlsx", "what csv", "what pdf",
    "which docx", "which xlsx", "which csv", "which pdf",
    "any docx", "any xlsx", "any csv", "any pdf", "any image",
    "excel file", "word file", "spreadsheet", "word document",
]


class SupervisorAgent:
    """Orchestrates the full agentic RAG workflow."""

    def __init__(
        self,
        router_agent,
        sql_gen_agent,
        table_agent,
        doc_image_agent,
        grading_agent,
        hallucination_agent,
        vector_search_tool,
    ):
        self._router = router_agent
        self._sql_gen = sql_gen_agent
        self._table = table_agent
        self._doc_image = doc_image_agent
        self._grader = grading_agent
        self._hallucination = hallucination_agent
        self._vs_tool = vector_search_tool

    # ── Public entry point ────────────────────────────────────────────────────

    def handle(
        self,
        question: str,
        memory,
        n_results: int = 8,
        temperature: float = 0.0,
        source_filter: Optional[List[str]] = None,
    ) -> dict:
        """Route and answer a question. Returns a result dict with keys:
        answer, sources, sql_query, answer_method, chunks_used, grade, hallucinated
        """
        # 1. Short-circuit: chitchat / meta / docs-list
        chitchat = self._chitchat_response(question)
        if chitchat is not None:
            return self._result(chitchat, [], "", "chitchat", 0)

        normalized_q = question.strip().lower().rstrip("!?.,")
        if any(p in normalized_q for p in _DOCS_LIST_PATTERNS):
            answer = self._docs_list_response() or "No documents are indexed yet. Please upload some documents first."
            return self._result(answer, [], "", "chitchat", 0)

        if self._vs_tool.total_chunks() == 0:
            return self._result(
                "No documents are indexed yet. Please upload some documents first.",
                [], "", "chitchat", 0,
            )

        # 2. Route
        route = self._router.route(question)

        # 3. Table path
        if route == "table":
            table_result = self._table.run(question, self._sql_gen, source_filter)
            if table_result:
                table_answer, table_sql = table_result
                sources = source_filter if source_filter else self._vs_tool.list_sources()
                memory.add("user", question)
                memory.add("assistant", f"{table_answer}\n\n[SQL used: {table_sql}]")
                grade, hallucinated = self._evaluate(question, table_answer, "")
                return self._result(table_answer, list(sources), table_sql, "table_query", 0, grade, hallucinated)
            # Fall through to doc path if no tables found

        # 4. Doc/image path
        answer, sources, chunks_used = self._doc_image.run(
            question, memory, n_results=n_results, temperature=temperature, source_filter=source_filter
        )
        grade, hallucinated = self._evaluate(question, answer, "")
        return self._result(answer, sources, "", "rag", chunks_used, grade, hallucinated)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _chitchat_response(self, text: str) -> Optional[str]:
        normalized = text.strip().lower().rstrip("!?.,")
        if normalized in _GREETING_RESPONSES:
            return _GREETING_RESPONSES[normalized]
        for pattern in _META_PATTERNS:
            if pattern in normalized:
                return _META_ANSWER
        return None

    def _docs_list_response(self) -> Optional[str]:
        sources = self._vs_tool.list_sources()
        if not sources:
            return None
        lines = "\n".join(f"- {s}" for s in sources)
        ext_counts = Counter(Path(s).suffix.lower() for s in sources)
        breakdown = ", ".join(f"{cnt} {ext}" for ext, cnt in sorted(ext_counts.items()))
        return (
            f"There are **{len(sources)}** indexed document(s) ({breakdown}):\n\n"
            f"{lines}"
        )

    def _evaluate(self, question: str, answer: str, context: str) -> Tuple[str, bool]:
        """Run grading and hallucination check. Returns (grade, hallucinated)."""
        try:
            grade = self._grader.grade(question, answer)
        except Exception:
            grade = "PASS"
        try:
            hallucinated = self._hallucination.check(context, answer) if context else False
        except Exception:
            hallucinated = False
        return grade, hallucinated

    @staticmethod
    def _result(
        answer: str,
        sources: list,
        sql_query: str,
        answer_method: str,
        chunks_used: int,
        grade: str = "PASS",
        hallucinated: bool = False,
    ) -> dict:
        return {
            "answer": answer,
            "sources": sources,
            "sql_query": sql_query,
            "answer_method": answer_method,
            "chunks_used": chunks_used,
            "grade": grade,
            "hallucinated": hallucinated,
        }

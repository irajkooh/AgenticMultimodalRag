"""
LLMTool — thin wrapper around the active LLM backend (Groq / HF / Ollama).

Instantiate once and pass into agents that need direct LLM access.
"""
import logging
import os

logger = logging.getLogger(__name__)


class LLMTool:
    """Calls the LLM directly (no RAG context). Shares the same client as RAGEngine."""

    def __init__(self, rag_engine):
        """
        Parameters
        ----------
        rag_engine : RAGEngine
            Already-initialised RAGEngine — we borrow its client, model, and backend.
        """
        self._rag = rag_engine
        from utils.rag_engine import BACKEND
        self._backend = BACKEND

    def call(self, messages: list, max_tokens: int = 512) -> str:
        """Synchronous LLM call. Returns the response text."""
        if self._backend == "groq":
            return self._call_groq(messages, max_tokens)
        elif self._backend == "hf":
            return self._call_hf(messages, max_tokens)
        else:
            return self._call_ollama(messages)

    def _call_groq(self, messages: list, max_tokens: int) -> str:
        try:
            resp = self._rag._client.chat.completions.create(
                model=self._rag.model,
                messages=messages,
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate_limit" in msg or "rate limit" in msg:
                logger.warning("Groq rate limit in LLMTool — falling back to Ollama")
                return self._ollama_fallback(messages)
            raise

    def _call_hf(self, messages: list, max_tokens: int) -> str:
        try:
            resp = self._rag._client.chat_completion(
                model=self._rag.model,
                messages=messages,
                temperature=0.01,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "402" in str(e) or "payment" in str(e).lower() or "depleted" in str(e).lower():
                logger.warning("HF Inference credits depleted in LLMTool — falling back to Ollama")
                return self._ollama_fallback(messages)
            raise

    def _call_ollama(self, messages: list) -> str:
        response = self._rag._client.chat(
            model=self._rag.model,
            messages=messages,
            options={"temperature": 0.0},
        )
        return response["message"]["content"]

    def _ollama_fallback(self, messages: list) -> str:
        import ollama
        from utils.rag_engine import OLLAMA_HOST, DEFAULT_OLLAMA_MODEL
        client = ollama.Client(host=OLLAMA_HOST)
        model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0.0},
        )
        return response["message"]["content"]

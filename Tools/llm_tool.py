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
        # Import backend constant from rag_engine's module
        from utils.rag_engine import BACKEND, DEFAULT_HF_MODEL
        self._backend = BACKEND
        self._default_hf_model = DEFAULT_HF_MODEL
        self._hf_token = os.environ.get("HF_TOKEN") or os.environ.get("AgenticMultiModalRag_Token", "")

    def call(self, messages: list, max_tokens: int = 512) -> str:
        """Synchronous LLM call. Returns the response text.

        Falls back to HuggingFace Inference on Groq rate-limit (if HF_TOKEN set).
        """
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
            if ("429" in msg or "rate_limit" in msg or "rate limit" in msg) and self._hf_token:
                logger.warning("Groq rate limit in LLMTool — falling back to HF Inference")
                return self._hf_fallback(messages, max_tokens)
            raise

    def _call_hf(self, messages: list, max_tokens: int) -> str:
        resp = self._rag._client.chat_completion(
            model=self._rag.model,
            messages=messages,
            temperature=0.01,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    def _call_ollama(self, messages: list) -> str:
        response = self._rag._client.chat(
            model=self._rag.model,
            messages=messages,
            options={"temperature": 0.0},
        )
        return response["message"]["content"]

    def _hf_fallback(self, messages: list, max_tokens: int) -> str:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=self._hf_token)
        model = os.environ.get("HF_MODEL", self._default_hf_model)
        resp = client.chat_completion(
            model=model,
            messages=messages,
            temperature=0.01,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

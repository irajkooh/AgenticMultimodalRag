"""
RouterAgent — decides whether a question should be answered via SQL/tables or via RAG.
"""
import re

_TABLE_INTENT_RE = re.compile(
    r"\b(sum|total|average|avg|mean|maximum|minimum|count|how much|how many|"
    r"calculate|tallest|largest|smallest|highest|lowest|most|least|"
    r"(?:max|min)(?!\s+\d)|"
    r"per (month|year|day|week|item|person|category)|"
    r"sell|sells|sold|selling|"
    r"(sales|revenue|profit|cost|price|amount|balance|credit|debit|spendings?|paid|owe)\b.*\b(of|for|by|in|per)\b|"
    r"\b(of|for|by|in)\b.*\b(sales|revenue|profit|cost|price|amount|balance|credit|debit|spendings?))\b",
    re.IGNORECASE,
)


class RouterAgent:
    """Routes a question to either the table (SQL) path or the document/image (RAG) path."""

    def is_table_question(self, question: str) -> bool:
        """Return True only if the question asks for quantitative/analytical data from tables."""
        return bool(_TABLE_INTENT_RE.search(question))

    def route(self, question: str) -> str:
        """Return 'table' or 'doc'."""
        return "table" if self.is_table_question(question) else "doc"

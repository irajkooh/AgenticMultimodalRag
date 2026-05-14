"""
ClaudeSQLMCP — MCP fallback using Claude API for table selection, SQL generation,
and answer synthesis. Activated when ANTHROPIC_API_KEY is set in the environment.
"""
import logging
import os
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def _build_schema_text(schema_info: list) -> str:
    parts = []
    for s in schema_info:
        parts.append(
            f"Table: {s['table_name']} ({s['nrows']} rows)\n"
            f"Numeric cols: {', '.join(s['numeric_cols']) or 'none'}\n"
            f"Text cols: {', '.join(s['text_cols']) or 'none'}\n"
            f"Sample:\n{s['sample']}"
        )
    return "\n\n---\n\n".join(parts)


class ClaudeSQLMCP:
    """Claude API-backed MCP for table selection, SQL generation, and answer synthesis."""

    _MODEL = "claude-haiku-4-5-20251001"

    def __init__(self):
        self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _call(self, system: str, user: str, max_tokens: int = 512) -> str:
        client = self._get_client()
        resp = client.messages.create(
            model=self._MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip()

    def select_tables(self, question: str, schema_info: list, max_tables: int = 5) -> list:
        """Use Claude to pick the most relevant tables for the question."""
        if len(schema_info) <= 1:
            return schema_info
        schema_text = _build_schema_text(schema_info)
        names = [s["table_name"] for s in schema_info]
        system = (
            "You are a SQL expert. Given a user question and table schemas with sample data, "
            "identify which tables contain the data needed to answer the question. "
            "Return ONLY a comma-separated list of exact table names, nothing else."
        )
        user = (
            f"Question: {question}\n\n"
            f"Available tables:\n{schema_text}\n\n"
            f"Which tables are needed? Choose from: {', '.join(names)}"
        )
        try:
            response = self._call(system, user, max_tokens=128)
            selected_names = {n.strip() for n in response.split(",")}
            selected = [s for s in schema_info if s["table_name"] in selected_names]
            return selected[:max_tables] if selected else schema_info[:max_tables]
        except Exception as e:
            logger.warning(f"ClaudeSQLMCP.select_tables failed: {e}")
            return schema_info[:max_tables]

    def generate_sql(self, question: str, schema_info: list) -> Optional[str]:
        """Use Claude to generate a SQLite SELECT query for the question."""
        schema_text = _build_schema_text(schema_info)
        system = (
            "You are a SQLite expert. Generate a single correct SQLite SELECT query "
            "to answer the user question. Use only exact column names shown in the schema. "
            "Return ONLY the SQL query — no markdown fences, no explanations."
        )
        user = f"Question: {question}\n\nTable schemas:\n{schema_text}"
        try:
            sql = self._call(system, user, max_tokens=256)
            sql = re.sub(r"```(?:sql)?\s*", "", sql, flags=re.IGNORECASE).strip().strip("`").strip()
            return sql if sql.upper().lstrip().startswith("SELECT") else None
        except Exception as e:
            logger.warning(f"ClaudeSQLMCP.generate_sql failed: {e}")
            return None

    def synthesize_answer(
        self,
        question: str,
        sql: str,
        result_str: str,
        detail_str: str = "",
    ) -> str:
        """Use Claude to produce a natural language answer from SQL results."""
        system = (
            "You are a data analyst. Given a question, the SQL used, and the query results, "
            "provide a clear, concise natural language answer focusing on key numbers and insights. "
            "Do not repeat the SQL or raw table data verbatim."
        )
        detail_section = f"\n\nAdditional detail rows:\n{detail_str}" if detail_str else ""
        user = (
            f"Question: {question}\n\n"
            f"SQL used: {sql}\n\n"
            f"Query results:\n{result_str}{detail_section}"
        )
        try:
            return self._call(system, user, max_tokens=512)
        except Exception as e:
            logger.warning(f"ClaudeSQLMCP.synthesize_answer failed: {e}")
            return result_str

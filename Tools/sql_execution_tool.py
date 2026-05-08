"""
SQL execution helpers: fence-stripping and safe query execution.
"""
import re as _re
from typing import Optional, Tuple, List


def strip_sql_fences(raw: str) -> str:
    """Remove markdown code fences from an LLM-generated SQL string."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:])
        if "```" in raw:
            raw = raw[: raw.rfind("```")]
    return raw.strip()


def execute_sql(
    sql: str,
    conn,
) -> Optional[Tuple[str, List, List[str]]]:
    """Execute sql on conn. Returns (sql, rows, col_names) or None on failure."""
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        return sql, rows, col_names
    except Exception:
        return None


def build_detail_sql(aggregate_sql: str) -> Optional[str]:
    """From a single-value aggregate SQL, build a SELECT * query for underlying rows."""
    m = _re.search(r"(FROM\s+\S+(?:\s+WHERE\s+.+)?)", aggregate_sql, _re.IGNORECASE | _re.DOTALL)
    if m:
        return f"SELECT * {m.group(1).rstrip(';')}"
    return None

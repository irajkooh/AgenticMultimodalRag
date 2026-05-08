"""
SQLGenAgent — converts natural language questions to SQLite SQL and executes them.
"""
import logging
import re as _re
from typing import Optional, Tuple, List

from Prompts.sql_prompts import build_sql_prompt, SQL_SYSTEM
from Tools.sql_execution_tool import strip_sql_fences, execute_sql

logger = logging.getLogger(__name__)


class SQLGenAgent:
    """Selects relevant tables, generates SQL via LLM, executes, and returns results."""

    def __init__(self, llm_tool):
        self._llm = llm_tool

    def select_relevant_tables(self, question: str, schema_info: list) -> list:
        """Return the most relevant table(s) based on keyword overlap."""
        if len(schema_info) <= 1:
            return schema_info
        q_words = set(_re.sub(r"[^a-z0-9]", " ", question.lower()).split())
        scores = []
        for s in schema_info:
            col_words = set(_re.sub(r"[^a-z0-9]", " ", " ".join(s["numeric_cols"] + s["text_cols"]).lower()).split())
            sample_words = set(_re.sub(r"[^a-z0-9]", " ", s["sample"].lower()).split())
            score = len(q_words & col_words) + 0.3 * len(q_words & sample_words)
            scores.append(score)
        max_score = max(scores)
        if max_score == 0:
            return schema_info
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] >= 2 * sorted_scores[1] + 0.01:
            return [schema_info[scores.index(max_score)]]
        return [s for s, sc in zip(schema_info, scores) if sc >= max_score * 0.5]

    def generate_and_execute(
        self,
        question: str,
        schema_info: list,
        conn,
    ) -> Optional[Tuple[str, List, List[str]]]:
        """Generate SQL, execute it, retry once on error. Returns (sql, rows, col_names) or None."""
        relevant = self.select_relevant_tables(question, schema_info)
        sql_prompt = build_sql_prompt(question, relevant)
        messages = [
            {"role": "system", "content": SQL_SYSTEM},
            {"role": "user", "content": sql_prompt},
        ]

        for attempt in range(2):
            try:
                llm_out = self._llm.call(messages, max_tokens=512)
            except Exception as e:
                logger.warning(f"LLM SQL generation failed: {e}")
                return None
            if not llm_out:
                return None
            sql = strip_sql_fences(llm_out.strip())
            result = execute_sql(sql, conn)
            if result is not None:
                return result
            logger.warning(f"SQL exec failed (attempt {attempt + 1}): SQL: {sql}")
            if attempt == 0:
                messages = messages + [
                    {"role": "assistant", "content": sql},
                    {"role": "user", "content": "That SQL failed. Return only the corrected SQL query."},
                ]
        return None

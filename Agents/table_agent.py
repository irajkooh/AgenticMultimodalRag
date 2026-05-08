"""
TableAgent — answers quantitative questions using stored tables via SQL + LLM synthesis.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from Prompts.table_prompts import TABLE_ANALYST_SYSTEM, build_table_answer_prompt
from Tools.sql_execution_tool import build_detail_sql

logger = logging.getLogger(__name__)


class TableAgent:
    """Full table query pipeline: extract → load → SQL → synthesize."""

    def __init__(self, llm_tool, table_store, table_extraction_tool, data_dir: str, supported_extensions):
        self._llm = llm_tool
        self._ts = table_store
        self._extractor = table_extraction_tool
        self._data_dir = data_dir
        self._supported_exts = supported_extensions

    def run(
        self,
        question: str,
        sql_gen_agent,
        source_filter: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Answer question via SQL. Returns (answer, sql) or None if no tables available."""
        import pandas as pd

        sources = source_filter or [
            f.name for f in Path(self._data_dir).iterdir()
            if f.suffix.lower() in self._supported_exts
        ]

        # On-demand extraction for sources not yet attempted
        for src in sources:
            if not self._ts.was_attempted(src):
                fp = Path(self._data_dir) / src
                if fp.exists():
                    try:
                        extracted = self._extractor.extract(str(fp))
                        self._ts.save(src, extracted)
                    except Exception as e:
                        logger.warning(f"On-demand table extraction failed for '{src}': {e}")
                        self._ts.save(src, [])

        conn, schema_info = self._ts.load_into_memory(sources)
        if not schema_info:
            conn.close()
            return None

        result = sql_gen_agent.generate_and_execute(question, schema_info, conn)
        if result is None:
            conn.close()
            return None

        sql, rows, col_names = result

        if not rows:
            conn.close()
            return "No matching data found in the tables.", sql

        result_df = pd.DataFrame(rows, columns=col_names) if col_names else pd.DataFrame(rows)
        result_str = result_df.to_string(index=False)

        # For a single aggregate: return directly without LLM synthesis
        if len(rows) == 1 and len(col_names) == 1:
            detail_str = self._fetch_detail_rows(conn, sql)
            conn.close()
            raw_val = rows[0][0]
            col_label = col_names[0]
            answer = f"**{col_label}:** {raw_val}"
            if detail_str:
                answer += detail_str
            return answer, sql

        # Fetch detail rows for aggregate context
        detail_str = self._fetch_detail_rows(conn, sql) if len(rows) == 1 else ""
        conn.close()

        messages = [
            {"role": "system", "content": TABLE_ANALYST_SYSTEM},
            {"role": "user", "content": build_table_answer_prompt(question, sql, result_str, detail_str)},
        ]
        answer = self._llm.call(messages, max_tokens=1024).strip()
        return (answer if answer else result_str), sql

    def _fetch_detail_rows(self, conn, sql: str) -> str:
        detail_sql = build_detail_sql(sql)
        if not detail_sql:
            return ""
        try:
            dcursor = conn.execute(detail_sql)
            drows = dcursor.fetchall()
            if not drows:
                return ""
            import pandas as pd
            dcols = [d[0] for d in dcursor.description]
            ddf = pd.DataFrame(drows, columns=dcols)
            return f"\n\nUnderlying rows:\n{ddf.to_string(index=False)}"
        except Exception:
            return ""

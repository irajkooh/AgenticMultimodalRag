"""
TableAgent — answers quantitative questions using stored tables via SQL + LLM synthesis.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, List

from Prompts.table_prompts import TABLE_ANALYST_SYSTEM, build_table_answer_prompt
from Tools.sql_execution_tool import build_detail_sql

logger = logging.getLogger(__name__)


_ONDEMAND_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}


class TableAgent:
    """Full table query pipeline: extract → load → SQL → synthesize."""

    def __init__(self, llm_tool, table_store, table_extraction_tool, data_dir: str, supported_extensions, mcp=None):
        self._llm = llm_tool
        self._ts = table_store
        self._extractor = table_extraction_tool
        self._data_dir = data_dir
        self._supported_exts = supported_extensions
        self._mcp = mcp

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

        # On-demand extraction only for image files — PDFs/DOCX require explicit indexing
        for src in sources:
            if not self._ts.was_attempted(src) and Path(src).suffix.lower() in _ONDEMAND_EXTS:
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
        try:
            result_str = result_df.to_markdown(index=False)
        except ImportError:
            result_str = result_df.to_string(index=False)

        # NULL aggregate: no rows matched the filter
        if len(rows) == 1 and len(col_names) == 1 and rows[0][0] is None:
            conn.close()
            return "No matching data found for that filter.", sql

        # Fetch detail rows for context on single-value aggregates
        detail_str = self._fetch_detail_rows(conn, sql) if len(rows) == 1 else ""
        conn.close()

        # Use MCP for synthesis when available (better accuracy); fall back to local LLM
        if self._mcp and self._mcp.is_available():
            answer = self._mcp.synthesize_answer(question, sql, result_str, detail_str)
        else:
            messages = [
                {"role": "system", "content": TABLE_ANALYST_SYSTEM},
                {"role": "user", "content": build_table_answer_prompt(question, sql, result_str, detail_str)},
            ]
            answer = self._llm.call(messages, max_tokens=1024).strip()
        final_answer = answer if answer else result_str
        if detail_str:
            final_answer += detail_str
        else:
            final_answer += f"\n\n{result_str}"
        return final_answer, sql

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
            try:
                detail_md = ddf.to_markdown(index=False)
            except ImportError:
                detail_md = ddf.to_string(index=False)
            return f"\n\nUnderlying rows:\n{detail_md}"
        except Exception:
            return ""

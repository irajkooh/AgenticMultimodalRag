"""
TableExtractionTool — extracts DataFrames from files.
Falls back to LLM-based parsing for images and PDFs where heuristics fail.
"""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
_FALLBACK_EXTS = _IMAGE_EXTS | {".pdf"}


class TableExtractionTool:
    def __init__(self, llm_tool):
        self._llm = llm_tool

    def extract(self, filepath: str) -> List:
        """Extract DataFrames from filepath. Returns list of DataFrames (may be empty)."""
        from utils.document_processor import extract_dataframes
        dfs = extract_dataframes(filepath)
        if not dfs and Path(filepath).suffix.lower() in _FALLBACK_EXTS:
            dfs = self._llm_extract(filepath)
        return dfs

    def _llm_extract(self, filepath: str) -> List:
        """LLM-based table extraction fallback for images and PDFs."""
        import io as _io
        import pandas as _pd
        from Prompts.table_prompts import LLM_TABLE_EXTRACT_PROMPT

        ext = Path(filepath).suffix.lower()
        ocr_text = ""
        try:
            if ext in _IMAGE_EXTS:
                from utils.document_processor import ocr_image
                from PIL import Image
                ocr_text = ocr_image(Image.open(filepath).convert("RGB"))
            elif ext == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(filepath)
                ocr_text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            logger.warning(f"LLM table fallback OCR failed for '{filepath}': {e}")
            return []

        if not ocr_text.strip():
            return []

        prompt = LLM_TABLE_EXTRACT_PROMPT + ocr_text
        try:
            csv_text = self._llm.call([{"role": "user", "content": prompt}]).strip()
            if not csv_text or csv_text.startswith("NO_TABLE"):
                return []
            if csv_text.startswith("```"):
                csv_text = "\n".join(l for l in csv_text.splitlines() if not l.startswith("```"))
            df = _pd.read_csv(_io.StringIO(csv_text.strip()))
            return [df] if not df.empty else []
        except Exception as e:
            logger.warning(f"LLM table fallback parse failed for '{filepath}': {e}")
            return []

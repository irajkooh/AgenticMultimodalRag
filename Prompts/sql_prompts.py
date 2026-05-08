SQL_SYSTEM = "Output only a valid SQLite SELECT statement. No markdown. No explanation."


def build_sql_prompt(question: str, schema_info: list) -> str:
    """Build the NL-to-SQL prompt given schema info dicts and the user question."""
    schema_text = "\n\n".join(
        f"Table `{s['table_name']}` (source: {s['source']}, {s['nrows']} rows)\n"
        f"  Numeric columns (REAL — safe for SUM/AVG/MIN/MAX): {', '.join(s['numeric_cols']) or 'none'}\n"
        f"  Text columns (strings only — NEVER use in SUM/AVG/MIN/MAX): {', '.join(s['text_cols'])}\n"
        f"  Sample rows:\n{s['sample']}"
        for s in schema_info
    )
    table_col_blocks = "\n".join(
        f"  `{s['table_name']}` valid columns: "
        + ", ".join(f"`{c}`" for c in s["numeric_cols"] + s["text_cols"])
        for s in schema_info
    )
    text_col_warnings = []
    for s in schema_info:
        bad_cols = [c for c in s["text_cols"] if c.lower() in ("balance", "total", "amount", "price")]
        if bad_cols:
            text_col_warnings.append(
                f"WARNING: {', '.join(bad_cols)} in `{s['table_name']}` are TEXT (OCR output with symbols like '$', '€', ',') — "
                f"they CANNOT be summed. Use {', '.join(s['numeric_cols']) or 'numeric columns'} instead."
            )
    warnings_block = ("\n".join(text_col_warnings) + "\n\n") if text_col_warnings else ""

    return (
        "SQLite database tables:\n\n"
        + schema_text
        + "\n\nCOLUMN CONSTRAINTS (only these column names are valid — no others exist):\n"
        + table_col_blocks
        + "\n\n"
        + warnings_block
        + "RULES:\n"
        "  1. NEVER use text columns in SUM/AVG/MIN/MAX — they contain strings, not numbers.\n"
        "  2. For date comparisons ALWAYS wrap with DATE(): WHERE DATE(Date)='2022-01-02'\n"
        "     WRONG: WHERE Date='2022-01-02'   CORRECT: WHERE DATE(Date)='2022-01-02'\n"
        "  3. For spending/expense queries on a bank table: filter Credit < 0 (negative = debit/spending)\n"
        "  4. ONLY filter on columns explicitly mentioned in the question. Do NOT infer extra filters from sample data.\n"
        "     WRONG (question says 'all items'): WHERE LOWER(Item)='apple' AND LOWER(Sales_Rep)='william'\n"
        "     CORRECT:                           WHERE LOWER(Sales_Rep)='william'\n"
        "  5. Always use LOWER() for text column comparisons.\n"
        "     For exact names: WHERE LOWER(Sales_Rep)='william'\n"
        "     For categories/keywords: WHERE LOWER(Description) LIKE '%grocery%'\n"
        "     NEVER use bare equality for text without LOWER().\n"
        "\n"
        "Query patterns:\n"
        "  Specific day:           WHERE DATE(Date)='2022-01-02'\n"
        "  Month filter:           WHERE strftime('%Y-%m', Date)='2026-03'\n"
        "  Exact text filter:      WHERE LOWER(Sales_Rep)='william'\n"
        "  Category/keyword:       WHERE LOWER(Description) LIKE '%grocery%'\n"
        "  All items for person:   SELECT SUM(Sales) FROM tbl WHERE LOWER(Sales_Rep)='william'\n"
        "  Numeric aggregation:    SELECT SUM(Sales) FROM tbl WHERE ...\n"
        "  Spending total:         SELECT SUM(Credit) FROM tbl WHERE strftime('%Y-%m', Date)='2026-03' AND Credit < 0\n"
        "  Spending by category:   SELECT SUM(Credit) FROM tbl WHERE strftime('%Y-%m', Date)='2026-03' AND LOWER(Description) LIKE '%grocery%' AND Credit < 0\n"
        "  Spending list:          SELECT Date, Description, Credit FROM tbl WHERE Credit < 0 ORDER BY Credit\n"
        "\n"
        f"Question: {question}\n\n"
        "Write one SQLite SELECT statement. OUTPUT ONLY THE SQL — no markdown, no explanation.\n"
    )

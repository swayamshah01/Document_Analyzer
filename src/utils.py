"""
utils.py
--------
Shared helpers for data export and formatting.
"""

import io
import pandas as pd


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes (for st.download_button)."""
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Entities") -> bytes:
    """Serialise a DataFrame to Excel (.xlsx) bytes (for st.download_button)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Return text truncated to max_chars with an ellipsis indicator."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n… [truncated – {len(text) - max_chars:,} more characters]"


def highlight_query_in_segment(segment: str, query: str) -> str:
    """Wrap occurrences of query words in the segment for Markdown bold display."""
    import re
    words = [w.strip() for w in query.split() if len(w.strip()) > 3]
    for word in words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        segment = pattern.sub(f"**{word}**", segment)
    return segment

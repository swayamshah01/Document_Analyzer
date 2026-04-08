"""
extractor.py
------------
PDF Text Extraction Engine.
Handles multi-column layouts, embedded tables, and common insurance
document quirks using pdfplumber. Includes a regex-based text cleaner
that strips noise before NLP processing.
"""

import re
import pdfplumber
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> tuple[str, list[str], int]:
    """
    Extract raw text from every page of an uploaded PDF.

    Args:
        uploaded_file: Streamlit UploadedFile object (BytesIO-compatible).

    Returns:
        (full_text, page_texts, page_count)
        - full_text   : entire document as a single string (cleaned)
        - page_texts  : list of per-page cleaned strings
        - page_count  : total number of pages
    """
    page_texts: list[str] = []

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                raise ValueError("The uploaded PDF contains no pages.")

            for page in pdf.pages:
                # --- Primary extraction: layout-aware text ---
                text = page.extract_text(layout=True) or ""

                # --- If layout mode misses content, fall back to table scan ---
                if len(text.strip()) < 30:
                    tables = page.extract_tables()
                    if tables:
                        rows = []
                        for table in tables:
                            for row in table:
                                rows.append(
                                    " | ".join(
                                        cell.strip() if cell else ""
                                        for cell in row
                                    )
                                )
                        text = "\n".join(rows)

                page_texts.append(clean_text(text))

        full_text = "\n\n".join(page_texts)

        if not full_text.strip():
            raise ValueError(
                "No readable text found. The PDF may be image-based (scanned). "
                "Please use a text-selectable PDF."
            )

        return full_text, page_texts, page_count

    except Exception as exc:
        msg = str(exc).lower()
        if "password" in msg or "encrypted" in msg:
            raise ValueError("The PDF is password-protected. Please provide an unencrypted file.") from exc
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

# Common noise patterns in insurance PDFs — compiled individually
# (Python disallows inline flags like (?im) after position 0 when joined with |)
_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\d+\s*$", re.MULTILINE),
    re.compile(r"^confidential\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^proprietary\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[-─═]{3,}$", re.MULTILINE),
    re.compile(r"^www\.\S+\.\S+$", re.MULTILINE),
    re.compile(r"\x0c"),
]


def clean_text(raw: str) -> str:
    """
    Normalise raw PDF text for NLP consumption.

    Pipeline:
        1. Remove header/footer noise patterns
        2. Replace non-breaking spaces and control chars
        3. Collapse 2+ blank lines into a single blank line
        4. Strip leading/trailing whitespace per line
        5. Strip the whole document
    """
    if not raw:
        return ""

    # Step 1 — remove known noise patterns (applied sequentially)
    text = raw
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub("", text)

    # Step 2 — normalise whitespace chars
    text = text.replace("\xa0", " ")          # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)       # multiple spaces/tabs → single space

    # Step 3 — clean per-line leading/trailing spaces
    lines = [line.strip() for line in text.splitlines()]

    # Step 4 — collapse runs of blank lines
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append("")
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_text_stats(full_text: str, page_count: int) -> dict:
    """Return a dict of basic document statistics for dashboard display."""
    words = full_text.split()
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    return {
        "pages": page_count,
        "words": len(words),
        "sentences": len(sentences),
        "characters": len(full_text),
        "avg_words_per_page": round(len(words) / max(page_count, 1)),
    }

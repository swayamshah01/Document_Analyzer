"""
nlp_pipeline.py
---------------
spaCy NER Intelligence Layer.
Loads en_core_web_sm, processes document text, filters entities
relevant to insurance/financial analysis, and builds export DataFrames.
"""

import pandas as pd
import spacy
import streamlit as st
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Entity category definitions
# ─────────────────────────────────────────────────────────────────────────────

# General labels for broad document intelligence
BUSINESS_LABELS = {
    "PERSON", "ORG", "MONEY", "DATE", "TIME", "GPE", "LOC", 
    "LAW", "PERCENT", "PRODUCT", "EVENT", "FAC"
}

# Human-readable descriptions for the UI
LABEL_DESCRIPTIONS = {
    "PERSON":  "People & Individuals",
    "ORG":     "Organizations & Companies",
    "MONEY":   "Financial Values & Currency",
    "DATE":    "Dates & Deadlines",
    "TIME":    "Time & Duration",
    "GPE":     "Geo-Political (Countries, Cities, States)",
    "LOC":     "Locations & Geography",
    "LAW":     "Legal & Regulatory References",
    "PERCENT": "Percentages & Rates",
    "PRODUCT": "Products & Goods",
    "EVENT":   "Events & Occurrences",
    "FAC":     "Facilities & Infrastructure",
}

# displaCy colour overrides so we match the Navy/Teal theme
DISPLACY_COLORS = {
    "PERSON":  "#d6e3ff",
    "ORG":     "#d5e0f7",
    "MONEY":   "#b1f0ce",
    "DATE":    "#e9e7eb",
    "TIME":    "#e9e7eb",
    "GPE":     "#d6e3ff",
    "LOC":     "#d5e0f7",
    "LAW":     "#b1f0ce",
    "PERCENT": "#b1f0ce",
    "PRODUCT": "#f4f3f7",
    "EVENT":   "#f4f3f7",
    "FAC":     "#f4f3f7",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Load and cache a spaCy model.
    Falls back gracefully if the model is not installed.
    """
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        raise OSError(
            f"spaCy model '{model_name}' not found. "
            f"Run: python -m spacy download {model_name}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Document processing
# ─────────────────────────────────────────────────────────────────────────────

def process_document(text: str, nlp) -> spacy.tokens.Doc:
    """
    Run the spaCy NLP pipeline on cleaned text.
    Truncates to 1 000 000 chars to avoid memory issues on huge PDFs.
    """
    if not text or not text.strip():
        raise ValueError("Cannot process empty text.")

    truncated = text[:1_000_000]
    # Increase max_length if needed
    if len(truncated) > nlp.max_length:
        nlp.max_length = len(truncated) + 100

    return nlp(truncated)


# ─────────────────────────────────────────────────────────────────────────────
# Entity filtering & structuring
# ─────────────────────────────────────────────────────────────────────────────

def filter_business_entities(
    doc: spacy.tokens.Doc,
    labels: Optional[set] = None,
    min_length: int = 2,
) -> list[dict]:
    """
    Extract only business-relevant entities from a spaCy Doc.

    Args:
        doc        : Processed spaCy document.
        labels     : Set of entity labels to keep (defaults to BUSINESS_LABELS).
        min_length : Minimum character length of entity text to include.

    Returns:
        List of dicts: {text, label, description, start_char, end_char}
    """
    target_labels = labels or BUSINESS_LABELS
    seen: set[str] = set()
    results: list[dict] = []

    for ent in doc.ents:
        if ent.label_ not in target_labels:
            continue
        text = ent.text.strip()
        if len(text) < min_length:
            continue
        key = f"{ent.label_}::{text.lower()}"
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "Entity": text,
                "Label": ent.label_,
                "Description": LABEL_DESCRIPTIONS.get(ent.label_, ent.label_),
                "Start": ent.start_char,
                "End": ent.end_char,
            }
        )

    return results


def entities_to_dataframe(entities: list[dict]) -> pd.DataFrame:
    """Convert the entity list to a sorted Pandas DataFrame."""
    if not entities:
        return pd.DataFrame(
            columns=["Entity", "Label", "Description", "Start", "End"]
        )
    df = pd.DataFrame(entities)
    df = df.sort_values(["Label", "Entity"]).reset_index(drop=True)
    return df


def get_entity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivot-style summary showing counts per label.
    Useful for the metrics row in the dashboard.
    """
    if df.empty:
        return pd.DataFrame(columns=["Label", "Count", "Description"])
    summary = (
        df.groupby(["Label", "Description"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )
    return summary

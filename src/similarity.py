"""
similarity.py
-------------
Semantic Similarity Engine.
Uses sentence-transformers (all-MiniLM-L6-v2) to convert text segments
into vector embeddings and finds the most semantically similar passages
to a user query via cosine similarity.
"""

import re
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_similarity_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and cache the sentence-transformer model.
    Downloads on first run (~80 MB); subsequent calls return the cached instance.
    """
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load similarity model '{model_name}': {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Text chunking
# ─────────────────────────────────────────────────────────────────────────────

def split_into_segments(
    text: str,
    segment_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    Split document text into overlapping word-based windows.

    Args:
        text         : Full document text.
        segment_size : Number of words per segment.
        overlap      : Number of words to overlap between segments.

    Returns:
        List of text segments (strings).
    """
    if not text or not text.strip():
        return []

    # Split on whitespace
    words = text.split()
    if len(words) == 0:
        return []

    step = max(segment_size - overlap, 1)
    segments: list[str] = []

    for i in range(0, len(words), step):
        chunk = words[i : i + segment_size]
        if len(chunk) < 10:          # ignore micro-fragments at tail
            break
        segments.append(" ".join(chunk))

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Semantic search
# ─────────────────────────────────────────────────────────────────────────────

def encode_segments(
    segments: list[str],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode a list of text segments into embedding vectors.

    Returns:
        numpy array of shape (N, embedding_dim)
    """
    if not segments:
        raise ValueError("No segments provided for encoding.")
    embeddings = model.encode(
        segments,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalised → dot product == cosine
    )
    return embeddings


def find_similar_segments(
    query: str,
    segments: list[str],
    model: SentenceTransformer,
    segment_embeddings: np.ndarray | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Find the top-k most semantically similar segments to a query.

    Args:
        query              : User-provided master clause or search text.
        segments           : List of document text segments.
        model              : Loaded SentenceTransformer model.
        segment_embeddings : Pre-computed embeddings (optional; avoids re-encoding).
        top_k              : Number of results to return.

    Returns:
        List of dicts: {rank, segment, score, score_pct}
        sorted by descending similarity.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if not segments:
        raise ValueError("No document segments to search.")

    # Encode query
    query_embedding = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )  # shape (1, dim)

    # Encode segments if not pre-computed
    if segment_embeddings is None:
        segment_embeddings = encode_segments(segments, model)

    # Cosine similarity — since embeddings are L2-normalised,
    # dot product is equivalent and faster
    scores = cosine_similarity(query_embedding, segment_embeddings)[0]  # shape (N,)

    # Get top-k indices (descending)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict] = []
    # Set a minimum threshold so completely unrelated questions don't just 
    # return the topmost noise segments. 30% scaled matches ~0.09 raw cosine.
    MIN_SCALED_SCORE = 30.0
    
    current_rank = 1
    for idx in top_indices:
        score = float(scores[idx])
        
        # Scale score to make it intuitive for users. Cosine similarities from 
        # SentenceTransformers often yield computationally correct but visually 
        # "low" scores (e.g., 0.40). A square root curve maps them better to expectations.
        scaled_score_pct = (score ** 0.5) * 100 if score > 0 else 0
        
        # Ignore wildly irrelevant matches
        if scaled_score_pct < MIN_SCALED_SCORE:
            continue
            
        results.append(
            {
                "rank": current_rank,
                "segment": segments[idx],
                "score": round(score, 4),
                "score_pct": round(scaled_score_pct, 1),
                "segment_index": int(idx),
            }
        )
        current_rank += 1

    return results

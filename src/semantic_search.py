"""
src/semantic_search.py
-----------------------
Fast semantic search using precomputed candidate embeddings.

Instead of encoding all 100,000 candidates on every run (1000–2000 seconds),
this module loads the pre-generated embedding matrix and only encodes the JD
(< 1 second), then computes similarity via a single matrix-vector dot product
(< 2 seconds). Total semantic search time: < 3 seconds.

This is the same technique used in production vector search systems
(LinkedIn Talent Solutions, Indeed, FAISS-based retrieval pipelines):
    1. Candidate profiles change slowly → encode offline, reuse many times.
    2. JDs change frequently → encode on demand (fast: single short text).
    3. Similarity = O(N·D) matrix-vector multiply → ~1s for 100k × 384-dim.

Public API:
    compute_semantic_scores(jd_text, model) -> list[dict]
        Returns [{"candidate_id": str, "embedding_score": float}, ...]
        sorted descending by embedding_score, scores in range 0–100.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Point sentence-transformers to local model cache (consistent with other modules)
_PROJECT_ROOT    = Path(__file__).resolve().parent.parent
_LOCAL_MODEL_DIR = str(_PROJECT_ROOT / "models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _LOCAL_MODEL_DIR)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from src.embedding_store import load_embeddings, get_candidate_ids, EmbeddingStoreError
from src.hybrid_ranker   import get_model, JD_TEXT


# ---------------------------------------------------------------------------
# Process-level JD embedding cache (avoids re-encoding same JD text)
# ---------------------------------------------------------------------------

_jd_cache: Dict[str, np.ndarray] = {}


def _encode_jd(jd_text: str, model) -> np.ndarray:
    """
    Encode JD text with L2 normalization. Cached by jd_text content.
    """
    if jd_text not in _jd_cache:
        logger.info("Encoding JD (%d chars)...", len(jd_text))
        emb = model.encode(jd_text, normalize_embeddings=True)
        _jd_cache[jd_text] = emb
        logger.info("JD encoded — shape: %s", emb.shape)
    else:
        logger.debug("JD embedding cache hit.")
    return _jd_cache[jd_text]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_semantic_scores(
    jd_text: str = JD_TEXT,
    model=None,
    top_n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Compute cosine similarity between a JD and all precomputed candidate embeddings.

    Does NOT encode candidate profiles — loads the pre-generated matrix from disk.
    Candidate encoding is done offline by scripts/generate_embeddings.py.

    Steps
    -----
    1. Load cached embedding matrix (N × D) from disk — once per process.
    2. Encode JD text only — < 1 second.
    3. Compute dot product (= cosine similarity, since embeddings are L2-normalized).
    4. Return sorted list of (candidate_id, embedding_score).

    Parameters
    ----------
    jd_text : Job Description text to encode and compare.
    model   : Pre-loaded SentenceTransformer. If None, loads from local cache.
    top_n   : If provided, return only the top-N results. Default: all candidates.

    Returns
    -------
    list[dict] sorted descending by embedding_score:
        [{"candidate_id": str, "embedding_score": float}, ...]
        embedding_score is in range 0–100 (cosine similarity × 100).

    Raises
    ------
    EmbeddingStoreError
        if precomputed embeddings are not found.
        Run: python scripts/generate_embeddings.py
    """
    import time

    # ── Step 1: Load precomputed embeddings ───────────────────────────────────
    t0 = time.perf_counter()
    embeddings = load_embeddings()           # (N, 384) float32, cached after first call
    candidate_ids = get_candidate_ids()      # (N,) str
    t_load = time.perf_counter() - t0
    n = len(embeddings)
    logger.info(
        "Embeddings loaded: %d candidates × %d dims (%.2f s)",
        n, embeddings.shape[1], t_load,
    )

    # ── Step 2: Encode JD ─────────────────────────────────────────────────────
    t1 = time.perf_counter()
    _model = model if model is not None else get_model()
    jd_emb = _encode_jd(jd_text, _model)    # shape (384,), L2-normalized
    t_jd = time.perf_counter() - t1
    logger.info("JD encoded in %.2f s.", t_jd)

    # ── Step 3: Cosine similarity (dot product on L2-normalized vectors) ───────
    # embeddings: (N, D), jd_emb: (D,) → result: (N,)
    t2 = time.perf_counter()
    similarity: np.ndarray = embeddings @ jd_emb   # cosine similarity
    t_sim = time.perf_counter() - t2
    logger.info(
        "Similarity computed for %d candidates in %.2f s.", n, t_sim,
    )

    # ── Step 4: Build result list ─────────────────────────────────────────────
    t3 = time.perf_counter()

    # Scale to 0–100
    scores_scaled = (similarity * 100.0).tolist()

    # Combine with IDs and sort
    results = [
        {
            "candidate_id":    str(candidate_ids[i]),
            "embedding_score": round(scores_scaled[i], 4),
        }
        for i in range(n)
    ]
    results.sort(key=lambda r: r["embedding_score"], reverse=True)

    if top_n is not None:
        results = results[:top_n]

    t_build = time.perf_counter() - t3

    # ── Summary ───────────────────────────────────────────────────────────────
    total = t_load + t_jd + t_sim + t_build
    logger.info(
        "compute_semantic_scores: total=%.2fs "
        "[load=%.2fs, jd=%.2fs, sim=%.2fs, build=%.2fs]",
        total, t_load, t_jd, t_sim, t_build,
    )

    if results:
        logger.info(
            "Score range: %.2f (rank 1) – %.2f (rank %d)",
            results[0]["embedding_score"],
            results[-1]["embedding_score"],
            len(results),
        )

    return results


# ---------------------------------------------------------------------------
# Convenience: build a candidate_id → embedding_score lookup dict
# ---------------------------------------------------------------------------

def get_score_lookup(jd_text: str = JD_TEXT, model=None) -> Dict[str, float]:
    """
    Return a dict mapping candidate_id -> embedding_score (0–100).
    Useful for merging embedding scores with feature scores in the hybrid pipeline.
    """
    scores = compute_semantic_scores(jd_text=jd_text, model=model)
    return {r["candidate_id"]: r["embedding_score"] for r in scores}

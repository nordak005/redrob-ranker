"""
src/embedding_store.py
-----------------------
Thread-safe, process-level singleton for precomputed candidate embeddings.

Embeddings are generated offline by scripts/generate_embeddings.py and
loaded once per process. Subsequent calls return the cached arrays.

Public API:
    load_embeddings()   -> np.ndarray  shape (N, 384) float32, L2-normalized
    get_candidate_ids() -> np.ndarray  shape (N,)     str
    validate_embeddings()              raises EmbeddingStoreError on problems
    get_metadata()      -> dict        contents of embedding_metadata.json
    is_loaded()         -> bool        True if cache is populated
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

_PROJECT_ROOT   = Path(__file__).resolve().parent.parent
_DATA_DIR       = _PROJECT_ROOT / "data"
_EMB_PATH       = _DATA_DIR / "candidate_embeddings.npy"
_IDS_PATH       = _DATA_DIR / "candidate_ids.npy"
_META_PATH      = _DATA_DIR / "embedding_metadata.json"

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class EmbeddingStoreError(RuntimeError):
    """Raised when the embedding store cannot be loaded or is invalid."""
    pass

# ---------------------------------------------------------------------------
# Module-level cache (process singleton)
# ---------------------------------------------------------------------------

_embeddings:    Optional[np.ndarray] = None
_candidate_ids: Optional[np.ndarray] = None
_metadata:      Optional[dict]       = None


def is_loaded() -> bool:
    """Return True if embeddings are already loaded into memory."""
    return _embeddings is not None


def load_embeddings(
    embeddings_path: Path = _EMB_PATH,
    ids_path:        Path = _IDS_PATH,
    meta_path:       Path = _META_PATH,
    force_reload:    bool = False,
) -> np.ndarray:
    """
    Load precomputed candidate embeddings from disk into memory.

    Loads only ONCE per process — subsequent calls return the cached array.
    Pass force_reload=True to refresh from disk (e.g. after re-generating).

    Parameters
    ----------
    embeddings_path : path to candidate_embeddings.npy
    ids_path        : path to candidate_ids.npy
    meta_path       : path to embedding_metadata.json
    force_reload    : if True, reload from disk even if already cached

    Returns
    -------
    np.ndarray — shape (N, 384), dtype float32, L2-normalized

    Raises
    ------
    EmbeddingStoreError
        if files are missing, shapes are inconsistent, or arrays are invalid
    """
    global _embeddings, _candidate_ids, _metadata

    if _embeddings is not None and not force_reload:
        logger.debug("Embedding store: cache hit (%d candidates).", len(_embeddings))
        return _embeddings

    # ── Check files exist ─────────────────────────────────────────────────────
    missing = [p for p in (embeddings_path, ids_path) if not p.exists()]
    if missing:
        raise EmbeddingStoreError(
            f"Precomputed embedding files not found: {[str(p) for p in missing]}\n"
            f"Run:  python scripts/generate_embeddings.py"
        )

    # ── Load arrays ───────────────────────────────────────────────────────────
    import time
    t0 = time.perf_counter()

    logger.info("Loading embeddings from %s ...", embeddings_path)
    emb = np.load(str(embeddings_path))

    logger.info("Loading candidate IDs from %s ...", ids_path)
    ids = np.load(str(ids_path), allow_pickle=True)

    elapsed = time.perf_counter() - t0

    # ── Load metadata (optional — don't fail if missing) ──────────────────────
    meta: dict = {}
    if meta_path.exists():
        with open(str(meta_path), "r", encoding="utf-8") as f:
            meta = json.load(f)

    # ── Validate ──────────────────────────────────────────────────────────────
    if emb.ndim != 2:
        raise EmbeddingStoreError(
            f"Expected 2D embedding array, got shape {emb.shape}"
        )
    if len(emb) != len(ids):
        raise EmbeddingStoreError(
            f"Embeddings and IDs have different lengths: "
            f"{len(emb)} embeddings vs {len(ids)} IDs"
        )
    if emb.dtype != np.float32:
        logger.info("Casting embeddings from %s to float32.", emb.dtype)
        emb = emb.astype(np.float32)

    # ── Populate cache ────────────────────────────────────────────────────────
    _embeddings    = emb
    _candidate_ids = ids
    _metadata      = meta

    n, dim = emb.shape
    logger.info(
        "Embedding store ready: %d candidates × %d dims, loaded in %.2f s.",
        n, dim, elapsed,
    )
    return _embeddings


def get_candidate_ids() -> np.ndarray:
    """
    Return the array of candidate ID strings corresponding to each embedding row.

    Raises EmbeddingStoreError if load_embeddings() has not been called.
    """
    if _candidate_ids is None:
        raise EmbeddingStoreError(
            "Embeddings not loaded. Call load_embeddings() first."
        )
    return _candidate_ids


def get_metadata() -> dict:
    """
    Return the embedding metadata dict (model name, timestamp, dimensions, etc.).

    Returns an empty dict if metadata.json was not found.
    """
    if _metadata is None:
        raise EmbeddingStoreError(
            "Embeddings not loaded. Call load_embeddings() first."
        )
    return _metadata


def validate_embeddings(
    embeddings_path: Path = _EMB_PATH,
    ids_path:        Path = _IDS_PATH,
) -> dict:
    """
    Validate that precomputed embeddings exist and are internally consistent.

    Does NOT require load_embeddings() to have been called first —
    loads and checks from disk directly.

    Returns
    -------
    dict with validation results:
        ok           : bool
        num_candidates, embedding_dim, dtype, normalized, ids_match

    Raises
    ------
    EmbeddingStoreError if critical checks fail.
    """
    # Re-use the load function for validation (populates cache as side effect)
    emb = load_embeddings(embeddings_path, ids_path)
    ids = get_candidate_ids()
    n, dim = emb.shape

    # Check L2 norms (should be ~1.0 for normalized embeddings)
    sample_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    norms = np.linalg.norm(emb[sample_idx], axis=1)
    normalized = bool(np.allclose(norms, 1.0, atol=1e-5))

    result = {
        "ok":             True,
        "num_candidates": n,
        "embedding_dim":  dim,
        "dtype":          str(emb.dtype),
        "normalized":     normalized,
        "ids_match":      len(ids) == n,
        "sample_norms":   [round(float(x), 6) for x in norms],
    }

    if not normalized:
        logger.warning(
            "Embeddings do not appear L2-normalized (sample norms: %s). "
            "Cosine similarity via dot-product may be inaccurate.",
            result["sample_norms"],
        )

    logger.info(
        "Validation: %d candidates, %d dims, dtype=%s, normalized=%s",
        n, dim, emb.dtype, normalized,
    )
    return result

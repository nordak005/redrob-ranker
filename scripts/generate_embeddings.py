#!/usr/bin/env python3
"""
scripts/generate_embeddings.py
-------------------------------
OFFLINE embedding pipeline — run ONCE to precompute all candidate embeddings.

After this script completes, runtime ranking never needs to re-encode candidates.
Semantic scoring becomes: encode JD (< 1s) + cosine similarity (< 2s) = < 3s total.

Usage:
    python scripts/generate_embeddings.py

Outputs:
    data/candidate_embeddings.npy    shape (N, 384) float32
    data/candidate_ids.npy           shape (N,)     string array
    data/embedding_metadata.json     provenance record

Runtime: ~90 seconds on CPU for 100,000 candidates.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Local model cache (consistent with hybrid_ranker.py) ─────────────────────
_LOCAL_MODEL_DIR = str(_PROJECT_ROOT / "models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _LOCAL_MODEL_DIR)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME      = "all-MiniLM-L6-v2"
BATCH_SIZE      = 64
CANDIDATES_PATH = _PROJECT_ROOT / "data" / "raw" / "candidates.jsonl.gz"
OUT_DIR         = _PROJECT_ROOT / "data"
EMB_PATH        = OUT_DIR / "candidate_embeddings.npy"
IDS_PATH        = OUT_DIR / "candidate_ids.npy"
META_PATH       = OUT_DIR / "embedding_metadata.json"


# ---------------------------------------------------------------------------
# Candidate text builder — MUST match hybrid_ranker.py exactly
# ---------------------------------------------------------------------------

def _candidate_text(candidate: dict) -> str:
    """
    Build structured text for embedding.
    Identical to hybrid_ranker._candidate_text() so scores are consistent.
    """
    profile = candidate.get("profile", {})
    title    = profile.get("current_title", "").strip()
    headline = profile.get("headline", "").strip()
    summary  = profile.get("summary", "").strip()

    career_titles = ", ".join(
        role.get("title", "").strip()
        for role in candidate.get("career_history", [])
        if role.get("title", "").strip()
    )
    skills = ", ".join(
        s.get("name", "").strip()
        for s in candidate.get("skills", [])
        if s.get("name", "").strip()
    )
    return (
        f"Title: {title}\n"
        f"Headline: {headline}\n"
        f"Summary: {summary}\n"
        f"Career Titles: {career_titles}\n"
        f"Skills: {skills}"
    ).strip()


# ---------------------------------------------------------------------------
# Candidate loader
# ---------------------------------------------------------------------------

def _load_candidates(path: Path) -> tuple[list[str], list[str]]:
    """
    Stream candidates from JSONL.gz.

    Returns
    -------
    candidate_ids : list[str]
    texts         : list[str]  (pre-built for embedding)
    """
    logger.info("Loading candidates from %s ...", path)
    t0 = time.perf_counter()
    ids, texts = [], []
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                cid  = cand.get("candidate_id", f"UNKNOWN_{i}")
                ids.append(cid)
                texts.append(_candidate_text(cand))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", i, exc)

            if i % 20_000 == 0:
                logger.info("  Streamed %6d candidates...", i)

    elapsed = time.perf_counter() - t0
    logger.info("Loaded %d candidates in %.2f s.", len(ids), elapsed)
    return ids, texts


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model():
    """Load MiniLM from local cache; download only if not cached."""
    from sentence_transformers import SentenceTransformer

    models_dir = Path(_LOCAL_MODEL_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    hf_cache_dir = models_dir / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    if hf_cache_dir.exists():
        snapshots = sorted(hf_cache_dir.iterdir())
        if snapshots:
            local_path = str(snapshots[-1])
            logger.info("Loading model from local cache: %s", local_path)
            return SentenceTransformer(local_path, cache_folder=_LOCAL_MODEL_DIR)

    logger.info("Downloading %s (first run only)...", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME, cache_folder=_LOCAL_MODEL_DIR)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("=" * 62)
    logger.info("Redrob — Offline Embedding Pipeline")
    logger.info("Model : %s", MODEL_NAME)
    logger.info("Input : %s", CANDIDATES_PATH)
    logger.info("Output: %s", OUT_DIR)
    logger.info("=" * 62)

    if not CANDIDATES_PATH.exists():
        logger.error("Candidates file not found: %s", CANDIDATES_PATH)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # ── Step 1: Load candidates ───────────────────────────────────────────────
    candidate_ids, texts = _load_candidates(CANDIDATES_PATH)
    n = len(candidate_ids)
    if n == 0:
        logger.error("No candidates loaded. Aborting.")
        return 1

    # ── Step 2: Load model ────────────────────────────────────────────────────
    logger.info("Loading embedding model...")
    t0 = time.perf_counter()
    model = _load_model()
    logger.info("Model loaded in %.2f s.", time.perf_counter() - t0)

    # ── Step 3: Encode all candidates ─────────────────────────────────────────
    logger.info("Encoding %d candidates (batch_size=%d)...", n, BATCH_SIZE)
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # L2-normalize for cosine via dot-product
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    encode_time = time.perf_counter() - t0
    logger.info(
        "Encoded %d candidates in %.2f s (%.1f cands/sec).",
        n, encode_time, n / encode_time,
    )

    emb_dim = embeddings.shape[1]
    logger.info("Embedding shape: %s  dtype: %s", embeddings.shape, embeddings.dtype)

    # ── Step 4: Save embeddings ───────────────────────────────────────────────
    logger.info("Saving embeddings to %s ...", EMB_PATH)
    np.save(str(EMB_PATH), embeddings.astype(np.float32))

    logger.info("Saving candidate IDs to %s ...", IDS_PATH)
    np.save(str(IDS_PATH), np.array(candidate_ids, dtype=str))

    # ── Step 5: Write metadata ────────────────────────────────────────────────
    metadata = {
        "model_name":       MODEL_NAME,
        "num_candidates":   n,
        "embedding_dim":    emb_dim,
        "normalized":       True,
        "batch_size":       BATCH_SIZE,
        "encode_time_sec":  round(encode_time, 2),
        "timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "embeddings_file":  str(EMB_PATH.name),
        "ids_file":         str(IDS_PATH.name),
        "source_file":      str(CANDIDATES_PATH.name),
    }
    with open(str(META_PATH), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata written to %s", META_PATH)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    emb_size_mb = embeddings.nbytes / 1_048_576

    logger.info("=" * 62)
    logger.info("DONE")
    logger.info("  Candidates  : %d", n)
    logger.info("  Dimensions  : %d", emb_dim)
    logger.info("  Encode time : %.2f s", encode_time)
    logger.info("  Total time  : %.2f s", total_time)
    logger.info("  File size   : %.1f MB (float32)", emb_size_mb)
    logger.info("  Saved to    : %s", OUT_DIR.resolve())
    logger.info("=" * 62)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run benchmark : python scripts/benchmark_embeddings.py")
    logger.info("  2. Run ranking   : python scripts/generate_submission.py")
    logger.info("  3. Launch app    : streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

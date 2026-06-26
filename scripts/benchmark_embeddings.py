#!/usr/bin/env python3
"""
scripts/benchmark_embeddings.py
---------------------------------
Measure the runtime of the precomputed embedding pipeline.

Benchmarks:
    1. Embedding load time (data/candidate_embeddings.npy)
    2. JD encoding time
    3. Cosine similarity computation time
    4. Total semantic search time

Usage:
    python scripts/benchmark_embeddings.py

Prerequisites:
    python scripts/generate_embeddings.py   # must be run first
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_LOCAL_MODEL_DIR = str(_PROJECT_ROOT / "models")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _LOCAL_MODEL_DIR)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

logging.basicConfig(
    level=logging.WARNING,   # suppress verbose logs during benchmark
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from src.embedding_store import load_embeddings, get_candidate_ids, get_metadata, validate_embeddings
from src.hybrid_ranker   import get_model, get_jd_embedding, JD_TEXT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char="-", width=57) -> str:
    return char * width


def _fmt(seconds: float) -> str:
    return f"{seconds:.2f} sec"


# ---------------------------------------------------------------------------
# Benchmark sections
# ---------------------------------------------------------------------------

def bench_load() -> tuple[np.ndarray, np.ndarray, float]:
    """Benchmark embedding load time."""
    t0 = time.perf_counter()
    embeddings    = load_embeddings()
    candidate_ids = get_candidate_ids()
    elapsed = time.perf_counter() - t0
    return embeddings, candidate_ids, elapsed


def bench_model_load() -> tuple[object, float]:
    """Benchmark model load time from local cache."""
    t0 = time.perf_counter()
    model = get_model()
    elapsed = time.perf_counter() - t0
    return model, elapsed


def bench_jd_encode(model) -> tuple[np.ndarray, float]:
    """Benchmark JD encoding time."""
    t0 = time.perf_counter()
    jd_emb = get_jd_embedding(model, JD_TEXT)
    elapsed = time.perf_counter() - t0
    return jd_emb, elapsed


def bench_similarity(embeddings: np.ndarray, jd_emb: np.ndarray) -> tuple[np.ndarray, float]:
    """Benchmark cosine similarity computation (matrix-vector dot product)."""
    t0 = time.perf_counter()
    scores = embeddings @ jd_emb   # (N,)
    elapsed = time.perf_counter() - t0
    return scores, elapsed


def bench_sort(scores: np.ndarray, candidate_ids: np.ndarray) -> tuple[list, float]:
    """Benchmark building and sorting the final result list."""
    t0 = time.perf_counter()
    idx_sorted = np.argsort(-scores)   # descending
    top100_ids    = candidate_ids[idx_sorted[:100]].tolist()
    top100_scores = (scores[idx_sorted[:100]] * 100).tolist()
    results = [
        {"rank": i + 1, "candidate_id": cid, "embedding_score": round(s, 4)}
        for i, (cid, s) in enumerate(zip(top100_ids, top100_scores))
    ]
    elapsed = time.perf_counter() - t0
    return results, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    print(_sep("="))
    print("  Redrob — Embedding Pipeline Benchmark")
    print(_sep("="))
    print()

    # ── Check embeddings exist ────────────────────────────────────────────────
    emb_path = _PROJECT_ROOT / "data" / "candidate_embeddings.npy"
    if not emb_path.exists():
        print("ERROR: Precomputed embeddings not found.")
        print(f"  Expected: {emb_path}")
        print()
        print("  Run this first:")
        print("      python scripts/generate_embeddings.py")
        print()
        return 1

    t_start = time.perf_counter()

    # ── 1. Load embeddings ────────────────────────────────────────────────────
    print("[ 1 / 5 ]  Loading precomputed embeddings...")
    embeddings, candidate_ids, t_load = bench_load()
    n, dim = embeddings.shape
    meta = get_metadata()
    print(f"           Done in {_fmt(t_load)}")
    print()

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print("[ 2 / 5 ]  Loading MiniLM model (local cache)...")
    model, t_model = bench_model_load()
    print(f"           Done in {_fmt(t_model)}")
    print()

    # ── 3. Encode JD ─────────────────────────────────────────────────────────
    print("[ 3 / 5 ]  Encoding Job Description...")
    jd_emb, t_jd = bench_jd_encode(model)
    print(f"           Done in {_fmt(t_jd)}")
    print()

    # ── 4. Cosine similarity ──────────────────────────────────────────────────
    print(f"[ 4 / 5 ]  Computing cosine similarity ({n:,} candidates × {dim} dims)...")
    scores, t_sim = bench_similarity(embeddings, jd_emb)
    print(f"           Done in {_fmt(t_sim)}")
    print()

    # ── 5. Sort + build top-100 ───────────────────────────────────────────────
    print("[ 5 / 5 ]  Sorting and building top-100...")
    results, t_sort = bench_sort(scores, candidate_ids)
    print(f"           Done in {_fmt(t_sort)}")
    print()

    # ── Validate results ──────────────────────────────────────────────────────
    val = validate_embeddings()

    # ── Summary table ─────────────────────────────────────────────────────────
    t_total_semantic = t_load + t_jd + t_sim + t_sort
    t_total          = time.perf_counter() - t_start

    print(_sep())
    print(f"  Embeddings loaded    : {n:,}")
    print(f"  Embedding dimension  : {dim}")
    print(f"  Model                : {meta.get('model_name', 'all-MiniLM-L6-v2')}")
    print(f"  Generated            : {meta.get('timestamp_utc', 'unknown')}")
    print(_sep())
    print(f"  Load time (npy)      : {_fmt(t_load)}")
    print(f"  Model load           : {_fmt(t_model)}")
    print(f"  JD encoding          : {_fmt(t_jd)}")
    print(f"  Similarity ({n:,}) : {_fmt(t_sim)}")
    print(f"  Sort + build top-100 : {_fmt(t_sort)}")
    print(_sep())
    print(f"  Semantic total       : {_fmt(t_total_semantic)}")
    print(f"  Wall time (incl. model load): {_fmt(t_total)}")
    print(_sep())
    print()
    print(f"  Normalized embeddings: {val['normalized']}")
    print()

    # ── Top-10 preview ────────────────────────────────────────────────────────
    print("  Top-10 by semantic similarity:")
    print(f"  {'Rank':<5}  {'Candidate ID':<16}  {'Embed Score':>11}")
    print("  " + "-" * 38)
    for r in results[:10]:
        print(f"  {r['rank']:<5}  {r['candidate_id']:<16}  {r['embedding_score']:>11.4f}")
    print()

    # ── Comparison ────────────────────────────────────────────────────────────
    old_time_min = 1000
    old_time_max = 2000
    speedup_min  = old_time_min / max(t_total_semantic, 0.001)
    speedup_max  = old_time_max / max(t_total_semantic, 0.001)

    print(_sep())
    print(f"  Previous runtime (live encoding) : {old_time_min}–{old_time_max} sec")
    print(f"  New runtime (precomputed)        : {t_total_semantic:.1f} sec")
    print(f"  Speedup                          : {speedup_min:.0f}x – {speedup_max:.0f}x faster")
    print(_sep())
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

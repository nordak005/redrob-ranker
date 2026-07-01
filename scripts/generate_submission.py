#!/usr/bin/env python3
"""
scripts/generate_submission.py
-------------------------------
Single-command final submission generator for the Redrob AI Engineer Ranker.

Usage:
    python scripts/generate_submission.py

Produces:
    outputs/final_submission.csv  (100 rows, hybrid-scored, natural reasoning)

Runtime target: < 5 minutes on CPU (16 GB RAM).

Pipeline:
    1. Load candidates from data/raw/candidates.jsonl.gz
    2. Compute feature scores
    3. Load precomputed embeddings & compute semantic similarity
    4. Compute hybrid_score = 0.85 * feature_score + 0.15 * embedding_score
    5. Take top 100, generate reasoning, write output
    6. Validate the output against submission rules
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import sys
import time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import build_final_score
from src.reasoning import build_reasoning
from src.semantic_search import get_score_lookup

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
CANDIDATES_PATH   = _PROJECT_ROOT / "data" / "raw" / "candidates.jsonl.gz"
OUTPUT_PATH       = _PROJECT_ROOT / "outputs" / "final_submission.csv"
TOP_N             = 100

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_candidates_gz(path: Path) -> dict[str, dict]:
    """Load all candidates from JSONL.gz into a dict keyed by candidate_id."""
    logger.info("Loading candidates from %s ...", path)
    t0 = time.perf_counter()
    candidates: dict[str, dict] = {}
    with gzip.open(str(path), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = record.get("candidate_id", "")
            if cid:
                candidates[cid] = record
    elapsed = time.perf_counter() - t0
    logger.info("Loaded %d candidates in %.2f s.", len(candidates), elapsed)
    return candidates

# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_final_submission(records: list[dict], output_path: Path) -> None:
    """Write the final_submission.csv with required columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["candidate_id", "rank", "score", "reasoning"]
    with open(str(output_path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "candidate_id": r["candidate_id"],
                "rank":         r["rank"],
                "score":        f"{r['score']:.6f}",
                "reasoning":    r["reasoning"],
            })
    logger.info("Wrote %d rows to %s", len(records), output_path)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_submission(records: list[dict]) -> bool:
    """
    Verify the submission meets spec requirements.
    Returns True if valid, raises AssertionError with details if not.
    """
    assert len(records) == TOP_N, f"Expected {TOP_N} rows, got {len(records)}"

    ranks = [r["rank"] for r in records]
    assert sorted(ranks) == list(range(1, TOP_N + 1)), "Ranks must be 1–100 exactly once"

    cids = [r["candidate_id"] for r in records]
    assert len(set(cids)) == TOP_N, "candidate_id values must be unique"

    scores = [float(r["score"]) for r in records]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Score not monotonically decreasing at rank {i+1}: "
            f"{scores[i]:.6f} -> {scores[i+1]:.6f}"
        )

    for r in records:
        assert r["reasoning"].strip(), "reasoning must not be empty"

    logger.info("✅ Validation PASSED: 100 rows, ranks 1–100, scores decreasing, all IDs unique.")
    return True

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("=" * 60)
    logger.info("Redrob — Final Submission Generator")
    logger.info("=" * 60)

    # ── Startup Validation ───────────────────────────────────────────────────
    required_files = [
        _PROJECT_ROOT / "data" / "candidate_embeddings.npy",
        _PROJECT_ROOT / "data" / "candidate_ids.npy",
        _PROJECT_ROOT / "data" / "embedding_metadata.json",
    ]
    missing = [p.name for p in required_files if not p.exists()]
    if missing:
        logger.error("ERROR: The embedding cache is missing.")
        logger.error("Missing files: %s", ", ".join([f"data/{f}" for f in missing]))
        logger.error("Run: python scripts/generate_embeddings.py")
        logger.error("=" * 60)
        return 1

    t_start = time.perf_counter()

    # ── Step 1: Load full candidate profiles ───────────────────────────────
    if not CANDIDATES_PATH.exists():
        logger.error("Candidates file not found at %s", CANDIDATES_PATH)
        return 1

    candidate_map = load_candidates_gz(CANDIDATES_PATH)
    if not candidate_map:
        logger.error("No candidates loaded.")
        return 1

    # ── Step 2: Semantic search via precomputed embeddings ─────────────────
    logger.info("Fetching semantic scores using precomputed embeddings...")
    semantic_scores = get_score_lookup()  # returns dict[candidate_id, embedding_score]

    # ── Step 3: Compute feature scores and hybrid scores ───────────────────
    logger.info("Computing feature scores and hybrid rankings...")
    scored_candidates = []
    
    t0 = time.perf_counter()
    for cid, candidate in candidate_map.items():
        # Feature score
        try:
            scores = build_final_score(candidate)
            feature_score = scores.get("final_score", 0.0) * 100.0
        except Exception as exc:
            logger.warning("Score computation failed for %s: %s", cid, exc)
            feature_score = 0.0
            scores = {}

        # Semantic score
        embedding_score = semantic_scores.get(cid, 0.0)
        
        # Hybrid score
        hybrid_score = 0.85 * feature_score + 0.15 * embedding_score

        scored_candidates.append({
            "candidate_id": cid,
            "candidate_obj": candidate,
            "feature_scores_dict": scores,
            "hybrid_score": hybrid_score
        })
        
    elapsed = time.perf_counter() - t0
    logger.info("Computed hybrid scores for %d candidates in %.2f s", len(scored_candidates), elapsed)

    # ── Step 4: Sort and take top 100 ──────────────────────────────────────
    scored_candidates.sort(key=lambda x: (-x["hybrid_score"], x["candidate_id"]))
    top_candidates = scored_candidates[:TOP_N]

    # ── Step 5: Build final submission records ─────────────────────────────
    logger.info("Generating reasoning for top-%d candidates...", TOP_N)
    submission_records = []

    for rank_idx, cand_info in enumerate(top_candidates, start=1):
        cid = cand_info["candidate_id"]
        hybrid_score = cand_info["hybrid_score"]
        candidate = cand_info["candidate_obj"]
        scores = cand_info["feature_scores_dict"]

        # Generate natural-language reasoning
        reasoning = build_reasoning(candidate, scores)

        submission_records.append({
            "candidate_id": cid,
            "rank":         rank_idx,
            "score":        hybrid_score,
            "reasoning":    reasoning,
        })

    # ── Step 6: Write CSV ──────────────────────────────────────────────────
    write_final_submission(submission_records, OUTPUT_PATH)

    # ── Step 7: Validate ───────────────────────────────────────────────────
    try:
        validate_submission(submission_records)
    except AssertionError as e:
        logger.error("❌ Validation FAILED: %s", e)
        return 1

    # ── Summary ────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("DONE. Total time: %.2f s", total_time)
    logger.info("Final submission: %s", OUTPUT_PATH.resolve())
    logger.info("=" * 60)

    # Print sample output
    logger.info("\nSample output (first 5 rows):")
    for r in submission_records[:5]:
        logger.info("  Rank %3d | %s | %.4f", r["rank"], r["candidate_id"], r["score"])
        logger.info("           %s", r["reasoning"])

    return 0


if __name__ == "__main__":
    sys.exit(main())

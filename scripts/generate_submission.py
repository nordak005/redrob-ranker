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
    2. Load hybrid scores from outputs/hybrid_rankings.csv
    3. Merge hybrid_score into feature-ranked candidates (top-100 by hybrid_rank)
    4. Generate natural-language reasoning via src.reasoning.build_reasoning()
    5. Write outputs/final_submission.csv
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

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
CANDIDATES_PATH   = _PROJECT_ROOT / "data" / "raw" / "candidates.jsonl.gz"
HYBRID_CSV_PATH   = _PROJECT_ROOT / "outputs" / "hybrid_rankings.csv"
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


def load_hybrid_top100(path: Path) -> list[dict]:
    """
    Load the top-100 candidates by hybrid_rank from hybrid_rankings.csv.
    Returns a list of dicts sorted by hybrid_rank ascending.
    """
    logger.info("Loading hybrid scores from %s ...", path)
    rows = []
    with open(str(path), "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hybrid_rank = int(row.get("hybrid_rank", 999999))
            if hybrid_rank <= TOP_N:
                rows.append(row)
    rows.sort(key=lambda r: int(r["hybrid_rank"]))
    logger.info("Loaded %d hybrid-ranked candidates.", len(rows))
    return rows


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

    t_start = time.perf_counter()

    # ── Step 1: Load hybrid top-100 ────────────────────────────────────────
    if not HYBRID_CSV_PATH.exists():
        logger.error("hybrid_rankings.csv not found at %s", HYBRID_CSV_PATH)
        logger.error("Run the hybrid ranking pipeline first.")
        return 1

    hybrid_rows = load_hybrid_top100(HYBRID_CSV_PATH)

    if len(hybrid_rows) < TOP_N:
        logger.error(
            "Only %d hybrid-ranked candidates found (need %d).",
            len(hybrid_rows), TOP_N
        )
        return 1

    # ── Step 2: Load full candidate profiles ───────────────────────────────
    if not CANDIDATES_PATH.exists():
        logger.error("Candidates file not found at %s", CANDIDATES_PATH)
        return 1

    # We only need profiles for the top-100 hybrid candidates
    top_ids = {r["candidate_id"] for r in hybrid_rows}
    logger.info("Loading candidate profiles for top %d hybrid candidates...", len(top_ids))
    t0 = time.perf_counter()
    candidate_map: dict[str, dict] = {}
    with gzip.open(str(CANDIDATES_PATH), "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cid = record.get("candidate_id", "")
            if cid in top_ids:
                candidate_map[cid] = record
                if len(candidate_map) == len(top_ids):
                    break  # Early exit once all top candidates found
    elapsed = time.perf_counter() - t0
    logger.info(
        "Loaded %d candidate profiles in %.2f s.",
        len(candidate_map), elapsed
    )

    # ── Step 3: Build final submission records ─────────────────────────────
    logger.info("Generating reasoning for top-%d candidates...", TOP_N)
    submission_records = []

    for hybrid_row in hybrid_rows:
        cid = hybrid_row["candidate_id"]
        hybrid_rank = int(hybrid_row["hybrid_rank"])
        hybrid_score = float(hybrid_row["hybrid_score"])

        candidate = candidate_map.get(cid)
        if candidate is None:
            logger.warning("Profile not found for %s — using empty profile.", cid)
            candidate = {"candidate_id": cid}

        # Compute feature scores for the reasoning generator
        try:
            scores = build_final_score(candidate)
        except Exception as exc:
            logger.warning("Score computation failed for %s: %s", cid, exc)
            scores = {}

        # Generate natural-language reasoning
        reasoning = build_reasoning(candidate, scores)

        submission_records.append({
            "candidate_id": cid,
            "rank":         hybrid_rank,
            "score":        hybrid_score,
            "reasoning":    reasoning,
        })

    # ── Step 4: Write CSV ──────────────────────────────────────────────────
    write_final_submission(submission_records, OUTPUT_PATH)

    # ── Step 5: Validate ───────────────────────────────────────────────────
    try:
        validate_submission(submission_records)
    except AssertionError as e:
        logger.error("❌ Validation FAILED: %s", e)
        return 1

    # ── Summary ────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("DONE. Total time: %.2f s (%.2f min)", total_time, total_time / 60)
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

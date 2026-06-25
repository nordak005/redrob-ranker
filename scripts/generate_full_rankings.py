#!/usr/bin/env python3
"""
scripts/generate_full_rankings.py
----------------------------------
Score every candidate in the dataset and write a full 100 000-row
debug-style CSV (same schema as debug_scores.csv) ranked 1 → N.

Output:  outputs/full_rankings.csv

Usage:
    python scripts/generate_full_rankings.py
    python scripts/generate_full_rankings.py --input data/raw/candidates.jsonl.gz --output outputs/full_rankings.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import sys
import time
from pathlib import Path

# ── project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import build_final_score, build_reasoning  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_HEADERS = [
    "candidate_id",
    "rank",
    "final_score",
    "semantic_score",
    "title_score",
    "career_score",
    "assessment_score",
    "skill_trust_score",
    "retrieval_score",
    "behavioral_multiplier",
    "reasoning",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate full 100 k rankings CSV.")
    p.add_argument(
        "--input",
        default="data/raw/candidates.jsonl.gz",
        help="Path to candidates JSONL or JSONL.gz (default: data/raw/candidates.jsonl.gz)",
    )
    p.add_argument(
        "--output",
        default="outputs/full_rankings.csv",
        help="Output CSV path (default: outputs/full_rankings.csv)",
    )
    return p.parse_args()


def _iter_candidates(filepath: str):
    """Yield candidate dicts from a .jsonl or .jsonl.gz file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    opener = gzip.open if filepath.endswith(".gz") else open
    mode = "rt"
    with opener(filepath, mode, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    args = _parse_args()
    input_path = str(_PROJECT_ROOT / args.input)
    output_path = _PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Full Rankings Generator")
    logger.info("Input :  %s", input_path)
    logger.info("Output:  %s", output_path)
    logger.info("=" * 60)

    # ── Score all candidates ──────────────────────────────────────────────────
    scored: list[dict] = []
    t0 = time.perf_counter()

    for i, cand in enumerate(_iter_candidates(input_path), start=1):
        cid = cand.get("candidate_id", f"UNKNOWN_{i}")
        try:
            scores = build_final_score(cand)
            reasoning = build_reasoning(cand, scores)
        except Exception as exc:
            logger.warning("Scoring failed for %s: %s", cid, exc)
            scores = {
                "title_score": 0.0,
                "career_score": 0.0,
                "assessment_score": 0.0,
                "skill_trust_score": 0.0,
                "retrieval_score": 0.0,
                "semantic_score": 0.0,
                "behavioral_multiplier": 1.0,
                "final_score": 0.0,
            }
            reasoning = "Scoring error – defaulted to 0.0"

        scored.append({"candidate_id": cid, **scores, "reasoning": reasoning})

        if i % 10_000 == 0:
            elapsed = time.perf_counter() - t0
            logger.info("  Processed %6d candidates  (%.1f s elapsed)", i, elapsed)

    score_time = time.perf_counter() - t0
    logger.info("Scored %d candidates in %.2f s.", len(scored), score_time)

    # ── Sort: descending final_score, tie-break by candidate_id asc ──────────
    logger.info("Sorting...")
    scored.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

    # ── Assign ranks ─────────────────────────────────────────────────────────
    for rank_idx, row in enumerate(scored, start=1):
        row["rank"] = rank_idx

    # ── Write CSV ─────────────────────────────────────────────────────────────
    logger.info("Writing CSV to %s ...", output_path)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_HEADERS, extrasaction="ignore")
        writer.writeheader()
        for row in scored:
            writer.writerow({
                "candidate_id":        row["candidate_id"],
                "rank":                row["rank"],
                "final_score":         f"{row['final_score']:.6f}",
                "semantic_score":      f"{row.get('semantic_score', 0):.4f}",
                "title_score":         f"{row.get('title_score', 0):.4f}",
                "career_score":        f"{row.get('career_score', 0):.4f}",
                "assessment_score":    f"{row.get('assessment_score', 0):.4f}",
                "skill_trust_score":   f"{row.get('skill_trust_score', 0):.4f}",
                "retrieval_score":     f"{row.get('retrieval_score', 0):.4f}",
                "behavioral_multiplier": f"{row.get('behavioral_multiplier', 0):.4f}",
                "reasoning":           row["reasoning"],
            })

    total_time = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("DONE  –  %d rows written in %.2f s", len(scored), total_time)
    logger.info("Output: %s", output_path.resolve())
    if scored:
        logger.info(
            "Score range: rank-1 = %.6f  |  rank-%d = %.6f",
            scored[0]["final_score"],
            scored[-1]["rank"],
            scored[-1]["final_score"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

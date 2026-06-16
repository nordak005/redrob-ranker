#!/usr/bin/env python3
"""
scripts/run_ranker.py
---------------------
Production entry point for the Redrob AI Engineer Ranker.

Usage:
    python scripts/run_ranker.py \\
        --input  data/raw/candidates.jsonl.gz \\
        --output outputs/submission.csv \\
        [--top-n 100] \\
        [--validate] \\
        [--debug-scores outputs/debug_scores.csv]

Arguments:
    --input          Path to candidates JSONL or JSONL.gz file.
    --output         Path for the submission CSV (must end in .csv).
    --top-n          Number of candidates to include (default: 100).
    --validate       If set, runs validate_submission.py on the output.
    --debug-scores   Optional CSV with all component scores (for analysis).
    --log-level      Logging level: DEBUG | INFO | WARNING (default: INFO).

Runtime: ~20–40 s for 100 k candidates on a standard CPU.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when the script
# is run from any working directory (e.g. project root or scripts/).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ranker import rank_candidates
from src.utils import Timer, load_candidates

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_ranker",
        description="Rank 100 k candidates for Senior AI Engineer fit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to candidates JSONL or JSONL.gz file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output CSV path (e.g. outputs/submission.csv).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        metavar="N",
        help="Number of top candidates to include (default: 100).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_submission.py on the output after writing.",
    )
    parser.add_argument(
        "--debug-scores",
        metavar="PATH",
        default=None,
        help="Write all component scores to this CSV for analysis.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

_SUBMISSION_HEADERS = ["candidate_id", "rank", "score", "reasoning"]

_DEBUG_HEADERS = [
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


def _write_submission_csv(ranked: list[dict], output_path: Path) -> None:
    """
    Write the top-N ranked candidates to a submission-format CSV.

    Columns: candidate_id, rank, score, reasoning

    Rules enforced:
      - Exactly 100 rows (or top_n rows if top_n != 100).
      - Scores are non-increasing by rank.
      - Scores are rounded to 6 decimal places (floats allowed per spec).
      - reasoning is a concise single-line string.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUBMISSION_HEADERS, extrasaction="ignore")
        writer.writeheader()
        for r in ranked:
            writer.writerow(
                {
                    "candidate_id": r["candidate_id"],
                    "rank": r["rank"],
                    "score": f"{r['final_score']:.6f}",
                    "reasoning": r["reasoning"],
                }
            )

    logger.info("Submission CSV written to: %s", output_path)


def _write_debug_csv(ranked: list[dict], debug_path: Path) -> None:
    """Write all component scores for the top-N candidates (analysis only)."""
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    with open(debug_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DEBUG_HEADERS, extrasaction="ignore")
        writer.writeheader()
        for r in ranked:
            writer.writerow(
                {
                    "candidate_id": r.get("candidate_id", ""),
                    "rank": r.get("rank", ""),
                    "final_score": f"{r.get('final_score', 0):.6f}",
                    "semantic_score": f"{r.get('semantic_score', 0):.4f}",
                    "title_score": f"{r.get('title_score', 0):.4f}",
                    "career_score": f"{r.get('career_score', 0):.4f}",
                    "assessment_score": f"{r.get('assessment_score', 0):.4f}",
                    "skill_trust_score": f"{r.get('skill_trust_score', 0):.4f}",
                    "retrieval_score": f"{r.get('retrieval_score', 0):.4f}",
                    "behavioral_multiplier": f"{r.get('behavioral_multiplier', 0):.4f}",
                    "reasoning": r.get("reasoning", ""),
                }
            )

    logger.info("Debug scores CSV written to: %s", debug_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_output(csv_path: Path) -> bool:
    """
    Run the competition's validate_submission.py against the output.

    Returns True if valid, False if errors found.
    """
    validator_path = _PROJECT_ROOT / "validate_submission.py"
    if not validator_path.exists():
        logger.warning("validate_submission.py not found at %s; skipping.", validator_path)
        return True

    import subprocess

    result = subprocess.run(
        [sys.executable, str(validator_path), str(csv_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        logger.info("Validation PASSED: %s", result.stdout.strip())
        return True
    else:
        logger.error("Validation FAILED:\n%s", result.stdout + result.stderr)
        return False


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _preflight(args: argparse.Namespace) -> None:
    """Validate paths and arguments before expensive operations."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_path.suffix.lower() in {".jsonl", ".gz", ".json"}:
        raise ValueError(
            f"Unexpected input extension '{input_path.suffix}'. "
            "Expected .jsonl, .jsonl.gz, or .json"
        )

    if output_path.suffix.lower() != ".csv":
        raise ValueError(f"Output file must have .csv extension, got: {output_path}")

    if args.top_n < 1 or args.top_n > 10_000:
        raise ValueError(f"--top-n must be between 1 and 10000, got: {args.top_n}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """
    Full ranking pipeline:
      1. Load candidates from JSONL/JSONL.gz
      2. Score + rank all candidates
      3. Write submission CSV
      4. Optionally write debug scores
      5. Optionally validate output

    Returns:
        0 on success, 1 on error.
    """
    args = _parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("Redrob AI Engineer Ranker – Phase 3")
    logger.info("=" * 60)
    logger.info("Input:    %s", args.input)
    logger.info("Output:   %s", args.output)
    logger.info("Top-N:    %d", args.top_n)

    # Pre-flight validation
    try:
        _preflight(args)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Pre-flight check failed: %s", e)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)

    # ---- Step 1: Load candidates ----
    logger.info("Loading candidates from %s ...", input_path)
    with Timer() as load_timer:
        candidates = load_candidates(str(input_path))

    n_loaded = len(candidates)
    logger.info(
        "Loaded %d candidates in %.2f s.",
        n_loaded,
        load_timer.interval,
    )

    if n_loaded == 0:
        logger.error("No candidates loaded. Check input file.")
        return 1

    # ---- Step 2: Score and rank ----
    logger.info("Ranking %d candidates (top-%d)...", n_loaded, args.top_n)
    t0 = time.perf_counter()

    ranked = rank_candidates(
        candidates=iter(candidates),
        top_n=args.top_n,
        log_interval=10_000,
    )

    rank_time = time.perf_counter() - t0
    logger.info(
        "Ranking complete in %.2f s (%.0f candidates/s).",
        rank_time,
        n_loaded / rank_time if rank_time > 0 else 0,
    )

    if not ranked:
        logger.error("Ranking produced 0 results. Aborting.")
        return 1

    # ---- Step 3: Print top-10 summary ----
    logger.info("\nTop-10 Candidates:")
    logger.info(
        "  %-4s  %-14s  %-6s  %-6s  %-6s  %-6s  %-5s  %s",
        "Rank", "candidate_id", "Final", "Sem", "Title", "Career", "Behav", "Summary",
    )
    for r in ranked[:10]:
        profile = next(
            (c.get("profile", {}) for c in candidates
             if c.get("candidate_id") == r["candidate_id"]),
            {},
        )
        title = profile.get("current_title", "?")[:25]
        logger.info(
            "  %-4d  %-14s  %-6.4f  %-6.2f  %-6.2f  %-6.2f  %-5.3f  %s",
            r["rank"],
            r["candidate_id"],
            r["final_score"],
            r.get("semantic_score", 0),
            r.get("title_score", 0),
            r.get("career_score", 0),
            r.get("behavioral_multiplier", 1),
            title,
        )

    # ---- Step 4: Write submission CSV ----
    _write_submission_csv(ranked, output_path)

    # ---- Step 5: Write debug CSV (optional) ----
    if args.debug_scores:
        _write_debug_csv(ranked, Path(args.debug_scores))

    # ---- Step 6: Validate (optional) ----
    if args.validate:
        logger.info("Running submission validator...")
        valid = _validate_output(output_path)
        if not valid:
            logger.error("Submission validation failed. Fix errors before submitting.")
            return 1

    # ---- Summary ----
    total_time = load_timer.interval + rank_time
    logger.info("=" * 60)
    logger.info("DONE. Total time: %.2f s", total_time)
    logger.info("Submission saved to: %s", output_path.resolve())

    if ranked:
        best = ranked[0]
        worst = ranked[-1]
        logger.info(
            "Score range: rank-1 = %.4f | rank-%d = %.4f",
            best["final_score"],
            worst["rank"],
            worst["final_score"],
        )
    logger.info("=" * 60)

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())

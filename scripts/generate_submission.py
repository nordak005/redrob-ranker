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

Embedding cache:
    If data/candidate_embeddings.npy or data/candidate_ids.npy are missing,
    the script presents an interactive menu:
      [1] Download precomputed embeddings from the GitHub Release (recommended)
      [2] Generate embeddings locally via generate_embeddings.py
      [3] Exit

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
import urllib.request
import zipfile
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

# ── Embedding download ─────────────────────────────────────────────────────
# URL of the official precomputed embedding package on the GitHub Release page.
# Update <OWNER>/<REPO> if the repository is ever renamed or forked.
EMBEDDING_DOWNLOAD_URL = (
    "https://github.com/nordak005/redrob-ranker/releases/download/v1.0/embedding.zip.zip"
)

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
# Embedding cache helpers
# ---------------------------------------------------------------------------

# Files that must be present for the pipeline to run.
_REQUIRED_EMBEDDING_FILES = [
    "candidate_embeddings.npy",
    "candidate_ids.npy",
    "embedding_metadata.json",
]


def _show_menu(missing: list[Path]) -> None:
    """Print the embedding-recovery menu to stdout."""
    sep = "=" * 60
    print(sep)
    print("Embedding cache not found.\n")
    print("Required files:")
    for p in missing:
        print(f"  \u2022 data/{p.name}")
    print()
    print("Choose one of the following options:\n")
    print("[1] Download precomputed embeddings (Recommended \u26a1)")
    print("    \u2714 Fastest setup (typically under a minute)")
    print("    \u2714 Downloads the official embedding package from the project\u2019s GitHub Release")
    print("    \u2714 Automatically extracts the files into the correct data/ directory")
    print("    \u2714 Recommended for most users\n")
    print("[2] Generate embeddings locally")
    print("    \u2714 No download required")
    print("    \u2714 Uses the existing embedding generation pipeline")
    print("    \u2714 May take several minutes depending on hardware and dataset size\n")
    print("[3] Exit\n")


def _download_embeddings() -> int:
    """
    Download the precomputed embedding ZIP from EMBEDDING_DOWNLOAD_URL,
    extract it into data/, verify the required files, and remove the ZIP.

    Returns 0 on success, 1 on any failure.
    """
    data_dir = _PROJECT_ROOT / "data"
    zip_path = data_dir / "embeddings.zip"
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading embeddings from:")
    logger.info("  %s", EMBEDDING_DOWNLOAD_URL)

    # ── Progress callback ──────────────────────────────────────────────────
    _last_pct: list[int] = [-1]  # mutable cell for closure

    def _progress(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        pct = int(downloaded * 100 / total_size)
        if pct != _last_pct[0] and pct % 5 == 0:
            bar_filled = pct // 5
            bar = "\u2588" * bar_filled + "\u2591" * (20 - bar_filled)
            mb_done = downloaded / 1_048_576
            mb_total = total_size / 1_048_576
            print(
                f"\r  [{bar}] {pct:3d}%  {mb_done:.1f} / {mb_total:.1f} MB",
                end="",
                flush=True,
            )
            _last_pct[0] = pct

    # ── Download ───────────────────────────────────────────────────────────
    try:
        urllib.request.urlretrieve(EMBEDDING_DOWNLOAD_URL, str(zip_path), _progress)
        print()  # newline after progress bar
    except Exception as exc:
        print()  # newline after partial progress bar
        logger.error("Download failed: %s", exc)
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
        return 1

    logger.info("Download complete. Extracting...")

    # ── Extract ────────────────────────────────────────────────────────────
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(data_dir))
    except zipfile.BadZipFile as exc:
        logger.error("Extraction failed — ZIP appears corrupt: %s", exc)
        zip_path.unlink(missing_ok=True)
        return 1
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        zip_path.unlink(missing_ok=True)
        return 1

    # ── Verify ─────────────────────────────────────────────────────────────
    still_missing = [
        name for name in _REQUIRED_EMBEDDING_FILES
        if not (data_dir / name).exists()
    ]
    if still_missing:
        logger.error(
            "Extraction succeeded but required files are still missing: %s",
            ", ".join(f"data/{f}" for f in still_missing),
        )
        zip_path.unlink(missing_ok=True)
        return 1

    # ── Cleanup ────────────────────────────────────────────────────────────
    zip_path.unlink(missing_ok=True)
    logger.info("\u2713 Embedding cache downloaded successfully.")
    return 0


def _generate_embeddings_locally() -> int:
    """
    Run the existing offline embedding pipeline from generate_embeddings.py
    in-process.  No code is duplicated — only the module's main() is called.

    Returns 0 on success, 1 on failure.
    """
    logger.info("Launching embedding generation pipeline...")
    logger.info("=" * 60)

    # Prefer package import; fall back to direct file load (handles cases
    # where scripts/ is not on sys.path as a package).
    try:
        import scripts.generate_embeddings as _gen_emb  # type: ignore[import]
    except ImportError:
        import importlib
        import importlib.util
        _scripts_dir = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location(
            "generate_embeddings",
            str(_scripts_dir / "generate_embeddings.py"),
        )
        _gen_emb = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(_gen_emb)  # type: ignore[union-attr]

    try:
        rc = _gen_emb.main()
    except Exception as exc:
        logger.error("Embedding generation failed with an unexpected error: %s", exc)
        return 1

    if rc != 0:
        logger.error(
            "Embedding generation exited with code %d. "
            "Cannot continue without the embedding cache.",
            rc,
        )
        return 1

    return 0


def _handle_missing_embeddings(missing: list[Path]) -> int:
    """
    Present the interactive three-option recovery menu to the user.

    Loops until the user makes a valid choice (1, 2, or 3).

    Returns
    -------
    0  — embeddings are now in place; caller should continue.
    1  — user chose to exit, or recovery failed.
    """
    sep = "=" * 60

    while True:
        _show_menu(missing)

        try:
            choice = input("Enter your choice (1/2/3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Submission cancelled.")
            print("No embedding cache available.")
            print(sep)
            return 1

        print(sep)

        if choice == "1":
            rc = _download_embeddings()
            if rc != 0:
                logger.error(
                    "Download failed. Please retry or choose option [2] to "
                    "generate embeddings locally."
                )
                # Return to menu instead of exiting outright.
                print()
                continue
            return 0

        elif choice == "2":
            rc = _generate_embeddings_locally()
            if rc != 0:
                # Error already logged inside helper; return to menu.
                print()
                continue
            return 0

        elif choice == "3":
            print("Submission cancelled.")
            print("No embedding cache available.")
            return 1

        else:
            logger.warning("Invalid choice %r — please enter 1, 2, or 3.", choice)
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger.info("=" * 60)
    logger.info("Redrob — Final Submission Generator")
    logger.info("=" * 60)

    # ── Startup Validation ───────────────────────────────────────────────────
    embedding_files = [
        _PROJECT_ROOT / "data" / "candidate_embeddings.npy",
        _PROJECT_ROOT / "data" / "candidate_ids.npy",
        _PROJECT_ROOT / "data" / "embedding_metadata.json",
    ]
    missing = [p for p in embedding_files if not p.exists()]
    if missing:
        rc = _handle_missing_embeddings(missing)
        if rc != 0:
            return rc

        # Verify that generation actually produced all required files.
        still_missing = [p for p in embedding_files if not p.exists()]
        if still_missing:
            logger.error(
                "Embedding generation finished but files are still missing: %s",
                ", ".join(f"data/{p.name}" for p in still_missing),
            )
            return 1

        logger.info("Embedding cache verified. Continuing with submission generation...")
        logger.info("=" * 60)

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

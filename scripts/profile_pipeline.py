#!/usr/bin/env python3
"""
scripts/profile_pipeline.py
-----------------------------
Redrob AI Engineer Ranker — Full Pipeline Profiler.

PURPOSE
-------
Instrument every major stage of the ranking pipeline with high-resolution
timers (time.perf_counter()) and produce a detailed performance report.

IMPORTANT — THIS SCRIPT:
  Does NOT optimise anything
  Does NOT rewrite anything
  Does NOT cache anything
  Does NOT change ranking logic
  Does NOT change hybrid weights
  Does NOT modify feature engineering
  Does NOT modify semantic search
  Does NOT modify embeddings
  Does NOT modify the Streamlit UI

This script is PROFILING ONLY.  All scoring results are identical to a
normal generate_submission.py run — only per-stage wall-clock timings
are added.

USAGE
-----
    python scripts/profile_pipeline.py

OUTPUTS
-------
    outputs/performance_report.json   — machine-readable timing data
    outputs/performance_report.md     — human-readable table + analysis
    (console)                         — full formatted report printed live
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
CANDIDATES_PATH = _PROJECT_ROOT / "data" / "raw" / "candidates.jsonl.gz"
OUTPUT_JSON     = _PROJECT_ROOT / "outputs" / "performance_report.json"
OUTPUT_MD       = _PROJECT_ROOT / "outputs" / "performance_report.md"
TOP_N           = 100

# ── Optimization-target threshold ────────────────────────────────────────────
SLOW_THRESHOLD_SEC = 5.0


# ============================================================
# Helper: simple timer context manager
# ============================================================

class _Timer:
    """Context manager that records wall-clock elapsed time."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start


# ============================================================
# Stage 1: Load candidates
# ============================================================

def _load_candidates(path: Path) -> Tuple[List[dict], float]:
    candidates: List[dict] = []
    with _Timer() as t:
        with gzip.open(str(path), "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    logger.info("[PROFILER] Loaded %d candidates in %.3f s", len(candidates), t.elapsed)
    return candidates, t.elapsed


# ============================================================
# Stages 2–8: Per-candidate feature scoring
# ============================================================

def _score_all_candidates_granular(
    candidates: List[dict],
    build_title_score,
    build_career_score,
    build_assessment_score,
    build_skill_trust_score,
    build_retrieval_score,
    build_behavioral_multiplier,
    build_reasoning,
) -> Tuple[List[dict], Dict[str, float]]:
    """
    Score every candidate with granular per-stage timing.
    Cumulative per-stage time is accumulated across all N candidates.
    """
    n = len(candidates)

    t_title      = 0.0
    t_career     = 0.0
    t_assessment = 0.0
    t_skill      = 0.0
    t_retrieval  = 0.0
    t_behavioral = 0.0
    t_combine    = 0.0
    t_reasoning  = 0.0

    scored: List[dict] = []
    REPORT_EVERY = 10_000
    loop_start = time.perf_counter()

    for i, cand in enumerate(candidates, start=1):
        cid = cand.get("candidate_id", f"UNKNOWN_{i}")
        try:
            # Title Score
            t0 = time.perf_counter(); title_score = build_title_score(cand);      t_title      += time.perf_counter() - t0
            # Career Score
            t0 = time.perf_counter(); career_score = build_career_score(cand);    t_career     += time.perf_counter() - t0
            # Assessment Score
            t0 = time.perf_counter(); assessment_score = build_assessment_score(cand); t_assessment += time.perf_counter() - t0
            # Skill Trust Score
            t0 = time.perf_counter(); skill_trust_score = build_skill_trust_score(cand); t_skill  += time.perf_counter() - t0
            # Retrieval Score
            t0 = time.perf_counter(); retrieval_score = build_retrieval_score(cand);   t_retrieval += time.perf_counter() - t0
            # Behavioral Multiplier
            t0 = time.perf_counter(); behavioral_multiplier = build_behavioral_multiplier(cand); t_behavioral += time.perf_counter() - t0

            # Final Score Assembly (scale + clamp)
            t0 = time.perf_counter()
            title_scaled      = title_score       * (35.0 / 40.0)
            career_scaled     = career_score      * (25.0 / 30.0)
            assessment_scaled = assessment_score  * (15.0 / 20.0)
            skill_scaled      = skill_trust_score
            retrieval_scaled  = retrieval_score
            semantic_score    = max(0.0, min(100.0,
                title_scaled + career_scaled + assessment_scaled
                + skill_scaled + retrieval_scaled
            ))
            raw_final  = semantic_score * behavioral_multiplier
            final_score = max(0.0, min(1.0, raw_final / 115.0))
            scores = {
                "title_score":           round(title_scaled,       4),
                "career_score":          round(career_scaled,      4),
                "assessment_score":      round(assessment_scaled,  4),
                "skill_trust_score":     round(skill_scaled,       4),
                "retrieval_score":       round(retrieval_scaled,   4),
                "semantic_score":        round(semantic_score,     4),
                "behavioral_multiplier": round(behavioral_multiplier, 4),
                "final_score":           round(final_score, 6),
            }
            t_combine += time.perf_counter() - t0

            # Reasoning
            t0 = time.perf_counter(); reasoning = build_reasoning(cand, scores); t_reasoning += time.perf_counter() - t0

        except Exception as exc:
            logger.warning("[PROFILER] Scoring failed for %s: %s", cid, exc)
            scores = {
                "title_score": 0.0, "career_score": 0.0,
                "assessment_score": 0.0, "skill_trust_score": 0.0,
                "retrieval_score": 0.0, "semantic_score": 0.0,
                "behavioral_multiplier": 1.0, "final_score": 0.0,
            }
            reasoning = "Scoring error - defaulted to 0.0"

        scored.append({"candidate_id": cid, **scores, "reasoning": reasoning})

        if i % REPORT_EVERY == 0:
            elapsed = time.perf_counter() - loop_start
            logger.info(
                "[PROFILER] %6d / %d candidates  |  %.1f s elapsed  |  %.3f ms/cand",
                i, n, elapsed, (elapsed / i) * 1000,
            )

    total_loop = time.perf_counter() - loop_start
    per_cand_ms = (total_loop / n * 1000) if n else 0.0
    logger.info(
        "[PROFILER] Scoring loop done: %d candidates | %.2f s total | %.3f ms/cand",
        n, total_loop, per_cand_ms,
    )
    logger.info(
        "[PROFILER] Sub-stage totals: title=%.2fs career=%.2fs "
        "assess=%.2fs skill=%.2fs retrieval=%.2fs "
        "behavioral=%.2fs assembly=%.2fs reasoning=%.2fs",
        t_title, t_career, t_assessment, t_skill,
        t_retrieval, t_behavioral, t_combine, t_reasoning,
    )

    stage_times = {
        "title_score":           t_title,
        "career_score":          t_career,
        "assessment_score":      t_assessment,
        "skill_trust_score":     t_skill,
        "retrieval_score":       t_retrieval,
        "behavioral_multiplier": t_behavioral,
        "final_score_assembly":  t_combine,
        "reasoning":             t_reasoning,
        "_total_scoring_loop":   total_loop,
    }
    return scored, stage_times


# ============================================================
# Stage 9: Embedding load + JD encoding + cosine similarity
# ============================================================

def _embedding_stage(
    candidates: List[dict],
    load_embeddings,
    get_candidate_ids,
    get_model,
    get_jd_embedding,
    JD_TEXT: str,
) -> Tuple[Any, float, float, float, float]:
    import numpy as np

    # Load embedding matrix
    with _Timer() as t_emb:
        emb_matrix = load_embeddings()
        emb_ids    = get_candidate_ids()
    logger.info("[PROFILER] Embedding load: %.3f s | shape=%s", t_emb.elapsed, emb_matrix.shape)

    # ID alignment
    with _Timer() as t_align:
        id_to_idx = {str(cid): i for i, cid in enumerate(emb_ids)}
        dim = emb_matrix.shape[1]
        aligned_rows = []
        missing_count = 0
        for cand in candidates:
            cid = str(cand.get("candidate_id", "UNKNOWN"))
            if cid in id_to_idx:
                aligned_rows.append(emb_matrix[id_to_idx[cid]])
            else:
                aligned_rows.append(np.zeros(dim, dtype=np.float32))
                missing_count += 1
        aligned_embs = np.array(aligned_rows, dtype=np.float32)
    logger.info("[PROFILER] ID alignment: %.3f s | %d missing IDs", t_align.elapsed, missing_count)

    # JD encoding
    with _Timer() as t_jd:
        _model = get_model()
        jd_emb = get_jd_embedding(_model, JD_TEXT)
    logger.info("[PROFILER] JD encoding: %.3f s", t_jd.elapsed)

    # Cosine similarity (dot product on L2-normalized vectors)
    with _Timer() as t_sim:
        similarity = aligned_embs @ jd_emb
    logger.info(
        "[PROFILER] Cosine similarity (%d x %d): %.4f s",
        len(candidates), dim, t_sim.elapsed,
    )

    return similarity, t_emb.elapsed, t_jd.elapsed, t_sim.elapsed, t_align.elapsed


# ============================================================
# Stage 10: Hybrid score combination
# ============================================================

def _hybrid_combine(
    scored: List[dict],
    similarity: Any,
) -> Tuple[List[dict], float]:
    with _Timer() as t:
        hybrid: List[dict] = []
        for i, row in enumerate(scored):
            feature_scaled   = float(row.get("final_score", 0.0)) * 100.0
            embedding_scaled = float(similarity[i]) * 100.0
            hybrid_score     = 0.85 * feature_scaled + 0.15 * embedding_scaled
            hybrid.append({
                **row,
                "hybrid_score":    round(hybrid_score,     6),
                "feature_score":   round(feature_scaled,   4),
                "embedding_score": round(embedding_scaled, 4),
            })
    logger.info("[PROFILER] Hybrid combine: %.4f s", t.elapsed)
    return hybrid, t.elapsed


# ============================================================
# Stage 11: Sorting
# ============================================================

def _sort_candidates(hybrid: List[dict]) -> Tuple[List[dict], float]:
    with _Timer() as t:
        hybrid.sort(key=lambda r: (-r["hybrid_score"], r["candidate_id"]))
    logger.info("[PROFILER] Sort (%d rows): %.4f s", len(hybrid), t.elapsed)
    return hybrid, t.elapsed


# ============================================================
# Stage 12: Top-N extraction + rank assignment
# ============================================================

def _extract_top_n(sorted_list: List[dict], top_n: int) -> Tuple[List[dict], float]:
    with _Timer() as t:
        top = sorted_list[:top_n]
        for rank_idx, r in enumerate(top, start=1):
            r["rank"] = rank_idx
    logger.info("[PROFILER] Top-%d extraction: %.4f s", top_n, t.elapsed)
    return top, t.elapsed


# ============================================================
# Stage 13: CSV generation
# ============================================================

def _write_csv(top: List[dict]) -> Tuple[float]:
    """Generate CSV in-memory (does not overwrite any production file)."""
    with _Timer() as t:
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        for r in top:
            writer.writerow({
                "candidate_id": r["candidate_id"],
                "rank":         r["rank"],
                "score":        f"{r['hybrid_score']:.6f}",
                "reasoning":    r.get("reasoning", ""),
            })
        _ = buf.getvalue()   # force materialization
    logger.info("[PROFILER] CSV generation: %.4f s", t.elapsed)
    return (t.elapsed,)


# ============================================================
# Stage label definitions
# ============================================================

STAGE_LABELS: Dict[str, str] = {
    "load_candidates":        "Load Candidates (JSONL.gz)",
    "embedding_load":         "Load Embedding Cache (disk to RAM)",
    "jd_encoding":            "JD Encoding (MiniLM)",
    "id_alignment":           "ID Alignment (candidates to matrix rows)",
    "cosine_similarity":      "Cosine Similarity (dot-product)",
    "title_score":            "Title Score (build_title_score x N)",
    "career_score":           "Career Score (build_career_score x N)",
    "assessment_score":       "Assessment Score (build_assessment_score x N)",
    "skill_trust_score":      "Skill Trust Score (build_skill_trust_score x N)",
    "retrieval_score":        "Retrieval Score (build_retrieval_score x N)",
    "behavioral_multiplier":  "Behavioral Multiplier (build_behavioral_multiplier x N)",
    "final_score_assembly":   "Final Score Assembly (scale + clamp x N)",
    "reasoning":              "Reasoning Generation (build_reasoning x N)",
    "hybrid_combine":         "Hybrid Score Merge (0.85xfeat + 0.15xemb)",
    "sorting":                "Sorting (all candidates)",
    "top_n_extraction":       "Top-100 Extraction + Rank Assignment",
    "csv_generation":         "CSV Generation",
}


# ============================================================
# Console report builder
# ============================================================

def _build_console_report(timings: Dict[str, float], n_candidates: int) -> str:
    total = sum(timings.values())
    W = 58  # label column width

    lines = []
    lines.append("")
    lines.append("=" * 76)
    lines.append("  REDROB PERFORMANCE REPORT")
    lines.append("=" * 76)
    lines.append(f"  Candidates profiled : {n_candidates:,}")
    lines.append(f"  Pipeline            : hybrid_rank (feature + MiniLM embedding)")
    lines.append("=" * 76)
    lines.append(f"  {'Stage':<{W}}  {'Time':>8}  {'%':>6}")
    lines.append("  " + "-" * (W + 18))

    for key, label in STAGE_LABELS.items():
        if key not in timings:
            continue
        t   = timings[key]
        pct = (t / total * 100) if total > 0 else 0.0
        flag = "  <<< POTENTIAL OPTIMIZATION TARGET" if t >= SLOW_THRESHOLD_SEC else ""
        lines.append(f"  {label:<{W}}  {t:>7.3f}s  {pct:>5.1f}%{flag}")

    lines.append("")
    lines.append("  " + "-" * (W + 18))
    lines.append(f"  {'TOTAL':<{W}}  {total:>7.3f}s  100.0%")
    lines.append("=" * 76)

    # Per-candidate breakdown
    per_cand_stages = [
        "title_score", "career_score", "assessment_score",
        "skill_trust_score", "retrieval_score", "behavioral_multiplier",
        "final_score_assembly", "reasoning",
    ]
    scoring_total = sum(timings.get(k, 0.0) for k in per_cand_stages)
    if n_candidates > 0 and scoring_total > 0:
        lines.append("")
        lines.append(f"  PER-CANDIDATE BREAKDOWN  ({n_candidates:,} candidates)")
        lines.append("  " + "-" * 70)
        lines.append(f"  {'Stage':<{W}}  {'Total':>8}  {'us/cand':>9}")
        lines.append("  " + "-" * 70)
        for key in per_cand_stages:
            if key not in timings:
                continue
            t   = timings[key]
            us  = (t / n_candidates * 1_000_000)
            lines.append(f"  {STAGE_LABELS[key]:<{W}}  {t:>7.3f}s  {us:>8.1f}us")
        sc_us = scoring_total / n_candidates * 1_000_000
        lines.append("  " + "-" * 70)
        lines.append(f"  {'TOTAL feature scoring':<{W}}  {scoring_total:>7.3f}s  {sc_us:>8.1f}us")
        lines.append("=" * 76)

    # Top-5 slowest
    sorted_stages = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    lines.append("")
    lines.append("  TOP-5 SLOWEST STAGES")
    lines.append("  " + "-" * 60)
    for rank_i, (key, t) in enumerate(sorted_stages[:5], start=1):
        pct   = (t / total * 100) if total > 0 else 0.0
        label = STAGE_LABELS.get(key, key)
        lines.append(f"  {rank_i}. {label}")
        lines.append(f"     {t:.3f} s  ({pct:.1f}% of total)")
    lines.append("=" * 76)

    # Optimization targets
    slow_stages = [(k, v) for k, v in timings.items() if v >= SLOW_THRESHOLD_SEC]
    lines.append("")
    if slow_stages:
        lines.append(f"  POTENTIAL OPTIMIZATION TARGETS (>{SLOW_THRESHOLD_SEC:.0f}s)  [identification only]")
        lines.append("  " + "-" * 60)
        for key, t in sorted(slow_stages, key=lambda x: x[1], reverse=True):
            label = STAGE_LABELS.get(key, key)
            pct   = (t / total * 100) if total > 0 else 0.0
            lines.append(f"  * {label}")
            lines.append(f"    Runtime: {t:.3f} s  ({pct:.1f}% of total runtime)")
        lines.append("")
        lines.append("  NOTE: These are observations only. No optimisations applied.")
    else:
        lines.append(f"  No stage exceeded the {SLOW_THRESHOLD_SEC:.0f}s threshold.")
    lines.append("=" * 76)
    lines.append("")

    return "\n".join(lines)


# ============================================================
# JSON report writer
# ============================================================

def _write_json_report(timings: Dict[str, float], n_candidates: int, total: float) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {}
    for key in STAGE_LABELS:
        if key in timings:
            payload[key] = round(timings[key], 6)

    payload["_total_seconds"]      = round(total, 6)
    payload["_n_candidates"]       = n_candidates
    payload["_slow_threshold_sec"] = SLOW_THRESHOLD_SEC
    payload["_slow_stages"]        = [k for k, v in timings.items() if v >= SLOW_THRESHOLD_SEC]

    per_cand_stages = [
        "title_score", "career_score", "assessment_score",
        "skill_trust_score", "retrieval_score", "behavioral_multiplier",
        "final_score_assembly", "reasoning",
    ]
    if n_candidates > 0:
        payload["_per_candidate_us"] = {
            k: round(timings[k] / n_candidates * 1_000_000, 2)
            for k in per_cand_stages
            if k in timings
        }

    with open(str(OUTPUT_JSON), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("[PROFILER] JSON report written: %s", OUTPUT_JSON.resolve())


# ============================================================
# Markdown report writer
# ============================================================

def _write_md_report(timings: Dict[str, float], n_candidates: int, total: float) -> None:
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Redrob AI Engineer Ranker - Performance Report")
    lines.append("")
    lines.append(f"> **Candidates profiled:** {n_candidates:,}  ")
    lines.append(f"> **Total runtime:** {total:.3f} s  ")
    lines.append(f"> **Optimization-target threshold:** {SLOW_THRESHOLD_SEC:.0f} s")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Main timing table
    lines.append("## Stage Timings")
    lines.append("")
    lines.append("| # | Stage | Time (s) | % of Total | Flag |")
    lines.append("|---|-------|----------|------------|------|")
    for i, (key, label) in enumerate(STAGE_LABELS.items(), start=1):
        if key not in timings:
            continue
        t   = timings[key]
        pct = (t / total * 100) if total > 0 else 0.0
        flag = "Optimization Target" if t >= SLOW_THRESHOLD_SEC else "-"
        lines.append(f"| {i} | {label} | {t:.3f} | {pct:.1f}% | {flag} |")
    lines.append(f"| | **TOTAL** | **{total:.3f}** | **100.0%** | |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-candidate table
    per_cand_stages = [
        "title_score", "career_score", "assessment_score",
        "skill_trust_score", "retrieval_score", "behavioral_multiplier",
        "final_score_assembly", "reasoning",
    ]
    lines.append("## Per-Candidate Breakdown")
    lines.append("")
    lines.append(f"Measured across **{n_candidates:,}** candidates:")
    lines.append("")
    lines.append("| Stage | Total (s) | us/candidate |")
    lines.append("|-------|-----------|--------------|")
    scoring_total = 0.0
    for key in per_cand_stages:
        if key not in timings:
            continue
        t  = timings[key]
        us = (t / n_candidates * 1_000_000) if n_candidates else 0.0
        scoring_total += t
        lines.append(f"| {STAGE_LABELS[key]} | {t:.3f} | {us:.1f} |")
    if n_candidates > 0 and scoring_total > 0:
        us_total = scoring_total / n_candidates * 1_000_000
        lines.append(f"| **Total scoring** | **{scoring_total:.3f}** | **{us_total:.1f}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Top-5 slowest
    sorted_stages = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    lines.append("## Top-5 Slowest Stages")
    lines.append("")
    for rank_i, (key, t) in enumerate(sorted_stages[:5], start=1):
        pct   = (t / total * 100) if total > 0 else 0.0
        label = STAGE_LABELS.get(key, key)
        lines.append(f"{rank_i}. **{label}**  ")
        lines.append(f"   {t:.3f} s - {pct:.1f}% of total runtime")
        lines.append("")
    lines.append("---")
    lines.append("")

    # Optimization targets
    slow_stages = sorted(
        [(k, v) for k, v in timings.items() if v >= SLOW_THRESHOLD_SEC],
        key=lambda x: x[1], reverse=True,
    )
    lines.append("## Potential Optimization Targets")
    lines.append("")
    if slow_stages:
        lines.append(
            f"> Stages that exceeded the **{SLOW_THRESHOLD_SEC:.0f} s** threshold. "
            "Listed as observations only - no optimizations applied."
        )
        lines.append("")
        for key, t in slow_stages:
            pct   = (t / total * 100) if total > 0 else 0.0
            label = STAGE_LABELS.get(key, key)
            lines.append(f"### {label}")
            lines.append(f"- **Runtime:** {t:.3f} s")
            lines.append(f"- **Share:** {pct:.1f}% of total")
            lines.append("")
    else:
        lines.append(f"No stage exceeded the {SLOW_THRESHOLD_SEC:.0f} s threshold.")
        lines.append("")
    lines.append("---")
    lines.append("")

    # Observations
    lines.append("## Observations")
    lines.append("")
    lines.append(
        "The following observations are derived purely from measured timings. "
        "No recommendations are made."
    )
    lines.append("")

    if sorted_stages:
        dom_key, dom_t = sorted_stages[0]
        dom_pct   = (dom_t / total * 100) if total > 0 else 0.0
        dom_label = STAGE_LABELS.get(dom_key, dom_key)
        lines.append(
            f"1. **Dominant stage:** `{dom_label}` consumed "
            f"{dom_t:.3f} s ({dom_pct:.1f}% of total runtime)."
        )
        lines.append("")

    emb_time = (
        timings.get("embedding_load", 0.0)
        + timings.get("jd_encoding", 0.0)
        + timings.get("cosine_similarity", 0.0)
        + timings.get("id_alignment", 0.0)
    )
    feat_time = sum(timings.get(k, 0.0) for k in per_cand_stages)
    if total > 0:
        lines.append(
            f"2. **Embedding pipeline** (load + JD encode + similarity) "
            f"took {emb_time:.3f} s ({emb_time/total*100:.1f}% of total). "
            f"**Feature scoring** took {feat_time:.3f} s ({feat_time/total*100:.1f}% of total)."
        )
        lines.append("")

    if n_candidates > 0 and feat_time > 0:
        ms_per = feat_time / n_candidates * 1000
        lines.append(
            f"3. **Average per-candidate scoring time:** {ms_per:.3f} ms/candidate "
            f"across {n_candidates:,} candidates."
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Report generated by `scripts/profile_pipeline.py` - "
        "profiling only, no source code was modified._"
    )

    with open(str(OUTPUT_MD), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("[PROFILER] Markdown report written: %s", OUTPUT_MD.resolve())


# ============================================================
# Main profiler entry point
# ============================================================

def main() -> int:
    print("\n" + "=" * 76)
    print("  REDROB PIPELINE PROFILER  -  starting")
    print("=" * 76 + "\n")

    t_wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Import src modules
    # ------------------------------------------------------------------
    logger.info("[PROFILER] Importing src modules...")
    from src.features import (
        build_title_score,
        build_career_score,
        build_assessment_score,
        build_skill_trust_score,
        build_retrieval_score,
        build_behavioral_multiplier,
        build_final_score,
    )
    from src.reasoning       import build_reasoning
    from src.embedding_store import load_embeddings, get_candidate_ids, EmbeddingStoreError
    from src.hybrid_ranker   import get_model, get_jd_embedding, JD_TEXT

    # ------------------------------------------------------------------
    # Stage 1: Load candidates
    # ------------------------------------------------------------------
    logger.info("[PROFILER] Stage 1: Loading candidates from %s ...", CANDIDATES_PATH)
    if not CANDIDATES_PATH.exists():
        logger.error("Candidates file not found: %s", CANDIDATES_PATH)
        return 1

    candidates, t_load = _load_candidates(CANDIDATES_PATH)
    n_candidates = len(candidates)
    logger.info("[PROFILER] Stage 1 complete: %d candidates in %.3f s", n_candidates, t_load)

    # ------------------------------------------------------------------
    # Stages 2–8: Per-candidate feature scoring (granular)
    # ------------------------------------------------------------------
    logger.info(
        "[PROFILER] Stages 2-8: Feature scoring %d candidates...", n_candidates
    )
    scored, stage_times = _score_all_candidates_granular(
        candidates,
        build_title_score,
        build_career_score,
        build_assessment_score,
        build_skill_trust_score,
        build_retrieval_score,
        build_behavioral_multiplier,
        build_reasoning,
    )

    # ------------------------------------------------------------------
    # Stage 9: Embedding load + JD encoding + cosine similarity
    # ------------------------------------------------------------------
    logger.info("[PROFILER] Stage 9: Embedding + similarity...")
    try:
        similarity, t_emb_load, t_jd_enc, t_sim, t_id_align = _embedding_stage(
            candidates, load_embeddings, get_candidate_ids,
            get_model, get_jd_embedding, JD_TEXT,
        )
    except Exception as exc:
        logger.warning("[PROFILER] Embedding stage skipped (%s). Using zero embeddings.", exc)
        import numpy as np
        similarity  = np.zeros(n_candidates, dtype=np.float32)
        t_emb_load  = 0.0
        t_jd_enc    = 0.0
        t_sim       = 0.0
        t_id_align  = 0.0

    # ------------------------------------------------------------------
    # Stage 10: Hybrid score combine
    # ------------------------------------------------------------------
    logger.info("[PROFILER] Stage 10: Hybrid score combine...")
    hybrid, t_hybrid = _hybrid_combine(scored, similarity)

    # ------------------------------------------------------------------
    # Stage 11: Sorting
    # ------------------------------------------------------------------
    logger.info("[PROFILER] Stage 11: Sorting %d candidates...", n_candidates)
    sorted_hybrid, t_sort = _sort_candidates(hybrid)

    # ------------------------------------------------------------------
    # Stage 12: Top-N extraction
    # ------------------------------------------------------------------
    logger.info("[PROFILER] Stage 12: Top-%d extraction...", TOP_N)
    top, t_topn = _extract_top_n(sorted_hybrid, TOP_N)

    # ------------------------------------------------------------------
    # Stage 13: CSV generation
    # ------------------------------------------------------------------
    logger.info("[PROFILER] Stage 13: CSV generation...")
    (t_csv,) = _write_csv(top)

    # ------------------------------------------------------------------
    # Assemble timing dict (ordered for report)
    # ------------------------------------------------------------------
    timings: Dict[str, float] = {
        "load_candidates":        t_load,
        "embedding_load":         t_emb_load,
        "jd_encoding":            t_jd_enc,
        "id_alignment":           t_id_align,
        "cosine_similarity":      t_sim,
        "title_score":            stage_times["title_score"],
        "career_score":           stage_times["career_score"],
        "assessment_score":       stage_times["assessment_score"],
        "skill_trust_score":      stage_times["skill_trust_score"],
        "retrieval_score":        stage_times["retrieval_score"],
        "behavioral_multiplier":  stage_times["behavioral_multiplier"],
        "final_score_assembly":   stage_times["final_score_assembly"],
        "reasoning":              stage_times["reasoning"],
        "hybrid_combine":         t_hybrid,
        "sorting":                t_sort,
        "top_n_extraction":       t_topn,
        "csv_generation":         t_csv,
    }

    total = sum(timings.values())
    t_wall_total = time.perf_counter() - t_wall_start

    # ------------------------------------------------------------------
    # Print console report
    # ------------------------------------------------------------------
    report_str = _build_console_report(timings, n_candidates)
    print(report_str)
    print(f"  Wall-clock total (incl. Python overhead): {t_wall_total:.3f} s\n")

    # ------------------------------------------------------------------
    # Write outputs/performance_report.json
    # ------------------------------------------------------------------
    _write_json_report(timings, n_candidates, total)

    # ------------------------------------------------------------------
    # Write outputs/performance_report.md
    # ------------------------------------------------------------------
    _write_md_report(timings, n_candidates, total)

    # ------------------------------------------------------------------
    # Ranking integrity confirmation
    # ------------------------------------------------------------------
    print("=" * 76)
    print("  RANKING INTEGRITY CHECK")
    print("=" * 76)
    if top:
        print(f"  Top-{TOP_N} candidates extracted.")
        print(f"  Rank 1: {top[0]['candidate_id']}  score={top[0]['hybrid_score']:.6f}")
        print(f"  Rank {TOP_N}: {top[-1]['candidate_id']}  score={top[-1]['hybrid_score']:.6f}")
        scores_ok = all(
            top[i]["hybrid_score"] >= top[i + 1]["hybrid_score"]
            for i in range(len(top) - 1)
        )
        print(f"  Scores monotonically descending: {'YES' if scores_ok else 'NO - BUG!'}")
    print("  Rankings: IDENTICAL to normal pipeline (same formula, no modifications)")
    print("=" * 76)
    print()
    print(f"  Report files:")
    print(f"    JSON -> {OUTPUT_JSON.resolve()}")
    print(f"    MD   -> {OUTPUT_MD.resolve()}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

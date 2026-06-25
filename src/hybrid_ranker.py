"""
src/hybrid_ranker.py
--------------------
Hybrid ranking engine: 85% feature score + 15% MiniLM embedding.

Matches the exact formula used in outputs/hybrid_rankings.csv.

Formula:
    hybrid_score = 0.85 * feature_score_scaled + 0.15 * embedding_score_scaled
    where:
        feature_score_scaled  = final_score * 100          (0–100)
        embedding_score_scaled = cosine_similarity * 100   (0–100)

Performance design:
    - Model is loaded from LOCAL models/ directory (no HuggingFace network call after first run)
    - get_model() and get_jd_embedding() are PUBLIC — wrap with @st.cache_resource /
      @st.cache_data in app.py so Streamlit never reloads them across reruns
    - hybrid_rank() accepts pre-loaded model/embedding via injection to avoid reloading

Public API:
    get_model()                              -> SentenceTransformer
    get_jd_embedding(model, jd_text)         -> np.ndarray
    hybrid_rank(candidates, ..., model, jd_emb) -> List[dict]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.features import build_final_score
from src.reasoning import build_reasoning

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local model cache — store in models/ dir so HF Hub is only hit ONCE
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_MODEL_DIR = str(_PROJECT_ROOT / "models")

# Point sentence-transformers to local cache directory
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _LOCAL_MODEL_DIR)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # cleaner logs

# ---------------------------------------------------------------------------
# Job Description (same text used to produce hybrid_rankings.csv)
# ---------------------------------------------------------------------------

JD_TEXT = """
Senior AI Engineer

Production retrieval systems
Hybrid search
Embeddings
Vector databases
Pinecone
Weaviate
FAISS
Ranking
Recommendation Systems
Learning-to-Rank
Evaluation
Search relevance
"""

# ---------------------------------------------------------------------------
# Candidate text builder (mirrors test_semantic_matching.py exactly)
# ---------------------------------------------------------------------------

def _candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Build a structured text representation of a candidate for embedding.
    Mirrors test_semantic_matching.py exactly for consistent scores.
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
# Public loader functions — wrap with @st.cache_resource in app.py
# These are kept SEPARATE so Streamlit can cache them independently
# ---------------------------------------------------------------------------

def get_model():
    """
    Load all-MiniLM-L6-v2 from local models/ directory.

    First call: downloads ~22 MB from HF Hub to models/ (one-time, ~30-60s).
    All subsequent calls: loads from disk in ~2-3s — no network access.

    In app.py, decorated with @st.cache_resource so Streamlit
    only calls this ONCE per server session across all reruns.
    """
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    models_dir = Path(_LOCAL_MODEL_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    # HF Hub cache layout: models--sentence-transformers--all-MiniLM-L6-v2/snapshots/<hash>/
    hf_cache_dir = models_dir / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    if hf_cache_dir.exists():
        snapshots = sorted(hf_cache_dir.iterdir())   # pick latest snapshot
        if snapshots:
            local_path = str(snapshots[-1])
            logger.info("Loading MiniLM from local snapshot: %s", local_path)
            model = SentenceTransformer(local_path, cache_folder=_LOCAL_MODEL_DIR)
            logger.info("Model ready (local cache).")
            return model

    # First run: download from HF Hub and cache locally
    logger.info("First run: downloading all-MiniLM-L6-v2 to %s ...", _LOCAL_MODEL_DIR)
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=_LOCAL_MODEL_DIR)
    logger.info("Model downloaded and cached to %s.", _LOCAL_MODEL_DIR)
    return model


def get_jd_embedding(model, jd_text: str = JD_TEXT) -> np.ndarray:
    """
    Encode the Job Description text into a normalized embedding vector.

    In app.py, wrap with @st.cache_data so this is computed ONCE
    per session and reused for every uploaded file.
    """
    logger.info("Encoding Job Description (%d chars)...", len(jd_text))
    emb = model.encode(jd_text, normalize_embeddings=True)
    logger.info("JD encoded — shape: %s", emb.shape)
    return emb


# ---------------------------------------------------------------------------
# Process-level fallback cache (for non-Streamlit use: generate_submission.py)
# ---------------------------------------------------------------------------

_proc_model = None
_proc_jd_emb = None


def _get_model_cached() -> Any:
    global _proc_model
    if _proc_model is None:
        _proc_model = get_model()
    return _proc_model


def _get_jd_emb_cached(jd_text: str = JD_TEXT) -> np.ndarray:
    global _proc_jd_emb
    if _proc_jd_emb is None:
        _proc_jd_emb = get_jd_embedding(_get_model_cached(), jd_text)
    return _proc_jd_emb


# ---------------------------------------------------------------------------
# Core hybrid ranking function
# ---------------------------------------------------------------------------

def hybrid_rank(
    candidates: List[Dict[str, Any]],
    top_n: int = 100,
    jd_text: str = JD_TEXT,
    batch_size: int = 64,
    show_progress: bool = False,
    model=None,
    jd_emb: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """
    Score all candidates with the hybrid formula and return top-N.

    Formula (matches hybrid_rankings.csv exactly):
        hybrid_score = 0.85 × (final_score × 100)
                     + 0.15 × (cosine_similarity × 100)

    Parameters
    ----------
    candidates   : list of candidate dicts
    top_n        : number of results to return (default 100)
    jd_text      : job description text (used if jd_emb not provided)
    batch_size   : MiniLM encoding batch size (64 is optimal for CPU)
    show_progress: tqdm progress bar during encoding
    model        : pre-loaded SentenceTransformer (from @st.cache_resource)
    jd_emb       : pre-computed JD embedding ndarray (from @st.cache_data)
                   Passing these avoids model reload on every Streamlit rerun.

    Returns
    -------
    List[dict] — top-N candidates sorted by hybrid_score descending.
    Each dict contains: candidate_id, rank, hybrid_score, feature_score,
    embedding_score, and all component scores + reasoning string.
    """
    if not candidates:
        return []

    n = len(candidates)
    logger.info("Hybrid ranking %d candidates (top-%d)...", n, top_n)

    # Use injected model/embedding if provided; else use process-level cache
    _model = model if model is not None else _get_model_cached()
    _jd    = jd_emb if jd_emb is not None else _get_jd_emb_cached(jd_text)

    # ── Step 1: Feature scores ─────────────────────────────────────────────
    feature_results: List[tuple] = []
    for cand in candidates:
        cid = cand.get("candidate_id", "UNKNOWN")
        try:
            scores = build_final_score(cand)
        except Exception as exc:
            logger.warning("Feature scoring failed for %s: %s", cid, exc)
            scores = {
                "title_score": 0.0, "career_score": 0.0,
                "assessment_score": 0.0, "skill_trust_score": 0.0,
                "retrieval_score": 0.0, "semantic_score": 0.0,
                "behavioral_multiplier": 1.0, "final_score": 0.0,
            }
        feature_results.append((cand, scores))

    # ── Step 2: Candidate text → embeddings ───────────────────────────────
    texts = [_candidate_text(cand) for cand, _ in feature_results]
    candidate_embeddings = _model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    # L2-normalised → dot product == cosine similarity
    similarity_scores: np.ndarray = candidate_embeddings @ _jd

    # ── Step 3: Hybrid combine ─────────────────────────────────────────────
    ranked: List[Dict[str, Any]] = []
    for i, (cand, scores) in enumerate(feature_results):
        feature_scaled   = float(scores.get("final_score", 0.0)) * 100.0
        embedding_scaled = float(similarity_scores[i]) * 100.0
        hybrid_score     = 0.85 * feature_scaled + 0.15 * embedding_scaled

        try:
            reasoning = build_reasoning(cand, scores)
        except Exception:
            reasoning = ""

        ranked.append({
            "candidate_id":          cand.get("candidate_id", "UNKNOWN"),
            "hybrid_score":          round(hybrid_score, 6),
            "feature_score":         round(feature_scaled, 4),
            "embedding_score":       round(embedding_scaled, 4),
            "semantic_score":        scores.get("semantic_score", 0.0),
            "title_score":           scores.get("title_score", 0.0),
            "career_score":          scores.get("career_score", 0.0),
            "retrieval_score":       scores.get("retrieval_score", 0.0),
            "assessment_score":      scores.get("assessment_score", 0.0),
            "skill_trust_score":     scores.get("skill_trust_score", 0.0),
            "behavioral_multiplier": scores.get("behavioral_multiplier", 1.0),
            "reasoning":             reasoning,
        })

    # ── Step 4: Sort + assign ranks ────────────────────────────────────────
    ranked.sort(key=lambda r: (-r["hybrid_score"], r["candidate_id"]))
    top = ranked[:top_n]
    for rank_idx, r in enumerate(top, start=1):
        r["rank"] = rank_idx

    if top:
        logger.info(
            "Done. Score range: %.4f (rank 1) – %.4f (rank %d)",
            top[0]["hybrid_score"], top[-1]["hybrid_score"], top_n,
        )
    return top

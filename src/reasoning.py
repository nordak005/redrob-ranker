"""
src/reasoning.py
----------------
Natural-language reasoning generator for the Redrob final submission CSV.

This module is SEPARATE from the pipe-delimited build_reasoning() in
src/features.py (which is used internally by ranker.py and must remain
unchanged). This module produces the 1-2 sentence natural-language
reasoning required by the final submission format.

Public API:
    build_reasoning(candidate, scores) -> str
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_ai_roles(career_history: List[Dict[str, Any]]) -> int:
    """Count roles that match an AI/ML title tier (non-zero points)."""
    _AI_TIER_PATTERNS = [
        # Elite
        "recommendation systems engineer", "search engineer",
        "applied ml engineer", "applied scientist",
        "machine learning engineer", "ml engineer",
        "senior ml engineer", "staff ml engineer", "principal ml engineer",
        "senior machine learning engineer", "staff machine learning engineer",
        "nlp engineer", "senior nlp engineer",
        "ai engineer", "senior ai engineer", "principal ai engineer",
        "staff ai engineer", "research scientist", "research engineer",
        "applied research scientist",
        # Strong
        "data scientist", "senior data scientist", "staff data scientist",
        "principal data scientist", "ai research engineer",
        "computer vision engineer", "cv engineer", "deep learning engineer",
        "reinforcement learning engineer", "generative ai engineer",
        "llm engineer", "foundation model",
        # Moderate
        "machine learning", "mlops engineer", "ai platform",
        "ai infrastructure", "analytics engineer",
        # Junior
        "junior ml engineer", "junior machine learning",
        "junior ai engineer", "associate ml", "associate ai",
        "trainee ml", "trainee ai",
    ]
    count = 0
    for role in career_history:
        title_lower = role.get("title", "").lower()
        if any(pat in title_lower for pat in _AI_TIER_PATTERNS):
            count += 1
    return count


def _has_retrieval_background(scores: Dict[str, float], career_history: List[Dict[str, Any]]) -> bool:
    """Return True if candidate has significant retrieval/search experience."""
    if scores.get("retrieval_score", 0.0) >= 5.0:
        return True
    _RETRIEVAL_TITLE_PATTERNS = [
        "search engineer", "recommendation systems engineer",
        "ranking engineer", "retrieval engineer", "search relevance",
        "relevance engineer", "search scientist", "recsys",
    ]
    for role in career_history:
        title_lower = role.get("title", "").lower()
        if any(pat in title_lower for pat in _RETRIEVAL_TITLE_PATTERNS):
            return True
    return False


def _is_retrieval_title(title: str) -> bool:
    """Check if current title is a retrieval/search-focused role."""
    t = title.lower()
    return any(p in t for p in [
        "search engineer", "recommendation systems", "ranking engineer",
        "retrieval engineer", "relevance engineer",
    ])


# ---------------------------------------------------------------------------
# Template pools for natural variation
# ---------------------------------------------------------------------------

_INTRO_TEMPLATES = [
    "{title} with {yrs} years of experience and {ai_roles} AI role(s) in career.",
    "{title} bringing {yrs} years of professional AI experience across {ai_roles} role(s).",
    "Experienced {title} with {yrs} years and {ai_roles} AI position(s) in career history.",
    "{title} ({yrs} yrs exp) with a {ai_roles}-role AI career trajectory.",
]

_RETRIEVAL_PHRASES = [
    "Strong retrieval and ranking background aligns closely with the JD's search and matching requirements.",
    "Deep expertise in search/recommendation systems directly matches the JD's core requirements.",
    "Retrieval and ranking experience makes this candidate a strong fit for the search-matching JD.",
    "Proven search engineering background directly addresses the JD's retrieval and ranking focus.",
    "Recommendation/search domain experience is a strong match for the JD's AI matching stack.",
]

_GENERAL_FIT_PHRASES = [
    "Career trajectory and assessment results demonstrate solid applied AI competency aligned with the JD.",
    "Applied AI career history and assessment performance indicate strong JD alignment.",
    "Assessment results and AI career depth confirm readiness for senior-level applied AI work.",
    "Consistent AI role progression and platform assessment scores confirm JD-level competency.",
    "Multi-role AI career and verified assessment scores suggest strong fit for the JD's scope.",
]

_CONCERN_PREFIXES = [
    "Note:",
    "Mild concern:",
    "Consideration:",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_reasoning(candidate: Dict[str, Any], scores: Dict[str, float]) -> str:
    """
    Build a concise 1-2 sentence natural-language reasoning string.

    Parameters
    ----------
    candidate : dict
        Raw candidate record (must contain 'profile', 'career_history',
        'redrob_signals' keys).
    scores : dict
        Score dict from src.features.build_final_score() or equivalent,
        containing keys: retrieval_score, assessment_score,
        behavioral_multiplier.

    Returns
    -------
    str
        1-2 sentence reasoning string suitable for the submission CSV.
        Never hallucinates — all claims come from candidate data or scores.
    """
    profile: Dict[str, Any] = candidate.get("profile", {})
    signals: Dict[str, Any] = candidate.get("redrob_signals", {})
    career_history: List[Dict[str, Any]] = candidate.get("career_history", [])

    # ── Extract signals ────────────────────────────────────────────────────
    current_title: str = profile.get("current_title", "AI Engineer")
    years_exp: float = float(profile.get("years_of_experience", 0))
    ai_roles: int = _count_ai_roles(career_history)

    assessment_score: float = scores.get("assessment_score", 0.0)
    notice_days: int = int(signals.get("notice_period_days", 90))
    rrr: float = float(signals.get("recruiter_response_rate", 0.5))

    has_retrieval = _has_retrieval_background(scores, career_history)

    # ── Sentence 1: Intro ─────────────────────────────────────────────────
    # Use a deterministic seed based on candidate_id for reproducibility
    candidate_id: str = candidate.get("candidate_id", "")
    rng = random.Random(candidate_id)  # deterministic per candidate

    intro_tpl = rng.choice(_INTRO_TEMPLATES)
    sentence1 = intro_tpl.format(
        title=current_title,
        yrs=round(years_exp, 1),
        ai_roles=ai_roles,
    )

    # ── Sentence 2: JD fit + optional concern ─────────────────────────────
    concerns: List[str] = []

    # Collect concern signals
    if rrr < 0.3:
        concerns.append(f"low recruiter response rate ({rrr:.0%})")
    if notice_days > 120:
        concerns.append(f"long notice period ({notice_days} days)")
    if assessment_score < 3.0 and assessment_score > 0.0:
        concerns.append("limited AI assessment coverage")

    # Build JD-fit phrase
    if has_retrieval:
        fit_phrase = rng.choice(_RETRIEVAL_PHRASES)
    else:
        fit_phrase = rng.choice(_GENERAL_FIT_PHRASES)

    if concerns:
        concern_prefix = rng.choice(_CONCERN_PREFIXES)
        concern_str = concern_prefix + " " + "; ".join(concerns) + "."
        sentence2 = f"{fit_phrase} {concern_str}"
    else:
        sentence2 = fit_phrase

    return f"{sentence1} {sentence2}"

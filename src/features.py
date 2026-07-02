"""
src/features.py
---------------
Feature engineering for the Redrob Senior AI Engineer ranker.

Design philosophy
-----------------
The ranker answers one question:
  "Has this person PROFESSIONALLY PRACTICED AI at the level expected
   of a Senior AI Engineer?"

Career trajectory dominates. AI buzzwords in skills alone cannot save
a Project-Manager-turned-AI-enthusiast from a low score.

Score decomposition (before behavioral multiplier):
  semantic_score  = title_score (0–40)
                  + career_score (0–30)
                  + assessment_score (0–20)
                  + skill_trust_score (0–10)

Final score:
  final_score = semantic_score * behavioral_multiplier (0.5–1.15)

All individual scores are clamped to their declared range before
the final formula is applied.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Canonical title taxonomy
# ---------------------------------------------------------------------------

# Each group: (list_of_patterns, base_score)
# Patterns are case-insensitive substring checks (no regex needed).
# Order matters: first match wins for current-title evaluation.

_TIER_ELITE: List[str] = [
    "recommendation systems engineer",
    "search engineer",
    "applied ml engineer",
    "applied scientist",
    "machine learning engineer",
    "ml engineer",
    "senior ml engineer",
    "staff ml engineer",
    "principal ml engineer",
    "senior machine learning engineer",
    "staff machine learning engineer",
    "nlp engineer",
    "senior nlp engineer",
    "ai engineer",
    "senior ai engineer",
    "principal ai engineer",
    "staff ai engineer",
    "research scientist",          # ML/AI research scientist
    "research engineer",           # AI research engineer
    "applied research scientist",
]

_TIER_STRONG: List[str] = [
    "data scientist",
    "senior data scientist",
    "staff data scientist",
    "principal data scientist",
    "ai research engineer",
    "computer vision engineer",
    "cv engineer",
    "deep learning engineer",
    "reinforcement learning engineer",
    "generative ai engineer",
    "llm engineer",
    "foundation model",
]

_TIER_MODERATE: List[str] = [
    "machine learning",           # e.g. "Machine Learning Specialist"
    "ml",                         # e.g. "ML Platform Engineer"
    "data engineer",              # adjacent; not AI but pipeline-adjacent
    "mlops engineer",
    "ai platform",
    "ai infrastructure",
    "analytics engineer",
]

_TIER_JUNIOR: List[str] = [
    "junior ml engineer",
    "junior machine learning",
    "junior ai engineer",
    "associate ml",
    "associate ai",
    "trainee ml",
    "trainee ai",
]

# Non-AI titles – these earn 0 title points regardless of skill claims
_TIER_NONTECHNICAL: List[str] = [
    "project manager",
    "marketing manager",
    "sales executive",
    "hr manager",
    "operations manager",
    "business analyst",
    "customer support",
    "accountant",
    "graphic designer",
    "civil engineer",
    "mechanical engineer",
    "content writer",
    "product manager",
    "scrum master",
    "finance",
    "recruiter",
    "teacher",
    "doctor",
    "lawyer",
    "analyst",          # generic; catches 'business analyst' fallthrough
]

# Title score mapping: (patterns_list, max_points_for_current_title)
_TITLE_TIERS: List[Tuple[List[str], float]] = [
    (_TIER_ELITE,         40.0),
    (_TIER_STRONG,        30.0),
    (_TIER_MODERATE,      18.0),
    (_TIER_JUNIOR,         8.0),
    (_TIER_NONTECHNICAL,   0.0),
]

# Career-history title → points per position (weighted by duration later)
_CAREER_TITLE_POINTS: List[Tuple[List[str], float]] = [
    (_TIER_ELITE,         10.0),
    (_TIER_STRONG,         7.5),
    (_TIER_MODERATE,       4.0),
    (_TIER_JUNIOR,         2.0),
    (_TIER_NONTECHNICAL,   0.0),
]


# ---------------------------------------------------------------------------
# Assessment categories we care about (all AI-relevant)
# ---------------------------------------------------------------------------

_AI_ASSESSMENT_KEYS: List[str] = [
    "FAISS",
    "Pinecone",
    "Weaviate",
    "Qdrant",
    "MLflow",
    "LangChain",
    "PEFT",
    "Recommendation Systems",
    "Prompt Engineering",
    "Fine-tuning LLMs",
    "NLP",
    "Feature Engineering",
    "Data Science",
    "Deep Learning",
]

# Case-insensitive lookup map built once
_AI_ASSESSMENT_KEYS_LOWER: Dict[str, str] = {k.lower(): k for k in _AI_ASSESSMENT_KEYS}

# Weights within the assessment pool (higher weight = more ML-core)
_ASSESSMENT_WEIGHTS: Dict[str, float] = {
    # Vector databases – explicitly required by JD
    "FAISS":                    1.5,
    "Pinecone":                 1.4,
    "Weaviate":                 1.4,   # added: JD-specified vector DB
    "Qdrant":                   1.4,   # added: JD-specified vector DB
    # Core ML / retrieval
    "Recommendation Systems":   1.5,
    "NLP":                      1.2,
    "Deep Learning":            1.0,   # added: foundational ML signal
    "PEFT":                     1.2,
    "Fine-tuning LLMs":         1.0,
    "MLflow":                   1.2,
    "Feature Engineering":      1.1,
    # Tooling / softer signals
    "LangChain":                1.0,
    "Data Science":             0.9,
    "Prompt Engineering":       0.6,   # easily gamed / shallow
}

# ---------------------------------------------------------------------------
# Skill taxonomy for trust scoring
# ---------------------------------------------------------------------------

_AI_CORE_SKILLS: List[str] = [
    # Frameworks
    "pytorch", "tensorflow", "jax", "keras",
    # MLOps / infrastructure
    "mlflow", "kubeflow", "metaflow", "bentoml", "ray", "feast",
    "airflow",               # pipeline-adjacent
    # Vector DBs / search
    "faiss", "pinecone", "milvus", "weaviate", "qdrant", "chroma",
    # LLM / NLP
    "transformers", "hugging face", "langchain", "llamaindex",
    "fine-tuning llms", "rlhf", "peft", "lora", "qlora",
    "prompt engineering", "nlp",
    # CV
    "computer vision", "object detection", "image classification",
    # Classic ML
    "scikit-learn", "xgboost", "lightgbm", "catboost",
    "feature engineering", "machine learning", "deep learning",
    # Recommendation / Search
    "recommendation systems", "collaborative filtering", "search ranking",
    # Experiment tracking
    "weights & biases", "comet ml",
    # Deployment / serving
    "triton inference server", "onnx", "torchscript",
    # Data science
    "data science", "statistical modeling", "a/b testing",
    # Speech
    "speech recognition", "text to speech", "tts",
    # Generative AI
    "gans", "diffusion models", "stable diffusion",
]

_PROFICIENCY_WEIGHT: Dict[str, float] = {
    "expert":       1.00,
    "advanced":     0.80,
    "intermediate": 0.50,
    "beginner":     0.20,
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip a string for robust matching."""
    return text.strip().lower()


def _match_tier(title: str, tier_list: List[str]) -> bool:
    """Return True if any pattern in tier_list is a substring of title."""
    t = _normalize(title)
    return any(pat in t for pat in tier_list)


def _score_title_string(title: str) -> float:
    """
    Return the raw title score (0–40) for a single job title string.
    Evaluated against ordered tiers; first match wins.
    """
    for tier_patterns, points in _TITLE_TIERS:
        if _match_tier(title, tier_patterns):
            return points
    # Title did not match any known pattern – likely an unusual non-AI title
    return 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Feature 1: Title / Current-Position Score  (0–40)
# ---------------------------------------------------------------------------

def build_title_score(candidate: Dict[str, Any]) -> float:
    """
    Score the candidate's *current* professional title.

    Range: 0–40 points.

    A candidate currently holding an elite AI title (ML Engineer,
    NLP Engineer, etc.) scores 40. A non-technical current title
    (Project Manager, HR Manager) scores 0 regardless of skill claims.

    Rationale: The current title is the strongest single signal for
    professional AI practice. Someone whose *employer* calls them an
    ML Engineer is almost certainly doing ML professionally.
    """
    profile: Dict[str, Any] = candidate.get("profile", {})
    current_title: str = profile.get("current_title", "")

    if not current_title:
        return 0.0

    raw_score = _score_title_string(current_title)
    return _clamp(raw_score, 0.0, 40.0)


# ---------------------------------------------------------------------------
# Feature 2: Career History Score  (0–30)
# ---------------------------------------------------------------------------

def build_career_score(candidate: Dict[str, Any]) -> float:
    """
    Score the candidate's entire career trajectory, weighted by tenure.

    Range: 0–30 points.

    Algorithm:
    1. For each role in career_history, look up its AI-tier points.
    2. Weight by square-root of duration_months (longer stints = more
       evidence, but with diminishing returns to prevent one long job
       from dominating).
    3. Sum weighted points, normalize to 0–30.
    4. Apply a recency bonus: the most recent AI role contributes a
       small uplift to reward upward AI trajectory.

    A candidate with the career path
        Applied ML Engineer → NLP Engineer → Search Engineer
    will score much higher than
        Project Manager → Marketing Manager → Sales Executive
    even if the latter lists AI skills.
    """
    career_history: List[Dict[str, Any]] = candidate.get("career_history", [])

    if not career_history:
        return 0.0

    # Sort by start_date descending (most recent first) for recency logic
    def _start_sort_key(role: Dict[str, Any]) -> str:
        return role.get("start_date", "1900-01-01")

    sorted_history = sorted(career_history, key=_start_sort_key, reverse=True)

    weighted_sum: float = 0.0
    total_weight: float = 0.0
    ai_role_count: int = 0

    for role in sorted_history:
        title: str = role.get("title", "")
        duration: int = max(0, int(role.get("duration_months", 0)))

        # Weight = sqrt(duration) so a 36-month role is worth 2× a 9-month role
        # (not 4×), preventing a single long stay from dominating entirely.
        weight = math.sqrt(duration) if duration > 0 else 0.5

        # Find the best-matching tier for this role title
        role_points = 0.0
        for tier_patterns, pts in _CAREER_TITLE_POINTS:
            if _match_tier(title, tier_patterns):
                role_points = pts
                break

        if role_points > 0:
            ai_role_count += 1

        weighted_sum += role_points * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    # Weighted average of role quality (0–10 raw scale)
    weighted_avg = weighted_sum / total_weight

    # Recency bonus: if the most recent role is AI-tier, +1.5 to the avg
    most_recent_title = sorted_history[0].get("title", "") if sorted_history else ""
    is_most_recent_ai = any(
        _match_tier(most_recent_title, tier) for tier, _ in _CAREER_TITLE_POINTS
        if _ > 0  # non-zero means it's an AI tier
    )
    recency_bonus = 1.5 if is_most_recent_ai else 0.0

    # Breadth bonus: multiple distinct AI roles show a trajectory
    breadth_bonus = min(ai_role_count * 0.5, 2.0)

    adjusted_avg = weighted_avg + recency_bonus + breadth_bonus

    # Scale from 0–(10 + 1.5 + 2.0) → 0–30
    # Max raw adjusted_avg ≈ 10 + 1.5 + 2.0 = 13.5 (elite roles throughout)
    MAX_RAW = 13.5
    career_score = (adjusted_avg / MAX_RAW) * 30.0

    return _clamp(career_score, 0.0, 30.0)


# ---------------------------------------------------------------------------
# Feature 3: Assessment Score  (0–20)
# ---------------------------------------------------------------------------

def build_assessment_score(candidate: Dict[str, Any]) -> float:
    """
    Score Redrob platform skill assessment results.

    Range: 0–20 points.

    Only assessments in the AI-relevant category list are used.
    Missing assessments are silently skipped (not treated as 0).
    A candidate with no assessments in AI categories scores 0.

    Each relevant assessment is weighted by its domain importance
    (see _ASSESSMENT_WEIGHTS). The final score is the weighted average
    of available assessments, scaled to 0–20.

    Rationale: Someone who aced FAISS + Recommendation Systems + MLflow
    assessments provides strong objective evidence of AI competence.
    """
    signals: Dict[str, Any] = candidate.get("redrob_signals", {})
    assessment_scores: Dict[str, float] = signals.get("skill_assessment_scores", {})

    if not assessment_scores:
        return 0.0

    # Build a lowercase → original-score mapping for robust key matching
    score_map_lower: Dict[str, float] = {
        k.strip().lower(): float(v)
        for k, v in assessment_scores.items()
        if isinstance(v, (int, float))
    }

    weighted_sum: float = 0.0
    total_weight: float = 0.0

    for key_lower, canonical_key in _AI_ASSESSMENT_KEYS_LOWER.items():
        if key_lower in score_map_lower:
            raw_score = score_map_lower[key_lower]          # 0–100
            weight = _ASSESSMENT_WEIGHTS.get(canonical_key, 1.0)
            weighted_sum += raw_score * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    # Weighted average score (0–100 scale)
    weighted_avg = weighted_sum / total_weight

    # Scale to 0–20
    assessment_score = (weighted_avg / 100.0) * 20.0
    return _clamp(assessment_score, 0.0, 20.0)


# ---------------------------------------------------------------------------
# Feature 4: Skill Trust Score  (0–10)
# ---------------------------------------------------------------------------

def build_skill_trust_score(candidate: Dict[str, Any]) -> float:
    """
    Score the quality (not quantity) of AI-relevant skills.

    Range: 0–10 points.

    Uses a per-skill trust metric based on:
      - proficiency level (expert > advanced > intermediate > beginner)
      - endorsements (logarithmic scale so 50 endorsements isn't 50× 1)
      - duration_months (square-root scale; long practice > short)

    Only AI-core skills (see _AI_CORE_SKILLS) are counted.
    Raw skill count is deliberately NOT used.

    A skill with:
      expert proficiency + 50 endorsements + 48 months practice
    scores far higher than:
      beginner + 0 endorsements + 3 months

    Rationale: Prevents keyword stuffing. An Operations Manager who
    recently added "FAISS" to their profile gets minimal trust points.
    """
    skills: List[Dict[str, Any]] = candidate.get("skills", [])

    if not skills:
        return 0.0

    skill_scores: List[float] = []

    for skill in skills:
        name: str = _normalize(skill.get("name", ""))

        # Check if this is an AI-relevant skill
        is_ai_skill = any(ai_term in name for ai_term in _AI_CORE_SKILLS)
        if not is_ai_skill:
            continue

        proficiency: str = _normalize(skill.get("proficiency", "beginner"))
        endorsements: int = max(0, int(skill.get("endorsements", 0)))
        duration: int = max(0, int(skill.get("duration_months", 0)))

        # Three independent sub-signals, each normalized to 0–1
        prof_score = _PROFICIENCY_WEIGHT.get(proficiency, 0.1)

        # Endorsement score: log scale, saturates at ~100 endorsements
        endorse_score = math.log1p(endorsements) / math.log1p(100)

        # Duration score: sqrt scale, saturates at ~60 months (5 years)
        duration_score = math.sqrt(duration) / math.sqrt(60)
        duration_score = min(1.0, duration_score)

        # Combined trust for this skill
        # Proficiency carries the most weight (it requires actual practice)
        skill_trust = (
            0.45 * prof_score +
            0.25 * endorse_score +
            0.30 * duration_score
        )
        skill_scores.append(skill_trust)

    if not skill_scores:
        return 0.0

    # Use the top-5 skills to prevent a long tail of weak skills padding the score
    top_k = sorted(skill_scores, reverse=True)[:5]

    # Average of top skills, scaled to 0–10
    avg_trust = sum(top_k) / len(top_k)
    skill_trust_score = avg_trust * 10.0

    return _clamp(skill_trust_score, 0.0, 10.0)


# ---------------------------------------------------------------------------
# Feature 5: Behavioral Multiplier  (0.50–1.15)
# ---------------------------------------------------------------------------

def build_behavioral_multiplier(candidate: Dict[str, Any]) -> float:
    """
    Compute a behavioral engagement multiplier.

    Range: 0.50–1.15 (never the primary ranking signal).

    The multiplier modulates the semantic score but can never rescue a
    semantically weak candidate. A highly responsive, active AI engineer
    gets a slight boost; an unresponsive, ghost-profile candidate gets
    a slight penalty.

    Signals used:
      - recruiter_response_rate   (0–1; key engagement signal)
      - open_to_work_flag         (boolean)
      - github_activity_score     (0–100; -1 = missing → skip)
      - notice_period_days        (30–60 strong, 90 moderate, 120–150 penalty)
      - profile_completeness_score (0–100)
      - interview_completion_rate  (0–1)
      - offer_acceptance_rate      (-1 = missing → skip)

    Returns a float in [0.50, 1.15].
    """
    signals: Dict[str, Any] = candidate.get("redrob_signals", {})

    # ---- Recruiter Response Rate (most important behavioral signal) ----
    rrr: float = float(signals.get("recruiter_response_rate", 0.5))
    rrr = _clamp(rrr, 0.0, 1.0)
    # Map 0→-0.15, 0.5→0, 1.0→+0.10
    rrr_delta = (rrr - 0.5) * 0.25       # range: -0.125 to +0.125

    # ---- Open To Work ----
    open_to_work: bool = bool(signals.get("open_to_work_flag", False))
    otw_delta = 0.04 if open_to_work else 0.0

    # ---- GitHub Activity Score ----
    github_raw = signals.get("github_activity_score", -1)
    github_delta = 0.0
    if github_raw != -1 and github_raw is not None:
        github_score = _clamp(float(github_raw), 0.0, 100.0)
        # +0.05 for top (100), 0 for ~35, -0.03 for zero
        github_delta = (github_score - 35.0) / 65.0 * 0.08
        github_delta = _clamp(github_delta, -0.03, 0.05)

    # ---- Notice Period ----
    notice: int = int(signals.get("notice_period_days", 90))
    if notice <= 60:
        notice_delta = 0.05        # immediately or quickly available
    elif notice <= 90:
        notice_delta = 0.02        # standard notice
    elif notice <= 120:
        notice_delta = -0.02       # moderate inconvenience
    else:
        notice_delta = -0.05       # long notice = penalty

    # ---- Profile Completeness ----
    completeness: float = float(signals.get("profile_completeness_score", 50.0))
    completeness = _clamp(completeness, 0.0, 100.0)
    # +0.03 for perfect (100), 0 for ~50, -0.03 for empty (0)
    completeness_delta = (completeness - 50.0) / 50.0 * 0.03

    # ---- Interview Completion Rate ----
    icr: float = float(signals.get("interview_completion_rate", 0.5))
    icr = _clamp(icr, 0.0, 1.0)
    icr_delta = (icr - 0.5) * 0.06     # range: -0.03 to +0.03

    # ---- Offer Acceptance Rate (optional) ----
    oar_raw = signals.get("offer_acceptance_rate", -1)
    oar_delta = 0.0
    if oar_raw != -1 and oar_raw is not None:
        oar = _clamp(float(oar_raw), 0.0, 1.0)
        oar_delta = (oar - 0.5) * 0.04   # range: -0.02 to +0.02

    # ---- Assemble multiplier ----
    # Base = 1.0; deltas push it up or down
    multiplier = (
        1.0
        + rrr_delta
        + otw_delta
        + github_delta
        + notice_delta
        + completeness_delta
        + icr_delta
        + oar_delta
    )

    return _clamp(multiplier, 0.50, 1.15)


# ---------------------------------------------------------------------------
# Retrieval / Search Domain Constants
# ---------------------------------------------------------------------------

# Skills that indicate deep retrieval / search system expertise
_RETRIEVAL_SKILLS: frozenset = frozenset({
    "faiss",
    "pinecone",
    "qdrant",
    "weaviate",
    "milvus",
    "elasticsearch",
    "bm25",
    "information retrieval",
    "learning to rank",
    "embeddings",
    "sentence transformers",
    "opensearch",
    "solr",
    "annoy",
    "scann",
    "hnsw",
    "approximate nearest neighbor",
})

# Current/historical job titles that are deeply retrieval-focused
_RETRIEVAL_TITLE_PATTERNS: list = [
    "search engineer",
    "recommendation systems engineer",
    "ranking engineer",
    "retrieval engineer",
    "search relevance",
    "relevance engineer",
    "search scientist",
    "recsys",
]

# Keywords to scan inside career role descriptions
_RETRIEVAL_DESC_TERMS: list = [
    "ranking",
    "retrieval",
    "recommendation",
    "search relevance",
    "learning-to-rank",
    "learning to rank",
    "hybrid retrieval",
    "semantic search",
    "candidate matching",
    "vector search",
    "dense retrieval",
    "sparse retrieval",
    "bm25",
    "reranking",
    "re-ranking",
    "recall",
    "precision",
    "embedding-based",
]


# ---------------------------------------------------------------------------
# Feature 5 (new): Retrieval / Search Score  (0–15)
# ---------------------------------------------------------------------------

def build_retrieval_score(candidate: Dict[str, Any]) -> float:
    """
    Score a candidate's depth of expertise in retrieval and search systems.

    Range: 0–15 points.

    Three sub-signals, each contributing independently:

    1. Skill signal (0–5 pts)
       Counts retrieval-specific skills (FAISS, Pinecone, Qdrant, etc.)
       weighted by proficiency + endorsements. Saturates at 5 skills.

    2. Title signal (0–5 pts)
       Checks current and historical job titles for retrieval-domain roles
       (Search Engineer, Recommendation Systems Engineer, Ranking Engineer).
       Current title earns more than past titles.

    3. Description signal (0–5 pts)
       Scans career role descriptions for retrieval terminology
       (ranking, retrieval, semantic search, learning-to-rank, etc.).
       More distinct terms + more roles containing them = higher score.

    Rationale: A candidate who professionally built ranking/retrieval
    systems is uniquely valuable. This score cannot be faked by adding
    FAISS as a skill — the title and description signals ensure genuine
    professional practice is required for a high score.
    """

    # ── Sub-signal 1: Retrieval Skills ────────────────────────────────────
    skills: List[Dict[str, Any]] = candidate.get("skills", [])

    skill_pts: float = 0.0
    for skill in skills:
        name_lower = _normalize(skill.get("name", ""))
        if any(term in name_lower for term in _RETRIEVAL_SKILLS):
            proficiency = _normalize(skill.get("proficiency", "beginner"))
            endorsements = max(0, int(skill.get("endorsements", 0)))
            prof_w = _PROFICIENCY_WEIGHT.get(proficiency, 0.1)
            endorse_w = math.log1p(endorsements) / math.log1p(50)   # saturates at 50
            # Each skill contributes up to 1.0 point; 5 strong skills → max
            skill_pts += (0.6 * prof_w + 0.4 * endorse_w)
            if skill_pts >= 5.0:
                break

    skill_pts = _clamp(skill_pts, 0.0, 5.0)

    # ── Sub-signal 2: Retrieval Titles ────────────────────────────────────
    profile: Dict[str, Any] = candidate.get("profile", {})
    career_history: List[Dict[str, Any]] = candidate.get("career_history", [])

    current_title = _normalize(profile.get("current_title", ""))
    current_is_retrieval = any(pat in current_title for pat in _RETRIEVAL_TITLE_PATTERNS)
    title_pts: float = 5.0 if current_is_retrieval else 0.0

    if title_pts < 5.0:
        # Past retrieval title: partial credit, capped at 3.0 pts
        past_retrieval_count = sum(
            1 for role in career_history
            if any(pat in _normalize(role.get("title", "")) for pat in _RETRIEVAL_TITLE_PATTERNS)
        )
        title_pts = _clamp(past_retrieval_count * 1.5, 0.0, 3.0)

    title_pts = _clamp(title_pts, 0.0, 5.0)

    # ── Sub-signal 3: Description Keywords ────────────────────────────────
    desc_pts: float = 0.0
    distinct_terms_found: set = set()

    for role in career_history:
        desc = _normalize(role.get("description", ""))
        if not desc:
            continue
        role_terms = {term for term in _RETRIEVAL_DESC_TERMS if term in desc}
        if role_terms:
            distinct_terms_found |= role_terms
            # Each role that mentions retrieval work: +0.75, up to 3 roles
            desc_pts = min(desc_pts + 0.75, 3.0)

    # Bonus for breadth: many distinct retrieval terms across career
    breadth_bonus = min(len(distinct_terms_found) * 0.2, 2.0)
    desc_pts = _clamp(desc_pts + breadth_bonus, 0.0, 5.0)

    # ── Combine ───────────────────────────────────────────────────────────
    retrieval_score = skill_pts + title_pts + desc_pts
    return _clamp(retrieval_score, 0.0, 15.0)


# ---------------------------------------------------------------------------
# Final Score
# ---------------------------------------------------------------------------

def build_final_score(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute the final ranking score for a single candidate.

    Returns a dict with individual component scores for transparency:
      {
        "title_score":           float,  # 0–35
        "career_score":          float,  # 0–25
        "assessment_score":      float,  # 0–15
        "skill_trust_score":     float,  # 0–10
        "retrieval_score":       float,  # 0–15
        "semantic_score":        float,  # 0–100
        "behavioral_multiplier": float,  # 0.50–1.15
        "final_score":           float,  # 0–1 (normalized)
      }

    Score weights (sum to 100):
      title_score       35   – professional AI title (strongest signal)
      career_score      25   – career trajectory weighted by tenure
      retrieval_score   15   – retrieval/search/ranking domain depth
      assessment_score  15   – objective Redrob AI assessment results
      skill_trust_score 10   – proficiency × endorsements × duration

    Formula:
      semantic_score  = title + career + retrieval + assessment + skill  (0–100)
      raw_final       = semantic_score × behavioral_multiplier            (0–115)
      final_score     = raw_final / 115.0                                 (0–1)

    Behavioral score is a *multiplier*. It cannot rescue a near-zero
    semantic score (0 × 1.15 = 0). It provides at most ±15% adjustment.
    """
    title_score       = build_title_score(candidate)
    career_score      = build_career_score(candidate)
    assessment_score  = build_assessment_score(candidate)
    skill_trust_score = build_skill_trust_score(candidate)
    retrieval_score   = build_retrieval_score(candidate)

    # Scale each component to its declared max before summing
    # title_score is already 0–40 raw; scale to 0–35
    title_scaled       = title_score       * (35.0 / 40.0)
    # career_score is already 0–30 raw; scale to 0–25
    career_scaled      = career_score      * (25.0 / 30.0)
    # assessment_score is already 0–20 raw; scale to 0–15
    assessment_scaled  = assessment_score  * (15.0 / 20.0)
    # skill_trust_score is already 0–10; no scaling needed
    skill_scaled       = skill_trust_score
    # retrieval_score is already 0–15; no scaling needed
    retrieval_scaled   = retrieval_score

    semantic_score = (
        title_scaled
        + career_scaled
        + assessment_scaled
        + skill_scaled
        + retrieval_scaled
    )
    semantic_score = _clamp(semantic_score, 0.0, 100.0)

    behavioral_multiplier = build_behavioral_multiplier(candidate)

    raw_final   = semantic_score * behavioral_multiplier   # 0–115
    final_score = raw_final / 115.0                        # normalized to 0–1
    final_score = _clamp(final_score, 0.0, 1.0)

    return {
        "title_score":           round(title_scaled,       4),
        "career_score":          round(career_scaled,      4),
        "assessment_score":      round(assessment_scaled,  4),
        "skill_trust_score":     round(skill_scaled,       4),
        "retrieval_score":       round(retrieval_scaled,   4),
        "semantic_score":        round(semantic_score,     4),
        "behavioral_multiplier": round(behavioral_multiplier, 4),
        "final_score":           round(final_score, 6),
    }


# ---------------------------------------------------------------------------
# Reasoning string builder
# ---------------------------------------------------------------------------

def build_reasoning(candidate: Dict[str, Any], scores: Dict[str, float]) -> str:
    """
    Build a concise, human-readable reasoning string for the submission CSV.

    Format mirrors the sample_submission.csv style:
      "[Title] with [N] yrs; [M] AI career roles; assess=[score]; behavioral=[mult]"
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career_history = candidate.get("career_history", [])

    current_title = profile.get("current_title", "Unknown")
    years_exp = profile.get("years_of_experience", 0)

    # Count career positions that were AI roles
    ai_career_roles = sum(
        1 for role in career_history
        if any(
            _match_tier(role.get("title", ""), tier)
            for tier, pts in _CAREER_TITLE_POINTS
            if pts > 0
        )
    )

    assess = scores.get("assessment_score", 0.0)
    mult = scores.get("behavioral_multiplier", 1.0)
    rrr = signals.get("recruiter_response_rate", 0.0)

    return (
        f"{current_title} | {years_exp:.1f} yrs exp | "
        f"{ai_career_roles} AI role(s) in career | "
        f"assess={assess:.1f} | "
        f"resp_rate={rrr:.2f} | "
        f"mult={mult:.3f}"
    )

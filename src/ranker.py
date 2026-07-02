"""
src/ranker.py
-------------
Core ranking engine for the Redrob Senior AI Engineer discovery challenge.

Responsibilities:
  1. Accept an iterable of raw candidate dicts (from JSONL/JSON).
  2. Score each candidate via src.features.build_final_score.
  3. Return the top-N ranked candidates with metadata.
  4. Guarantee deterministic tie-breaking (candidate_id ascending).
  5. Stay well within the 5-minute CPU runtime for 100 k candidates.

Performance notes:
  - All feature functions are pure Python with no heavy imports.
  - No embeddings, no model inference, no API calls.
  - Processing 100 k candidates takes ~20–40 seconds on a modern CPU.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List

from src.features import build_final_score, build_reasoning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Candidate = Dict[str, Any]
RankedCandidate = Dict[str, Any]   # scored candidate with rank metadata


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

class AIEngineerRanker:
    """
    Ranks candidates by their Senior AI Engineer fit score.

    Usage:
        ranker = AIEngineerRanker(top_n=100)
        results = ranker.rank(candidates)
    """

    def __init__(
        self,
        top_n: int = 100,
        log_interval: int = 10_000,
    ) -> None:
        """
        Parameters
        ----------
        top_n : int
            Number of top candidates to return. Default 100.
        log_interval : int
            Log progress every N candidates. Default 10 000.
        """
        self.top_n = top_n
        self.log_interval = log_interval

    def score_candidate(self, candidate: Candidate) -> RankedCandidate:
        """
        Score a single candidate and return an enriched dict.

        Returns the original candidate dict merged with:
          - all component scores from build_final_score
          - a human-readable 'reasoning' string

        Never raises: exceptions from malformed candidates are caught
        and the candidate receives final_score=0.0.
        """
        candidate_id: str = candidate.get("candidate_id", "UNKNOWN")
        try:
            scores = build_final_score(candidate)
            reasoning = build_reasoning(candidate, scores)
        except Exception as exc:                         # noqa: BLE001
            logger.warning(
                "Scoring failed for %s: %s – assigning score=0.0",
                candidate_id,
                exc,
            )
            scores = {
                "title_score": 0.0,
                "career_score": 0.0,
                "assessment_score": 0.0,
                "skill_trust_score": 0.0,
                "semantic_score": 0.0,
                "behavioral_multiplier": 1.0,
                "final_score": 0.0,
            }
            reasoning = "Scoring error – defaulted to 0.0"

        return {
            "candidate_id": candidate_id,
            **scores,
            "reasoning": reasoning,
        }

    def rank(self, candidates: Iterable[Candidate]) -> List[RankedCandidate]:
        """
        Score all candidates and return the top-N sorted by final_score.

        Tie-breaking rule (per challenge spec):
          Equal scores → candidate_id ascending (lexicographic).

        Parameters
        ----------
        candidates : iterable of candidate dicts

        Returns
        -------
        List of top-N RankedCandidate dicts, each containing:
            candidate_id, rank, score (=final_score), reasoning,
            plus all component scores for analysis.
        """
        logger.info("Starting ranking pipeline...")
        scored: List[RankedCandidate] = []

        for i, candidate in enumerate(candidates, start=1):
            result = self.score_candidate(candidate)
            scored.append(result)

            if i % self.log_interval == 0:
                logger.info("  Processed %d candidates...", i)

        logger.info("Scored %d candidates total.", len(scored))

        # Sort: primary = final_score descending; secondary = candidate_id ascending
        scored.sort(
            key=lambda r: (-r["final_score"], r["candidate_id"])
        )

        top_candidates = scored[: self.top_n]

        # Assign ranks 1–N
        for rank_idx, result in enumerate(top_candidates, start=1):
            result["rank"] = rank_idx

        logger.info(
            "Top-%d selected. Score range: %.4f – %.4f",
            self.top_n,
            top_candidates[-1]["final_score"] if top_candidates else 0.0,
            top_candidates[0]["final_score"] if top_candidates else 0.0,
        )

        return top_candidates


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: Iterable[Candidate],
    top_n: int = 100,
    log_interval: int = 10_000,
) -> List[RankedCandidate]:
    """
    Convenience wrapper around AIEngineerRanker.rank().

    Parameters
    ----------
    candidates : iterable of raw candidate dicts
    top_n : int
        Number of top candidates to return.
    log_interval : int
        Progress log frequency.

    Returns
    -------
    List of top-N scored, ranked candidate dicts.
    """
    ranker = AIEngineerRanker(top_n=top_n, log_interval=log_interval)
    return ranker.rank(candidates)

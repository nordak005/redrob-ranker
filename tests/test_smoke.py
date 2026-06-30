"""Smoke test – run against the 100-candidate sample to validate feature logic."""
import json
import sys
from pathlib import Path

# Ensure project root on path (works regardless of CWD when invoked via pytest)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import build_final_score, build_reasoning

SAMPLE_PATH = _PROJECT_ROOT / "data" / "sample" / "sample_candidates.json"

with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
    candidates = json.load(f)

print(f"Loaded {len(candidates)} sample candidates\n")

results = []
for c in candidates:
    scores = build_final_score(c)
    scores["candidate_id"] = c["candidate_id"]
    scores["current_title"] = c.get("profile", {}).get("current_title", "?")
    scores["reasoning"] = build_reasoning(c, scores)
    results.append(scores)

results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

header = f"{'Rank':<5} {'ID':<14} {'Final':<7} {'Sem':>5} {'Title':>6} {'Career':>7} {'Assess':>7} {'SkTrs':>6} {'Mult':>6}  Current Title"
print(header)
print("-" * len(header))
for i, r in enumerate(results[:20], 1):
    print(
        f"{i:<5} {r['candidate_id']:<14} {r['final_score']:<7.4f}"
        f" {r['semantic_score']:>5.1f}"
        f" {r['title_score']:>6.1f}"
        f" {r['career_score']:>7.2f}"
        f" {r['assessment_score']:>7.2f}"
        f" {r['skill_trust_score']:>6.2f}"
        f" {r['behavioral_multiplier']:>6.3f}"
        f"  {r['current_title']}"
    )

print("\n--- Bottom 5 ---")
for r in results[-5:]:
    print(f"  {r['candidate_id']}: {r['final_score']:.4f} | {r['current_title']}")

# Sanity checks
print("\n--- Sanity Checks ---")
top = results[0]
bottom = results[-1]
assert top["final_score"] >= bottom["final_score"], "Top should have higher score than bottom"

# All scores should be in valid range
for r in results:
    assert 0.0 <= r["final_score"] <= 1.0, f"final_score out of range: {r}"
    assert 0.5 <= r["behavioral_multiplier"] <= 1.15, f"multiplier out of range: {r}"

print("All sanity checks PASSED.")

# Check that non-AI current titles score low on title_score
non_ai_titles = [r for r in results if r["current_title"] in (
    "Project Manager", "Marketing Manager", "HR Manager",
    "Business Analyst", "Accountant", "Sales Executive"
)]
if non_ai_titles:
    max_non_ai_title_score = max(r["title_score"] for r in non_ai_titles)
    print(f"Max title_score for non-AI current titles: {max_non_ai_title_score:.2f} (should be 0.0)")
    assert max_non_ai_title_score == 0.0, "Non-AI current titles must score 0 on title_score"
    print("Non-AI title_score check PASSED.")

print("\nSmoke test complete.")

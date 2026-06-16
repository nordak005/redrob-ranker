#!/usr/bin/env python3
"""
scripts/export_top100_profiles.py
----------------------------------
Reads the ranked top-100 from debug_scores.csv, looks up each candidate
in the full dataset, and writes outputs/top100.csv with complete details.

Columns = rank + score columns + all 47 profile detail columns.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEBUG_CSV   = _ROOT / "outputs" / "debug_scores.csv"
DATASET_GZ  = _ROOT / "data" / "raw" / "candidates.jsonl.gz"
OUTPUT_CSV  = _ROOT / "outputs" / "top100.csv"

_AI_SKILL_TERMS = {
    "pytorch", "tensorflow", "jax", "keras",
    "mlflow", "kubeflow", "metaflow", "bentoml", "ray", "feast",
    "faiss", "pinecone", "milvus", "weaviate", "qdrant", "chroma",
    "transformers", "hugging face", "langchain", "llamaindex",
    "fine-tuning llms", "rlhf", "peft", "lora", "qlora",
    "prompt engineering", "nlp",
    "computer vision", "object detection", "image classification",
    "scikit-learn", "xgboost", "lightgbm", "catboost",
    "feature engineering", "machine learning", "deep learning",
    "recommendation systems", "collaborative filtering", "search ranking",
    "weights & biases", "comet ml", "triton inference server", "onnx",
    "data science", "statistical modeling", "speech recognition",
    "text to speech", "tts", "gans", "diffusion models",
    "openai", "opensearch", "yolo", "opencv", "airflow", "spark",
}

def _pipe(items: list) -> str:
    return " | ".join(str(x) for x in items) if items else ""

def _is_ai_skill(name: str) -> bool:
    n = name.strip().lower()
    return any(term in n for term in _AI_SKILL_TERMS)

def flatten_candidate(c: dict, rank_row: dict) -> dict:
    profile   = c.get("profile", {})
    career    = c.get("career_history", [])
    education = c.get("education", [])
    skills    = c.get("skills", [])
    certs     = c.get("certifications", [])
    signals   = c.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})

    career_sorted = sorted(career, key=lambda r: r.get("start_date", ""), reverse=True)

    skills_sorted = sorted(skills, key=lambda s: (
        {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(s.get("proficiency", "beginner"), 0),
        s.get("endorsements", 0)
    ), reverse=True)

    ai_skills = [s["name"] for s in skills if _is_ai_skill(s.get("name", ""))]
    salary    = signals.get("expected_salary_range_inr_lpa", {})

    return {
        # ── Ranking columns first ──────────────────────────────────────────
        "rank":                      rank_row.get("rank", ""),
        "final_score":               rank_row.get("final_score", ""),
        "semantic_score":            rank_row.get("semantic_score", ""),
        "title_score":               rank_row.get("title_score", ""),
        "career_score":              rank_row.get("career_score", ""),
        "assessment_score":          rank_row.get("assessment_score", ""),
        "skill_trust_score":         rank_row.get("skill_trust_score", ""),
        "behavioral_multiplier":     rank_row.get("behavioral_multiplier", ""),
        "ranker_reasoning":          rank_row.get("reasoning", ""),
        # ── Identity ──────────────────────────────────────────────────────
        "candidate_id":              c.get("candidate_id", ""),
        "name":                      profile.get("anonymized_name", ""),
        "headline":                  profile.get("headline", ""),
        "summary":                   profile.get("summary", "").replace("\n", " ").strip(),
        "location":                  profile.get("location", ""),
        "country":                   profile.get("country", ""),
        "years_of_experience":       profile.get("years_of_experience", ""),
        # ── Current role ──────────────────────────────────────────────────
        "current_title":             profile.get("current_title", ""),
        "current_company":           profile.get("current_company", ""),
        "current_company_size":      profile.get("current_company_size", ""),
        "current_industry":          profile.get("current_industry", ""),
        # ── Career history ────────────────────────────────────────────────
        "career_titles":             _pipe([r.get("title", "") for r in career_sorted]),
        "career_companies":          _pipe([r.get("company", "") for r in career_sorted]),
        "career_duration_months":    _pipe([str(r.get("duration_months", 0)) for r in career_sorted]),
        "career_industries":         _pipe(list({r.get("industry", "") for r in career_sorted if r.get("industry")})),
        "total_career_roles":        len(career),
        "career_descriptions":       _pipe([r.get("description", "")[:200] for r in career_sorted]),
        # ── Education ─────────────────────────────────────────────────────
        "education_degrees":         _pipe([e.get("degree", "") for e in education]),
        "education_fields":          _pipe([e.get("field_of_study", "") for e in education]),
        "education_institutions":    _pipe([e.get("institution", "") for e in education]),
        "education_tiers":           _pipe([e.get("tier", "") for e in education]),
        # ── Skills ────────────────────────────────────────────────────────
        "top_skills":                _pipe([
            f"{s['name']} ({s.get('proficiency','?')}, {s.get('endorsements',0)} endorse, {s.get('duration_months',0)}mo)"
            for s in skills_sorted[:15]
        ]),
        "ai_skills":                 _pipe(ai_skills),
        "ai_skill_count":            len(ai_skills),
        "total_skill_count":         len(skills),
        # ── Certifications ────────────────────────────────────────────────
        "certifications":            _pipe([
            f"{cert.get('name','')} ({cert.get('issuer','')}, {cert.get('year','')})"
            for cert in certs
        ]),
        # ── Assessments ───────────────────────────────────────────────────
        "assessment_scores":         _pipe([f"{k}: {v:.1f}" for k, v in sorted(assessments.items())]),
        "assessment_count":          len(assessments),
        # ── Platform signals ──────────────────────────────────────────────
        "profile_completeness_score": signals.get("profile_completeness_score", ""),
        "open_to_work":              signals.get("open_to_work_flag", ""),
        "github_activity_score":     signals.get("github_activity_score", ""),
        "recruiter_response_rate":   signals.get("recruiter_response_rate", ""),
        "notice_period_days":        signals.get("notice_period_days", ""),
        "preferred_work_mode":       signals.get("preferred_work_mode", ""),
        "willing_to_relocate":       signals.get("willing_to_relocate", ""),
        "expected_salary_lpa":       (f"{salary.get('min','')}–{salary.get('max','')} LPA" if salary else ""),
        "interview_completion_rate": signals.get("interview_completion_rate", ""),
        "offer_acceptance_rate":     signals.get("offer_acceptance_rate", ""),
        "connection_count":          signals.get("connection_count", ""),
        "endorsements_received":     signals.get("endorsements_received", ""),
        "verified_email":            signals.get("verified_email", ""),
        "verified_phone":            signals.get("verified_phone", ""),
        "linkedin_connected":        signals.get("linkedin_connected", ""),
        "last_active_date":          signals.get("last_active_date", ""),
        "signup_date":               signals.get("signup_date", ""),
        "profile_views_30d":         signals.get("profile_views_received_30d", ""),
        "saved_by_recruiters_30d":   signals.get("saved_by_recruiters_30d", ""),
        "search_appearance_30d":     signals.get("search_appearance_30d", ""),
        "avg_response_time_hours":   signals.get("avg_response_time_hours", ""),
    }


def main() -> None:
    # Step 1: Load top-100 ranked rows from debug_scores.csv
    print(f"Loading ranked top-100 from: {DEBUG_CSV}")
    ranked_rows: dict[str, dict] = {}
    with open(DEBUG_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row.get("candidate_id", "").strip()
            if cid:
                ranked_rows[cid] = row
    print(f"  {len(ranked_rows)} ranked candidates loaded.")

    # Step 2: Stream dataset and collect matching candidates
    print(f"Streaming full dataset from: {DATASET_GZ}")
    matched: dict[str, dict] = {}
    total_read = 0
    with gzip.open(DATASET_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_read += 1
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = candidate.get("candidate_id", "")
            if cid in ranked_rows:
                matched[cid] = candidate
            if len(matched) == len(ranked_rows):
                print(f"  All {len(ranked_rows)} candidates found after reading {total_read:,} records.")
                break

    print(f"  Matched {len(matched)} / {len(ranked_rows)} candidates.")

    # Step 3: Flatten and write, in rank order
    print(f"Writing to: {OUTPUT_CSV}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Sort by rank (integer)
    ordered_ids = sorted(ranked_rows.keys(), key=lambda cid: int(ranked_rows[cid].get("rank", 9999)))

    rows = []
    for cid in ordered_ids:
        if cid in matched:
            rows.append(flatten_candidate(matched[cid], ranked_rows[cid]))

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! {OUTPUT_CSV.resolve()}")
    print(f"  Rows: {len(rows)}  |  Columns: {len(fieldnames)}")
    print()
    print(f"{'Rank':<5} {'Score':<8} {'Name':<22} {'Current Title':<35} {'Company'}")
    print("-" * 100)
    for r in rows[:10]:
        print(f"{r['rank']:<5} {r['final_score']:<8} {r['name']:<22} {r['current_title']:<35} {r['current_company']}")
    print("  ...")
    for r in rows[-3:]:
        print(f"{r['rank']:<5} {r['final_score']:<8} {r['name']:<22} {r['current_title']:<35} {r['current_company']}")


if __name__ == "__main__":
    main()

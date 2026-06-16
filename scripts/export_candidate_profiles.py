#!/usr/bin/env python3
"""
scripts/export_candidate_profiles.py
--------------------------------------
Reads the ai_candidate_pool.csv (which contains only candidate_ids),
looks them up in the full 100k dataset, and writes a rich CSV with
complete candidate profile details.

Output: outputs/ai_candidate_profiles.csv

Columns:
  candidate_id, name, headline, summary, location, country,
  years_of_experience, current_title, current_company,
  current_company_size, current_industry,
  career_titles (pipe-separated), career_companies (pipe-separated),
  career_durations_months (pipe-separated),
  education_degrees, education_fields, education_institutions, education_tiers,
  top_skills (pipe-separated with proficiency),
  skill_count, ai_skill_count,
  certifications,
  assessment_scores (key:value pairs),
  profile_completeness_score, open_to_work, github_activity_score,
  recruiter_response_rate, notice_period_days,
  preferred_work_mode, willing_to_relocate,
  verified_email, verified_phone, linkedin_connected,
  last_active_date, connection_count
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

AI_POOL_CSV   = _ROOT / "outputs" / "ai_candidate_pool.csv"
CANDIDATES_GZ = _ROOT / "data" / "raw" / "candidates.jsonl.gz"
OUTPUT_CSV    = _ROOT / "outputs" / "ai_candidate_profiles.csv"

# AI skill keywords for counting
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
    "weights & biases", "comet ml",
    "triton inference server", "onnx",
    "data science", "statistical modeling",
    "speech recognition", "text to speech", "tts",
    "gans", "diffusion models",
    "openai", "openSearch", "yolo", "opencv",
    "airflow", "spark", "dbt",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipe(items: list) -> str:
    """Join a list with ' | ' separator, safe for CSV."""
    return " | ".join(str(x) for x in items) if items else ""


def _is_ai_skill(name: str) -> bool:
    n = name.strip().lower()
    return any(term in n for term in _AI_SKILL_TERMS)


def flatten_candidate(c: dict) -> dict:
    """Flatten a full candidate JSON into a single dict row for CSV."""
    profile      = c.get("profile", {})
    career       = c.get("career_history", [])
    education    = c.get("education", [])
    skills       = c.get("skills", [])
    certs        = c.get("certifications", [])
    signals      = c.get("redrob_signals", {})
    assessments  = signals.get("skill_assessment_scores", {})

    # --- Career ---
    career_sorted = sorted(career, key=lambda r: r.get("start_date", ""), reverse=True)
    career_titles      = [r.get("title", "") for r in career_sorted]
    career_companies   = [r.get("company", "") for r in career_sorted]
    career_durations   = [str(r.get("duration_months", 0)) for r in career_sorted]
    career_industries  = list({r.get("industry", "") for r in career_sorted if r.get("industry")})

    # --- Education ---
    edu_degrees      = [e.get("degree", "") for e in education]
    edu_fields       = [e.get("field_of_study", "") for e in education]
    edu_institutions = [e.get("institution", "") for e in education]
    edu_tiers        = [e.get("tier", "") for e in education]

    # --- Skills ---
    skills_sorted = sorted(skills, key=lambda s: (
        {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(s.get("proficiency", "beginner"), 0),
        s.get("endorsements", 0)
    ), reverse=True)

    top_skills_str = _pipe([
        f"{s['name']} ({s.get('proficiency','?')}, {s.get('endorsements',0)} endorse, {s.get('duration_months',0)}mo)"
        for s in skills_sorted[:15]
    ])
    ai_skills = [s["name"] for s in skills if _is_ai_skill(s.get("name", ""))]
    ai_skills_str = _pipe(ai_skills)

    # --- Assessments ---
    assess_str = _pipe([f"{k}: {v:.1f}" for k, v in sorted(assessments.items())])

    # --- Certifications ---
    certs_str = _pipe([f"{cert.get('name','')} ({cert.get('issuer','')}, {cert.get('year','')})" for cert in certs])

    # --- Salary ---
    salary = signals.get("expected_salary_range_inr_lpa", {})
    salary_str = f"{salary.get('min', '')}–{salary.get('max', '')} LPA" if salary else ""

    return {
        "candidate_id":             c.get("candidate_id", ""),
        "name":                     profile.get("anonymized_name", ""),
        "headline":                 profile.get("headline", ""),
        "summary":                  profile.get("summary", "").replace("\n", " ").strip(),
        "location":                 profile.get("location", ""),
        "country":                  profile.get("country", ""),
        "years_of_experience":      profile.get("years_of_experience", ""),
        "current_title":            profile.get("current_title", ""),
        "current_company":          profile.get("current_company", ""),
        "current_company_size":     profile.get("current_company_size", ""),
        "current_industry":         profile.get("current_industry", ""),
        # Career
        "career_titles":            _pipe(career_titles),
        "career_companies":         _pipe(career_companies),
        "career_duration_months":   _pipe(career_durations),
        "career_industries":        _pipe(career_industries),
        "total_career_roles":       len(career),
        # Education
        "education_degrees":        _pipe(edu_degrees),
        "education_fields":         _pipe(edu_fields),
        "education_institutions":   _pipe(edu_institutions),
        "education_tiers":          _pipe(edu_tiers),
        # Skills
        "top_skills":               top_skills_str,
        "total_skill_count":        len(skills),
        "ai_skill_count":           len(ai_skills),
        "ai_skills":                ai_skills_str,
        # Certifications
        "certifications":           certs_str,
        # Assessments
        "assessment_scores":        assess_str,
        "assessment_count":         len(assessments),
        # Platform signals
        "profile_completeness_score": signals.get("profile_completeness_score", ""),
        "open_to_work":             signals.get("open_to_work_flag", ""),
        "github_activity_score":    signals.get("github_activity_score", ""),
        "recruiter_response_rate":  signals.get("recruiter_response_rate", ""),
        "notice_period_days":       signals.get("notice_period_days", ""),
        "preferred_work_mode":      signals.get("preferred_work_mode", ""),
        "willing_to_relocate":      signals.get("willing_to_relocate", ""),
        "expected_salary_lpa":      salary_str,
        "interview_completion_rate": signals.get("interview_completion_rate", ""),
        "offer_acceptance_rate":    signals.get("offer_acceptance_rate", ""),
        "connection_count":         signals.get("connection_count", ""),
        "endorsements_received":    signals.get("endorsements_received", ""),
        "verified_email":           signals.get("verified_email", ""),
        "verified_phone":           signals.get("verified_phone", ""),
        "linkedin_connected":       signals.get("linkedin_connected", ""),
        "last_active_date":         signals.get("last_active_date", ""),
        "signup_date":              signals.get("signup_date", ""),
        "profile_views_30d":        signals.get("profile_views_received_30d", ""),
        "saved_by_recruiters_30d":  signals.get("saved_by_recruiters_30d", ""),
        "search_appearance_30d":    signals.get("search_appearance_30d", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Step 1: Load candidate IDs from ai_candidate_pool.csv
    print(f"Reading candidate pool from: {AI_POOL_CSV}")
    if not AI_POOL_CSV.exists():
        print(f"ERROR: {AI_POOL_CSV} not found.")
        sys.exit(1)

    with open(AI_POOL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Handle both with-header and header-less files
        first_field = reader.fieldnames[0] if reader.fieldnames else "candidate_id"
        target_ids = set()
        for row in reader:
            cid = row.get("candidate_id") or row.get(first_field, "")
            if cid.strip():
                target_ids.add(cid.strip())

    # If the file has no header and IDs start from line 1
    if not target_ids:
        with open(AI_POOL_CSV, "r", encoding="utf-8") as f:
            for line in f:
                cid = line.strip()
                if cid and cid != "candidate_id":
                    target_ids.add(cid)

    print(f"  Found {len(target_ids)} candidate IDs in pool.")

    # Step 2: Stream through full dataset and collect matching candidates
    print(f"Streaming candidates from: {CANDIDATES_GZ}")
    matched: dict[str, dict] = {}
    total_read = 0

    with gzip.open(CANDIDATES_GZ, "rt", encoding="utf-8") as f:
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
            if cid in target_ids:
                matched[cid] = candidate

            if total_read % 10_000 == 0:
                print(f"  Read {total_read:,} candidates, matched {len(matched):,}...")

            # Early exit if we've found all
            if len(matched) == len(target_ids):
                print(f"  All {len(target_ids)} candidates found. Stopping early.")
                break

    print(f"Total read: {total_read:,} | Matched: {len(matched):,}")

    if not matched:
        print("ERROR: No candidates matched. Check that candidate_ids align with the dataset.")
        sys.exit(1)

    # Step 3: Flatten and write output CSV
    print(f"Writing profiles to: {OUTPUT_CSV}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid in target_ids:
        if cid in matched:
            rows.append(flatten_candidate(matched[cid]))
        # Skip IDs not found in dataset (shouldn't happen but handle gracefully)

    # Sort by candidate_id for reproducibility
    rows.sort(key=lambda r: r["candidate_id"])

    if not rows:
        print("ERROR: No rows to write.")
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Wrote {len(rows)} candidate profiles.")
    print(f"Output: {OUTPUT_CSV.resolve()}")
    print(f"Columns: {len(fieldnames)}")

    # Quick stats
    ai_titles = sum(1 for r in rows if any(
        t in r["current_title"].lower()
        for t in ["ml engineer", "ai engineer", "nlp engineer", "data scientist",
                  "machine learning", "applied ml", "search engineer", "recommendation"]
    ))
    print(f"\nQuick stats:")
    print(f"  Candidates with AI current title : {ai_titles}")
    print(f"  Candidates with AI skills        : {sum(1 for r in rows if int(r['ai_skill_count']) > 0)}")
    print(f"  Candidates with assessments      : {sum(1 for r in rows if int(r['assessment_count']) > 0)}")
    print(f"  Open to work                     : {sum(1 for r in rows if str(r['open_to_work']) == 'True')}")


if __name__ == "__main__":
    main()

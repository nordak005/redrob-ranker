import gzip
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

def get_nested_depth(d: Any) -> int:
    """
    Returns the maximum depth of a nested dictionary/list.
    """
    if isinstance(d, dict):
        if not d:
            return 1
        return 1 + max(get_nested_depth(v) for v in d.values())
    elif isinstance(d, list):
        if not d:
            return 1
        return 1 + max(get_nested_depth(item) for item in d)
    return 0


def collect_field_stats(
    data: Any, path: str, stats: Dict[str, Dict[str, Any]], example_limit: int = 3
) -> None:
    """
    Recursively profiles the keys, types, and values of a nested data structure.
    """
    if data is None:
        return

    # Initialize stats for this path
    if path not in stats:
        stats[path] = {
            "count": 0,
            "types": set(),
            "examples": [],
            "depth": path.count(".") + path.count("["),
        }

    stats[path]["count"] += 1
    t_name = type(data).__name__
    stats[path]["types"].add(t_name)

    if isinstance(data, dict):
        # Add dict value as an example if it's small, otherwise we recurse
        for k, v in data.items():
            sub_path = f"{path}.{k}" if path else k
            collect_field_stats(v, sub_path, stats, example_limit)
    elif isinstance(data, list):
        # We collect stats for the list itself, and recurse on elements
        # To avoid path explosion, we use [*] to denote array items
        for item in data:
            sub_path = f"{path}[*]"
            collect_field_stats(item, sub_path, stats, example_limit)
    else:
        # Scalar value
        if len(stats[path]["examples"]) < example_limit:
            if data not in stats[path]["examples"]:
                stats[path]["examples"].append(data)


def analyze_candidates(
    filepath: str, schema_path: str
) -> Tuple[int, Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Reads the candidates file and profiles fields, structures, values, and anomalies.
    """
    stats: Dict[str, Dict[str, Any]] = {}
    total_candidates = 0

    # Lists for specific field profiling
    titles: List[str] = []
    skills: List[str] = []
    years_exp: List[float] = []
    locations: List[str] = []
    career_lens: List[int] = []
    skills_lens: List[int] = []

    # Honeypot checks
    honeypot_anomalies: List[Dict[str, Any]] = []

    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_candidates += 1
            cand = json.loads(line)

            # 1. Profile general fields
            collect_field_stats(cand, "", stats)

            # 2. Extract profile fields dynamically
            profile = cand.get("profile", {})
            
            # Years of experience distribution
            y_exp = profile.get("years_of_experience")
            if isinstance(y_exp, (int, float)):
                years_exp.append(float(y_exp))
            
            # Location distribution
            loc = profile.get("location")
            if loc:
                locations.append(str(loc))

            # Current title
            curr_title = profile.get("current_title")
            if curr_title:
                titles.append(str(curr_title))

            # 3. Extract career history fields
            history = cand.get("career_history", [])
            if isinstance(history, list):
                career_lens.append(len(history))
                for entry in history:
                    t = entry.get("title")
                    if t:
                        titles.append(str(t))

            # 4. Extract skills fields
            sk = cand.get("skills", [])
            if isinstance(sk, list):
                skills_lens.append(len(sk))
                for s_entry in sk:
                    s_name = s_entry.get("name")
                    if s_name:
                        skills.append(str(s_name))

            # 5. Dynamic Honeypot / Anomaly check (Rule: Never invent candidate fields!)
            # Check for skill anomalies
            skill_anomaly = False
            expert_with_zero_duration = []
            if isinstance(sk, list):
                for s_entry in sk:
                    s_name = s_entry.get("name")
                    s_prof = s_entry.get("proficiency")
                    s_dur = s_entry.get("duration_months")
                    if s_prof == "expert" and s_dur == 0:
                        skill_anomaly = True
                        expert_with_zero_duration.append(s_name)

            # Check for experience time anomaly
            exp_anomaly = False
            total_history_months = 0
            if isinstance(history, list):
                for entry in history:
                    dur = entry.get("duration_months")
                    if isinstance(dur, int):
                        total_history_months += dur
            
            # Let's check if the candidate has years_of_experience in profile, e.g. 5,
            # but their total history months is, say, 30 years (360 months), or if years_of_experience
            # is 8 years but they spent 10 years at a single company founded recently.
            # Let's check if any company duration exceeds their years_of_experience by a wide margin,
            # or if years_of_experience is 0 but they have multiple years of history.
            if isinstance(y_exp, (int, float)):
                history_years = total_history_months / 12.0
                # If total history years is dramatically different from profile years_of_experience (e.g. diff > 10 years)
                if abs(y_exp - history_years) > 10.0:
                    exp_anomaly = True

            if skill_anomaly or exp_anomaly:
                honeypot_anomalies.append({
                    "candidate_id": cand.get("candidate_id"),
                    "skills_anomaly": skill_anomaly,
                    "expert_zero_dur_skills": expert_with_zero_duration,
                    "exp_anomaly": exp_anomaly,
                    "profile_exp": y_exp,
                    "history_exp_years": total_history_months / 12.0 if total_candidates > 0 else 0
                })

    # Summary metrics
    analysis_results = {
        "candidate_count": total_candidates,
        "titles_counter": Counter(titles),
        "skills_counter": Counter(skills),
        "years_exp": sorted(years_exp),
        "locations_counter": Counter(locations),
        "career_lens": career_lens,
        "skills_lens": skills_lens,
        "honeypots": honeypot_anomalies,
        "max_depth": get_nested_depth(samples := json.load(open(schema_path)))
    }

    return total_candidates, stats, analysis_results


def write_schema_report(
    output_path: str,
    total_candidates: int,
    stats: Dict[str, Dict[str, Any]],
    results: Dict[str, Any]
) -> None:
    """
    Formats the profiling and schema data into a clean markdown document.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Pre-calculate experience distributions
    y_exp = results["years_exp"]
    n_exp = len(y_exp)
    min_exp = y_exp[0] if n_exp > 0 else 0
    max_exp = y_exp[-1] if n_exp > 0 else 0
    mean_exp = sum(y_exp) / n_exp if n_exp > 0 else 0
    median_exp = y_exp[n_exp // 2] if n_exp > 0 else 0

    # Pre-calculate array lengths
    career_lens = results["career_lens"]
    n_career = len(career_lens)
    mean_career = sum(career_lens) / n_career if n_career > 0 else 0
    max_career = max(career_lens) if n_career > 0 else 0

    skills_lens = results["skills_lens"]
    n_skills = len(skills_lens)
    mean_skills = sum(skills_lens) / n_skills if n_skills > 0 else 0
    max_skills = max(skills_lens) if n_skills > 0 else 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Redrob Candidate Schema Exploration & Dataset Profile\n\n")
        f.write("This document profiles the 100K candidate dataset and acts as the blueprint for Phase 2 ranking.\n\n")

        f.write("## 1. Executive Summary\n")
        f.write(f"- **Total Candidates**: {total_candidates:,}\n")
        f.write(f"- **Nested Schema Max Depth**: {results['max_depth']}\n")
        f.write(f"- **Anomalous (Potential Honeypot) Records**: {len(results['honeypots'])} candidates detected with impossible profiles\n\n")

        f.write("## 2. Field Analysis & Types\n")
        f.write("| Field Path | Detected Types | Count | Null Rate (%) | Depth | Examples |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        # Sort fields by path name
        sorted_fields = sorted(stats.keys())
        for path in sorted_fields:
            meta = stats[path]
            count = meta["count"]
            null_pct = ((total_candidates - count) / total_candidates) * 100
            types_str = ", ".join(meta["types"])
            examples_str = ", ".join(repr(x) for x in meta["examples"])
            f.write(f"| `{path}` | {types_str} | {count:,} | {null_pct:.2f}% | {meta['depth']} | {examples_str[:60]} |\n")
        
        f.write("\n## 3. Experience and Career Distributions\n")
        f.write("### Years of Experience (Profile-level)\n")
        f.write(f"- **Minimum**: {min_exp:.1f} years\n")
        f.write(f"- **Maximum**: {max_exp:.1f} years\n")
        f.write(f"- **Mean**: {mean_exp:.2f} years\n")
        f.write(f"- **Median**: {median_exp:.1f} years\n\n")

        f.write("### Career History Lengths\n")
        f.write(f"- **Mean Job Entries Per Candidate**: {mean_career:.2f}\n")
        f.write(f"- **Maximum Job Entries Per Candidate**: {max_career}\n\n")

        f.write("## 4. Skills Profiling\n")
        f.write(f"- **Mean Skills Listed Per Candidate**: {mean_skills:.2f}\n")
        f.write(f"- **Maximum Skills Listed Per Candidate**: {max_skills}\n\n")
        
        f.write("### Top 20 Most Frequent Skills\n")
        f.write("| Rank | Skill Name | Occurrences | Frequency (%) |\n")
        f.write("| --- | --- | --- | --- |\n")
        for idx, (skill_name, count) in enumerate(results["skills_counter"].most_common(20), start=1):
            pct = (count / total_candidates) * 100
            f.write(f"| {idx} | {skill_name} | {count:,} | {pct:.2f}% |\n")

        f.write("\n## 5. Job Titles Profiling\n")
        f.write("### Top 20 Job Titles (Current & Historic)\n")
        f.write("| Rank | Job Title | Occurrences |\n")
        f.write("| --- | --- | --- |\n")
        for idx, (title, count) in enumerate(results["titles_counter"].most_common(20), start=1):
            f.write(f"| {idx} | {title} | {count:,} |\n")

        f.write("\n## 6. Location Distribution\n")
        f.write("### Top 20 Candidate Locations\n")
        f.write("| Rank | Location | Occurrences | Frequency (%) |\n")
        f.write("| --- | --- | --- |\n")
        for idx, (loc, count) in enumerate(results["locations_counter"].most_common(20), start=1):
            pct = (count / total_candidates) * 100
            f.write(f"| {idx} | {loc} | {count:,} | {pct:.2f}% |\n")

        f.write("\n## 7. Potential Honeypot / Trap Analysis\n")
        f.write("A honeypot is defined as a record with impossible combinations of skills/experience:\n")
        f.write("- **Expert proficiency** listed with **0 months duration** used.\n")
        f.write("- Profile **years_of_experience** is highly inconsistent with job history (difference > 10 years).\n\n")
        f.write(f"Total anomalies flagged: **{len(results['honeypots'])}** candidates.\n\n")
        
        f.write("### Sample flagged candidates:\n")
        f.write("| Candidate ID | Profile Experience | History Experience (Years) | Anomaly Reason |\n")
        f.write("| --- | --- | --- | --- |\n")
        for h in results["honeypots"][:10]:
            reason = []
            if h["skills_anomaly"]:
                reason.append(f"Expert with 0 duration in: {', '.join(h['expert_zero_dur_skills'])}")
            if h["exp_anomaly"]:
                reason.append(f"Profile experience ({h['profile_exp']}) differs from history ({h['history_exp_years']:.1f} years)")
            
            f.write(f"| {h['candidate_id']} | {h['profile_exp']} | {h['history_exp_years']:.1f} | {' & '.join(reason)} |\n")


def main() -> None:
    data_file = "data/raw/candidates.jsonl.gz"
    schema_file = "data/raw/candidate_schema.json"
    output_report = "outputs/schema_report.md"

    print("[*] Starting candidate dataset schema exploration...")
    total_candidates, stats, results = analyze_candidates(data_file, schema_file)
    
    print(f"[*] Generating report at: {output_report}")
    write_schema_report(output_report, total_candidates, stats, results)
    
    print("[+] SUCCESS: Schema exploration and dataset profiling complete.")


if __name__ == "__main__":
    main()

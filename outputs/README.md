# outputs/

Generated outputs from the ranking pipeline.

> **Note:** Large intermediate files (>1 MB) are gitignored and/or moved to `archive/`.
> Only final, meaningful outputs are committed here.

---

## Files

| File | Size | Description |
|---|---|---|
| `final_submission.csv` | ~20 KB | **← FINAL SUBMISSION** — Top-100 candidates, hybrid-ranked. 100 rows, 4 columns: `candidate_id`, `rank`, `score`, `reasoning`. |
| `submission.csv` | ~13 KB | Feature-only ranking (no embeddings). Top-100 by pure feature score. |
| `performance_report.md` | ~3.5 KB | Human-readable pipeline performance report. |
| `performance_report.json` | ~1 KB | Machine-readable pipeline metrics (JSON). |
| `debug_scores.csv` | ~18 KB | Per-component score breakdown for top-100 candidates. All 5 feature components + multiplier. |
| `ai_candidate_pool.csv` | ~16 KB | Candidate IDs for the AI-relevant candidate pool (feature-filtered). |
| `top100.csv` | ~272 KB | Full profile detail for the top-100 ranked candidates. |

---

## Gitignored Outputs

The following large files are gitignored (listed in `.gitignore`):

| File | Size | Regenerate With |
|---|---|---|
| `full_rankings.csv` | ~17 MB | `python scripts/generate_full_rankings.py` |
| `hybrid_rankings.csv` | ~20 MB | `python scripts/run_ranker.py` |
| `debug_*.csv` | varies | `python scripts/run_ranker.py` |

Large intermediate CSVs are stored in `archive/` (also gitignored):

| File | Size | Regenerate With |
|---|---|---|
| `../archive/semantic_results.csv` | ~5 MB | `python scripts/generate_embeddings.py` |
| `../archive/retrieval_candidates.csv` | ~3 MB | `python scripts/run_ranker.py` |
| `../archive/ai_candidate_profiles.csv` | ~2 MB | `python scripts/export_candidate_profiles.py` |

---

## Reproducing the Final Submission

```bash
python scripts/generate_submission.py
# Produces: outputs/final_submission.csv (~6 seconds on CPU)
```

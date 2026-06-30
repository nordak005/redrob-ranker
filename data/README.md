# data/

Runtime data for the ranking pipeline.

> **Important:** Large binary files are gitignored. See below for how to obtain them.

---

## Directory Structure

```
data/
├── embedding_metadata.json          # Metadata for precomputed embeddings
├── candidate_embeddings.npy         # ← GITIGNORED (153 MB) — precomputed 384-dim embeddings
├── candidate_ids.npy                # ← GITIGNORED (4.8 MB)  — candidate ID index
├── raw/
│   ├── candidates.jsonl.gz          # ← GITIGNORED (54 MB)   — full 100k candidate dataset
│   └── candidate_schema.json        # JSON schema definition for the dataset
└── sample/
    ├── sample_candidates.json       # ~100 candidate demo subset
    └── sample_submission.csv        # Submission format reference (10 rows)
```

---

## Obtaining the Dataset

The full dataset (`data/raw/candidates.jsonl.gz`) is not committed to git due to its size.

Place the competition-provided file at:
```
data/raw/candidates.jsonl.gz
```

---

## Generating Embeddings

Precomputed embeddings (`candidate_embeddings.npy`, `candidate_ids.npy`) are generated
once offline and loaded at runtime for fast semantic search.

```bash
# Run once — takes ~90 seconds on CPU
python scripts/generate_embeddings.py
```

Output:
- `data/candidate_embeddings.npy` — (100000, 384) float32, L2-normalized
- `data/candidate_ids.npy` — (100000,) string array
- `data/embedding_metadata.json` — generation timestamp, model name, shape

---

## Schema Reference

See `data/raw/candidate_schema.json` for the full JSON schema.

Key fields:
- `candidate_id` — unique identifier (e.g. `CAND_0000001`)
- `profile.current_title` — current job title
- `profile.years_of_experience` — self-reported years
- `career_history[]` — list of past roles
- `skills[]` — list of skills with proficiency/endorsements
- `redrob_signals` — platform behavioral signals (assessments, response rate, etc.)

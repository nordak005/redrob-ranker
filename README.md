# Redrob AI Engineer Ranker — Final Submission

> **Stage 3 Submission** · Hybrid CPU Ranking · 100k candidates · <20 seconds · No GPUs · No External APIs

---

## 1. Problem

Redrob needs to discover the top Senior AI Engineer candidates from a pool of **100,000 anonymised candidate profiles**. The challenge: identify individuals who have *professionally practiced* AI at the senior level — not just those who list AI buzzwords on their profiles.

**Job Description core requirements:**
- Retrieval, ranking, or recommendation systems experience
- Demonstrated AI career trajectory (not just skills claimed)
- Verifiable AI competency (platform assessment scores)
- Active engagement with the platform (behavioral signals)

---

## 2. Dataset

| Property | Value |
|---|---|
| Total candidates | 100,000 |
| Format | JSONL (gzip-compressed), ~54 MB |
| Fields per candidate | profile, career_history, skills, redrob_signals |
| Key signals | current_title, years_of_experience, career_history titles, skill assessments, recruiter_response_rate, notice_period_days |
| Honeypot detection | Title-tier system penalizes non-AI titles regardless of skill claims |

---

## 3. Architecture

```
Candidate Profiles (100,000)
        ↓
Feature Engineering
  ├── Title Score         (0–35 pts)  current professional title
  ├── Career Score        (0–25 pts)  career trajectory × tenure
  ├── Retrieval Score     (0–15 pts)  search/ranking domain depth
  ├── Assessment Score    (0–15 pts)  Redrob AI platform assessments
  └── Skill Trust Score   (0–10 pts)  proficiency × endorsements × duration
        ↓
Behavioral Multiplier (×0.50–1.15)
  └── recruiter_response_rate, github_activity, notice_period, ...
        ↓
Feature Ranker  →  feature_score = semantic × multiplier / 115
        ↓
Precomputed Embeddings  (all-MiniLM-L6-v2, 384-dim, float32)
  ├── Generated offline once for all candidates
  └── Loaded into memory (singleton store)
        ↓
Semantic Search
  └── Encode JD only → Fast matrix multiply → 0-100 embedding_score
        ↓
Hybrid Scoring
  └── hybrid_score = 0.85 × feature_score + 0.15 × embedding_score
        ↓
Top 100 Candidates  →  outputs/final_submission.csv
```

---

## 4. Feature Engineering

Five deterministic scoring components, all pure Python, no model inference:

### Title Score (0–35 points)
Evaluates the candidate's **current job title** against a curated 4-tier taxonomy:
- **Elite** (40 pts raw → 35 scaled): ML Engineer, Search Engineer, NLP Engineer, Applied Scientist
- **Strong** (30 pts → ~26): Data Scientist, Computer Vision Engineer, LLM Engineer
- **Moderate** (18 pts → ~16): Data Engineer, MLOps Engineer, Analytics Engineer
- **Junior** (8 pts → ~7): Junior ML Engineer, Associate AI
- **Non-technical** (0 pts): Project Manager, HR Manager — blocked regardless of skills listed

### Career Score (0–25 points)
Weighted career trajectory scoring:
- Each role scored by AI-tier value
- Weight = √(duration_months) — diminishing returns prevent one long role dominating
- Recency bonus: +1.5 if most recent role is AI-tier
- Breadth bonus: up to +2.0 for multiple distinct AI roles

### Retrieval Score (0–15 points)
Three independent sub-signals for retrieval/search/ranking domain depth:
1. **Skill signal** (0–5): FAISS, Pinecone, Qdrant, Elasticsearch, BM25, embeddings, etc.
2. **Title signal** (0–5): Search Engineer, Recommendation Systems Engineer, Ranking Engineer in current or past titles
3. **Description signal** (0–5): retrieval terminology in career role descriptions (ranking, dense retrieval, LTR, reranking, etc.)

### Assessment Score (0–15 points)
Weighted average of Redrob platform AI assessment results across 14 categories:
- Highest weight: FAISS (×1.5), Recommendation Systems (×1.5), Pinecone (×1.4)
- Lowest weight: Prompt Engineering (×0.6) — easily gamed, shallow signal

### Skill Trust Score (0–10 points)
Quality-over-quantity trust metric:
- Only AI-core skills counted (PyTorch, FAISS, transformers, scikit-learn, etc.)
- Per-skill trust = 0.45 × proficiency + 0.25 × log(endorsements) + 0.30 × √(duration)
- Top-5 skills averaged — prevents padding with weak skills

---

## 5. Semantic Search

**Model:** `all-MiniLM-L6-v2` (sentence-transformers, 384-dim, CPU, ~22 MB)

Candidate text constructed from:
- Current title + years of experience
- Career history titles and descriptions
- Top AI-relevant skills

JD embedding computed once at startup. Cosine similarity produces a 0–100 embedding score for each candidate.

**Why MiniLM over larger models:**
- Runs on CPU in < 1ms per candidate
- 384-dim is sufficient for title/career text similarity
- No GPU required — satisfies the compute constraint
- 22 MB model fits comfortably in RAM

---

## 6. Precomputed Embeddings (Offline Pipeline)

Encoding 100,000 candidates live takes 1000–2000 seconds on CPU. To achieve production-level latency (< 5s), the system uses an **offline embedding pipeline**:

1. **`scripts/generate_embeddings.py`**: Runs once offline. Encodes all candidate text into a single `(100000, 384)` float32 NumPy array (`data/candidate_embeddings.npy`).
2. **`src/embedding_store.py`**: Loads the precomputed array into memory once per process.
3. **`src/semantic_search.py`**: At query time, only the JD is encoded. Semantic similarity is computed via a highly optimized matrix-vector dot product (`embeddings @ jd_emb`).

**Result:** Semantic search latency drops from **~1500 seconds to < 3 seconds** for 100,000 candidates.

---

## 7. Hybrid Ranking

```
hybrid_score = 0.85 × feature_score_scaled + 0.15 × embedding_score
```

**Feature score** (85%) — deterministic, explainable, anti-gaming  
**Embedding score** (15%) — semantic generalization, catches non-obvious fits

The 85/15 split was chosen because:
- Feature scores encode hard professional facts (titles, tenures, assessments)
- Embeddings add recall for candidates whose profiles use different terminology
- A higher embedding weight would let keyword-only profiles game the system

**Effect on Top-500:** Embeddings introduce only 13 new candidates into the top-500 vs. pure feature ranking — confirming the feature ranker's high precision, with embeddings providing modest recall improvement.

---

## 8. Evaluation

| Metric | Value |
|---|---|
| Mean experience (top-100) | 6.45 years |
| Search/Recommendation titles (top-100) | 32% |
| Top-500 overlap: feature vs semantic | 46 candidates |
| Top-500 churn from embeddings | 13 new candidates |
| Candidate 0018499 hybrid rank | 8 |
| Candidate 0000031 hybrid rank | 2 |
| Feature rank for 0018499 | 5 (feature) → 8 (hybrid) |
| Semantic rank for 0018499 | 70 (pure embedding) |

### Case Study: CAND_0000031 (Hybrid Rank #2)
Recommendation Systems Engineer with 6 years of experience and 4 AI roles in career history. Scores maximum on retrieval-domain metrics. Achieves hybrid rank #2 despite not topping either the feature-only or embedding-only rankings individually — the hybrid score correctly identifies depth across both dimensions.

### Case Study: CAND_0018499 (Hybrid Rank #8)
Feature Rank = 5, Semantic Rank = 70. This candidate has strong structured features (title, career, assessments) but below-average embedding similarity. The 85/15 hybrid correctly keeps them in the top-10 rather than letting a mediocre embedding similarity drop them significantly. This demonstrates the hybrid's resilience against noisy embedding signals.

### Why Hybrid Works
Pure feature ranking misses candidates who describe their work differently (synonym variation). Pure semantic ranking misses candidates with strong objective credentials but atypical text. The hybrid combines both: structured professional facts dominate (85%), while semantic similarity provides a supplementary recall signal (15%).

---

## 9. Runtime

| Step | Time (100k candidates) |
|---|---|
| Load JSONL.gz | ~6 s |
| Feature scoring (all 100k) | ~18 s |
| Offline MiniLM embedding (all 100k) | ~90 s (CPU batch) — *Run once offline* |
| Load precomputed embeddings | < 1 s |
| JD Encoding + Similarity compute | < 2 s |
| Hybrid ranking + CSV write | < 1 s |
| **Total Online Runtime** | **< 30 seconds** |

**Submission generation** (`generate_submission.py` — top-100 only): **~6 seconds**

---

## 10. Reproducibility

### Environment Setup

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Single-Command Submission

```bash
python scripts/generate_submission.py
```

Produces: `outputs/final_submission.csv` (100 rows, < 10 seconds)

### Full Pipeline

```bash
# Full feature ranking (100k candidates)
python scripts/run_ranker.py \
    --input  data/raw/candidates.jsonl.gz \
    --output outputs/submission.csv \
    --validate

# Streamlit sandbox
streamlit run app.py
```

### Validation

```bash
python scripts/validate_submission.py outputs/final_submission.csv
pytest
```

### Compute Environment

| Property | Value |
|---|---|
| Platform | CPU only |
| Python | 3.11+ |
| GPU | Not used |
| Network during ranking | None (fully offline) |
| RAM requirement | ~4 GB (16 GB recommended) |

---

## 11. Future Work

1. **Fine-tune MiniLM on Redrob JDs** — domain-adapted embeddings for higher precision
2. **Learning-to-Rank** — train a LambdaMART model on recruiter feedback signals
3. **Candidate clustering** — identify diverse archetypes beyond the top-100
4. **FAISS ANN index** — approximate nearest-neighbor search for 10M+ candidate scale
5. **Score calibration** — Platt scaling to convert raw scores to calibrated probabilities
6. **Explainability layer** — per-candidate SHAP values for each scoring component
7. **Incremental updates** — streaming JSONL ingestion for real-time candidate updates

---

## File Tree

```
project_root/
├── src/                              # Core library (feature eng, ranking, embeddings)
├── scripts/
│   ├── run_ranker.py                 # Full pipeline CLI (100k candidates)
│   ├── generate_submission.py        # Final submission generator (< 10s)
│   ├── generate_embeddings.py        # Offline embedding pipeline (run once)
│   ├── benchmark_embeddings.py       # Embedding benchmark
│   ├── validate_submission.py        # Competition submission validator
│   └── ...                          # Utilities and export scripts
├── tests/
│   ├── test_validation.py            # Submission format validation tests
│   ├── test_smoke.py                 # Feature scoring smoke test (sample data)
│   └── test_hybrid.py               # Hybrid ranker timing test
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_real_data_exploration.ipynb
│   ├── 03_retrieval_signal_audit.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_final_audit.ipynb
│   └── 06_hybrid_ranking.ipynb
├── research/                         # EDA scripts (schema explorer, memory report)
├── docs/
│   ├── interview_notes.md            # Technical interview Q&A
│   ├── evaluation_report.md          # Full evaluation report
│   └── schema_report.md             # Dataset schema analysis
├── configs/
│   └── submission_metadata.yaml      # Submission metadata (fill TODO fields)
├── outputs/
│   ├── final_submission.csv          # ← FINAL SUBMISSION (100 rows, hybrid)
│   ├── submission.csv                # Feature-only ranking (100 rows)
│   └── ...                          # Performance reports, debug scores
├── data/
│   ├── raw/candidates.jsonl.gz       # 100k candidate profiles (gitignored)
│   └── sample/                       # Sample data for testing
├── archive/                          # Preserved non-production files
├── app.py                            # Streamlit sandbox
├── requirements.txt                  # Runtime dependencies
├── requirements-dev.txt              # Development dependencies
├── Dockerfile
└── README.md
```

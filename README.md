# Redrob AI Candidate Ranking System

> **Redrob Hackathon 2026 Submission**  
> *A Production-Ready, CPU-Optimized Hybrid Ranking Pipeline for Senior AI Engineers.*  
> **Offline Performance**: Semantic Search & Feature Ranking on 100,000 Candidates in **< 30 seconds** on standard CPU.  
> **Zero Network Overhead** · **No GPU Required** · **Fully Isolated Docker Containerization**.

---

## 1. Problem Statement

Modern talent acquisition platforms contain vast databases of candidate profiles. Discovering high-quality candidates for specialized roles like **Senior AI/ML Engineers** is a challenging ranking task. Recruiters are frequently overwhelmed by candidate pools containing thousands of profiles, many of which use keyword-stuffing tactics without possessing actual, verified domain depth. 

Conversely, relying solely on semantic text matches (sentence embeddings) often misses structural signals such as years of experience, platform assessment scores, historical career trajectory, and behavioral signals (e.g., recruiter response rates or notice periods).

### The Hybrid Solution
This repository implements a production-grade **Hybrid Ranker** combining:
1. **Deterministic Feature Engineering (85%)**: An explainable, rule-based feature pipeline evaluating profile completeness, professional seniority taxonomy, recency-weighted career trajectory, verified platform assessments, and credential trust metrics.
2. **Semantic Similarity Search (15%)**: A sentence-embedding similarity scoring layer based on the `all-MiniLM-L6-v2` transformer model, mapping the candidate's holistic profile text to the Job Description (JD).

---
<img width="1896" height="900" alt="image" src="https://github.com/user-attachments/assets/321bede9-e8c4-4e9a-b341-192b9c4f4f36" />


## 2. Architecture & Pipeline

The system uses a two-phase architecture to achieve low-latency inference: an **offline precomputation phase** that caches candidate embeddings into a flat matrix, and an **online execution phase** that performs fast cosine similarities and feature engineering.

### Pipeline Flow Diagram
```text
Job Description (JD)                   Candidate Profile JSONL
        │                                         │
        ▼                                         ▼
  MiniLM Encoder                         Feature Engineering
        │                                 ├── Title Score      (0-35)
        ▼                                 ├── Career Score     (0-25)
  JD Embedding [384-dim]                  ├── Retrieval Score  (0-15)
        │                                 ├── Assessment Score (0-15)
        │                                 └── Skill Trust Score(0-10)
        │                                         │
        │                                         ▼
        │                               Behavioral Multiplier [x0.50-1.15]
        │                                         │
        ▼                                         ▼
  Cosine Similarity <──[Precomputed]───► Scaled Feature Score [0-100]
  (via Dot Product)    Candidate Embs             │
        │                                         │
        ▼                                         ▼
  Semantic Score [0-100]                  Final Feature Score [0-100]
        │                                         │
        └───────────────────┬─────────────────────┘
                            │
                            ▼
                     Hybrid Scoring
          (0.85 * Feature + 0.15 * Semantic)
                            │
                            ▼
                    Sort & Tie-Breaker
                            │
                            ▼
                   Natural Reasoning
                            │
                            ▼
                  Top 100 Candidate CSV
```

---

## 3. Core Features

### 🧠 Semantic Search (Sentence Embeddings)
- **Model**: `all-MiniLM-L6-v2` (SentenceTransformer, 384-dimensional dense vectors).
- **Holistic Profiles**: Constructs candidate search texts from current titles, headline summaries, years of experience, historical titles, and core skills.
- **Optimized Similarity**: Utilizes standard L2-normalized vectors so that the cosine similarity is computed instantly via a matrix-vector dot product (`embeddings @ jd_emb`), running in under 2 seconds for 100K profiles.

### 📊 Feature Engineering (Deterministic Python)
- **Title Score (0–35 pts)**: Ranks candidate's current title against a strict AI taxonomy (Elite, Strong, Moderate, Junior). Irrelevant titles (e.g. Project Manager, HR) are automatically scored as `0.0`.
- **Career Score (0–25 pts)**: Rates career trajectory by weighting historical roles. Sub-signal scaling uses a square-root tenure dampener (`sqrt(duration_months)`) to prevent a single long tenure from dominating the score.
- **Retrieval Score (0–15 pts)**: Evaluates deep domain experience in search/ranking/recommendation systems using three criteria: skill keyword match, title keyword match, and role description keywords.
- **Assessment Score (0–15 pts)**: Calculates weighted average of Redrob platform AI assessments, prioritizing core skills (e.g., Pinecone, FAISS, Recommendation Systems) and dampening shallow skills (e.g. Prompt Engineering).
- **Skill Trust Score (0–10 pts)**: Penalizes keyword stuffing. Aggregates and averages the top 5 skills based on proficiency levels, user endorsements, and role-duration history.
- **Behavioral Multiplier (×0.50–1.15)**: Adjusts scores based on recruiter response rates, GitHub activity, notice periods, and profile completeness.

### ✍️ Natural Reasoning Generation
For each of the top 100 candidates, the ranker synthesizes a natural language rationale highlighting:
- The candidate's current title and overall years of experience.
- Verified platform assessment highlights.
- Domain fit (retrieval systems, semantic search, etc.).
- Critical behavioral signals (such as short notice periods or high responsiveness).
---
<img width="1905" height="892" alt="image" src="https://github.com/user-attachments/assets/3f9b3e78-9241-4d99-94b1-7e0818cfd3a5" />
---

### 💾 Caching & Performance Optimizations
- **Streamlit `@st.cache_resource`**: Caches the SentenceTransformer model and precomputed embedding matrix in RAM, ensuring zero reload overhead across frontend user interactions.
- **Precomputed Embeddings**: Candidate embeddings are generated once offline, removing live inference bottlenecks and reducing online execution to CPU matrix operations.

---

## 4. Repository Structure

```text
nordak005-redrob-ranker/
├── app.py                      # Main Streamlit web application & reproducibility sandbox
├── Dockerfile                  # Self-contained container setup (Offline mode configured)
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Testing & development dependencies
├── submission.yaml             # Contest metadata configuration template
├── submission_metadata.yaml    # Registered participant submission metadata
├── README.md                   # Project documentation
├── configs/                    # Environment and validation configuration
├── data/
│   ├── raw/                    # Contains raw gzipped candidates dataset (100k rows)
│   └── sample/                 # Small candidate sample dataset (~20 candidates)
├── docs/
│   ├── evaluation_report.md    # Metrics audit, case studies, and hybrid calibration logs
│   ├── schema_report.md        # Deep-dive dataset schema exploration
│   └── interview_notes.md      # Architecture and design Q&A notes
├── models/                     # Cached SentenceTransformer models folder (local inference)
├── outputs/
│   ├── final_submission.csv    # Final 100 rows CSV (ranked and validated)
│   └── hybrid_ranked_debug.csv # Full breakdown scores CSV for auditing
├── scripts/
│   ├── generate_submission.py  # Production CLI script to generate the final CSV submission
│   ├── generate_embeddings.py  # Offline script to precompute candidate embeddings
│   ├── run_ranker.py           # CLI script to execute the full pipeline on arbitrary files
│   └── validate_submission.py  # Submission CSV schema and logic validator
├── src/
│   ├── embedding_store.py      # Cache manager for precomputed candidate embeddings
│   ├── features.py             # Feature engineering functions
│   ├── hybrid_ranker.py        # Core hybrid scorer and sorting logic
│   ├── reasoning.py            # Natural language rationale builder
│   ├── semantic_search.py      # Vector similarity utilities
│   └── submission_validator.py # Internal checker verifying competition rules
└── tests/                      # pytest unit and smoke tests
```

---

## 5. Installation & Setup

### Local Python Environment
Prerequisites: Python 3.11 or 3.13 (recommended).

```powershell
# Clone the repository
git clone https://github.com/nordak005/redrob-ranker.git
cd redrob-ranker

# Create and activate virtual environment
python -m venv .venv
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
<img width="865" height="643" alt="image" src="https://github.com/user-attachments/assets/dd3b7a6d-2701-4161-9775-320812d98105" />

### Docker Container Setup
The project is packaged to run in fully isolated environments (offline) without requiring GPU acceleration.

```bash
# Build the Docker image
docker build -t redrob-ai .
```
OR
## Docker Deployment((preferred)

### Pull Docker Image

```bash
docker pull nordak005/redrob-ai:v1.1
```

### Run Locally

```bash
docker run -p 7860:7860 nordak005/redrob-ai:v1.1
```
---

## 6. Running Production Mode

Production Mode processes the complete dataset of **100,000 candidates** and generates the competition-ready submission containing the **Top 100 ranked candidates**.

### Automatic Embedding Management

The production pipeline requires a semantic embedding cache.

Simply run:

```bash
python scripts/generate_submission.py
```

If the embedding cache is already available, the submission generation starts immediately.

If the cache is missing, an interactive menu is displayed:

```
Embedding cache not found.

Choose one option:

[1] Download Official Embeddings (Recommended ⚡)
    • Fastest setup (~1 minute)
    • Downloads the official precomputed embedding package
    • Automatically extracts it into the correct location

[2] Generate Embeddings Locally
    • No download required
    • Uses the local SentenceTransformer model
    • Recommended for offline environments
    • May take longer depending on hardware

[3] Exit
```

### Generate the Competition Submission

After the embeddings are available (downloaded or generated automatically), the pipeline continues without any additional commands.

```bash
python scripts/generate_submission.py
```

Typical execution time:

- **With precomputed embeddings:** ~6 seconds
- **First-time setup (download):** ~1 minute
- **First-time setup (local generation):** Depends on hardware and dataset size

The generated submission is saved to:

```
outputs/final_submission.csv
```

---

### Running Inside Docker

```bash
docker run --rm \
    -v "${PWD}/outputs:/app/outputs" \
    nordak005/redrob-ai:v1.2
```

The Docker container follows the same workflow:

- Detects existing embeddings automatically.
- Prompts to download the official embedding package if missing.
- Falls back to local embedding generation if preferred.
- Produces the final submission in the mounted `outputs/` directory.
#### 3. Web UI Application (Local or Docker)
To launch the interactive, high-performance Streamlit interface:
```bash
# Local
streamlit run app.py

# Docker
docker run -d -p 7860:7860 --name redrob-ui -v "${PWD}/outputs:/app/outputs" redrob-ai
```
Open **http://localhost:7860** in your browser.

---

## 7. Running Sandbox Mode( Deployed on hugging face)

Sandbox Mode is designed for evaluation and reproducibility on small candidate datasets.

### Redrob Section 10.5 Compliance
Per **Redrob Submission Specification Section 10.5**, the hosted sandbox environment only needs to demonstrate reproducibility on a small sample of candidates (≤100). The full 100K evaluation is performed later inside Redrob's private Docker sandbox.
```bash
https://huggingface.co/spaces/Nordak005/ranking_optimal
```

To activate the Sandbox Demo:
1. Launch the Streamlit Web UI (locally or via Docker).
2. Toggle the **Execution Mode** radio selector from **Production** to **Sandbox Demo**.
3. The warning banner will appear: `"This mode is intended for Redrob Hackathon reproducibility (≤100 candidates)."`
4. Drop or select any candidate `.json` or `.jsonl` file.
5. Files containing **more than 100 candidates will be rejected** with a friendly validation error, blocking execution. Only files matching the size criteria are parsed and hybrid-ranked.

---

<img width="1898" height="896" alt="image" src="https://github.com/user-attachments/assets/929a61c1-70d4-476d-8963-3edcf8a114fa" />

## 8. System Performance

| Step | Scope | Hardware | Memory | Runtime |
|---|---|---|---|---|
| **Embedding Generation** | 100K Profiles | Standard CPU (8-cores) | ~2 GB | ~90 seconds |
| **Embedding Loading** | 100K Matrices | RAM (Singleton Cache) | ~150 MB | < 1 second |
| **JD Encoding & Matching**| 100K Embeddings| Standard CPU | - | < 2 seconds |
| **Feature Engineering** | 100K Profiles | Standard CPU | ~4 GB | ~18 seconds |
| **Submission Output** | Top 100 | Standard CPU | - | < 1 second |
| **Full CLI Pipeline** | Top 100 | Standard CPU (Production) | ~4 GB | **~6 seconds** |

---

## 9. Key Design Decisions

- **Why SentenceTransformer `all-MiniLM-L6-v2`?**  
  At 22 MB, it loads instantly into memory and executes cosine similarities in milliseconds on CPU, satisfying the hackathon's strict compute limits while providing high quality semantic alignment.
- **Why an 85/15 Hybrid Split?**  
  We prioritize structured, verifiable professional credentials (experience, assessments, exact titles) over raw semantic matching. This prevents candidate profiles with shallow, keyword-stuffed summaries from ranking above candidates with verified industry accomplishments.
- **Why Precompute Embeddings Offline?**  
  Encoding 100K profiles on CPU at query time would take over 20 minutes. Precomputing embeddings compresses runtime into a few seconds, making the system responsive and production-viable.
- **Why Fully Offline Containerization?**  
  All models, scripts, and precomputed embeddings are self-contained. No network calls to HuggingFace or external LLM API keys are made, satisfying security and platform-isolation constraints.

---

## 10. Future Improvements

1. **FAISS ANN Indexing**: Implement Approximate Nearest Neighbors using FAISS to scale candidate matching from 100K to 10M+ candidates in sub-millisecond range.
2. **Incremental Indexing**: Implement streaming vector updates to ingest new candidates and incrementally update the precomputed embedding matrix without full rebuilds.
3. **Distributed Scoring**: Utilize PySpark or Dask parallel workers to distribute feature scoring and matching across horizontal compute clusters for enterprise scale.
4. **Learning to Rank (LTR)**: Integrate a LambdaMART or XGBoost ranker trained on historic recruiter feedback data (accept/reject signals) to dynamically adjust feature weights.

---
## 🤖 AI-Assisted Development

Throughout the development of **RedRob AI Ranker**, OpenAI ChatGPT was used as an engineering copilot to accelerate development and improve software quality.
```bash
AI assistance was primarily used for:
- Designing and refining the system architecture
- Debugging complex implementation and deployment issues
- Optimizing Docker and Hugging Face deployment
- Reviewing code and suggesting performance improvements
- Enhancing the UI/UX and user feedback
- Preparing technical documentation and project reports
```

All architectural decisions, implementation, integration, testing, and final validation were performed by the development team. Every AI-generated suggestion was critically reviewed, adapted where necessary, and verified through local testing and production deployment before being incorporated into the final project.

**Outcome:** AI significantly reduced development and debugging time, enabling faster iteration while maintaining complete human oversight over the final solution.
---

## 12. License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 13. Acknowledgements

- **Redrob Platform**: For providing the raw candidate profiles and verified assessment scores.
- **SentenceTransformers**: For the high-performance offline MiniLM model.
- **Streamlit**: For the interactive dashboard capabilities.

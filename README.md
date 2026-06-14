<<<<<<< HEAD
# Redrob Ranker - Candidate Discovery Ranking System

This repository implements **Phase 1 (Development Environment & Data Validation)** of the candidate discovery and ranking engine. It establishes a production-grade development environment, profiles candidate schema properties recursively, estimates dataset memory consumption under constraints, and validates formatting.

No ranking model or scoring logic is implemented during this phase.

## 1. Project Directory Structure

```text
project_root/
│
├── data/
│   ├── raw/                 # Raw datasets (candidate_schema.json, candidates.jsonl.gz)
│   └── sample/              # Sample references (sample_candidates.json, sample_submission.csv)
│
├── src/
│   ├── __init__.py          # Source package initialization
│   ├── utils.py             # Reusable file loaders, timers, and helpers
│   └── submission_validator.py # Submission format constraints validator
│
├── scripts/
│   ├── setup_env.py         # Verifies python and package installations
│   ├── validate_data.py     # Validates candidates JSONL pool integrity
│   ├── explore_schema.py    # Recurses schema, top frequencies, auto-reports
│   └── dataset_memory_report.py # Estimates DataFrame and embedding memory
│
├── notebooks/
│   └── schema_exploration.ipynb # Jupyter notebook for interactive exploration
│
├── tests/
│   ├── __init__.py          # Test package initialization
│   └── test_validation.py   # Pytest unit tests for loaders & validators
│
├── outputs/                 # Stores generated reports (e.g., schema_report.md)
├── models/                  # Stores precomputed embeddings & indices (Phase 2)
│
├── Dockerfile               # Reproducibility environment recipe
├── requirements.txt         # Frozen exact dependency versions
├── README.md                # Project documentation
└── .gitignore               # Ignored caches, temporary data, and virtualenv files
```

---

## 2. Virtual Environment Setup

Since we are on Windows, follow these instructions to create and activate your environment:

### PowerShell
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

### Command Prompt (CMD)
```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate.bat
```

---

## 3. Installation

Once the virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt --no-cache-dir
```

---

## 4. Execution & Validation Steps

You can run each of the environment checks, validation tools, and profiling scripts:

### A. Environment Status Check
Verifies your Python version (requires Python 3.11+) and checks that all dependencies are installed:
```bash
python scripts/setup_env.py
```

### B. Raw Dataset Integrity Check
Validates that `candidates.jsonl.gz` exists, checks size, integrity, malformed JSON lines, and verifies fields against the JSON Schema:
```bash
python scripts/validate_data.py
```

### C. Schema Profiling & Report Auto-Generation
Performs deep dataset profiling (extracts frequencies, null percentages, arrays, top 20 skills/titles/locations, anomalies) and writes the blueprint to `outputs/schema_report.md`:
```bash
python scripts/explore_schema.py
```

### D. Memory Budget Profiling
Estimates memory usage for Pandas loading and float32 embedding matrices (for dims 384, 768, 1024, 1536) to ensure the system stays within the 16 GB constraint:
```bash
python scripts/dataset_memory_report.py
```

### E. Run Unit Tests
Executes unit tests verifying loader utilities and formatting constraints:
```bash
pytest
```
=======
# redrob-ranker
>>>>>>> a14cd9a89af904c5c3ea9bba326c50195fe14190

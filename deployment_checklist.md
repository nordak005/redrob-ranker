# Deployment Readiness Checklist

This checklist verifies that the Redrob AI Candidate Ranking System is fully portable, environment-independent, and ready for deployment to the Hackathon Sandbox.

---

## 1. Path Portability & OS-Independence
- [x] **No Drive Letters**: No absolute Windows paths (e.g. `C:\Users\...`) exist in the codebase.
- [x] **Standard Path Joining**: The codebase uses `pathlib.Path` or `os.path.join` for all file path operations. This prevents issues with backslashes (`\`) vs forward slashes (`/`).
- [x] **Project Root Anchor**: Key directories (`src/`, `data/`, `models/`, `outputs/`) are anchored dynamically to the project root relative to the running module's location:
  - `_PROJECT_ROOT = Path(__file__).resolve().parent.parent` (or similar).
- [x] **Case-Sensitivity**: All file names in imports match the filesystem exactly (crucial since Windows is case-insensitive, but Linux/Docker is case-sensitive).

---

## 2. Environment Isolation & Cache Handling
- [x] **Offline Capability**: The SentenceTransformer model folder (`models/`) is fully packaged inside the container. The loader check avoids making any network calls to HuggingFace Hub at runtime if local files exist.
- [x] **Cache Control**: Environment variable overrides are set inside `Dockerfile` to control model caches:
  - `SENTENCE_TRANSFORMERS_HOME=/app/models`
  - `HF_HUB_DISABLE_PROGRESS_BARS=1`
- [x] **Streamlit Sandbox Isolation**: Streamlit is configured to run in headless mode (`--server.headless=true`) and telemetry collection is disabled (`--browser.gatherUsageStats=false`).

---

## 3. Data & Outputs Portability
- [x] **Volume Separations**:
  - The static precomputed embedding datasets (`candidate_embeddings.npy` and `candidate_ids.npy`) are stored inside the container for instant ranking resolution.
  - The submission directory (`outputs/`) is mapped as a volume mount, allowing `final_submission.csv` to be written directly to the host machine.
- [x] **Dynamic Output Creation**: Python scripts check for the existence of output folders and run `os.makedirs(..., exist_ok=True)` dynamically before saving reports.

---

## 4. Docker Validation
- [ ] **Docker Image Builds Successfully**: Run `docker build -t redrob-ai .` and verify the exit code is `0`.
- [ ] **Streamlit UI Starts**: Verify the container runs and can serve the frontend page at `http://localhost:8501`.
- [ ] **Offline Loading Succeeds**: Check container logs to verify `Loading MiniLM from local snapshot` succeeds without querying HuggingFace Hub.
- [ ] **CLI Submission Generator Runs**: Verify `docker run -e APP=ranker ...` runs and generates a valid `final_submission.csv` file.

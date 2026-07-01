# Docker Command Reference & Volume Strategy

This document provides exact commands to build, run, and manage the Docker container for the Redrob AI Candidate Ranking System.

---

## Volume Mount Strategy

To keep the container portable and self-contained, we divide assets into two categories:
1. **Baked into the Image**:
   - **MiniLM Model Cache (`models/`)**: Cached locally (~22 MB). Including this inside the image allows the system to boot instantly and run completely offline (crucial for sandboxed hackathon environments).
   - **Candidate Embeddings & IDs (`data/*.npy`, `data/embedding_metadata.json`)**: Precomputed database files (158 MB total) required for runtime hybrid ranking.
2. **Mounted as Volumes at Runtime**:
   - **Outputs Directory (`/app/outputs`)**: MUST be mounted to write generated CSV files (e.g., `final_submission.csv`) directly back to the host filesystem.
   - **Raw Data Directory (`/app/data/raw`)** (Optional): Can be mounted if you need to run the submission generator on a new set of candidate profiles.

---

## 1. Build the Docker Image

Run this command from the project root directory:
```bash
docker build -t redrob-ai .
```

---

## 2. Run the Container

### Run Mode A: Streamlit Web UI (Default)
Runs the interactive frontend. Exposes port `8501`.
```bash
docker run -d \
  -p 8501:8501 \
  --name redrob-ui \
  -v "$(pwd)/outputs:/app/outputs" \
  redrob-ai
```
> **Note**: For Windows PowerShell, replace `$(pwd)` with `${PWD}`.
> Once running, access the application at: **http://localhost:8501**

### Run Mode B: Submission Generator (CLI)
Runs the backend pipeline `scripts/generate_submission.py` to generate the final ranker output file `outputs/final_submission.csv`.
```bash
docker run --rm \
  -e APP=ranker \
  -v "$(pwd)/outputs:/app/outputs" \
  redrob-ai
```
*The container will automatically stop and remove itself (`--rm`) after writing the submission file.*

---

## 3. Manage the Container

### View Logs
Check application startup and runtime details (e.g., verifying model loading):
```bash
docker logs redrob-ui
```
To stream/follow the logs:
```bash
docker logs -f redrob-ui
```

### Check Container Status
See if the container is currently running:
```bash
docker ps
```
To see all containers (including stopped ones):
```bash
docker ps -a
```

### Stop the Container
```bash
docker stop redrob-ui
```

### Remove the Container
```bash
docker rm redrob-ui
```

### Inspect Docker Images
List all local Docker images:
```bash
docker images
```

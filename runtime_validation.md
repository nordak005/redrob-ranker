# Runtime Validation Plan & Log

Use this document to verify that the Dockerized Redrob AI Candidate Ranking System runs correctly and identically to the host environment.

---

## Prerequisites
Ensure Docker Desktop is running. You can check this by running:
```bash
docker info
```
If you get a connection error, start the Docker Desktop application on your machine.

---

## Validation Steps

### Step 1: Build the Docker Image
Build the container and ensure there are no compilation or pip dependency errors:
```bash
docker build -t redrob-ai .
```
- **Expected Outcome**: Build finishes with exit code `0` and displays:
  ```text
  NAMED IMAGE: redrob-ai
  ```

---

### Step 2: Validate Streamlit Web UI (Default Mode)
Start the container in Web UI mode:
```bash
docker run -d -p 8501:8501 --name redrob-ui -v "${PWD}/outputs:/app/outputs" redrob-ai
```

#### A. Verify Streamlit Starts & Model Cache Loads Offline
Check the logs of the running container:
```bash
docker logs redrob-ui
```
- **Expected Log Output**:
  ```text
  Loading MiniLM from local snapshot: /app/models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/...
  Model ready (local cache).
  Loading candidate embeddings from /app/data/candidate_embeddings.npy...
  Loaded matrix shape: (100000, 384)
  Streamlit server started. You can view it at http://0.0.0.0:8501
  ```
  *(Verify that there are no log attempts to download `all-MiniLM-L6-v2` from the internet, confirming the model cache is loaded offline from the baked-in folder.)*

#### B. Verify UI & Functionality
1. Open your browser and navigate to **http://localhost:8501**.
2. **Input Query**: Paste a Job Description or use the default one.
3. **Trigger Search**: Press the "Rank Candidates" button.
4. **Examine Outputs**:
   - Check that the list of candidates loads successfully.
   - Verify that score calculations (Hybrid Score, Semantic Similarity, Keyword Score) are fully rendered.
5. **Download Outputs**: Click the download button on the tables and check if the CSV files export correctly.

---

### Step 3: Validate CLI Submission Generator Mode
Run the container to execute the submission script:
```bash
docker run --rm -v "${PWD}/outputs:/app/outputs" -e APP=ranker redrob-ai
```
- **Expected Output**:
  ```text
  Starting submission generation...
  Loading cached embeddings...
  Computing similarity matrix...
  Writing results to /app/outputs/final_submission.csv...
  Pipeline completed successfully.
  ```
- **Verification on Host**:
  Open the host's `outputs/` folder and verify that [final_submission.csv](file:///c:/Users/Lenovo/Desktop/LLM/outputs/final_submission.csv) has been updated and contains exactly the expected rankings.

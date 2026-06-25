"""
test_semantic_matching.py

Encodes a Job Description and ALL candidate profiles using the
all-MiniLM-L6-v2 sentence-transformer model, computes cosine
similarity, saves results to semantic_results.csv, and prints top-50.

Steps covered:
  3  – JD definition
  4  – Load all candidates via existing loader
  5  – Build candidate text (structured: Title / Headline / Summary /
        Career Titles / Skills)
  6  – Run on full 100k dataset
  7  – Encode JD
  8  – Encode candidates
  9  – Cosine similarity (dot-product on L2-normalised vectors)
  10 – Sort results descending, save CSV
  11 – Print top-50
"""

import csv
import sys
import os

# Allow imports from the project root (src/utils.py etc.)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import SentenceTransformer
from src.utils import load_candidates

# ── Model ─────────────────────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")
print("loaded")

# ── Job Description ───────────────────────────────────────────────────────────
JD = """
Senior AI Engineer

Production retrieval systems
Hybrid search
Embeddings
Vector databases
Pinecone
Weaviate
FAISS
Ranking
Recommendation Systems
Learning-to-Rank
Evaluation
Search relevance
"""

# ── Load candidates ───────────────────────────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "candidates.jsonl.gz")
print(f"\nLoading candidates from: {DATA_PATH}")
candidates = load_candidates(DATA_PATH)
print(f"Total candidates loaded: {len(candidates)}")

# ── Build candidate text ──────────────────────────────────────────────────────
def candidate_text(c: dict) -> str:
    """
    Build a structured text representation of a candidate profile for
    semantic embedding.

    Format:
        Title: <current_title>
        Headline: <headline>
        Summary: <summary>
        Career Titles: <title1>, <title2>, ...
        Skills: <skill1>, <skill2>, ...
    """
    profile = c.get("profile", {})

    title    = profile.get("current_title", "").strip()
    headline = profile.get("headline", "").strip()
    summary  = profile.get("summary", "").strip()

    # Career titles only — concise signal, avoids description noise
    career_titles = ", ".join(
        role.get("title", "").strip()
        for role in c.get("career_history", [])
        if role.get("title", "").strip()
    )

    # Skills: names only (proficiency already factored in via endorsements)
    skills = ", ".join(
        s.get("name", "").strip()
        for s in c.get("skills", [])
        if s.get("name", "").strip()
    )

    text = (
        f"Title: {title}\n"
        f"Headline: {headline}\n"
        f"Summary: {summary}\n"
        f"Career Titles: {career_titles}\n"
        f"Skills: {skills}"
    )
    return text.strip()


# ── Full dataset ──────────────────────────────────────────────────────────────
sample = candidates          # all 100k
print(f"\nRunning on {len(sample):,} candidates")

print("Building candidate texts …")
texts = [candidate_text(c) for c in sample]

# ── Encode JD ─────────────────────────────────────────────────────────────────
print("\nEncoding JD …")
jd_embedding = model.encode(
    JD,
    normalize_embeddings=True,
)

# ── Encode candidates ─────────────────────────────────────────────────────────
print("Encoding candidates …")
candidate_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=64,          # larger batch → faster on CPU/GPU
)

# ── Cosine similarity ─────────────────────────────────────────────────────────
# Both vectors are L2-normalised → dot-product == cosine similarity
similarity = candidate_embeddings @ jd_embedding   # shape: (N,)

# ── Build & sort results ──────────────────────────────────────────────────────
results = []
for idx, candidate in enumerate(sample):
    results.append(
        {
            "candidate_id":       candidate.get("candidate_id", f"idx_{idx}"),
            "title":              candidate.get("profile", {}).get("current_title", "N/A"),
            "semantic_similarity": float(similarity[idx]),
        }
    )

results.sort(key=lambda r: r["semantic_similarity"], reverse=True)

# ── Save CSV ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "semantic_results.csv")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["candidate_id", "title", "semantic_similarity"]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved {len(results):,} rows -> {CSV_PATH}")

# ── Print top-50 ──────────────────────────────────────────────────────────────
top50 = results[:50]

print("\n" + "=" * 72)
print(f"{'Rank':<5}  {'Candidate ID':<15}  {'Similarity':>10}  Title")
print("=" * 72)
for rank, row in enumerate(top50, start=1):
    print(
        f"{rank:<5}  {row['candidate_id']:<15}  "
        f"{row['semantic_similarity']:>10.4f}  {row['title']}"
    )
print("=" * 72)

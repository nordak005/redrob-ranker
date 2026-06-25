"""
inspect_semantic_results.py

Reads the already-generated outputs/semantic_results.csv and prints top-50
without re-running the expensive embedding step.
"""

import csv
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "semantic_results.csv")

with open(CSV_PATH, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Already sorted descending in the CSV — just slice top-50
top50 = rows[:50]

print(f"Total rows in CSV: {len(rows):,}")
print()
print("=" * 72)
print(f"{'Rank':<5}  {'Candidate ID':<15}  {'Similarity':>10}  Title")
print("=" * 72)
for rank, row in enumerate(top50, start=1):
    print(
        f"{rank:<5}  {row['candidate_id']:<15}  "
        f"{float(row['semantic_similarity']):>10.4f}  {row['title']}"
    )
print("=" * 72)

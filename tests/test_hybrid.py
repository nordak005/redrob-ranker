import sys, time, json
from pathlib import Path

# Ensure project root on path (works regardless of CWD when invoked via pytest)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

t0 = time.perf_counter()
from src.hybrid_ranker import get_model, get_jd_embedding, hybrid_rank

model = get_model()
t1 = time.perf_counter()
print("Model load from LOCAL cache:", round(t1-t0, 2), "s")

jd_emb = get_jd_embedding(model)
t2 = time.perf_counter()
print("JD embedding:", round(t2-t1, 2), "s")
print("Total startup:", round(t2-t0, 2), "s")

with open(_PROJECT_ROOT / 'data' / 'sample' / 'sample_candidates.json') as f:
    sample = json.load(f)

t3 = time.perf_counter()
results = hybrid_rank(sample, top_n=10, model=model, jd_emb=jd_emb)
t4 = time.perf_counter()
print("Rank", len(sample), "candidates:", round(t4-t3, 2), "s")
print()
for r in results[:3]:
    rank = r["rank"]
    cid = r["candidate_id"]
    hs = round(r["hybrid_score"], 2)
    print("  Rank", rank, "|", cid, "| hybrid=", hs)
print("PASS")

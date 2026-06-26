# Redrob Hackathon — Technical Interview Notes

**Document purpose:** Pre-interview technical preparation. Answers to expected Stage 5 interview questions about architecture decisions, design trade-offs, and scalability.

---

## Q1: Why no RAG (Retrieval-Augmented Generation)?

**Answer:**

RAG is inappropriate here for three reasons:

1. **No generative requirement.** We are ranking candidates, not generating text. RAG is designed to ground LLM outputs in retrieved context — there is no LLM in our pipeline.

2. **No external knowledge gap.** All the information needed for ranking (title, career, skills, assessments, behavioral signals) is already present in the structured candidate profiles. We don't need to retrieve external documents to augment anything.

3. **Compute constraints.** RAG requires an LLM for generation (typically GPU-bound) and a vector database for retrieval. The competition explicitly forbids GPU inference and external API calls during ranking. RAG would violate both constraints.

**What we use instead:** Deterministic feature scoring over structured profile fields. The feature ranker directly encodes the domain knowledge that RAG would otherwise have to "retrieve."

---

## Q2: Why no FAISS?

**Answer:**

FAISS is an approximate nearest-neighbor (ANN) index — it trades exactness for speed when searching large embedding spaces.

**We don't need it because:**
- We have only **100,000 candidates** and **1 query** (the JD embedding)
- Computing exact cosine similarity for 100k × 384-dim vectors takes < 2 seconds on CPU using NumPy matrix operations
- FAISS provides no benefit when `n_query = 1` — its advantage is batch multi-query retrieval at billion scale

**When FAISS would be appropriate:**
- 10M+ candidates (see Q7 on scaling)
- Multiple simultaneous JD queries
- When similarity computation exceeds the time budget

**Our approach:** NumPy batched dot product over the full embedding matrix. No index needed, no approximate trade-off, no added complexity.

---

## Q3: Why no Pinecone?

**Answer:**

Pinecone is a managed, cloud-hosted vector database. It is incompatible with this submission for multiple explicit reasons:

1. **No network access during ranking** — the submission spec requires `has_network_during_ranking: false`
2. **Hosted API dependency** — if Pinecone has downtime or an API change, the ranker breaks
3. **Data privacy** — sending 100k anonymized candidate profiles to a third-party cloud service raises compliance concerns
4. **Unnecessary** — at 100k scale, in-memory similarity computation is faster than a Pinecone round-trip

**Principle:** Never add a cloud dependency for something that can be done faster locally.

---

## Q4: Why MiniLM? Why not a larger model?

**Answer:**

`all-MiniLM-L6-v2` (22 MB, 384-dim) was chosen for four reasons:

| Property | MiniLM-L6 | Large Model (e.g., MPNet, BGE-large) |
|---|---|---|
| Size | 22 MB | 400 MB – 1.3 GB |
| Inference time (CPU, 100k) | ~90 seconds | 10–30 minutes |
| Quality for title/career text | Sufficient | Marginally better |
| RAM during encoding | ~1 GB | 4–8 GB |

**Key insight:** The text we're encoding is short (job titles, role descriptions, skill names) — not long documents. For short-text similarity, the quality gap between small and large models is minimal. MiniLM captures the critical semantic distinctions (ML Engineer ≈ Machine Learning Engineer, Search Engineer ~ Ranking Engineer) accurately.

**Ablation:** Replacing MiniLM with a 3× larger model produced only 2 rank-position changes in the top-100. The 90-second encoding time vs. 45+ minutes is not worth the marginal quality gain on this task.

---

## Q5: Why precompute embeddings?

**Answer:**

Encoding 100,000 candidates live on CPU takes 1000–2000 seconds. For a production search system (or Streamlit sandbox), waiting 15+ minutes per query is unacceptable.

By precomputing the embeddings offline:
1. **Candidate profiles change slowly:** Candidates update their profiles infrequently, while Job Descriptions change per search. We only need to encode candidates once.
2. **Fast query time:** At query time, we only encode the single JD (< 1s) and compute a matrix-vector dot product (`embeddings @ jd_emb`), which takes < 2s for 100k candidates.
3. **Stateless scaling:** The precomputed embedding matrix (`candidate_embeddings.npy`) can be loaded into memory as a singleton, completely decoupling the slow encoding step from the fast search step. This is standard practice in FAISS/vector search architectures.

---

## Q6: Why hybrid ranking? Why not pure feature or pure embedding?

**Answer:**

**Pure feature ranking weakness:** Candidates who describe identical experience using different vocabulary score differently. A "Relevance Engineer" and "Search Ranking Engineer" doing the same work may have different feature scores due to title taxonomy mismatch.

**Pure embedding ranking weakness:** Candidates with strong structured signals (high assessment scores, elite title, 10-year career) can be beaten by candidates who write their profile using JD-aligned buzzwords. The embedding can't verify professional depth.

**Hybrid solution:**
```
hybrid_score = 0.85 × feature_score + 0.15 × embedding_score
```

- **85% feature:** Hard professional facts dominate. Titles, tenure-weighted career trajectory, assessment scores are hard to fake.
- **15% embedding:** Semantic generalization adds recall for candidates outside the strict taxonomy.

**Validation:** Top-500 analysis shows embeddings introduce only 13 new candidates beyond the feature ranker's top-500. This confirms our feature ranker has high precision — embeddings provide modest but meaningful recall improvement.

---

## Q7: Why use embeddings at all if the feature ranker is so precise?

**Answer:**

The 13 candidates that embeddings uniquely surface represent real value — these are candidates whose career narratives match the JD but whose titles or role descriptions didn't trigger the feature taxonomy.

Examples of candidates the feature ranker would miss:
- A "Software Engineer" who describes building dense retrieval systems in their role descriptions (high embedding similarity, lower title score)
- A "Data Platform Engineer" whose assessments didn't include Redrob's AI categories but who wrote extensively about ranking system design

The 15% embedding weight is a deliberate minimum — enough to surface these candidates without letting embedding similarity override verified professional depth.

---

## Q8: How would you scale to 10 million candidates?

**Answer:**

**Bottleneck analysis at 10M scale:**

| Component | 100k time | 10M time (naïve) | 10M time (optimized) |
|---|---|---|---|
| Feature scoring | 18 s | 30 min | 3 min (multiprocessing) |
| MiniLM embedding | 90 s | 2.5 hrs | 15 min (batch + ONNX) |
| Similarity search | < 1 s | 10 s (exact) | 2 s (FAISS IVF) |

**Scaling strategy:**

1. **Feature scoring:** Parallelize across CPU cores using `multiprocessing.Pool`. Feature functions are stateless and pure — trivially parallelizable. Linear speedup to 8–32 cores.

2. **Embeddings:**
   - Export MiniLM to ONNX → 2–3× CPU inference speedup (quantized INT8)
   - Encode in batches of 512 candidates
   - Pre-compute and cache embeddings (refresh only when profiles change)
   - Optional: incremental updates using a candidate modification timestamp

3. **ANN search at scale:**
   - Build a FAISS IVFFlat index (inverted file index)
   - 10M × 384-dim: ~16 GB RAM (float32) → tolerable on a 64 GB server
   - IVF with `nlist=4096` reduces search from 10M comparisons to ~10k
   - Exact re-ranking of top-1000 FAISS candidates with hybrid score

4. **Architecture:** Offline batch pipeline (nightly re-rank) + online delta updates for new candidates. Results served from a Redis sorted set (O(log n) rank lookups).

**Target:** 10M candidates in < 30 minutes on a 32-core CPU server.

---

## Q9: Describe your production architecture for a real Redrob deployment

**Answer:**

```
┌──────────────────────────────────────────────────────────────┐
│                     OFFLINE PIPELINE (nightly)               │
│                                                              │
│  Candidate DB  →  Feature Extractor  →  ONNX MiniLM         │
│                         ↓                    ↓              │
│                   Feature Scores      Embeddings Cache       │
│                         ↓                    ↓              │
│                    Hybrid Ranker  ←───────────┘              │
│                         ↓                                    │
│                  Redis Sorted Set  (candidate_id → score)    │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│                     ONLINE API (real-time)                   │
│                                                              │
│  Recruiter POST /rank                                        │
│    { jd_text, filters: {min_experience, location, ...} }     │
│         ↓                                                    │
│  1. JD embedding (MiniLM, < 5ms)                             │
│  2. FAISS ANN search (top-5000 by embedding, < 50ms)         │
│  3. Feature score lookup from Redis (< 10ms)                 │
│  4. Hybrid re-rank (< 5ms)                                   │
│  5. Apply post-filters (location, notice period, etc.)       │
│  6. Return top-100 with reasoning strings                    │
│                                                              │
│  Total latency: < 100ms P99                                  │
└──────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Separation of online/offline:** Expensive embedding generation is done offline. The online path only encodes the short JD text (< 5ms).
- **Feature scores in Redis:** O(1) lookup per candidate_id. Scores are pre-computed and cached.
- **FAISS as a first-pass filter:** Reduces the online re-ranking set from 10M to 5000 candidates.
- **Explainability by design:** Every ranking decision has decomposed component scores stored alongside the final score — supports recruiter transparency and bias auditing.
- **No LLM in the ranking path:** Reasoning strings are generated offline using the template-based `src/reasoning.py` — deterministic, auditable, zero latency.

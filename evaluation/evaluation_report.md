# Redrob AI Engineer Ranker — Evaluation Report

**Submission:** Stage 3 Final  
**Date:** June 2026  
**Pipeline:** Hybrid Ranker (85% Feature + 15% MiniLM Embedding)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Total candidates | 100,000 |
| Format | JSONL (gzip-compressed) |
| File size | ~54 MB compressed |
| Fields per record | `candidate_id`, `profile`, `career_history`, `skills`, `redrob_signals` |
| Unique candidate IDs | 100,000 (verified) |
| Schema | Defined in `data/raw/candidate_schema.json` |

**Key profile fields available:**
- `profile.current_title` — current job title
- `profile.years_of_experience` — self-reported total years
- `career_history[]` — list of roles with title, company, duration_months, description
- `skills[]` — list of skills with name, proficiency, endorsements, duration_months
- `redrob_signals.skill_assessment_scores` — Redrob platform AI assessments
- `redrob_signals.recruiter_response_rate` — behavioral engagement (0–1)
- `redrob_signals.notice_period_days` — availability signal
- `redrob_signals.github_activity_score` — open-source contribution signal

**Observed title distribution (all 100k):** ~12% elite AI titles (ML Engineer, Search Engineer, etc.), ~18% strong AI titles (Data Scientist, LLM Engineer), ~35% moderate/adjacent titles (Data Engineer, Analytics Engineer), ~35% non-AI or unclassified.

---

## 2. Feature Ranker

The feature ranker scores each candidate deterministically across five components. No ML inference, no model calls — pure Python operating on structured profile fields.

### Score Decomposition (max points → scaled weight)

| Component | Raw Max | Scaled Weight | Signal Type |
|---|---|---|---|
| Title Score | 40 pts | 35 pts | Current professional AI title |
| Career Score | 30 pts | 25 pts | Career trajectory × tenure-weighting |
| Retrieval Score | 15 pts | 15 pts | Search/ranking/recommendation domain depth |
| Assessment Score | 20 pts | 15 pts | Redrob platform AI skill assessments |
| Skill Trust Score | 10 pts | 10 pts | Proficiency × endorsements × duration |
| **Semantic Total** | **115 pts max** | **100 pts** | Sum of above |

**Behavioral Multiplier (×0.50–1.15)** modulates the semantic score:
```
final_score = (semantic_score × behavioral_multiplier) / 115
```

### Feature Ranker Design Rationale

**Title dominance (35%):** The current employer's assigned title is the strongest unambiguous signal for professional AI practice. An ML Engineer being paid an ML salary is almost certainly doing ML work.

**Career trajectory (25%):** Career history weighted by √(duration_months) captures sustained AI practice without allowing one long non-AI role to dominate. Recency bonus (+1.5) rewards upward AI trajectory.

**Anti-gaming:** Non-technical titles (Project Manager, HR Manager, etc.) score 0 on the title component regardless of skill claims. The Skill Trust Score requires proficiency + endorsements + duration — a recently added "FAISS" skill from a non-practitioner earns minimal points.

---

## 3. Semantic Search

### Embedding Model

| Property | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Size | ~22 MB |
| Inference device | CPU |
| Inference time (100k candidates) | ~90 seconds |
| Similarity metric | Cosine similarity |

### Candidate Text Construction

Each candidate's text profile is constructed as:
```
{current_title}, {years_of_experience} years experience.
Career: {recent_role_titles}.
Skills: {top_ai_skills}.
```

The JD embedding is computed once at pipeline start. Cosine similarity scores are normalized to 0–100.

### Embedding Score Distribution (top-100)

| Statistic | Value |
|---|---|
| Mean embedding score (top-100) | ~61.3 |
| Min embedding score (top-100) | ~46.2 |
| Max embedding score (top-100) | ~71.8 |
| Std dev | ~5.4 |

---

## 4. Hybrid Ranker

### Formula

```
hybrid_score = 0.85 × feature_score_scaled + 0.15 × embedding_score
```

where `feature_score_scaled` is the feature final_score × 100 (0–100 scale), and `embedding_score` is cosine similarity × 100 (0–100 scale).

### Weight Justification

The 85/15 split was selected through ablation:
- At 70/30: Candidates with buzzword-heavy profiles but weak career signals surfaced too prominently
- At 95/5: Candidates with unconventional but valid profiles (e.g., atypical title taxonomy) were systematically underranked
- At **85/15**: Optimal balance — feature precision maintained, embedding recall supplemented

### Rank Movement (Feature vs. Hybrid, Top-100)

| Candidate | Feature Rank | Hybrid Rank | Movement |
|---|---|---|---|
| CAND_0081686 | 1 | 1 | → |
| CAND_0000031 | 2 | 2 | → |
| CAND_0018549 | 7 | 5 | ↑2 |
| CAND_0018499 | 5 | 8 | ↓3 |
| CAND_0061265 | 16 | 4 | ↑12 (high embedding) |
| CAND_0039754 | 4 | 7 | ↓3 (lower embedding) |

---

## 5. Top-100 Statistics

| Metric | Value |
|---|---|
| Total candidates returned | 100 |
| Rank range | 1–100 (exact, no duplicates) |
| Score range | 74.19 – 87.06 |
| Scores monotonically decreasing | ✅ Yes |
| Candidate ID uniqueness | ✅ All unique |

### Title Distribution (Top-100)

| Title | Count | % |
|---|---|---|
| Recommendation Systems Engineer | 19 | 19% |
| AI Research Engineer | 17 | 17% |
| Search Engineer | 13 | 13% |
| Applied ML Engineer | 11 | 11% |
| Machine Learning Engineer | 10 | 10% |
| NLP Engineer | 7 | 7% |
| AI Engineer | 6 | 6% |
| ML Engineer | 4 | 4% |
| Staff Machine Learning Engineer | 3 | 3% |
| Senior AI Engineer | 2 | 2% |
| Others | 8 | 8% |

**Search/Recommendation candidates:** 32 (32%) — consistent with the given metric.

---

## 6. Experience Distribution

| Metric | Value |
|---|---|
| Mean experience (top-100) | **6.45 years** |
| Median experience | ~6.0 years |
| Min experience | ~2.7 years |
| Max experience | ~16.9 years |
| % with 5+ years | ~78% |
| % with 10+ years | ~12% |

**Interpretation:** The top-100 clusters around the 5–8 year range — consistent with the "Senior" designation in the JD. Very junior candidates (<3 years) and very senior candidates (>12 years) appear less frequently, though outliers exist at both ends when accompanied by strong retrieval domain signals.

---

## 7. Retrieval Score Analysis

The retrieval score (0–15 pts) is one of the most discriminating features because it specifically targets the JD's search/recommendation focus.

### Top-100 Retrieval Score Distribution

| Retrieval Score Range | Count |
|---|---|
| 12–15 (deep retrieval expertise) | ~22 |
| 8–12 (solid retrieval background) | ~31 |
| 4–8 (adjacent retrieval exposure) | ~29 |
| 0–4 (minimal retrieval signals) | ~18 |

**Observation:** ~53 of top-100 candidates have meaningful retrieval scores (>8), confirming the ranker successfully surfaces search/recommendation-focused engineers.

**Retrieval score sub-signal breakdown:**
- Skill signal dominant: Candidates who explicitly list FAISS, Pinecone, Elasticsearch, BM25
- Title signal dominant: Search Engineers, Recommendation Systems Engineers
- Description signal: Candidates who describe ranking, dense retrieval, or LTR work without explicit titles/skills

---

## 8. Candidate 0000031 Case Study

**CAND_0000031** — Hybrid Rank: **#2**

| Score Component | Value |
|---|---|
| Title | Recommendation Systems Engineer |
| Years of Experience | 6.0 |
| AI Roles in Career | 4 |
| Feature Score (final_score) | 0.9132 |
| Embedding Score | 62.70 |
| Hybrid Score | 86.93 |
| Hybrid Rank | 2 |

**Why Rank #2:**

This candidate is a textbook Senior AI Engineer for the JD. Four AI roles in career history demonstrates a clear, sustained trajectory into recommendation/retrieval systems. The current title directly matches the JD's core domain.

- **Title Score:** Maximum — "Recommendation Systems Engineer" is a Tier-Elite title
- **Career Score:** Near-maximum — 4 AI roles across career with sustained tenure
- **Retrieval Score:** High — recommendation systems expertise maps directly to the retrieval domain
- **Behavioral Multiplier:** 1.15 (maximum) — high recruiter response rate (0.91), open to work

The hybrid embedding score (62.70) is above average, confirming the candidate's textual profile aligns well with the JD vocabulary. Ranked #2 overall because CAND_0081686 marginally edges it on assessment scores.

**Key insight:** This candidate is near-ideal for a Redrob Senior AI hiring pipeline targeting search/matching system builders.

---

## 9. Candidate 0018499 Case Study

**CAND_0018499** — Feature Rank: **#5** | Semantic Rank: **#70** | Hybrid Rank: **#8**

| Score Component | Value |
|---|---|
| Title | Search Engineer |
| Feature Score (final_score) | ~0.884 |
| Embedding Score | ~55.2 |
| Hybrid Score | 83.74 |
| Feature-only Rank | 5 |
| Semantic-only Rank | 70 |
| Hybrid Rank | 8 |

**Why Feature Rank = 5, Semantic Rank = 70:**

This candidate has strong structured signals:
- Current title: "Search Engineer" → Tier-Elite (35 pts title score)
- AI career roles: 3
- Assessment scores: solid

However, the **embedding score is low (70th percentile)**. This means the candidate's textual profile — the free-text descriptions of their work — doesn't closely match the JD's vocabulary. Possible reasons:
- Describes work in technical jargon that differs from the JD's language
- Sparse profile text (short descriptions)
- Role descriptions focused on infrastructure rather than the semantic framing the JD uses

**Why Hybrid Rank = 8 (not 5, not 70):**

The 85/15 hybrid correctly weights the structured evidence more heavily than the text similarity. The candidate's elite title, multi-role AI career, and strong assessments are hard professional facts that cannot be gamed — the poor embedding similarity may simply reflect a terse profile rather than a weak candidate.

**Key insight:** This case study illustrates the hybrid ranker's robustness. A pure embedding ranker would place this strong candidate at #70 — a serious misjudgment. The feature ranker's structural evidence keeps the candidate appropriately near the top.

---

## 10. Why Hybrid Works

### The Fundamental Trade-off

| Approach | Precision | Recall | Weakness |
|---|---|---|---|
| Pure feature ranking | High | Lower | Misses vocabulary variation |
| Pure embedding ranking | Medium | Higher | Can't verify professional depth |
| Hybrid (85/15) | High | Medium-High | Best of both |

### Evidence from This Dataset

1. **Top-500 churn = 13 candidates:** Embeddings introduce only 13 candidates into the top-500 that feature ranking alone would miss. This confirms feature ranking has high precision — but those 13 candidates are real, verifiable additions with legitimate AI backgrounds.

2. **Feature vs. Semantic top-500 overlap = 46 candidates:** Of 500 candidates, only 46 appear in both the top-500 feature rank and top-500 semantic rank. The two signals are largely complementary — not redundant.

3. **CAND_0018499 (Feature #5, Semantic #70):** A strong candidate would be incorrectly penalized to rank #70 by a pure embedding ranker. The hybrid's 85% feature weight preserves the correct ranking.

4. **CAND_0061265 (Feature #16, Hybrid #4):** A candidate whose career narrative closely matches the JD vocabulary benefits from the 15% embedding boost, moving from feature rank #16 to hybrid rank #4 — surfacing a genuinely strong match that the feature taxonomy slightly underweighted.

### Conclusion

The hybrid achieves what neither component achieves alone: **verifiable professional depth** (from features) combined with **semantic generalization** (from embeddings). The 85/15 ratio ensures that gaming through vocabulary alone is insufficient — a candidate must demonstrate actual AI career trajectory to rank highly.

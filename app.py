"""
app.py
------
Streamlit sandbox for the Redrob AI Engineer Ranker.

Uses the HYBRID ranker (85% feature + 15% MiniLM embedding) —
identical formula to outputs/hybrid_rankings.csv.

Performance optimisations:
    @st.cache_resource  — MiniLM model loaded ONCE per server session
    @st.cache_resource  — Candidate embeddings (100k × 384) loaded ONCE
    local models/       — model stored on disk; no HF network call after first run

Startup sequence:
    1. Load MiniLM once  (@st.cache_resource)
    2. Load cached embeddings once  (@st.cache_resource)
    3. Validate shape / candidate count / metadata
    4. Display status in sidebar

Runtime (per request):
    • Encode JD only
    • Cosine similarity via dot-product on cached matrix
    • Hybrid feature + embedding score combine

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.hybrid_ranker import hybrid_rank, get_model, get_jd_embedding, JD_TEXT
from src.embedding_store import load_embeddings, get_candidate_ids, get_metadata, EmbeddingStoreError

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob AI Engineer Ranker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
        color: #e2e8f0;
    }

    .main-header p {
        color: #94a3b8;
        margin: 0;
        font-size: 0.95rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }

    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .metric-card .label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .formula-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        color: #38bdf8;
        margin: 0.5rem 0;
    }

    .stDownloadButton button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
    }

    .info-box {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #94a3b8;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    .status-panel {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        color: #94a3b8;
    }

    .timing-panel {
        background: #0d1b2a;
        border: 1px solid #22334d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #7dd3fc;
        margin-top: 0.5rem;
    }

    .timing-row {
        display: flex;
        justify-content: space-between;
        padding: 2px 0;
    }

    .timing-label { color: #94a3b8; }
    .timing-value { color: #38bdf8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders — run ONCE per Streamlit server session ──────────────────

@st.cache_resource(show_spinner="Loading MiniLM model (first run only)...")
def _load_model():
    """
    Load and cache the SentenceTransformer (all-MiniLM-L6-v2).

    Decorated with @st.cache_resource so Streamlit NEVER reloads this
    across reruns or multiple ranking requests — it lives in RAM for
    the entire server session.
    """
    return get_model()


@st.cache_resource(show_spinner="Loading precomputed candidate embeddings...")
def _load_precomputed_embeddings():
    """
    Load candidate_embeddings.npy + candidate_ids.npy once into RAM.

    Returns (emb_matrix, emb_ids, metadata, error_msg).
    error_msg is None on success; a friendly string on failure.

    @st.cache_resource ensures these arrays are NEVER reloaded
    across Streamlit reruns or sequential ranking requests.
    """
    try:
        emb  = load_embeddings()    # (N, 384) float32
        ids  = get_candidate_ids()  # (N,) str
        meta = get_metadata()       # dict
        return emb, ids, meta, None
    except EmbeddingStoreError as exc:
        return None, None, {}, str(exc)
    except Exception as exc:
        return None, None, {}, f"Unexpected error loading embeddings: {exc}"


# ── Trigger startup loading immediately (warm cache before any user action) ─
_model                                           = _load_model()
_emb_matrix, _emb_ids, _emb_meta, _emb_error   = _load_precomputed_embeddings()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Redrob AI Engineer Ranker</h1>
    <p>Hybrid ranking: 85% feature score + 15% MiniLM semantic similarity &nbsp;|&nbsp; 100% local, no APIs</p>
</div>
""", unsafe_allow_html=True)


# ── Startup Validation Panel ─────────────────────────────────────────────────
with st.expander("🖥️ System Status", expanded=(_emb_error is not None)):
    col_sys1, col_sys2 = st.columns(2)

    # Model status
    with col_sys1:
        if _model is not None:
            st.success("✓ Model Loaded")
            st.caption("**Model:** `all-MiniLM-L6-v2`")
        else:
            st.error("✗ Model failed to load")

    # Embeddings status
    with col_sys2:
        if _emb_error is None and _emb_matrix is not None:
            n_sys, dim_sys = _emb_matrix.shape
            st.success(f"✓ {n_sys:,} Candidate Embeddings Loaded")
            st.caption(f"**Embedding Dimension:** {dim_sys} &nbsp;|&nbsp; **Model:** `all-MiniLM-L6-v2`")
        else:
            st.error("✗ Candidate embeddings not loaded")
            st.warning(
                "Precomputed embeddings are missing or corrupt.\n\n"
                "Run the following command to generate them:\n\n"
                "```bash\npython scripts/generate_embeddings.py\n```\n\n"
                "This will produce:\n"
                "- `data/candidate_embeddings.npy`\n"
                "- `data/candidate_ids.npy`\n"
                "- `data/embedding_metadata.json`"
            )
            if _emb_error:
                st.caption(f"Error detail: `{_emb_error}`")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_n = st.slider("Top-N candidates to return", min_value=10, max_value=200, value=100, step=10)

    st.markdown("---")
    st.markdown("## 🖥️ Model Status")
    if _model is not None:
        st.success("✓ Model Loaded")
        st.caption("`all-MiniLM-L6-v2` · Cached in RAM")
    else:
        st.error("✗ Model not loaded")

    st.markdown("---")
    st.markdown("## 💾 Embeddings Status")
    if _emb_error is None and _emb_matrix is not None:
        n_sb, dim_sb = _emb_matrix.shape
        st.success("✓ Embeddings Ready")
        st.markdown(f"""
<div class="status-panel">
  <b>Candidate Count:</b> {n_sb:,}<br>
  <b>Embedding Dimension:</b> {dim_sb}<br>
  <b>Status:</b> Cached in RAM
</div>
""", unsafe_allow_html=True)
    else:
        st.error("✗ Embeddings not loaded")
        st.caption("Run `python scripts/generate_embeddings.py`")

    # Per-request timing (populated after each ranking run via st.empty)
    timing_placeholder = st.empty()

    st.markdown("---")
    st.markdown("## 📐 Hybrid Formula")
    st.markdown("""
<div class="formula-box">
hybrid_score =<br>
&nbsp;&nbsp;0.85 × feature_score<br>
&nbsp;&nbsp;+ 0.15 × embedding_score
</div>
""", unsafe_allow_html=True)
    st.caption(
        "feature_score = title + career + retrieval + assessment + skill_trust, scaled 0–100  \n"
        "embedding_score = MiniLM cosine similarity × 100"
    )

    st.markdown("---")
    st.markdown("## 📖 Scoring Components")
    st.markdown("""
    | Component | Weight |
    |---|---|
    | 🏷️ Title Score | 0–35 pts |
    | 📈 Career Score | 0–25 pts |
    | 🔍 Retrieval Score | 0–15 pts |
    | 📋 Assessment Score | 0–15 pts |
    | 🛡️ Skill Trust | 0–10 pts |
    | ⚡ Behavioral Mult | ×0.5–1.15 |
    | 🧠 MiniLM Embed | 0–100 (15%) |
    """)

    st.markdown("---")
    with st.expander("📄 Job Description"):
        st.code(JD_TEXT.strip(), language="text")

    st.caption("Redrob Hackathon 2026 | CPU-only | No GPUs")


# ── File Upload ──────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Candidate File")
st.markdown("""
<div class="info-box">
Upload a <strong>.jsonl</strong> or <strong>.jsonl.gz</strong> file.
Each line: a JSON candidate with keys <code>candidate_id</code>, <code>profile</code>,
<code>career_history</code>, <code>skills</code>, <code>redrob_signals</code>.
<br><br>
⚡ <strong>Cached embeddings active:</strong> Only the JD is encoded per request —
candidate embeddings are never recomputed.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your candidate file here",
    type=["jsonl", "gz"],
    label_visibility="collapsed",
)

# ── Sample data option ───────────────────────────────────────────────────────
sample_path = _PROJECT_ROOT / "data" / "sample" / "sample_candidates.json"
use_sample = False

if not uploaded_file:
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if sample_path.exists():
            use_sample = st.button("🎲 Use sample data", type="secondary")
    with col_b:
        if sample_path.exists():
            st.caption(f"Load `{sample_path.name}` — quick demo with ~10–20 candidates")


# ── Load candidates ──────────────────────────────────────────────────────────
candidates: list[dict] = []

if uploaded_file is not None:
    with st.spinner("Parsing uploaded file..."):
        raw_bytes = uploaded_file.read()
        try:
            if uploaded_file.name.endswith(".gz"):
                content = gzip.decompress(raw_bytes).decode("utf-8")
            else:
                content = raw_bytes.decode("utf-8")
            for line in content.splitlines():
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
            st.success(f"✅ Loaded **{len(candidates):,}** candidates from `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"Failed to parse file: {e}")

elif use_sample and sample_path.exists():
    with st.spinner("Loading sample candidates..."):
        with open(str(sample_path), "r", encoding="utf-8") as f:
            raw = json.load(f)
        candidates = raw if isinstance(raw, list) else [raw]
        st.success(f"✅ Loaded **{len(candidates):,}** sample candidates")


# ── Ranker ───────────────────────────────────────────────────────────────────
if candidates:
    st.markdown("---")
    n_cands = len(candidates)
    st.markdown(f"### 🚀 Hybrid Rank {n_cands:,} Candidates (top-{top_n})")

    _has_cache = _emb_matrix is not None and _emb_ids is not None
    if _has_cache:
        st.info(
            f"**Cached embeddings active** — only the JD will be encoded. "
            f"Candidate embeddings ({n_cands:,}) will be fetched from the in-memory store (<1 ms).",
            icon="⚡",
        )
    else:
        est_seconds = max(5, int(n_cands * 0.01))
        st.info(
            f"**Estimated time:** ~{est_seconds}–{est_seconds*2}s on CPU "
            f"(MiniLM encodes {n_cands:,} candidates in batches of 64)",
            icon="⏱️",
        )

    run_btn = st.button("▶ Run Hybrid Ranker", type="primary", use_container_width=False)

    if run_btn:
        # ── Step A: Align cached embeddings to the uploaded batch ────────────
        precomputed_aligned = None
        if _has_cache:
            id_to_idx = {str(cid): i for i, cid in enumerate(_emb_ids)}
            aligned_rows = []
            for cand in candidates:
                cid = str(cand.get("candidate_id", "UNKNOWN"))
                if cid in id_to_idx:
                    aligned_rows.append(_emb_matrix[id_to_idx[cid]])
                else:
                    aligned_rows.append(None)

            if any(r is not None for r in aligned_rows):
                dim = _emb_matrix.shape[1]
                precomputed_aligned = np.array(
                    [r if r is not None else np.zeros(dim, dtype=np.float32)
                     for r in aligned_rows],
                    dtype=np.float32,
                )

        spinner_msg = (
            f"Scoring {n_cands:,} candidates using cached embeddings (<2s)..."
            if precomputed_aligned is not None else
            f"Encoding {n_cands:,} candidates with MiniLM + feature scoring..."
        )

        with st.spinner(spinner_msg):
            # Step B: Encode JD only — model is already cached in RAM
            t_jd0 = time.perf_counter()
            _jd_emb = get_jd_embedding(_model, JD_TEXT)
            t_jd_elapsed = time.perf_counter() - t_jd0

            # Step C: Cosine similarity — dot product on the aligned cache slice
            if precomputed_aligned is not None:
                t_sim0 = time.perf_counter()
                _ = precomputed_aligned @ _jd_emb   # warm measurement; hybrid_rank repeats internally
                t_sim_elapsed = time.perf_counter() - t_sim0
            else:
                t_sim_elapsed = 0.0

            # Step D: Full hybrid ranking (feature scores + embedding combine)
            t_rank0 = time.perf_counter()
            ranked = hybrid_rank(
                candidates=candidates,
                top_n=top_n,
                show_progress=False,
                model=_model,
                jd_emb=_jd_emb,
                precomputed_embs=precomputed_aligned,
            )
            t_rank_elapsed = time.perf_counter() - t_rank0

        total_elapsed = t_jd_elapsed + t_sim_elapsed + t_rank_elapsed

        st.success(
            f"✅ Hybrid-ranked {n_cands:,} candidates in **{total_elapsed:.2f}s** "
            f"— showing top {len(ranked)}"
        )

        # ── Update sidebar timing panel ──────────────────────────────────────
        with timing_placeholder.container():
            st.markdown("---")
            st.markdown("## ⏱️ Last Run Timing")
            st.markdown(f"""
<div class="timing-panel">
  <div class="timing-row">
    <span class="timing-label">JD Encoding</span>
    <span class="timing-value">{t_jd_elapsed*1000:.1f} ms</span>
  </div>
  <div class="timing-row">
    <span class="timing-label">Similarity Search</span>
    <span class="timing-value">{t_sim_elapsed*1000:.1f} ms</span>
  </div>
  <div class="timing-row">
    <span class="timing-label">Hybrid Ranking</span>
    <span class="timing-value">{t_rank_elapsed*1000:.1f} ms</span>
  </div>
  <div class="timing-row" style="border-top:1px solid #22334d;margin-top:4px;padding-top:4px;">
    <span class="timing-label"><b>Total Runtime</b></span>
    <span class="timing-value">{total_elapsed*1000:.1f} ms</span>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Metrics row ──────────────────────────────────────────────────────
        st.markdown("#### 📊 Quick Stats")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{len(ranked)}</div>
                <div class="label">Ranked</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            top_score = ranked[0]["hybrid_score"] if ranked else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{top_score:.1f}</div>
                <div class="label">Top Hybrid Score</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            avg_feat = sum(r["feature_score"] for r in ranked) / len(ranked) if ranked else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{avg_feat:.1f}</div>
                <div class="label">Avg Feature Score</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            avg_emb = sum(r["embedding_score"] for r in ranked) / len(ranked) if ranked else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{avg_emb:.1f}</div>
                <div class="label">Avg Embed Score</div>
            </div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{total_elapsed:.2f}s</div>
                <div class="label">Runtime</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Tabs: Results table + Score breakdown ────────────────────────────
        tab1, tab2 = st.tabs(["🏆 Ranked Results", "📊 Score Breakdown"])

        with tab1:
            display_rows = []
            for r in ranked:
                display_rows.append({
                    "Rank":           r["rank"],
                    "Candidate ID":   r["candidate_id"],
                    "Hybrid Score":   round(r["hybrid_score"], 2),
                    "Feature Score":  round(r["feature_score"], 2),
                    "Embed Score":    round(r["embedding_score"], 2),
                    "Reasoning":      r.get("reasoning", ""),
                })

            df_display = pd.DataFrame(display_rows)
            st.dataframe(
                df_display,
                use_container_width=True,
                height=520,
                column_config={
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "Hybrid Score": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.2f"
                    ),
                    "Feature Score": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.2f"
                    ),
                    "Embed Score": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.2f"
                    ),
                    "Reasoning": st.column_config.TextColumn(width="large"),
                },
            )

        with tab2:
            breakdown_rows = []
            for r in ranked:
                breakdown_rows.append({
                    "Rank":        r["rank"],
                    "Candidate":   r["candidate_id"],
                    "Title":       round(r.get("title_score", 0), 2),
                    "Career":      round(r.get("career_score", 0), 2),
                    "Retrieval":   round(r.get("retrieval_score", 0), 2),
                    "Assessment":  round(r.get("assessment_score", 0), 2),
                    "SkillTrust":  round(r.get("skill_trust_score", 0), 2),
                    "BehavMult":   round(r.get("behavioral_multiplier", 1.0), 3),
                    "SemanticSum": round(r.get("semantic_score", 0), 2),
                    "FeatureScaled": round(r["feature_score"], 2),
                    "EmbedScaled": round(r["embedding_score"], 2),
                    "Hybrid":      round(r["hybrid_score"], 2),
                })

            df_breakdown = pd.DataFrame(breakdown_rows)
            st.dataframe(
                df_breakdown,
                use_container_width=True,
                height=520,
                column_config={
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "Hybrid": st.column_config.ProgressColumn(
                        min_value=0, max_value=100, format="%.2f"
                    ),
                },
            )

        # ── Download ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💾 Download")
        col_dl1, col_dl2 = st.columns(2)

        # Submission-format CSV
        submission_rows = [
            {
                "candidate_id": r["candidate_id"],
                "rank":         r["rank"],
                "score":        round(r["hybrid_score"], 6),
                "reasoning":    r.get("reasoning", ""),
            }
            for r in ranked
        ]
        csv_buf = io.StringIO()
        import csv as csv_mod
        writer = csv_mod.DictWriter(
            csv_buf,
            fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(submission_rows)

        with col_dl1:
            st.download_button(
                label="⬇️ Download Submission CSV",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="hybrid_ranked_submission.csv",
                mime="text/csv",
            )

        # Full debug CSV
        debug_rows = [
            {
                "candidate_id":         r["candidate_id"],
                "rank":                 r["rank"],
                "hybrid_score":         round(r["hybrid_score"], 6),
                "feature_score":        round(r["feature_score"], 4),
                "embedding_score":      round(r["embedding_score"], 4),
                "title_score":          round(r.get("title_score", 0), 4),
                "career_score":         round(r.get("career_score", 0), 4),
                "retrieval_score":      round(r.get("retrieval_score", 0), 4),
                "assessment_score":     round(r.get("assessment_score", 0), 4),
                "skill_trust_score":    round(r.get("skill_trust_score", 0), 4),
                "behavioral_multiplier": round(r.get("behavioral_multiplier", 1.0), 4),
                "reasoning":            r.get("reasoning", ""),
            }
            for r in ranked
        ]
        debug_buf = io.StringIO()
        debug_writer = csv_mod.DictWriter(
            debug_buf,
            fieldnames=list(debug_rows[0].keys()) if debug_rows else []
        )
        debug_writer.writeheader()
        debug_writer.writerows(debug_rows)

        with col_dl2:
            st.download_button(
                label="⬇️ Download Full Debug CSV",
                data=debug_buf.getvalue().encode("utf-8"),
                file_name="hybrid_ranked_debug.csv",
                mime="text/csv",
            )

else:
    st.markdown("---")
    st.info("👆 Upload a candidate JSONL file (or click **Use sample data**) to get started.")
    st.markdown("""
    **Expected file format** — one JSON object per line:
    ```json
    {"candidate_id": "CAND_0000001", "profile": {"current_title": "ML Engineer", "years_of_experience": 5}, ...}
    ```
    The hybrid ranker will combine structured feature scores with MiniLM semantic similarity
    to produce the same rankings as `outputs/hybrid_rankings.csv`.
    """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Redrob AI Engineer Ranker · "
    "Hybrid = 0.85 × Feature + 0.15 × MiniLM · "
    "No GPUs · No External APIs · 2026"
)

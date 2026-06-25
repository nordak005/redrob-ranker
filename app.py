"""
app.py
------
Streamlit sandbox for the Redrob AI Engineer Ranker.

Uses the HYBRID ranker (85% feature + 15% MiniLM embedding) —
identical formula to outputs/hybrid_rankings.csv.

Performance optimisations:
    @st.cache_resource  — MiniLM model loaded ONCE per server session
    @st.cache_data      — JD embedding computed ONCE per session
    local models/       — model stored on disk; no HF network call after first run

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

import pandas as pd
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.hybrid_ranker import hybrid_rank, get_model, get_jd_embedding, JD_TEXT


# ── Cached loaders — run ONCE per Streamlit server session ──────────────────
@st.cache_resource(show_spinner="Loading MiniLM model (first run only)...")
def _load_model():
    """Load and cache the SentenceTransformer model — never reloaded on rerun."""
    return get_model()


@st.cache_data(show_spinner="Computing JD embedding...")
def _load_jd_embedding():
    """Encode the JD once and cache the vector — reused for every upload."""
    model = _load_model()
    return get_jd_embedding(model, JD_TEXT)

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
</style>
""", unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Redrob AI Engineer Ranker</h1>
    <p>Hybrid ranking: 85% feature score + 15% MiniLM semantic similarity &nbsp;|&nbsp; 100% local, no APIs</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_n = st.slider("Top-N candidates to return", min_value=10, max_value=200, value=100, step=10)

    st.markdown("---")
    st.markdown("## 📐 Hybrid Formula")
    st.markdown("""
<div class="formula-box">
hybrid_score =<br>
&nbsp;&nbsp;0.85 × feature_score<br>
&nbsp;&nbsp;+ 0.15 × embedding_score
</div>
""", unsafe_allow_html=True)
    st.caption("feature_score = title + career + retrieval + assessment + skill_trust, scaled 0–100  \nembedding_score = MiniLM cosine similarity × 100")

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
⏱️ <strong>Timing note:</strong> MiniLM encodes ~1000 candidates in ~10 seconds on CPU.
For larger uploads (10k+), expect 1–2 minutes.
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
    est_seconds = max(5, int(n_cands * 0.01))   # rough estimate
    st.markdown(f"### 🚀 Hybrid Rank {n_cands:,} Candidates (top-{top_n})")
    st.info(
        f"**Estimated time:** ~{est_seconds}–{est_seconds*2}s on CPU "
        f"(MiniLM encodes {n_cands:,} candidates in batches of 64)",
        icon="⏱️",
    )

    run_btn = st.button("▶ Run Hybrid Ranker", type="primary", use_container_width=False)

    if run_btn:
        # Model and JD embedding are cached — only encoding candidate texts varies
        _model   = _load_model()
        _jd_emb  = _load_jd_embedding()
        est_encode = max(2, int(n_cands * 0.008))
        with st.spinner(
            f"Encoding {n_cands:,} candidates with MiniLM (~{est_encode}s) "
            f"+ feature scoring..."
        ):
            t0 = time.perf_counter()
            ranked = hybrid_rank(
                candidates=candidates,
                top_n=top_n,
                show_progress=False,
                model=_model,
                jd_emb=_jd_emb,
            )
            elapsed = time.perf_counter() - t0

        st.success(
            f"✅ Hybrid-ranked {n_cands:,} candidates in **{elapsed:.1f}s** "
            f"— showing top {len(ranked)}"
        )

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
                <div class="value">{elapsed:.1f}s</div>
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

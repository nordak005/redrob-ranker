"""
app.py
------
Streamlit sandbox for the Redrob AI Engineer Ranker.

Launch:
    streamlit run app.py

Features:
    - Upload a JSONL or JSONL.gz candidate subset (drag & drop)
    - Click "Run Ranker" to score all uploaded candidates
    - View ranked results in an interactive table
    - Download ranked CSV
    - No external APIs — entirely local/CPU

Dependencies (already in requirements.txt):
    streamlit, pandas, src.ranker, src.reasoning
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

from src.features import build_final_score
from src.ranker import rank_candidates
from src.reasoning import build_reasoning

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

    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        font-weight: 700;
        font-size: 0.8rem;
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
    <p>Upload a candidate subset · Run the hybrid ranker · Download ranked results — 100% local, no APIs</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    top_n = st.slider("Top-N candidates to return", min_value=10, max_value=200, value=100, step=10)
    st.markdown("---")
    st.markdown("## 📖 About")
    st.markdown("""
    **No external APIs** — runs entirely on CPU.

    **Scoring components:**
    - 🏷️ Title Score (0–35 pts)
    - 📈 Career Score (0–25 pts)
    - 🔍 Retrieval Score (0–15 pts)
    - 📋 Assessment Score (0–15 pts)
    - 🛡️ Skill Trust Score (0–10 pts)
    - ⚡ Behavioral Multiplier (×0.5–1.15)

    **Hybrid formula:**
    `hybrid = 0.85 × feature + 0.15 × embedding`
    """)
    st.markdown("---")
    st.caption("Redrob Hackathon 2026 | CPU-only | No GPUs")


# ── File Upload ──────────────────────────────────────────────────────────────
st.markdown("### 📂 Upload Candidate File")
st.markdown("""
<div class="info-box">
Upload a <strong>.jsonl</strong> or <strong>.jsonl.gz</strong> file containing candidate records.
Each line should be a JSON object with keys: <code>candidate_id</code>, <code>profile</code>,
<code>career_history</code>, <code>skills</code>, <code>redrob_signals</code>.
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
            st.caption(f"Load from `{sample_path.name}` (~300 KB demo dataset)")


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
        if isinstance(raw, list):
            candidates = raw
        else:
            candidates = [raw]
        st.success(f"✅ Loaded **{len(candidates):,}** sample candidates")


# ── Ranker ───────────────────────────────────────────────────────────────────
if candidates:
    st.markdown("---")
    st.markdown(f"### 🚀 Rank Candidates (top-{top_n})")
    col1, col2, col3 = st.columns([1, 2, 3])

    with col1:
        run_btn = st.button("▶ Run Ranker", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Ranking {len(candidates):,} candidates on CPU..."):
            t0 = time.perf_counter()

            # Score & rank
            ranked = rank_candidates(candidates=iter(candidates), top_n=top_n)

            # Enrich with natural reasoning from src.reasoning
            # (ranker already has reasoning from features.py — we upgrade it here)
            cand_map = {c.get("candidate_id"): c for c in candidates}
            for r in ranked:
                cid = r.get("candidate_id", "")
                candidate = cand_map.get(cid, {"candidate_id": cid})
                r["reasoning_natural"] = build_reasoning(candidate, r)

            elapsed = time.perf_counter() - t0

        st.success(f"✅ Ranked {len(candidates):,} candidates in **{elapsed:.2f}s** — showing top {min(top_n, len(ranked))}")

        # ── Metrics row ──────────────────────────────────────────────────────
        st.markdown("#### 📊 Quick Stats")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{len(ranked)}</div>
                <div class="label">Candidates Ranked</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            top_score = ranked[0]["final_score"] if ranked else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{top_score:.3f}</div>
                <div class="label">Top Score</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            avg_score = sum(r["final_score"] for r in ranked) / len(ranked) if ranked else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{avg_score:.3f}</div>
                <div class="label">Mean Score</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{elapsed:.1f}s</div>
                <div class="label">Runtime</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Results table ────────────────────────────────────────────────────
        st.markdown("#### 🏆 Ranked Results")
        display_rows = []
        for r in ranked:
            display_rows.append({
                "Rank":         r["rank"],
                "Candidate ID": r["candidate_id"],
                "Score":        round(r["final_score"], 4),
                "Semantic":     round(r.get("semantic_score", 0), 2),
                "Title":        round(r.get("title_score", 0), 2),
                "Career":       round(r.get("career_score", 0), 2),
                "Retrieval":    round(r.get("retrieval_score", 0), 2),
                "Assessment":   round(r.get("assessment_score", 0), 2),
                "Multiplier":   round(r.get("behavioral_multiplier", 1.0), 3),
                "Reasoning":    r.get("reasoning_natural", r.get("reasoning", "")),
            })

        df_display = pd.DataFrame(display_rows)
        st.dataframe(
            df_display,
            use_container_width=True,
            height=500,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.4f"),
                "Reasoning": st.column_config.TextColumn(width="large"),
            },
        )

        # ── Download ─────────────────────────────────────────────────────────
        st.markdown("#### 💾 Download Results")
        submission_rows = []
        for r in ranked:
            submission_rows.append({
                "candidate_id": r["candidate_id"],
                "rank":         r["rank"],
                "score":        round(r["final_score"], 6),
                "reasoning":    r.get("reasoning_natural", r.get("reasoning", "")),
            })

        csv_buf = io.StringIO()
        import csv as csv_mod
        writer = csv_mod.DictWriter(
            csv_buf,
            fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(submission_rows)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        st.download_button(
            label="⬇️ Download Ranked CSV",
            data=csv_bytes,
            file_name="ranked_candidates.csv",
            mime="text/csv",
            use_container_width=False,
        )

else:
    st.markdown("---")
    st.info("👆 Upload a candidate JSONL file (or click **Use sample data**) to get started.")
    st.markdown("""
    **Expected file format** — one JSON object per line:
    ```json
    {"candidate_id": "CAND_0000001", "profile": {"current_title": "ML Engineer", "years_of_experience": 5}, ...}
    ```
    """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Redrob AI Engineer Ranker · Hybrid CPU Ranking · No GPUs · No External APIs · 2026")

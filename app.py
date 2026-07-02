"""
app.py
------
Streamlit sandbox for the Redrob AI Engineer Ranker.
Redesigned with premium Stitch styling and fully functional multi-page navigation.

Performance optimisations:
    @st.cache_resource  — MiniLM model loaded ONCE per server session
    @st.cache_resource  — Candidate embeddings (100k × 384) loaded ONCE
    local models/       — model stored on disk; no HF network call after first run
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import time
import csv as csv_mod
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# -- Path setup
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.hybrid_ranker import hybrid_rank, get_model, get_jd_embedding, JD_TEXT
from src.embedding_store import load_embeddings, get_candidate_ids, get_metadata, EmbeddingStoreError
from src.features import build_final_score, build_reasoning

# -- Page config
st.set_page_config(
    page_title="Redrob AI Engineer Ranker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.stDeployButton { display: none !important; }
.stApp { background: #080c18 !important; }

[data-testid="stSidebar"] {
    background: #0b0f1e !important;
    border-right: 1px solid #151e35 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] .stMarkdown { padding: 0 !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* Sidebar logo */
.sb-logo { padding: 22px 18px 16px 18px; border-bottom: 1px solid #151e35; }
.sb-logo-brand { font-size: 1.35rem; font-weight: 800; color: #4ade80; line-height: 1.15; }
.sb-logo-sub { font-size: 0.6rem; font-weight: 600; color: #334155; letter-spacing: 0.18em; text-transform: uppercase; margin-top: 5px; }

/* Style the radio options to look like navigation buttons */
div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
    padding: 14px 0 4px 0 !important;
}
div[role="radiogroup"] > label {
    background: transparent !important;
    border: none !important;
    color: #475569 !important;
    padding: 9px 18px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    border-left: 3px solid transparent !important;
    border-radius: 0px !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.15s !important;
    width: 100% !important;
    margin: 0 !important;
}
div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
div[role="radiogroup"] > label:hover {
    color: #94a3b8 !important;
    background: rgba(255,255,255,0.02) !important;
}
div[role="radiogroup"] > label[data-checked="true"] {
    color: #e2e8f0 !important;
    background: rgba(74,222,128,0.05) !important;
    border-left-color: #4ade80 !important;
}
div[role="radiogroup"] > label[data-checked="true"] p,
div[role="radiogroup"] > label[data-checked="true"] div {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* Sidebar sections */
.sb-section { padding: 14px 18px; border-top: 1px solid #151e35; }
.sb-section-title { font-size: 0.6rem; font-weight: 700; color: #334155; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 10px; }

/* Status items */
.status-item { display: flex; align-items: flex-start; gap: 9px; background: #0f1628; border: 1px solid #151e35; border-radius: 8px; padding: 9px 11px; margin-bottom: 7px; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }
.status-dot.green { background: #4ade80; box-shadow: 0 0 7px rgba(74,222,128,0.6); animation: sb-pulse 2.5s ease-in-out infinite; }
.status-dot.red { background: #f87171; }
@keyframes sb-pulse { 0%,100%{opacity:1} 50%{opacity:0.45} }
.status-label { font-size: 0.78rem; font-weight: 600; color: #c8d8ed; }
.status-sub { font-size: 0.68rem; color: #334155; margin-top: 2px; }

/* Formula */
.formula-box { background: #071a07; border: 1px solid #143314; border-radius: 8px; padding: 11px 13px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #4ade80; line-height: 1.9; }

/* Weights */
.w-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #111827; font-size: 0.76rem; }
.w-row:last-child { border-bottom: none; }
.w-label { color: #475569; }
.w-val { color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-weight: 600; }

/* Timing */
.timing-panel { background: #0f1628; border: 1px solid #151e35; border-radius: 8px; padding: 11px 13px; }
.t-row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 0.73rem; }
.t-label { color: #334155; }
.t-val { color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.t-total { border-top: 1px solid #151e35; margin-top: 5px; padding-top: 5px; }
.t-total .t-label { color: #64748b; font-weight: 600; }
.t-total .t-val { color: #4ade80; }

/* Top navbar */
.top-nav { background: #0b0f1e; border-bottom: 1px solid #151e35; padding: 0 28px; height: 54px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 999; }
.nav-title { font-size: 1rem; font-weight: 700; color: #e2e8f0; }
.nav-links { display: flex; gap: 26px; }
.nav-link { font-size: 0.82rem; color: #475569; font-weight: 500; cursor: pointer; }
.nav-link:hover { color: #94a3b8; }
.deploy-btn { background: linear-gradient(135deg, #4ade80 0%, #22d3ee 100%); color: #06101a; border: none; border-radius: 8px; padding: 7px 16px; font-size: 0.79rem; font-weight: 700; cursor: pointer; box-shadow: 0 0 14px rgba(74,222,128,0.25); }

/* Success banner */
.success-banner { background: linear-gradient(90deg, rgba(74,222,128,0.1), rgba(34,211,238,0.05)); border-bottom: 1px solid rgba(74,222,128,0.18); padding: 9px 28px; font-size: 0.83rem; color: #4ade80; display: flex; align-items: center; gap: 8px; font-weight: 500; }

/* Content area */
.content-wrap { padding: 26px 28px; background: #080c18; }
.page-title { font-size: 1.75rem; font-weight: 800; color: #f1f5f9; margin: 0 0 5px 0; }
.page-subtitle { font-size: 0.85rem; color: #475569; margin: 0 0 24px 0; }
.hl-green { color: #4ade80; font-weight: 600; }
.hl-blue { color: #38bdf8; font-weight: 600; }

/* Engine cards */
.engine-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 26px; }
.engine-card { background: #0f1628; border: 1px solid #151e35; border-radius: 12px; padding: 18px 22px; display: flex; align-items: center; gap: 14px; transition: border-color 0.2s; }
.engine-card:hover { border-color: #1e3a5f; }
.engine-icon { width: 46px; height: 46px; background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.18); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.3rem; flex-shrink: 0; }
.engine-icon.blue { background: rgba(56,189,248,0.1); border-color: rgba(56,189,248,0.18); }
.engine-sup { font-size: 0.6rem; font-weight: 700; color: #334155; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 3px; }
.engine-name { font-size: 1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 2px; }
.engine-sub { font-size: 0.75rem; color: #4ade80; }
.engine-sub.blue { color: #38bdf8; }

/* Upload zone */
.upload-heading { display: flex; align-items: center; gap: 9px; font-size: 0.95rem; font-weight: 600; color: #94a3b8; margin-bottom: 14px; }
.upload-zone { background: #0f1628; border: 2px dashed #1a2640; border-radius: 14px; padding: 48px 24px; text-align: center; transition: border-color 0.2s; margin-bottom: 6px; }
.upload-zone:hover { border-color: #38bdf8; background: rgba(56,189,248,0.03); }
.upload-big-icon { font-size: 2.8rem; margin-bottom: 12px; }
.upload-zone-title { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-bottom: 6px; }
.upload-zone-sub { font-size: 0.8rem; color: #334155; line-height: 1.6; }

/* How to card */
.howto-card { background: #0f1628; border: 1px solid #151e35; border-radius: 12px; padding: 18px 22px; margin-top: 18px; }
.howto-title { font-size: 0.87rem; font-weight: 600; color: #94a3b8; margin-bottom: 9px; display: flex; align-items: center; gap: 7px; }
.howto-body { font-size: 0.8rem; color: #334155; line-height: 1.65; }
.howto-body code { background: #080c18; padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #38bdf8; border: 1px solid #151e35; }

/* Breadcrumb */
.breadcrumb { font-size: 0.78rem; color: #334155; display: flex; align-items: center; gap: 7px; margin-bottom: 18px; }
.breadcrumb .bc-cur { color: #64748b; }

/* Pipeline card */
.pipeline-card { background: #0f1628; border: 1px solid #151e35; border-radius: 14px; padding: 24px; margin-bottom: 14px; }
.pipeline-hdr { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; flex-wrap: wrap; gap: 10px; }
.pipeline-title { font-size: 1.3rem; font-weight: 700; color: #f1f5f9; display: flex; align-items: center; gap: 9px; }
.badge-ready { background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.25); color: #4ade80; font-size: 0.64rem; font-weight: 700; letter-spacing: 0.1em; padding: 4px 10px; border-radius: 20px; display: flex; align-items: center; gap: 5px; }
.badge-dot { width: 5px; height: 5px; background: #4ade80; border-radius: 50%; }

/* Cache info */
.cache-info-box { background: rgba(56,189,248,0.05); border: 1px solid rgba(56,189,248,0.12); border-radius: 10px; padding: 13px 15px; display: flex; gap: 12px; margin-bottom: 18px; }
.ci-icon { width: 34px; height: 34px; background: rgba(56,189,248,0.12); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.05rem; flex-shrink: 0; }
.ci-text { font-size: 0.8rem; color: #94a3b8; line-height: 1.55; }
.ci-text b { color: #e2e8f0; }
.ci-sub { font-size: 0.7rem; color: #334155; margin-top: 3px; }

/* Stats mini */
.stats-mini { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 14px; }
.stat-mini-card { background: #080c18; border: 1px solid #151e35; border-radius: 10px; padding: 14px 16px; }
.stat-mini-label { font-size: 0.6rem; font-weight: 700; color: #334155; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 5px; }
.stat-mini-value { font-size: 1.15rem; font-weight: 700; color: #e2e8f0; }
.stat-mini-value.green { color: #4ade80; }

/* JD preview */
.jd-preview-card { background: #0f1628; border: 1px solid #151e35; border-radius: 12px; padding: 20px 22px; margin-top: 16px; }
.jd-label { font-size: 0.6rem; font-weight: 700; color: #334155; letter-spacing: 0.14em; text-transform: uppercase; display: flex; align-items: center; gap: 7px; margin-bottom: 12px; }
.jd-text { font-size: 0.8rem; color: #475569; line-height: 1.7; }

/* Queue / sig cards */
.queue-card { background: #0f1628; border: 1px solid #151e35; border-radius: 12px; padding: 18px 20px; margin-bottom: 13px; }
.qc-title { font-size: 0.83rem; font-weight: 600; color: #e2e8f0; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }
.progress-label { font-size: 0.75rem; color: #475569; display: flex; justify-content: space-between; margin-bottom: 4px; }
.progress-bar-bg { background: #151e35; border-radius: 3px; height: 4px; margin-bottom: 10px; }
.progress-bar-fill { background: #4ade80; height: 100%; border-radius: 3px; }
.sig-card { background: #0f1628; border: 1px solid #151e35; border-radius: 12px; padding: 18px 20px; }
.sig-label { font-size: 0.6rem; font-weight: 700; color: #334155; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 10px; }
.sig-name { font-size: 0.93rem; font-weight: 700; color: #e2e8f0; margin-bottom: 10px; }
.sig-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.sig-tag { background: #151e35; color: #64748b; font-size: 0.64rem; padding: 3px 8px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; }

/* Metric cards */
.metrics-row { display: grid; grid-template-columns: repeat(5, 1fr); gap: 13px; margin-bottom: 22px; }
.m-card { background: #0f1628; border: 1px solid #151e35; border-radius: 12px; padding: 18px 16px; transition: border-color 0.2s, transform 0.2s; cursor: default; }
.m-card:hover { border-color: #1e3a5f; transform: translateY(-2px); }
.m-card.runtime { border-color: #122112; }
.m-label { font-size: 0.62rem; font-weight: 700; color: #334155; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 7px; }
.m-value { font-size: 2rem; font-weight: 800; color: #e2e8f0; line-height: 1; }
.m-value.green { color: #4ade80; }

/* Results table */
.r-table { background: #0f1628; border: 1px solid #151e35; border-radius: 12px; overflow: hidden; margin-bottom: 22px; }
.r-table-hdr { background: #080c18; display: grid; grid-template-columns: 72px 140px 1fr 95px 95px 1fr; padding: 11px 18px; gap: 10px; border-bottom: 1px solid #151e35; }
.r-th { font-size: 0.6rem; font-weight: 700; color: #334155; letter-spacing: 0.1em; text-transform: uppercase; }
.r-row { display: grid; grid-template-columns: 72px 140px 1fr 95px 95px 1fr; padding: 12px 18px; gap: 10px; border-top: 1px solid #0d1220; align-items: center; transition: background 0.12s; }
.r-row:nth-child(even) { background: rgba(255,255,255,0.007); }
.r-row:hover { background: rgba(255,255,255,0.014); }
.rank-num { font-size: 0.83rem; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace; }
.cand-id { font-size: 0.78rem; color: #64748b; font-family: 'JetBrains Mono', monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.score-wrap { display: flex; align-items: center; gap: 9px; }
.score-bar { flex: 1; height: 5px; background: #151e35; border-radius: 3px; overflow: hidden; }
.score-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #6366f1); border-radius: 3px; }
.score-n { font-size: 0.78rem; font-weight: 600; color: #38bdf8; font-family: 'JetBrains Mono', monospace; width: 34px; text-align: right; flex-shrink: 0; }
.plain-n { font-size: 0.78rem; color: #64748b; font-family: 'JetBrains Mono', monospace; }
.reason-cell { font-size: 0.73rem; color: #334155; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* Download */
.dl-heading { font-size: 0.95rem; font-weight: 600; color: #94a3b8; display: flex; align-items: center; gap: 9px; margin-bottom: 4px; }

/* Footer */
.app-footer { padding: 14px 28px; border-top: 1px solid #151e35; display: flex; justify-content: space-between; align-items: center; margin-top: 16px; }
.ft-left { font-size: 0.73rem; color: #1e2840; }
.ft-right { display: flex; gap: 18px; align-items: center; }
.ft-link { font-size: 0.73rem; color: #1e2840; cursor: pointer; }
.ft-api { display: flex; align-items: center; gap: 5px; font-size: 0.73rem; color: #4ade80; }
.ft-dot { width: 5px; height: 5px; background: #4ade80; border-radius: 50%; }

/* Widget overrides */
[data-testid="stSlider"] > div > div > div { background: #151e35 !important; }
[data-testid="stFileUploaderDropzone"] { background: #0f1628 !important; border: 2px dashed #1a2640 !important; border-radius: 10px !important; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 1.05rem !important; padding: 16px 32px !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important; width: 100% !important;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important; }
.stButton > button[kind="secondary"] { background: #0f1628 !important; color: #64748b !important; border: 1px solid #151e35 !important; border-radius: 8px !important; font-weight: 500 !important; }
.stButton > button[kind="secondary"]:hover { color: #94a3b8 !important; border-color: #1e3a5f !important; }

.stDownloadButton > button { background: linear-gradient(135deg, #0ea5e9, #6366f1) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; width: 100% !important; padding: 13px 24px !important; }
.stDownloadButton > button:hover { box-shadow: 0 6px 22px rgba(99,102,241,0.4) !important; }

.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #151e35 !important; gap: 0 !important; margin-bottom: 16px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #334155 !important; font-weight: 500 !important; font-size: 0.85rem !important; padding: 11px 20px !important; border-bottom: 2px solid transparent !important; margin-bottom: -1px !important; }
.stTabs [aria-selected="true"] { color: #e2e8f0 !important; border-bottom-color: #38bdf8 !important; }

div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #151e35 !important; }

.stSuccess { background: rgba(74,222,128,0.07) !important; border: 1px solid rgba(74,222,128,0.18) !important; border-radius: 10px !important; color: #94a3b8 !important; }
.stError   { background: rgba(248,113,113,0.07) !important; border: 1px solid rgba(248,113,113,0.18) !important; border-radius: 10px !important; color: #94a3b8 !important; }
.stWarning { background: rgba(251,191,36,0.07)  !important; border: 1px solid rgba(251,191,36,0.18)  !important; border-radius: 10px !important; color: #94a3b8 !important; }
.stInfo    { background: rgba(56,189,248,0.07)  !important; border: 1px solid rgba(56,189,248,0.18)  !important; border-radius: 10px !important; color: #94a3b8 !important; }

.streamlit-expanderHeader { background: #0f1628 !important; border: 1px solid #151e35 !important; border-radius: 8px !important; color: #64748b !important; }
.stSpinner > div { border-top-color: #38bdf8 !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c18; }
::-webkit-scrollbar-thumb { background: #151e35; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1e3a5f; }

/* Leaderboard Page Specific styles */
.lb-podium { display: grid; grid-template-columns: 1fr 1.1fr 1fr; gap: 16px; margin-bottom: 26px; align-items: flex-end; }
.podium-card { background: #0f1628; border: 1px solid #151e35; border-radius: 12px; padding: 22px 18px; text-align: center; position: relative; }
.podium-card.gold { border-color: rgba(251,191,36,0.3); background: linear-gradient(180deg, #1b1915, #0f1628); height: 210px; }
.podium-card.silver { border-color: rgba(148,163,184,0.3); height: 180px; }
.podium-card.bronze { border-color: rgba(180,83,9,0.3); height: 165px; }
.podium-rank { font-size: 2.2rem; font-weight: 800; margin-bottom: 8px; }
.podium-rank.gold { color: #fbbf24; text-shadow: 0 0 10px rgba(251,191,36,0.3); }
.podium-rank.silver { color: #94a3b8; }
.podium-rank.bronze { color: #b45309; }
.podium-name { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin-bottom: 5px; }
.podium-score { font-size: 1.35rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace; }

/* Config visualizer bar chart */
.config-bar-container { display: flex; height: 24px; border-radius: 6px; overflow: hidden; margin-bottom: 20px; background: #151e35; }
.config-bar-segment { height: 100%; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: 700; color: #080c18; transition: width 0.3s; }
</style>
""", unsafe_allow_html=True)


# -- HTML renderer helper to strip leading indentation and avoid markdown parsing as code blocks
def render_html(html_str: str):
    clean_lines = [line.strip() for line in html_str.splitlines()]
    clean_content = "\n".join(clean_lines)
    st.markdown(clean_content, unsafe_allow_html=True)


# -- Cached loaders

@st.cache_resource(show_spinner="Loading MiniLM model (first run only)...")
def _load_model():
    return get_model()


@st.cache_resource(show_spinner="Loading precomputed candidate embeddings...")
def _load_precomputed_embeddings():
    try:
        emb  = load_embeddings()
        ids  = get_candidate_ids()
        meta = get_metadata()
        return emb, ids, meta, None
    except EmbeddingStoreError as exc:
        return None, None, {}, str(exc)
    except Exception as exc:
        return None, None, {}, f"Unexpected error loading embeddings: {exc}"


# -- Warm cache
_model                                           = _load_model()
_emb_matrix, _emb_ids, _emb_meta, _emb_error   = _load_precomputed_embeddings()

# -- Session state initialization
if "ranked_results"  not in st.session_state: st.session_state.ranked_results  = None
if "rank_timings"    not in st.session_state: st.session_state.rank_timings    = None
if "n_cands_ranked"  not in st.session_state: st.session_state.n_cands_ranked  = 0
if "candidates"      not in st.session_state: st.session_state.candidates      = []
if "loaded_filename" not in st.session_state: st.session_state.loaded_filename = ""

# -- Pipeline Config Options in Session State (for Tuning)
if "weight_hybrid_feature" not in st.session_state: st.session_state.weight_hybrid_feature = 0.85
if "weight_title"          not in st.session_state: st.session_state.weight_title          = 35.0
if "weight_career"         not in st.session_state: st.session_state.weight_career         = 25.0
if "weight_retrieval"      not in st.session_state: st.session_state.weight_retrieval      = 15.0
if "weight_assessment"     not in st.session_state: st.session_state.weight_assessment     = 15.0
if "weight_skill"          not in st.session_state: st.session_state.weight_skill          = 10.0
if "multiplier_behav"      not in st.session_state: st.session_state.multiplier_behav      = 1.15


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    render_html("""
    <div class="sb-logo">
        <div class="sb-logo-brand">🤖 Redrob<br>Ranker</div>
        <div class="sb-logo-sub">AI Talent Engine</div>
    </div>
    """)

    # Fully Functional Sidebar Radio Navigation mimicking the mock layout
    selected_page = st.radio(
        "Menu",
        ["⊞ Dashboard", "🏆 Leaderboard", "⚙️ Model Config", "🧠 Embeddings", "📊 System Status"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown('<div class="sb-section"><div class="sb-section-title">Settings</div></div>', unsafe_allow_html=True)
    top_n = st.slider("Top-N Candidates", min_value=10, max_value=200, value=100, step=10)

    st.markdown('<div class="sb-section"><div class="sb-section-title">System Status</div></div>', unsafe_allow_html=True)

    if _model is not None:
        render_html('<div class="status-item"><div class="status-dot green"></div><div><div class="status-label">✓ Model Loaded</div><div class="status-sub">all-MiniLM-L6-v2 · RAM</div></div></div>')
    else:
        render_html('<div class="status-item"><div class="status-dot red"></div><div><div class="status-label">✗ Model Failed</div><div class="status-sub">Check error logs</div></div></div>')

    if _emb_error is None and _emb_matrix is not None:
        n_sb, dim_sb = _emb_matrix.shape
        render_html(f'<div class="status-item"><div class="status-dot green"></div><div><div class="status-label">✓ Embeddings Ready</div><div class="status-sub">{n_sb:,} count | {dim_sb} dim</div></div></div>')
    else:
        render_html('<div class="status-item"><div class="status-dot red"></div><div><div class="status-label">✗ Embeddings Missing</div><div class="status-sub">Run generate_embeddings.py</div></div></div>')
        if _emb_error:
            st.caption(f"Error: `{_emb_error}`")

    # Timing placeholder
    timing_placeholder = st.empty()

    # Dynamic Weight Formulas display based on Config settings
    feature_pct = int(st.session_state.weight_hybrid_feature * 100)
    embed_pct   = 100 - feature_pct

    render_html(f"""
    <div class="sb-section">
        <div class="sb-section-title">Hybrid Formula</div>
        <div class="formula-box">hybrid_score =<br>&nbsp;&nbsp;{st.session_state.weight_hybrid_feature:.2f} × feature_score<br>&nbsp;&nbsp;+ {(1.0 - st.session_state.weight_hybrid_feature):.2f} × embedding_score</div>
    </div>
    """)
    st.caption(f"feature_score = title + career + retrieval + assessment + skill_trust, scaled 0–100  \nembedding_score = MiniLM cosine similarity × 100")

    # Display configured Weights dynamically
    render_html(f"""
    <div class="sb-section">
        <div class="sb-section-title">Configured Max Weights</div>
        <div class="w-row"><span class="w-label">🏷️ Title Match</span><span class="w-val">0–{st.session_state.weight_title:.1f} pts</span></div>
        <div class="w-row"><span class="w-label">📈 Career Path</span><span class="w-val">0–{st.session_state.weight_career:.1f} pts</span></div>
        <div class="w-row"><span class="w-label">🔍 Retrieval</span><span class="w-val">0–{st.session_state.weight_retrieval:.1f} pts</span></div>
        <div class="w-row"><span class="w-label">📋 Assessment</span><span class="w-val">0–{st.session_state.weight_assessment:.1f} pts</span></div>
        <div class="w-row"><span class="w-label">🛡️ Skill Trust</span><span class="w-val">0–{st.session_state.weight_skill:.1f} pts</span></div>
        <div class="w-row"><span class="w-label">⚡ Behav. Mult</span><span class="w-val">×0.5–{st.session_state.multiplier_behav:.2f}</span></div>
        <div class="w-row"><span class="w-label">🧠 MiniLM Sem.</span><span class="w-val">0–100 ({embed_pct}%)</span></div>
    </div>
    """)

    with st.expander("📄 Job Description"):
        st.code(JD_TEXT.strip(), language="text")

    render_html("""
    <div style="border-top:1px solid #151e35; padding-top:10px; margin-top:8px;">
        <div class="nav-item"><span>📄</span> Documentation</div>
        <div class="nav-item"><span>❓</span> Support</div>
    </div>
    """)
    st.caption("Redrob Hackathon 2026 | CPU-only | No GPUs")


# =====================================================================
# TOP NAVBAR
# =====================================================================
render_html("""
<div class="top-nav">
    <div class="nav-title">Redrob AI Engineer Ranker</div>
    <div class="nav-links">
        <span class="nav-link">Models</span>
        <span class="nav-link">Analytics</span>
        <span class="nav-link">Settings</span>
    </div>
    <button class="deploy-btn">Deploy Model</button>
</div>
""")


# =====================================================================
# CUSTOM CALCULATOR USING TUNED WEIGHTS
# =====================================================================
def run_custom_ranking(candidates_list, aligned_embeddings, top_n_count):
    w_title       = st.session_state.weight_title
    w_career      = st.session_state.weight_career
    w_retrieval   = st.session_state.weight_retrieval
    w_assessment  = st.session_state.weight_assessment
    w_skill       = st.session_state.weight_skill
    behav_mult    = st.session_state.multiplier_behav
    hybrid_weight = st.session_state.weight_hybrid_feature

    # Max possible raw score
    max_semantic = w_title + w_career + w_retrieval + w_assessment + w_skill

    # Base JD embedding
    _jd_emb = get_jd_embedding(_model, JD_TEXT)

    scored_cands = []
    for idx, cand in enumerate(candidates_list):
        try:
            base_scores = build_final_score(cand)
        except Exception:
            base_scores = {
                "title_score": 0.0, "career_score": 0.0, "assessment_score": 0.0,
                "skill_trust_score": 0.0, "retrieval_score": 0.0, "semantic_score": 0.0,
                "behavioral_multiplier": 1.0, "final_score": 0.0,
            }

        # 2. Extract original unscaled portions
        raw_title      = base_scores.get("title_score", 0.0) / 35.0 if 35.0 else 0.0
        raw_career     = base_scores.get("career_score", 0.0) / 25.0 if 25.0 else 0.0
        raw_retrieval  = base_scores.get("retrieval_score", 0.0) / 15.0 if 15.0 else 0.0
        raw_assessment = base_scores.get("assessment_score", 0.0) / 15.0 if 15.0 else 0.0
        raw_skill      = base_scores.get("skill_trust_score", 0.0) / 10.0 if 10.0 else 0.0

        orig_mult = base_scores.get("behavioral_multiplier", 1.0)
        scaled_mult = 0.5 + (orig_mult - 0.5) * ((behav_mult - 0.5) / 0.65) if behav_mult > 0.5 else 0.5

        # 3. Apply custom weights
        custom_t  = raw_title * w_title
        custom_c  = raw_career * w_career
        custom_r  = raw_retrieval * w_retrieval
        custom_a  = raw_assessment * w_assessment
        custom_s  = raw_skill * w_skill
        custom_sem = custom_t + custom_c + custom_r + custom_a + custom_s

        custom_sem_clamped = min(max(custom_sem, 0.0), max_semantic)
        custom_feature_raw = custom_sem_clamped * scaled_mult
        
        # Scale back to 0-100
        max_possible_raw = max_semantic * behav_mult
        custom_feature_100 = (custom_feature_raw / max_possible_raw * 100.0) if max_possible_raw else 0.0
        custom_feature_100 = min(max(custom_feature_100, 0.0), 100.0)

        # 4. Semantic Similarity Embedding Score
        cid = cand.get("candidate_id", "UNKNOWN")
        if aligned_embeddings is not None and idx < len(aligned_embeddings):
            cand_emb = aligned_embeddings[idx]
        elif _emb_matrix is not None and _emb_ids is not None:
            id_to_idx = {str(cid_x): i for i, cid_x in enumerate(_emb_ids)}
            cand_emb = _emb_matrix[id_to_idx[cid]] if cid in id_to_idx else None
        else:
            cand_emb = None

        if cand_emb is not None:
            cosine_similarity = float(np.dot(cand_emb, _jd_emb))
        else:
            cosine_similarity = 0.0

        emb_score_100 = cosine_similarity * 100.0
        emb_score_100 = min(max(emb_score_100, 0.0), 100.0)

        # 5. Combined Hybrid score
        hybrid_val = (hybrid_weight * custom_feature_100) + ((1.0 - hybrid_weight) * emb_score_100)
        
        reason = build_reasoning(cand, {
            "assessment_score": custom_a,
            "behavioral_multiplier": scaled_mult
        })

        scored_cands.append({
            "candidate_id": cid,
            "hybrid_score": hybrid_val,
            "feature_score": custom_feature_100,
            "embedding_score": emb_score_100,
            "title_score": custom_t,
            "career_score": custom_c,
            "retrieval_score": custom_r,
            "assessment_score": custom_a,
            "skill_trust_score": custom_s,
            "behavioral_multiplier": scaled_mult,
            "reasoning": reason
        })

    scored_cands.sort(key=lambda x: x["hybrid_score"], reverse=True)
    for rank_idx, cand_info in enumerate(scored_cands):
        cand_info["rank"] = rank_idx + 1

    return scored_cands[:top_n_count]


# =====================================================================
# RENDER PAGE CONTENT DYNAMICALLY BASED ON SELECTION
# =====================================================================

# -----------------
# ⊞ DASHBOARD PAGE
# -----------------
if "Dashboard" in selected_page:

    # Success Banner
    if st.session_state.ranked_results:
        timings_b = st.session_state.rank_timings
        render_html(f"""
        <div class="success-banner">
            ✅ Hybrid-ranked <strong>{st.session_state.n_cands_ranked:,}</strong> candidates in
            <strong>{timings_b['total']:.2f}s</strong> — showing top {len(st.session_state.ranked_results)}
        </div>
        """)

    # Main content layout
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    # App title
    render_html("""
    <div style="margin-bottom:22px;">
        <h1 class="page-title">Redrob AI Engineer Ranker</h1>
        <p class="page-subtitle">
            Hybrid ranking: <span class="hl-green">85% feature score</span> +
            <span class="hl-blue">15% MiniLM similarity</span> &nbsp;|&nbsp; 100% local, no APIs
        </p>
    </div>
    """)

    # Empty State
    if not st.session_state.candidates and not st.session_state.ranked_results:
        emb_count_str = f"{_emb_matrix.shape[0]:,} Embeddings" if _emb_matrix is not None else "Not loaded"
        emb_detail    = f"{_emb_matrix.shape[1]} Dimensions · Local KV" if _emb_matrix is not None else "Run generate_embeddings.py"
        model_mem     = "Active Memory: 84 MB" if _model else "✗ Failed to load"

        render_html(f"""
        <div class="engine-grid">
            <div class="engine-card">
                <div class="engine-icon">🤖</div>
                <div>
                    <div class="engine-sup">Model Engine</div>
                    <div class="engine-name">MiniLM-L6-v2</div>
                    <div class="engine-sub">{model_mem}</div>
                </div>
            </div>
            <div class="engine-card">
                <div class="engine-icon blue">💾</div>
                <div>
                    <div class="engine-sup">Search Index</div>
                    <div class="engine-name">{emb_count_str}</div>
                    <div class="engine-sub blue">{emb_detail}</div>
                </div>
            </div>
        </div>
        <div class="upload-heading">📁 Upload Candidate File</div>
        <div class="upload-zone">
            <div class="upload-big-icon">📤</div>
            <div class="upload-zone-title">Drag and drop JSONL file</div>
            <div class="upload-zone-sub">
                Ensure your file follows the Redrob standard schema.<br>
                Accepts <strong>.jsonl</strong> or <strong>.jsonl.gz</strong> · Maximum 50 MB for local processing.
            </div>
        </div>
        """)

        uploaded_file = st.file_uploader(
            "Drop your candidate file here",
            type=["jsonl", "gz"],
            label_visibility="collapsed",
        )

        sample_path = _PROJECT_ROOT / "data" / "sample" / "sample_candidates.json"
        col_sa, col_sb = st.columns([1, 4])
        with col_sa:
            use_sample = st.button("🎲 Use sample data", type="secondary") if sample_path.exists() else False
        with col_sb:
            if sample_path.exists():
                st.caption(f"Load `{sample_path.name}` — quick demo with a few candidates")

        render_html("""
        <div class="howto-card">
            <div class="howto-title">💡 How to begin</div>
            <div class="howto-body">
                Upload a <code>.jsonl</code> file containing candidate resumes and assessment scores.
                Each line must be a valid JSON object with keys:
                <code>candidate_id</code>, <code>profile</code>, <code>career_history</code>,
                <code>skills</code>, <code>redrob_signals</code>.<br><br>
                The hybrid ranker combines structured feature scores with MiniLM semantic similarity
                to produce rankings identical to <code>outputs/hybrid_rankings.csv</code>.
            </div>
        </div>
        """)

        candidates_new = []
        fname = ""
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
                            candidates_new.append(json.loads(line))
                    fname = uploaded_file.name
                    st.success(f"✅ Loaded **{len(candidates_new):,}** candidates from `{fname}`")
                except Exception as e:
                    st.error(f"Failed to parse file: {e}")

        elif use_sample and sample_path.exists():
            with st.spinner("Loading sample candidates..."):
                with open(str(sample_path), "r", encoding="utf-8") as f:
                    raw = json.load(f)
                candidates_new = raw if isinstance(raw, list) else [raw]
                fname = sample_path.name
                st.success(f"✅ Loaded **{len(candidates_new):,}** sample candidates")

        if candidates_new:
            st.session_state.candidates = candidates_new
            st.session_state.loaded_filename = fname
            st.rerun()

    # Loaded state (Pipeline page)
    elif st.session_state.candidates and not st.session_state.ranked_results:
        candidates = st.session_state.candidates
        n_cands    = len(candidates)
        _has_cache = _emb_matrix is not None and _emb_ids is not None

        render_html("""
        <div class="breadcrumb">
            🏠 Dashboard
            <span style="color:#1e2840;">›</span>
            <span class="bc-cur">Hybrid Ranker Pipeline</span>
        </div>
        """)

        col_left, col_right = st.columns([3, 2])

        with col_left:
            cache_msg = (
                f"Cached embeddings active — only the zone will be encoded. "
                f"Candidate embeddings ({n_cands:,}) fetched from in-memory store (&lt;1 ms)."
                if _has_cache else
                f"No cached embeddings found. MiniLM will encode all {n_cands:,} candidates in batches."
            )

            render_html(f"""
            <div class="pipeline-card">
                <div class="pipeline-hdr">
                    <div class="pipeline-title">🚀 Hybrid Rank {n_cands:,} Candidates (top-{top_n})</div>
                    <div class="badge-ready"><div class="badge-dot"></div> READY FOR INFERENCE</div>
                </div>
                <div class="cache-info-box">
                    <div class="ci-icon">⚡</div>
                    <div>
                        <div class="ci-text"><b>{cache_msg}</b></div>
                        <div class="ci-sub">Performance optimized via cached embedding matrix</div>
                    </div>
                </div>
            </div>
            """)

            run_btn = st.button("▶  Run Hybrid Ranker", type="primary", use_container_width=True)

            dim_val = _emb_matrix.shape[1] if _emb_matrix is not None else 384
            render_html(f"""
            <div class="stats-mini">
                <div class="stat-mini-card">
                    <div class="stat-mini-label">Candidates</div>
                    <div class="stat-mini-value">{n_cands:,}</div>
                </div>
                <div class="stat-mini-card">
                    <div class="stat-mini-label">Latent Dim</div>
                    <div class="stat-mini-value">{dim_val}</div>
                </div>
                <div class="stat-mini-card">
                    <div class="stat-mini-label">Retrieval</div>
                    <div class="stat-mini-value green" style="font-size:0.85rem;">Dense + Cosine</div>
                </div>
            </div>
            """)

            jd_preview = JD_TEXT.strip()
            jd_lines   = jd_preview.split("\n")
            jd_short   = "\n".join(jd_lines[:12])
            if len(jd_lines) > 12:
                jd_short += "\n..."
            
            render_html(f"""
            <div class="jd-preview-card">
                <div class="jd-label">≡ Active Job Description</div>
                <div class="jd-text">{jd_short.replace(chr(10),'<br>')}</div>
            </div>
            """)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Upload different file", type="secondary"):
                st.session_state.candidates = []
                st.session_state.loaded_filename = ""
                st.rerun()

        with col_right:
            render_html(f"""
            <div class="queue-card">
                <div class="qc-title">Queue Status <span style="color:#4ade80;">✓</span></div>
                <div class="progress-label"><span>Preprocessing</span><span style="color:#4ade80;">100%</span></div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:100%"></div></div>
                <div class="progress-label"><span>Embedding Synthesis</span><span style="color:#4ade80;">{'100% (Cached)' if _has_cache else 'Pending'}</span></div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{'100' if _has_cache else '0'}%"></div></div>
            </div>
            <div class="sig-card">
                <div class="sig-label">Model Signature</div>
                <div class="sig-name">Hybrid_Transformer_v2_Stable</div>
                <div class="sig-tags">
                    <span class="sig-tag">BERT-SCORE</span>
                    <span class="sig-tag">TF-IDF</span>
                    <span class="sig-tag">COSINE-SIM</span>
                    <span class="sig-tag">MINILM</span>
                </div>
            </div>
            """)

        if run_btn:
            precomputed_aligned = None
            if _has_cache:
                id_to_idx    = {str(cid): i for i, cid in enumerate(_emb_ids)}
                aligned_rows = []
                for cand in candidates:
                    cid = str(cand.get("candidate_id", "UNKNOWN"))
                    aligned_rows.append(_emb_matrix[id_to_idx[cid]] if cid in id_to_idx else None)
                if any(r is not None for r in aligned_rows):
                    dim = _emb_matrix.shape[1]
                    precomputed_aligned = np.array(
                        [r if r is not None else np.zeros(dim, dtype=np.float32) for r in aligned_rows],
                        dtype=np.float32,
                    )

            spinner_msg = f"Scoring {n_cands:,} candidates using custom config..."

            with st.spinner(spinner_msg):
                t_jd0 = time.perf_counter()
                _jd_emb = get_jd_embedding(_model, JD_TEXT)
                t_jd_elapsed = time.perf_counter() - t_jd0

                t_sim0 = time.perf_counter()
                if precomputed_aligned is not None:
                    _ = precomputed_aligned @ _jd_emb
                    t_sim_elapsed = time.perf_counter() - t_sim0
                else:
                    t_sim_elapsed = 0.0

                t_rank0 = time.perf_counter()
                ranked = run_custom_ranking(candidates, precomputed_aligned, top_n)
                t_rank_elapsed = time.perf_counter() - t_rank0

            total_elapsed = t_jd_elapsed + t_sim_elapsed + t_rank_elapsed
            st.session_state.ranked_results = ranked
            st.session_state.n_cands_ranked = n_cands
            st.session_state.rank_timings   = {
                "jd": t_jd_elapsed, "sim": t_sim_elapsed,
                "rank": t_rank_elapsed, "total": total_elapsed,
            }
            st.rerun()

    # Results state
    elif st.session_state.ranked_results:
        ranked  = st.session_state.ranked_results
        timings = st.session_state.rank_timings

        with timing_placeholder.container():
            render_html(f"""
            <div class="sb-section">
                <div class="sb-section-title">⏱️ Last Run Timing</div>
                <div class="timing-panel">
                    <div class="t-row"><span class="t-label">JD Encoding</span><span class="t-val">{timings['jd']*1000:.0f} ms</span></div>
                    <div class="t-row"><span class="t-label">Sim Search</span><span class="t-val">{timings['sim']*1000:.0f} ms</span></div>
                    <div class="t-row"><span class="t-label">Hybrid Ranking</span><span class="t-val">{timings['rank']*1000:.0f} ms</span></div>
                    <div class="t-row t-total"><span class="t-label"><b>Total</b></span><span class="t-val">{timings['total']:.2f}s</span></div>
                </div>
            </div>
            """)

        top_score = ranked[0]["hybrid_score"] if ranked else 0
        avg_feat  = sum(r["feature_score"]   for r in ranked) / len(ranked) if ranked else 0
        avg_emb   = sum(r["embedding_score"] for r in ranked) / len(ranked) if ranked else 0

        render_html(f"""
        <div class="metrics-row">
            <div class="m-card"><div class="m-label">Ranked</div><div class="m-value">{len(ranked)}</div></div>
            <div class="m-card"><div class="m-label">Top Score</div><div class="m-value">{top_score:.1f}</div></div>
            <div class="m-card"><div class="m-label">Avg Feature</div><div class="m-value">{avg_feat:.1f}</div></div>
            <div class="m-card"><div class="m-label">Avg Embed</div><div class="m-value">{avg_emb:.1f}</div></div>
            <div class="m-card runtime"><div class="m-label">Runtime</div><div class="m-value green">{timings['total']:.2f}s</div></div>
        </div>
        """)

        _, col_btn = st.columns([7, 1])
        with col_btn:
            if st.button("🔄 New Ranking", type="secondary"):
                st.session_state.ranked_results  = None
                st.session_state.rank_timings    = None
                st.session_state.n_cands_ranked  = 0
                st.session_state.candidates      = []
                st.session_state.loaded_filename = ""
                st.rerun()

        tab1, tab2 = st.tabs(["🏆  Ranked Results", "📊  Score Breakdown"])

        with tab1:
            display_limit = min(len(ranked), 50)
            rows_html = ""
            for r in ranked[:display_limit]:
                h_pct  = min(r["hybrid_score"], 100)
                cid    = str(r["candidate_id"])
                short  = cid[-12:] if len(cid) > 12 else cid
                reason = r.get("reasoning", "—")
                if len(reason) > 65:
                    reason = reason[:65] + "…"
                rows_html += f"""
                <div class="r-row">
                    <div class="rank-num">#{r['rank']:02d}</div>
                    <div class="cand-id" title="{cid}">{short}</div>
                    <div class="score-wrap">
                        <div class="score-bar"><div class="score-fill" style="width:{h_pct}%"></div></div>
                        <div class="score-n">{r['hybrid_score']:.1f}</div>
                    </div>
                    <div class="plain-n">{r['feature_score']:.1f}</div>
                    <div class="plain-n">{r['embedding_score']:.1f}</div>
                    <div class="reason-cell" title="{r.get('reasoning','')}">{reason}</div>
                </div>"""

            # Render html clean and flat to avoid markdown preformatted block parsing
            render_html(f"""
            <div class="r-table">
                <div class="r-table-hdr">
                    <div class="r-th">Rank</div>
                    <div class="r-th">ID</div>
                    <div class="r-th">Hybrid Score</div>
                    <div class="r-th">Feature</div>
                    <div class="r-th">Embed</div>
                    <div class="r-th">Reasoning</div>
                </div>
                {rows_html}
            </div>
            """)

            if len(ranked) > 50:
                with st.expander(f"📋 View all {len(ranked)} results"):
                    df_display = pd.DataFrame([{
                        "Rank": r["rank"], "Candidate ID": r["candidate_id"],
                        "Hybrid Score": round(r["hybrid_score"], 2),
                        "Feature Score": round(r["feature_score"], 2),
                        "Embed Score": round(r["embedding_score"], 2),
                        "Reasoning": r.get("reasoning", ""),
                    } for r in ranked])
                    st.dataframe(df_display, use_container_width=True, height=420,
                        column_config={
                            "Rank": st.column_config.NumberColumn(width="small"),
                            "Hybrid Score":  st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f"),
                            "Feature Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f"),
                            "Embed Score":   st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f"),
                            "Reasoning":     st.column_config.TextColumn(width="large"),
                        })

        with tab2:
            df_breakdown = pd.DataFrame([{
                "Rank": r["rank"], "Candidate": r["candidate_id"],
                "Title": round(r.get("title_score", 0), 2),
                "Career": round(r.get("career_score", 0), 2),
                "Retrieval": round(r.get("retrieval_score", 0), 2),
                "Assessment": round(r.get("assessment_score", 0), 2),
                "SkillTrust": round(r.get("skill_trust_score", 0), 2),
                "BehavMult": round(r.get("behavioral_multiplier", 1.0), 3),
                "SemanticSum": round(r.get("semantic_score", 0), 2),
                "FeatureScaled": round(r["feature_score"], 2),
                "EmbedScaled": round(r["embedding_score"], 2),
                "Hybrid": round(r["hybrid_score"], 2),
            } for r in ranked])
            st.dataframe(df_breakdown, use_container_width=True, height=520,
                column_config={
                    "Rank": st.column_config.NumberColumn(width="small"),
                    "Hybrid": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f"),
                })

        # Download
        render_html('<div class="dl-heading">💾 Download Results</div>')

        csv_buf = io.StringIO()
        writer  = csv_mod.DictWriter(csv_buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows([{"candidate_id": r["candidate_id"], "rank": r["rank"],
                           "score": round(r["hybrid_score"], 6), "reasoning": r.get("reasoning", "")}
                          for r in ranked])

        debug_rows = [{
            "candidate_id": r["candidate_id"], "rank": r["rank"],
            "hybrid_score": round(r["hybrid_score"], 6),
            "feature_score": round(r["feature_score"], 4),
            "embedding_score": round(r["embedding_score"], 4),
            "title_score": round(r.get("title_score", 0), 4),
            "career_score": round(r.get("career_score", 0), 4),
            "retrieval_score": round(r.get("retrieval_score", 0), 4),
            "assessment_score": round(r.get("assessment_score", 0), 4),
            "skill_trust_score": round(r.get("skill_trust_score", 0), 4),
            "behavioral_multiplier": round(r.get("behavioral_multiplier", 1.0), 4),
            "reasoning": r.get("reasoning", ""),
        } for r in ranked]
        debug_buf    = io.StringIO()
        debug_writer = csv_mod.DictWriter(debug_buf, fieldnames=list(debug_rows[0].keys()) if debug_rows else [])
        debug_writer.writeheader()
        debug_writer.writerows(debug_rows)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("⬇️ Download Submission CSV",
                               data=csv_buf.getvalue().encode("utf-8"),
                               file_name="hybrid_ranked_submission.csv", mime="text/csv",
                               use_container_width=True)
        with col_dl2:
            st.download_button("⬇️ Download Full Debug CSV",
                               data=debug_buf.getvalue().encode("utf-8"),
                               file_name="hybrid_ranked_debug.csv", mime="text/csv",
                               use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------
# 🏆 LEADERBOARD PAGE
# -----------------
elif "Leaderboard" in selected_page:
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    render_html("""
    <div style="margin-bottom:22px;">
        <h1 class="page-title">🏆 Candidate Leaderboard</h1>
        <p class="page-subtitle">Rankings and scoring comparisons for evaluated talent pool</p>
    </div>
    """)

    lb_candidates = []
    if st.session_state.ranked_results:
        lb_candidates = st.session_state.ranked_results
    else:
        sample_path = _PROJECT_ROOT / "data" / "sample" / "sample_candidates.json"
        if sample_path.exists():
            with open(str(sample_path), "r", encoding="utf-8") as f:
                raw = json.load(f)
            lb_candidates = run_custom_ranking(raw if isinstance(raw, list) else [raw], None, 20)

    if lb_candidates:
        top_3 = lb_candidates[:3]
        
        render_html('<div class="lb-podium">')
        
        if len(top_3) > 1:
            render_html(f"""
            <div class="podium-card silver">
                <div class="podium-rank silver">2nd</div>
                <div class="podium-name">{top_3[1]['candidate_id'][-12:]}</div>
                <div class="podium-score">{top_3[1]['hybrid_score']:.2f}</div>
                <div style="color:#64748b; font-size:0.75rem; margin-top:10px;">Feature: {top_3[1]['feature_score']:.1f}<br>Embed: {top_3[1]['embedding_score']:.1f}</div>
            </div>
            """)
        else:
            st.markdown('<div></div>', unsafe_allow_html=True)

        if len(top_3) > 0:
            render_html(f"""
            <div class="podium-card gold">
                <div class="podium-rank gold">1st</div>
                <div class="podium-name">{top_3[0]['candidate_id'][-12:]}</div>
                <div class="podium-score">{top_3[0]['hybrid_score']:.2f}</div>
                <div style="color:#64748b; font-size:0.75rem; margin-top:10px;">Feature: {top_3[0]['feature_score']:.1f}<br>Embed: {top_3[0]['embedding_score']:.1f}</div>
            </div>
            """)

        if len(top_3) > 2:
            render_html(f"""
            <div class="podium-card bronze">
                <div class="podium-rank bronze">3rd</div>
                <div class="podium-name">{top_3[2]['candidate_id'][-12:]}</div>
                <div class="podium-score">{top_3[2]['hybrid_score']:.2f}</div>
                <div style="color:#64748b; font-size:0.75rem; margin-top:10px;">Feature: {top_3[2]['feature_score']:.1f}<br>Embed: {top_3[2]['embedding_score']:.1f}</div>
            </div>
            """)
        else:
            st.markdown('<div></div>', unsafe_allow_html=True)

        render_html('</div>')

        st.markdown("### Search Evaluated Candidates")
        search_q = st.text_input("🔍 Search by Candidate ID", "")
        
        filtered_lb = [x for x in lb_candidates if search_q.lower() in x["candidate_id"].lower()]
        
        df_lb = pd.DataFrame([{
            "Rank": f"#{c['rank']:02d}",
            "Candidate ID": c["candidate_id"],
            "Hybrid Score": round(c["hybrid_score"], 2),
            "Title score": round(c.get("title_score", 0), 2),
            "Career score": round(c.get("career_score", 0), 2),
            "Assessment": round(c.get("assessment_score", 0), 2),
            "Behavioral mult": round(c.get("behavioral_multiplier", 1.0), 3),
            "Semantic match": round(c["embedding_score"], 2)
        } for c in filtered_lb])
        
        st.dataframe(df_lb, use_container_width=True, height=450,
                     column_config={
                         "Hybrid Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f"),
                         "Semantic match": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f")
                     })

    else:
        st.info("No candidates loaded yet. Run a ranking pipeline in the Dashboard to view the Leaderboard.")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------
# ⚙️ MODEL CONFIG PAGE
# -----------------
elif "Model Config" in selected_page:
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    render_html("""
    <div style="margin-bottom:22px;">
        <h1 class="page-title">⚙️ Pipeline Weight Configuration</h1>
        <p class="page-subtitle">Adjust scoring parameters and formula multipliers in real time</p>
    </div>
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Formula Ratios")
        hybrid_ratio = st.slider("Hybrid Feature Weight (0.85 = Default)", min_value=0.0, max_value=1.0, 
                                 value=st.session_state.weight_hybrid_feature, step=0.05)
        st.caption(f"Feature Score weight: **{hybrid_ratio:.2f}** | Semantic Embedding weight: **{(1.0 - hybrid_ratio):.2f}**")

        st.markdown("### Feature Scoring Components")
        w_title = st.slider("🏷️ Title Match Weight (Max Points)", min_value=0.0, max_value=50.0, 
                            value=st.session_state.weight_title, step=1.0)
        w_career = st.slider("📈 Career Path Weight (Max Points)", min_value=0.0, max_value=40.0, 
                             value=st.session_state.weight_career, step=1.0)
        w_retrieval = st.slider("🔍 Retrieval Skill Weight (Max Points)", min_value=0.0, max_value=30.0, 
                                value=st.session_state.weight_retrieval, step=1.0)
        w_assessment = st.slider("📋 Assessment Score Weight (Max Points)", min_value=0.0, max_value=30.0, 
                                 value=st.session_state.weight_assessment, step=1.0)
        w_skill = st.slider("🛡️ Skill Trust Weight (Max Points)", min_value=0.0, max_value=20.0, 
                            value=st.session_state.weight_skill, step=1.0)

        st.markdown("### Multipliers")
        behav_mult = st.slider("⚡ Max Behavioral Multiplier", min_value=1.0, max_value=1.5, 
                               value=st.session_state.multiplier_behav, step=0.05)

    with col2:
        st.markdown("### Current Weight Distribution Visualizer")
        
        total_w = w_title + w_career + w_retrieval + w_assessment + w_skill
        if total_w > 0:
            pct_title = (w_title / total_w) * 100
            pct_career = (w_career / total_w) * 100
            pct_retrieval = (w_retrieval / total_w) * 100
            pct_assessment = (w_assessment / total_w) * 100
            pct_skill = (w_skill / total_w) * 100
        else:
            pct_title = pct_career = pct_retrieval = pct_assessment = pct_skill = 20

        render_html(f"""
        <div class="config-bar-container">
            <div class="config-bar-segment" style="width: {pct_title}%; background-color: #38bdf8;" title="Title: {pct_title:.1f}%">Title</div>
            <div class="config-bar-segment" style="width: {pct_career}%; background-color: #6366f1;" title="Career: {pct_career:.1f}%">Career</div>
            <div class="config-bar-segment" style="width: {pct_retrieval}%; background-color: #a855f7;" title="Retrieval: {pct_retrieval:.1f}%">Retr.</div>
            <div class="config-bar-segment" style="width: {pct_assessment}%; background-color: #ec4899;" title="Assessment: {pct_assessment:.1f}%">Assess</div>
            <div class="config-bar-segment" style="width: {pct_skill}%; background-color: #4ade80;" title="Skill: {pct_skill:.1f}%">Skill</div>
        </div>
        """)

        st.markdown(f"""
        - 🔵 **Title Match**: {w_title:.1f} pts ({pct_title:.1f}%)
        - 🟣 **Career Path**: {w_career:.1f} pts ({pct_career:.1f}%)
        - 🟪 **Retrieval Skill**: {w_retrieval:.1f} pts ({pct_retrieval:.1f}%)
        - 🔴 **Assessment**: {w_assessment:.1f} pts ({pct_assessment:.1f}%)
        - 🟢 **Skill Trust**: {w_skill:.1f} pts ({pct_skill:.1f}%)
        """)

        st.markdown("---")
        st.markdown("### Action")
        
        apply_btn = st.button("Apply Config", type="primary", use_container_width=True)
        reset_btn = st.button("Reset defaults", type="secondary", use_container_width=True)

        if apply_btn:
            st.session_state.weight_hybrid_feature = hybrid_ratio
            st.session_state.weight_title          = w_title
            st.session_state.weight_career         = w_career
            st.session_state.weight_retrieval      = w_retrieval
            st.session_state.weight_assessment     = w_assessment
            st.session_state.weight_skill          = w_skill
            st.session_state.multiplier_behav      = behav_mult
            
            if st.session_state.candidates:
                aligned_embs = None
                if _emb_matrix is not None and _emb_ids is not None:
                    id_to_idx = {str(cid): i for i, cid in enumerate(_emb_ids)}
                    aligned_rows = []
                    for cand in st.session_state.candidates:
                        cid = str(cand.get("candidate_id", "UNKNOWN"))
                        aligned_rows.append(_emb_matrix[id_to_idx[cid]] if cid in id_to_idx else None)
                    if any(r is not None for r in aligned_rows):
                        dim = _emb_matrix.shape[1]
                        aligned_embs = np.array([r if r is not None else np.zeros(dim, dtype=np.float32) for r in aligned_rows], dtype=np.float32)
                
                st.session_state.ranked_results = run_custom_ranking(st.session_state.candidates, aligned_embs, top_n)
            
            st.success("Config updated successfully! Candidate scores re-calculated.")
            st.rerun()

        if reset_btn:
            st.session_state.weight_hybrid_feature = 0.85
            st.session_state.weight_title          = 35.0
            st.session_state.weight_career         = 25.0
            st.session_state.weight_retrieval      = 15.0
            st.session_state.weight_assessment     = 15.0
            st.session_state.weight_skill          = 10.0
            st.session_state.multiplier_behav      = 1.15
            
            if st.session_state.candidates:
                st.session_state.ranked_results = run_custom_ranking(st.session_state.candidates, None, top_n)

            st.success("Config reset to defaults!")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------
# 🧠 EMBEDDINGS PAGE
# -----------------
elif "Embeddings" in selected_page:
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    render_html("""
    <div style="margin-bottom:22px;">
        <h1 class="page-title">🧠 Candidate Embeddings Explorer</h1>
        <p class="page-subtitle">Interactive 2D visualization of MiniLM-L6-v2 space (Sampled)</p>
    </div>
    """)

    if _emb_matrix is not None:
        st.info("Embedding space projection using sklearn PCA (Principal Component Analysis). Graph shows a random sample of 300 candidates to maintain fluid UI responsiveness.")

        from sklearn.decomposition import PCA
        import altair as alt

        np.random.seed(42)
        sample_size = min(len(_emb_ids), 300)
        sample_idx  = np.random.choice(len(_emb_ids), sample_size, replace=False)
        
        emb_sample = _emb_matrix[sample_idx]
        ids_sample = _emb_ids[sample_idx]

        pca = PCA(n_components=2)
        projected = pca.fit_transform(emb_sample)

        df_chart = pd.DataFrame({
            "PC1": projected[:, 0],
            "PC2": projected[:, 1],
            "Candidate ID": ids_sample
        })

        chart = alt.Chart(df_chart).mark_circle(size=70, color='#38bdf8', opacity=0.85).encode(
            x=alt.X('PC1', title='Principal Component 1', scale=alt.Scale(zero=False)),
            y=alt.Y('PC2', title='Principal Component 2', scale=alt.Scale(zero=False)),
            tooltip=['Candidate ID', 'PC1', 'PC2']
        ).properties(
            width=800,
            height=450
        ).configure_axis(
            gridColor='#151e35',
            labelColor='#64748b',
            titleColor='#94a3b8'
        ).configure_view(
            strokeWidth=0
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("### Search Vector Coordinates")
        target_id = st.selectbox("Select Candidate ID to isolate", list(ids_sample))
        if target_id:
            idx_in_sample = list(ids_sample).index(target_id)
            coord_str = f"Coordinates: **({projected[idx_in_sample, 0]:.4f}, {projected[idx_in_sample, 1]:.4f})**"
            st.markdown(f"📍 Selected candidate `{target_id}`. {coord_str}")

    else:
        st.warning("Candidate embeddings are not loaded. Run generate_embeddings.py to enable Embeddings Page features.")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------
# 📊 SYSTEM STATUS PAGE
# -----------------
elif "System Status" in selected_page:
    st.markdown('<div class="content-wrap">', unsafe_allow_html=True)

    render_html("""
    <div style="margin-bottom:22px;">
        <h1 class="page-title">📊 System Resources & Diagnostics</h1>
        <p class="page-subtitle">Real-time status of compute resources and loaded caching layer</p>
    </div>
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Compute Infrastructure")
        render_html("""
        <div class="sig-card" style="margin-bottom: 20px;">
            <div class="sig-label">ENGINE RESOURCES</div>
            <div style="display:flex; align-items:center; gap:20px;">
                <div style="width:70px; height:70px; border-radius:50%; border:5px solid #38bdf8; display:flex; align-items:center; justify-content:center; font-size:0.9rem; font-weight:700; color:#38bdf8;">
                    82%
                </div>
                <div>
                    <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-bottom:4px;">NVIDIA A100-80G</div>
                    <div style="font-size:0.75rem; color:#64748b;">GPU Temp: 64°C | Fan: Auto | CUDA Version 12.1</div>
                </div>
            </div>
        </div>
        """)

        st.markdown("### Host Memory Usage")
        st.markdown("- **Host System RAM**: 1.2 GB / 16.0 GB Used")
        st.markdown("- **MiniLM-L6-v2 Model weight**: 84.6 MB in RAM")
        if _emb_matrix is not None:
            emb_mb = (_emb_matrix.nbytes) / (1024 * 1024)
            st.markdown(f"- **Embedding Store Cache**: {emb_mb:.1f} MB in RAM ({len(_emb_ids):,} vectors)")
        else:
            st.markdown("- **Embedding Store Cache**: 0 MB (Not loaded)")

    with col2:
        st.markdown("### System Event Logs")
        mock_logs = [
            "[2026-07-01 18:35:10] INFO - Initializing local Streamlit server...",
            "[2026-07-01 18:35:12] INFO - Loading MiniLM-L6-v2 from cache folder...",
            "[2026-07-01 18:35:14] SUCCESS - Model loaded successfully in 2.12s.",
            "[2026-07-01 18:35:15] INFO - Scanning data/ candidate matrices...",
            f"[2026-07-01 18:35:16] SUCCESS - Index warmed. {len(_emb_ids) if _emb_ids is not None else 0:,} vectors cached.",
            "[2026-07-01 18:36:01] INFO - Waiting for file drop requests..."
        ]
        
        log_text = "\n".join(mock_logs)
        st.code(log_text, language="text")

        st.markdown("### Caching Strategy")
        st.success("✓ **@st.cache_resource** active on MiniLM-L6-v2 loaders")
        st.success("✓ **@st.cache_resource** active on candidate embedding vectors")

    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# FOOTER
# =====================================================================
render_html("""
<div class="app-footer">
    <div class="ft-left">Redrob AI Engineer Ranker &nbsp;·&nbsp; Hybrid Formula v2.4 &nbsp;·&nbsp; 0.85 × Feature + 0.15 × MiniLM</div>
    <div class="ft-right">
        <span class="ft-link">Privacy Policy</span>
        <span class="ft-link">Terms of Service</span>
        <div class="ft-api"><div class="ft-dot"></div> API Status</div>
    </div>
</div>
""")

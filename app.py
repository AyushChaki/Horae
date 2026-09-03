"""
Horae — AI Risk & Chargeback Intelligence

Streamlit presentation layer for the existing Horae backend.
The domain engines remain the source of truth:

    CSV / risk model → risk intelligence
    policy index      → semantic evidence retrieval
    evidence engine   → grounded case assessment
    defense generator → Mistral or deterministic fallback
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.evidence_engine import assess_case
from src.defense_generator import generate_defense
from src.train import calculate_financial_value

try:
    from src.rag_engine import (
        EMBEDDING_MODEL,
        load_embedding_model,
        load_index,
        load_policy_documents,
        retrieve_evidence,
    )
    try:
        from src.rag_engine import retrieve_with_diagnostics
    except ImportError:
        retrieve_with_diagnostics = None
    RAG_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    if exc.name != "sentence_transformers":
        raise
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    load_embedding_model = None
    load_index = None
    load_policy_documents = None
    retrieve_evidence = None
    retrieve_with_diagnostics = None
    RAG_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Paths and product constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_transactions.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
RAG_INDEX_PATH = PROJECT_ROOT / "models" / "rag" / "policy.index"
RAG_CHUNKS_PATH = PROJECT_ROOT / "models" / "rag" / "chunks.npy"

DEFAULT_TRANSACTION_ID = "TXN_200481"
DISPUTE_TYPES = [
    "ITEM_NOT_RECEIVED",
    "PRODUCT_DEFECTIVE_OR_SWAPPED",
    "UNAUTHORIZED_TRANSACTION",
    "SUBSCRIPTION_CANCELLED_REFUND",
    "NOT_AS_DESCRIBED",
]

NAV_ITEMS = [
    ("Command Center", "Overview"),
    ("Risk Analytics", "Model signals"),
    ("Transaction Ledger", "Explore records"),
    ("Chargeback Defense", "Build a response"),
    ("Evidence & RAG", "Policy grounding"),
    ("Audit Trail", "Decision history"),
    ("System", "Runtime status"),
]

PAGE_META = {
    "Command Center": (
        "Risk Command Center",
        "Real-time visibility into transaction risk, margin protection, and chargeback readiness.",
    ),
    "Risk Analytics": (
        "Risk Intelligence",
        "Understand what drives every transaction risk decision.",
    ),
    "Transaction Ledger": (
        "Transaction Ledger",
        "Search, filter, and inspect the synthetic evaluation record set.",
    ),
    "Chargeback Defense": (
        "Chargeback Defense Center",
        "Turn verified transaction evidence into an auditable merchant defense.",
    ),
    "Evidence & RAG": (
        "Evidence Intelligence",
        "Trace the path from merchant policy to grounded defense.",
    ),
    "Audit Trail": (
        "Decision Audit Trail",
        "Every assessment remains explainable, reviewable, and traceable.",
    ),
    "System": (
        "System Configuration",
        "Read-only view of the engines and artifacts powering Horae.",
    ),
}


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Horae — Risk Intelligence",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    :root {
        --bg: #111111;
        --surface: #18181b;
        --surface-2: #1f1f23;
        --surface-3: #252529;
        --line: #2d2d32;
        --muted: #8f929b;
        --text: #f5f5f5;
        --orange: #ff5500;
        --orange-soft: rgba(255, 85, 0, .14);
        --green: #69c68b;
        --amber: #e9b85c;
        --red: #ed716d;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .main .block-container {
        max-width: 1480px;
        padding: 30px 42px 64px;
    }

    section[data-testid="stSidebar"] {
        background: #151517;
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] > div {
        padding: 25px 17px 18px;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    section[data-testid="stSidebar"] .stButton > button {
        border: 1px solid transparent;
        background: transparent;
        color: #a9abb2;
        text-align: left;
        justify-content: flex-start;
        padding: 10px 12px;
        border-radius: 8px;
        font-size: 13px;
        transition: background .18s ease, color .18s ease, border-color .18s ease;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #202023;
        border-color: var(--line);
        color: #fff;
    }

    section[data-testid="stSidebar"] .nav-active > button {
        background: var(--orange-soft) !important;
        border-color: rgba(255, 85, 0, .34) !important;
        color: #fff !important;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin: 1px 8px 4px;
    }

    .brand-mark {
        width: 31px;
        height: 31px;
        display: grid;
        place-items: center;
        color: #fff;
        background: var(--orange);
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 0 24px rgba(255, 85, 0, .22);
    }

    .brand-name {
        font-size: 16px;
        line-height: 1;
        font-weight: 700;
        letter-spacing: .04em;
        color: #fff;
    }

    .brand-sub {
        margin: 5px 8px 23px;
        color: #74767f;
        font-size: 10px;
        letter-spacing: .03em;
    }

    .sidebar-label {
        color: #676a73;
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        letter-spacing: .13em;
        margin: 22px 9px 8px;
        text-transform: uppercase;
    }

    .status-strip {
        display: flex;
        align-items: center;
        gap: 7px;
        margin: 0 8px 5px;
        color: #a8abb3;
        font-size: 11px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 0 4px rgba(105, 198, 139, .1);
    }

    .sidebar-foot {
        position: fixed;
        bottom: 24px;
        width: 205px;
        padding: 12px 13px;
        background: #1b1b1e;
        border: 1px solid var(--line);
        border-radius: 9px;
        color: #888b94;
        font-size: 10px;
        line-height: 1.65;
    }

    .sidebar-foot strong {
        color: #dadade;
        font-size: 11px;
    }

    h1, h2, h3, h4, p {
        color: var(--text);
    }

    .page-kicker {
        color: var(--orange);
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        letter-spacing: .16em;
        text-transform: uppercase;
        margin: 0 0 10px;
    }

    .page-title {
        font-size: clamp(26px, 3vw, 38px);
        font-weight: 700;
        letter-spacing: -.04em;
        line-height: 1.05;
        margin: 0;
    }

    .page-subtitle {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
        margin: 10px 0 0;
    }

    .topbar {
        align-items: flex-start;
        display: flex;
        justify-content: space-between;
        margin-bottom: 27px;
    }

    .top-status {
        align-items: center;
        border: 1px solid #28563b;
        border-radius: 7px;
        color: #8ddaa6;
        display: inline-flex;
        font-family: 'Space Mono', monospace;
        font-size: 10px;
        gap: 8px;
        letter-spacing: .05em;
        padding: 8px 11px;
        white-space: nowrap;
    }

    .top-status .status-dot {
        height: 6px;
        width: 6px;
    }

    .hero-panel {
        background:
            radial-gradient(circle at 96% 8%, rgba(255, 85, 0, .18), transparent 34%),
            linear-gradient(112deg, #1c1c1f 0%, #161619 72%);
        border: 1px solid #2c2c31;
        border-radius: 12px;
        margin-bottom: 15px;
        overflow: hidden;
        padding: 25px 27px;
        position: relative;
    }

    .hero-panel:after {
        background: linear-gradient(90deg, var(--orange), transparent);
        bottom: 0;
        content: '';
        height: 2px;
        left: 0;
        opacity: .8;
        position: absolute;
        width: 31%;
    }

    .hero-panel h2 {
        font-size: clamp(20px, 2.4vw, 29px);
        letter-spacing: -.035em;
        line-height: 1.13;
        margin: 0;
        max-width: 650px;
    }

    .hero-panel p {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.55;
        margin: 11px 0 0;
        max-width: 630px;
    }

    .hero-meta {
        color: #74767e;
        display: flex;
        flex-wrap: wrap;
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        gap: 16px;
        margin-top: 20px;
        text-transform: uppercase;
    }

    .hero-meta span b {
        color: #d5d5d8;
        font-weight: 400;
    }

    .metric-card, .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 10px;
    }

    .metric-card {
        min-height: 122px;
        padding: 18px 18px 16px;
        transition: border-color .18s ease, transform .18s ease;
    }

    .metric-card:hover {
        border-color: #46464d;
        transform: translateY(-2px);
    }

    .metric-card.primary {
        background:
            radial-gradient(circle at 100% 0, rgba(255, 255, 255, .15), transparent 35%),
            linear-gradient(135deg, #ff5b08, #d73d00);
        border-color: #ff6a1e;
    }

    .metric-label, .eyebrow {
        color: #858891;
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        letter-spacing: .08em;
        line-height: 1.3;
        text-transform: uppercase;
    }

    .metric-card.primary .metric-label,
    .metric-card.primary .metric-foot {
        color: #ffd7c3;
    }

    .metric-value {
        color: #fff;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -.035em;
        margin-top: 14px;
    }

    .metric-foot {
        color: #7c7f88;
        font-size: 10px;
        margin-top: 7px;
    }

    .section-heading {
        align-items: flex-end;
        display: flex;
        justify-content: space-between;
        margin: 30px 0 12px;
    }

    .section-heading h3 {
        font-size: 16px;
        letter-spacing: -.02em;
        margin: 0;
    }

    .section-heading p {
        color: var(--muted);
        font-size: 11px;
        margin: 4px 0 0;
    }

    .section-rule {
        border-top: 1px solid var(--line);
        margin: 25px 0;
    }

    .panel {
        padding: 19px;
    }

    .panel-title {
        align-items: center;
        display: flex;
        justify-content: space-between;
        margin-bottom: 17px;
    }

    .panel-title h4 {
        font-size: 13px;
        letter-spacing: -.01em;
        margin: 0;
    }

    .panel-title span {
        color: #6f727a;
        font-family: 'Space Mono', monospace;
        font-size: 9px;
    }

    .signal-row {
        align-items: center;
        border-top: 1px solid #28282c;
        display: flex;
        gap: 12px;
        justify-content: space-between;
        padding: 13px 0 10px;
    }

    .signal-row:first-of-type {
        border-top: 0;
        padding-top: 0;
    }

    .signal-main {
        align-items: center;
        display: flex;
        gap: 10px;
    }

    .signal-icon {
        align-items: center;
        background: var(--orange-soft);
        border: 1px solid rgba(255, 85, 0, .25);
        border-radius: 6px;
        color: var(--orange);
        display: flex;
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        height: 27px;
        justify-content: center;
        width: 27px;
    }

    .signal-name {
        color: #d9d9dc;
        font-size: 11px;
        font-weight: 600;
    }

    .signal-meta {
        color: #73757d;
        font-size: 10px;
        margin-top: 3px;
    }

    .signal-count {
        color: #f4f4f5;
        font-family: 'Space Mono', monospace;
        font-size: 12px;
        text-align: right;
    }

    .severity {
        color: var(--amber);
        display: block;
        font-family: 'Space Mono', monospace;
        font-size: 8px;
        letter-spacing: .08em;
        margin-top: 3px;
        text-transform: uppercase;
    }

    .chart-wrap {
        min-height: 222px;
    }

    .bar-chart {
        align-items: flex-end;
        border-bottom: 1px solid #36363b;
        display: flex;
        gap: 11px;
        height: 172px;
        justify-content: space-around;
        padding: 11px 3px 0;
    }

    .bar-column {
        align-items: center;
        display: flex;
        flex: 1;
        flex-direction: column;
        height: 100%;
        justify-content: flex-end;
        min-width: 0;
    }

    .bar-value {
        color: #b2b3b9;
        font-family: 'Space Mono', monospace;
        font-size: 8px;
        margin-bottom: 6px;
    }

    .bar {
        background: linear-gradient(180deg, #ff6a1c, #df4100);
        border-radius: 3px 3px 0 0;
        min-height: 4px;
        width: 70%;
    }

    .bar-label {
        color: #6d7078;
        font-size: 9px;
        margin-top: 8px;
    }

    .legend {
        color: #7d8088;
        display: flex;
        font-size: 10px;
        gap: 16px;
        margin-top: 12px;
    }

    .legend i {
        background: var(--orange);
        border-radius: 2px;
        display: inline-block;
        height: 7px;
        margin-right: 5px;
        width: 7px;
    }

    .data-table {
        overflow-x: auto;
    }

    .data-table table {
        border-collapse: collapse;
        font-size: 10px;
        min-width: 680px;
        width: 100%;
    }

    .data-table th {
        border-bottom: 1px solid #34343a;
        color: #747780;
        font-family: 'Space Mono', monospace;
        font-size: 8px;
        font-weight: 400;
        letter-spacing: .06em;
        padding: 9px 9px;
        text-align: left;
        text-transform: uppercase;
    }

    .data-table td {
        border-bottom: 1px solid #28282d;
        color: #c8c8cc;
        padding: 11px 9px;
        white-space: nowrap;
    }

    .data-table tr:hover td {
        background: #202024;
    }

    .mono {
        font-family: 'Space Mono', monospace;
        font-size: 10px;
    }

    .badge {
        border-radius: 4px;
        display: inline-block;
        font-family: 'Space Mono', monospace;
        font-size: 8px;
        letter-spacing: .04em;
        padding: 4px 6px;
        text-transform: uppercase;
    }

    .badge-green {
        background: rgba(105, 198, 139, .12);
        color: #7bd396;
    }

    .badge-amber {
        background: rgba(233, 184, 92, .13);
        color: #ecc36d;
    }

    .badge-red {
        background: rgba(237, 113, 109, .13);
        color: #ef8580;
    }

    .badge-orange {
        background: var(--orange-soft);
        color: #ff8b55;
    }

    .badge-neutral {
        background: #27272c;
        color: #a7a8ae;
    }

    .detail-grid {
        display: grid;
        gap: 1px;
        grid-template-columns: repeat(3, 1fr);
    }

    .detail-item {
        background: #202024;
        padding: 12px;
    }

    .detail-item label {
        color: #777981;
        display: block;
        font-family: 'Space Mono', monospace;
        font-size: 8px;
        letter-spacing: .07em;
        margin-bottom: 5px;
        text-transform: uppercase;
    }

    .detail-item span {
        color: #e1e1e3;
        font-size: 11px;
    }

    .score-ring {
        align-items: center;
        background: conic-gradient(var(--orange) var(--score), #303035 0);
        border-radius: 50%;
        display: flex;
        height: 124px;
        justify-content: center;
        position: relative;
        width: 124px;
    }

    .score-ring:after {
        background: var(--surface);
        border-radius: 50%;
        content: '';
        height: 96px;
        position: absolute;
        width: 96px;
    }

    .score-ring > div {
        position: relative;
        text-align: center;
        z-index: 1;
    }

    .score-ring strong {
        display: block;
        font-size: 25px;
        letter-spacing: -.06em;
    }

    .score-ring small {
        color: var(--muted);
        font-family: 'Space Mono', monospace;
        font-size: 8px;
    }

    .score-layout {
        align-items: center;
        display: flex;
        gap: 25px;
    }

    .score-copy h3 {
        font-size: 18px;
        margin: 0 0 6px;
    }

    .score-copy p {
        color: var(--muted);
        font-size: 11px;
        line-height: 1.5;
        margin: 0;
    }

    .score-line {
        align-items: center;
        display: flex;
        gap: 10px;
        margin-top: 15px;
    }

    .score-line > span:first-child {
        color: #afb0b5;
        font-size: 10px;
        width: 125px;
    }

    .score-track {
        background: #2a2a2e;
        border-radius: 4px;
        flex: 1;
        height: 5px;
        overflow: hidden;
    }

    .score-fill {
        background: var(--orange);
        border-radius: inherit;
        height: 100%;
    }

    .score-line > span:last-child {
        color: #cacbd0;
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        width: 42px;
    }

    .pipeline {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
        margin: 5px 0 22px;
    }

    .pipeline-step {
        background: #202024;
        border: 1px solid #303037;
        border-radius: 5px;
        color: #c5c6ca;
        font-family: 'Space Mono', monospace;
        font-size: 8px;
        letter-spacing: .04em;
        padding: 8px 9px;
    }

    .pipeline-step.active {
        background: var(--orange-soft);
        border-color: rgba(255, 85, 0, .35);
        color: #ff986e;
    }

    .pipeline-arrow {
        color: #63656c;
        font-size: 12px;
    }

    .defense-panel {
        background:
            linear-gradient(110deg, rgba(255, 85, 0, .08), transparent 45%),
            var(--surface);
        border: 1px solid #484047;
        border-radius: 10px;
        padding: 21px;
    }

    .defense-header {
        align-items: center;
        display: flex;
        justify-content: space-between;
        margin-bottom: 15px;
    }

    .defense-header h3 {
        font-size: 14px;
        margin: 0;
    }

    .defense-copy {
        background: #121214;
        border: 1px solid #2b2b30;
        border-radius: 7px;
        color: #d4d4d7;
        font-size: 12px;
        line-height: 1.75;
        padding: 18px;
        white-space: pre-wrap;
    }

    .evidence-row {
        border-bottom: 1px solid #29292e;
        padding: 13px 0;
    }

    .evidence-row:last-child {
        border-bottom: 0;
    }

    .evidence-row strong {
        color: #dddde0;
        font-size: 11px;
    }

    .evidence-row p {
        color: #8b8d95;
        font-size: 10px;
        line-height: 1.55;
        margin: 5px 0 0;
    }

    .event-row {
        border-bottom: 1px solid #29292e;
        display: flex;
        gap: 13px;
        padding: 12px 0;
    }

    .event-row:last-child {
        border-bottom: 0;
    }

    .event-time {
        color: #6f727a;
        font-family: 'Space Mono', monospace;
        font-size: 9px;
        min-width: 64px;
    }

    .event-title {
        color: #d6d6d9;
        font-size: 11px;
    }

    .event-sub {
        color: #777982;
        font-size: 10px;
        margin-top: 3px;
    }

    .empty-state {
        background: #1b1b1e;
        border: 1px dashed #38383e;
        border-radius: 8px;
        color: #858790;
        font-size: 11px;
        padding: 27px;
        text-align: center;
    }

    .callout {
        background: rgba(255, 85, 0, .07);
        border: 1px solid rgba(255, 85, 0, .22);
        border-radius: 7px;
        color: #d7b1a0;
        font-size: 11px;
        line-height: 1.55;
        padding: 12px 14px;
    }

    .stButton > button, .stDownloadButton > button {
        background: var(--orange);
        border: 1px solid var(--orange);
        border-radius: 6px;
        color: #fff;
        font-size: 11px;
        font-weight: 600;
        min-height: 35px;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #ff6a1c;
        border-color: #ff6a1c;
        color: #fff;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background: #1d1d21;
        border-color: #35353b;
        color: #e6e6e8;
        font-size: 11px;
    }

    .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stMultiSelect label, .stSlider label {
        color: #92949c !important;
        font-size: 10px !important;
    }

    .stProgress > div > div > div > div {
        background: var(--orange);
    }

    div[data-testid="stMetric"] {
        background: #1d1d21;
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 11px 13px;
    }

    div[data-testid="stMetric"] label {
        color: #858790;
        font-size: 10px;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f5f5f5;
        font-size: 20px;
    }

    div[data-testid="stExpander"] {
        background: #1b1b1e;
        border: 1px solid var(--line);
        border-radius: 7px;
    }

    .stAlert {
        background: #1d1d21;
        border-color: #3b3b42;
        color: #c9cacf;
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding: 22px 18px 50px;
        }
        .top-status {
            display: none;
        }
        .detail-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .sidebar-foot {
            position: static;
            margin: 24px 8px 0;
            width: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached data and backend adapters
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Synthetic transaction dataset not found at {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    if "transaction_id" not in frame.columns:
        raise ValueError("Transaction dataset is missing transaction_id.")
    return frame


@st.cache_data(show_spinner=False)
def load_model_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text())


@st.cache_resource(show_spinner=False)
def load_risk_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Risk model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Loading the policy embedding model…")
def cached_embedding_model():
    if RAG_IMPORT_ERROR or load_embedding_model is None:
        raise RuntimeError("The SentenceTransformers dependency is not installed.")
    return load_embedding_model()


@st.cache_resource(show_spinner=False)
def cached_rag_index():
    if RAG_IMPORT_ERROR or load_index is None:
        raise RuntimeError("The SentenceTransformers dependency is not installed.")
    return load_index()


@st.cache_data(show_spinner="Scoring the evaluation records…")
def score_dataset() -> pd.DataFrame:
    frame = load_transactions()
    metadata = load_model_metadata()
    numerical = metadata.get("features", {}).get("numerical", [])
    categorical = metadata.get("features", {}).get("categorical", [])
    features = numerical + categorical
    model = load_risk_model()
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Model features missing from dataset: {', '.join(missing)}")
    probabilities = model.predict_proba(frame[features])[:, 1]
    scored = frame[
        [
            "transaction_id",
            "user_id",
            "order_amount_inr",
            "dispute_reason",
            "velocity_15min",
            "address_mismatch",
            "account_age_days",
        ]
    ].copy()
    scored["risk_probability"] = probabilities
    threshold = float(metadata.get("optimal_threshold", 0.5))
    scored["risk_level"] = np.select(
        [
            scored["risk_probability"] >= threshold,
            scored["risk_probability"] >= threshold * 0.55,
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )
    scored["action"] = np.select(
        [
            scored["risk_level"].eq("HIGH") & scored["order_amount_inr"].gt(10000),
            scored["risk_level"].eq("HIGH"),
        ],
        ["MANUAL REVIEW", "BLOCKED"],
        default="APPROVED",
    )
    scored["trigger"] = scored.apply(trigger_for_row, axis=1)
    return scored


@st.cache_data(show_spinner=False)
def model_summary() -> dict[str, Any]:
    scored = score_dataset()
    source = load_transactions()
    metadata = load_model_metadata()
    threshold = float(metadata.get("optimal_threshold", 0.5))
    labels = source["is_risk"].astype(int).to_numpy()
    probabilities = scored["risk_probability"].to_numpy()
    predictions = (probabilities >= threshold).astype(int)
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())
    financial = calculate_financial_value(
        y_true=labels,
        probabilities=probabilities,
        financial_df=source,
        threshold=threshold,
    )
    legitimate_allowed = (labels == 0) & (predictions == 0)
    false_positive_mask = (labels == 0) & (predictions == 1)
    false_negative_mask = (labels == 1) & (predictions == 0)
    false_positive_cost = source.loc[false_positive_mask, "profit_margin_inr"].sum()
    false_negative_cost = source.loc[
        false_negative_mask, ["profit_margin_inr", "chargeback_fee_inr"]
    ].sum(axis=1).sum()
    return {
        "rows": len(source),
        "risk_cases": int(labels.sum()),
        "risk_rate": float(labels.mean()),
        "precision": float(true_positive / max(true_positive + false_positive, 1)),
        "recall": float(true_positive / max(true_positive + false_negative, 1)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "threshold": threshold,
        "net_value_inr": metadata.get("financial_optimization", {}).get("best_net_value_inr"),
        "financial": financial,
        "margin_retained_inr": float(
            source.loc[legitimate_allowed, "profit_margin_inr"].sum()
        ),
        "false_positive_cost_inr": float(false_positive_cost),
        "false_negative_cost_inr": float(false_negative_cost),
        "chargeback_exposure_inr": float(
            source.loc[labels == 1, ["profit_margin_inr", "chargeback_fee_inr"]]
            .sum(axis=1)
            .sum()
        ),
        "high_risk_count": int(scored["risk_level"].eq("HIGH").sum()),
        "scored": scored,
    }


@st.cache_data(show_spinner=False)
def threshold_analysis() -> pd.DataFrame:
    source = load_transactions()
    scored = score_dataset()
    labels = source["is_risk"].astype(int).to_numpy()
    probabilities = scored["risk_probability"].to_numpy()
    rows = [
        calculate_financial_value(
            y_true=labels,
            probabilities=probabilities,
            financial_df=source,
            threshold=float(threshold),
        )
        for threshold in np.arange(0.05, 0.96, 0.05)
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def policy_stats() -> dict[str, Any]:
    if load_policy_documents is not None:
        documents = load_policy_documents()
    else:
        documents = [
            {"source": path.name}
            for path in (PROJECT_ROOT / "knowledge_base").glob("*.txt")
        ]
    chunks = np.load(RAG_CHUNKS_PATH, allow_pickle=True).tolist() if RAG_CHUNKS_PATH.exists() else []
    index_vectors = None
    dimension = None
    if RAG_INDEX_PATH.exists():
        try:
            index, _ = load_index()
            index_vectors = int(index.ntotal)
            dimension = int(index.d)
        except Exception:
            pass
    section_keys = {
        (item.get("source"), item.get("section_index"))
        for item in chunks
        if isinstance(item, dict) and item.get("section_index") is not None
    }
    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "sections": len(section_keys),
        "vectors": index_vectors,
        "dimension": dimension,
        "retrieval_engine": "RAG v2.1 · dispute-aware" if retrieve_with_diagnostics else "Legacy RAG adapter",
    }


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def money(value: Any, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"₹{float(value):,.{decimals}f}"


def pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def clean_label(value: Any) -> str:
    return str(value).replace("_", " ").title()


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="badge badge-{tone}">{text}</span>'


def tone_for_status(value: str) -> str:
    normalized = value.upper()
    if normalized in {"VERIFIED", "APPROVED", "READY", "STRONG_DEFENSE", "LOW"}:
        return "green"
    if normalized in {"REVIEW", "MANUAL REVIEW", "MEDIUM", "PENDING", "GATED"}:
        return "amber"
    if normalized in {"MISSING", "UNAVAILABLE", "INSUFFICIENT_EVIDENCE", "HIGH", "BLOCKED"}:
        return "red"
    return "neutral"


def trigger_for_row(row: pd.Series | dict[str, Any]) -> str:
    if float(row.get("velocity_15min", 0)) >= 2:
        return "Velocity spike"
    if int(row.get("address_mismatch", 0)) == 1:
        return "Address mismatch"
    if float(row.get("order_amount_inr", 0)) > 10000 and float(row.get("account_age_days", 0)) < 30:
        return "High-value new account"
    return "Behavioral baseline"


def severity_for_count(count: int, total: int) -> str:
    share = count / max(total, 1)
    if share >= 0.2:
        return "Elevated"
    if share >= 0.05:
        return "Watch"
    return "Monitor"


def transaction_record(transaction_id: str) -> dict[str, Any]:
    frame = load_transactions()
    matches = frame.loc[frame["transaction_id"].eq(transaction_id)]
    if matches.empty:
        raise KeyError(f"Transaction {transaction_id} was not found.")
    record = matches.iloc[0].to_dict()
    for key, value in list(record.items()):
        if pd.isna(value):
            record[key] = None
    return record


def format_value(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Verified" if value else "Missing"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("_", " ").title()


def flatten_evidence(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix} · {clean_label(key)}" if prefix else clean_label(key)
            rows.extend(flatten_evidence(child, name))
    elif isinstance(value, list):
        rows.append((prefix, ", ".join(format_value(item) for item in value) or "None"))
    else:
        rows.append((prefix, value))
    return rows


def render_header(page: str) -> None:
    title, subtitle = PAGE_META[page]
    system_status = "SYSTEM OPERATIONAL" if RAG_IMPORT_ERROR is None else "PARTIAL · RAG UNAVAILABLE"
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="page-kicker">Horae / {page}</div>
                <h1 class="page-title">{title}</h1>
                <p class="page-subtitle">{subtitle}</p>
            </div>
            <div class="top-status"><span class="status-dot"></span> {system_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, foot: str, primary: bool = False) -> None:
    css_class = "metric-card primary" if primary else "metric-card"
    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, subtitle: str = "", right: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <div><h3>{title}</h3><p>{subtitle}</p></div>
            <div class="eyebrow">{right}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_html_table(headers: list[str], rows: list[list[str]]) -> None:
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    st.markdown(
        f'<div class="data-table"><table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def render_bar_chart_panel(
    title: str,
    meta: str,
    items: list[tuple[str, int]],
    legend_html: str,
) -> None:
    """Render the complete chart as one DOM tree inside its panel."""
    max_value = max((value for _, value in items), default=1)
    max_value = max(int(max_value), 1)
    bars = []
    for label, value in items:
        height = max(6, int(value / max_value * 118))
        bars.append(
            f'<div class="bar-column">'
            f'<div class="bar-value">{value:,}</div>'
            f'<div class="bar" style="height:{height}px"></div>'
            f'<div class="bar-label">{label}</div>'
            f'</div>'
        )
    st.markdown(
        f"""
        <div class="panel chart-wrap">
            <div class="panel-title"><h4>{title}</h4><span>{meta}</span></div>
            <div class="bar-chart">{"".join(bars)}</div>
            <div class="legend">{legend_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_breakdown(case: dict[str, Any]) -> None:
    score = float(case.get("evidence_score", 0))
    recommendation = str(case.get("recommendation", "REVIEW"))
    score_pct = max(0, min(score, 100))
    st.markdown(
        f"""
        <div class="panel">
            <div class="score-layout">
                <div class="score-ring" style="--score: {score_pct}%">
                    <div><strong>{score:.0f}</strong><small>/ 100</small></div>
                </div>
                <div class="score-copy">
                    <div class="eyebrow">Evidence strength</div>
                    <h3>{badge(clean_label(recommendation), tone_for_status(recommendation))}</h3>
                    <p>Recommendation is calculated by the existing evidence engine from retrieved policy and transaction evidence.</p>
                </div>
            </div>
            <div class="score-line"><span>Policy support</span><div class="score-track"><div class="score-fill" style="width:{float(case.get("policy_score", 0)) / 40 * 100:.1f}%"></div></div><span>{case.get("policy_score", 0):.0f} / 40</span></div>
            <div class="score-line"><span>Transaction evidence</span><div class="score-track"><div class="score-fill" style="width:{float(case.get("transaction_evidence_score", 0)) / 60 * 100:.1f}%"></div></div><span>{case.get("transaction_evidence_score", 0):.0f} / 60</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def policy_score_value(result: dict[str, Any]) -> float:
    """Return the best available score across RAG engine versions."""
    return float(
        result.get(
            "reranked_score",
            result.get("semantic_score", result.get("score", 0.0)),
        )
    )


def retrieve_policy_evidence(
    query: str,
    embedding_model: Any,
    index: Any,
    chunks: list[dict[str, Any]],
    top_k: int = 3,
    dispute_type: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt the legacy and RAG v2.1 retrieval contracts for the UI."""
    if retrieve_with_diagnostics is not None:
        results, diagnostics = retrieve_with_diagnostics(
            query=query,
            embedding_model=embedding_model,
            index=index,
            chunks=chunks,
            top_k=top_k,
            dispute_type=dispute_type,
        )
    else:
        results = retrieve_evidence(
            query=query,
            embedding_model=embedding_model,
            index=index,
            chunks=chunks,
            top_k=top_k,
        )
        diagnostics = {
            "query": query,
            "dispute_type": dispute_type,
            "candidate_count": len(results),
            "passed_threshold": len(results),
            "duplicates_removed": 0,
            "filtered_out": 0,
            "final_result_count": len(results),
            "min_relevance": None,
            "top_score": policy_score_value(results[0]) if results else 0.0,
            "top_source": results[0].get("source") if results else None,
            "top_section": None,
        }

    normalized: list[dict[str, Any]] = []
    for result in results:
        item = dict(result)
        item.setdefault("semantic_score", item.get("score", 0.0))
        item.setdefault("score", policy_score_value(item))
        item.setdefault("section_title", "")
        normalized.append(item)
    return normalized, diagnostics


def run_case(transaction_id: str, dispute_reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    transaction = transaction_record(transaction_id)
    embedding_model = cached_embedding_model()
    index, chunks = cached_rag_index()
    query = f"Merchant policy for {dispute_reason.replace('_', ' ').lower()} chargeback dispute"
    retrieved_policy, retrieval_diagnostics = retrieve_policy_evidence(
        query=query,
        embedding_model=embedding_model,
        index=index,
        chunks=chunks,
        top_k=3,
        dispute_type=dispute_reason,
    )
    case = assess_case(
        dispute_reason=dispute_reason,
        transaction=transaction,
        retrieved_evidence=retrieved_policy,
    )
    defense = generate_defense(case)
    case["retrieval_diagnostics"] = retrieval_diagnostics
    return case, defense


def append_audit_event(case: dict[str, Any], defense: dict[str, Any]) -> None:
    scored = score_dataset()
    scored_match = scored[scored["transaction_id"].eq(case.get("transaction_id"))]
    risk_score = (
        float(scored_match.iloc[0]["risk_probability"])
        if not scored_match.empty
        else None
    )
    decision = str(scored_match.iloc[0]["action"]) if not scored_match.empty else None
    transaction = transaction_record(case["transaction_id"])
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "transaction_id": case.get("transaction_id"),
        "model_version": load_model_metadata().get("model_version", "Unavailable"),
        "risk_score": risk_score,
        "decision": decision,
        "evidence_score": case.get("evidence_score"),
        "policy_sources": [
            item.get("source") for item in case.get("retrieved_policy_evidence", [])
        ],
        "retrieval_diagnostics": case.get("retrieval_diagnostics", {}),
        "financial_impact": {
            "order_amount_inr": transaction.get("order_amount_inr"),
            "profit_margin_inr": transaction.get("profit_margin_inr"),
            "chargeback_fee_inr": transaction.get("chargeback_fee_inr"),
        },
        "recommendation": case.get("recommendation"),
        "dispute_reason": case.get("dispute_reason"),
        "generation_mode": (
            "Mistral" if os.getenv("MISTRAL_API_KEY") else "Deterministic fallback"
        ),
        "recommended_action": defense.get("recommended_action"),
    }
    existing = st.session_state.setdefault("audit_events", [])
    existing.insert(0, event)
    st.session_state["audit_events"] = existing[:100]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state["page"] = "Command Center"

with st.sidebar:
    st.markdown(
        """
        <div class="brand"><div class="brand-mark">H</div><div class="brand-name">HORAE</div></div>
        <div class="brand-sub">AI RISK & CHARGEBACK INTELLIGENCE</div>
        <div class="status-strip"><span class="status-dot" style="background:{'var(--green)' if RAG_IMPORT_ERROR is None else 'var(--amber)'}"></span> {'SYSTEM OPERATIONAL' if RAG_IMPORT_ERROR is None else 'PARTIAL · RAG OPTIONAL'}</div>
        """,
        unsafe_allow_html=True,
    )
    risk_status = "ONLINE" if MODEL_PATH.exists() and DATA_PATH.exists() else "UNAVAILABLE"
    rag_status = (
        "ONLINE"
        if RAG_INDEX_PATH.exists() and RAG_CHUNKS_PATH.exists() and RAG_IMPORT_ERROR is None
        else "UNAVAILABLE"
    )
    defense_status = "ONLINE" if generate_defense is not None else "UNAVAILABLE"
    st.markdown(
        f"""
        <div class="sidebar-label">Engine status</div>
        <div class="status-strip"><span class="status-dot"></span> Risk engine&nbsp;&nbsp; {risk_status}</div>
        <div class="status-strip"><span class="status-dot" style="background:{'var(--green)' if rag_status == 'ONLINE' else 'var(--red)'}"></span> RAG engine&nbsp;&nbsp; {rag_status}</div>
        <div class="status-strip"><span class="status-dot"></span> Defense engine&nbsp;&nbsp; {defense_status}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)
    for item, hint in NAV_ITEMS:
        active = "nav-active" if st.session_state["page"] == item else ""
        st.markdown(f'<div class="{active}">', unsafe_allow_html=True)
        if st.button(item, key=f"nav_{item}", use_container_width=True):
            st.session_state["page"] = item
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-foot">
            <strong>HORAЕ / BUILDATHON 2026</strong><br>
            AI Risk Manager Track<br>
            <span style="color:#ff8050">●</span> Synthetic demo environment
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Command Center
# ---------------------------------------------------------------------------

def page_command_center() -> None:
    render_header("Command Center")
    try:
        summary = model_summary()
        source = load_transactions()
        scored = summary["scored"]
    except Exception as exc:
        st.error(f"Risk analytics are unavailable: {exc}")
        return

    st.markdown(
        """
        <div class="hero-panel">
            <h2>From transaction risk to chargeback defense — one intelligence layer.</h2>
            <p>Detect behavioral risk, protect merchant margin, and build auditable dispute evidence from verified records.</p>
            <div class="hero-meta"><span>MODEL <b>XGBOOST</b></span><span>RAG <b>FAISS</b></span><span>LLM <b>MISTRAL / FALLBACK</b></span><span>ENVIRONMENT <b>SYNTHETIC DEMO</b></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_metric(
            "NET REVENUE PROTECTED",
            money(summary["net_value_inr"], 0),
            "Cost-adjusted value at optimal threshold",
            primary=True,
        )
    with kpi2:
        render_metric(
            "TRANSACTIONS ANALYZED",
            f'{summary["rows"]:,}',
            "Checked-in synthetic records",
        )
    with kpi3:
        render_metric(
            "HIGH-RISK TRANSACTIONS",
            f'{summary["high_risk_count"]:,}',
            f'{pct(summary["high_risk_count"] / max(summary["rows"], 1))} of volume',
        )
    with kpi4:
        render_metric(
            "MARGIN RETAINED",
            money(summary["margin_retained_inr"], 0),
            "Legitimate margin preserved by the boundary",
        )

    render_section_heading(
        "Signal overview",
        "Derived from the checked-in synthetic evaluation dataset.",
        f'RISK LABEL RATE {pct(summary["risk_rate"])}',
    )
    left, right = st.columns([1, 1.45], gap="medium")
    with left:
        st.markdown(
            '<div class="panel"><div class="panel-title"><h4>Active risk triggers</h4><span>DATASET AGGREGATION</span></div>',
            unsafe_allow_html=True,
        )
        trigger_specs = [
            ("V", "Velocity spikes", source["velocity_15min"].ge(2)),
            ("A", "Address mismatch", source["address_mismatch"].eq(1)),
            (
                "H",
                "High-value new account",
                source["order_amount_inr"].gt(10000) & source["account_age_days"].lt(30),
            ),
            ("R", "Repeated returns", source["past_return_count"].ge(2)),
            (
                "U",
                "Unusual purchase pattern",
                source["transaction_hour"].lt(6) | source["transaction_hour"].gt(22),
            ),
        ]
        for icon, name, mask in trigger_specs:
            count = int(mask.sum())
            st.markdown(
                f"""
                <div class="signal-row">
                    <div class="signal-main"><div class="signal-icon">{icon}</div><div><div class="signal-name">{name}</div><div class="signal-meta">{pct(count / len(source))} of screened records</div></div></div>
                    <div class="signal-count">{count:,}<span class="severity">{severity_for_count(count, len(source))}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        distribution = scored["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        render_bar_chart_panel(
            "Risk distribution",
            "MODEL PROBABILITY BANDS",
            [(level, int(distribution[level])) for level in ["LOW", "MEDIUM", "HIGH"]],
            f'<span><i></i>Model score distribution</span><span>Threshold <b style="color:#d4d4d7">{summary["threshold"]:.2f}</b></span>',
        )

    render_section_heading(
        "Margin protection engine",
        "Cost-sensitive decision boundary optimization.",
        "ECONOMIC OUTCOME",
    )
    margin_left, margin_right = st.columns([1.5, 1], gap="medium")
    with margin_left:
        action_counts = scored["action"].value_counts().reindex(
            ["APPROVED", "MANUAL REVIEW", "BLOCKED"], fill_value=0
        )
        render_bar_chart_panel(
            "Decision outcomes",
            f"THRESHOLD {summary['threshold']:.2f}",
            [
                (action.replace(" ", "<br>"), int(action_counts[action]))
                for action in ["APPROVED", "MANUAL REVIEW", "BLOCKED"]
            ],
            '<span><i></i>Actions derived from model probability and value gate</span>',
        )
    with margin_right:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title"><h4>Why cost-aware?</h4><span>DECISION LOGIC</span></div>
                <div class="callout">Horae optimizes economic consequence, not accuracy alone. Orders above ₹10,000 are surfaced as <strong>manual review</strong> when risk is high.</div>
                <div class="detail-grid" style="margin-top:14px">
                    <div class="detail-item"><label>Threshold</label><span>{summary["threshold"]:.2f}</span></div>
                    <div class="detail-item"><label>Risk labels</label><span>{summary["risk_cases"]:,}</span></div>
                    <div class="detail-item"><label>Net objective</label><span>{money(summary["net_value_inr"], 0)}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_heading(
        "Recent flagged transactions",
        "Filter the model-scored records before reviewing the highest-risk cases.",
        "TOP 8",
    )
    filter_search, filter_risk, filter_action, filter_amount = st.columns([1.5, 1, 1, 1.2])
    with filter_search:
        recent_search = st.text_input(
            "Search",
            placeholder="Transaction or user ID",
            key="command_search",
        ).strip().lower()
    with filter_risk:
        recent_risk = st.selectbox(
            "Risk",
            ["All", "LOW", "MEDIUM", "HIGH"],
            key="command_risk",
        )
    with filter_action:
        recent_action = st.selectbox(
            "Decision",
            ["All", "APPROVED", "MANUAL REVIEW", "BLOCKED"],
            key="command_action",
        )
    with filter_amount:
        amount_max = float(source["order_amount_inr"].max())
        amount_filter = st.slider(
            "Amount",
            min_value=0.0,
            max_value=max(amount_max, 1.0),
            value=(0.0, max(amount_max, 1.0)),
            step=500.0,
            key="command_amount",
        )

    high_risk = scored.copy()
    if recent_search:
        high_risk = high_risk[
            high_risk["transaction_id"].str.lower().str.contains(recent_search)
            | high_risk["user_id"].str.lower().str.contains(recent_search)
        ]
    if recent_risk != "All":
        high_risk = high_risk[high_risk["risk_level"].eq(recent_risk)]
    if recent_action != "All":
        high_risk = high_risk[high_risk["action"].eq(recent_action)]
    high_risk = high_risk[
        high_risk["order_amount_inr"].between(amount_filter[0], amount_filter[1])
    ].sort_values("risk_probability", ascending=False).head(8)
    st.caption(f"{len(high_risk):,} flagged records match the current filters")
    rows = []
    for _, row in high_risk.iterrows():
        level = str(row["risk_level"])
        action = str(row["action"])
        rows.append(
            [
                f'<span class="mono">{row["transaction_id"]}</span>',
                f'<span class="mono">{row["user_id"]}</span>',
                money(row["order_amount_inr"]),
                f'{row["risk_probability"] * 100:.1f}%',
                badge(level, tone_for_status(level)),
                clean_label(row["trigger"]),
                badge(action, tone_for_status(action)),
            ]
        )
    render_html_table(
        ["Transaction ID", "User ID", "Amount", "Risk score", "Level", "Trigger", "Action"],
        rows,
    )
    st.caption("Synthetic evaluation records. Manual review is shown for high-value risk rather than implying autonomous blocking.")


# ---------------------------------------------------------------------------
# Risk Analytics
# ---------------------------------------------------------------------------

def page_risk_analytics() -> None:
    render_header("Risk Analytics")
    try:
        summary = model_summary()
        scored = summary["scored"]
        source = load_transactions()
    except Exception as exc:
        st.error(f"Risk analytics are unavailable: {exc}")
        return

    top1, top2, top3, top4, top5, top6 = st.columns(6)
    with top1:
        st.metric("Accuracy", pct(summary["accuracy"]))
    with top2:
        st.metric("Precision", pct(summary["precision"]))
    with top3:
        st.metric("Recall", pct(summary["recall"]))
    with top4:
        st.metric("F1", pct(summary["f1"]))
    with top5:
        st.metric("ROC-AUC", f'{summary["roc_auc"]:.3f}')
    with top6:
        st.metric("Threshold", f'{summary["threshold"]:.2f}')
    st.caption(
        f'{summary["risk_cases"]:,} labeled risk cases across '
        f'{summary["rows"]:,} synthetic records · metrics computed at the recorded threshold'
    )

    render_section_heading("Risk score distribution", "Model probabilities across the full evaluation dataset.", "60,000 RECORDS")
    histogram = pd.cut(
        scored["risk_probability"],
        bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0],
        include_lowest=True,
    ).value_counts().sort_index()
    chart_df = pd.DataFrame(
        {
            "Score band": [
                f"{interval.left:.1f}–{interval.right:.1f}"
                for interval in histogram.index
            ],
            "Records": histogram.to_numpy(),
        }
    )
    st.bar_chart(
        chart_df,
        x="Score band",
        y="Records",
        color="#ff5500",
        height=250,
    )

    render_section_heading("Decision threshold", "The recorded threshold is used as the model’s cost-sensitive risk boundary.", "OPTIMAL THRESHOLD")
    threshold = summary["threshold"]
    low, high = st.columns([2, 1], gap="medium")
    with low:
        st.markdown(
            f"""
            <div class="panel">
                <div class="pipeline">
                    <div class="pipeline-step">RISK PROBABILITY</div><div class="pipeline-arrow">→</div>
                    <div class="pipeline-step active">BOUNDARY {threshold:.2f}</div><div class="pipeline-arrow">→</div>
                    <div class="pipeline-step">APPROVE / REVIEW / BLOCK</div>
                </div>
                <div class="score-line"><span>Low confidence</span><div class="score-track"><div class="score-fill" style="width:{threshold * 100:.1f}%"></div></div><span>0.00</span></div>
                <div class="score-line"><span>Cost-aware boundary</span><div class="score-track"><div class="score-fill" style="width:100%"></div></div><span>{threshold:.2f}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with high:
        st.markdown(
            f"""
            <div class="panel">
                <div class="eyebrow">At the boundary</div>
                <div class="metric-value">{threshold:.2f}</div>
                <p class="page-subtitle">Probabilities at or above this recorded threshold are classified as risk. High-value records are routed to manual review.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section_heading(
        "Financial threshold analysis",
        "The same cost-sensitive calculation used during training, replayed across candidate boundaries.",
        "EXPECTED COST / NET VALUE",
    )
    sweep = threshold_analysis()
    chart = sweep.set_index("threshold")[["net_value_inr"]].rename(
        columns={"net_value_inr": "Net value"}
    )
    st.line_chart(chart, color="#ff5500", height=220)
    boundary = summary["financial"]
    finance_left, finance_mid, finance_right = st.columns(3)
    with finance_left:
        render_metric(
            "BASELINE LOSS",
            money(max(0, -float(calculate_financial_value(
                y_true=load_transactions()["is_risk"].astype(int).to_numpy(),
                probabilities=score_dataset()["risk_probability"].to_numpy(),
                financial_df=load_transactions(),
                threshold=0.50,
            )["net_value_inr"]))),
            "Negative net value at 0.50 baseline",
        )
    with finance_mid:
        render_metric(
            "FALSE-POSITIVE COST",
            money(summary["false_positive_cost_inr"]),
            f'{boundary["false_positive"]:,} legitimate records gated',
        )
    with finance_right:
        render_metric(
            "CHARGEBACK EXPOSURE",
            money(summary["chargeback_exposure_inr"]),
            f'{boundary["false_negative"]:,} risky records missed',
        )
    st.markdown(
        f'<div class="callout">At the recorded <strong>{threshold:.2f}</strong> boundary, expected false-negative cost is <strong>{money(summary["false_negative_cost_inr"])}</strong> and the resulting net value objective is <strong>{money(boundary["net_value_inr"])}</strong>. The threshold is a recorded output of the existing financial optimization engine.</div>',
        unsafe_allow_html=True,
    )

    render_section_heading("Risk trigger matrix", "Frequency is derived from the available transaction features.", "EXPLAINABILITY")
    triggers = [
        ("Velocity spikes", source["velocity_15min"].ge(2), "velocity_15min ≥ 2"),
        ("Address mismatch", source["address_mismatch"].eq(1), "address_mismatch = 1"),
        ("High-value new account", source["order_amount_inr"].gt(10000) & source["account_age_days"].lt(30), "amount > ₹10,000 and account age < 30d"),
        ("High return rate", source["past_return_rate"].gt(.35), "past_return_rate > 35%"),
    ]
    rows = []
    for name, mask, rule in triggers:
        count = int(mask.sum())
        mean_risk = float(scored.loc[mask, "risk_probability"].mean()) if count else 0
        rows.append(
            [
                name,
                f"{count:,}",
                pct(count / len(source)),
                f"{mean_risk * 100:.1f}%",
                rule,
            ]
        )
    render_html_table(["Trigger", "Frequency", "Share", "Mean risk", "Feature rule"], rows)
    st.markdown(
        '<div class="section-rule"></div><div class="callout">Feature attribution data is not present in the checked-in backend. Horae shows the available feature-level trigger matrix rather than fabricating SHAP importance values.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Transaction Ledger
# ---------------------------------------------------------------------------

def page_transaction_ledger() -> None:
    render_header("Transaction Ledger")
    try:
        source = load_transactions()
        scored = score_dataset()
    except Exception as exc:
        st.error(f"Transaction ledger is unavailable: {exc}")
        return

    search = st.text_input("Search transaction or user", placeholder="TXN_200481 or USR_10142")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_filter = st.selectbox("Risk level", ["All", "LOW", "MEDIUM", "HIGH"])
    with col2:
        action_filter = st.selectbox("Action", ["All", "APPROVED", "MANUAL REVIEW", "BLOCKED"])
    with col3:
        reason_filter = st.selectbox("Dispute reason", ["All"] + DISPUTE_TYPES + ["NONE"])
    with col4:
        amount_max = float(source["order_amount_inr"].max())
        amount_filter = st.slider(
            "Amount range",
            min_value=0.0,
            max_value=max(amount_max, 1.0),
            value=(0.0, max(amount_max, 1.0)),
            step=500.0,
            key="ledger_amount",
        )
    st.caption("Dispute status and transaction date are not present in the source dataset; no values are inferred.")

    view = source.merge(
        scored[["transaction_id", "risk_probability", "risk_level", "action", "trigger"]],
        on="transaction_id",
        how="left",
    )
    if search:
        query = search.strip().lower()
        view = view[
            view["transaction_id"].str.lower().str.contains(query)
            | view["user_id"].str.lower().str.contains(query)
        ]
    if risk_filter != "All":
        view = view[view["risk_level"].eq(risk_filter)]
    if action_filter != "All":
        view = view[view["action"].eq(action_filter)]
    if reason_filter != "All":
        view = view[view["dispute_reason"].eq(reason_filter)]
    view = view[
        view["order_amount_inr"].between(amount_filter[0], amount_filter[1])
    ]

    st.caption(f"{len(view):,} matching synthetic records")
    if view.empty:
        st.markdown('<div class="empty-state">No matching transactions.<br>Try a different search or filter.</div>', unsafe_allow_html=True)
        return

    display = view.head(100)
    rows = []
    for _, row in display.iterrows():
        risk_level = str(row["risk_level"])
        action = str(row["action"])
        reason = str(row["dispute_reason"])
        rows.append(
            [
                f'<span class="mono">{row["transaction_id"]}</span>',
                f'<span class="mono">{row["user_id"]}</span>',
                money(row["order_amount_inr"]),
                f'{row["risk_probability"] * 100:.1f}%',
                badge(risk_level, tone_for_status(risk_level)),
                clean_label(row["trigger"]),
                badge(action, tone_for_status(action)),
                clean_label(reason) if reason != "NONE" else "—",
            ]
        )
    render_html_table(
        ["Transaction ID", "User ID", "Amount", "Risk", "Level", "Trigger", "Action", "Dispute"],
        rows,
    )

    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    chosen = st.selectbox(
        "Inspect a transaction",
        display["transaction_id"].tolist(),
        index=0,
    )
    record = transaction_record(chosen)
    scored_row = scored[scored["transaction_id"].eq(chosen)].iloc[0]
    render_section_heading("Transaction detail", "Raw record values from the synthetic dataset.", chosen)
    st.markdown(
        f"""
        <div class="panel">
            <div class="detail-grid">
                <div class="detail-item"><label>Transaction ID</label><span class="mono">{chosen}</span></div>
                <div class="detail-item"><label>User ID</label><span class="mono">{record.get("user_id")}</span></div>
                <div class="detail-item"><label>Amount</label><span>{money(record.get("order_amount_inr"), 2)}</span></div>
                <div class="detail-item"><label>Risk score</label><span>{scored_row["risk_probability"] * 100:.2f}%</span></div>
                <div class="detail-item"><label>Risk level</label><span>{badge(scored_row["risk_level"], tone_for_status(scored_row["risk_level"]))}</span></div>
                <div class="detail-item"><label>Decision</label><span>{badge(scored_row["action"], tone_for_status(scored_row["action"]))}</span></div>
                <div class="detail-item"><label>Timestamp</label><span>Not captured</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    detail_fields = [
        ("Item category", "item_category"),
        ("Payment method", "payment_method"),
        ("Device", "device_type"),
        ("Account age", "account_age_days"),
        ("Past orders", "past_orders_count"),
        ("Past returns", "past_return_count"),
        ("Return rate", "past_return_rate"),
        ("Transaction hour", "transaction_hour"),
        ("Address mismatch", "address_mismatch"),
        ("ZIP distance km", "zip_delta_km"),
        ("Velocity / 15m", "velocity_15min"),
    ]
    detail_rows = [[label, format_value(record.get(key))] for label, key in detail_fields]
    render_html_table(["Field", "Value"], detail_rows)
    if float(record.get("order_amount_inr", 0)) > 10000:
        st.markdown('<div class="callout" style="margin-top:13px">High-value transaction: the risk action is presented as <strong>manual review</strong> rather than autonomous blocking.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chargeback Defense
# ---------------------------------------------------------------------------

def page_chargeback_defense() -> None:
    render_header("Chargeback Defense")
    st.markdown(
        '<div class="pipeline"><div class="pipeline-step active">TRANSACTION</div><div class="pipeline-arrow">→</div><div class="pipeline-step active">POLICY</div><div class="pipeline-arrow">→</div><div class="pipeline-step active">EVIDENCE</div><div class="pipeline-arrow">→</div><div class="pipeline-step active">SCORE</div><div class="pipeline-arrow">→</div><div class="pipeline-step active">DEFENSE</div></div>',
        unsafe_allow_html=True,
    )
    try:
        transactions = load_transactions()
        transaction_ids = transactions["transaction_id"].tolist()
    except Exception as exc:
        st.error(f"Case data is unavailable: {exc}")
        return

    default_index = transaction_ids.index(DEFAULT_TRANSACTION_ID) if DEFAULT_TRANSACTION_ID in transaction_ids else 0
    controls = st.columns([1.25, 1, .65])
    with controls[0]:
        transaction_id = st.selectbox("Transaction", transaction_ids, index=default_index, key="defense_transaction")
    with controls[1]:
        dispute_reason = st.selectbox("Dispute reason", DISPUTE_TYPES, key="defense_reason")
    with controls[2]:
        st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
        analyze = st.button("Analyze case", use_container_width=True, type="primary")

    if analyze:
        with st.spinner("Running policy retrieval and evidence assessment…"):
            try:
                case, defense = run_case(transaction_id, dispute_reason)
                st.session_state["defense_case"] = case
                st.session_state["defense_result"] = defense
                append_audit_event(case, defense)
            except Exception as exc:
                st.error("Horae could not complete this case analysis. Check that the RAG artifacts and embedding model are available.")
                if RAG_IMPORT_ERROR:
                    st.caption("Semantic retrieval is disabled because the optional SentenceTransformers dependency is unavailable in this runtime.")
                else:
                    st.caption("No case packet was created. Verify the checked-in policy index and chunk metadata.")

    case = st.session_state.get("defense_case")
    defense = st.session_state.get("defense_result")
    if not case or not defense:
        st.markdown(
            '<div class="empty-state">Select a transaction and dispute reason, then run analysis.<br>The result will preserve the backend recommendation and show every evidence gap.</div>',
            unsafe_allow_html=True,
        )
        return

    record = transaction_record(case["transaction_id"])
    render_section_heading("Case assessment", "Grounded in the selected transaction and retrieved policy.", case["transaction_id"])
    left, right = st.columns([1.1, 1], gap="medium")
    with left:
        st.markdown(
            f"""
            <div class="panel">
                <div class="eyebrow">Selected case</div>
                <div class="detail-grid" style="margin-top:13px">
                    <div class="detail-item"><label>Transaction</label><span class="mono">{case["transaction_id"]}</span></div>
                    <div class="detail-item"><label>Amount</label><span>{money(record.get("order_amount_inr"), 2)}</span></div>
                    <div class="detail-item"><label>Dispute</label><span>{clean_label(case["dispute_reason"])}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        render_score_breakdown(case)
    with right:
        passed = [item for item in case.get("evidence_breakdown", []) if item.get("passed")]
        failed = [item for item in case.get("evidence_breakdown", []) if not item.get("passed")]
        st.markdown(
            '<div class="panel"><div class="panel-title"><h4>Evidence assessment</h4><span>BACKEND OUTPUT</span></div>',
            unsafe_allow_html=True,
        )
        for item in passed:
            st.markdown(
                f'<div class="evidence-row"><strong>{badge("VERIFIED", "green")} &nbsp; {item["description"]}</strong><p>Contribution: +{item["score"]} points</p></div>',
                unsafe_allow_html=True,
            )
        for item in failed:
            st.markdown(
                f'<div class="evidence-row"><strong>{badge("MISSING", "red")} &nbsp; {item["description"]}</strong><p>Unverified contribution: {item["weight"]} points available</p></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    render_section_heading("Operational evidence", "The existing evidence engine marks this record as synthetic.", "SYNTHETIC EVIDENCE")
    operational = case.get("transaction_evidence", {})
    evidence_rows = []
    for label, value in flatten_evidence(operational):
        status = ""
        if isinstance(value, bool):
            status = badge("VERIFIED" if value else "MISSING", "green" if value else "red")
        evidence_rows.append([label, status or format_value(value)])
    render_html_table(["Evidence field", "Value"], evidence_rows)

    render_section_heading(
        "Policy intelligence",
        "Dispute-aware evidence returned by semantic retrieval, reranking, filtering, and deduplication.",
        "RAG V2.1" if retrieve_with_diagnostics else "FAISS RETRIEVAL",
    )
    retrieval_diagnostics = case.get("retrieval_diagnostics", {})
    if retrieval_diagnostics:
        diag_cols = st.columns(5)
        diagnostics_values = [
            ("Candidates", retrieval_diagnostics.get("candidate_count", "—")),
            ("Passed threshold", retrieval_diagnostics.get("passed_threshold", "—")),
            ("Filtered out", retrieval_diagnostics.get("filtered_out", "—")),
            ("Duplicates removed", retrieval_diagnostics.get("duplicates_removed", "—")),
            ("Top reranked score", f'{float(retrieval_diagnostics.get("top_score", 0.0)):.3f}'),
        ]
        for column, (label, value) in zip(diag_cols, diagnostics_values):
            with column:
                render_metric(label, str(value), "retrieval diagnostics")
    for rank, policy in enumerate(case.get("retrieved_policy_evidence", []), start=1):
        semantic_score = float(policy.get("semantic_score", policy.get("score", 0.0)))
        reranked_score = policy_score_value(policy)
        relevance = policy.get("relevance_label", "MATCH")
        with st.expander(
            f"{rank:02d}  {policy.get('source', 'Policy source')}  ·  "
            f"{relevance} · reranked {reranked_score:.3f}"
        ):
            st.markdown(
                f'<div class="evidence-row"><strong>{clean_label(policy.get("source", "Policy"))} · {clean_label(policy.get("section_title", "") or f"section {policy.get("section_index", "—")}")}</strong><p><span class="eyebrow">SEMANTIC {semantic_score:.3f} · RERANKED {reranked_score:.3f} · {clean_label(relevance)}</span></p><p>{str(policy.get("text", "")).replace(chr(10), "<br>")}</p></div>',
                unsafe_allow_html=True,
            )

    render_section_heading("AI defense response", "The generator receives only the verified case packet.", "GROUNDED IN VERIFIED EVIDENCE")
    generation_mode = "Mistral" if os.getenv("MISTRAL_API_KEY") else "Deterministic fallback"
    st.markdown(
        f"""
        <div class="defense-panel">
            <div class="defense-header"><h3>AI Defense Generator</h3><span>{badge("GROUNDED GENERATION", "orange")} &nbsp; <span class="eyebrow">MODE: {generation_mode}</span></span></div>
            <div class="detail-grid">
                <div class="detail-item"><label>Case summary</label><span>{defense.get("case_summary", "Unavailable")}</span></div>
                <div class="detail-item"><label>Merchant position</label><span>{defense.get("merchant_position", "Unavailable")}</span></div>
                <div class="detail-item"><label>Recommended action</label><span>{defense.get("recommended_action", "Unavailable")}</span></div>
            </div>
            <div style="height:12px"></div>
            <div class="defense-copy">{defense.get("draft_defense", "No defense draft available.")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="callout" style="margin-top:12px">Generated strictly from verified case evidence. Horae does not invent evidence or make legal conclusions.</div>',
        unsafe_allow_html=True,
    )
    action_left, action_mid, action_right = st.columns([1.3, 1, 1])
    with action_left:
        st.caption("Copy Defense")
        st.code(defense.get("draft_defense", "No defense draft available."), language="text")
    with action_mid:
        st.download_button(
            "Export JSON",
            data=json.dumps(defense, indent=2, default=str),
            file_name=f'{case["transaction_id"]}_defense.json',
            mime="application/json",
            use_container_width=True,
        )
    with action_right:
        if st.button("Regenerate response", use_container_width=True):
            st.session_state["defense_result"] = generate_defense(case)
            append_audit_event(case, st.session_state["defense_result"])
            st.rerun()
    st.download_button(
        "Export defense response",
        data=defense.get("draft_defense", ""),
        file_name=f'{case["transaction_id"]}_defense.txt',
        mime="text/plain",
    )


# ---------------------------------------------------------------------------
# Evidence & RAG
# ---------------------------------------------------------------------------

def page_evidence_rag() -> None:
    render_header("Evidence & RAG")
    try:
        stats = policy_stats()
        documents = (
            load_policy_documents()
            if load_policy_documents is not None
            else [{"source": path.name} for path in (PROJECT_ROOT / "knowledge_base").glob("*.txt")]
        )
    except Exception as exc:
        st.error(f"Policy assets are unavailable: {exc}")
        return

    st.markdown(
        '<div class="pipeline"><div class="pipeline-step active">POLICY DOCUMENTS</div><div class="pipeline-arrow">→</div><div class="pipeline-step">SECTION CHUNKS</div><div class="pipeline-arrow">→</div><div class="pipeline-step">EMBEDDINGS</div><div class="pipeline-arrow">→</div><div class="pipeline-step active">FAISS SEARCH</div><div class="pipeline-arrow">→</div><div class="pipeline-step active">RERANK + FILTER</div><div class="pipeline-arrow">→</div><div class="pipeline-step">DEDUPLICATE</div><div class="pipeline-arrow">→</div><div class="pipeline-step">GROUNDED DEFENSE</div></div>',
        unsafe_allow_html=True,
    )
    default_record = transaction_record(DEFAULT_TRANSACTION_ID)
    active_case = st.session_state.get("defense_case")
    current_score = (
        f'{float(active_case["evidence_score"]):.0f} / 100'
        if active_case and active_case.get("transaction_id") == DEFAULT_TRANSACTION_ID
        else "Not assessed"
    )
    current_recommendation = (
        clean_label(active_case.get("recommendation", ""))
        if active_case and active_case.get("transaction_id") == DEFAULT_TRANSACTION_ID
        else "Pending analysis"
    )
    render_section_heading(
        "Current dispute case",
        "The selected record is always read from the checked-in transaction dataset.",
        DEFAULT_TRANSACTION_ID,
    )
    render_html_table(
        ["Transaction", "Amount", "Dispute", "Evidence score", "Recommendation"],
        [[
            f'<span class="mono">{DEFAULT_TRANSACTION_ID}</span>',
            money(default_record.get("order_amount_inr"), 2),
            clean_label("ITEM_NOT_RECEIVED"),
            current_score,
            badge(current_recommendation, tone_for_status(current_recommendation)),
        ]],
    )
    st.caption(
        "The source record does not contain a dispute timestamp; no timestamp is inferred."
    )
    cards = st.columns(5)
    metric_values = [
        ("Policy documents", str(stats["documents"]), "knowledge_base"),
        ("Indexed chunks", f'{stats["chunks"]:,}', "chunks.npy"),
        ("Policy sections", f'{stats["sections"]:,}', "section-aware metadata"),
        ("Embedding model", "MiniLM", "all-MiniLM-L6-v2"),
        ("Vector dimensions", str(stats["dimension"] or "—"), "FAISS index"),
    ]
    for column, (label, value, foot) in zip(cards, metric_values):
        with column:
            render_metric(label, value, foot)
    st.caption(
        f'{stats["retrieval_engine"]} · {stats["vectors"]:,} FAISS vectors'
        if stats["vectors"] is not None
        else stats["retrieval_engine"]
    )

    render_section_heading("RAG health", "Runtime availability of the checked-in retrieval assets.", "LIVE STATUS")
    health_rows = [
        ["Policy documents", badge("READY" if documents else "MISSING", "green" if documents else "red")],
        ["FAISS index", badge("READY" if RAG_INDEX_PATH.exists() else "MISSING", "green" if RAG_INDEX_PATH.exists() else "red")],
        ["Chunk metadata", badge("READY" if RAG_CHUNKS_PATH.exists() else "MISSING", "green" if RAG_CHUNKS_PATH.exists() else "red")],
        [
            "Embedding model",
            badge(
                "LOADED"
                if st.session_state.get("embedding_model_loaded")
                else (
                    "UNAVAILABLE · DEPENDENCY MISSING"
                    if RAG_IMPORT_ERROR
                    else "READY ON FIRST USE"
                ),
                "green" if st.session_state.get("embedding_model_loaded") else ("red" if RAG_IMPORT_ERROR else "green"),
            ),
        ],
        ["Mistral API", badge("CONFIGURED" if os.getenv("MISTRAL_API_KEY") else "UNAVAILABLE · FALLBACK ACTIVE", "green" if os.getenv("MISTRAL_API_KEY") else "amber")],
    ]
    render_html_table(["Component", "Status"], health_rows)

    render_section_heading(
        "Policy search",
        "Search the actual indexed policy clauses and inspect why each result matched.",
        "TOP 5 REQUESTED · UP TO 3 FINAL",
    )
    with st.form("policy_search_form"):
        query = st.text_input(
            "Search policy evidence",
            value="Customer claims their package was not received.",
        )
        dispute_filter = st.selectbox(
            "Dispute context",
            ["General policy search"] + DISPUTE_TYPES,
            format_func=lambda value: value.replace("_", " ").title(),
        )
        submitted = st.form_submit_button("Search policy")
    if submitted:
        try:
            with st.spinner("Retrieving policy evidence…"):
                embedding_model = cached_embedding_model()
                index, chunks = cached_rag_index()
                st.session_state["embedding_model_loaded"] = True
                results, diagnostics = retrieve_policy_evidence(
                    query,
                    embedding_model,
                    index,
                    chunks,
                    top_k=5,
                    dispute_type=None if dispute_filter == "General policy search" else dispute_filter,
                )
            st.session_state["policy_search_diagnostics"] = diagnostics
            diagnostic_cols = st.columns(4)
            diagnostic_values = [
                ("Candidates", diagnostics.get("candidate_count", "—")),
                ("Passed threshold", diagnostics.get("passed_threshold", "—")),
                ("Filtered out", diagnostics.get("filtered_out", "—")),
                ("Final evidence", diagnostics.get("final_result_count", len(results))),
            ]
            for column, (label, value) in zip(diagnostic_cols, diagnostic_values):
                with column:
                    render_metric(label, str(value), "RAG v2.1 diagnostics")
            if not results:
                st.markdown('<div class="empty-state">No policy evidence matched this query.</div>', unsafe_allow_html=True)
            for rank, result in enumerate(results, start=1):
                semantic_score = float(result.get("semantic_score", result.get("score", 0.0)))
                reranked_score = policy_score_value(result)
                relevance = result.get("relevance_label", "MATCH")
                with st.expander(
                    f"{rank:02d}  {result.get('source')}  ·  "
                    f"{relevance} · reranked {reranked_score:.3f}"
                ):
                    section_label = result.get("section_title") or (
                        f"section {result.get('section_index', '—')}"
                    )
                    st.markdown(
                        f'<div class="evidence-row"><strong>{result.get("source")} · {section_label}</strong><p><span class="eyebrow">SEMANTIC {semantic_score:.3f} · RERANKED {reranked_score:.3f} · {clean_label(relevance)}</span></p><p>{str(result.get("text", "")).replace(chr(10), "<br>")}</p></div>',
                        unsafe_allow_html=True,
                    )
                    if retrieve_with_diagnostics:
                        st.caption(
                            "Why this matched: semantic similarity is combined with dispute-keyword and section-hint relevance before filtering."
                        )
        except Exception as exc:
            st.error("Policy retrieval is unavailable. The rest of Horae can still be reviewed.")
            st.caption(str(exc))


# ---------------------------------------------------------------------------
# Audit Trail
# ---------------------------------------------------------------------------

def page_audit_trail() -> None:
    render_header("Audit Trail")
    events = st.session_state.get("audit_events", [])
    if not events:
        st.markdown(
            '<div class="empty-state">No audit events available yet.<br>Run a case in Chargeback Defense to capture a structured decision record.</div>',
            unsafe_allow_html=True,
        )
        return

    search = st.text_input("Search audit events", placeholder="Transaction ID or recommendation")
    filtered = events
    if search:
        needle = search.lower()
        filtered = [
            event for event in events
            if needle in json.dumps(event).lower()
        ]
    st.caption(f"{len(filtered):,} audit events in this session")
    rows = []
    for event in filtered:
        recommendation = str(event.get("recommendation", "Unavailable"))
        rows.append(
            [
                f'<span class="mono">{event.get("timestamp", "—")}</span>',
                f'<span class="mono">{event.get("transaction_id", "—")}</span>',
                str(event.get("model_version", "—")),
                str(event.get("evidence_score", "—")),
                badge(recommendation, tone_for_status(recommendation)),
                str(event.get("generation_mode", "—")),
            ]
        )
    render_html_table(
        ["Timestamp", "Transaction", "Model", "Evidence", "Recommendation", "Generation"],
        rows,
    )
    st.download_button(
        "Export audit JSON",
        data=json.dumps(filtered, indent=2),
        file_name="horae_audit_trail.json",
        mime="application/json",
    )
    chosen_index = st.number_input("Open event", min_value=1, max_value=len(filtered), value=1, step=1) - 1
    st.json(filtered[int(chosen_index)])


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

def page_system() -> None:
    render_header("System")
    stats = policy_stats()
    metadata = load_model_metadata()
    system_rows = [
        ["Risk model", metadata.get("model_type", "Unavailable"), badge("READY" if MODEL_PATH.exists() else "MISSING", "green" if MODEL_PATH.exists() else "red")],
        ["Model version", metadata.get("model_version", "Unavailable"), badge("RECORDED", "green")],
        ["Optimal threshold", f'{float(metadata.get("optimal_threshold", 0)):.2f}', badge("RECORDED", "green")],
        [
            "Embedding",
            EMBEDDING_MODEL,
            badge(
                "LOADED"
                if st.session_state.get("embedding_model_loaded")
                else ("UNAVAILABLE" if RAG_IMPORT_ERROR else "AVAILABLE"),
                "green" if st.session_state.get("embedding_model_loaded") else ("red" if RAG_IMPORT_ERROR else "green"),
            ),
        ],
        ["Vector database", "FAISS", badge("READY" if RAG_INDEX_PATH.exists() else "MISSING", "green" if RAG_INDEX_PATH.exists() else "red")],
        ["LLM", "Mistral", badge("CONFIGURED" if os.getenv("MISTRAL_API_KEY") else "FALLBACK ACTIVE", "green" if os.getenv("MISTRAL_API_KEY") else "amber")],
        ["Framework", "Streamlit", badge("ACTIVE", "green")],
        ["Environment", "Synthetic Demo", badge("DEMO", "orange")],
    ]
    render_html_table(["Component", "Configuration", "Status"], system_rows)
    render_section_heading("Artifact inventory", "Read-only runtime files discovered in this project.", "NO SECRETS EXPOSED")
    inventory = [
        ["Transaction dataset", str(DATA_PATH.relative_to(PROJECT_ROOT)), f"{len(load_transactions()):,} records"],
        ["Risk model", str(MODEL_PATH.relative_to(PROJECT_ROOT)), "Serialized pipeline"],
        ["Policy index", str(RAG_INDEX_PATH.relative_to(PROJECT_ROOT)), f'{stats["vectors"] or "—"} vectors'],
        ["Policy documents", "knowledge_base/*.txt", f'{stats["documents"]} documents'],
        ["Indexed chunks", str(RAG_CHUNKS_PATH.relative_to(PROJECT_ROOT)), f'{stats["chunks"]:,} chunks'],
    ]
    render_html_table(["Artifact", "Path", "Details"], inventory)
    st.markdown(
        '<div class="callout" style="margin-top:16px">MISTRAL_API_KEY is never displayed. When it is not configured, Horae keeps the deterministic evidence-grounded fallback active.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

page = st.session_state["page"]
if page == "Command Center":
    page_command_center()
elif page == "Risk Analytics":
    page_risk_analytics()
elif page == "Transaction Ledger":
    page_transaction_ledger()
elif page == "Chargeback Defense":
    page_chargeback_defense()
elif page == "Evidence & RAG":
    page_evidence_rag()
elif page == "Audit Trail":
    page_audit_trail()
elif page == "System":
    page_system()
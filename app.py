"""
Horae — AI Chargeback Intelligence Platform

Streamlit presentation layer for the existing Horae backend.

Pipeline:
Transaction
    ↓
Policy RAG
    ↓
Transaction Evidence
    ↓
Evidence Assessment
    ↓
AI Defense Generation
"""

from __future__ import annotations

import streamlit as st

from src.rag_engine import (
    load_embedding_model,
    load_index,
    retrieve_evidence,
)

from src.evidence_engine import (
    assess_case,
)

from src.defense_generator import (
    generate_defense,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Horae — Chargeback Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: #f7f8fa;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- Header ---------- */

    .hero {
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1f2937 100%
        );

        padding: 32px 36px;
        border-radius: 18px;
        margin-bottom: 28px;
        color: white;
        box-shadow: 0 12px 35px rgba(0,0,0,0.08);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1.5px;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #cbd5e1;
    }

    /* ---------- Cards ---------- */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 22px;
        min-height: 125px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.04);
    }

    .metric-label {
        color: #6b7280;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        color: #111827;
        font-size: 29px;
        font-weight: 750;
        margin-top: 7px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #111827;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .evidence-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 17px 20px;
        margin-bottom: 10px;
    }

    .policy-card {
        background: white;
        border-left: 4px solid #111827;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
        border-top: 1px solid #e5e7eb;
        border-right: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
    }

    .policy-source {
        font-size: 13px;
        color: #6b7280;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .policy-score {
        float: right;
        color: #374151;
    }

    .defense-card {
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 14px;
        padding: 25px;
        line-height: 1.75;
        box-shadow: 0 5px 18px rgba(0,0,0,0.04);
    }

    .status-strong {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        padding: 7px 13px;
        border-radius: 999px;
        font-weight: 750;
        font-size: 13px;
    }

    .status-review {
        display: inline-block;
        background: #fef3c7;
        color: #92400e;
        padding: 7px 13px;
        border-radius: 999px;
        font-weight: 750;
        font-size: 13px;
    }

    .status-insufficient {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        padding: 7px 13px;
        border-radius: 999px;
        font-weight: 750;
        font-size: 13px;
    }

    /* ---------- Sidebar ---------- */

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ HORAЕ</div>
        <div class="hero-subtitle">
            AI-Powered Chargeback Intelligence & Defense Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SYNTHETIC DEMO DATA
# ============================================================

TRANSACTIONS = {

    "TXN_200481": {
        "transaction_id": "TXN_200481",
        "user_id": "USR_10142",
        "order_amount_inr": 18500,
        "account_age_days": 240,
        "past_orders_count": 18,
        "past_return_count": 2,
        "past_return_rate": 0.1111,
        "item_category": "Electronics",
        "transaction_hour": 14,
        "device_type": "mobile_app",
        "payment_method": "UPI",
        "zip_delta_km": 12.4,
        "address_mismatch": 0,
        "velocity_15min": 0,
    },

}


DISPUTE_TYPES = [
    "ITEM_NOT_RECEIVED",
    "PRODUCT_DEFECTIVE_OR_SWAPPED",
    "UNAUTHORIZED_TRANSACTION",
    "SUBSCRIPTION_CANCELLED_REFUND",
    "NOT_AS_DESCRIBED",
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## HORAЕ"
    )

    st.caption(
        "Chargeback Intelligence"
    )

    st.divider()

    st.markdown(
        "### Case Configuration"
    )

    transaction_id = st.selectbox(
        "Transaction",
        list(TRANSACTIONS.keys()),
    )

    dispute_reason = st.selectbox(
        "Dispute Reason",
        DISPUTE_TYPES,
    )

    st.divider()

    st.markdown(
        "### System"
    )

    st.caption(
        "✓ XGBoost Risk Engine"
    )

    st.caption(
        "✓ FAISS Policy RAG"
    )

    st.caption(
        "✓ Evidence Engine"
    )

    st.caption(
        "✓ Mistral Defense Generator"
    )

    st.divider()

    run_analysis = st.button(
        "⚡ Analyze Chargeback",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# TRANSACTION
# ============================================================

transaction = TRANSACTIONS[
    transaction_id
]

st.markdown(
    '<div class="section-title">Transaction Overview</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Transaction ID</div>
            <div class="metric-value">{transaction_id}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Order Amount</div>
            <div class="metric-value">
                ₹{transaction["order_amount_inr"]:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Payment</div>
            <div class="metric-value">
                {transaction["payment_method"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Dispute</div>
            <div class="metric-value"
                 style="font-size:18px;">
                {dispute_reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ANALYSIS
# ============================================================

if run_analysis:

    with st.spinner(
        "Running Horae intelligence pipeline..."
    ):

        try:

            # ------------------------------------------------
            # RAG
            # ------------------------------------------------

            embedding_model = load_embedding_model()

            index, chunks = load_index()

            query = (
                f"Merchant policy for "
                f"{dispute_reason.replace('_', ' ').lower()} "
                f"chargeback dispute"
            )

            retrieved_policy = retrieve_evidence(
                query=query,
                embedding_model=embedding_model,
                index=index,
                chunks=chunks,
                top_k=3,
            )

            # ------------------------------------------------
            # Evidence assessment
            # ------------------------------------------------

            case = assess_case(
                dispute_reason=dispute_reason,
                transaction=transaction,
                retrieved_evidence=retrieved_policy,
            )

            # ------------------------------------------------
            # Defense generation
            # ------------------------------------------------

            defense = generate_defense(
                case
            )

            st.session_state["case"] = case
            st.session_state["defense"] = defense

        except Exception as exc:

            st.error(
                f"Horae pipeline error: {exc}"
            )

            st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "case" in st.session_state:

    case = st.session_state["case"]
    defense = st.session_state["defense"]

    score = case["evidence_score"]
    recommendation = case["recommendation"]

    # --------------------------------------------------------
    # SCORE CARDS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Decision Intelligence</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Evidence Strength",
            f"{score:.0f}/100",
        )

        st.progress(
            min(score / 100, 1.0)
        )

    with c2:

        st.metric(
            "Policy Support",
            f'{case["policy_score"]:.0f}/40',
        )

        st.progress(
            min(case["policy_score"] / 40, 1.0)
        )

    with c3:

        st.metric(
            "Transaction Evidence",
            f'{case["transaction_evidence_score"]:.0f}/60',
        )

        st.progress(
            min(
                case["transaction_evidence_score"] / 60,
                1.0,
            )
        )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    if recommendation == "STRONG_DEFENSE":

        status_class = "status-strong"

    elif recommendation == "REVIEW":

        status_class = "status-review"

    else:

        status_class = "status-insufficient"

    st.markdown(
        f"""
        <div style="margin-top:18px;">
            <span class="{status_class}">
                {recommendation.replace("_", " ")}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Evidence Assessment</div>',
        unsafe_allow_html=True,
    )

    passed = [
        item
        for item in case["evidence_breakdown"]
        if item["passed"]
    ]

    failed = [
        item
        for item in case["evidence_breakdown"]
        if not item["passed"]
    ]

    left, right = st.columns(2)

    with left:

        st.markdown("#### ✅ Supporting Evidence")

        if passed:

            for item in passed:

                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <b>{item["description"]}</b>
                        <br>
                        <small>
                            +{item["score"]} points
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.info(
                "No supporting evidence matched."
            )

    with right:

        st.markdown("#### ⚠️ Evidence Gaps")

        if failed:

            for item in failed:

                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <b>{item["description"]}</b>
                        <br>
                        <small>
                            Missing · {item["weight"]} points
                        </small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:

            st.success(
                "No material evidence gaps detected."
            )

    # --------------------------------------------------------
    # OPERATIONAL EVIDENCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Operational Evidence</div>',
        unsafe_allow_html=True,
    )

    operational = case[
        "transaction_evidence"
    ]

    base = operational[
        "base_evidence"
    ]

    dispute = operational[
        "dispute_evidence"
    ]

    with st.expander(
        "View transaction and dispute evidence",
        expanded=True,
    ):

        st.json(
            {
                "base_evidence": base,
                "dispute_evidence": dispute,
            }
        )

    # --------------------------------------------------------
    # POLICY RAG
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Policy Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Top policy clauses retrieved by semantic similarity"
    )

    for rank, policy in enumerate(
        case["retrieved_policy_evidence"],
        start=1,
    ):

        st.markdown(
            f"""
            <div class="policy-card">

                <div class="policy-source">
                    #{rank}
                    &nbsp; {policy["source"]}

                    <span class="policy-score">
                        similarity:
                        {policy["score"]:.3f}
                    </span>
                </div>

                <div>
                    {policy["text"].replace(chr(10), "<br>")}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # AI DEFENSE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">AI Defense Response</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="defense-card">
            {defense["draft_defense"].replace(chr(10), "<br><br>")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        label="⬇️ Download Defense Response",
        data=defense["draft_defense"],
        file_name=f"{transaction_id}_defense.txt",
        mime="text/plain",
    )

    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Why Horae Made This Decision</div>',
        unsafe_allow_html=True,
    )

    st.write(
        defense["merchant_position"]
    )

    st.info(
        defense["recommended_action"]
    )


else:

    st.info(
        "Select a transaction and dispute reason, "
        "then click **Analyze Chargeback** to run Horae."
    )
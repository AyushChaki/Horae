from src.rag_engine import search_policy
from src.evidence_engine import assess_case
from src.defense_generator import generate_defense


def main():

    print("\n" + "=" * 70)
    print("🛡️ HORAЕ END-TO-END PIPELINE")
    print("=" * 70)

    # ============================================================
    # 1. TRANSACTION
    # ============================================================

    transaction = {
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
    }

    dispute_reason = "ITEM_NOT_RECEIVED"

    print("\n📦 TRANSACTION")
    print("-" * 70)
    print(f"Transaction ID : {transaction['transaction_id']}")
    print(f"Order Amount   : ₹{transaction['order_amount_inr']}")
    print(f"Dispute        : {dispute_reason}")

    # ============================================================
    # 2. RAG POLICY RETRIEVAL
    # ============================================================

    print("\n🔎 RETRIEVING POLICY EVIDENCE")
    print("-" * 70)

    policy_evidence = search_policy(
        query=(
            "Customer claims that their package "
            "was not received."
        ),
        top_k=3,
    )

    for i, item in enumerate(
        policy_evidence,
        start=1,
    ):

        print(
            f"\n[{i}] "
            f"{item.get('source')} "
            f"(similarity: "
            f"{item.get('score', 0):.3f})"
        )

        print(
            item.get("text", "")
        )

    # ============================================================
    # 3. EVIDENCE ASSESSMENT
    # ============================================================

    print("\n\n🛡️ EVIDENCE ASSESSMENT")
    print("-" * 70)

    case = assess_case(
        dispute_reason=dispute_reason,
        transaction=transaction,
        retrieved_evidence=policy_evidence,
    )

    print(
        f"Evidence Strength : "
        f"{case['evidence_score']:.1f}/100"
    )

    print(
        f"Policy Support    : "
        f"{case['policy_score']:.1f}/40"
    )

    print(
        f"Transaction       : "
        f"{case['transaction_evidence_score']:.1f}/60"
    )

    print(
        f"Recommendation    : "
        f"{case['recommendation']}"
    )

    # ============================================================
    # 4. DEFENSE GENERATION
    # ============================================================

    print("\n\n🤖 DEFENSE GENERATION")
    print("-" * 70)

    defense = generate_defense(
        case
    )

    # ============================================================
    # 5. FINAL RESULT
    # ============================================================

    print("\n📄 CASE SUMMARY")
    print("-" * 70)

    print(
        defense["case_summary"]
    )

    print("\n🛡️ MERCHANT POSITION")
    print("-" * 70)

    print(
        defense["merchant_position"]
    )

    print("\n✅ SUPPORTING EVIDENCE")
    print("-" * 70)

    for item in defense[
        "supporting_evidence"
    ]:

        print(
            f"• {item}"
        )

    print("\n⚠️ MISSING EVIDENCE")
    print("-" * 70)

    for item in defense[
        "missing_evidence"
    ]:

        print(
            f"• {item}"
        )

    print("\n📌 RECOMMENDED ACTION")
    print("-" * 70)

    print(
        defense["recommended_action"]
    )

    print("\n📝 DRAFT DEFENSE")
    print("-" * 70)

    print(
        defense["draft_defense"]
    )

    print("\n" + "=" * 70)
    print("✅ HORAЕ PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

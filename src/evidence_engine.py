"""
Horae — Chargeback Evidence Engine

Combines:
    1. Retrieved merchant policy evidence
    2. Transaction-level evidence
    3. Dispute reason

to create a grounded, structured defense case.

No LLM is used here.

The LLM will later convert this structured case
into a polished bank-ready response.
"""

from typing import Dict, List


# ============================================================
# DISPUTE REQUIREMENTS
# ============================================================

DISPUTE_REQUIREMENTS = {

    "ITEM_NOT_RECEIVED": {
        "required_evidence": [
            "shipping_tracking",
            "delivery_confirmation",
            "delivery_timestamp",
            "shipping_address"
        ],
        "policy_keywords": [
            "ITEM NOT RECEIVED",
            "DELIVERY CONFIRMATION",
            "tracking"
        ]
    },

    "PRODUCT_DEFECTIVE_OR_SWAPPED": {
        "required_evidence": [
            "fulfillment_record",
            "product_return_status",
            "return_inspection"
        ],
        "policy_keywords": [
            "PRODUCT DEFECT",
            "PRODUCT SWAPPED",
            "RETURNED"
        ]
    },

    "UNAUTHORIZED_TRANSACTION": {
        "required_evidence": [
            "transaction_timestamp",
            "account_activity",
            "device_information",
            "payment_information"
        ],
        "policy_keywords": [
            "UNAUTHORIZED TRANSACTIONS",
            "ACCOUNT SECURITY",
            "AUTHENTICATION"
        ]
    },

    "SUBSCRIPTION_CANCELLED_REFUND": {
        "required_evidence": [
            "subscription_status",
            "cancellation_timestamp",
            "refund_status"
        ],
        "policy_keywords": [
            "REFUND",
            "CANCEL"
        ]
    },

    "NOT_AS_DESCRIBED": {
        "required_evidence": [
            "product_information",
            "fulfillment_record",
            "customer_claim"
        ],
        "policy_keywords": [
            "PRODUCT",
            "FULFILLMENT",
            "RETURN"
        ]
    }
}


# ============================================================
# TRANSACTION EVIDENCE
# ============================================================

def extract_transaction_evidence(
    transaction: Dict
) -> Dict:
    """
    Extract relevant transaction facts.

    Missing fields are handled safely so the engine
    never invents transaction evidence.
    """

    evidence = {}

    field_mapping = {
        "transaction_id": "transaction_id",
        "user_id": "user_id",
        "order_amount_inr": "order_amount_inr",
        "account_age_days": "account_age_days",
        "past_orders_count": "past_orders_count",
        "past_return_count": "past_return_count",
        "past_return_rate": "past_return_rate",
        "item_category": "item_category",
        "transaction_hour": "transaction_hour",
        "device_type": "device_type",
        "payment_method": "payment_method",
        "zip_delta_km": "zip_delta_km",
        "address_mismatch": "address_mismatch",
        "velocity_15min": "velocity_15min",
    }

    for output_name, input_name in field_mapping.items():

        if input_name in transaction:

            value = transaction[input_name]

            if value is not None:

                # Convert NumPy values into native Python values.
                if hasattr(value, "item"):
                    value = value.item()

                evidence[output_name] = value

    return evidence


# ============================================================
# CASE ASSESSMENT
# ============================================================

def assess_case(
    dispute_reason: str,
    transaction: Dict,
    retrieved_evidence: List[Dict],
) -> Dict:
    """
    Assess whether the available evidence supports
    a merchant defense.

    This is deterministic and transparent.
    """

    requirements = DISPUTE_REQUIREMENTS.get(
        dispute_reason,
        {}
    )

    policy_keywords = requirements.get(
        "policy_keywords",
        []
    )

    policy_text = " ".join(
        result.get("text", "")
        for result in retrieved_evidence
    ).upper()

    matched_policy_keywords = [
        keyword
        for keyword in policy_keywords
        if keyword.upper() in policy_text
    ]

    transaction_evidence = (
        extract_transaction_evidence(
            transaction
        )
    )

    evidence_count = len(
        transaction_evidence
    )

    policy_support = len(
        matched_policy_keywords
    )

    # Transparent evidence score.
    #
    # Policy support = up to 50 points
    # Transaction evidence = up to 50 points

    policy_score = min(
        policy_support / max(
            len(policy_keywords), 1
        ),
        1.0
    ) * 50

    transaction_score = min(
        evidence_count / 8,
        1.0
    ) * 50

    evidence_score = round(
        policy_score + transaction_score,
        2
    )

    if evidence_score >= 70:
        recommendation = "STRONG_DEFENSE"

    elif evidence_score >= 45:
        recommendation = "REVIEW"

    else:
        recommendation = "INSUFFICIENT_EVIDENCE"

    return {
        "dispute_reason": dispute_reason,
        "evidence_score": evidence_score,
        "recommendation": recommendation,
        "matched_policy_keywords": matched_policy_keywords,
        "transaction_evidence": transaction_evidence,
        "retrieved_policy_evidence": retrieved_evidence,
        "required_evidence": requirements.get(
            "required_evidence",
            []
        )
    }


# ============================================================
# HUMAN-READABLE SUMMARY
# ============================================================

def build_case_summary(
    case: Dict
) -> str:
    """
    Generate a deterministic summary suitable
    for display in the Streamlit dashboard.
    """

    recommendation = case[
        "recommendation"
    ]

    score = case[
        "evidence_score"
    ]

    reason = case[
        "dispute_reason"
    ]

    if recommendation == "STRONG_DEFENSE":

        conclusion = (
            "Available merchant policy and "
            "transaction evidence strongly support "
            "contesting this dispute."
        )

    elif recommendation == "REVIEW":

        conclusion = (
            "Some supporting evidence is available, "
            "but manual review is recommended before "
            "submitting the dispute response."
        )

    else:

        conclusion = (
            "Available evidence is insufficient to "
            "confidently support a merchant defense."
        )

    return (
        f"Dispute Reason: {reason}\n"
        f"Evidence Strength: {score}/100\n"
        f"Recommendation: {recommendation}\n\n"
        f"Assessment:\n{conclusion}"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("🛡️ HORAЕ EVIDENCE ENGINE TEST")
    print("=" * 70)

    # Mock transaction representing an
    # ITEM_NOT_RECEIVED dispute.

    test_transaction = {
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

    # Mock retrieved evidence.
    # Later this will come directly from rag_engine.py.

    retrieved_policy = [
        {
            "score": 0.629,
            "source": "refund_policy.txt",
            "section_index": 1,
            "text": (
                "2. ITEM NOT RECEIVED\n"
                "For an ITEM_NOT_RECEIVED claim, "
                "the merchant may provide shipment "
                "tracking information, carrier delivery "
                "confirmation, delivery timestamps, "
                "and proof of delivery.\n"
                "Where the carrier confirms successful "
                "delivery to the shipping address provided "
                "during checkout, the merchant may contest "
                "an unsupported non-delivery claim."
            )
        }
    ]

    case = assess_case(
        dispute_reason="ITEM_NOT_RECEIVED",
        transaction=test_transaction,
        retrieved_evidence=retrieved_policy,
    )

    print(
        "\n" + build_case_summary(case)
    )

    print("\n" + "-" * 70)
    print("📋 TRANSACTION EVIDENCE")
    print("-" * 70)

    for key, value in case[
        "transaction_evidence"
    ].items():

        print(
            f"{key}: {value}"
        )

    print("\n" + "-" * 70)
    print("📚 POLICY EVIDENCE")
    print("-" * 70)

    for result in case[
        "retrieved_policy_evidence"
    ]:

        print(
            f"\nSource: {result['source']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.3f}"
        )

        print(
            result["text"]
        )

    print("\n" + "=" * 70)
    print("✅ EVIDENCE ENGINE TEST COMPLETE")
    print("=" * 70)
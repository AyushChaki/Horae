"""
Horae — Chargeback Evidence Engine

Combines:
    1. Retrieved merchant policy evidence
    2. Synthetic operational transaction evidence
    3. Dispute-specific rules

to produce a transparent defense assessment.

The engine is deterministic and does NOT use an LLM.
"""

from __future__ import annotations

from typing import Dict, List, Any

from transaction_evidence import (
    generate_transaction_evidence,
)


# ============================================================
# DISPUTE EVIDENCE RULES
# ============================================================

DISPUTE_RULES = {

    "ITEM_NOT_RECEIVED": {
        "policy_weight": 40,

        "checks": [
            (
                "delivery_confirmation",
                lambda e: e.get("delivery_confirmation") is True,
                25,
                "Carrier delivery confirmation available",
            ),
            (
                "proof_of_delivery",
                lambda e: e.get("proof_of_delivery_available") is True,
                20,
                "Proof of delivery available",
            ),
            (
                "address_match",
                lambda e: e.get("address_match") is True,
                15,
                "Shipping address verified",
            ),
        ],
    },

    "PRODUCT_DEFECTIVE_OR_SWAPPED": {
        "policy_weight": 40,

        "checks": [
            (
                "return_received",
                lambda e: e.get("return_received") is True,
                20,
                "Returned product received",
            ),
            (
                "inspection_record",
                lambda e: e.get("inspection_record_available") is True,
                15,
                "Return inspection record available",
            ),
            (
                "fulfillment_record",
                lambda e: e.get("fulfillment_record_available") is True,
                15,
                "Original fulfillment record available",
            ),
            (
                "product_mismatch",
                lambda e: e.get("original_product_match") is False,
                10,
                "Returned item differs from original fulfillment",
            ),
        ],
    },

    "UNAUTHORIZED_TRANSACTION": {
        "policy_weight": 40,

        "checks": [
            (
                "authentication",
                lambda e: e.get("authentication_status") == "VERIFIED",
                25,
                "Transaction authentication verified",
            ),
            (
                "account_activity",
                lambda e: e.get("account_activity") == "NORMAL",
                15,
                "Account activity appears normal",
            ),
            (
                "device_information",
                lambda e: e.get("device_information_available") is True,
                10,
                "Device information available",
            ),
            (
                "device_consistency",
                lambda e: e.get("device_consistency") == "MATCHED",
                10,
                "Device is consistent with previous activity",
            ),
        ],
    },

    "SUBSCRIPTION_CANCELLED_REFUND": {
        "policy_weight": 40,

        "checks": [
            (
                "cancellation",
                lambda e: e.get("subscription_status") == "CANCELLED",
                25,
                "Subscription cancellation recorded",
            ),
            (
                "refund",
                lambda e: e.get("refund_status") == "PROCESSED",
                20,
                "Refund processing record available",
            ),
            (
                "refund_record",
                lambda e: e.get("refund_record_available") is True,
                15,
                "Refund record available",
            ),
        ],
    },

    "NOT_AS_DESCRIBED": {
        "policy_weight": 40,

        "checks": [
            (
                "product_information",
                lambda e: e.get("product_information_available") is True,
                20,
                "Product information available",
            ),
            (
                "fulfillment_record",
                lambda e: e.get("fulfillment_record_available") is True,
                20,
                "Fulfillment record available",
            ),
            (
                "description_match",
                lambda e: e.get("product_description_match") is True,
                20,
                "Product matches recorded description",
            ),
            (
                "customer_claim",
                lambda e: e.get("customer_claim_recorded") is True,
                10,
                "Customer claim recorded",
            ),
        ],
    },
}


# ============================================================
# POLICY SUPPORT
# ============================================================

def calculate_policy_support(
    dispute_reason: str,
    retrieved_evidence: List[Dict[str, Any]],
) -> tuple[float, List[str]]:
    """
    Determine how strongly the retrieved policy evidence
    supports the dispute type.

    Policy support contributes up to 40 points.
    """

    rules = DISPUTE_RULES.get(
        dispute_reason,
        {},
    )

    policy_weight = rules.get(
        "policy_weight",
        0,
    )

    if not retrieved_evidence:
        return 0.0, []

    # Combine retrieved policy text.
    policy_text = " ".join(
        result.get("text", "")
        for result in retrieved_evidence
    ).upper()

    # Dispute-specific terms.
    keywords = {
        "ITEM_NOT_RECEIVED": [
            "ITEM NOT RECEIVED",
            "TRACKING",
            "DELIVERY",
            "PROOF OF DELIVERY",
        ],

        "PRODUCT_DEFECTIVE_OR_SWAPPED": [
            "PRODUCT DEFECT",
            "PRODUCT SWAPPED",
            "INSPECTION",
            "FULFILLMENT",
        ],

        "UNAUTHORIZED_TRANSACTION": [
            "UNAUTHORIZED TRANSACTIONS",
            "AUTHENTICATION",
            "ACCOUNT",
            "DEVICE",
        ],

        "SUBSCRIPTION_CANCELLED_REFUND": [
            "REFUND",
            "CANCEL",
            "SUBSCRIPTION",
        ],

        "NOT_AS_DESCRIBED": [
            "PRODUCT",
            "FULFILLMENT",
            "DESCRIPTION",
            "RETURN",
        ],
    }

    relevant_keywords = keywords.get(
        dispute_reason,
        [],
    )

    matched = [
        keyword
        for keyword in relevant_keywords
        if keyword.upper() in policy_text
    ]

    if not relevant_keywords:
        return 0.0, matched

    coverage = (
        len(matched)
        / len(relevant_keywords)
    )

    score = round(
        coverage * policy_weight,
        2,
    )

    return score, matched


# ============================================================
# DISPUTE EVIDENCE SCORING
# ============================================================

def calculate_dispute_evidence_score(
    dispute_reason: str,
    dispute_evidence: Dict[str, Any],
) -> tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate evidence using only facts relevant to
    the specific dispute type.

    Returns:
        score
        evidence breakdown
    """

    rules = DISPUTE_RULES.get(
        dispute_reason
    )

    if not rules:
        return 0.0, []

    total_score = 0.0
    breakdown = []

    for (
        evidence_name,
        condition,
        weight,
        description,
    ) in rules["checks"]:

        passed = False

        try:
            passed = bool(
                condition(
                    dispute_evidence
                )
            )
        except Exception:
            passed = False

        awarded = weight if passed else 0

        total_score += awarded

        breakdown.append({
            "evidence": evidence_name,
            "description": description,
            "weight": weight,
            "passed": passed,
            "score": awarded,
        })

    return round(
        total_score,
        2,
    ), breakdown


# ============================================================
# COMPLETE CASE ASSESSMENT
# ============================================================

def assess_case(
    dispute_reason: str,
    transaction: Dict[str, Any],
    retrieved_evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produce a complete chargeback defense assessment.
    """

    # Generate synthetic operational evidence.
    transaction_evidence = (
        generate_transaction_evidence(
            transaction,
            dispute_reason,
        )
    )

    dispute_specific_evidence = (
        transaction_evidence[
            "dispute_evidence"
        ]
    )

    # Policy score: maximum 40.
    policy_score, matched_policy_keywords = (
        calculate_policy_support(
            dispute_reason,
            retrieved_evidence,
        )
    )

    # Transaction evidence score: maximum 60.
    transaction_score, evidence_breakdown = (
        calculate_dispute_evidence_score(
            dispute_reason,
            dispute_specific_evidence,
        )
    )

    final_score = round(
        policy_score + transaction_score,
        2,
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    if final_score >= 75:

        recommendation = (
            "STRONG_DEFENSE"
        )

    elif final_score >= 50:

        recommendation = (
            "REVIEW"
        )

    else:

        recommendation = (
            "INSUFFICIENT_EVIDENCE"
        )

    return {

        "transaction_id":
            transaction.get(
                "transaction_id"
            ),

        "dispute_reason":
            dispute_reason,

        "evidence_score":
            final_score,

        "policy_score":
            policy_score,

        "transaction_evidence_score":
            transaction_score,

        "recommendation":
            recommendation,

        "matched_policy_keywords":
            matched_policy_keywords,

        "evidence_breakdown":
            evidence_breakdown,

        "transaction_evidence":
            transaction_evidence,

        "retrieved_policy_evidence":
            retrieved_evidence,
    }


# ============================================================
# CASE SUMMARY
# ============================================================

def build_case_summary(
    case: Dict[str, Any]
) -> str:

    score = case[
        "evidence_score"
    ]

    recommendation = case[
        "recommendation"
    ]

    reason = case[
        "dispute_reason"
    ]

    if recommendation == "STRONG_DEFENSE":

        conclusion = (
            "Available policy and transaction "
            "evidence strongly support contesting "
            "this dispute."
        )

    elif recommendation == "REVIEW":

        conclusion = (
            "Supporting evidence exists, but "
            "manual review is recommended before "
            "submission."
        )

    else:

        conclusion = (
            "Available evidence is insufficient "
            "to confidently support a merchant defense."
        )

    return (
        f"Dispute Reason: {reason}\n"
        f"Evidence Strength: {score}/100\n"
        f"Policy Support: "
        f"{case['policy_score']}/40\n"
        f"Transaction Evidence: "
        f"{case['transaction_evidence_score']}/60\n"
        f"Recommendation: {recommendation}\n\n"
        f"Assessment:\n{conclusion}"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(
        "🛡️ HORAЕ EVIDENCE ENGINE TEST"
    )
    print("=" * 70)

    test_transaction = {

        "transaction_id":
            "TXN_200481",

        "user_id":
            "USR_10142",

        "order_amount_inr":
            18500,

        "account_age_days":
            240,

        "past_orders_count":
            18,

        "past_return_count":
            2,

        "past_return_rate":
            0.1111,

        "item_category":
            "Electronics",

        "transaction_hour":
            14,

        "device_type":
            "mobile_app",

        "payment_method":
            "UPI",

        "zip_delta_km":
            12.4,

        "address_mismatch":
            0,

        "velocity_15min":
            0,
    }

    retrieved_policy = [

        {
            "score": 0.629,

            "source":
                "refund_policy.txt",

            "section_index":
                1,

            "text": (
                "2. ITEM NOT RECEIVED\n"
                "For an ITEM_NOT_RECEIVED claim, "
                "the merchant may provide shipment "
                "tracking information, carrier delivery "
                "confirmation, delivery timestamps, "
                "and proof of delivery.\n"
                "Where the carrier confirms successful "
                "delivery to the shipping address provided "
                "during checkout, the merchant may "
                "contest an unsupported non-delivery claim."
            ),
        }
    ]

    case = assess_case(
        dispute_reason=
            "ITEM_NOT_RECEIVED",

        transaction=
            test_transaction,

        retrieved_evidence=
            retrieved_policy,
    )

    print(
        "\n"
        + build_case_summary(case)
    )

    print("\n" + "-" * 70)
    print("📊 EVIDENCE BREAKDOWN")
    print("-" * 70)

    for item in case[
        "evidence_breakdown"
    ]:

        status = (
            "✅"
            if item["passed"]
            else "❌"
        )

        print(
            f"{status} "
            f"{item['description']} "
            f"({item['score']}/{item['weight']})"
        )

    print("\n" + "-" * 70)
    print("📦 OPERATIONAL EVIDENCE")
    print("-" * 70)

    dispute_evidence = (
        case[
            "transaction_evidence"
        ][
            "dispute_evidence"
        ]
    )

    for key, value in dispute_evidence.items():

        label = key.replace(
            "_",
            " "
        ).title()

        print(
            f"{label}: {value}"
        )

    print("\n" + "=" * 70)
    print(
        "✅ EVIDENCE ENGINE TEST COMPLETE"
    )
    print("=" * 70)
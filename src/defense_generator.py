"""
Horae — AI Chargeback Defense Generator

Converts verified policy + transaction evidence into a
structured merchant defense response.

IMPORTANT:
The LLM is only a language-generation layer.
It must NOT invent evidence, timestamps, tracking numbers,
delivery confirmations, or policy claims.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from unittest import case, result

load_dotenv()


# ============================================================
# DEFENSE PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Horae, an AI chargeback defense assistant.

Your job is to draft a professional merchant response using
ONLY the evidence provided in the case packet.

STRICT RULES:

1. Never invent evidence.
2. Never invent tracking numbers.
3. Never invent delivery confirmation.
4. Never invent timestamps.
5. Never invent authentication records.
6. Never claim that a document exists unless it appears in
   the supplied evidence.
7. Never change the meaning of the evidence.
8. If evidence is missing, explicitly state that it is missing.
9. Do not exaggerate the strength of the merchant's case.
10. The final recommendation must reflect the supplied
    evidence assessment.
11. Do not make legal conclusions.
12. Do not fabricate policy clauses.

Write in a concise, professional tone suitable for a merchant
reviewing a chargeback case.

Return valid JSON with exactly these fields:

{
  "case_summary": "...",
  "merchant_position": "...",
  "supporting_evidence": [],
  "missing_evidence": [],
  "recommended_action": "...",
  "draft_defense": "..."
}
"""


# ============================================================
# CASE PACKET
# ============================================================

def build_case_packet(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the minimal verified evidence packet that can be
    passed to the language model.
    """

    transaction = case.get(
        "transaction_evidence",
        {},
    )

    dispute_evidence = transaction.get(
        "dispute_evidence",
        {},
    )

    policy_evidence = case.get(
        "retrieved_policy_evidence",
        [],
    )

    # Only expose the relevant policy fields.
    policies: List[Dict[str, Any]] = []

    for item in policy_evidence:

        policies.append({
            "source": item.get(
                "source"
            ),

            "similarity": item.get(
                "score"
            ),

            "text": item.get(
                "text"
            ),
        })

    return {

        "transaction_id":
            case.get(
                "transaction_id"
            ),

        "dispute_reason":
            case.get(
                "dispute_reason"
            ),

        "evidence_score":
            case.get(
                "evidence_score"
            ),

        "policy_score":
            case.get(
                "policy_score"
            ),

        "transaction_evidence_score":
            case.get(
                "transaction_evidence_score"
            ),

        "recommendation":
            case.get(
                "recommendation"
            ),

        "dispute_evidence":
            dispute_evidence,

        "policy_evidence":
            policies,

        "evidence_breakdown":
            case.get(
                "evidence_breakdown",
                [],
            ),
    }


# ============================================================
# FALLBACK GENERATOR
# ============================================================

def generate_fallback_defense(
    case: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deterministic fallback used when no LLM API key is
    available.

    This makes the system fully demoable without an external
    API dependency.
    """

    reason = case.get(
        "dispute_reason",
        "UNKNOWN",
    )

    score = case.get(
        "evidence_score",
        0,
    )

    recommendation = case.get(
        "recommendation",
        "REVIEW",
    )

    transaction_id = case.get(
        "transaction_id",
        "UNKNOWN",
    )

    breakdown = case.get(
        "evidence_breakdown",
        [],
    )

    supporting = [
        item["description"]
        for item in breakdown
        if item.get("passed") is True
    ]

    missing = [
        item["description"]
        for item in breakdown
        if item.get("passed") is False
    ]

    # --------------------------------------------------------
    # Case summary
    # --------------------------------------------------------

    summary = (
        f"Chargeback case {transaction_id} "
        f"was assessed for dispute reason "
        f"{reason}. Horae assigned an evidence "
        f"strength score of {score}/100."
    )

    # --------------------------------------------------------
    # Merchant position
    # --------------------------------------------------------

    if recommendation == "STRONG_DEFENSE":

        position = (
            "The available evidence provides strong "
            "support for contesting the dispute."
        )

        action = (
            "Proceed with merchant defense submission "
            "using the verified supporting evidence."
        )

    elif recommendation == "REVIEW":

        position = (
            "The merchant has supporting evidence, "
            "but material gaps remain."
        )

        action = (
            "Route the case for manual review and "
            "obtain the missing evidence before submission."
        )

    else:

        position = (
            "The currently available evidence does not "
            "provide sufficient support for a strong defense."
        )

        action = (
            "Do not submit an unsupported defense. "
            "Seek additional transaction evidence."
        )

    # --------------------------------------------------------
    # Evidence paragraphs
    # --------------------------------------------------------

    evidence_text = ""

    if supporting:

        evidence_text += (
            "Verified supporting evidence includes: "
            + "; ".join(supporting)
            + ". "
        )

    if missing:

        evidence_text += (
            "The following evidence is currently "
            "unavailable: "
            + "; ".join(missing)
            + "."
        )

    draft = (
        f"Re: Chargeback dispute {transaction_id}\n\n"
        f"Dispute reason: {reason}.\n\n"
        f"{position} "
        f"{evidence_text}\n\n"
        f"Based on the currently available records, "
        f"Horae recommends: {action}"
    )

    return {

        "case_summary": summary,

        "merchant_position": position,

        "supporting_evidence": supporting,

        "missing_evidence": missing,

        "recommended_action": action,

        "draft_defense": draft,
    }


# ============================================================
# LLM GENERATOR
# ============================================================
# ============================================================
# GROUNDING GUARD
# ============================================================

def validate_defense_grounding(
    result: Dict[str, Any],
    case: Dict[str, Any],
) -> tuple[bool, List[str]]:
    """
    Validate that the generated defense stays grounded in the
    verified case packet.

    This is intentionally conservative:
    if the LLM introduces a material claim that is not supported,
    the response is rejected and the deterministic fallback is used.
    """

    violations: List[str] = []

    draft = str(
        result.get("draft_defense", "")
    ).lower()

    transaction_evidence = case.get(
        "transaction_evidence",
        {},
    )

    dispute_evidence = transaction_evidence.get(
        "dispute_evidence",
        {},
    )

    reason = case.get(
        "dispute_reason",
        "",
    )

    # --------------------------------------------------------
    # ITEM NOT RECEIVED
    # --------------------------------------------------------

    if reason == "ITEM_NOT_RECEIVED":

        delivery_confirmation = (
            dispute_evidence.get(
                "delivery_confirmation",
                False,
            )
        )

        proof_of_delivery = (
            dispute_evidence.get(
                "proof_of_delivery_available",
                False,
            )
        )

        # If delivery was NOT confirmed, the LLM must not
        # claim that successful delivery occurred.
        if not delivery_confirmation:

            unsupported_delivery_terms = [
                "successfully delivered",
                "successfully received",
                "delivery was confirmed",
                "proof of delivery confirms",
                "customer received the item",
            ]

            for term in unsupported_delivery_terms:

                if term in draft:

                    violations.append(
                        f"Unsupported delivery claim: '{term}'"
                    )

        # If proof of delivery is unavailable, the LLM
        # must not claim that POD exists.
        if not proof_of_delivery:

            unsupported_pod_terms = [
                "proof of delivery confirms",
                "proof of delivery shows",
                "pod confirms",
                "delivery receipt confirms",
            ]

            for term in unsupported_pod_terms:

                if term in draft:

                    violations.append(
                        f"Unsupported proof-of-delivery claim: '{term}'"
                    )

        # We do NOT have return evidence for this dispute.
        # Therefore the model should not invent a returned-item
        # outcome.
        unsupported_return_terms = [
            "item was returned",
            "item was received back",
            "returned item was inspected",
            "merchant received the item back",
            "item was not returned",
            "merchant did not receive the item back",
        ]

        for term in unsupported_return_terms:

            if term in draft:

                violations.append(
                    f"Unsupported return claim: '{term}'"
                )

    # --------------------------------------------------------
    # TRACKING NUMBER VALIDATION
    # --------------------------------------------------------

    tracking_number = dispute_evidence.get(
        "tracking_number"
    )

    if tracking_number:

        # If the model mentions a tracking number,
        # it must be the verified one.
        import re

        tracking_numbers = re.findall(
            r"\bTRK_\d+\b",
            draft.upper(),
        )

        for generated_tracking in tracking_numbers:

            if generated_tracking != str(
                tracking_number
            ).upper():

                violations.append(
                    "Generated tracking number does not "
                    "match verified evidence."
                )

    # --------------------------------------------------------
    # TRANSACTION ID VALIDATION
    # --------------------------------------------------------

    transaction_id = case.get(
        "transaction_id"
    )

    if transaction_id:

        import re

        transaction_ids = re.findall(
            r"\bTXN_\d+\b",
            draft.upper(),
        )

        for generated_id in transaction_ids:

            if generated_id != str(
                transaction_id
            ).upper():

                violations.append(
                    "Generated transaction ID does not "
                    "match verified evidence."
                )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    return (
        len(violations) == 0,
        violations,
    )

# ============================================================
# LLM GENERATOR
# ============================================================

def generate_defense(
    case: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate a defense response.

    Uses an LLM when configured.
    Otherwise falls back to deterministic generation.

    The LLM output is validated before being returned.
    """

    api_key = os.getenv("MISTRAL_API_KEY")


    # --------------------------------------------------------
    # No API key → deterministic fallback
    # --------------------------------------------------------

    if not api_key:
        print(
            "⚠️ MISTRAL_API_KEY not configured."
        )
        print(
            "↪ Using deterministic fallback."
        )

        return generate_fallback_defense(case)
    print("🔑 Mistral API key detected.")

    # --------------------------------------------------------
    # LLM GENERATION
    # --------------------------------------------------------

    try:

        from mistralai.client import Mistral

        client = Mistral(
            api_key=api_key
        )

        packet = build_case_packet(
            case
        )

        response = client.chat.complete(

            model="mistral-small-latest",

            messages=[

                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },

                {
                    "role": "user",
                    "content": (
                        "Generate the defense response "
                        "from this verified case packet:\n\n"
                        + json.dumps(
                            packet,
                            indent=2,
                            default=str,
                        )
                    ),
                },
            ],

            temperature=0.1,
        )

        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:
            print(
                "⚠️ Empty LLM response."
            )

            print(
                "↪ Using deterministic fallback."
            )

            return generate_fallback_defense(
                case
            )

        content = content.strip()

        # ----------------------------------------------------
        # Remove accidental markdown fences
        # ----------------------------------------------------

        if content.startswith("```"):

            content = (
                content
                .replace(
                    "```json",
                    "",
                )
                .replace(
                    "```",
                    "",
                )
                .strip()
            )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        result = json.loads(
            content
        )

        # ----------------------------------------------------
        # Validate required fields
        # ----------------------------------------------------

        required_fields = [
            "case_summary",
            "merchant_position",
            "supporting_evidence",
            "missing_evidence",
            "recommended_action",
            "draft_defense",
        ]

        valid_result = (
            isinstance(result, dict)
            and all(
                field in result
                for field in required_fields
            )
        )

        if not valid_result:

            print(
                "⚠️ LLM response rejected."
            )

            print(
                "↪ Using deterministic fallback."
            )

            return generate_fallback_defense(
                case
            )

        # ----------------------------------------------------
        # Additional safety validation
        # ----------------------------------------------------

        if (
            not isinstance(
                result["supporting_evidence"],
                list,
            )
            or not isinstance(
                result["missing_evidence"],
                list,
            )
        ):

            print(
                "⚠️ Invalid evidence format "
                "returned by LLM."
            )

            print(
                "↪ Using deterministic fallback."
            )

            return generate_fallback_defense(
                case
            )

        # ----------------------------------------------------
        # Valid LLM response
        # ----------------------------------------------------

        return result

    # --------------------------------------------------------
    # ANY LLM/API/PARSING FAILURE
    # --------------------------------------------------------

    except Exception as exc:

        print(
            f"⚠️ LLM generation failed: {exc}"
        )

        print(
            "↪ Using deterministic fallback."
        )

        return generate_fallback_defense(
            case
        )

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(
        "🤖 HORAЕ DEFENSE GENERATOR TEST"
    )
    print("=" * 70)

    test_case = {

        "transaction_id":
            "TXN_200481",

        "dispute_reason":
            "ITEM_NOT_RECEIVED",

        "evidence_score":
            55.0,

        "policy_score":
            40.0,

        "transaction_evidence_score":
            15.0,

        "recommendation":
            "REVIEW",

        "evidence_breakdown": [

            {
                "description":
                    "Carrier delivery confirmation available",

                "passed":
                    False,

                "score":
                    0,

                "weight":
                    25,
            },

            {
                "description":
                    "Proof of delivery available",

                "passed":
                    False,

                "score":
                    0,

                "weight":
                    20,
            },

            {
                "description":
                    "Shipping address verified",

                "passed":
                    True,

                "score":
                    15,

                "weight":
                    15,
            },
        ],

        "transaction_evidence": {

            "dispute_evidence": {

                "tracking_number":
                    "TRK_113121",

                "delivery_status":
                    "DELIVERY_EXCEPTION",

                "delivery_confirmation":
                    False,

                "proof_of_delivery_available":
                    False,

                "address_match":
                    True,
            }
        },

        "retrieved_policy_evidence": [

            {
                "source":
                    "refund_policy.txt",

                "score":
                    0.629,

                "text":
                    (
                        "For an ITEM_NOT_RECEIVED claim, "
                        "the merchant may provide shipment "
                        "tracking information, carrier "
                        "delivery confirmation, delivery "
                        "timestamps, and proof of delivery."
                    ),
            }
        ],
    }

    result = generate_defense(
        test_case
    )

    print(
        "\n📄 CASE SUMMARY"
    )
    print("-" * 70)
    print(
        result[
            "case_summary"
        ]
    )

    print(
        "\n🛡️ MERCHANT POSITION"
    )
    print("-" * 70)
    print(
        result[
            "merchant_position"
        ]
    )

    print(
        "\n✅ SUPPORTING EVIDENCE"
    )
    print("-" * 70)

    for item in result[
        "supporting_evidence"
    ]:

        print(
            f"• {item}"
        )

    print(
        "\n⚠️ MISSING EVIDENCE"
    )
    print("-" * 70)

    for item in result[
        "missing_evidence"
    ]:

        print(
            f"• {item}"
        )

    print(
        "\n📌 RECOMMENDED ACTION"
    )
    print("-" * 70)
    print(
        result[
            "recommended_action"
        ]
    )

    print(
        "\n📝 DRAFT DEFENSE"
    )
    print("-" * 70)
    print(
        result[
            "draft_defense"
        ]
    )

    print("\n" + "=" * 70)
    print(
        "✅ DEFENSE GENERATOR TEST COMPLETE"
    )
    print("=" * 70)
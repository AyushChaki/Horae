"""
Horae — Transaction Evidence Simulator

Generates deterministic operational evidence for a transaction.

Important:
This is a SYNTHETIC evidence layer for the hackathon dataset.
It does not claim to represent real payment/shipping records.

The generated evidence is deterministic for a transaction,
which means the same transaction will produce the same
evidence every time.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any


# ============================================================
# HELPERS
# ============================================================

def _stable_number(
    transaction_id: str,
    salt: str,
    modulo: int,
) -> int:
    """
    Generate a deterministic pseudo-random integer.

    Using a hash instead of random.seed() means the output
    remains stable across different executions.
    """

    value = (
        f"{transaction_id}:{salt}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        value
    ).hexdigest()

    return int(digest[:12], 16) % modulo


def _stable_bool(
    transaction_id: str,
    salt: str,
    probability_percent: int,
) -> bool:
    """
    Generate a deterministic boolean with an approximate
    probability.
    """

    return (
        _stable_number(
            transaction_id,
            salt,
            100,
        )
        < probability_percent
    )


# ============================================================
# BASE TRANSACTION EVIDENCE
# ============================================================

def generate_base_evidence(
    transaction: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate operational records associated with a transaction.

    These are synthetic records designed for demonstration.
    """

    transaction_id = str(
        transaction["transaction_id"]
    )

    # Stable synthetic tracking ID.
    tracking_number = (
        f"TRK_"
        f"{_stable_number(transaction_id, 'tracking', 900000) + 100000}"
    )

    # Synthetic fulfillment date.
    days_ago = _stable_number(
        transaction_id,
        "fulfillment_days",
        15,
    ) + 1

    fulfillment_date = (
        datetime.now()
        - timedelta(days=days_ago)
    )

    # Delivery occurs 1–5 days after fulfillment.
    delivery_delay = (
        _stable_number(
            transaction_id,
            "delivery_delay",
            5,
        )
        + 1
    )

    delivery_date = (
        fulfillment_date
        + timedelta(days=delivery_delay)
    )

    # Whether the shipment was successfully delivered.
    delivered = _stable_bool(
        transaction_id,
        "delivered",
        92,
    )

    # Address verification based partly on the
    # existing transaction address mismatch feature.
    address_match = (
        int(
            transaction.get(
                "address_mismatch",
                0,
            )
        )
        == 0
    )

    return {
        "tracking_number": tracking_number,

        "fulfillment_timestamp":
            fulfillment_date.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "delivery_timestamp":
            (
                delivery_date.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if delivered
                else None
            ),

        "delivery_status":
            "DELIVERED"
            if delivered
            else "DELIVERY_EXCEPTION",

        "shipping_address_verified":
            address_match,

        "carrier_event":
            (
                "Successful delivery confirmed"
                if delivered
                else "Delivery exception recorded"
            ),
    }


# ============================================================
# DISPUTE-SPECIFIC EVIDENCE
# ============================================================

def generate_dispute_evidence(
    transaction: Dict[str, Any],
    dispute_reason: str,
) -> Dict[str, Any]:
    """
    Generate evidence specifically relevant to a dispute type.
    """

    transaction_id = str(
        transaction["transaction_id"]
    )

    # --------------------------------------------------------
    # ITEM NOT RECEIVED
    # --------------------------------------------------------

    if dispute_reason == "ITEM_NOT_RECEIVED":

        base = generate_base_evidence(
            transaction
        )

        return {
            **base,

            "evidence_type":
                "DELIVERY_EVIDENCE",

            "delivery_confirmation":
                base["delivery_status"]
                == "DELIVERED",

            "address_match":
                base["shipping_address_verified"],

            "proof_of_delivery_available":
                base["delivery_status"]
                == "DELIVERED",
        }

    # --------------------------------------------------------
    # PRODUCT DEFECTIVE / SWAPPED
    # --------------------------------------------------------

    if dispute_reason == "PRODUCT_DEFECTIVE_OR_SWAPPED":

        return_received = _stable_bool(
            transaction_id,
            "return_received",
            85,
        )

        # Synthetic inspection outcome.
        swapped_item = _stable_bool(
            transaction_id,
            "swapped_item",
            15,
        )

        if swapped_item:

            inspection_status = (
                "SWAPPED_ITEM"
            )

            product_match = False

        else:

            inspection_status = (
                "PRODUCT_MATCH_CONFIRMED"
            )

            product_match = True

        return {
            "evidence_type":
                "RETURN_INSPECTION_EVIDENCE",

            "return_requested": True,

            "return_received":
                return_received,

            "inspection_status":
                (
                    inspection_status
                    if return_received
                    else "AWAITING_RETURN"
                ),

            "original_product_match":
                (
                    product_match
                    if return_received
                    else None
                ),

            "fulfillment_record_available":
                True,

            "inspection_record_available":
                return_received,
        }

    # --------------------------------------------------------
    # UNAUTHORIZED TRANSACTION
    # --------------------------------------------------------

    if dispute_reason == "UNAUTHORIZED_TRANSACTION":

        authentication_verified = _stable_bool(
            transaction_id,
            "authentication",
            94,
        )

        device_match = _stable_bool(
            transaction_id,
            "device_match",
            88,
        )

        account_activity_normal = _stable_bool(
            transaction_id,
            "account_activity",
            90,
        )

        return {
            "evidence_type":
                "TRANSACTION_AUTHENTICATION_EVIDENCE",

            "transaction_timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "authentication_status":
                (
                    "VERIFIED"
                    if authentication_verified
                    else "NOT_VERIFIED"
                ),

            "device_consistency":
                (
                    "MATCHED"
                    if device_match
                    else "NEW_DEVICE"
                ),

            "account_activity":
                (
                    "NORMAL"
                    if account_activity_normal
                    else "ANOMALOUS"
                ),

            "payment_method":
                transaction.get(
                    "payment_method"
                ),

            "device_information_available":
                True,
        }

    # --------------------------------------------------------
    # SUBSCRIPTION CANCELLED / REFUND
    # --------------------------------------------------------

    if dispute_reason == "SUBSCRIPTION_CANCELLED_REFUND":

        cancellation_exists = _stable_bool(
            transaction_id,
            "cancellation",
            70,
        )

        refund_processed = _stable_bool(
            transaction_id,
            "refund",
            65,
        )

        return {
            "evidence_type":
                "SUBSCRIPTION_REFUND_EVIDENCE",

            "subscription_status":
                "ACTIVE"
                if not cancellation_exists
                else "CANCELLED",

            "cancellation_timestamp":
                (
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if cancellation_exists
                    else None
                ),

            "refund_status":
                (
                    "PROCESSED"
                    if refund_processed
                    else "NOT_PROCESSED"
                ),

            "refund_record_available":
                refund_processed,
        }

    # --------------------------------------------------------
    # NOT AS DESCRIBED
    # --------------------------------------------------------

    if dispute_reason == "NOT_AS_DESCRIBED":

        fulfillment_available = True

        product_information_available = True

        return_match = _stable_bool(
            transaction_id,
            "description_match",
            90,
        )

        return {
            "evidence_type":
                "PRODUCT_FULFILLMENT_EVIDENCE",

            "product_information_available":
                product_information_available,

            "fulfillment_record_available":
                fulfillment_available,

            "product_description_match":
                return_match,

            "customer_claim_recorded":
                True,
        }

    # --------------------------------------------------------
    # UNKNOWN DISPUTE
    # --------------------------------------------------------

    return {
        "evidence_type":
            "GENERAL_TRANSACTION_EVIDENCE",

        "transaction_record_available":
            True,
    }


# ============================================================
# COMPLETE EVIDENCE PACKAGE
# ============================================================

def generate_transaction_evidence(
    transaction: Dict[str, Any],
    dispute_reason: str,
) -> Dict[str, Any]:
    """
    Generate a complete operational evidence package.
    """

    base_evidence = generate_base_evidence(
        transaction
    )

    dispute_evidence = generate_dispute_evidence(
        transaction,
        dispute_reason,
    )

    return {
        "transaction_id":
            transaction.get(
                "transaction_id"
            ),

        "dispute_reason":
            dispute_reason,

        "synthetic_evidence":
            True,

        "base_evidence":
            base_evidence,

        "dispute_evidence":
            dispute_evidence,
    }


# ============================================================
# HUMAN-READABLE FORMATTER
# ============================================================

def format_evidence(
    evidence: Dict[str, Any]
) -> str:
    """
    Convert evidence into a readable format for the UI
    and future LLM prompt.
    """

    lines = []

    lines.append(
        f"Transaction: "
        f"{evidence['transaction_id']}"
    )

    lines.append(
        f"Dispute Reason: "
        f"{evidence['dispute_reason']}"
    )

    lines.append("")

    lines.append(
        "TRANSACTION EVIDENCE"
    )

    lines.append("-" * 40)

    for key, value in evidence[
        "base_evidence"
    ].items():

        label = key.replace(
            "_",
            " "
        ).title()

        lines.append(
            f"{label}: {value}"
        )

    lines.append("")

    lines.append(
        "DISPUTE-SPECIFIC EVIDENCE"
    )

    lines.append("-" * 40)

    for key, value in evidence[
        "dispute_evidence"
    ].items():

        label = key.replace(
            "_",
            " "
        ).title()

        lines.append(
            f"{label}: {value}"
        )

    return "\n".join(lines)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(
        "📦 HORAЕ TRANSACTION EVIDENCE TEST"
    )
    print("=" * 70)

    test_transaction = {
        "transaction_id":
            "TXN_200481",

        "user_id":
            "USR_10142",

        "order_amount_inr":
            18500,

        "payment_method":
            "UPI",

        "address_mismatch":
            0,
    }

    dispute_types = [
        "ITEM_NOT_RECEIVED",
        "PRODUCT_DEFECTIVE_OR_SWAPPED",
        "UNAUTHORIZED_TRANSACTION",
        "SUBSCRIPTION_CANCELLED_REFUND",
        "NOT_AS_DESCRIBED",
    ]

    for dispute_reason in dispute_types:

        print("\n" + "=" * 70)

        print(
            f"DISPUTE: {dispute_reason}"
        )

        print("=" * 70)

        evidence = (
            generate_transaction_evidence(
                test_transaction,
                dispute_reason,
            )
        )

        print(
            format_evidence(evidence)
        )

    print("\n" + "=" * 70)
    print(
        "✅ TRANSACTION EVIDENCE TEST COMPLETE"
    )
    print("=" * 70)
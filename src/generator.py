import numpy as np
import pandas as pd
import random
import os

def generate_synthetic_transactions(num_records=60000, seed=42):
    """
    Generates a high-volume (60k) synthetic e-commerce transaction dataset
    tailored for Return Fraud & Chargeback Risk Scoring.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"🚀 Generating {num_records:,} synthetic transaction records...")
    
    # 1. Base Customer Metadata (scaling user pool to ~20,000 unique buyers)
    num_users = num_records // 3
    user_ids = [f"USR_{10000 + i}" for i in range(num_users)]
    assigned_users = np.random.choice(user_ids, size=num_records)
    
    account_age_days = np.random.exponential(scale=200, size=num_records).astype(int) + 1
    past_orders_count = np.clip((account_age_days / 12) + np.random.normal(0, 4, num_records), 0, 150).astype(int)
    past_return_count = np.clip(past_orders_count * np.random.uniform(0.02, 0.45, num_records), 0, past_orders_count).astype(int)
    
    # Calculate historical return rate safely
    past_return_rate = np.where(past_orders_count > 0, past_return_count / past_orders_count, 0.0)

    # 2. Transaction Categories & Pricing Dynamics
    categories = ['Electronics', 'Apparel', 'Digital Goods', 'Home & Kitchen', 'Footwear']
    category_weights = [0.25, 0.35, 0.15, 0.15, 0.10]
    item_category = np.random.choice(categories, size=num_records, p=category_weights)
    
    # Vectorized category price assignment
    category_price_mean = {'Electronics': 15000, 'Apparel': 2800, 'Digital Goods': 1200, 'Home & Kitchen': 4500, 'Footwear': 3800}
    category_price_std = {'Electronics': 6000, 'Apparel': 1200, 'Digital Goods': 500, 'Home & Kitchen': 1800, 'Footwear': 1500}
    
    means = np.array([category_price_mean[cat] for cat in item_category])
    stds = np.array([category_price_std[cat] for cat in item_category])
    order_amount = np.round(np.clip(np.random.normal(means, stds), 300, 80000), 2)

    # 3. Behavioral Features & Signals
    transaction_hour = np.random.randint(0, 24, size=num_records)
    device_type = np.random.choice(['mobile_app', 'mobile_web', 'desktop_web'], size=num_records, p=[0.60, 0.28, 0.12])
    payment_method = np.random.choice(['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'COD'], size=num_records, p=[0.52, 0.23, 0.15, 0.05, 0.05])
    
    # Shipping vs Billing distance delta
    shipping_billing_zip_delta_km = np.round(np.random.exponential(scale=20, size=num_records), 2)
    address_mismatch = (shipping_billing_zip_delta_km > 80).astype(int)
    
    # Order velocity in past 15 minutes
    velocity_15min = np.random.poisson(lam=0.35, size=num_records)
    
    # 4. Latent Risk Model Synthesis
    risk_score_raw = (
        (account_age_days < 5) * 2.8 +                     # Brand new account
        (past_return_rate > 0.35) * 3.2 +                 # Chronic returner
        (velocity_15min >= 2) * 3.8 +                      # Velocity spike (bot/card testing)
        (address_mismatch == 1) * 2.2 +                   # Address mismatch
        ((transaction_hour >= 1) & (transaction_hour <= 4)) * 1.8 + # Off-peak hour activity
        np.isin(item_category, ['Electronics', 'Digital Goods']) * 1.4 +
        (order_amount > 20000) * 2.1 +                    # High cart value
        np.random.normal(0, 1.1, num_records)             # Unobserved factors
    )
    
    # Convert to probability distribution via Sigmoid transform
    prob_risk = 1 / (1 + np.exp(-(risk_score_raw - 4.8)))
    is_risk = (prob_risk > 0.68).astype(int)

    # 5. Assign Chargeback / Return Dispute Reason Codes (for RAG module integration)
    reason_codes = [
        "ITEM_NOT_RECEIVED",
        "PRODUCT_DEFECTIVE_OR_SWAPPED",
        "UNAUTHORIZED_TRANSACTION",
        "SUBSCRIPTION_CANCELLED_REFUND",
        "NOT_AS_DESCRIBED"
    ]
    # Assign reasons to flagged cases, leave clean transactions as "NONE"
    dispute_reasons = np.where(
        is_risk == 1,
        np.random.choice(reason_codes, size=num_records, p=[0.30, 0.25, 0.25, 0.10, 0.10]),
        "NONE"
    )

    # 6. Financial Economics Metrics
    profit_margin_amount = np.round(order_amount * 0.18, 2)  # ~18% average margin
    chargeback_fee = 500.0  # Fixed gateway penalty fee per disputed transaction

    # Build DataFrame
    df = pd.DataFrame({
        'transaction_id': [f"TXN_{200000 + i}" for i in range(num_records)],
        'user_id': assigned_users,
        'account_age_days': account_age_days,
        'past_orders_count': past_orders_count,
        'past_return_count': past_return_count,
        'past_return_rate': np.round(past_return_rate, 4),
        'item_category': item_category,
        'order_amount_inr': order_amount,
        'profit_margin_inr': profit_margin_amount,
        'chargeback_fee_inr': chargeback_fee,
        'transaction_hour': transaction_hour,
        'device_type': device_type,
        'payment_method': payment_method,
        'zip_delta_km': shipping_billing_zip_delta_km,
        'address_mismatch': address_mismatch,
        'velocity_15min': velocity_15min,
        'dispute_reason': dispute_reasons,
        'is_risk': is_risk  # Ground Truth
    })

    return df

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    df_60k = generate_synthetic_transactions(num_records=60000, seed=42)
    output_path = "data/synthetic_transactions.csv"
    
    df_60k.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset successfully generated and saved to '{output_path}'")
    print(f"📊 Dataset Size: {df_60k.shape[0]:,} rows × {df_60k.shape[1]} columns")
    print(f"🎯 Positive Fraud/Risk Rate: {df_60k['is_risk'].mean() * 100:.2f}% ({df_60k['is_risk'].sum():,} flagged cases)")
"""
Horae - Risk Scoring & Financial Cost-Boundary Optimization

This module:
1. Loads the synthetic transaction dataset.
2. Preprocesses numerical and categorical features.
3. Trains an XGBoost risk classifier.
4. Evaluates statistical performance.
5. Optimizes the decision threshold using financial value.
6. Saves the trained model and optimization metadata.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_transactions.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "risk_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 3. FEATURE DEFINITIONS
# ============================================================

NUMERICAL_FEATURES = [
    "account_age_days",
    "past_orders_count",
    "past_return_count",
    "past_return_rate",
    "order_amount_inr",
    "profit_margin_inr",
    "chargeback_fee_inr",
    "transaction_hour",
    "zip_delta_km",
    "address_mismatch",
    "velocity_15min",
]

CATEGORICAL_FEATURES = [
    "item_category",
    "device_type",
    "payment_method",
]


TARGET = "is_risk"


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data():
    """Load the generated transaction dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found at:\n{DATA_PATH}\n\n"
            "Run the following command first:\n"
            "python src/generator.py"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 70)
    print("📂 DATASET LOADED")
    print("=" * 70)

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Risk cases : {df[TARGET].sum():,}")
    print(
        f"Risk rate  : {df[TARGET].mean() * 100:.2f}%"
    )

    return df


# ============================================================
# 5. PREPROCESSING
# ============================================================

def build_preprocessor():
    """Create preprocessing pipelines for numerical/categorical data."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


# ============================================================
# 6. TRAIN MODEL
# ============================================================

def train_model(X_train, y_train, preprocessor):
    """Train the XGBoost risk classifier."""

    print("\n" + "=" * 70)
    print("🤖 TRAINING XGBOOST RISK MODEL")
    print("=" * 70)

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    print("✅ Model training completed.")

    return pipeline


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model using standard classification metrics."""

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.50).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print("\n" + "=" * 70)
    print("📊 MODEL PERFORMANCE @ 0.50 THRESHOLD")
    print("=" * 70)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Risk"],
            zero_division=0,
        )
    )

    return probabilities


# ============================================================
# 8. FINANCIAL COST-BOUNDARY OPTIMIZATION
# ============================================================

def calculate_financial_value(
    y_true,
    probabilities,
    financial_df,
    threshold,
):
    """
    Calculate the financial value of a classification threshold.

    Decision:
        probability >= threshold -> BLOCK / REVIEW AS RISK
        probability < threshold  -> ALLOW

    For legitimate transactions:
        Correctly allowing the transaction preserves profit margin.

    For risky transactions:
        Correctly identifying risk avoids:
            profit margin + chargeback fee

    For incorrectly blocked legitimate transactions:
        We lose the legitimate profit margin.

    For incorrectly allowed risky transactions:
        We lose profit margin + chargeback fee.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    profit_margin = financial_df[
        "profit_margin_inr"
    ].to_numpy()

    chargeback_fee = financial_df[
        "chargeback_fee_inr"
    ].to_numpy()

    # Legitimate transaction
    legitimate = y_true == 0

    # Risky transaction
    risky = y_true == 1

    # Correctly allowed legitimate transaction
    true_negative = legitimate & (predictions == 0)

    # Incorrectly blocked legitimate transaction
    false_positive = legitimate & (predictions == 1)

    # Correctly blocked risky transaction
    true_positive = risky & (predictions == 1)

    # Incorrectly allowed risky transaction
    false_negative = risky & (predictions == 0)

    # Revenue/profit preserved
    value_from_legitimate = profit_margin[true_negative].sum()

    # Cost of blocking legitimate customer
    cost_false_positive = profit_margin[
        false_positive
    ].sum()

    # Cost avoided by catching risky transactions
    value_from_true_positive = (
        profit_margin[true_positive]
        + chargeback_fee[true_positive]
    ).sum()

    # Cost from missing risky transactions
    cost_false_negative = (
        profit_margin[false_negative]
        + chargeback_fee[false_negative]
    ).sum()

    net_value = (
        value_from_legitimate
        - cost_false_positive
        + value_from_true_positive
        - cost_false_negative
    )

    return {
        "threshold": float(threshold),
        "net_value_inr": float(net_value),
        "true_positive": int(true_positive.sum()),
        "true_negative": int(true_negative.sum()),
        "false_positive": int(false_positive.sum()),
        "false_negative": int(false_negative.sum()),
    }


def optimize_threshold(
    y_test,
    probabilities,
    financial_df,
):
    """Find the threshold producing maximum financial value."""

    print("\n" + "=" * 70)
    print("💰 FINANCIAL COST-BOUNDARY OPTIMIZATION")
    print("=" * 70)

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    results = []

    for threshold in thresholds:
        result = calculate_financial_value(
            y_true=y_test,
            probabilities=probabilities,
            financial_df=financial_df,
            threshold=threshold,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["net_value_inr"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    print(
        f"\n🏆 Optimal Threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"💰 Maximum Net Value  : "
        f"₹{best_row['net_value_inr']:,.2f}"
    )

    print(
        f"TP: {int(best_row['true_positive']):,} | "
        f"TN: {int(best_row['true_negative']):,} | "
        f"FP: {int(best_row['false_positive']):,} | "
        f"FN: {int(best_row['false_negative']):,}"
    )

    return best_threshold, results_df


# ============================================================
# 9. SAVE MODEL
# ============================================================

def save_artifacts(
    model,
    optimal_threshold,
    financial_results,
):
    """Save model and metadata for inference/dashboard use."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    best_result = financial_results.loc[
        financial_results["net_value_inr"].idxmax()
    ]

    metadata = {
        "model_type": "XGBClassifier",
        "model_version": "horae-risk-v1",
        "optimal_threshold": float(
            optimal_threshold
        ),
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "target": TARGET,
        "financial_optimization": {
            "objective": "maximize_net_value_inr",
            "best_net_value_inr": float(
                best_result["net_value_inr"]
            ),
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("💾 MODEL ARTIFACTS SAVED")
    print("=" * 70)

    print(f"Model    : {MODEL_PATH}")
    print(f"Metadata : {METADATA_PATH}")


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " H O R A E  —  R I S K  M O D E L  T R A I N I N G ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Load data
    df = load_data()

    # Features and target
    feature_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = df[feature_columns]
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 70)
    print("✂️ TRAIN / TEST SPLIT")
    print("=" * 70)

    print(f"Training records : {len(X_train):,}")
    print(f"Testing records  : {len(X_test):,}")

    # Preprocessor
    preprocessor = build_preprocessor()

    # Train
    model = train_model(
        X_train,
        y_train,
        preprocessor,
    )

    # Evaluate
    probabilities = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # Financial data corresponding to test set
    financial_test = df.loc[
        X_test.index,
        [
            "profit_margin_inr",
            "chargeback_fee_inr",
        ],
    ].copy()

    # Optimize threshold
    optimal_threshold, financial_results = (
        optimize_threshold(
            y_test=y_test.to_numpy(),
            probabilities=probabilities,
            financial_df=financial_test,
        )
    )

    # Save model
    save_artifacts(
        model=model,
        optimal_threshold=optimal_threshold,
        financial_results=financial_results,
    )

    print("\n" + "=" * 70)
    print("🎉 HORAЕ TRAINING PIPELINE COMPLETE")
    print("=" * 70)

    print("\nNext step:")
    print("  → Build src/rag_engine.py")


if __name__ == "__main__":
    main()
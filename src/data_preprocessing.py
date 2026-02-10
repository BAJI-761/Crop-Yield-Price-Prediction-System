"""
data_preprocessing.py - Data cleaning, encoding, scaling, and feature selection.

Loads raw CSVs from data/raw/, applies the following transformations:
  1. Handle missing values  (median for numeric, mode for categorical)
  2. Label-encode categorical features
  3. StandardScaler on numeric features
  4. Save processed DataFrames + fitted encoders/scalers to data/processed/

Usage:
    python src/data_preprocessing.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    YIELD_TARGET, PRICE_TARGET,
    ensure_dirs, print_section,
)


# ─── Step 1: Load Data ──────────────────────────────────────────────────────────

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw yield and price CSVs."""
    yield_path = os.path.join(DATA_RAW_DIR, "crop_yield_data.csv")
    price_path = os.path.join(DATA_RAW_DIR, "crop_price_data.csv")

    df_yield = pd.read_csv(yield_path)
    df_price = pd.read_csv(price_path)

    print(f"  Loaded yield data : {df_yield.shape}")
    print(f"  Loaded price data : {df_price.shape}")
    return df_yield, df_price


# ─── Step 2: Handle Missing Values ──────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Fill missing values — median for numeric, mode for categorical."""
    missing_before = df.isna().sum().sum()

    for col in NUMERICAL_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Handle target columns if present
    for tgt in [YIELD_TARGET, PRICE_TARGET]:
        if tgt in df.columns and df[tgt].isna().any():
            df[tgt].fillna(df[tgt].median(), inplace=True)

    missing_after = df.isna().sum().sum()
    print(f"  [{label}] Missing values: {missing_before} → {missing_after}")
    return df


# ─── Step 3: Encode Categorical Variables ────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame, label: str, encoders: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    If `encoders` dict is provided, reuse existing encoders (for inference).
    Returns (encoded_df, encoders_dict).
    """
    if encoders is None:
        encoders = {}

    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue

        if col not in encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else -1
            )

    print(f"  [{label}] Encoded {len(encoders)} categorical columns")
    return df, encoders


# ─── Step 4: Scale Numeric Features ─────────────────────────────────────────────

def scale_features(
    df: pd.DataFrame, label: str, scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler to numeric feature columns.
    If `scaler` is provided, use transform only (for inference).
    """
    df = df.copy()
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]

    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print(f"  [{label}] Fitted & scaled {len(num_cols)} numeric features")
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        print(f"  [{label}] Transformed {len(num_cols)} numeric features (existing scaler)")

    return df, scaler


# ─── Step 5: Feature Selection (correlation-based) ──────────────────────────────

def show_feature_correlations(df: pd.DataFrame, target: str, label: str):
    """Print correlation of each feature with the target variable."""
    if target not in df.columns:
        return
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()[target].drop(target, errors="ignore").sort_values(
        ascending=False, key=abs
    )
    print(f"\n  [{label}] Feature correlations with '{target}':")
    for feat, val in corr.items():
        bar = "█" * int(abs(val) * 30)
        sign = "+" if val >= 0 else "-"
        print(f"    {sign} {feat:<35s} {val:+.4f}  {bar}")


# ─── Pipeline ───────────────────────────────────────────────────────────────────

def preprocess_dataset(df: pd.DataFrame, target: str, label: str) -> pd.DataFrame:
    """Full preprocessing pipeline for a single dataset."""
    print_section(f"Preprocessing: {label}")

    # 1. Missing values
    df = handle_missing(df, label)

    # 2. Encode categoricals
    df, encoders = encode_categoricals(df, label)

    # 3. Scale numerics
    df, scaler = scale_features(df, label)

    # 4. Feature correlations (informational)
    show_feature_correlations(df, target, label)

    # 5. Save artefacts
    processed_path = os.path.join(DATA_PROCESSED_DIR, f"{label}_processed.csv")
    df.to_csv(processed_path, index=False)
    print(f"\n  ✓ Saved processed data → {processed_path}")

    encoders_path = os.path.join(DATA_PROCESSED_DIR, f"{label}_encoders.pkl")
    joblib.dump(encoders, encoders_path)
    print(f"  ✓ Saved encoders       → {encoders_path}")

    scaler_path = os.path.join(DATA_PROCESSED_DIR, f"{label}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved scaler         → {scaler_path}")

    return df


def main():
    ensure_dirs()
    print_section("Data Preprocessing Pipeline")

    df_yield, df_price = load_raw_data()

    df_yield_processed = preprocess_dataset(df_yield, YIELD_TARGET, "yield")
    df_price_processed = preprocess_dataset(df_price, PRICE_TARGET, "price")

    print_section("Preprocessing Complete")
    print(f"  Yield processed shape : {df_yield_processed.shape}")
    print(f"  Price processed shape : {df_price_processed.shape}")
    print(f"  All artefacts saved to: {DATA_PROCESSED_DIR}\n")


if __name__ == "__main__":
    main()

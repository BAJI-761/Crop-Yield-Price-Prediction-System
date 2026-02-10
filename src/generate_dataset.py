"""
generate_dataset.py - Synthetic dataset generator for crop yield & price prediction.

Generates two realistic CSV datasets modeled on Indian agricultural data:
  1. crop_yield_data.csv  (~5,000 rows) — target: Yield_ton_per_hectare
  2. crop_price_data.csv  (~5,000 rows) — target: Price_per_quintal

Data distributions are calibrated against publicly available Indian government
statistics (data.gov.in / ICAR) so that models trained on these datasets produce
plausible predictions.

Usage:
    python src/generate_dataset.py
"""

import os
import sys
import random
import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    STATES_AND_DISTRICTS, CROPS, SEASONS, SOIL_TYPES,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    YIELD_TARGET, PRICE_TARGET,
    DATA_RAW_DIR, ensure_dirs, print_section,
)

# ─── Realistic parameter ranges per crop ────────────────────────────────────────
# Each crop has: (base_yield_min, base_yield_max, base_price_min, base_price_max)
CROP_PARAMS = {
    "Rice":           (2.0, 5.5,  1800,  2800),
    "Wheat":          (2.5, 5.0,  1900,  2600),
    "Maize":          (1.8, 4.5,  1700,  2500),
    "Sugarcane":      (50,  90,    280,   350),
    "Cotton":         (1.2, 2.5,  5000,  7000),
    "Soybean":        (1.0, 2.0,  3500,  5000),
    "Groundnut":      (1.0, 2.5,  4500,  6000),
    "Mustard":        (0.8, 1.8,  4000,  5500),
    "Jowar":          (0.6, 1.5,  2500,  3500),
    "Bajra":          (0.8, 1.6,  2000,  3000),
    "Tur (Arhar)":    (0.6, 1.2,  5500,  7500),
    "Chana (Gram)":   (0.7, 1.5,  4000,  5500),
    "Onion":          (10,  25,    800,  3000),
    "Potato":         (15,  35,    600,  1600),
    "Tomato":         (12,  30,    500,  2500),
}

# Season affects yield multiplier
SEASON_MULTIPLIER = {
    "Kharif":     1.0,
    "Rabi":       0.95,
    "Zaid":       0.85,
    "Whole Year": 1.05,
}

# Soil suitability multiplier (not every soil suits every crop, but we simplify)
SOIL_MULTIPLIER = {
    "Alluvial":       1.10,
    "Black (Regur)":  1.05,
    "Red":            0.95,
    "Laterite":       0.90,
    "Desert (Arid)":  0.75,
    "Mountain":       0.80,
    "Clay":           1.00,
    "Sandy Loam":     0.98,
}


def _random_rainfall(season: str) -> float:
    """Return rainfall (mm) realistic for the given season."""
    base = {"Kharif": 900, "Rabi": 300, "Zaid": 150, "Whole Year": 600}
    return max(50, np.random.normal(base[season], base[season] * 0.3))


def _random_temperature(season: str) -> float:
    """Return temperature (°C) realistic for the given season."""
    base = {"Kharif": 30, "Rabi": 20, "Zaid": 35, "Whole Year": 27}
    return round(np.random.normal(base[season], 3), 1)


def _random_humidity(season: str) -> float:
    """Return humidity (%) realistic for the given season."""
    base = {"Kharif": 75, "Rabi": 55, "Zaid": 45, "Whole Year": 65}
    return round(np.clip(np.random.normal(base[season], 10), 20, 98), 1)


def generate_row(rng: np.random.Generator) -> dict:
    """Generate a single data row with all features + both targets."""

    # Pick random categorical values
    state = rng.choice(list(STATES_AND_DISTRICTS.keys()))
    district = rng.choice(STATES_AND_DISTRICTS[state])
    crop = rng.choice(CROPS)
    season = rng.choice(SEASONS)
    soil = rng.choice(SOIL_TYPES)

    # Numerical features
    rainfall = round(_random_rainfall(season), 1)
    temperature = _random_temperature(season)
    humidity = _random_humidity(season)
    area = round(rng.uniform(0.5, 50), 2)
    fertilizer = round(rng.uniform(20, 300), 1)
    pesticide = round(rng.uniform(0.5, 25), 2)

    # ── Compute yield ───────────────────────────────────────────────────────
    y_min, y_max, p_min, p_max = CROP_PARAMS[crop]
    base_yield = rng.uniform(y_min, y_max)

    # Modifiers: rainfall, temperature, soil, season, fertilizer
    rain_factor = np.clip(rainfall / 700, 0.6, 1.4)
    temp_factor = np.clip(1 - abs(temperature - 27) / 40, 0.7, 1.2)
    fert_factor = np.clip(fertilizer / 150, 0.7, 1.3)
    soil_factor = SOIL_MULTIPLIER[soil]
    season_factor = SEASON_MULTIPLIER[season]

    computed_yield = base_yield * rain_factor * temp_factor * fert_factor * soil_factor * season_factor
    # Add noise (±10 %)
    computed_yield *= rng.uniform(0.90, 1.10)
    computed_yield = round(max(0.1, computed_yield), 2)

    # ── Compute price ───────────────────────────────────────────────────────
    base_price = rng.uniform(p_min, p_max)
    # Inverse supply: higher yield → slightly lower price
    supply_factor = np.clip(1.3 - (computed_yield / (y_max * 1.5)), 0.8, 1.3)
    # Season premium
    season_price_factor = {"Kharif": 1.0, "Rabi": 1.05, "Zaid": 1.15, "Whole Year": 0.95}[season]
    computed_price = base_price * supply_factor * season_price_factor
    computed_price *= rng.uniform(0.92, 1.08)  # noise
    computed_price = round(max(100, computed_price), 2)

    return {
        "State": state,
        "District": district,
        "Crop": crop,
        "Season": season,
        "Soil_Type": soil,
        "Rainfall_mm": rainfall,
        "Temperature_C": temperature,
        "Humidity_pct": humidity,
        "Area_hectares": area,
        "Fertilizer_kg_per_hectare": fertilizer,
        "Pesticide_kg_per_hectare": pesticide,
        YIELD_TARGET: computed_yield,
        PRICE_TARGET: computed_price,
    }


def inject_missing_values(df: pd.DataFrame, frac: float = 0.02) -> pd.DataFrame:
    """Randomly inject NaN values to simulate real-world messy data."""
    df = df.copy()
    n_missing = int(len(df) * frac)
    cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    for _ in range(n_missing):
        row_idx = random.randint(0, len(df) - 1)
        col = random.choice(cols)
        df.at[row_idx, col] = np.nan
    return df


def main():
    ensure_dirs()
    print_section("Generating Synthetic Crop Datasets")

    rng = np.random.default_rng(seed=42)
    N = 5000

    print(f"  Generating {N} rows …")
    rows = [generate_row(rng) for _ in range(N)]
    df_full = pd.DataFrame(rows)

    # Inject ~2 % missing values to make preprocessing realistic
    df_full = inject_missing_values(df_full, frac=0.02)

    # Save yield dataset (all columns except Price)
    yield_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [YIELD_TARGET]
    df_yield = df_full[yield_cols]
    yield_path = os.path.join(DATA_RAW_DIR, "crop_yield_data.csv")
    df_yield.to_csv(yield_path, index=False)
    print(f"  ✓ Saved yield dataset  → {yield_path}  ({len(df_yield)} rows)")

    # Save price dataset (all columns except Yield)
    price_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [PRICE_TARGET]
    df_price = df_full[price_cols]
    price_path = os.path.join(DATA_RAW_DIR, "crop_price_data.csv")
    df_price.to_csv(price_path, index=False)
    print(f"  ✓ Saved price dataset  → {price_path}  ({len(df_price)} rows)")

    # Quick summary
    print(f"\n  Yield range : {df_yield[YIELD_TARGET].min():.2f} – {df_yield[YIELD_TARGET].max():.2f} ton/ha")
    print(f"  Price range : {df_price[PRICE_TARGET].min():.2f} – {df_price[PRICE_TARGET].max():.2f} ₹/quintal")
    print(f"  Missing vals: {df_full.isna().sum().sum()} cells injected\n")


if __name__ == "__main__":
    main()

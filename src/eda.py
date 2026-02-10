"""
eda.py - Exploratory Data Analysis and Visualization.

Generates publication-quality plots and saves them to visualizations/:
  1. Correlation heatmap
  2. Target distribution histograms
  3. Crop-wise yield & price box plots
  4. State-wise average yield & price bar charts
  5. Feature importance charts (from Random Forest)
  6. Rainfall vs. Yield scatter

Usage:
    python src/eda.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, VISUALIZATIONS_DIR,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    YIELD_TARGET, PRICE_TARGET,
    ensure_dirs, print_section,
)

# ─── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.1)
FIGSIZE = (12, 8)
DPI = 150


def save_fig(fig, name: str):
    """Save figure and close."""
    path = os.path.join(VISUALIZATIONS_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved → {path}")


# ─── 1. Correlation Heatmap ─────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, title: str, filename: str):
    """Generate a triangular correlation heatmap for numeric features."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    save_fig(fig, filename)


# ─── 2. Distribution Plots ──────────────────────────────────────────────────────

def plot_distributions(df_yield: pd.DataFrame, df_price: pd.DataFrame):
    """Histograms of yield and price targets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Yield distribution
    axes[0].hist(df_yield[YIELD_TARGET].dropna(), bins=50, color="#2ecc71", edgecolor="white", alpha=0.85)
    axes[0].set_title("Crop Yield Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Yield (ton / hectare)")
    axes[0].set_ylabel("Frequency")

    # Price distribution
    axes[1].hist(df_price[PRICE_TARGET].dropna(), bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    axes[1].set_title("Crop Market Price Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Price (₹ / quintal)")
    axes[1].set_ylabel("Frequency")

    fig.suptitle("Target Variable Distributions", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, "target_distributions.png")


# ─── 3. Crop-wise Box Plots ─────────────────────────────────────────────────────

def plot_cropwise_boxplots(df_yield: pd.DataFrame, df_price: pd.DataFrame):
    """Box plots of yield and price grouped by crop type."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    sns.boxplot(data=df_yield, x="Crop", y=YIELD_TARGET, ax=axes[0],
                palette="Set2", showfliers=False)
    axes[0].set_title("Crop-wise Yield Distribution", fontsize=14, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)

    sns.boxplot(data=df_price, x="Crop", y=PRICE_TARGET, ax=axes[1],
                palette="Set3", showfliers=False)
    axes[1].set_title("Crop-wise Market Price Distribution", fontsize=14, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    save_fig(fig, "cropwise_boxplots.png")


# ─── 4. State-wise Bar Charts ───────────────────────────────────────────────────

def plot_statewise_averages(df_yield: pd.DataFrame, df_price: pd.DataFrame):
    """Average yield and price by state."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    yield_by_state = df_yield.groupby("State")[YIELD_TARGET].mean().sort_values(ascending=False)
    yield_by_state.plot(kind="barh", ax=axes[0], color="#27ae60", edgecolor="white")
    axes[0].set_title("Average Yield by State", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Yield (ton / hectare)")

    price_by_state = df_price.groupby("State")[PRICE_TARGET].mean().sort_values(ascending=False)
    price_by_state.plot(kind="barh", ax=axes[1], color="#2980b9", edgecolor="white")
    axes[1].set_title("Average Market Price by State", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Price (₹ / quintal)")

    fig.tight_layout()
    save_fig(fig, "statewise_averages.png")


# ─── 5. Feature Importance (Random Forest) ──────────────────────────────────────

def plot_feature_importance(df: pd.DataFrame, target: str, title: str, filename: str):
    """Train a quick Random Forest and plot feature importances."""
    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importances)))
    importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Importance Score")

    # Annotate values
    for i, (val, name) in enumerate(zip(importances, importances.index)):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    save_fig(fig, filename)


# ─── 6. Rainfall vs Yield Scatter ───────────────────────────────────────────────

def plot_rainfall_vs_yield(df: pd.DataFrame):
    """Scatter of rainfall against yield, coloured by crop."""
    if "Rainfall_mm" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    crops = df["Crop"].unique() if "Crop" in df.columns else ["All"]
    palette = sns.color_palette("husl", len(crops))

    for crop, color in zip(crops, palette):
        subset = df[df["Crop"] == crop] if "Crop" in df.columns else df
        ax.scatter(subset["Rainfall_mm"], subset[YIELD_TARGET],
                   alpha=0.4, s=12, label=crop, color=color)

    ax.set_title("Rainfall vs Crop Yield", fontsize=15, fontweight="bold")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Yield (ton / hectare)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    save_fig(fig, "rainfall_vs_yield.png")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    print_section("Exploratory Data Analysis")

    # Load raw data (for readable labels in box/bar plots)
    df_yield_raw = pd.read_csv(os.path.join(DATA_RAW_DIR, "crop_yield_data.csv"))
    df_price_raw = pd.read_csv(os.path.join(DATA_RAW_DIR, "crop_price_data.csv"))

    # Load processed data (for correlation / feature importance with encoded values)
    df_yield_proc = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "yield_processed.csv"))
    df_price_proc = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "price_processed.csv"))

    print("  Generating visualizations …\n")

    # 1. Correlation heatmaps
    plot_correlation_heatmap(df_yield_proc, "Correlation Heatmap — Yield Dataset", "correlation_heatmap_yield.png")
    plot_correlation_heatmap(df_price_proc, "Correlation Heatmap — Price Dataset", "correlation_heatmap_price.png")

    # 2. Target distributions
    plot_distributions(df_yield_raw, df_price_raw)

    # 3. Crop-wise box plots
    plot_cropwise_boxplots(df_yield_raw, df_price_raw)

    # 4. State-wise averages
    plot_statewise_averages(df_yield_raw, df_price_raw)

    # 5. Feature importance
    plot_feature_importance(df_yield_proc, YIELD_TARGET,
                           "Feature Importance — Yield Prediction",
                           "feature_importance_yield.png")
    plot_feature_importance(df_price_proc, PRICE_TARGET,
                           "Feature Importance — Price Prediction",
                           "feature_importance_price.png")

    # 6. Rainfall vs Yield
    plot_rainfall_vs_yield(df_yield_raw)

    print_section("EDA Complete")
    print(f"  All plots saved to: {VISUALIZATIONS_DIR}\n")


if __name__ == "__main__":
    main()

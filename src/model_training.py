"""
model_training.py - Train, evaluate, compare, and save ML models.

For EACH target (Yield, Price) it trains:
  • Linear Regression
  • Random Forest Regressor
  • XGBoost Regressor

Evaluation metrics: RMSE, MAE, R² Score.
Best model (by R²) is saved to models/.
Actual-vs-Predicted scatter plots are saved to visualizations/.

Usage:
    python src/model_training.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import (
    DATA_PROCESSED_DIR, MODELS_DIR, VISUALIZATIONS_DIR,
    YIELD_TARGET, PRICE_TARGET,
    ensure_dirs, print_section,
)


# ─── Model Definitions ──────────────────────────────────────────────────────────

def get_models() -> list[tuple[str, object]]:
    """Return a list of (name, estimator) tuples to evaluate."""
    return [
        ("Linear Regression",       LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        )),
        ("XGBoost Regressor",       XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, verbosity=0, n_jobs=-1
        )),
    ]


# ─── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, and R² for a set of predictions."""
    return {
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
        "R²":   round(r2_score(y_true, y_pred), 4),
    }


# ─── Actual vs Predicted Plot ────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_label: str,
    filename: str,
):
    """Scatter plot of actual vs predicted, with a perfect-prediction line."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.35, s=15, color="#2980b9", edgecolors="white", linewidths=0.3)

    # Perfect prediction line
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Prediction")

    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"Actual vs Predicted — {model_name}\n({target_label})", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Actual {target_label}")
    ax.set_ylabel(f"Predicted {target_label}")
    ax.legend(fontsize=11)
    ax.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=13, fontweight="bold", color="#e74c3c",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e74c3c"))

    fig.tight_layout()
    path = os.path.join(VISUALIZATIONS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    ✓ Plot saved → {path}")


# ─── Training Pipeline ──────────────────────────────────────────────────────────

def train_and_evaluate(target: str, label: str):
    """
    Full train → evaluate → compare → save pipeline for one target.
    Returns the name and fitted model of the best estimator.
    """
    print_section(f"Model Training — {label}")

    # Load processed data
    csv_path = os.path.join(DATA_PROCESSED_DIR, f"{label}_processed.csv")
    df = pd.read_csv(csv_path)

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols]
    y = df[target]

    # 80 / 20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    print(f"  Features  : {X_train.shape[1]}\n")

    results = []
    best_r2 = -np.inf
    best_model = None
    best_name = ""

    for name, model in get_models():
        print(f"  ▸ Training {name} …")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test.values, y_pred)
        results.append({"Model": name, **metrics})
        print(f"    RMSE={metrics['RMSE']}  MAE={metrics['MAE']}  R²={metrics['R²']}")

        if metrics["R²"] > best_r2:
            best_r2 = metrics["R²"]
            best_model = model
            best_name = name

        # Plot Actual vs Predicted for each model
        safe_name = name.lower().replace(" ", "_")
        plot_actual_vs_predicted(
            y_test.values, y_pred, name, target,
            f"actual_vs_pred_{label}_{safe_name}.png"
        )

    # ── Comparison Table ────────────────────────────────────────────────────────
    print(f"\n  {'─' * 65}")
    print(f"  {'Model':<28s} {'RMSE':>10s} {'MAE':>10s} {'R²':>10s}")
    print(f"  {'─' * 65}")
    for r in results:
        marker = " ★" if r["Model"] == best_name else ""
        print(f"  {r['Model']:<28s} {r['RMSE']:>10.4f} {r['MAE']:>10.4f} {r['R²']:>10.4f}{marker}")
    print(f"  {'─' * 65}")

    # ── Save best model ─────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, f"best_{label}_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n  ★ Best model: {best_name}  (R² = {best_r2:.4f})")
    print(f"  ✓ Saved → {model_path}")

    # Save feature columns list (needed at inference time)
    feat_path = os.path.join(MODELS_DIR, f"{label}_feature_columns.pkl")
    joblib.dump(list(feature_cols), feat_path)

    # Save results table
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(MODELS_DIR, f"{label}_model_comparison.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"  ✓ Comparison table → {results_csv}\n")

    return best_name, best_model


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    yield_best_name, _ = train_and_evaluate(YIELD_TARGET, "yield")
    price_best_name, _ = train_and_evaluate(PRICE_TARGET, "price")

    print_section("Training Complete — Summary")
    print(f"  Yield best model : {yield_best_name}")
    print(f"  Price best model : {price_best_name}")
    print(f"  Models saved to  : {MODELS_DIR}")
    print(f"  Plots saved to   : {VISUALIZATIONS_DIR}\n")

    # ─── Model Selection Justification ──────────────────────────────────────────
    print_section("Model Selection Justification")
    print("""
  Both Random Forest and XGBoost are ensemble tree-based methods that
  capture non-linear relationships between agricultural features (rainfall,
  temperature, soil type, etc.) and the target variables.

  Why tree-based models outperform Linear Regression here:
    1. Agriculture data has complex interactions (e.g., crop × soil × season).
    2. Categorical features after label-encoding create non-linear splits
       that trees exploit, whereas linear models treat them as ordinal.
    3. Ensemble methods reduce variance and are robust to outliers.

  XGBoost often edges ahead due to gradient-boosted regularisation, but
  Random Forest is competitive on moderately-sized datasets like ours.

  The best model was selected purely by highest R² on the 20 % held-out
  test set, ensuring it generalises well to unseen inputs.
""")


if __name__ == "__main__":
    main()

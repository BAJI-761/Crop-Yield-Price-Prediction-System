# 🌾 Crop Yield & Market Price Prediction System

> An end-to-end Machine Learning system that predicts **crop yield** (ton/hectare) and **market price** (₹/quintal) for Indian farmers — powered by **Scikit-Learn**, **XGBoost**, and **Streamlit**.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Model Selection Logic](#model-selection-logic)
- [Visualizations](#visualizations)
- [Real-World Impact](#real-world-impact)
- [Resume Bullet Points](#resume-bullet-points)

---

## 🏗️ Project Overview

Indian agriculture is the backbone of the nation's economy, yet millions of farmers lack access to data-driven tools for decision making. This project addresses two critical needs:

1. **Yield Prediction** — How much crop output (tons/hectare) can a farmer expect given location, crop type, soil, weather, and farming practices?
2. **Price Prediction** — What market price (₹/quintal) can a farmer expect for the harvested crop?

The system trains & compares three ML models, selects the best-performing one, and deploys it via a farmer-friendly **Streamlit** web application.

---

## 🏛️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Raw Dataset │────▶│  Preprocessing   │────▶│  EDA & Visuals  │
│  (CSV files) │     │  (clean, encode,  │     │  (heatmaps,     │
│              │     │   scale)          │     │   box plots)    │
└──────────────┘     └────────┬─────────┘     └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Model Training  │
                    │  ├─ LinearReg    │
                    │  ├─ RandomForest │
                    │  └─ XGBoost     │
                    └────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Best Model (.pkl) │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Streamlit App   │
                    │  (User Inputs →   │
                    │   Predictions)    │
                    └───────────────────┘
```

---

## 🔄 Data Flow

1. **Data Generation** — `generate_dataset.py` creates 5,000-row synthetic datasets calibrated to Indian crop statistics
2. **Preprocessing** — Missing values filled (median/mode), categoricals label-encoded, numerics standard-scaled
3. **EDA** — 8+ visualizations generated for exploratory analysis
4. **Training** — 3 models trained per target with 80/20 split; evaluated by RMSE, MAE, R²
5. **Deployment** — Best model served via Streamlit; user inputs encoded with saved encoders/scalers

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌱 Dual Prediction | Predict both yield AND market price in one click |
| 📊 Model Comparison | Side-by-side RMSE / MAE / R² for 3 models |
| 📈 Rich Visualizations | Heatmaps, feature importance, actual vs predicted |
| 💵 Revenue Estimate | Combines yield × price × area for total revenue |
| 🧑‍🌾 Farmer-Friendly UI | Clean Streamlit interface with dropdowns & sliders |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Libraries | Scikit-Learn, XGBoost |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Serialization | Joblib |

---

## 📁 Folder Structure

```
Crop Yield & Price Prediction System/
├── data/
│   ├── raw/                          # Raw CSV datasets
│   │   ├── crop_yield_data.csv
│   │   └── crop_price_data.csv
│   └── processed/                    # Cleaned & encoded data + encoders
│       ├── yield_processed.csv
│       ├── price_processed.csv
│       ├── yield_encoders.pkl
│       ├── price_encoders.pkl
│       ├── yield_scaler.pkl
│       └── price_scaler.pkl
├── models/                           # Trained model files
│   ├── best_yield_model.pkl
│   ├── best_price_model.pkl
│   ├── yield_model_comparison.csv
│   └── price_model_comparison.csv
├── visualizations/                   # Saved PNG plots
├── src/
│   ├── __init__.py
│   ├── utils.py                      # Shared constants & paths
│   ├── generate_dataset.py           # Dataset generator
│   ├── data_preprocessing.py         # Cleaning & encoding pipeline
│   ├── eda.py                        # Exploratory analysis plots
│   └── model_training.py             # Train, evaluate, compare models
├── app/
│   └── streamlit_app.py              # Streamlit web application
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/crop-yield-price-prediction.git
cd "Crop Yield & Price Prediction System"

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset
python src/generate_dataset.py

# 5. Run preprocessing
python src/data_preprocessing.py

# 6. Run EDA (generates visualizations)
python src/eda.py

# 7. Train models
python src/model_training.py

# 8. Launch Streamlit app
streamlit run app/streamlit_app.py
```

---

## 📊 Model Performance

### Yield Prediction

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 13.6973 | 7.8493 | 0.1391 |
| **Random Forest** ★ | **3.6237** | **1.4322** | **0.9397** |
| XGBoost | 3.8627 | 1.4743 | 0.9315 |

### Price Prediction

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | 1914.5467 | 1582.1259 | 0.1448 |
| **Random Forest** ★ | **563.8055** | **424.7541** | **0.9258** |
| XGBoost | 566.1396 | 426.6924 | 0.9252 |

> ★ = Best model selected based on highest R² on 20% held-out test set

---

## 🧠 Model Selection Logic

| Criterion | Linear Regression | Random Forest | XGBoost |
|---|---|---|---|
| Handles non-linearity | ❌ | ✅ | ✅ |
| Captures feature interactions | ❌ | ✅ | ✅ |
| Robust to outliers | ❌ | ✅ | ✅ |
| Regularization | ❌ | Implicit (bagging) | ✅ (L1/L2) |
| Interpretability | ✅ High | ⚡ Medium | ⚡ Medium |

**Why tree-based models win on agricultural data:**
- Crop yield depends on **complex interactions** (crop × soil × season × weather)
- Label-encoded categoricals create **non-linear decision boundaries** that trees exploit
- Ensemble methods **reduce variance** and are robust to noisy real-world data
- XGBoost's gradient boosting with regularization typically achieves the **highest R²**

---

## 📈 Visualizations

The project generates the following plots (saved in `visualizations/`):

- **Correlation Heatmaps** — Reveal multicollinearity and feature-target relationships
- **Target Distributions** — Show yield and price spread across the dataset
- **Crop-wise Box Plots** — Compare yield/price distributions across 15 crops
- **State-wise Averages** — Geographic patterns in yield and pricing
- **Feature Importance** — Random Forest-derived importance scores
- **Actual vs Predicted** — Scatter plots with R² for each model
- **Rainfall vs Yield** — Scatter coloured by crop type

---

## 🌍 Real-World Impact

| Impact Area | Description |
|---|---|
| 🧑‍🌾 Farmer Decision Making | Helps farmers choose which crop to grow based on predicted yield & price |
| 💰 Revenue Planning | Revenue estimates help farmers negotiate better with middlemen (mandis) |
| 📦 Supply Chain | Aggregated predictions can inform cold storage & logistics planning |
| 🏦 Credit & Insurance | Banks and insurers can use yield predictions for loan/premium assessment |
| 🏛️ Policy Making | Government can use state/district-level predictions for MSP and subsidy planning |

---

## 📝 Resume Bullet Points

- **Built an end-to-end ML pipeline** for predicting crop yield (ton/ha) and market price (₹/quintal) using Scikit-Learn and XGBoost, achieving **R² = 0.94** on yield and **R² = 0.93** on price test data
- **Designed and deployed** a Streamlit web application enabling farmers to get real-time predictions with revenue estimates based on location, crop, soil, and weather inputs
- **Compared 3 regression models** (Linear Regression, Random Forest, XGBoost) using RMSE, MAE, and R² metrics; selected Random Forest for best generalization on agricultural data
- **Engineered a complete data pipeline** including missing value imputation, label encoding, standard scaling, and correlation-based feature analysis on 5,000+ agricultural records
- **Created 8+ publication-quality visualizations** (correlation heatmaps, feature importance charts, actual vs. predicted plots) for data-driven model interpretation

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for Indian Farmers
</p>

"""
streamlit_app.py - Crop Yield & Price Prediction Web Application.

Clean, minimal Streamlit UI for predicting crop yield and market price.

Launch:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    STATES_AND_DISTRICTS, CROPS, SEASONS, SOIL_TYPES,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    YIELD_TARGET, PRICE_TARGET,
    DATA_PROCESSED_DIR, MODELS_DIR, VISUALIZATIONS_DIR,
)

# ── Page config ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Crop Yield & Price Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal, clean CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(label):
    model = joblib.load(os.path.join(MODELS_DIR, f"best_{label}_model.pkl"))
    encoders = joblib.load(os.path.join(DATA_PROCESSED_DIR, f"{label}_encoders.pkl"))
    scaler = joblib.load(os.path.join(DATA_PROCESSED_DIR, f"{label}_scaler.pkl"))
    feat_cols = joblib.load(os.path.join(MODELS_DIR, f"{label}_feature_columns.pkl"))
    return model, encoders, scaler, feat_cols


@st.cache_data
def load_comparison(label):
    path = os.path.join(MODELS_DIR, f"{label}_model_comparison.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def encode_and_scale(user_data, encoders, scaler, feat_cols):
    df = pd.DataFrame([user_data])
    for col in CATEGORICAL_FEATURES:
        if col in encoders and col in df.columns:
            le = encoders[col]
            val = str(df[col].iloc[0])
            df[col] = le.transform([val])[0] if val in set(le.classes_) else -1
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    df[num_cols] = scaler.transform(df[num_cols])
    return df.reindex(columns=feat_cols, fill_value=0).values


# ── Sidebar inputs ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Input Parameters")

    st.subheader("Location")
    state = st.selectbox("State", list(STATES_AND_DISTRICTS.keys()))
    district = st.selectbox("District", STATES_AND_DISTRICTS[state])

    st.subheader("Crop information")
    crop = st.selectbox("Crop type", CROPS)
    season = st.selectbox("Season", SEASONS)
    soil_type = st.selectbox("Soil type", SOIL_TYPES)

    st.subheader("Weather")
    rainfall = st.slider("Rainfall (mm)", 50.0, 2000.0, 800.0, step=10.0)
    temperature = st.slider("Temperature (°C)", 10.0, 50.0, 28.0, step=0.5)
    humidity = st.slider("Humidity (%)", 20.0, 98.0, 65.0, step=1.0)

    st.subheader("Farm details")
    area = st.number_input("Area (hectares)", 0.1, 500.0, 5.0, step=0.5)
    fertilizer = st.slider("Fertilizer (kg/ha)", 10.0, 500.0, 120.0, step=5.0)
    pesticide = st.slider("Pesticide (kg/ha)", 0.5, 50.0, 8.0, step=0.5)

    st.divider()
    predict = st.button("Predict", use_container_width=True, type="primary")


# ── Main area ────────────────────────────────────────────────────────────────────

st.title("Crop Yield & Market Price Predictor")
st.caption("Estimate how much your farm will produce and what price you can expect at the market.")

st.divider()

if predict:
    user_data = {
        "State": state, "District": district, "Crop": crop,
        "Season": season, "Soil_Type": soil_type,
        "Rainfall_mm": rainfall, "Temperature_C": temperature,
        "Humidity_pct": humidity, "Area_hectares": area,
        "Fertilizer_kg_per_hectare": fertilizer,
        "Pesticide_kg_per_hectare": pesticide,
    }

    try:
        y_model, y_enc, y_scaler, y_feats = load_model("yield")
        p_model, p_enc, p_scaler, p_feats = load_model("price")

        y_pred = max(0.01, y_model.predict(encode_and_scale(user_data, y_enc, y_scaler, y_feats))[0])
        p_pred = max(1.0, p_model.predict(encode_and_scale(user_data, p_enc, p_scaler, p_feats))[0])

        total_tons = y_pred * area
        total_qtl = total_tons * 10
        revenue = total_qtl * p_pred

        # Results
        st.subheader("Prediction results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Yield", f"{y_pred:.2f} ton/ha")
        c2.metric("Market price", f"₹ {p_pred:,.0f} / quintal")
        c3.metric("Estimated revenue", f"₹ {revenue:,.0f}")

        st.divider()

        # Breakdown
        st.subheader("Revenue breakdown")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Yield per hectare", f"{y_pred:.2f} t")
        b2.metric("Total production", f"{total_tons:.1f} tons")
        b3.metric("Total quintals", f"{total_qtl:.1f} qtl")
        b4.metric("Total revenue", f"₹ {revenue:,.0f}")

        st.caption(f"Based on **{crop}** grown on **{area} ha** in **{district}, {state}** during **{season}** season.")

        # Input summary
        with st.expander("View input summary"):
            st.dataframe(pd.DataFrame([user_data]), use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error("Models not found. Run `python src/model_training.py` first.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Yield prediction")
        st.write("How much crop output you can expect per hectare given your soil, weather, and practices.")
    with col2:
        st.subheader("Price forecast")
        st.write("Expected market price per quintal so you can time your sales better.")
    with col3:
        st.subheader("Revenue estimate")
        st.write("Total expected income combining predicted yield, price, and your farm area.")

    st.info("Fill in the details on the left and click **Predict**.")


# ── Model comparison ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("Model performance")

tab1, tab2 = st.tabs(["Yield models", "Price models"])

with tab1:
    df_y = load_comparison("yield")
    if not df_y.empty:
        best = df_y.loc[df_y["R²"].idxmax(), "Model"]
        st.write(f"Best model: **{best}**")
        st.dataframe(df_y, use_container_width=True, hide_index=True)

with tab2:
    df_p = load_comparison("price")
    if not df_p.empty:
        best = df_p.loc[df_p["R²"].idxmax(), "Model"]
        st.write(f"Best model: **{best}**")
        st.dataframe(df_p, use_container_width=True, hide_index=True)


# ── Visualizations ───────────────────────────────────────────────────────────────

st.divider()
st.subheader("Visualizations")

viz_map = {
    "Correlation heatmap (Yield)": "correlation_heatmap_yield.png",
    "Correlation heatmap (Price)": "correlation_heatmap_price.png",
    "Feature importance (Yield)": "feature_importance_yield.png",
    "Feature importance (Price)": "feature_importance_price.png",
    "Target distributions": "target_distributions.png",
    "Crop-wise box plots": "cropwise_boxplots.png",
    "State-wise averages": "statewise_averages.png",
    "Rainfall vs Yield": "rainfall_vs_yield.png",
}

# Add actual-vs-predicted plots dynamically
if os.path.exists(VISUALIZATIONS_DIR):
    for f in sorted(os.listdir(VISUALIZATIONS_DIR)):
        if f.startswith("actual_vs_pred_") and f.endswith(".png"):
            nice_name = f.replace("actual_vs_pred_", "Actual vs Predicted - ").replace(".png", "").replace("_", " ").title()
            viz_map[nice_name] = f

choice = st.selectbox("Select a chart", list(viz_map.keys()), label_visibility="collapsed",
                       placeholder="Choose a visualization")
if choice:
    path = os.path.join(VISUALIZATIONS_DIR, viz_map[choice])
    if os.path.exists(path):
        st.image(path, caption=choice, use_container_width=True)
    else:
        st.warning(f"{viz_map[choice]} not found. Run EDA first.")

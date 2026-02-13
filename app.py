import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ── Load model & scaler
model  = joblib.load("crop_yield_model.pkl")
scaler = joblib.load("crop_yield_scaler.pkl")

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")

st.title("🌾 Crop Yield Predictor")
st.markdown("Fill in the field conditions below to predict the expected crop yield.")
st.divider()

# ── Input Form ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)",     min_value=0.0,   max_value=60.0,  value=25.0)
    humidity    = st.number_input("Humidity (%)",         min_value=0.0,   max_value=100.0, value=60.0)
    moisture    = st.number_input("Moisture (%)",         min_value=0.0,   max_value=100.0, value=40.0)
    nitrogen    = st.number_input("Nitrogen (N)",         min_value=0.0,   max_value=200.0, value=50.0)

with col2:
    potassium   = st.number_input("Potassium (K)",        min_value=0.0,   max_value=200.0, value=50.0)
    phosphorus  = st.number_input("Phosphorus (P)",       min_value=0.0,   max_value=200.0, value=50.0)
    soil_type   = st.selectbox("Soil Type",   ["Sandy", "Loamy", "Black", "Red", "Clayey"])
    crop_type   = st.selectbox("Crop Type",   ["Maize", "Cotton", "Tobacco", "Paddy", 
                                                "Barley", "Wheat", "Millets", 
                                                "Oil seeds", "Pulses", "Ground Nuts"])

fertilizer = st.selectbox("Fertilizer Used", ["Urea", "DAP", "14-35-14", "28-28", "17-17-17", "20-20", "10-26-26"])

st.divider()

# ── Predict Button ───────────────────────────────────────────────
if st.button("🔍 Predict Yield", use_container_width=True):

    # Build input dataframe matching training structure
    input_data = {
        "Temperature": [temperature],
        "Humidity":    [humidity],
        "Moisture":    [moisture],
        "Nitrogen":    [nitrogen],
        "Potassium":   [potassium],
        "Phosphorus":  [phosphorus],
        "Soil_type":   [soil_type],
        "Crop_type":   [crop_type],
        "Fertilizer_name": [fertilizer]
    }

    input_df = pd.DataFrame(input_data)

    # One-hot encode (must match training encoding exactly)
    input_encoded = pd.get_dummies(input_df, columns=["Soil_type", "Crop_type", "Fertilizer_name"], drop_first=True)

    # Align columns with training data (fill any missing columns with 0)
    model_columns = scaler.feature_names_in_
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale and predict
    input_scaled = scaler.transform(input_encoded)
    prediction   = model.predict(input_scaled)[0]

    # Display result
    st.success(f"### 🌱 Predicted Crop Yield: **{prediction:,.2f} units**")
    st.caption("This prediction is based on the entered field conditions using a trained Linear Regression model.")

import streamlit as st
import joblib
import numpy as np
import os

# 1. Page Setup
st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠")

# 2. Path Handling (Fixes the "Model Not Found" issue)
# This finds the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear_regression_model.joblib")

# 3. Model Loading Logic
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_model()

# 4. UI Header
st.title("🏠 California House Price Predictor")
st.markdown("---")

# If model is missing, show a clear message and stop
if model is None:
    st.error(f"❌ Model file not found!")
    st.code(f"Looking at: {MODEL_PATH}")
    st.info("Ensure 'linear_regression_model.joblib' is in the same GitHub folder as 'app.py'.")
    st.stop()

# 5. Input Form (All 8 Features)
st.subheader("Enter Property Details")
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (in $10k)", value=3.5, help="Example: 3.5 = $35,000")
    house_age = st.number_input("House Age", value=20.0)
    ave_rooms = st.number_input("Average Rooms", value=5.0)
    ave_bedrms = st.number_input("Average Bedrooms", value=1.0)

with col2:
    population = st.number_input("Population of Block", value=1000.0)
    ave_occup = st.number_input("Average Occupancy", value=3.0)
    lat = st.number_input("Latitude", value=34.0)
    lon = st.number_input("Longitude", value=-118.0)

# 6. Prediction Logic
if st.button("Predict House Value", type="primary", use_container_width=True):
    # Prepare input array (Must match the 8 features used during training)
    features = np.array([[
        med_inc, house_age, ave_rooms, ave_bedrms, 
        population, ave_occup, lat, lon
    ]])
    
    prediction = model.predict(features)
    
    # Scale output: California dataset target is in $100,000s
    predicted_value = prediction[0] * 100000
    
    st.markdown("---")
    st.success(f"### Predicted Market Value: ${predicted_value:,.2f}")
    st.caption("Result based on Experiment 1: Linear Regression Model.")

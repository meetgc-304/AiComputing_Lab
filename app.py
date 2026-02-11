import streamlit as st
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="CA House Price Predictor", layout="centered")

# 2. Load the model
# Ensure 'linear_regression_model.joblib' is in the same folder as this script
try:
    model = joblib.load('/Users/mac/Desktop/ONE/Sem 8/Ai Computing/Lab/exp-1/linear_regression_model.joblib')
except:
    st.error("Model file not found! Please place 'linear_regression_model.joblib' in the current directory.")

# 3. UI Elements
st.title("🏠 California House Price Predictor")
st.markdown("Enter the features below to predict the median house value.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Median Income (in $10k)", value=3.5, step=0.1)
    house_age = st.number_input("House Age", value=20, step=1)
    ave_rooms = st.number_input("Average Rooms", value=5.2, step=0.1)
    ave_bedrms = st.number_input("Average Bedrooms", value=1.0, step=0.1)

with col2:
    population = st.number_input("Population", value=1000, step=10)
    ave_occup = st.number_input("Average Occupancy", value=3.0, step=0.1)
    lat = st.number_input("Latitude", value=34.0, step=0.01)
    lon = st.number_input("Longitude", value=-118.0, step=0.01)

# 4. Prediction Logic
if st.button("Calculate Prediction", type="primary"):
    # Arrange features in the exact order the model expects
    features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, 
                          population, ave_occup, lat, lon]])
    
    prediction = model.predict(features)
    
    # The California dataset target is in units of $100,000
    final_price = prediction[0] * 100000
    
    st.divider()
    st.header(f"Estimated Value: ${final_price:,.2f}")
    st.info("Note: This prediction is based on the Linear Regression model from Experiment 1.")
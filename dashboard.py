import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("models/house_model.pkl")

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 Advanced House Price Prediction Dashboard")

st.markdown("### Enter Property Details")

# Inputs
area = st.slider("Area (sq ft)", 500, 4000, 2000)
bedrooms = st.slider("Bedrooms", 1, 6, 3)
bathrooms = st.slider("Bathrooms", 1, 4, 2)
floors = st.slider("Floors", 1, 3, 2)
age = st.slider("Age of Property", 0, 30, 5)
parking = st.selectbox("Parking Available", [0, 1])

# Prediction button
if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms, floors, age, parking]])
    prediction = model.predict(features)[0]

    st.success(f"💰 Estimated Price: ₹ {round(prediction, 2):,}")
    

    st.bar_chart({
    "Feature": ["Area", "Bedrooms", "Bathrooms"],
    "Impact": [area, bedrooms * 50000, bathrooms * 30000]
})

    # Insight section
    st.markdown("### 📊 Insights")
    st.write("- Larger area increases price")
    st.write("- More bedrooms & bathrooms increase value")
    st.write("- Older houses may reduce price")
    st.write("- Parking availability adds value")
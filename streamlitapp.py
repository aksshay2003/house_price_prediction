import streamlit as st
import pickle
import pandas as pd
import os

# Load model from file
model_path = os.path.join("models", "model_pickle_production.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # If model is wrapped in numpy array, extract the model
    if isinstance(model, (list, tuple)) or hasattr(model, "__len__") and not hasattr(model, "predict"):
        model = model[0]

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit App Title
st.title("🏠 House Price Prediction App")

# User Input
st.header("Enter House Details")

overall_qual = st.selectbox("Overall Quality (1-10)", options=list(range(1, 11)), index=6)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=100, max_value=10000, value=1500)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, value=800)
garage_cars = st.selectbox("Garage Cars Capacity", options=list(range(0, 5)), index=1)

# Optional: Location input if your model uses it, otherwise ignore
# location = st.selectbox("Location", options=["New York", "San Francisco", "Los Angeles", "Austin"])
# location_dict = {"New York": 0, "San Francisco": 1, "Los Angeles": 2, "Austin": 3}
# location_encoded = location_dict.get(location, 0)

# Prepare feature DataFrame (with exact column names expected by the model)
input_dict = {
    "Overall Qual": [overall_qual],
    "Gr Liv Area": [gr_liv_area],
    "Total Bsmt SF": [total_bsmt_sf],
    "Garage Cars": [garage_cars],
    # "location_encoded": [location_encoded],  # Include if your model needs this
}

features_df = pd.DataFrame(input_dict)

# Prediction
if st.button("Predict Price"):
    try:
        price = model.predict(features_df)[0]
        st.success(f"🏷️ Estimated House Price: $ {price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

import streamlit as st
import pickle

# Set page configuration
st.set_page_config(page_title="Car CO2 Emission Predictor", page_icon="🚗", layout="centered")

# Title of the app
st.title("🎈 Car CO2 Emission Predictor")

# Input fields for the features
f1 = st.number_input('Engine Size (L)', min_value=1, max_value=10, help="Enter the engine size in liters")
f2 = st.number_input('Cylinders', min_value=1, max_value=10, help="Enter the number of cylinders")
f3 = st.number_input('Fuel Consumption (L/100km)', min_value=1, max_value=100, help="Enter the fuel consumption in liters per 100km")

# Load the model
with open('model (1).pkl', 'rb') as file:
    model = pickle.load(file)

# Predict button
if st.button("Predict CO2 Emission"):
    res = model.predict([[f1, f2, f3]])  
    st.success(f"Predicted CO2 Emission: {res[0]:.2f} grams/km")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Made with ❤️ using Streamlit</p>', unsafe_allow_html=True)

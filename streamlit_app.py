import streamlit as st
import pickle
import numpy as np
import seaborn as sns

# Load the model from the pickle file
with open("model (1).pkl", "rb") as file:
    model = pickle.load(file)

# Custom colors using Seaborn palette
primary_color = sns.color_palette("coolwarm", as_cmap=True)

# Streamlit app design
st.set_page_config(page_title="Predictor App", page_icon="🔮", layout="centered")

# Header section with custom styles
st.markdown(
    """
    <style>
    .main-header {
        font-size:40px;
        font-weight:700;
        text-align:center;
        color:#4A148C;
    }
    .sub-header {
        font-size:20px;
        font-weight:500;
        text-align:center;
        color:#9C27B0;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<p class="main-header">Predictor App 🔮</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter values for the features to predict the output</p>', unsafe_allow_html=True)

# Create input fields organized in columns
col1, col2, col3 = st.columns(3)

with col1:
    feature1 = st.number_input("Feature 1", min_value=0.0, step=0.1, format="%.2f")
with col2:
    feature2 = st.number_input("Feature 2", min_value=0.0, step=0.1, format="%.2f")
with col3:
    feature3 = st.number_input("Feature 3", min_value=0.0, step=0.1, format="%.2f")

# Button to make a prediction
st.markdown("<hr>", unsafe_allow_html=True)

if st.button("Predict", help="Click to predict the output based on the features provided"):
    # Prepare the input as a numpy array
    input_data = np.array([[feature1, feature2, feature3]])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Display the prediction in a styled box
    st.markdown(
        f"""
        <div style="background-color:#EDE7F6;padding:20px;border-radius:10px;text-align:center;">
        <h2 style="color:#4A148C;">Prediction Result:</h2>
        <p style="font-size:24px;color:#1A237E;"><b>{prediction[0]}</b></p>
        </div>
        """, unsafe_allow_html=True
    )

# Footer section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Made with ❤️ using Streamlit</p>', unsafe_allow_html=True)

# Running the app
if __name__ == "__main__":
    st.run()






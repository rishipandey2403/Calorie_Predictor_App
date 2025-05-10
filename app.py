import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import zipfile
from PIL import Image  # For adding logo (optional)

# Unzip rfr.pkl if it doesn't exist
if not os.path.exists('rfr.pkl'):
    with zipfile.ZipFile('rfr.zip', 'r') as zip_ref:
        zip_ref.extractall()

# Set page config (first Streamlit command)
st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🔥",
    layout="wide"
)


# Load model and data with error handling
@st.cache_resource
def load_model():
    try:
        # For Render deployment - files must be in same directory
        model_path = os.path.join(os.path.dirname(__file__), 'rfr.pkl')
        data_path = os.path.join(os.path.dirname(__file__), 'X_train.csv')

        rfr = pickle.load(open(model_path, 'rb'))
        x_train = pd.read_csv(data_path)
        return rfr, x_train
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None


rfr, x_train = load_model()


# Prediction function
def predict_calories(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp):
    if rfr is None:
        return None
    try:
        features = np.array([[Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp]])
        prediction = rfr.predict(features).reshape(1, -1)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# --- Main App ---
st.title("🔥 Calorie Burn Prediction")
st.write("""
Predict how many calories you'll burn during exercise based on your metrics.
""")

# Add sidebar (optional)
with st.sidebar:
    st.header("🔥 CalorieBurn Predictor")
    st.markdown("""
    An intelligent app designed to estimate calories burned during physical activity using key physiological inputs like:
    - Gender
    - Age
    - Height
    - Weight
    - Duration
    - Heart Rate
    - Body Temperature

    **Powered by:**  
    ✔ Advanced Machine Learning  
    ✔ State-of-the-art regression algorithms  
    ✔ Data-driven precision

    *Helping users track fitness scientifically—whether optimizing workouts or managing health goals.*
    """)
    st.caption("Application is live on Render")

if x_train is not None:
    # Create input widgets
    col1, col2 = st.columns(2)

    with col1:
        gender_mapping = {0: 'Male', 1: 'Female'}
        Gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: gender_mapping[x])
        Age = st.number_input('Age',
                              min_value=10, max_value=100, value=30)
        Height = st.number_input('Height (cm)',
                                 min_value=100, max_value=250, value=170)
        Weight = st.number_input('Weight (kg)',
                                 min_value=30, max_value=200, value=70)

    with col2:
        Duration = st.number_input('Duration (minutes)',
                                   min_value=1, max_value=300, value=30)
        Heart_rate = st.number_input('Heart Rate (bpm)',
                                     min_value=50, max_value=200, value=120)
        Body_temp = st.number_input('Body Temperature (°C)',
                                    min_value=35.0, max_value=42.0, value=37.0, step=0.1)

    if st.button('Calculate Calories Burned'):
        result = predict_calories(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp)
        if result is not None:
            st.success(f"🔥 You burned approximately **{result[0]:.2f} calories**")
else:
    st.error("Failed to load required data. Please check the data files.")

# For Render deployment - add this to ensure proper port binding
if __name__ == '__main__':
    # When running locally, Streamlit automatically handles this
    # On Render, it will use the port from environment variable
    port = os.environ.get('PORT', 8501)
    #st.write(f"App running on port {port}")
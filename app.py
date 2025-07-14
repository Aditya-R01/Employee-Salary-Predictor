import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from streamlit_lottie import st_lottie

# ===== Load Model =====
model = joblib.load('rf_model_compressed.pkl')

# ===== Encodings =====
education_encoding = {"High School": 1, "Bachelor's":0, "Master's": 2, "PhD": 3}
location_encoding = {"Rural": 0, "Suburban": 1, "Urban": 2}
job_title_encoding = {
    'Clerk':0, 'Technician':9, 'Customer Support':1, 'Data Analyst':2,
    'Software Engineer':8, 'HR Manager':5,
    'Data Scientist':3, 'Product Manager':7, 'Director':4
}
gender_encoding = {'Female': 0, 'Male': 1}

# ===== Custom CSS =====
st.set_page_config(page_title="Salary Predictor 💼", page_icon="💸", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
    }
    .main {
        background: linear-gradient(145deg, #e0e0e0, #ffffff);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .css-1d391kg { background-color: #fff0;}
    </style>
""", unsafe_allow_html=True)

# ===== Lottie Animation Loader =====
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
lottie_ai = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")

# ===== Header =====
st_lottie(lottie_ai, height=200, key="ai")
st.title("💼 AI-Powered Salary Predictor")
st.markdown("Enter employee details below to predict **realistic salary** based on market standards.")

# ===== Input UI =====
with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox("📚 Education Level", list(education_encoding.keys()))
        experience = st.slider("👔 Years of Experience", 0, 40, 1)
        job_title = st.selectbox("💼 Job Title", list(job_title_encoding.keys()))
    
    with col2:
        location = st.selectbox("📍 Work Location", list(location_encoding.keys()))
        age = st.slider("🎂 Age", 18, 65, 30)
        gender = st.selectbox("👤 Gender", list(gender_encoding.keys()))

    submitted = st.form_submit_button("🚀 Predict Salary")

    if submitted:
        # Encode input
        input_vector = np.array([[
            education_encoding[education],
            experience,
            location_encoding[location],
            job_title_encoding[job_title],
            age,
            gender_encoding[gender]
        ]])
        input_df = pd.DataFrame(input_vector, columns=['education_level', 'experience', 'location', 'job_title', 'age', 'gender'])

        # Predict
        salary = model.predict(input_df)[0]
        st.success(f"🤑 Estimated Monthly Salary: ₹{salary:,.2f}")

# ===== Footer =====
st.markdown("""
<hr>
<div style="text-align:center">
    <strong>Made with ❤️ by Aditya Raj</strong><br>
    ECE Undergrad @ Birla Institute of Technology, Mesra<br><br>
    <a href="https://www.linkedin.com/in/adityaraj-bit/" target="_blank">🔗 LinkedIn</a> |
    <a href="https://www.instagram.com/adityar_a_j_?igsh=MTZicm1qejZmMWg4MQ==/" target="_blank">📸 Instagram</a>
</div>
""", unsafe_allow_html=True)

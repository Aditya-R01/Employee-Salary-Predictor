import streamlit as st
import pandas as pd
import pickle
import numpy as np


# Load trained model
model = pickle.load(open("lr_model.pkl", "rb"))

# ===== Encodings =====
education_encoding = {
    "High School": 0,
    "Bachelor's":1,
    "Master's": 2,
    "PhD": 3
}

location_encoding = {
    "Rural": 0,
    "Suburban": 1,
    "Urban": 2
}


job_title_encoding = {
    'Clerk':0,
    'Technician':1,
    'Customer Support':2,
    'Data Analyst':3,
    'Software Engineer':4,
    'Marketing Executive':5,
    'HR Manager':6,
    'Data Scientist':7,
    'Product Manager':8,
    'Director':9
}

gender_encoding = {
    'Female': 0,
    'Male': 1
}

# ===== Streamlit UI =====
st.title("💼 Employee Salary Predictor")

education = st.selectbox("📚 Education Level", list(education_encoding.keys()))
experience = st.slider("👔 Years of Experience", 0, 40, 1)
location = st.selectbox("📍 Work Location", list(location_encoding.keys()))
job_title = st.selectbox("💼 Job Title", list(job_title_encoding.keys()))
age = st.slider("🎂 Age", 18, 65, 30)
gender = st.selectbox("👤 Gender", list(gender_encoding.keys()))

# Encode input
input_vector = np.array([[
    education_encoding[education],
    experience,
    location_encoding[location],
    job_title_encoding[job_title],
    age,
    gender_encoding[gender]
]])


# Define the feature names used during training
feature_names =['education_level', 'experience', 'location', 'job_title', 'age', 'gender']
input_df = pd.DataFrame(input_vector, columns=feature_names)


# Predict and display
if st.button("💰 Predict Salary"):
    salary = model.predict(input_df)[0]
    st.success(f"🤑 Estimated Salary: ₹{salary:,.2f}")


# ===== Footer =====
st.markdown("""
<hr style="border:1px solid #ddd">
<div style="text-align:center">
    <strong>Made with ❤️ by Aditya Raj</strong><br>
    ECE Undergrad @ Birla Institute of Technology, Mesra<br><br>
    <a href="https://www.linkedin.com/in/adityar-a-j/" target="_blank">🔗 LinkedIn</a> |
    <a href="https://www.instagram.com/adityar_a_j_/" target="_blank">📸 Instagram</a>
</div>
""", unsafe_allow_html=True)

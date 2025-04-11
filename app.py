# app.py
import streamlit as st
import requests

st.set_page_config(page_title="Stroke Risk Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Stroke Risk Predictor")
st.markdown("Enter the following health data to assess stroke risk.")

# Form inputs
with st.form("stroke_form"):
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.slider("Age", 1, 100, 30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, step=0.1)
    bmi = st.number_input("BMI", min_value=0.0, step=0.1)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Format data for request
    data = {
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": Residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=data)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"🩺 Prediction: {prediction}")
        else:
            st.error("Error from prediction API.")
    except Exception as e:
        st.error(f"API call failed: {e}")

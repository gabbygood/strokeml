# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and preprocessors
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# Define input schema
class StrokeInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.post("/predict")
def predict(data: StrokeInput):
    # Encode categorical values
    input_data = {
        'gender': encoders['gender'].transform([data.gender])[0],
        'age': data.age,
        'hypertension': data.hypertension,
        'heart_disease': data.heart_disease,
        'ever_married': encoders['ever_married'].transform([data.ever_married])[0],
        'work_type': encoders['work_type'].transform([data.work_type])[0],
        'Residence_type': encoders['Residence_type'].transform([data.Residence_type])[0],
        'avg_glucose_level': data.avg_glucose_level,
        'bmi': data.bmi,
        'smoking_status': encoders['smoking_status'].transform([data.smoking_status])[0],
    }

    # Arrange in correct order
    features = np.array([list(input_data.values())])
    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]
    result = "High Risk of Stroke" if prediction == 1 else "Low Risk of Stroke"
    return {"prediction": result}

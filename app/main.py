from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from typing import List
app = FastAPI()

model = joblib.load(
    "artifacts/model.pkl"
)

scaler = joblib.load(
    "artifacts/scaler.pkl"
)
feature_names = joblib.load(
    "artifacts/features.pkl"
)
class FraudInput(BaseModel):

    features: List[float]
@app.get("/")
@app.get("/model-info")
def model_info():

    return {

        "model": "XGBoost Fraud Detection",

        "features": len(feature_names),

        "scaler": "StandardScaler"
    }
def home():

    return {
        "message": "Fraud Detection API"
    }


@app.post("/predict")
def predict(data: FraudInput):

    try:

        if len(data.features) != 30:

            return {
                "error": "Expected 30 input features"
            }

        input_data = np.array(data.features).reshape(1, -1)

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)

        probability = model.predict_proba(
            scaled_data
        )[0][1]

        if prediction[0] == 1:

            result = "Fraudulent Transaction"

        else:

            result = "Legitimate Transaction"

        return {

            "prediction": result,

            "fraud_probability": float(
                probability
            )
        }

    except Exception as e:

        return {
            "error": str(e)
        }
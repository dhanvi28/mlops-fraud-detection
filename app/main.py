from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load(
    "artifacts/model.pkl"
)

@app.get("/")
def home():

    return {
        "message": "Fraud Detection API"
    }

@app.post("/predict")
def predict(data: list):

    prediction = model.predict(
        [np.array(data)]
    )

    return {
        "prediction": int(prediction[0])
    }
from fastapi import FastAPI
import joblib
import json
import pandas as pd

app = FastAPI(title="Heart Disease Prediction API")

MODEL_PATH = "models/heart_disease_pipeline.joblib"
FEATURES_PATH = "models/features.json"

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    input_df = pd.DataFrame([[data[f] for f in feature_names]], columns=feature_names)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }

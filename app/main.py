import time
import csv
from datetime import datetime
import os

from fastapi import FastAPI, Response
import joblib
import json
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="Heart Disease Prediction API")

MODEL_PATH = "models/heart_disease_pipeline.joblib"
FEATURES_PATH = "models/features.json"

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of prediction requests in seconds"
)

@app.get("/")
def read_root():
    return {"message": "Heart Disease Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    start = time.time()

    input_df = pd.DataFrame([[data[f] for f in feature_names]], columns=feature_names)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    latency = time.time() - start

    # Update Prometheus metrics
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(latency)

    # Log to CSV
    os.makedirs("logs", exist_ok=True)
    with open("logs/predictions.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            *[data[f] for f in feature_names],
            int(pred),
            float(prob),
            round(latency, 4)
        ])

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

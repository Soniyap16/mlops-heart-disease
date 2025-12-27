import joblib
import json
import pandas as pd
import os

MODEL_PATH = "models/heart_disease_pipeline.joblib"
FEATURES_PATH = "models/features.json"

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Model file not found"

def test_features_file_exists():
    assert os.path.exists(FEATURES_PATH), "Features file not found"

def test_model_prediction_shape():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    # create a dummy sample
    sample = {f: 0 for f in features}
    df = pd.DataFrame([sample])

    preds = model.predict(df)
    assert len(preds) == 1, "Prediction should return one value"

def test_prediction_value_range():
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)

    sample = {f: 0 for f in features}
    df = pd.DataFrame([sample])

    pred = model.predict(df)[0]
    assert pred in [0, 1], "Prediction must be binary (0 or 1)"

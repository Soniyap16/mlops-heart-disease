import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/heart_clean.csv"
MODEL_OUT = "models/heart_disease_pipeline.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("num", axis=1)
    y = df["num"]

    pipeline = Pipeline([
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_OUT)
    print(f"Model trained and saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()


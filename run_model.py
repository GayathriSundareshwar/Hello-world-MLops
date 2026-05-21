import argparse
import joblib
import os
import json
from pathlib import Path
import numpy as np



MODEL_PATH = Path("/Users/gayus/Drive_D_Work/MLOps Zero to Hero/Iris_Train/artifacts/model.pkl")

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)   # <-- correct binary loading

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Feature list as JSON string. Example: \"[5.1,3.5,1.4,0.2]\"")
    args = parser.parse_args()

    # Parse input
    try:
        features = json.loads(args.input)
    except json.JSONDecodeError:
        raise ValueError("Invalid input. Use JSON list, e.g. --input \"[5.1,3.5,1.4,0.2]\"")

    X = np.array(features).reshape(1, -1)

    model = load_model()
    pred = model.predict(X)

    print(json.dumps({"prediction": pred.tolist()}))

    flower_names=["setosa", "versicolor", "virginica"]
    print(f"Predicted class: {flower_names[pred[0]]}")

if __name__ == "__main__":
    main()
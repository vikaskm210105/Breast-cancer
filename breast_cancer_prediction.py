"""Make an educational prediction using the saved breast cancer classifier.
This model is for learning only and must not be used for medical diagnosis.
"""
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "breast_cancer_logistic_model.pkl"
SCALER_PATH = BASE_DIR / "breast_cancer_scaler.pkl"
DATA_PATH = BASE_DIR / "breast-cancer.csv"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Demo: predict the first row of measurements from the dataset.
# Replace this with a new, valid row containing the same 30 feature columns.
df = pd.read_csv(DATA_PATH)
features = df.drop(columns=["id", "diagnosis"])
sample = features.iloc[[0]]
scaled_sample = scaler.transform(sample)
prediction = int(model.predict(scaled_sample)[0])
probability = float(model.predict_proba(scaled_sample)[0][1])

label = "Malignant" if prediction == 1 else "Benign"
print(f"Educational model prediction: {label}")
print(f"Model-estimated malignant probability: {probability:.1%}")
print("WARNING: This is a learning demo, not a medical diagnostic tool.")

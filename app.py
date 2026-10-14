"""Streamlit demo for a breast-cancer classification model.
For educational use only; not for medical diagnosis or treatment decisions.
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "breast_cancer_logistic_model.pkl")
scaler = joblib.load(BASE_DIR / "breast_cancer_scaler.pkl")
reference = pd.read_csv(BASE_DIR / "breast-cancer.csv")
FEATURES = reference.drop(columns=["id", "diagnosis"]).columns.tolist()

st.set_page_config(page_title="Breast Cancer ML Demo", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classification — ML Demo")
st.warning("Educational demonstration only. Do not use this app for medical diagnosis or clinical decisions.")
st.write("Enter the 30 numerical measurements expected by the trained model. Defaults are dataset feature averages, not patient measurements.")

with st.form("prediction_form"):
    values = {}
    for start in range(0, len(FEATURES), 3):
        cols = st.columns(3)
        for col, feature in zip(cols, FEATURES[start:start+3]):
            default = float(reference[feature].mean())
            low = float(reference[feature].min())
            high = float(reference[feature].max())
            # Number input accepts values outside the observed range, which is useful for entering raw measurements.
            values[feature] = col.number_input(feature, value=default, format="%.6f", help=f"Observed dataset range: {low:.6g} to {high:.6g}")
    submitted = st.form_submit_button("Run model prediction")

if submitted:
    input_df = pd.DataFrame([values], columns=FEATURES)
    scaled = scaler.transform(input_df)
    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])
    if prediction == 1:
        st.subheader("Model output: Malignant")
    else:
        st.subheader("Model output: Benign")
    st.metric("Model-estimated malignant probability", f"{probability:.1%}")
    st.caption("This probability is a model output, not a clinically validated risk estimate. Please do not use it to make health decisions.")

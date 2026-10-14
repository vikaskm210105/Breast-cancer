"""
Breast Cancer Classification demo.
Educational use only; not for medical diagnosis or treatment decisions.
"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧪",
    layout="wide",
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_DIR / "breast_cancer_logistic_model.pkl")
    scaler = joblib.load(BASE_DIR / "breast_cancer_scaler.pkl")
    return model, scaler

@st.cache_data
def load_reference():
    # Find the expected filename first; fall back to a matching CSV filename.
    expected = BASE_DIR / "breast-cancer.csv"
    if expected.exists():
        csv_path = expected
    else:
        candidates = sorted(BASE_DIR.glob("breast-cancer*.csv"))
        if not candidates:
            raise FileNotFoundError(
                "Dataset CSV not found. Place breast-cancer.csv beside app.py."
            )
        csv_path = candidates[0]
    return pd.read_csv(csv_path)

try:
    model, scaler = load_artifacts()
    reference = load_reference()
except Exception as exc:
    st.error(f"Could not load project files: {exc}")
    st.info("Keep app.py, the two .pkl files, and breast-cancer.csv in the same folder.")
    st.stop()

if not {"id", "diagnosis"}.issubset(reference.columns):
    st.error("The CSV does not have the expected 'id' and 'diagnosis' columns.")
    st.stop()

FEATURES = reference.drop(columns=["id", "diagnosis"]).columns.tolist()
if len(FEATURES) != getattr(scaler, "n_features_in_", len(FEATURES)):
    st.error("The CSV feature count does not match the saved scaler.")
    st.stop()

st.title("Breast Cancer Classification — ML Demo")
st.warning(
    "Educational demonstration only. This model is not clinically validated "
    "and must not be used for diagnosis or medical decisions."
)
st.write(
    "Enter 30 numerical tumor measurements. Default values are dataset averages, "
    "not measurements from a real patient."
)

with st.form("prediction_form"):
    values = {}
    for start in range(0, len(FEATURES), 3):
        cols = st.columns(3)
        for col, feature in zip(cols, FEATURES[start:start + 3]):
            default = float(reference[feature].mean())
            low = float(reference[feature].min())
            high = float(reference[feature].max())
            values[feature] = col.number_input(
                feature,
                value=default,
                format="%.6f",
                help=f"Observed dataset range: {low:.6g} to {high:.6g}",
            )
    submitted = st.form_submit_button("Run model prediction", type="primary")

if submitted:
    input_df = pd.DataFrame([values], columns=FEATURES)
    scaled = scaler.transform(input_df)
    prediction = int(model.predict(scaled)[0])
    malignant_probability = float(model.predict_proba(scaled)[0][1])

    if prediction == 1:
        st.subheader("Model output: Malignant")
    else:
        st.subheader("Model output: Benign")

    st.metric("Model-estimated malignant probability", f"{malignant_probability:.1%}")
    st.caption(
        "This is a model output, not a clinically validated risk estimate. "
        "Do not use it to make health decisions."
    )

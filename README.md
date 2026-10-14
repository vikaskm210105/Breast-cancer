# Breast Cancer Prediction ML Project

## Files
- `breast-cancer.csv` — dataset used by the project.
- `Breast_Cancer_Prediction_Implementation.ipynb` — notebook for exploring, training, evaluating, and saving the model.
- `breast_cancer_logistic_model.pkl` — trained Logistic Regression model.
- `breast_cancer_scaler.pkl` — fitted feature scaler.
- `breast_cancer_prediction.py` — command-line demo prediction using the first CSV row.
- `app.py` — Streamlit interface for entering the 30 model features.
- `requirements.txt` — Python dependencies.

## Setup (Windows)
Open Command Prompt or a terminal in this folder, then run:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook
```

Run the prediction script:

```bash
python breast_cancer_prediction.py
```

Launch the app:

```bash
streamlit run app.py
```

The saved model/scaler were trained using an 80/20 stratified split and `random_state=42`. The notebook can retrain and regenerate the `.pkl` files.

**Important:** This is a learning project only. It is not clinically validated and must not be used for medical diagnosis or treatment decisions. Model probabilities are not clinical risk estimates.

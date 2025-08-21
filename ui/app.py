import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Disease Risk App", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Risk — Interactive Demo")
st.write(
    "Enter patient information to get a **real-time prediction** of heart disease risk."
)

FEATURE_DOC = {
    "categorical": {
        "cp": "Chest pain type (1-4)",
        "exang": "Exercise induced angina (1 = yes; 0 = no)",
        "thal": "Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)",
        "sex": "Sex (1 = male; 0 = female)",
        "ca": "Number of major vessels (0-3) colored by fluoroscopy",
        "fbs": "Fasting blood sugar > 120 mg/dl (1 = yes; 0 = no)",
        "restecg": "Resting electrocardiographic results (0-2)",
        "slope": "Slope of the peak exercise ST segment (0-2)",
    },
    "numerical": {
        "age": "Age in years",
        "trestbps": "Resting blood pressure (in mm Hg)",
        "chol": "Serum cholesterol (in mg/dl)",
        "oldpeak": "Oldpeak (depression induced by exercise relative to rest)",
        "thalach": "Maximum heart rate achieved",
    }
}

RAW_COLUMNS = list(FEATURE_DOC["numerical"].keys()) + list(FEATURE_DOC["categorical"].keys())

@st.cache_resource
def load_model():
    try:
        local_path = Path("models\\final_model.pkl")
        model = joblib.load(local_path)
        return model
    except Exception as e:
        return f"Error loading model: {e}"

st.sidebar.header("Patient Inputs")

def number_input(label, min_value, max_value, value, help_text):
    return st.number_input(label, min_value=min_value, max_value=max_value, value=value, step=1.0, help=help_text)

def int_selectbox(label, options, index, help_text):
    return st.selectbox(label, options=options, index=index, help=help_text)

# Numerical
age = number_input("Age", 1.0, 120.0, 55.0, FEATURE_DOC["numerical"]["age"])
trestbps = number_input("Resting BP (mm Hg)", 50.0, 260.0, 130.0, FEATURE_DOC["numerical"]["trestbps"])
chol = number_input("Serum Cholesterol (mg/dl)", 80.0, 700.0, 245.0, FEATURE_DOC["numerical"]["chol"])
oldpeak = st.slider("Oldpeak", 0.0, 10.0, 1.0, 0.1, help=FEATURE_DOC["numerical"]["oldpeak"])
thalach = number_input("Max Heart Rate Achieved", 60.0, 250.0, 150.0, FEATURE_DOC["numerical"]["thalach"])

# Categorical
cp = int_selectbox("Chest Pain Type (1-4)", options=[1,2,3,4], index=0, help_text=FEATURE_DOC["categorical"]["cp"])
exang = int_selectbox("Exercise Induced Angina (0/1)", options=[0,1], index=0, help_text=FEATURE_DOC["categorical"]["exang"])
thal = int_selectbox("Thalassemia (1,2,3)", options=[1,2,3], index=0, help_text=FEATURE_DOC["categorical"]["thal"])
sex = int_selectbox("Sex (0=female,1=male)", options=[0,1], index=1, help_text=FEATURE_DOC["categorical"]["sex"])
ca = int_selectbox("No. Major Vessels (0-3)", options=[0,1,2,3], index=0, help_text=FEATURE_DOC["categorical"]["ca"])
fbs = int_selectbox("Fasting Blood Sugar >120 (0/1)", options=[0,1], index=0, help_text=FEATURE_DOC["categorical"]["fbs"])
restecg = int_selectbox("Resting ECG (0-2)", options=[0,1,2], index=0, help_text=FEATURE_DOC["categorical"]["restecg"])
slope = int_selectbox("ST Slope (0-2)", options=[0,1,2], index=1, help_text=FEATURE_DOC["categorical"]["slope"])


row_dict = {
    "age": age,
    "trestbps": trestbps,
    "chol": chol,
    "oldpeak": oldpeak,
    "thalach": thalach,
    "cp": int(cp),
    "exang": int(exang),
    "thal": int(thal),
    "sex": int(sex),
    "ca": int(ca),
    "fbs": int(fbs),
    "restecg": int(restecg),
    "slope": int(slope),
}
model = load_model()

# Ensure feature names and order match model expectations
if not isinstance(model, str) and hasattr(model, "feature_names_in_"):
    expected_columns = list(model.feature_names_in_)
    row = pd.DataFrame([[row_dict[col] for col in expected_columns]], columns=expected_columns)
else:
    row = pd.DataFrame([row_dict], columns=RAW_COLUMNS)

if not isinstance(model, str):
    st.markdown("### Real-time Prediction")
    label = int(model.predict(row)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0,1]
        st.metric("Prediction (1 = disease)", label)
        st.metric("Probability", f"{proba:.3f}")
    else:
        st.metric("Prediction (1 = disease)", label)

    st.divider()
    st.markdown("#### Model Inputs")
    st.dataframe(row)
else:
    st.warning(model)

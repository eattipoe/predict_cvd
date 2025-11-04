# app_cardio_test.py
import streamlit as st
import joblib
import pandas as pd

# Load model bundle
bundle = joblib.load("cardio_test_stacked_model.pkl")
model = bundle["model"]
feature_names = bundle["feature_names"]
feature_defaults = bundle["feature_defaults"]

st.title("AI-Powered Cardiovascular Disease (CVD) Risk Predictor")

st.markdown(
    "Enter patient details below and click **Predict** to estimate the risk of "
    "cardiovascular disease (CVD)."
)

# --- MAIN INPUTS ---
age_years = st.number_input("Age (years)", min_value=18, max_value=100, value=50)

gender_label = st.selectbox("Gender", options=["Male", "Female"])
# Dataset convention: 1=female, 2=male
gender = 2 if gender_label == "Male" else 1

bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

ap_hi = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=260, value=130)
ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=85)

# NEW: cholesterol in mg/dL
cholesterol = st.number_input(
    "Total Cholesterol (mg/dL)",
    min_value=120.0,
    max_value=320.0,
    value=200.0,
    step=1.0,
)

# --- OPTIONAL / ADVANCED INPUTS ---
with st.expander("Advanced options (optional)"):
    gluc = st.number_input(
        "Glucose (same scale as training data)",
        min_value=0.0,
        max_value=10.0,
        value=float(feature_defaults.get("gluc", 1.0)),
        step=0.1,
    )
    smoke = st.selectbox(
        "Smoker (smoke)",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        index=int(feature_defaults.get("smoke", 0)),
    )
    alco = st.selectbox(
        "Alcohol intake (alco)",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        index=int(feature_defaults.get("alco", 0)),
    )
    active = st.selectbox(
        "Physically active (active)",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        index=int(feature_defaults.get("active", 1)),
    )

st.markdown(
    "> This tool is for decision support only and does **not** replace professional medical advice."
)

if st.button("Predict CVD Risk"):
    # Start from defaults
    row = dict(feature_defaults)

    # Override with user inputs
    row["age_years"] = age_years
    row["gender"] = gender
    row["bmi"] = bmi
    row["ap_hi"] = ap_hi
    row["ap_lo"] = ap_lo
    row["cholesterol"] = cholesterol  # mg/dL
    row["gluc"] = gluc
    row["smoke"] = smoke
    row["alco"] = alco
    row["active"] = active

    # Build dataframe in correct order
    X_new = pd.DataFrame([row])[feature_names]

    # Predict
    prob = model.predict_proba(X_new)[0, 1]
    pred = model.predict(X_new)[0]

    label = "CVD (High Risk)" if pred == 1 else "No CVD (Low Risk)"
    st.subheader(f"Prediction: {label}")
    st.write(f"Estimated CVD probability: **{prob:.3f}**")

    if pred == 1:
        st.warning(
            "The model suggests elevated CVD risk. "
            "Kindly consider further evaluation and guideline-based management."
        )
    else:
        st.info(
            "The model suggests low CVD risk. Continue routine monitoring and healthy lifestyle."
        )

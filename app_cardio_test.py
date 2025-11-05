# app_cvd_mmol.py

import streamlit as st
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

# ---------------- Helper functions ----------------

def bmi_category(bmi: float) -> str:
    """BMI categories (kg/m²)."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    elif bmi < 35.0:
        return "Obese (Class I)"
    else:
        return "Obese (Class II/III)"


def bp_category(sys: float, dia: float) -> str:
    """
    BP categories (flexible):

      Normal:     systolic <120 and diastolic <80
      Elevated:   120–139 and diastolic <90
      Hypertension: systolic ≥140 or diastolic ≥90
      Otherwise:  At-Risk (Borderline)
    """
    if sys < 120 and dia < 80:
        return "Normal"
    if sys >= 140 or dia >= 90:
        return "Hypertension"
    if 120 <= sys <= 139 and dia < 90:
        return "Elevated"
    return "At-Risk (Borderline)"


def cholesterol_category(chol: float) -> str:
    """
    Total cholesterol in mmol/L:

      Normal:    < 5.2
      Borderline: 5.2–6.2
      High:      ≥ 6.2
    """
    if chol < 5.2:
        return "Normal"
    elif chol < 6.2:
        return "Borderline / At-Risk"
    else:
        return "Abnormal / High Risk"


def glucose_category(glu: float) -> str:
    """
    Fasting glucose in mmol/L:

      Normal:     < 5.6
      Prediabetes: 5.6–6.9
      Diabetes:   ≥ 7.0
    """
    if glu < 5.6:
        return "Normal"
    elif glu < 7.0:
        return "Prediabetes"
    else:
        return "Diabetes"


def overall_risk_level(bmi_cat: str, bp_cat: str, chol_cat: str, glu_cat: str) -> str:
    """
    Flexible overall risk:
      - High when multiple high-risk factors (Hypertension, Diabetes,
        Abnormal cholesterol, Overweight/Obese) are present.
      - Moderate for single high-risk or several borderline.
      - Low when most values are normal.
    """
    high_flags = 0
    moderate_flags = 0

    # BMI
    if bmi_cat in ["Overweight", "Obese (Class I)", "Obese (Class II/III)"]:
        high_flags += 1

    # BP
    if bp_cat == "Hypertension":
        high_flags += 1
    elif bp_cat in ["Elevated", "At-Risk (Borderline)"]:
        moderate_flags += 1

    # Cholesterol
    if chol_cat == "Abnormal / High Risk":
        high_flags += 1
    elif chol_cat == "Borderline / At-Risk":
        moderate_flags += 1

    # Glucose
    if glu_cat == "Diabetes":
        high_flags += 1
    elif glu_cat == "Prediabetes":
        moderate_flags += 1

    if high_flags >= 2:
        return "High Overall Risk"
    if high_flags == 1 and moderate_flags >= 1:
        return "High Overall Risk"
    if high_flags == 1 or moderate_flags >= 2:
        return "Moderate Overall Risk"
    return "Low Overall Risk"


# ---------------- DB SETUP ----------------

@st.cache_resource
def get_connection():
    # New DB name to avoid old schema conflicts
    conn = sqlite3.connect("cvd_predictions_v2.db", check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age_years REAL,
            gender INTEGER,
            gender_label TEXT,
            bmi REAL,
            systolic_bp REAL,
            diastolic_bp REAL,
            cholesterol_mmol REAL,
            glucose_mmol REAL,
            pred INTEGER,
            prob REAL,
            category_hint TEXT
        )
        """
    )
    return conn


conn = get_connection()


def save_prediction_to_db(
    age_years,
    gender,
    gender_label,
    bmi,
    systolic_bp,
    diastolic_bp,
    cholesterol,
    glucose,
    pred,
    prob,
    category_hint,
):
    ts = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO predictions (
            timestamp, age_years, gender, gender_label, bmi,
            systolic_bp, diastolic_bp, cholesterol_mmol, glucose_mmol,
            pred, prob, category_hint
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts,
            float(age_years),
            int(gender),
            gender_label,
            float(bmi),
            float(systolic_bp),
            float(diastolic_bp),
            float(cholesterol),
            float(glucose),
            int(pred),
            float(prob),
            category_hint,
        ),
    )
    conn.commit()


# ---------------- MODEL LOAD ----------------

bundle = joblib.load("cardio_mmol_stacked_model.pkl")
model = bundle["model"]
feature_names = bundle["feature_names"]
feature_defaults = bundle["feature_defaults"]

# ---------------- UI ----------------

st.title("AI-Powered Cardiovascular Disease (CVD) Risk Predictor")

st.markdown(
    "Enter patient details below and click **Predict** to estimate the risk of "
    "cardiovascular disease (CVD)."
)

# --- MAIN INPUTS ---

age_years = st.number_input("Age (years)", min_value=18, max_value=100, value=50)

gender_label = st.selectbox("Gender", options=["Male", "Female"])
# 1 = female, 2 = male
gender = 2 if gender_label == "Male" else 1

bmi = st.number_input(
    "Body Mass Index (BMI, kg/m²)",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.1,
)

systolic_bp = st.number_input(
    "Systolic BP (mmHg)",
    min_value=80.0,
    max_value=260.0,
    value=130.0,
    step=1.0,
)

diastolic_bp = st.number_input(
    "Diastolic BP (mmHg)",
    min_value=40.0,
    max_value=150.0,
    value=85.0,
    step=1.0,
)

cholesterol = st.number_input(
    "Total Cholesterol (mmol/L)",
    min_value=3.0,
    max_value=9.0,
    value=5.0,
    step=0.1,
)

glucose = st.number_input(
    "Fasting Glucose (mmol/L)",
    min_value=3.0,
    max_value=15.0,
    value=5.0,
    step=0.1,
)

st.markdown(
    "> This tool is for decision support only and does **not** replace professional medical advice."
)

# ---------------- PREDICTION ----------------

if st.button("Predict CVD Risk"):
    # Base row from training medians
    row = dict(feature_defaults)

    # Override with user inputs
    row["age_years"] = age_years
    row["gender"] = gender
    row["bmi"] = bmi
    row["systolic_bp"] = systolic_bp
    row["diastolic_bp"] = diastolic_bp
    row["cholesterol"] = cholesterol
    row["glucose"] = glucose

    X_new = pd.DataFrame([row])[feature_names]

    # Model probability
    prob_model = float(model.predict_proba(X_new)[0, 1])

    # Detailed categories
    bmi_cat = bmi_category(bmi)
    bp_cat = bp_category(systolic_bp, diastolic_bp)
    chol_cat = cholesterol_category(cholesterol)
    glu_cat = glucose_category(glucose)
    overall = overall_risk_level(bmi_cat, bp_cat, chol_cat, glu_cat)

    # ----- Reconcile prediction with overall risk -----
    # Overall drives final label to keep "Prediction" consistent
    if overall == "High Overall Risk":
        final_pred = 1
    elif overall == "Low Overall Risk":
        final_pred = 0
    else:  # Moderate Overall Risk
        final_pred = 1 if prob_model >= 0.5 else 0

    label = "CVD (High Risk)" if final_pred == 1 else "No CVD (Low Risk)"

    st.subheader(f"Prediction: {label}")
    st.write(f"Estimated model CVD probability: **{prob_model:.3f}**")

    category_hint = (
        f"BMI: {bmi_cat}; "
        f"BP: {bp_cat}; "
        f"Cholesterol: {chol_cat}; "
        f"Glucose: {glu_cat}; "
        f"Overall: {overall}"
    )

    st.markdown(f"**Risk Profile:** {category_hint}")

    # ---- Save to DB using final_pred ----
    save_prediction_to_db(
        age_years=age_years,
        gender=gender,
        gender_label=gender_label,
        bmi=bmi,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        cholesterol=cholesterol,
        glucose=glucose,
        pred=final_pred,
        prob=prob_model,
        category_hint=category_hint,
    )

    st.success("Record saved to database ✅")

# ---------------- VIEW RECENT RECORDS ----------------

with st.expander("Show recent saved predictions"):
    df_log = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 25", conn
    )
    st.dataframe(df_log)


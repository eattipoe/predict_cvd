# app_cardio_test.py
import streamlit as st
import joblib
import pandas as pd
import sqlite3
from datetime import datetime

# ---------------- DB SETUP ----------------
@st.cache_resource
def get_connection():
    conn = sqlite3.connect("cvd_predictions.db", check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            age_years REAL,
            gender INTEGER,
            gender_label TEXT,
            bmi REAL,
            ap_hi REAL,
            ap_lo REAL,
            cholesterol REAL,
            gluc REAL,
            smoke INTEGER,
            alco INTEGER,
            active INTEGER,
            pred INTEGER,
            prob REAL
        )
        """
    )
    return conn

conn = get_connection()

def save_prediction_to_db(
    age_years, gender, gender_label, bmi, ap_hi, ap_lo,
    cholesterol, gluc, smoke, alco, active, pred, prob
):
    ts = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO predictions (
            timestamp, age_years, gender, gender_label, bmi, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active, pred, prob
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ts,
            float(age_years),
            int(gender),
            gender_label,
            float(bmi),
            float(ap_hi),
            float(ap_lo),
            float(cholesterol),
            float(gluc),
            int(smoke),
            int(alco),
            int(active),
            int(pred),
            float(prob),
        ),
    )
    conn.commit()

# ---------------- MODEL LOAD ----------------
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
    prob = float(model.predict_proba(X_new)[0, 1])
    pred = int(model.predict(X_new)[0])

    label = "CVD (High Risk)" if pred == 1 else "No CVD (Low Risk)"
    st.subheader(f"Prediction: {label}")
    st.write(f"Estimated CVD probability: **{prob:.3f}**")

    # ---------- SAVE TO DB ----------
    save_prediction_to_db(
        age_years=age_years,
        gender=gender,
        gender_label=gender_label,
        bmi=bmi,
        ap_hi=ap_hi,
        ap_lo=ap_lo,
        cholesterol=cholesterol,
        gluc=gluc,
        smoke=smoke,
        alco=alco,
        active=active,
        pred=pred,
        prob=prob,
    )

    st.success("Record saved to database ✅")

# Optional: view last few records
with st.expander("Show recent saved predictions"):
    df_log = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT 10", conn
    )
    st.dataframe(df_log)

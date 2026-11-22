"""ML Assignment demo app: PM2.5 next-day forecast (regression) + stroke risk (classification)."""
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "models"

# same column order the models were trained on
PM_FEATURES = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone",
               "aerosol_optical_depth", "dust", "uv_index", "month",
               "pm25_lag1", "pm25_lag2", "pm25_roll7"]
STROKE_COLUMNS = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type",
                  "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]


@st.cache_resource
def load_models():
    return (joblib.load(MODEL_DIR / "ridge_pm25.joblib"),
            joblib.load(MODEL_DIR / "stroke_model.joblib"))


st.set_page_config(page_title="ML Assignment Demo", page_icon="📊", layout="centered")
pm_model, stroke_model = load_models()

st.title("ML Assignment Demo")
st.caption("Regression: Addis Ababa PM2.5 forecast · Classification: stroke risk")

tab_pm, tab_stroke = st.tabs(["🌫️ PM2.5 forecast (regression)", "🩺 Stroke risk (classification)"])

# ---------------------------------------------------------------- regression
with tab_pm:
    st.subheader("Predict tomorrow's PM2.5 in Addis Ababa")
    st.write("Defaults are the last real week in the dataset (12–18 Feb 2026). "
             "Model: Ridge regression, test RMSE 3.93 µg/m³.")

    date = st.date_input("Today's date", dt.date(2026, 2, 18))

    st.markdown("**PM2.5 of the last 7 days (µg/m³)**, oldest → today")
    last7_default = [15.77, 16.58, 14.43, 15.57, 17.92, 17.89, 19.31]
    labels = ["6 days ago", "5 days ago", "4 days ago", "3 days ago", "2 days ago", "Yesterday", "Today"]
    cols = st.columns(7)
    last7 = [c.number_input(lbl, 0.0, 200.0, val, 0.1, key=f"pm{i}")
             for i, (c, lbl, val) in enumerate(zip(cols, labels, last7_default))]

    st.markdown("**Today's other air-quality values**")
    c1, c2, c3 = st.columns(3)
    pm10 = c1.number_input("PM10 (µg/m³)", 0.0, 500.0, 49.18, 0.1)
    co = c2.number_input("Carbon monoxide (µg/m³)", 0.0, 10000.0, 396.5, 1.0)
    no2 = c3.number_input("Nitrogen dioxide (µg/m³)", 0.0, 100.0, 3.97, 0.1)
    so2 = c1.number_input("Sulphur dioxide (µg/m³)", 0.0, 100.0, 1.36, 0.1)
    ozone = c2.number_input("Ozone (µg/m³)", 0.0, 300.0, 69.92, 0.1)
    aod = c3.number_input("Aerosol optical depth", 0.0, 5.0, 0.235, 0.001, format="%.3f")
    dust = c1.number_input("Dust (µg/m³)", 0.0, 1000.0, 71.83, 0.1)
    uv = c2.number_input("UV index", 0.0, 15.0, 1.91, 0.01)

    if st.button("Forecast tomorrow", type="primary"):
        row = pd.DataFrame([{
            "pm10": pm10, "pm2_5": last7[-1], "carbon_monoxide": co, "nitrogen_dioxide": no2,
            "sulphur_dioxide": so2, "ozone": ozone, "aerosol_optical_depth": aod, "dust": dust,
            "uv_index": uv, "month": date.month,
            "pm25_lag1": last7[-2], "pm25_lag2": last7[-3], "pm25_roll7": float(np.mean(last7)),
        }])[PM_FEATURES]
        pred = float(pm_model.predict(row)[0])
        st.metric("Tomorrow's PM2.5", f"{pred:.1f} µg/m³", f"{pred - last7[-1]:+.1f} vs today")
        st.caption("Typical error is about ±4 µg/m³. WHO 24-hour guideline: 15 µg/m³.")

# ------------------------------------------------------------ classification
with tab_stroke:
    st.subheader("Estimate stroke risk for a patient")
    st.write("Model: Logistic Regression tuned for **recall**. It catches 82% of strokes on the test set, "
             "but it also raises many false alarms (precision 0.14), so treat it as a screening aid.")

    c1, c2 = st.columns(2)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    age = c2.number_input("Age", 0.0, 100.0, 67.0, 1.0)
    hypertension = c1.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = c2.selectbox("Heart disease", ["No", "Yes"])
    ever_married = c1.selectbox("Ever married", ["Yes", "No"])
    work_type = c2.selectbox("Work type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence = c1.selectbox("Residence type", ["Urban", "Rural"])
    smoking = c2.selectbox("Smoking status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    glucose = c1.number_input("Average glucose level (mg/dL)", 40.0, 300.0, 105.0, 0.5)
    bmi_known = c2.checkbox("BMI known", value=True)
    bmi = c2.number_input("BMI", 10.0, 100.0, 28.0, 0.1, disabled=not bmi_known)

    if st.button("Assess risk", type="primary"):
        patient = pd.DataFrame([{
            "gender": gender, "age": age,
            "hypertension": int(hypertension == "Yes"), "heart_disease": int(heart_disease == "Yes"),
            "ever_married": ever_married, "work_type": work_type, "Residence_type": residence,
            "avg_glucose_level": glucose, "bmi": bmi if bmi_known else np.nan,  # missing BMI is imputed
            "smoking_status": smoking,
        }])[STROKE_COLUMNS]
        flagged = int(stroke_model.predict(patient)[0])
        score = float(stroke_model.predict_proba(patient)[0, 1])

        if flagged:
            st.error(f"⚠️ Flagged as **at risk** (risk score {score:.2f}). Recommend a medical follow-up.")
        else:
            st.success(f"✅ Not flagged (risk score {score:.2f}).")
        st.caption("The risk score is a model score from class-balanced training, not a calibrated probability. "
                   "Educational project only, not medical advice.")

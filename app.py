"""
app.py - Customer Churn Prediction (Streamlit)

Run with:  streamlit run app.py

Requires the artifacts produced by train_and_save_model.py to be in the
same folder: rf_model.pkl, scaler.pkl, model_columns.pkl, total_charges_median.pkl
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---------- Load saved artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    total_charges_median = joblib.load("total_charges_median.pkl")
    return model, scaler, model_columns, total_charges_median

try:
    model, scaler, model_columns, total_charges_median = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_and_save_model.py` first "
        "(with WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder), "
        "then restart this app."
    )
    st.stop()

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details to estimate their probability of churning.")

# ---------- Input form ----------
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0)
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=float(total_charges_median), step=1.0
        )

    submitted = st.form_submit_button("Predict Churn")

# ---------- Prediction ----------
if submitted:
    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([raw_input])

    # One-hot encode the same way training data was encoded
    categorical_cols = input_df.select_dtypes(include=["object"]).columns
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # Align columns with training-time columns (fill any missing dummy cols with 0)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale numeric columns using the SAME fitted scaler from training
    input_encoded[NUM_COLS] = scaler.transform(input_encoded[NUM_COLS])

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ Likely to Churn — probability: {probability:.1%}")
    else:
        st.success(f"✅ Likely to Stay — churn probability: {probability:.1%}")

    st.progress(min(max(probability, 0.0), 1.0))

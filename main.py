import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Risk Predictor")
st.write("Input customer details below to evaluate real-time churn probability and risk level.")

# 1. Load trained artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# 2. Input Form
with st.form("customer_input_form"):
    st.subheader("Customer Demographics & Usage")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=6)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=300.0, value=75.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=450.0)
    
    st.subheader("Account & Behavioral Indicators")
    support_tickets = st.slider("Support Tickets Raised", 0, 10, 2)
    payment_delay = st.slider("Payment Delay (Days)", 0, 30, 5)
    contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    
    submit_button = st.form_submit_button("Predict Churn Risk")

# 3. Prediction Logic
if submit_button:
    # Map contract selection to binary features
    contract_one_year = 1 if contract_type == "One Year" else 0
    contract_two_year = 1 if contract_type == "Two Year" else 0

    raw_data = {
        'Age': age,
        'Tenure_Months': tenure,
        'Monthly_Charges': monthly_charges,
        'Total_Charges': total_charges,
        'Support_Tickets': support_tickets,
        'Payment_Delay_Days': payment_delay,
        'Contract_Type_One Year': contract_one_year,
        'Contract_Type_Two Year': contract_two_year
    }

    # Align columns to scaler inputs
    input_df = pd.DataFrame([raw_data])[scaler.feature_names_in_]
    scaled_data = scaler.transform(input_df)
    
    # Calculate churn probability
    churn_prob = model.predict_proba(scaled_data)[0, 1]

    st.markdown("---")
    st.subheader("Results")
    st.metric("Predicted Churn Probability", f"{churn_prob:.1%}")

    # Risk Tier Display
    if churn_prob >= 0.65:
        st.error("🔴 **High Risk Customer**: Recommended for immediate proactive retention outreach!")
    elif churn_prob >= 0.35:
        st.warning("🟡 **Medium Risk Customer**: Monitor usage and contract renewal timeline.")
    else:
        st.success("🟢 **Low Risk Customer**: Account is stable.")
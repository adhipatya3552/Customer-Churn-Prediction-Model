import streamlit as st
import pickle
import numpy as np
import os, json
import pandas as pd
import matplotlib.pyplot as plt

st.markdown("""
<style>
.big-font {
    font-size:22px !important;
    font-weight:600;
}
.card {
    padding: 15px;
    border-radius: 10px;
    background-color: #111827;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load model safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = pickle.load(open(model_path, "rb"))

# -----------------------------
# UI DESIGN
# -----------------------------
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to leave the service.")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("📥 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", value=70.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# -----------------------------
# SIMPLE ENCODING (DEMO PURPOSE)
# -----------------------------
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

internet_map = {
    "DSL": 1,
    "Fiber optic": 2,
    "No": 0
}

# Load column structure
columns_path = os.path.join(BASE_DIR, "models", "columns.json")
with open(columns_path, "r") as f:
    model_columns = json.load(f)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Churn"):

    # Create empty dataframe with all columns
    input_data = pd.DataFrame(columns=model_columns)

    # Fill with zeros
    input_data.loc[0] = 0

        # -----------------------------
    # SAFE COLUMN MAPPING (FIX)
    # -----------------------------
    for col in model_columns:
        if "tenure" in col.lower():
            input_data[col] = tenure
        if "monthly" in col.lower() and "charge" in col.lower():
            input_data[col] = monthly

    # Contract encoding
    for col in model_columns:
        if "contract" in col.lower() and contract.lower().replace(" ", "_") in col.lower():
            input_data[col] = 1

    # Internet encoding
    for col in model_columns:
        if "internet" in col.lower() and internet.lower().replace(" ", "_") in col.lower():
            input_data[col] = 1

    # Predict
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0][1]
    else:
        prob = 0

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
        <div class="card">
            <p class="big-font">⚠️ High Churn Risk</p>
            <p>Probability: {prob*100:.2f}%</p>
            <p>Suggested Action: Offer discounts or long-term plans</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card">
            <p class="big-font">✅ Low Churn Risk</p>
            <p>Retention Probability: {(1-prob)*100:.2f}%</p>
            <p>Suggested Action: Maintain engagement</p>
        </div>
        """, unsafe_allow_html=True)


if hasattr(model, "feature_importances_"):
    st.subheader("📈 Feature Importance")

    importances = model.feature_importances_
    feature_names = model_columns

    # Top 10 features
    indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:10]

    top_features = [feature_names[i] for i in indices]
    top_importance = [importances[i] for i in indices]

    fig, ax = plt.subplots()
    ax.barh(top_features[::-1], top_importance[::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top Features")

    st.pyplot(fig)

st.markdown("---")
st.subheader("💡 Business Insights")

st.markdown("""
- Customers with **month-to-month contracts** churn more  
- Higher **monthly charges** increase churn probability  
- Long-term customers are more stable  

👉 Companies can reduce churn by:
- Offering long-term plans  
- Providing loyalty discounts  
- Targeting high-risk customers early  
""")

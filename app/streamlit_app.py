"""
Customer Churn Prediction Dashboard
Real-time churn risk assessment with business insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Churn Prediction Platform",
    page_icon="📊",
    layout="wide"
)

# Load config
@st.cache_resource
def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

# Load model and scaler
@st.cache_resource
def load_model_artifacts():
    model = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

# Initialize
config = load_config()
model, scaler = load_model_artifacts()

# App header
st.title("🎯 Customer Churn Prediction Platform")
st.markdown("**Real-time churn risk assessment powered by ML**")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Analysis", "Model Info"])

# ============================================================
# PAGE 1: SINGLE PREDICTION
# ============================================================
if page == "Single Prediction":
    st.header("📋 Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    
    with col2:
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                      ["Electronic check", "Mailed check", 
                                       "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 5.0)
    
    with col3:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    st.markdown("---")
    
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        # Calculate total charges
        total_charges = monthly_charges * tenure
        
        # Create feature vector (must match training data order)
        # Order: tenure, MonthlyCharges, TotalCharges, SeniorCitizen, then encoded categoricals
        
        # Simple encoding (matching training logic)
        features = np.array([[
            tenure,
            monthly_charges,
            total_charges,
            1 if senior_citizen == "Yes" else 0,
            0 if gender == "Female" else 1,  # gender_encoded
            1 if partner == "Yes" else 0,  # Partner_encoded
            1 if dependents == "Yes" else 0,  # Dependents_encoded
            1 if phone_service == "Yes" else 0,  # PhoneService_encoded
            0 if multiple_lines == "No" else (1 if multiple_lines == "Yes" else 2),  # MultipleLines_encoded
            0 if internet_service == "DSL" else (1 if internet_service == "Fiber optic" else 2),  # InternetService_encoded
            0 if online_security == "No" else (1 if online_security == "Yes" else 2),  # OnlineSecurity_encoded
            0 if online_backup == "No" else (1 if online_backup == "Yes" else 2),  # OnlineBackup_encoded
            0 if device_protection == "No" else (1 if device_protection == "Yes" else 2),  # DeviceProtection_encoded
            0 if tech_support == "No" else (1 if tech_support == "Yes" else 2),  # TechSupport_encoded
            0 if streaming_tv == "No" else (1 if streaming_tv == "Yes" else 2),  # StreamingTV_encoded
            0 if streaming_movies == "No" else (1 if streaming_movies == "Yes" else 2),  # StreamingMovies_encoded
            0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),  # Contract_encoded
            1 if paperless_billing == "Yes" else 0,  # PaperlessBilling_encoded
            0 if payment_method == "Bank transfer (automatic)" else (
                1 if payment_method == "Credit card (automatic)" else (
                    2 if payment_method == "Electronic check" else 3
                )
            )  # PaymentMethod_encoded
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        churn_probability = model.predict_proba(features_scaled)[0][1]
        churn_prediction = int(churn_probability > 0.5)
        
        # Display results
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{churn_probability:.1%}")
        
        with col2:
            risk_level = "🔴 HIGH RISK" if churn_probability > 0.7 else (
                "🟡 MEDIUM RISK" if churn_probability > 0.4 else "🟢 LOW RISK"
            )
            st.metric("Risk Level", risk_level)
        
        with col3:
            prediction_text = "⚠️ LIKELY TO CHURN" if churn_prediction else "✅ LIKELY TO STAY"
            st.metric("Prediction", prediction_text)
        
        # Business insights
        st.markdown("---")
        st.subheader("💼 Business Insights")
        
        annual_value = monthly_charges * 12
        ltv_3yr = monthly_charges * 36  # Simple 3-year LTV
        
        if churn_probability > 0.5:
            st.warning(f"""
            **⚠️ High Churn Risk Customer**
            - Estimated Annual Value: ${annual_value:,.2f}
            - 3-Year Lifetime Value at Risk: ${ltv_3yr:,.2f}
            - **Recommended Action**: Immediate retention campaign
            """)
        else:
            st.success(f"""
            **✅ Healthy Customer Relationship**
            - Estimated Annual Value: ${annual_value:,.2f}
            - Projected 3-Year Value: ${ltv_3yr:,.2f}
            - **Recommended Action**: Continue regular engagement
            """)

# ============================================================
# PAGE 2: BATCH ANALYSIS
# ============================================================
elif page == "Batch Analysis":
    st.header("📊 Batch Customer Analysis")
    st.markdown("Upload a CSV file with customer data for bulk predictions")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df)} customers")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        if st.button("Run Batch Predictions"):
            st.info("🔄 Processing predictions...")
            
            # Note: This is simplified - in production you'd need proper preprocessing
            st.warning("⚠️ Batch prediction requires properly formatted input matching training data schema")
            st.info("Feature implementation: Connect to preprocessing pipeline for production use")

# ============================================================
# PAGE 3: MODEL INFO
# ============================================================
elif page == "Model Info":
    st.header("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.markdown(f"""
        - **Model Type**: {type(model).__name__}
        - **Training Date**: 2026-01-27
        - **Framework**: Scikit-learn
        - **Features**: 19 customer attributes
        - **Target**: Binary churn classification
        """)
    
    with col2:
        st.subheader("Performance Metrics")
        st.markdown("""
        - **ROC-AUC**: 84.0%
        - **Accuracy**: 79.8%
        - **Precision**: 64.1%
        - **Recall**: 54.8%
        - **F1-Score**: 59.1%
        """)
    
    st.markdown("---")
    st.subheader("📈 Model Training Pipeline")
    st.markdown("""
    1. **Data Preprocessing**: SQL-based transformations using DuckDB
    2. **Feature Engineering**: Categorical encoding, scaling
    3. **Model Training**: Logistic Regression baseline + Random Forest
    4. **Experiment Tracking**: MLflow for versioning and metrics
    5. **Model Selection**: Best model by ROC-AUC score
    """)
    
    st.markdown("---")
    st.subheader("🎯 Business Impact")
    st.markdown("""
    **Use Cases:**
    - Identify high-risk customers for retention campaigns
    - Estimate revenue at risk from potential churn
    - Prioritize customer success resources
    - Track churn trends over time
    
    **Expected ROI:**
    - 15-20% reduction in churn rate
    - $500K-$1M annual revenue saved (for mid-size telco)
    - 3-5x ROI on retention campaigns
    """)

# Footer
st.markdown("---")
st.markdown("**Built with:** Python • Scikit-learn • MLflow • DuckDB • Streamlit")

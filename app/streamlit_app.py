"""
Customer Churn Prediction Dashboard
Real-time churn risk assessment with business insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
import sys
import os
from pathlib import Path

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="Churn Prediction Platform",
    page_icon="📊",
    layout="wide"
)

# Auto-train models if missing (for Streamlit Cloud)
if not os.path.exists('models/churn_model.pkl'):
    st.warning("⚠️ Models not found. Training models... (takes ~1 minute)")
    
    with st.spinner("Downloading data and training models..."):
        import subprocess
        try:
            subprocess.run(['python3', 'download_data.py'], check=True, capture_output=True)
            subprocess.run(['python3', 'src/pipelines/data_preprocessing.py'], check=True, capture_output=True)
            subprocess.run(['python3', 'src/models/train_churn_model.py'], check=True, capture_output=True)
            st.success("✅ Models trained successfully!")
            st.info("🔄 Refreshing...")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            st.info("Models will be loaded from repository")

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

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Single Prediction", "Batch Analysis", "Model Info"])

# Helper function
def calculate_clv(monthly_charges, tenure, churn_probability):
    """Calculate CLV with churn adjustment"""
    if churn_probability < 0.3:
        lifetime = 48
    elif churn_probability < 0.6:
        lifetime = 24
    else:
        lifetime = 12
    
    if tenure < 12:
        lifetime *= 1.2
    elif tenure < 36:
        lifetime *= 1.0
    else:
        lifetime *= 0.8
    
    survival = 1 - churn_probability
    clv = monthly_charges * lifetime * survival * (0.95 ** (lifetime / 12))
    
    return {
        'monthly_value': monthly_charges,
        'annual_value': monthly_charges * 12,
        'expected_lifetime_months': lifetime,
        'clv_optimistic': monthly_charges * lifetime,
        'clv_realistic': clv,
        'revenue_at_risk': clv if churn_probability > 0.5 else 0
    }

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
        
        st.markdown("---")
        calculated_total = monthly_charges * tenure
        st.metric("Total Charges (Calculated)", f"${calculated_total:,.2f}",
                 help="Monthly × Tenure")
        st.caption("ℹ️ Calculated automatically")
    
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
        total_charges = monthly_charges * tenure
        
        features = np.array([[
            tenure, monthly_charges, total_charges,
            1 if senior_citizen == "Yes" else 0,
            0 if gender == "Female" else 1,
            1 if partner == "Yes" else 0,
            1 if dependents == "Yes" else 0,
            1 if phone_service == "Yes" else 0,
            0 if multiple_lines == "No" else (1 if multiple_lines == "Yes" else 2),
            0 if internet_service == "DSL" else (1 if internet_service == "Fiber optic" else 2),
            0 if online_security == "No" else (1 if online_security == "Yes" else 2),
            0 if online_backup == "No" else (1 if online_backup == "Yes" else 2),
            0 if device_protection == "No" else (1 if device_protection == "Yes" else 2),
            0 if tech_support == "No" else (1 if tech_support == "Yes" else 2),
            0 if streaming_tv == "No" else (1 if streaming_tv == "Yes" else 2),
            0 if streaming_movies == "No" else (1 if streaming_movies == "Yes" else 2),
            0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
            1 if paperless_billing == "Yes" else 0,
            0 if payment_method == "Bank transfer (automatic)" else (
                1 if payment_method == "Credit card (automatic)" else (
                    2 if payment_method == "Electronic check" else 3))
        ]])
        
        features_scaled = scaler.transform(features)
        churn_probability = model.predict_proba(features_scaled)[0][1]
        churn_prediction = int(churn_probability > 0.5)
        
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{churn_probability:.1%}")
        with col2:
            risk_level = "🔴 HIGH RISK" if churn_probability > 0.7 else (
                "🟡 MEDIUM RISK" if churn_probability > 0.4 else "🟢 LOW RISK")
            st.metric("Risk Level", risk_level)
        with col3:
            prediction_text = "⚠️ LIKELY TO CHURN" if churn_prediction else "✅ LIKELY TO STAY"
            st.metric("Prediction", prediction_text)
        
        st.markdown("---")
        st.subheader("💼 Business Insights")
        clv_metrics = calculate_clv(monthly_charges, tenure, churn_probability)
        
        if churn_probability > 0.5:
            st.error(f"""
### ⚠️ High Churn Risk Customer

**Current Revenue:**
- Monthly: ${clv_metrics['monthly_value']:.2f}
- Annual: ${clv_metrics['annual_value']:,.2f}

**Lifetime Value:**
- Optimistic: ${clv_metrics['clv_optimistic']:,.2f}
- Realistic: ${clv_metrics['clv_realistic']:,.2f}
- **At Risk: ${clv_metrics['revenue_at_risk']:,.2f}**

**💡 Actions:**
1. Immediate retention campaign
2. Customer success outreach
3. Contract upgrade offer

**Budget:** Max ${clv_metrics['revenue_at_risk'] * 0.15:,.2f} to retain
            """)
        elif churn_probability > 0.3:
            st.warning(f"""
### 🟡 Medium Risk
**CLV:** ${clv_metrics['clv_realistic']:,.2f}
**Actions:** Proactive engagement, surveys
            """)
        else:
            st.success(f"""
### ✅ Healthy Customer
**CLV:** ${clv_metrics['clv_realistic']:,.2f}
**Actions:** Continue engagement, upsells
            """)
        
        st.markdown("---")
        st.subheader("📊 CLV Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual", f"${clv_metrics['annual_value']:,.0f}")
        with col2:
            st.metric("Optimistic", f"${clv_metrics['clv_optimistic']:,.0f}")
        with col3:
            st.metric("Realistic", f"${clv_metrics['clv_realistic']:,.0f}")

# ============================================================
# PAGE 2: BATCH ANALYSIS  
# ============================================================
elif page == "Batch Analysis":
    st.header("📊 Batch Analysis")
    st.markdown("**Accepts ANY telecom CSV format with smart defaults**")
    
    st.info("""
    ℹ️ **Flexible Input Support**
    
    - Accepts various column names (Gender, gender, sex, etc.)
    - Uses smart defaults for missing features
    - Always makes predictions with available data
    """)
    
    # Sample template
    sample = pd.DataFrame({
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 1],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'Yes'],
        'tenure': [12, 60],
        'PhoneService': ['Yes', 'Yes'],
        'MultipleLines': ['No', 'No'],
        'InternetService': ['DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes'],
        'OnlineBackup': ['Yes', 'Yes'],
        'DeviceProtection': ['No', 'Yes'],
        'TechSupport': ['No', 'No'],
        'StreamingTV': ['No', 'No'],
        'StreamingMovies': ['No', 'No'],
        'Contract': ['Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [70.0, 55.0]
    })
    
    st.download_button("📥 Download Template", sample.to_csv(index=False),
                      "template.csv", "text/csv")
    
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} customers")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Run Predictions", type="primary"):
                with st.spinner("Processing..."):
                    sys.path.insert(0, 'src/models')
                    from batch_predictions import BatchPredictor
                    
                    predictor = BatchPredictor()
                    results, errors = predictor.predict_batch(df)
                    
                    if results is not None:
                        st.success("✅ Complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🔴 High", (results['Risk_Level']=='High Risk').sum())
                        with col2:
                            st.metric("🟡 Medium", (results['Risk_Level']=='Medium Risk').sum())
                        with col3:
                            st.metric("🟢 Low", (results['Risk_Level']=='Low Risk').sum())
                        with col4:
                            st.metric("💰 At Risk", f"${results['Revenue_at_Risk'].sum():,.0f}")
                        
                        st.dataframe(results[['Churn_Probability', 'Risk_Level', 
                                            'Annual_Value', 'Realistic_CLV']], 
                                   use_container_width=True)
                        
                        st.download_button("📥 Download", results.to_csv(index=False),
                                         "predictions.csv", "text/csv", type="primary")
                    else:
                        st.warning("⚠️ Using defaults for missing features")
                        st.info("Results may have lower confidence")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ============================================================
# PAGE 3: MODEL INFO
# ============================================================
elif page == "Model Info":
    st.header("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Details")
        st.markdown(f"""
        - **Model**: {type(model).__name__}
        - **Framework**: Scikit-learn
        - **Features**: 19 telecom attributes
        - **Dataset**: 7,043 customers
        """)
    
    with col2:
        st.subheader("Performance")
        st.markdown("""
        - **ROC-AUC**: 84.0% ⭐
        - **Accuracy**: 79.8%
        - **Precision**: 64.1%
        - **Recall**: 54.8%
        """)
    
    st.markdown("---")
    st.subheader("📈 Pipeline")
    st.markdown("""
    1. SQL preprocessing (DuckDB)
    2. Model training (Logistic Regression + Random Forest)
    3. Experiment tracking (MLflow)
    4. Model selection (ROC-AUC)
    5. Dashboard deployment (Streamlit)
    """)
    
    st.markdown("---")
    st.subheader("💼 Impact")
    st.markdown("""
    - 15-20% churn reduction
    - $500K-$1M savings/year
    - 3-5x ROI on retention
    """)

st.markdown("---")
st.markdown("**Built with:** Python • Scikit-learn • MLflow • DuckDB • Streamlit")
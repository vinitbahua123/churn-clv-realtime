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
# HELPER FUNCTION: Calculate CLV
# ============================================================
def calculate_clv(monthly_charges, tenure, churn_probability):
    """Calculate Customer Lifetime Value with churn adjustment"""
    if churn_probability < 0.3:
        expected_lifetime_months = 48
    elif churn_probability < 0.6:
        expected_lifetime_months = 24
    else:
        expected_lifetime_months = 12
    
    if tenure < 12:
        tenure_multiplier = 1.2
    elif tenure < 36:
        tenure_multiplier = 1.0
    else:
        tenure_multiplier = 0.8
    
    adjusted_lifetime = expected_lifetime_months * tenure_multiplier
    survival_probability = 1 - churn_probability
    clv_base = monthly_charges * adjusted_lifetime * survival_probability
    discount_factor = 0.95 ** (adjusted_lifetime / 12)
    clv_adjusted = clv_base * discount_factor
    
    return {
        'monthly_value': monthly_charges,
        'annual_value': monthly_charges * 12,
        'expected_lifetime_months': adjusted_lifetime,
        'clv_optimistic': monthly_charges * adjusted_lifetime,
        'clv_realistic': clv_adjusted,
        'revenue_at_risk': clv_adjusted if churn_probability > 0.5 else 0
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
        
        # Show calculated total charges
        st.markdown("---")
        calculated_total = monthly_charges * tenure
        st.metric(
            label="Total Charges (Calculated)", 
            value=f"${calculated_total:,.2f}",
            help="Automatically calculated as Monthly Charges × Tenure"
        )
        st.caption("ℹ️ This is calculated automatically")
    
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
        
        # Create feature vector
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
        
        # Scale and predict
        features_scaled = scaler.transform(features)
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
                "🟡 MEDIUM RISK" if churn_probability > 0.4 else "🟢 LOW RISK")
            st.metric("Risk Level", risk_level)
        with col3:
            prediction_text = "⚠️ LIKELY TO CHURN" if churn_prediction else "✅ LIKELY TO STAY"
            st.metric("Prediction", prediction_text)
        
        # Business insights
        st.markdown("---")
        st.subheader("💼 Business Insights")
        clv_metrics = calculate_clv(monthly_charges, tenure, churn_probability)
        
        if churn_probability > 0.5:
            st.error(f"""
            ### ⚠️ High Churn Risk Customer
            
            **Current Revenue:**
            - Monthly Value: ${clv_metrics['monthly_value']:.2f}
            - Annual Value: ${clv_metrics['annual_value']:,.2f}
            
            **Lifetime Value Analysis:**
            - Optimistic LTV (if retained): ${clv_metrics['clv_optimistic']:,.2f}
            - Realistic LTV (with churn risk): ${clv_metrics['clv_realistic']:,.2f}
            - **Revenue at Risk: ${clv_metrics['revenue_at_risk']:,.2f}**
            
            **Context:**
            - Customer tenure: {tenure} months
            - Churn probability: {churn_probability:.1%}
            - Expected remaining lifetime: {clv_metrics['expected_lifetime_months']:.0f} months
            
            **💡 Recommended Actions:**
            1. **Immediate retention campaign** - Offer targeted incentives
            2. **Customer success outreach** - Identify pain points
            3. **Loyalty program enrollment** - Lock in with benefits
            4. **Contract upgrade offer** - Month-to-month → 1-year contract
            
            **Retention Investment Budget:**
            - Max spend to retain: ${clv_metrics['revenue_at_risk'] * 0.15:,.2f} (15% of CLV)
            - Break-even retention cost: ${clv_metrics['revenue_at_risk'] * 0.30:,.2f} (30% of CLV)
            """)
        elif churn_probability > 0.3:
            st.warning(f"""
            ### 🟡 Medium Churn Risk Customer
            
            **Current Revenue:**
            - Monthly Value: ${clv_metrics['monthly_value']:.2f}
            - Annual Value: ${clv_metrics['annual_value']:,.2f}
            
            **Lifetime Value Analysis:**
            - Optimistic LTV: ${clv_metrics['clv_optimistic']:,.2f}
            - Realistic LTV: ${clv_metrics['clv_realistic']:,.2f}
            
            **💡 Recommended Actions:**
            1. **Proactive engagement** - Regular check-ins
            2. **Add-on services** - Increase stickiness
            3. **Survey feedback** - Understand satisfaction
            4. **Loyalty rewards** - Build commitment
            """)
        else:
            st.success(f"""
            ### ✅ Healthy Customer Relationship
            
            **Current Revenue:**
            - Monthly Value: ${clv_metrics['monthly_value']:.2f}
            - Annual Value: ${clv_metrics['annual_value']:,.2f}
            
            **Lifetime Value Analysis:**
            - Expected LTV: ${clv_metrics['clv_realistic']:,.2f}
            - Growth Potential: ${clv_metrics['clv_optimistic'] - clv_metrics['clv_realistic']:,.2f}
            
            **💡 Recommended Actions:**
            1. **Continue regular engagement**
            2. **Upsell opportunities**
            3. **Referral program**
            4. **Annual loyalty bonus**
            """)
        
        # CLV Comparison
        st.markdown("---")
        st.subheader("📊 CLV Comparison")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual Value", f"${clv_metrics['annual_value']:,.0f}",
                     help="Guaranteed revenue for next 12 months")
        with col2:
            st.metric("Optimistic LTV", f"${clv_metrics['clv_optimistic']:,.0f}",
                     delta=f"{clv_metrics['expected_lifetime_months']:.0f} mo lifetime",
                     help="Maximum potential if customer doesn't churn")
        with col3:
            st.metric("Realistic LTV", f"${clv_metrics['clv_realistic']:,.0f}",
                     delta=f"-{churn_probability:.0%} risk adjusted",
                     delta_color="inverse",
                     help="Expected value accounting for churn probability")

# ============================================================
# PAGE 2: BATCH ANALYSIS
# ============================================================
elif page == "Batch Analysis":
    st.header("📊 Batch Customer Analysis")
    st.markdown("Upload CSV file with customer data for bulk predictions")
    
    # Create sample template
    sample_data = {
        'gender': ['Female', 'Male', 'Female'],
        'SeniorCitizen': [0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes'],
        'tenure': [12, 34, 62],
        'PhoneService': ['Yes', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'No', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL'],
        'OnlineSecurity': ['No', 'No', 'Yes'],
        'OnlineBackup': ['Yes', 'No', 'Yes'],
        'DeviceProtection': ['No', 'No', 'Yes'],
        'TechSupport': ['No', 'No', 'No'],
        'StreamingTV': ['No', 'No', 'No'],
        'StreamingMovies': ['No', 'No', 'No'],
        'Contract': ['Month-to-month', 'Month-to-month', 'One year'],
        'PaperlessBilling': ['Yes', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Electronic check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [29.85, 70.70, 56.15]
    }
    sample_df = pd.DataFrame(sample_data)
    sample_csv = sample_df.to_csv(index=False)
    
    # Download button
    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_csv,
        file_name="churn_prediction_template.csv",
        mime="text/csv",
        help="Download this template to see the required format"
    )
    
    # Format guide
    with st.expander("📋 Required Columns & Accepted Values"):
        st.markdown("""
        **Demographics:**
        - `gender`: Female, Male (or F, M)
        - `SeniorCitizen`: 0, 1
        - `Partner`: Yes, No
        - `Dependents`: Yes, No
        
        **Account:**
        - `tenure`: Number (0-72 months)
        - `Contract`: Month-to-month, One year, Two year
        - `PaperlessBilling`: Yes, No
        - `PaymentMethod`: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)
        - `MonthlyCharges`: Number (e.g., 70.50)
        
        **Services:**
        - `PhoneService`: Yes, No
        - `MultipleLines`: Yes, No, No phone service
        - `InternetService`: DSL, Fiber optic, No
        - `OnlineSecurity`: Yes, No, No internet service
        - `OnlineBackup`: Yes, No, No internet service
        - `DeviceProtection`: Yes, No, No internet service
        - `TechSupport`: Yes, No, No internet service
        - `StreamingTV`: Yes, No, No internet service
        - `StreamingMovies`: Yes, No, No internet service
        
        **Note:** Column names accept variations (e.g., `gender` or `Gender`)
        """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} customers")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Run Batch Predictions", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Import batch predictor
                    sys.path.insert(0, 'src/models')
                    from batch_predictions import BatchPredictor
                    
                    # Run predictions
                    predictor = BatchPredictor()
                    results, errors = predictor.predict_batch(df)
                    
                    if results is not None:
                        # SUCCESS
                        st.success("✅ Predictions Complete!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            high_risk = (results['Risk_Level'] == 'High Risk').sum()
                            st.metric("🔴 High Risk", high_risk)
                        with col2:
                            medium_risk = (results['Risk_Level'] == 'Medium Risk').sum()
                            st.metric("🟡 Medium Risk", medium_risk)
                        with col3:
                            low_risk = (results['Risk_Level'] == 'Low Risk').sum()
                            st.metric("🟢 Low Risk", low_risk)
                        with col4:
                            if 'Revenue_at_Risk' in results.columns:
                                total_risk = results['Revenue_at_Risk'].sum()
                                st.metric("💰 Revenue at Risk", f"${total_risk:,.0f}")
                        
                        # Results table
                        st.subheader("📊 Prediction Results")
                        display_cols = ['Churn_Probability', 'Risk_Level', 'Churn_Status']
                        if 'Annual_Value' in results.columns:
                            display_cols.append('Annual_Value')
                        if 'Realistic_CLV' in results.columns:
                            display_cols.append('Realistic_CLV')
                        if 'Revenue_at_Risk' in results.columns:
                            display_cols.append('Revenue_at_Risk')
                        
                        st.dataframe(results[display_cols], use_container_width=True)
                        
                        # Download button
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Complete Results",
                            data=csv,
                            file_name="churn_predictions_results.csv",
                            mime="text/csv",
                            type="primary"
                        )
                    else:
                        # ERRORS
                        st.error("❌ Could not process file")
                        for error in errors:
                            st.error(error)
                        st.info("💡 Download the sample template above to see the correct format")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please ensure your file is a valid CSV")

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
    2. **Feature Engineering**: Categorical encoding, scaling, TotalCharges calculation
    3. **Model Training**: Logistic Regression baseline + Random Forest
    4. **Experiment Tracking**: MLflow for versioning and metrics
    5. **Model Selection**: Best model by ROC-AUC (Logistic Regression - 84.0%)
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
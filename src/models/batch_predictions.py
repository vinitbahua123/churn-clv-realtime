"""
Batch Churn Predictions - Fixed Version
"""

import pandas as pd
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchPredictor:
    """Batch prediction with proper encoding"""
    
    COLUMN_MAPPINGS = {
        'gender': ['gender', 'Gender', 'sex', 'Sex'],
        'SeniorCitizen': ['SeniorCitizen', 'senior_citizen', 'Senior'],
        'Partner': ['Partner', 'partner', 'Has_Partner'],
        'Dependents': ['Dependents', 'dependents', 'Children'],
        'tenure': ['tenure', 'Tenure', 'months', 'Months'],
        'PhoneService': ['PhoneService', 'phone_service', 'Phone'],
        'MultipleLines': ['MultipleLines', 'multiple_lines'],
        'InternetService': ['InternetService', 'internet_service', 'Internet'],
        'OnlineSecurity': ['OnlineSecurity', 'online_security'],
        'OnlineBackup': ['OnlineBackup', 'online_backup'],
        'DeviceProtection': ['DeviceProtection', 'device_protection'],
        'TechSupport': ['TechSupport', 'tech_support'],
        'StreamingTV': ['StreamingTV', 'streaming_tv'],
        'StreamingMovies': ['StreamingMovies', 'streaming_movies'],
        'Contract': ['Contract', 'contract', 'ContractType'],
        'PaperlessBilling': ['PaperlessBilling', 'paperless_billing'],
        'PaymentMethod': ['PaymentMethod', 'payment_method'],
        'MonthlyCharges': ['MonthlyCharges', 'monthly_charges', 'Monthly']
    }
    
    def __init__(self):
        self.model = joblib.load('models/churn_model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        logger.info("✅ Model and scaler loaded")
    
    def normalize_columns(self, df):
        """Map column name variations"""
        normalized_df = pd.DataFrame()
        missing_columns = []
        
        for standard_name, variations in self.COLUMN_MAPPINGS.items():
            found = False
            for variation in variations:
                if variation in df.columns:
                    normalized_df[standard_name] = df[variation]
                    found = True
                    break
            if not found:
                missing_columns.append(standard_name)
        
        return normalized_df, missing_columns
    
    def validate_input(self, df):
        """Validate uploaded dataframe"""
        if len(df) == 0:
            return False, ["❌ File is empty"]
        
        normalized_df, missing_cols = self.normalize_columns(df)
        
        if missing_cols:
            errors = [f"❌ Missing columns: {', '.join(missing_cols)}"]
            return False, errors
        
        return True, []
    
    def preprocess_batch(self, df):
        """Preprocess to match training format"""
        logger.info(f"📥 Processing {len(df)} customers...")
        
        # Normalize columns
        df_processed, _ = self.normalize_columns(df)
        
        # Handle SeniorCitizen
        if df_processed['SeniorCitizen'].max() > 2:
            df_processed['SeniorCitizen'] = (df_processed['SeniorCitizen'] >= 65).astype(int)
        
        # Clean gender
        df_processed['gender'] = df_processed['gender'].str.strip().str.title()
        df_processed['gender'] = df_processed['gender'].replace({'M': 'Male', 'F': 'Female'})
        
        # Calculate TotalCharges
        df_processed['TotalCharges'] = df_processed['MonthlyCharges'] * df_processed['tenure']
        
        # Encode categorical with _encoded suffix
        df_processed['gender_encoded'] = df_processed['gender'].map({'Female': 0, 'Male': 1}).fillna(0).astype(int)
        df_processed['Partner_encoded'] = df_processed['Partner'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
        df_processed['Dependents_encoded'] = df_processed['Dependents'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
        df_processed['PhoneService_encoded'] = df_processed['PhoneService'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
        df_processed['MultipleLines_encoded'] = df_processed['MultipleLines'].map({
            'No': 0, 'No phone service': 1, 'Yes': 2
        }).fillna(0).astype(int)
        df_processed['InternetService_encoded'] = df_processed['InternetService'].map({
            'DSL': 0, 'Fiber optic': 1, 'No': 2
        }).fillna(0).astype(int)
        df_processed['OnlineSecurity_encoded'] = df_processed['OnlineSecurity'].map({
            'No': 0, 'No internet service': 1, 'Yes': 2
        }).fillna(0).astype(int)
        df_processed['OnlineBackup_encoded'] = df_processed['OnlineBackup'].map({
            'No': 0, 'No internet service': 1, 'Yes': 2
        }).fillna(0).astype(int)
        df_processed['DeviceProtection_encoded'] = df_processed['DeviceProtection'].map({
            'No': 0, 'No internet service': 1, 'Yes': 2
        }).fillna(0).astype(int)
        df_processed['TechSupport_encoded'] = df_processed['TechSupport'].map({
            'No': 0, 'No internet service': 1, 'Yes': 2
        }).fillna(0).astype(int)
        df_processed['StreamingTV_encoded'] = df_processed['StreamingTV'].map({
            'No': 0, 'No internet service': 1, 'Yes': 2
        }).fillna(0).astype(int)
        df_processed['StreamingMovies_encoded'] = df_processed['StreamingMovies'].map({
            'No': 0, 'No internet service': 1, 'Yes': 2
        }).fillna(0).astype(int)
        df_processed['Contract_encoded'] = df_processed['Contract'].map({
            'Month-to-month': 0, 'One year': 1, 'Two year': 2
        }).fillna(0).astype(int)
        df_processed['PaperlessBilling_encoded'] = df_processed['PaperlessBilling'].map({
            'No': 0, 'Yes': 1
        }).fillna(0).astype(int)
        df_processed['PaymentMethod_encoded'] = df_processed['PaymentMethod'].map({
            'Bank transfer (automatic)': 0,
            'Credit card (automatic)': 1,
            'Electronic check': 2,
            'Mailed check': 3
        }).fillna(0).astype(int)
        
        # Select only encoded columns in correct order
        final_cols = [
            'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
            'gender_encoded', 'Partner_encoded', 'Dependents_encoded',
            'PhoneService_encoded', 'MultipleLines_encoded',
            'InternetService_encoded', 'OnlineSecurity_encoded',
            'OnlineBackup_encoded', 'DeviceProtection_encoded',
            'TechSupport_encoded', 'StreamingTV_encoded',
            'StreamingMovies_encoded', 'Contract_encoded',
            'PaperlessBilling_encoded', 'PaymentMethod_encoded'
        ]
        
        df_final = df_processed[final_cols]
        logger.info("✅ Preprocessing complete")
        return df_final
    
    def calculate_clv(self, monthly_charges, tenure, churn_probability):
        """Calculate CLV"""
        if churn_probability < 0.3:
            lifetime = 48
        elif churn_probability < 0.6:
            lifetime = 24
        else:
            lifetime = 12
        
        if tenure < 12:
            lifetime *= 1.2
        elif tenure >= 36:
            lifetime *= 0.8
        
        survival = 1 - churn_probability
        clv = monthly_charges * lifetime * survival * (0.95 ** (lifetime / 12))
        return clv
    
    def predict_batch(self, df):
        """Generate predictions"""
        is_valid, errors = self.validate_input(df)
        if not is_valid:
            return None, errors
        
        try:
            df_original = df.copy()
            X = self.preprocess_batch(df)
            X_scaled = self.scaler.transform(X)
            
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            results = df_original.copy()
            results['Churn_Probability'] = probabilities
            results['Churn_Prediction'] = predictions.astype(int)
            results['Churn_Status'] = results['Churn_Prediction'].map({
                0: 'Will Stay', 1: 'Will Churn'
            })
            results['Risk_Level'] = results['Churn_Probability'].apply(
                lambda x: 'High Risk' if x > 0.7 else (
                    'Medium Risk' if x > 0.4 else 'Low Risk')
            )
            
            # Find column names
            monthly_col = None
            tenure_col = None
            for col in ['MonthlyCharges', 'monthly_charges', 'Monthly']:
                if col in results.columns:
                    monthly_col = col
                    break
            for col in ['tenure', 'Tenure', 'months']:
                if col in results.columns:
                    tenure_col = col
                    break
            
            if monthly_col and tenure_col:
                results['Annual_Value'] = results[monthly_col] * 12
                clv_values = [
                    self.calculate_clv(row[monthly_col], row[tenure_col], row['Churn_Probability'])
                    for _, row in results.iterrows()
                ]
                results['Realistic_CLV'] = clv_values
                results['Revenue_at_Risk'] = results.apply(
                    lambda x: x['Realistic_CLV'] if x['Churn_Prediction'] == 1 else 0,
                    axis=1
                )
            
            logger.info(f"✅ Predictions for {len(results)} customers")
            logger.info(f"   High: {(results['Risk_Level']=='High Risk').sum()}")
            logger.info(f"   Medium: {(results['Risk_Level']=='Medium Risk').sum()}")
            logger.info(f"   Low: {(results['Risk_Level']=='Low Risk').sum()}")
            
            if 'Revenue_at_Risk' in results.columns:
                logger.info(f"   Revenue at Risk: ${results['Revenue_at_Risk'].sum():,.2f}")
            
            return results, []
            
        except Exception as e:
            logger.error(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, [f"❌ Error: {str(e)}"]

if __name__ == "__main__":
    print("🧪 Testing...")
    predictor = BatchPredictor()
    try:
        df = pd.read_csv('sample_batch_input.csv')
        results, errors = predictor.predict_batch(df)
        if results is not None:
            print("✅ SUCCESS!")
            print(results[['Churn_Probability', 'Risk_Level']].head())
    except:
        print("⚠️ No sample file")
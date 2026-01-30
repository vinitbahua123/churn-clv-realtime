"""
Smart Batch Predictor - Accepts ANY telecom data format
Uses intelligent defaults for missing features
"""

import pandas as pd
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchPredictor:
    """Smart predictor that accepts any telecom data with defaults"""
    
    def __init__(self):
        self.model = joblib.load('models/churn_model.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        logger.info("✅ Model loaded")
    
    def find_column(self, df, possible_names):
        """Find a column from list of possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        # Case-insensitive
        for name in possible_names:
            for col in df.columns:
                if col.lower() == name.lower():
                    return col
        return None
    
    def extract_feature(self, df, feature_name, possible_cols, default, mapping=None):
        """Extract feature with smart defaults"""
        col = self.find_column(df, possible_cols)
        
        if col is None:
            # Use default for all rows
            return pd.Series([default] * len(df))
        
        if mapping:
            # Categorical - map values
            result = df[col].astype(str).str.strip().str.title()
            result = result.map(mapping)
            return result.fillna(default).astype(int)
        else:
            # Numeric - convert
            return pd.to_numeric(df[col], errors='coerce').fillna(default)
    
    def preprocess_batch(self, df):
        """Preprocess ANY telecom data format"""
        logger.info(f"📥 Processing {len(df)} customers...")
        
        processed = pd.DataFrame()
        
        # Extract each feature with defaults
        processed['tenure'] = self.extract_feature(
            df, 'tenure', 
            ['tenure', 'Tenure', 'Account length', 'Account_length', 'months'],
            default=12
        )
        
        processed['MonthlyCharges'] = self.extract_feature(
            df, 'MonthlyCharges',
            ['MonthlyCharges', 'monthly_charges', 'Total day charge', 'bill_amount'],
            default=50.0
        )
        
        processed['TotalCharges'] = processed['MonthlyCharges'] * processed['tenure']
        
        processed['SeniorCitizen'] = self.extract_feature(
            df, 'SeniorCitizen',
            ['SeniorCitizen', 'Age', 'age', 'senior_citizen'],
            default=0
        )
        # Convert Age to binary if needed
        if processed['SeniorCitizen'].max() > 2:
            processed['SeniorCitizen'] = (processed['SeniorCitizen'] >= 65).astype(int)
        
        processed['gender_encoded'] = self.extract_feature(
            df, 'gender',
            ['gender', 'Gender', 'sex'],
            default=0,
            mapping={'Female': 0, 'Male': 1, 'F': 0, 'M': 1}
        )
        
        processed['Partner_encoded'] = self.extract_feature(
            df, 'Partner',
            ['Partner', 'partner', 'married'],
            default=0,
            mapping={'Yes': 1, 'No': 0}
        )
        
        processed['Dependents_encoded'] = self.extract_feature(
            df, 'Dependents',
            ['Dependents', 'dependents', 'children'],
            default=0,
            mapping={'Yes': 1, 'No': 0}
        )
        
        processed['PhoneService_encoded'] = self.extract_feature(
            df, 'PhoneService',
            ['PhoneService', 'phone_service', 'phone'],
            default=1,
            mapping={'Yes': 1, 'No': 0}
        )
        
        processed['MultipleLines_encoded'] = self.extract_feature(
            df, 'MultipleLines',
            ['MultipleLines', 'multiple_lines'],
            default=0,
            mapping={'No': 0, 'Yes': 2, 'No phone service': 1}
        )
        
        processed['InternetService_encoded'] = self.extract_feature(
            df, 'InternetService',
            ['InternetService', 'internet_service', 'internet'],
            default=0,
            mapping={'DSL': 0, 'Fiber optic': 1, 'Fiber': 1, 'No': 2}
        )
        
        processed['OnlineSecurity_encoded'] = self.extract_feature(
            df, 'OnlineSecurity',
            ['OnlineSecurity', 'online_security'],
            default=0,
            mapping={'No': 0, 'Yes': 2, 'No internet service': 1}
        )
        
        processed['OnlineBackup_encoded'] = self.extract_feature(
            df, 'OnlineBackup',
            ['OnlineBackup', 'online_backup'],
            default=0,
            mapping={'No': 0, 'Yes': 2, 'No internet service': 1}
        )
        
        processed['DeviceProtection_encoded'] = self.extract_feature(
            df, 'DeviceProtection',
            ['DeviceProtection', 'device_protection'],
            default=0,
            mapping={'No': 0, 'Yes': 2, 'No internet service': 1}
        )
        
        processed['TechSupport_encoded'] = self.extract_feature(
            df, 'TechSupport',
            ['TechSupport', 'tech_support'],
            default=0,
            mapping={'No': 0, 'Yes': 2, 'No internet service': 1}
        )
        
        processed['StreamingTV_encoded'] = self.extract_feature(
            df, 'StreamingTV',
            ['StreamingTV', 'streaming_tv'],
            default=0,
            mapping={'No': 0, 'Yes': 2, 'No internet service': 1}
        )
        
        processed['StreamingMovies_encoded'] = self.extract_feature(
            df, 'StreamingMovies',
            ['StreamingMovies', 'streaming_movies'],
            default=0,
            mapping={'No': 0, 'Yes': 2, 'No internet service': 1}
        )
        
        processed['Contract_encoded'] = self.extract_feature(
            df, 'Contract',
            ['Contract', 'contract', 'contract_type'],
            default=0,
            mapping={'Month-to-month': 0, 'Monthly': 0, 'One year': 1, 'Two year': 2}
        )
        
        processed['PaperlessBilling_encoded'] = self.extract_feature(
            df, 'PaperlessBilling',
            ['PaperlessBilling', 'paperless_billing'],
            default=1,
            mapping={'Yes': 1, 'No': 0}
        )
        
        processed['PaymentMethod_encoded'] = self.extract_feature(
            df, 'PaymentMethod',
            ['PaymentMethod', 'payment_method'],
            default=2,
            mapping={
                'Bank transfer (automatic)': 0,
                'Credit card (automatic)': 1,
                'Electronic check': 2,
                'Mailed check': 3
            }
        )
        
        # Exact column order
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
        
        return processed[final_cols]
    
    def calculate_clv(self, monthly, tenure, prob):
        """Calculate CLV"""
        if prob < 0.3:
            lifetime = 48
        elif prob < 0.6:
            lifetime = 24
        else:
            lifetime = 12
        
        if tenure < 12:
            lifetime *= 1.2
        elif tenure >= 36:
            lifetime *= 0.8
        
        survival = 1 - prob
        return monthly * lifetime * survival * (0.95 ** (lifetime / 12))
    
    def predict_batch(self, df):
        """Predict on ANY data - always works!"""
        try:
            df_original = df.copy()
            
            # Preprocess with defaults
            X = self.preprocess_batch(df)
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # Build results
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
            
            # Calculate business metrics
            results['Annual_Value'] = X['MonthlyCharges'] * 12
            results['Realistic_CLV'] = [
                self.calculate_clv(X.iloc[i]['MonthlyCharges'], X.iloc[i]['tenure'], probabilities[i])
                for i in range(len(X))
            ]
            results['Revenue_at_Risk'] = results.apply(
                lambda x: x['Realistic_CLV'] if x['Churn_Prediction'] == 1 else 0,
                axis=1
            )
            
            logger.info(f"✅ Processed {len(results)} customers")
            logger.info(f"   High Risk: {(results['Risk_Level']=='High Risk').sum()}")
            logger.info(f"   Revenue at Risk: ${results['Revenue_at_Risk'].sum():,.2f}")
            
            return results, []
            
        except Exception as e:
            logger.error(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, [f"Error: {str(e)}"]

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
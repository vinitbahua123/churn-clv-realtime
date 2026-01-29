"""
Check what features the model expects
"""
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

print("🔍 Checking model feature requirements...")
print("\n📊 Model type:", type(model).__name__)

# Check if model has feature_names_in_
if hasattr(model, 'feature_names_in_'):
    print(f"\n✅ Model expects {len(model.feature_names_in_)} features:")
    for i, name in enumerate(model.feature_names_in_):
        print(f"   {i+1}. {name}")
else:
    print("\n⚠️ Model doesn't store feature names")
    print(f"   Expected number of features: {model.n_features_in_}")

# Check scaler
if hasattr(scaler, 'feature_names_in_'):
    print(f"\n✅ Scaler expects {len(scaler.feature_names_in_)} features:")
    for i, name in enumerate(scaler.feature_names_in_):
        print(f"   {i+1}. {name}")
else:
    print("\n⚠️ Scaler doesn't store feature names")
    print(f"   Expected number of features: {scaler.n_features_in_}")

# Load the processed training data to see actual column names
print("\n📂 Checking training data format...")
try:
    df = pd.read_csv('data/processed/cleaned_churn_data.csv')
    print(f"✅ Training data has {len(df.columns)} columns:")
    for i, col in enumerate(df.columns):
        print(f"   {i+1}. {col}")
except Exception as e:
    print(f"❌ Could not load training data: {e}")

import urllib.request
import pandas as pd
import os

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
output_path = "data/raw/telco_churn.csv"

print("📥 Downloading dataset...")
urllib.request.urlretrieve(url, output_path)
print(f"✅ Dataset saved to {output_path}")

df = pd.read_csv(output_path)
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns[:5])}...")
print(f"\nTarget Distribution:")
print(df['Churn'].value_counts())

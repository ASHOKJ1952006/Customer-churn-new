"""
create_shap_explainer.py — Create SHAP explainer for optimized LightGBM model
"""

import pandas as pd
import numpy as np
import joblib
import shap
from lightgbm import LGBMClassifier

print("Loading optimized model...")
model = joblib.load('models/ds1optuna_churn_model.pkl')
scaler = joblib.load('models/ds1optuna_scaler.pkl')
feature_columns = joblib.load('models/ds1optuna_feature_columns.pkl')

print(f"Model type: {type(model)}")
print(f"Feature count: {len(feature_columns)}")

# Load sample data to create background dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df.drop('customerID', axis=1, inplace=True)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# Add engineered features
df_encoded['AvgMonthlySpend'] = df_encoded['TotalCharges'] / df_encoded['tenure'].replace(0, 1)
df_encoded['ChargeIncrease'] = (df_encoded['MonthlyCharges'] > df_encoded['AvgMonthlySpend']).astype(int)

service_cols = ['PhoneService', 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 
                'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']
df_encoded['NumServices'] = 0
for col in service_cols:
    if col in df_encoded.columns:
        df_encoded['NumServices'] += df_encoded[col]

df_encoded['IsNewCustomer'] = (df_encoded['tenure'] < 6).astype(int)

has_month_to_month = df_encoded['Contract_One year'] == 0
has_electronic_check = df_encoded['PaymentMethod_Electronic check'] == 1 if 'PaymentMethod_Electronic check' in df_encoded.columns else False
df_encoded['HighRiskCombo'] = (has_month_to_month & has_electronic_check).astype(int)

# Add aggressive features
df_encoded['TenureGroup'] = pd.cut(df_encoded['tenure'], bins=[-1, 12, 24, 48, 100], labels=[0, 1, 2, 3]).astype(int)
df_encoded['Contract_Payment_Risk'] = (has_month_to_month & has_electronic_check).astype(int)
df_encoded['Tenure_MonthlyCharges'] = df_encoded['tenure'] * df_encoded['MonthlyCharges']
df_encoded['ServiceCount_Charges'] = df_encoded['NumServices'] * df_encoded['MonthlyCharges']
df_encoded['Tenure_ServiceCount'] = df_encoded['tenure'] * df_encoded['NumServices']
df_encoded['Charges_Senior'] = df_encoded['MonthlyCharges'] * df_encoded['SeniorCitizen']
df_encoded['Contract_Partner'] = (has_month_to_month & df_encoded['Partner']).astype(int)
df_encoded['Tenure_Contract_Payment'] = df_encoded['tenure'] * has_month_to_month.astype(int) * has_electronic_check.astype(int)
df_encoded['Service_Contract_Payment'] = df_encoded['NumServices'] * has_month_to_month.astype(int) * has_electronic_check.astype(int)
df_encoded['Tenure_Service_Charges'] = df_encoded['tenure'] * df_encoded['NumServices'] * df_encoded['MonthlyCharges']

df_encoded['Charges_Tenure_Ratio'] = df_encoded['MonthlyCharges'] / (df_encoded['tenure'] + 1)
df_encoded['Total_Monthly_Ratio'] = df_encoded['TotalCharges'] / (df_encoded['MonthlyCharges'] * df_encoded['tenure'] + 1)
df_encoded['Service_Tenure_Ratio'] = df_encoded['NumServices'] / (df_encoded['tenure'] + 1)

df_encoded['poly_tenure'] = df_encoded['tenure']
df_encoded['poly_MonthlyCharges'] = df_encoded['MonthlyCharges']
df_encoded['poly_TotalCharges'] = df_encoded['TotalCharges']
df_encoded['poly_AvgMonthlySpend'] = df_encoded['AvgMonthlySpend']
df_encoded['poly_tenure^2'] = df_encoded['tenure'] ** 2
df_encoded['poly_tenure MonthlyCharges'] = df_encoded['tenure'] * df_encoded['MonthlyCharges']
df_encoded['poly_tenure TotalCharges'] = df_encoded['tenure'] * df_encoded['TotalCharges']
df_encoded['poly_tenure AvgMonthlySpend'] = df_encoded['tenure'] * df_encoded['AvgMonthlySpend']
df_encoded['poly_MonthlyCharges^2'] = df_encoded['MonthlyCharges'] ** 2
df_encoded['poly_MonthlyCharges TotalCharges'] = df_encoded['MonthlyCharges'] * df_encoded['TotalCharges']
df_encoded['poly_MonthlyCharges AvgMonthlySpend'] = df_encoded['MonthlyCharges'] * df_encoded['AvgMonthlySpend']
df_encoded['poly_TotalCharges^2'] = df_encoded['TotalCharges'] ** 2
df_encoded['poly_TotalCharges AvgMonthlySpend'] = df_encoded['TotalCharges'] * df_encoded['AvgMonthlySpend']
df_encoded['poly_AvgMonthlySpend^2'] = df_encoded['AvgMonthlySpend'] ** 2

df_encoded['Cluster_2'] = 0

# Ensure all feature columns exist
for col in feature_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

X = df_encoded[feature_columns].astype(float)

# Scale features
if hasattr(scaler, 'feature_names_in_'):
    scale_cols = list(scaler.feature_names_in_)
    scale_cols = [col for col in scale_cols if col in X.columns]
    X[scale_cols] = scaler.transform(X[scale_cols])

print(f"Data shape for SHAP: {X.shape}")

# Create SHAP explainer
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# Save explainer
print("Saving SHAP explainer...")
joblib.dump(explainer, 'models/ds1optuna_shap_explainer.pkl')

print("✅ SHAP explainer saved to models/ds1optuna_shap_explainer.pkl")

# Test it
print("Testing SHAP explainer...")
test_sample = X.iloc[[0]]
shap_vals = explainer.shap_values(test_sample)
print(f"SHAP values shape: {shap_vals[0].shape}")
print("✅ SHAP explainer working correctly")

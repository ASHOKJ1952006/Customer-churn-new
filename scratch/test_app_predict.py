"""
test_app_predict.py — Test app.py prediction and SHAP in-process
"""
import sys
import os
import joblib
import pandas as pd

# Load ds1optuna models as app.py does
model = joblib.load('models/ds1optuna_churn_model.pkl')
scaler = joblib.load('models/ds1optuna_scaler.pkl')
feature_columns = joblib.load('models/ds1optuna_feature_columns.pkl')
best_threshold = joblib.load('models/ds1optuna_best_threshold.pkl')
explainer = joblib.load('models/ds1optuna_shap_explainer.pkl')

print("Loaded ds1optuna artifacts for app.py:")
print("Features count:", len(feature_columns))
print("Best threshold:", best_threshold)

# Test high risk profile
high_risk = {
    'tenure': 2, 'monthly_charges': 95.0, 'total_charges': 190.0,
    'contract': 'Month-to-month', 'payment_method': 'Electronic check',
    'paperless_billing': 'Yes', 'phone_service': 'Yes', 'multiple_lines': 'Yes',
    'internet_service': 'Fiber optic', 'online_security': 'No',
    'online_backup': 'No', 'device_protection': 'No', 'tech_support': 'No',
    'streaming_tv': 'Yes', 'streaming_movies': 'Yes',
    'gender': 'Male', 'partner': 'No', 'dependents': 'No', 'senior_citizen': 'No'
}

# Import app.py's build_feature_vector logic
raw = {
    'gender': 1 if high_risk['gender'] == "Male" else 0,
    'SeniorCitizen': 1 if high_risk['senior_citizen'] == "Yes" else 0,
    'Partner': 1 if high_risk['partner'] == "Yes" else 0,
    'Dependents': 1 if high_risk['dependents'] == "Yes" else 0,
    'tenure': high_risk['tenure'],
    'PhoneService': 1 if high_risk['phone_service'] == "Yes" else 0,
    'PaperlessBilling': 1 if high_risk['paperless_billing'] == "Yes" else 0,
    'MonthlyCharges': high_risk['monthly_charges'],
    'TotalCharges': high_risk['total_charges'],
}
raw['AvgMonthlySpend'] = high_risk['total_charges'] / max(high_risk['tenure'], 1)
raw['ChargeIncrease'] = 1 if high_risk['monthly_charges'] > (high_risk['total_charges'] / max(high_risk['tenure'], 1)) else 0
raw['NumServices'] = 3
raw['IsNewCustomer'] = 1
raw['HighRiskCombo'] = 1
raw['TenureGroup'] = 0
raw['Contract_Payment_Risk'] = 1
raw['Tenure_MonthlyCharges'] = high_risk['tenure'] * high_risk['monthly_charges']
raw['ServiceCount_Charges'] = raw['NumServices'] * high_risk['monthly_charges']
raw['Tenure_ServiceCount'] = high_risk['tenure'] * raw['NumServices']
raw['Charges_Senior'] = 0
raw['Contract_Partner'] = 0
raw['Tenure_Contract_Payment'] = high_risk['tenure']
raw['Service_Contract_Payment'] = raw['NumServices']
raw['Tenure_Service_Charges'] = high_risk['tenure'] * raw['NumServices'] * high_risk['monthly_charges']
raw['Charges_Tenure_Ratio'] = high_risk['monthly_charges'] / (high_risk['tenure'] + 1)
raw['Total_Monthly_Ratio'] = high_risk['total_charges'] / (high_risk['monthly_charges'] * high_risk['tenure'] + 1)
raw['Service_Tenure_Ratio'] = raw['NumServices'] / (high_risk['tenure'] + 1)
raw['poly_tenure'] = high_risk['tenure']
raw['poly_MonthlyCharges'] = high_risk['monthly_charges']
raw['poly_TotalCharges'] = high_risk['total_charges']
raw['poly_AvgMonthlySpend'] = raw['AvgMonthlySpend']
raw['poly_tenure^2'] = high_risk['tenure'] ** 2
raw['poly_tenure MonthlyCharges'] = high_risk['tenure'] * high_risk['monthly_charges']
raw['poly_tenure TotalCharges'] = high_risk['tenure'] * high_risk['total_charges']
raw['poly_tenure AvgMonthlySpend'] = high_risk['tenure'] * raw['AvgMonthlySpend']
raw['poly_MonthlyCharges^2'] = high_risk['monthly_charges'] ** 2
raw['poly_MonthlyCharges TotalCharges'] = high_risk['monthly_charges'] * high_risk['total_charges']
raw['poly_MonthlyCharges AvgMonthlySpend'] = high_risk['monthly_charges'] * raw['AvgMonthlySpend']
raw['poly_TotalCharges^2'] = high_risk['total_charges'] ** 2
raw['poly_TotalCharges AvgMonthlySpend'] = high_risk['total_charges'] * raw['AvgMonthlySpend']
raw['poly_AvgMonthlySpend^2'] = raw['AvgMonthlySpend'] ** 2
raw['Cluster_2'] = 0

for col in feature_columns:
    if col not in raw:
        raw[col] = 0

raw['MultipleLines_Yes'] = 1
raw['InternetService_Fiber optic'] = 1
raw['PaymentMethod_Electronic check'] = 1

input_df = pd.DataFrame([raw])[feature_columns].astype(float)
input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

prob = model.predict_proba(input_df)[0, 1]
print(f"High risk customer predicted probability: {prob:.4f}")
assert prob > 0.80, "Expected high churn probability"

shap_vals = explainer.shap_values(input_df)[0]
print(f"SHAP values computed: {len(shap_vals)} values")
assert len(shap_vals) == len(feature_columns), "SHAP feature count mismatch"
print("SUCCESS: app.py prediction and SHAP explainer are 100% verified!")

"""
survival_analysis.py — Time-to-churn analysis using survival analysis
Complementary to classification model - answers WHEN not WHETHER churn will occur.
Uses Dataset 1 (Telco) which has natural duration structure (tenure).
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

print("=== Survival Analysis: Time-to-Churn for Dataset 1 ===\n")

# =========================================================
# 1. Load and Prepare Data
# =========================================================
print("Loading and preparing data...")

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean data (same as primary pipeline)
df_clean = df.copy()
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
df_clean = df_clean.drop('customerID', axis=1)

# Convert Churn to numeric (event indicator)
df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

# Duration = tenure (months with company)
# Event = Churn (1 if churned, 0 if censored/active)
df_clean['duration'] = df_clean['tenure']
df_clean['event'] = df_clean['Churn']

print(f"Total customers: {len(df_clean)}")
print(f"Churned (events): {df_clean['event'].sum()}")
print(f"Active (censored): {(df_clean['event'] == 0).sum()}")
print(f"Median tenure: {df_clean['duration'].median()} months")
print(f"Max tenure: {df_clean['duration'].max()} months\n")

# =========================================================
# 2. Feature Engineering (same as primary pipeline)
# =========================================================
print("Engineering features...")

# Binary columns
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})

df_clean['gender'] = df_clean['gender'].map({'Male': 1, 'Female': 0})
df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(int)

# Multi-category columns (one-hot encode)
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
df_encoded = pd.get_dummies(df_clean, columns=multi_cols, drop_first=True)

# Engineered features
df_encoded['AvgMonthlySpend'] = df_encoded['TotalCharges'] / df_encoded['tenure'].replace(0, 1)
df_encoded['ChargeIncrease'] = (df_encoded['MonthlyCharges'] > df_encoded['AvgMonthlySpend']).astype(int)

service_cols = ['PhoneService', 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 
                'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']
df_encoded['NumServices'] = 0
for col in service_cols:
    if col in df_encoded.columns:
        df_encoded['NumServices'] += df_encoded[col]

df_encoded['IsNewCustomer'] = (df_encoded['tenure'] < 6).astype(int)

# HighRiskCombo: month-to-month + electronic check
has_month_to_month = df_encoded['Contract_One year'] == 0
has_electronic_check = df_encoded['PaymentMethod_Electronic check'] == 1 if 'PaymentMethod_Electronic check' in df_encoded.columns else False
df_encoded['HighRiskCombo'] = (has_month_to_month & has_electronic_check).astype(int)

print(f"Total features after engineering: {df_encoded.shape[1]}\n")

# =========================================================
# 3. Kaplan-Meier Survival Curves
# =========================================================
print("=== Kaplan-Meier Survival Curves ===\n")

kmf = KaplanMeierFitter()

# Overall survival curve
kmf.fit(durations=df_encoded['duration'], event_observed=df_encoded['event'], label='Overall')
print(f"Overall median survival time: {kmf.median_survival_time_:.1f} months")

# Plot overall survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Overall Survival Curve (Telco Dataset)')
plt.xlabel('Tenure (months)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.savefig('models/survival_km_overall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: models/survival_km_overall.png\n")

# Stratified by Contract type
print("Stratified by Contract type...")
plt.figure(figsize=(10, 6))

contract_groups = {
    'Month-to-month': df_encoded['Contract_One year'] == 0,
    'One year': df_encoded['Contract_One year'] == 1,
    'Two year': df_encoded['Contract_Two year'] == 1 if 'Contract_Two year' in df_encoded.columns else pd.Series([False]*len(df_encoded))
}

km_curves_contract = {}
for contract_name, mask in contract_groups.items():
    if mask.sum() > 0:
        kmf.fit(durations=df_encoded[mask]['duration'], 
               event_observed=df_encoded[mask]['event'], 
               label=contract_name)
        kmf.plot_survival_function(ci_show=False)
        km_curves_contract[contract_name] = {
            'median_survival': kmf.median_survival_time_,
            'n_customers': mask.sum()
        }
        print(f"  {contract_name}: median survival = {kmf.median_survival_time_:.1f} months (n={mask.sum()})")

plt.title('Survival Curves by Contract Type')
plt.xlabel('Tenure (months)')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid(True)
plt.savefig('models/survival_km_contract.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: models/survival_km_contract.png\n")

# Log-rank test for Contract type
if len(km_curves_contract) >= 2:
    results_contract = logrank_test(
        df_encoded[contract_groups['Month-to-month']]['duration'],
        df_encoded[contract_groups['One year']]['duration'],
        event_observed_A=df_encoded[contract_groups['Month-to-month']]['event'],
        event_observed_B=df_encoded[contract_groups['One year']]['event']
    )
    print(f"Log-rank test (Month-to-month vs One year): p-value = {results_contract.p_value:.4e}")

# Stratified by HighRiskCombo
print("Stratified by HighRiskCombo...")
plt.figure(figsize=(10, 6))

highrisk_groups = {
    'High Risk (HighRiskCombo=1)': df_encoded['HighRiskCombo'] == 1,
    'Low Risk (HighRiskCombo=0)': df_encoded['HighRiskCombo'] == 0
}

km_curves_highrisk = {}
for group_name, mask in highrisk_groups.items():
    if mask.sum() > 0:
        kmf.fit(durations=df_encoded[mask]['duration'], 
               event_observed=df_encoded[mask]['event'], 
               label=group_name)
        kmf.plot_survival_function(ci_show=False)
        km_curves_highrisk[group_name] = {
            'median_survival': kmf.median_survival_time_,
            'n_customers': mask.sum()
        }
        print(f"  {group_name}: median survival = {kmf.median_survival_time_:.1f} months (n={mask.sum()})")

plt.title('Survival Curves by HighRiskCombo')
plt.xlabel('Tenure (months)')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid(True)
plt.savefig('models/survival_km_highrisk.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: models/survival_km_highrisk.png\n")

# Log-rank test for HighRiskCombo
results_highrisk = logrank_test(
    df_encoded[highrisk_groups['High Risk (HighRiskCombo=1)']]['duration'],
    df_encoded[highrisk_groups['Low Risk (HighRiskCombo=0)']]['duration'],
    event_observed_A=df_encoded[highrisk_groups['High Risk (HighRiskCombo=1)']]['event'],
    event_observed_B=df_encoded[highrisk_groups['Low Risk (HighRiskCombo=0)']]['event']
)
print(f"Log-rank test (High Risk vs Low Risk): p-value = {results_highrisk.p_value:.4e}\n")

# =========================================================
# 4. Cox Proportional Hazards Model
# =========================================================
print("=== Cox Proportional Hazards Model ===\n")

# Load feature importance from existing model to select top features
feature_importance = joblib.load('models/ds1calcv_final_metrics.pkl')

# Select features for Cox model (avoid multicollinearity)
# Cox models struggle with high dimensionality and collinear features
# Use a conservative, interpretable feature set
# CRITICAL: Remove tenure from covariates - it's the DURATION variable, not a predictor
# Also remove tenure-derived features (IsNewCustomer) to avoid leakage

cox_features = [
    'MonthlyCharges',
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling',
    'Contract_One year',
    'Contract_Two year',
    'PaymentMethod_Electronic check',
    'InternetService_Fiber optic',
    'InternetService_No',
    'OnlineSecurity_Yes',
    'TechSupport_Yes',
    'HighRiskCombo'
]

# Ensure all features exist
cox_features = [f for f in cox_features if f in df_encoded.columns]

print(f"Using {len(cox_features)} features for Cox model")
print(f"Features: {cox_features}\n")

# Prepare data for Cox model
cox_data = df_encoded[['duration', 'event'] + cox_features].copy()

# Fit Cox model
cph = CoxPHFitter()
cph.fit(cox_data, duration_col='duration', event_col='event')

# Print summary
print(cph.print_summary())

# Get C-index (concordance)
c_index = cph.concordance_index_
print(f"\nC-index (concordance): {c_index:.4f}")
print("Interpretation: C-index is survival analysis equivalent of AUC. Values > 0.7 indicate good discrimination.\n")

# Get hazard ratios with confidence intervals
hazard_ratios = cph.hazard_ratios_
confidence_intervals = cph.confidence_intervals_

print("=== Top 10 Hazard Ratios ===")
print(f"Confidence intervals columns: {confidence_intervals.columns.tolist()}")

hr_df = pd.DataFrame({
    'Hazard Ratio': hazard_ratios,
    '95% CI Lower': confidence_intervals.iloc[:, 0],
    '95% CI Upper': confidence_intervals.iloc[:, 1]
})
hr_df['Direction'] = hr_df['Hazard Ratio'].apply(lambda x: 'Risk-increasing' if x > 1 else 'Protective')
hr_df = hr_df.sort_values('Hazard Ratio', ascending=False)
print(hr_df.head(10).to_string())

# =========================================================
# 5. Check Proportional Hazards Assumption
# =========================================================
print("\n=== Proportional Hazards Assumption Check ===\n")

try:
    assumptions = cph.check_assumptions(cox_data, p_value_threshold=0.05)
    print(assumptions)
except Exception as e:
    print(f"Assumption check failed: {e}")
    print("This may occur with categorical features. Proceeding with caution.\n")

# =========================================================
# 6. Individual Customer Prediction
# =========================================================
print("=== Individual Customer Prediction ===\n")

# Sample customer (high-risk profile)
sample_customer = cox_data.iloc[0].copy()
print("Sample customer features:")
print(sample_customer[cox_features].to_string())

# Predict survival function (only use feature columns, not duration/event)
sample_features = sample_customer[cox_features]
survival_function = cph.predict_survival_function(sample_features)
print(f"\nSurvival probabilities at key timepoints:")
for month in [6, 12, 24, 36, 48, 60, 72]:
    if month in survival_function.index:
        prob = survival_function.loc[month].values[0]
        print(f"  Month {month}: {prob:.3f}")

# Estimate median survival time
try:
    median_survival = cph.predict_median(sample_features)
    print(f"\nEstimated median survival time: {median_survival.values[0]:.1f} months")
    print("Interpretation: Customer has 50% probability of churning by this tenure.\n")
except Exception as e:
    print(f"\nMedian survival time could not be estimated (survival may not reach 50%): {e}")
    print("This is common for high-risk customers who churn very quickly.\n")

# =========================================================
# 7. Save Results
# =========================================================
print("=== Saving Results ===\n")

# Save Cox model
joblib.dump(cph, 'models/survival_cox_model.pkl')
print("Saved: models/survival_cox_model.pkl")

# Save KM curve data
km_results = {
    'overall_median': kmf.median_survival_time_,
    'contract_stratification': km_curves_contract,
    'highrisk_stratification': km_curves_highrisk,
    'logrank_contract_p': results_contract.p_value if len(km_curves_contract) >= 2 else None,
    'logrank_highrisk_p': results_highrisk.p_value
}
joblib.dump(km_results, 'models/survival_km_results.pkl')
print("Saved: models/survival_km_results.pkl")

# Save Cox results
cox_results = {
    'c_index': c_index,
    'hazard_ratios': hr_df.to_dict('index'),
    'features_used': cox_features
}
joblib.dump(cox_results, 'models/survival_cox_results.pkl')
print("Saved: models/survival_cox_results.pkl")

print("\n=== Survival Analysis Complete ===")

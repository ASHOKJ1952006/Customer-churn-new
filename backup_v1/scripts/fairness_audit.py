"""
fairness_audit.py — Fairness audit across demographic groups
Checks for bias in model predictions across gender and senior citizen status.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

print("=== Fairness Audit Across Demographic Groups ===\n")

# Load Dataset 1 model and data
print("Loading model and data...")
model = joblib.load('models/ds1calcv_churn_model.pkl')
scaler = joblib.load('models/ds1calcv_scaler.pkl')
feature_columns = joblib.load('models/ds1calcv_feature_columns.pkl')
best_threshold = joblib.load('models/ds1calcv_best_threshold.pkl')

# Load Dataset 1
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess (same as training)
df_clean = df.copy()
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

# Store demographic info BEFORE preprocessing (keep original values)
demographics = df[['gender', 'SeniorCitizen']].copy()
demographics['gender'] = demographics['gender'].map({'Male': 1, 'Female': 0})

# Handle SeniorCitizen - might already be numeric or string
if demographics['SeniorCitizen'].dtype == object:
    demographics['SeniorCitizen'] = demographics['SeniorCitizen'].map({'Yes': 1, 'No': 0})
else:
    demographics['SeniorCitizen'] = demographics['SeniorCitizen'].astype(int)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})

df_clean['gender'] = df_clean['gender'].map({'Male': 1, 'Female': 0})
df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({'Yes': 1, 'No': 0})

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
df_encoded = pd.get_dummies(df_clean, columns=multi_cols, drop_first=True)

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

# Ensure all feature columns exist
for col in feature_columns:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

X = df_encoded[feature_columns].astype(float)
y = df_clean['Churn']

# Scale features
scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'ChargeIncrease', 'NumServices']
X[scale_cols] = scaler.transform(X[scale_cols])

# Train/test split
X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
    X, y, demographics, test_size=0.2, random_state=42, stratify=y
)

print(f"Test set size: {X_test.shape[0]}")
print(f"Churn rate in test: {y_test.mean():.2%}\n")

# Get predictions
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= best_threshold).astype(int)

# =========================================================
# Fairness Metrics by Group
# =========================================================

def calculate_group_metrics(y_true, y_pred, group_name, group_size):
    """Calculate metrics for a specific group."""
    if group_size == 0:
        return None
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Disparate impact: positive prediction rate
    ppr = (tp + fp) / group_size
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # False negative rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'Group': group_name,
        'Size': group_size,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'PPR': ppr,
        'FPR': fpr,
        'FNR': fnr,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

print("=== Fairness Analysis by Gender ===")

# Gender analysis
gender_groups = {
    'Female': demo_test['gender'] == 0,
    'Male': demo_test['gender'] == 1
}

gender_results = []
for group_name, mask in gender_groups.items():
    group_size = mask.sum()
    if group_size > 0:
        metrics = calculate_group_metrics(
            y_test[mask], preds[mask], group_name, group_size
        )
        gender_results.append(metrics)

gender_df = pd.DataFrame(gender_results)
print(gender_df.to_string(index=False))

# Calculate fairness ratios
if len(gender_results) == 2:
    female = gender_results[0]
    male = gender_results[1]
    
    print("\n=== Fairness Ratios (Female / Male) ===")
    print(f"Accuracy Ratio: {female['Accuracy'] / male['Accuracy']:.4f}")
    print(f"Recall Ratio: {female['Recall'] / male['Recall']:.4f}")
    print(f"FPR Ratio: {female['FPR'] / male['FPR']:.4f}")
    print(f"FNR Ratio: {female['FNR'] / male['FNR']:.4f}")
    print(f"PPR Ratio (Disparate Impact): {female['PPR'] / male['PPR']:.4f}")
    
    # 80% rule (disparate impact > 0.8 is acceptable)
    disparate_impact = female['PPR'] / male['PPR']
    if disparate_impact >= 0.8:
        print(f"\n✅ Disparate Impact ({disparate_impact:.4f}) meets 80% rule")
    else:
        print(f"\n⚠️ Disparate Impact ({disparate_impact:.4f}) violates 80% rule")

print("\n" + "="*80 + "\n")

print("=== Fairness Analysis by Senior Citizen Status ===")

# Senior citizen analysis
print(f"Senior citizen distribution in test:")
print(demo_test['SeniorCitizen'].value_counts())

senior_groups = {
    'Non-Senior': demo_test['SeniorCitizen'] == 0,
    'Senior': demo_test['SeniorCitizen'] == 1
}

senior_results = []
for group_name, mask in senior_groups.items():
    group_size = mask.sum()
    if group_size > 0:
        metrics = calculate_group_metrics(
            y_test[mask], preds[mask], group_name, group_size
        )
        senior_results.append(metrics)

senior_df = pd.DataFrame(senior_results)
print(senior_df.to_string(index=False))

# Calculate fairness ratios
if len(senior_results) == 2:
    non_senior = senior_results[0]
    senior = senior_results[1]
    
    print("\n=== Fairness Ratios (Non-Senior / Senior) ===")
    print(f"Accuracy Ratio: {non_senior['Accuracy'] / senior['Accuracy']:.4f}")
    print(f"Recall Ratio: {non_senior['Recall'] / senior['Recall']:.4f}")
    print(f"FPR Ratio: {non_senior['FPR'] / senior['FPR']:.4f}")
    print(f"FNR Ratio: {non_senior['FNR'] / senior['FNR']:.4f}")
    print(f"PPR Ratio (Disparate Impact): {non_senior['PPR'] / senior['PPR']:.4f}")
    
    # 80% rule
    disparate_impact = non_senior['PPR'] / senior['PPR']
    if disparate_impact >= 0.8:
        print(f"\n✅ Disparate Impact ({disparate_impact:.4f}) meets 80% rule")
    else:
        print(f"\n⚠️ Disparate Impact ({disparate_impact:.4f}) violates 80% rule")

print("\n" + "="*80 + "\n")

# =========================================================
# Overall Summary
# =========================================================
print("=== Fairness Audit Summary ===")

# Check for concerning disparities
concerns = []

# Gender concerns
if len(gender_results) == 2:
    recall_diff = abs(gender_results[0]['Recall'] - gender_results[1]['Recall'])
    fpr_diff = abs(gender_results[0]['FPR'] - gender_results[1]['FPR'])
    
    if recall_diff > 0.1:
        concerns.append(f"Gender recall disparity: {recall_diff:.4f}")
    if fpr_diff > 0.1:
        concerns.append(f"Gender FPR disparity: {fpr_diff:.4f}")

# Senior citizen concerns
if len(senior_results) == 2:
    recall_diff = abs(senior_results[0]['Recall'] - senior_results[1]['Recall'])
    fpr_diff = abs(senior_results[0]['FPR'] - senior_results[1]['FPR'])
    
    if recall_diff > 0.1:
        concerns.append(f"Senior citizen recall disparity: {recall_diff:.4f}")
    if fpr_diff > 0.1:
        concerns.append(f"Senior citizen FPR disparity: {fpr_diff:.4f}")

if concerns:
    print("⚠️ Concerning disparities found:")
    for concern in concerns:
        print(f"  - {concern}")
else:
    print("✅ No concerning disparities found (> 0.10 threshold)")

# =========================================================
# Save Results
# =========================================================
print("\n=== Saving Results ===")

fairness_results = {
    'gender_analysis': gender_df.to_dict('records'),
    'senior_analysis': senior_df.to_dict('records'),
    'concerns': concerns,
    'threshold_used': best_threshold
}

joblib.dump(fairness_results, 'models/ds1calcv_fairness_audit.pkl')

print("Fairness audit results saved to models/ds1calcv_fairness_audit.pkl")

print("\n=== Fairness Audit Complete ===")

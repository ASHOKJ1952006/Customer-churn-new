"""
cost_sensitive_threshold.py — Business-driven threshold selection
Replaces F1-maximizing threshold with cost-sensitive threshold based on business costs.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

print("=== Cost-Sensitive Threshold Selection ===\n")

# Load Dataset 1 model and data
print("Loading model and data...")
model = joblib.load('models/ds1calcv_churn_model.pkl')
scaler = joblib.load('models/ds1calcv_scaler.pkl')
feature_columns = joblib.load('models/ds1calcv_feature_columns.pkl')

# Load Dataset 1
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess (same as training)
df_clean = df.copy()
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)

# Convert Churn to numeric
df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Test set size: {X_test.shape[0]}")
print(f"Churn rate in test: {y_test.mean():.2%}\n")

# Get predictions
probs = model.predict_proba(X_test)[:, 1]

# =========================================================
# Business Cost Parameters
# =========================================================
print("=== Business Cost Parameters ===")

# Typical telecom costs (adjustable)
COST_FP = 50      # Cost of false positive (retention offer sent to non-churner)
COST_FN = 500     # Cost of false negative (losing a customer)
COST_TP = 50      # Cost of true positive (retention offer sent to churner who stays)
COST_TN = 0       # Cost of true negative (no action, customer stays)

print(f"False Positive Cost (retention offer to non-churner): ${COST_FP}")
print(f"False Negative Cost (losing a customer): ${COST_FN}")
print(f"True Positive Cost (retention offer to saved customer): ${COST_TP}")
print(f"True Negative Cost (no action, customer stays): ${COST_TN}")
print(f"\nCost Ratio (FN/FP): {COST_FN / COST_FP:.1f}x")
print("Interpretation: Losing a customer is 10x more expensive than a false alarm\n")

# =========================================================
# Find Cost-Optimal Threshold
# =========================================================
print("=== Finding Cost-Optimal Threshold ===")

precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

# Calculate expected cost for each threshold
costs = []
for threshold in thresholds:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    # Expected cost = FP * COST_FP + FN * COST_FN + TP * COST_TP + TN * COST_TN
    expected_cost = fp * COST_FP + fn * COST_FN + tp * COST_TP + tn * COST_TN
    costs.append(expected_cost)

# Find threshold with minimum cost
min_cost_idx = np.argmin(costs)
cost_optimal_threshold = thresholds[min_cost_idx]
min_expected_cost = costs[min_cost_idx]

print(f"Cost-optimal threshold: {cost_optimal_threshold:.4f}")
print(f"Minimum expected cost: ${min_expected_cost:.2f} per {len(y_test)} customers")
print(f"Average cost per customer: ${min_expected_cost / len(y_test):.2f}\n")

# =========================================================
# Compare with F1-Optimal Threshold
# =========================================================
print("=== Comparison with F1-Optimal Threshold ===")

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
f1_optimal_idx = f1_scores.argmax()
f1_optimal_threshold = thresholds[f1_optimal_idx]

print(f"F1-optimal threshold: {f1_optimal_threshold:.4f}")

# Calculate metrics for both thresholds
for name, threshold in [("F1-Optimal", f1_optimal_threshold), ("Cost-Optimal", cost_optimal_threshold)]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    expected_cost = fp * COST_FP + fn * COST_FN + tp * COST_TP + tn * COST_TN
    
    print(f"\n{name} (threshold={threshold:.4f}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Expected Cost: ${expected_cost:.2f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# =========================================================
# Sensitivity Analysis
# =========================================================
print("\n=== Sensitivity Analysis (Cost Ratios) ===")

cost_ratios = [5, 10, 15, 20, 30]  # FN/FP cost ratios
results = []

for ratio in cost_ratios:
    cost_fn = ratio * COST_FP
    costs = []
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        expected_cost = fp * COST_FP + fn * cost_fn + tp * COST_TP + tn * COST_TN
        costs.append(expected_cost)
    
    min_idx = np.argmin(costs)
    opt_threshold = thresholds[min_idx]
    preds = (probs >= opt_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    results.append({
        'Cost_Ratio': ratio,
        'Threshold': opt_threshold,
        'Recall': recall,
        'Expected_Cost': costs[min_idx]
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# =========================================================
# Save Results
# =========================================================
print("\n=== Saving Results ===")

# Save cost-optimal threshold
joblib.dump(cost_optimal_threshold, 'models/ds1calcv_cost_sensitive_threshold.pkl')

# Save comparison results
comparison_results = {
    'f1_threshold': f1_optimal_threshold,
    'cost_threshold': cost_optimal_threshold,
    'cost_parameters': {
        'cost_fp': COST_FP,
        'cost_fn': COST_FN,
        'cost_tp': COST_TP,
        'cost_tn': COST_TN
    },
    'sensitivity_analysis': results
}
joblib.dump(comparison_results, 'models/ds1calcv_cost_analysis.pkl')

print("Cost-sensitive threshold saved to models/ds1calcv_cost_sensitive_threshold.pkl")
print("Analysis results saved to models/ds1calcv_cost_analysis.pkl")

print("\n=== Cost-Sensitive Threshold Selection Complete ===")

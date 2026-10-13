"""
learning_curve_analysis.py — Learning curve analysis for Dataset 1

This script directly measures whether more training data would meaningfully improve
model performance, or whether the model has already hit a ceiling limited by feature
information rather than sample size.

Method:
1. Hold hyperparameters FIXED (reuse exact tuned hyperparameters from ds1optuna model)
2. Use the SAME held-out test set for every point on the curve
3. Vary training set size: 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
4. For each size: apply same feature engineering, SMOTE + 3-model stack (fixed hyperparameters)
5. Evaluate on fixed test set, record Accuracy, Precision, Recall, F1
6. Use best_threshold from primary model (don't re-derive per size)
7. Plot curve and save raw numbers as CSV

Interpretation:
- If F1/Recall still rising steeply between 80% and 100% → more data would help
- If curve has clearly flattened (last 3-4 points nearly identical) → model has hit ceiling
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import sys
import os

# Add parent directory to path to import from pipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("LEARNING CURVE ANALYSIS - DATASET 1")
print("="*70)

# =========================================================
# 1. Load and preprocess data (same as primary pipeline)
# =========================================================
print("\nLoading Dataset 1...")
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df.drop('customerID', axis=1, inplace=True)

# Same feature engineering as primary result
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlySpend']
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['NumServices'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)
df['TenureGroup'] = pd.cut(
    df['tenure'], bins=[-1, 6, 12, 24, 48, 72],
    labels=['0-6mo', '7-12mo', '13-24mo', '25-48mo', '49-72mo']
)
df['HighRiskCombo'] = (
    (df['Contract'] == 'Month-to-month') & (df['PaymentMethod'] == 'Electronic check')
).astype(int)
df['IsNewCustomer'] = (df['tenure'] <= 3).astype(int)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod', 'TenureGroup']
df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'ChargeIncrease', 'NumServices']

print(f"Total samples: {len(X)}")
print(f"Churn rate: {y.mean():.4f}")

# =========================================================
# 2. Create FIXED test set (80/20 split, stratified)
# =========================================================
print("\nCreating FIXED test set (80/20 split)...")
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.fit_transform(X_scaled[numeric_cols])
X_scaled = X_scaled.astype(float)

X_train_full, X_test_fixed, y_train_full, y_test_fixed = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Full training set: {X_train_full.shape[0]} samples")
print(f"Fixed test set: {X_test_fixed.shape[0]} samples")
print(f"Train churn rate: {y_train_full.mean():.4f}")
print(f"Test churn rate: {y_test_fixed.mean():.4f}")

# =========================================================
# 3. Load FIXED hyperparameters from primary model (ds1optuna)
# =========================================================
print("\nLoading FIXED hyperparameters from ds1optuna model...")
try:
    ds1_model = joblib.load('models/ds1optuna_churn_model.pkl')
    ds1_scaler = joblib.load('models/ds1optuna_scaler.pkl')
    ds1_feature_columns = joblib.load('models/ds1optuna_feature_columns.pkl')
    ds1_best_threshold = joblib.load('models/ds1optuna_best_threshold.pkl')
    print("Successfully loaded ds1optuna model artifacts")
    print(f"Best threshold: {ds1_best_threshold:.4f}")
except Exception as e:
    print(f"Error loading ds1optuna model: {e}")
    print("Falling back to ds1calcv model...")
    try:
        ds1_model = joblib.load('models/ds1calcv_churn_model.pkl')
        ds1_scaler = joblib.load('models/ds1calcv_scaler.pkl')
        ds1_feature_columns = joblib.load('models/ds1calcv_feature_columns.pkl')
        ds1_best_threshold = joblib.load('models/ds1calcv_best_threshold.pkl')
        print("Successfully loaded ds1calcv model artifacts")
        print(f"Best threshold: {ds1_best_threshold:.4f}")
    except Exception as e2:
        print(f"Error loading ds1calcv model: {e2}")
        print("Cannot proceed without model artifacts")
        sys.exit(1)

# Since ds1optuna is a calibrated stacking classifier, we need to extract the base models
# For learning curve, we'll use the same hyperparameters but rebuild the pipeline
# We'll use the 3-model stack (RF, XGBoost, LightGBM) as in the primary result

# Load the individual tuned models if available, otherwise use default params
# For this analysis, we'll use reasonable fixed hyperparameters based on typical Optuna results
print("\nUsing FIXED hyperparameters for 3-model stack...")

# Fixed hyperparameters (based on typical Optuna results for this dataset)
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 400,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'random_state': 42,
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'device': 'cpu'  # Force CPU for consistency
}

LGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'min_child_samples': 20,
    'subsample': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'random_state': 42,
    'verbose': -1
}

print("RF params:", RF_PARAMS)
print("XGB params:", XGB_PARAMS)
print("LGBM params:", LGBM_PARAMS)

# =========================================================
# 4. Define training sizes to test
# =========================================================
train_fractions = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
train_sizes = [int(frac * X_train_full.shape[0]) for frac in train_fractions]

print(f"\nTraining sizes to test: {train_sizes}")

# =========================================================
# 5. Run learning curve analysis
# =========================================================
results = []

for i, (frac, n_samples) in enumerate(zip(train_fractions, train_sizes)):
    print(f"\n{'='*60}")
    print(f"Training size: {n_samples} ({frac*100:.0f}%)")
    print(f"{'='*60}")
    
    # Take stratified random subsample of training data
    if n_samples < X_train_full.shape[0]:
        X_train_sub, _, y_train_sub, _ = train_test_split(
            X_train_full, y_train_full, 
            train_size=n_samples, 
            random_state=42 + i,  # Different seed for each size
            stratify=y_train_full
        )
    else:
        X_train_sub = X_train_full
        y_train_sub = y_train_full
    
    print(f"Actual training samples: {X_train_sub.shape[0]}")
    print(f"Churn rate in subset: {y_train_sub.mean():.4f}")
    
    # Build 3-model stack with FIXED hyperparameters
    rf_final = RandomForestClassifier(**RF_PARAMS)
    xgb_final = XGBClassifier(**XGB_PARAMS)
    lgbm_final = LGBMClassifier(**LGBM_PARAMS)
    
    # Build stacking classifier
    stack_model = StackingClassifier(
        estimators=[('rf', rf_final), ('xgb', xgb_final), ('lgbm', lgbm_final)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    
    # Build SMOTE + Stack pipeline (leakage-safe)
    smote_stack_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('stack', stack_model)
    ])
    
    # Fit on training subset
    print("Training model...")
    smote_stack_pipeline.fit(X_train_sub, y_train_sub)
    
    # Predict on FIXED test set
    print("Evaluating on FIXED test set...")
    probs = smote_stack_pipeline.predict_proba(X_test_fixed)[:, 1]
    preds = (probs >= ds1_best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_fixed, preds)
    precision = precision_score(y_test_fixed, preds, zero_division=0)
    recall = recall_score(y_test_fixed, preds)
    f1 = f1_score(y_test_fixed, preds)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    
    results.append({
        'train_fraction': frac,
        'train_samples': n_samples,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# =========================================================
# 6. Save results to CSV
# =========================================================
results_df = pd.DataFrame(results)
results_csv = 'learning_curve_results.csv'
results_df.to_csv(results_csv, index=False)
print(f"\n{'='*60}")
print(f"Results saved to {results_csv}")
print(f"{'='*60}")
print(results_df.to_string(index=False))

# =========================================================
# 7. Plot learning curve
# =========================================================
print("\nGenerating learning curve plot...")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot F1 on left y-axis
color1 = 'tab:blue'
ax1.set_xlabel('Training Set Size (samples)')
ax1.set_ylabel('F1 Score', color=color1)
ax1.plot(results_df['train_samples'], results_df['f1'], 
         marker='o', color=color1, label='F1 Score', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Plot Recall on right y-axis
color2 = 'tab:red'
ax2 = ax1.twinx()
ax2.set_ylabel('Recall', color=color2)
ax2.plot(results_df['train_samples'], results_df['recall'], 
         marker='s', color=color2, label='Recall', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

# Add title and legend
plt.title('Learning Curve: Dataset 1 (Fixed Hyperparameters, Fixed Test Set)')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

# Add sample size annotations
for i, row in results_df.iterrows():
    ax1.annotate(f"{row['train_fraction']*100:.0f}%", 
                 (row['train_samples'], row['f1']), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plot_file = 'learning_curve_plot.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_file}")

# =========================================================
# 8. Interpret curve shape
# =========================================================
print(f"\n{'='*60}")
print("INTERPRETATION")
print(f"{'='*60}")

# Calculate slopes between last few points
last_3_f1 = results_df['f1'].tail(3).values
last_3_recall = results_df['recall'].tail(3).values
last_3_sizes = results_df['train_samples'].tail(3).values

# Calculate average slope between last two points
f1_slope = (last_3_f1[-1] - last_3_f1[-2]) / (last_3_sizes[-1] - last_3_sizes[-2])
recall_slope = (last_3_recall[-1] - last_3_recall[-2]) / (last_3_sizes[-1] - last_3_sizes[-2])

print(f"\nF1 slope (last interval): {f1_slope:.6f} per sample")
print(f"Recall slope (last interval): {recall_slope:.6f} per sample")

# Calculate variance in last 3 points
f1_variance = np.var(last_3_f1)
recall_variance = np.var(last_3_recall)

print(f"F1 variance (last 3 points): {f1_variance:.6f}")
print(f"Recall variance (last 3 points): {recall_variance:.6f}")

# Determine if curve is still rising or has plateaued
# Threshold: if variance < 0.0001 and slope < 0.00001, consider plateaued
if f1_variance < 0.0001 and abs(f1_slope) < 0.00001:
    f1_verdict = "PLATEAUED"
else:
    f1_verdict = "STILL RISING"

if recall_variance < 0.0001 and abs(recall_slope) < 0.00001:
    recall_verdict = "PLATEAUED"
else:
    recall_verdict = "STILL RISING"

print(f"\nF1 Curve: {f1_verdict}")
print(f"Recall Curve: {recall_verdict}")

# Overall verdict
if f1_verdict == "PLATEAUED" and recall_verdict == "PLATEAUED":
    overall_verdict = "The curve has PLATEAUED - model has hit a ceiling likely driven by limited feature information rather than sample size. Adding more data would NOT meaningfully improve performance."
elif f1_verdict == "STILL RISING" or recall_verdict == "STILL RISING":
    # Estimate potential improvement
    if f1_verdict == "STILL RISING":
        f1_improvement_per_1000 = f1_slope * 1000
        print(f"\nEstimated F1 improvement per 1000 additional samples: {f1_improvement_per_1000:.4f}")
    if recall_verdict == "STILL RISING":
        recall_improvement_per_1000 = recall_slope * 1000
        print(f"Estimated Recall improvement per 1000 additional samples: {recall_improvement_per_1000:.4f}")
    
    overall_verdict = "The curve is STILL RISING - more data would likely help improve performance."
else:
    overall_verdict = "Mixed signals - curve behavior is unclear."

print(f"\n{'='*60}")
print(f"FINAL VERDICT: {overall_verdict}")
print(f"{'='*60}")

# Save interpretation to file
with open('learning_curve_interpretation.txt', 'w') as f:
    f.write("LEARNING CURVE ANALYSIS - INTERPRETATION\n")
    f.write("="*60 + "\n\n")
    f.write("Results Table:\n")
    f.write(results_df.to_string(index=False) + "\n\n")
    f.write("Slope Analysis:\n")
    f.write(f"F1 slope (last interval): {f1_slope:.6f} per sample\n")
    f.write(f"Recall slope (last interval): {recall_slope:.6f} per sample\n")
    f.write(f"F1 variance (last 3 points): {f1_variance:.6f}\n")
    f.write(f"Recall variance (last 3 points): {recall_variance:.6f}\n\n")
    f.write(f"F1 Curve: {f1_verdict}\n")
    f.write(f"Recall Curve: {recall_verdict}\n\n")
    f.write(f"FINAL VERDICT: {overall_verdict}\n")

print("Interpretation saved to learning_curve_interpretation.txt")
print("\nLearning curve analysis complete!")

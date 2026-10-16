import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, brier_score_loss

# Add project root and backend to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'backend'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

from feature_engineering_v2 import engineer_features_v2, get_churn_v2_feature_names, build_single_vector_v2
from stacking_model_v2 import ChurnV2StackingEnsemble
from meta_learner import LogitLogisticRegression

# 1. Load trained artifacts
scale_cols, feature_cols = get_churn_v2_feature_names()
rf = joblib.load('models/churn_v2_rf.pkl')
xgb = joblib.load('models/churn_v2_xgb.pkl')
lgbm = joblib.load('models/churn_v2_lgbm.pkl')
scaler = joblib.load('models/churn_v2_scaler.pkl')

# Train and package the optimized ensemble
data = joblib.load('scratch/cached_base_preds.pkl')
oof_rf, oof_xgb, oof_lgbm = data['oof_rf'], data['oof_xgb'], data['oof_lgbm']
test_rf, test_xgb, test_lgbm = data['test_rf'], data['test_xgb'], data['test_lgbm']
y_train, y_test = data['y_train'], data['y_test']

eps = 1e-5
oof_meta = np.column_stack([oof_rf, oof_xgb, oof_lgbm])

# Meta-learner: LogitLogisticRegression with C=0.1
opt_meta = LogitLogisticRegression(C=0.1, penalty='l2', eps=eps, random_state=42)
opt_meta.fit(oof_meta, y_train)

# Create optimized stacking ensemble
opt_ensemble = ChurnV2StackingEnsemble(
    rf=rf,
    xgb=xgb,
    lgbm=lgbm,
    meta=opt_meta,
    feature_cols=feature_cols,
    scale_cols=scale_cols
)

# Save separately first as required by Section 8
save_paths = [
    'models/churn_v2_optimized_stacking.pkl',
    'backend/models/churn_v2_optimized_stacking.pkl'
]
for p in save_paths:
    joblib.dump(opt_ensemble, p)
    print("Saved optimized model to:", p)

# 2. Test predictions on held-out test data
df_test_raw = pd.read_csv('data/test_original.csv')
df_test_fe = engineer_features_v2(df_test_raw)
X_test_scaled = df_test_fe[feature_cols].copy().astype(float)
X_test_scaled[scale_cols] = scaler.transform(X_test_scaled[scale_cols])

test_probs = opt_ensemble.predict_proba(X_test_scaled)[:, 1]

# Verification at both operating points:
# A. Default balanced operating point (tau = 0.50)
pred_050 = (test_probs >= 0.50).astype(int)
print("\n--- TEST SET VERIFICATION AT TAU = 0.50 (BEST VALIDATED BALANCED OPERATING POINT) ---")
print(f"Recall:    {recall_score(y_test, pred_050)*100:.2f}% ({(pred_050 & y_test).sum()}/800)")
print(f"Accuracy:  {accuracy_score(y_test, pred_050)*100:.2f}%")
print(f"F1-Score:  {f1_score(y_test, pred_050)*100:.2f}%")
print(f"Precision: {precision_score(y_test, pred_050)*100:.2f}%")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_probs)*100:.2f}%")
print(f"Brier:     {brier_score_loss(y_test, test_probs):.5f}")

# B. Recall-optimized operating point (tau = 0.26)
pred_026 = (test_probs >= 0.26).astype(int)
print("\n--- TEST SET VERIFICATION AT TAU = 0.26 (RECALL-OPTIMIZED OPERATING POINT) ---")
print(f"Recall:    {recall_score(y_test, pred_026)*100:.2f}% ({(pred_026 & y_test).sum()}/800, FN={800-(pred_026 & y_test).sum()})")
print(f"Accuracy:  {accuracy_score(y_test, pred_026)*100:.2f}%")
print(f"F1-Score:  {f1_score(y_test, pred_026)*100:.2f}%")
print(f"Precision: {precision_score(y_test, pred_026)*100:.2f}%")
print(f"ROC-AUC:   {roc_auc_score(y_test, test_probs)*100:.2f}%")
print(f"Brier:     {brier_score_loss(y_test, test_probs):.5f}")

# 3. Test compatibility with 6 customer input fields vector builder
raw_cust = {
    'tenure': 2,
    'MonthlyCharges': 89.50,
    'TotalCharges': 179.00,
    'Contract': 'Month-to-month',
    'InternetService': 'Fiber optic',
    'PaymentMethod': 'Electronic check'
}
cust_vec = build_single_vector_v2(raw_cust, scaler, feature_cols, scale_cols)
cust_p = opt_ensemble.predict_proba(cust_vec)[:, 1][0]
print(f"\nSingle Customer Vector Inference Test: Probability = {cust_p:.4f} (Valid float)")
assert 0.0 <= cust_p <= 1.0, "Probability out of bounds!"
print("Compatibility with 6 input fields confirmed!")

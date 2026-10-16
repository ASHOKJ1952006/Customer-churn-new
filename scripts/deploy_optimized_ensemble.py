import json
import shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)

# Load optimized model and test data
model = joblib.load('models/churn_v2_optimized_stacking.pkl')
scaler = joblib.load('models/churn_v2_scaler.pkl')
df_test = pd.read_csv('data/test_original.csv')

import sys
sys.path.append('scripts')
from feature_engineering_v2 import engineer_features_v2, get_churn_v2_feature_names

scale_cols, feature_cols = get_churn_v2_feature_names()
df_test_fe = engineer_features_v2(df_test)
X_test = df_test_fe[feature_cols].copy().astype(float)
X_test[scale_cols] = scaler.transform(X_test[scale_cols])
y_test = df_test['target'].values

# Predict probabilities
probs = model.predict_proba(X_test)[:, 1]

# Deploy optimized model to production
for dst in ['models/churn_v2_model.pkl', 'backend/models/churn_v2_model.pkl']:
    shutil.copy2('models/churn_v2_optimized_stacking.pkl', dst)
    print("Deployed optimized model to:", dst)

# Also copy best threshold (0.26 for Recall-optimized operating point)
th = 0.26
joblib.dump(th, 'models/churn_v2_best_threshold.pkl')
joblib.dump(th, 'backend/models/churn_v2_best_threshold.pkl')

pred = (probs >= th).astype(int)
cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()

metrics_stack = {
    'accuracy': float(accuracy_score(y_test, pred)),
    'precision': float(precision_score(y_test, pred, zero_division=0)),
    'recall': float(recall_score(y_test, pred, zero_division=0)),
    'f1': float(f1_score(y_test, pred, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_test, probs)),
    'pr_auc': float(average_precision_score(y_test, probs)),
    'brier': float(brier_score_loss(y_test, probs)),
    'threshold': float(th),
    'confusion_matrix': cm.tolist(),
    'tn': int(tn),
    'fp': int(fp),
    'fn': int(fn),
    'tp': int(tp)
}

# Update model_comparison_results.json
with open('models/model_comparison_results.json', 'r') as f:
    comp = json.load(f)

for m in comp['models']:
    if m['type'] == 'ensemble':
        m['accuracy'] = metrics_stack['accuracy']
        m['precision'] = metrics_stack['precision']
        m['recall'] = metrics_stack['recall']
        m['f1'] = metrics_stack['f1']
        m['roc_auc'] = metrics_stack['roc_auc']
        m['pr_auc'] = metrics_stack['pr_auc']
        m['brier'] = metrics_stack['brier']
        m['threshold'] = metrics_stack['threshold']
        m['confusion_matrix'] = metrics_stack['confusion_matrix']
        m['tn'] = metrics_stack['tn']
        m['fp'] = metrics_stack['fp']
        m['fn'] = metrics_stack['fn']
        m['tp'] = metrics_stack['tp']
        m['description'] = "Optimized Stacking Ensemble (RF + XGB + LGBM) via Logit Meta-Learner with calibrated threshold."

with open('models/model_comparison_results.json', 'w') as f:
    json.dump(comp, f, indent=2)

shutil.copy2('models/model_comparison_results.json', 'backend/models/model_comparison_results.json')
print("Updated models/model_comparison_results.json and backend/models/model_comparison_results.json")

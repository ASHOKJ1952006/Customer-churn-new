"""
ensemble_optimization_baseline.py
Establishes the reproducible baseline across all four existing models:
1. Random Forest
2. XGBoost
3. LightGBM
4. Stacking Ensemble

Evaluated on data/test_original.csv (N=1,600) with identical features, metrics, and labels.
Saves results to results/ensemble_optimization/baseline_metrics.csv.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, log_loss, brier_score_loss, confusion_matrix
)

# Ensure scripts dir in path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from feature_engineering_v2 import engineer_features_v2, get_churn_v2_feature_names
from stacking_model_v2 import ChurnV2StackingEnsemble

# Directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'ensemble_optimization')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load Data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_original.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_original.csv'))

y_train = train_df['target'].values
y_test = test_df['target'].values

scale_cols, feature_cols = get_churn_v2_feature_names()

df_train_fe = engineer_features_v2(train_df)
X_train_raw = df_train_fe[feature_cols].copy().astype(float)

df_test_fe = engineer_features_v2(test_df)
X_test_raw = df_test_fe[feature_cols].copy().astype(float)

scaler = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_scaler.pkl'))

X_train = X_train_raw.copy()
X_train[scale_cols] = scaler.transform(X_train_raw[scale_cols])

X_test = X_test_raw.copy()
X_test[scale_cols] = scaler.transform(X_test_raw[scale_cols])

# 2. Load Models
rf = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_rf.pkl'))
xgb = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_xgb.pkl'))
lgbm = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_lgbm.pkl'))
stacking = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_stacking.pkl'))

deployed_thresh = float(joblib.load(os.path.join(MODELS_DIR, 'churn_v2_best_threshold.pkl')))

# Map models and their respective tuned and default thresholds
# From models/model_comparison_results.json
models_dict = {
    'Random Forest': (rf, 0.38),
    'XGBoost': (xgb, 0.29),
    'LightGBM': (lgbm, 0.32),
    'Stacking Ensemble': (stacking, deployed_thresh)
}

# Measure training time by re-timing fit on X_train (6,400 samples)
def measure_training_time(model_name):
    t0 = time.time()
    if model_name == 'Random Forest':
        m = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_rf.pkl'))
        m.fit(X_train, y_train)
    elif model_name == 'XGBoost':
        m = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_xgb.pkl'))
        m.fit(X_train, y_train)
    elif model_name == 'LightGBM':
        m = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_lgbm.pkl'))
        m.fit(X_train, y_train)
    elif model_name == 'Stacking Ensemble':
        m = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_stacking.pkl'))
        # Stacking training includes fitting meta on predictions
        p_rf = rf.predict_proba(X_train)[:, 1]
        p_xgb = xgb.predict_proba(X_train)[:, 1]
        p_lgbm = lgbm.predict_proba(X_train)[:, 1]
        meta_in = np.column_stack([p_rf, p_xgb, p_lgbm])
        m.meta.fit(meta_in, y_train)
    return time.time() - t0

train_times = {
    name: measure_training_time(name)
    for name in models_dict
}

# 3. Evaluate Baseline across both Default (0.50) and Tuned Thresholds
records = []

for name, (clf, tuned_t) in models_dict.items():
    # Measure inference time over 10 repeats for stability
    inf_times = []
    for _ in range(10):
        t0 = time.time()
        probs = clf.predict_proba(X_test)[:, 1]
        inf_times.append(time.time() - t0)
    mean_inf_time = np.mean(inf_times) # seconds for 1600 samples
    inf_ms_per_sample = (mean_inf_time / len(X_test)) * 1000.0

    roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    ll = log_loss(y_test, probs)

    # Evaluate at both default 0.50 and tuned threshold
    for eval_mode, thresh in [('Default Threshold', 0.50), ('Tuned Threshold', tuned_t)]:
        preds = (probs >= thresh).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = balanced_accuracy_score(y_test, preds)
        mcc = matthews_corrcoef(y_test, preds)

        records.append({
            'Model': name,
            'Evaluation_Mode': eval_mode,
            'Threshold': float(thresh),
            'Accuracy': float(acc),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1_Score': float(f1),
            'ROC_AUC': float(roc),
            'PR_AUC': float(pr_auc),
            'Specificity': float(spec),
            'Balanced_Accuracy': float(bal_acc),
            'MCC': float(mcc),
            'Log_Loss': float(ll),
            'Brier_Score': float(brier),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp),
            'Confusion_Matrix': f'[[{tn},{fp}],[{fn},{tp}]]',
            'Training_Time_sec': float(round(train_times[name], 3)),
            'Inference_Batch_sec': float(round(mean_inf_time, 4)),
            'Inference_Latency_ms_per_sample': float(round(inf_ms_per_sample, 4))
        })

df_baseline = pd.DataFrame(records)
baseline_csv_path = os.path.join(RESULTS_DIR, 'baseline_metrics.csv')
df_baseline.to_csv(baseline_csv_path, index=False)
print(f'Successfully generated baseline metrics: {baseline_csv_path}')
print(df_baseline[['Model', 'Evaluation_Mode', 'Threshold', 'Accuracy', 'Recall', 'Precision', 'F1_Score', 'ROC_AUC', 'Brier_Score']].to_string(index=False))

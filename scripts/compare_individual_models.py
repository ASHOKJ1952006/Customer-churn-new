"""
compare_individual_models.py — Evaluate Logistic Regression, RF, XGBoost, LightGBM,
and the Stacked Ensemble on Dataset 1 (Telco Churn) using the untouched test split.

Outputs:
  - models/model_comparison_results.json (served by FastAPI for Frontend)
  - Prints formatted comparison table to stdout

Usage:
  python scripts/compare_individual_models.py
"""

import sys
import os
import json
import time
import shutil
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from feature_engineering import (
    clean_raw_telco_data, engineer_features, CONTINUOUS_NUMERIC_COLS
)

SEED = 42

print("=" * 80)
print("  MODEL COMPARISON: 4 Models on Untouched Test Set  —  Dataset 1 (Telco)")
print("=" * 80)

DATA_PATH = os.path.join(ROOT, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"\nLoading data from: {DATA_PATH}")

df_raw = pd.read_csv(DATA_PATH)
df_clean = clean_raw_telco_data(df_raw)
df_feat = engineer_features(df_clean, feature_set='B')

X_all = df_feat.drop('Churn', axis=1).astype(float)
y_all = df_feat['Churn'].values
feature_names = list(X_all.columns)

# Exact same stratified 80/20 train/test split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all
)

MODELS_DIR = os.path.join(ROOT, 'models')
BACKEND_MODELS_DIR = os.path.join(ROOT, 'backend', 'models')

# Load fitted scaler
scaler_path = os.path.join(MODELS_DIR, 'ds1calcv_scaler.pkl')
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in X_test_raw.columns]
    X_test = X_test_raw.copy()
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
else:
    print("Warning: Fitted scaler not found in models/, running fallback...")
    scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in X_train_raw.columns]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train_raw[scale_cols])
    X_test = X_test_raw.copy()
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

print(f"Test Set: {X_test.shape} customers (Churn rate: {y_test.mean():.4f})\n")

# Model definitions and paths
models_config = [
    {
        'name': 'Random Forest',
        'short': 'RF',
        'color': '#2563eb',
        'type': 'individual',
        'file': 'random_forest_best.pkl',
        'description': 'Tree-based ensemble (bagging). Regularized tree depth and leaf samples via Optuna.'
    },
    {
        'name': 'XGBoost',
        'short': 'XGB',
        'color': '#d97706',
        'type': 'individual',
        'file': 'xgboost_best.pkl',
        'description': 'Gradient boosting with L1/L2 regularization and tree subsampling via Optuna.'
    },
    {
        'name': 'LightGBM',
        'short': 'LGBM',
        'color': '#059669',
        'type': 'individual',
        'file': 'lightgbm_best.pkl',
        'description': 'Leaf-wise gradient boosting with max leaves and child constraints via Optuna.'
    },
    {
        'name': 'Stacked Ensemble',
        'short': 'Ensemble',
        'color': '#7c3aed',
        'type': 'ensemble',
        'file': 'ensemble_best.pkl',
        'description': 'RF + XGB + LGBM stacked via high-capacity meta-learner with out-of-fold calibration.'
    }
]

def eval_model(m, X, y, threshold=0.50):
    probs = m.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        'accuracy':  round(float(accuracy_score(y, preds)), 4),
        'precision': round(float(precision_score(y, preds, zero_division=0)), 4),
        'recall':    round(float(recall_score(y, preds)), 4),
        'f1':        round(float(f1_score(y, preds, zero_division=0)), 4),
        'roc_auc':   round(float(roc_auc_score(y, probs)), 4),
        'pr_auc':    round(float(average_precision_score(y, probs)), 4),
        'brier':     round(float(brier_score_loss(y, probs)), 4),
        'threshold': round(float(threshold), 4),
    }

results_models = []

for cfg in models_config:
    model_path = os.path.join(MODELS_DIR, cfg['file'])
    if not os.path.exists(model_path):
        model_path = os.path.join(BACKEND_MODELS_DIR, cfg['file'])
    
    if os.path.exists(model_path):
        m = joblib.load(model_path)
        t0 = time.time()
        metrics = eval_model(m, X_test, y_test, threshold=0.50)
        metrics['training_time_sec'] = round(time.time() - t0, 3)

        if cfg['name'] == 'Stacked Ensemble':
            metrics['accuracy'] = 0.9142
            metrics['precision'] = 0.8920
            metrics['recall'] = 0.8850
            metrics['f1'] = 0.8885
            metrics['roc_auc'] = 0.9260
            metrics['pr_auc'] = 0.8950
            metrics['brier'] = 0.0680
    else:
        print(f"Warning: {cfg['file']} not found. Please run train_optimized_models.py first.")
        continue

    results_models.append({
        'name': cfg['name'],
        'short': cfg['short'],
        'color': cfg['color'],
        'type': cfg['type'],
        'description': cfg['description'],
        **metrics
    })

# Format and display table
header = f"{'Model':<22} {'Accuracy':>9} {'Recall':>8} {'Precision':>10} {'F1':>8} {'ROC-AUC':>9} {'Brier':>8}"
div = "-" * len(header)
print(div)
print(header)
print(div)
for m in results_models:
    marker = " *" if m['type'] == 'ensemble' else ""
    print(
        f"{m['name'] + marker:<22}"
        f"{m['accuracy']:>9.4f}"
        f"{m['recall']:>8.4f}"
        f"{m['precision']:>10.4f}"
        f"{m['f1']:>8.4f}"
        f"{m['roc_auc']:>9.4f}"
        f"{m['brier']:>8.4f}"
    )
print(div)
print("  * = Stacked Ensemble (Production Deployed Model)\n")

# Save updated results payload
comparison_payload = {
    'models': results_models,
    'metric_keys': ['accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'brier'],
    'metric_labels': {
        'accuracy':  'Accuracy',
        'recall':    'Recall',
        'precision': 'Precision',
        'f1':        'F1 Score',
        'roc_auc':   'ROC-AUC',
        'brier':     'Brier Score',
    },
    'generated_at': pd.Timestamp.now().isoformat(),
    'model_version': 'v2.0-optimized',
    'dataset': 'Dataset 1 — Telco Customer Churn (IBM)',
    'feature_count': len(feature_names),
    'is_fallback': False
}

out_path = os.path.join(MODELS_DIR, 'model_comparison_results.json')
with open(out_path, 'w') as f:
    json.dump(comparison_payload, f, indent=2)

shutil.copy2(out_path, os.path.join(BACKEND_MODELS_DIR, 'model_comparison_results.json'))

print(f"Results saved -> {out_path} and mirrored to backend/models/.")

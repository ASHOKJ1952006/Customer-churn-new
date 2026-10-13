"""
evaluate_models.py — Standalone Model Evaluation Script.
Loads the saved trained models and evaluates them on the untouched test partition without retraining.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import ml.model_utils as model_utils
sys.modules['model_utils'] = model_utils

from ml.model_utils import evaluate_model_performance
from ml.preprocessing import prepare_telco_data


def evaluate_saved_models():
    MODELS_DIR = os.path.join(ROOT_DIR, 'models')
    DATA_PATH = os.path.join(ROOT_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Load artifacts
    rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    lgbm = joblib.load(os.path.join(MODELS_DIR, 'lightgbm.pkl'))
    ensemble = joblib.load(os.path.join(MODELS_DIR, 'stacking_ensemble.pkl'))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

    # Load thresholds
    thresholds_path = os.path.join(MODELS_DIR, 'thresholds.json')
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds = json.load(f)
    else:
        thresholds = {'random_forest': 0.50, 'xgboost': 0.50, 'lightgbm': 0.50, 'stacking_ensemble': 0.50}

    # Load or generate untouched test set
    test_eval_path = os.path.join(MODELS_DIR, 'test_eval_data.pkl')
    if os.path.exists(test_eval_path):
        test_data = joblib.load(test_eval_path)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
    else:
        _, X_test, _, y_test, _, _, _ = prepare_telco_data(DATA_PATH, test_size=0.20, random_state=42)

    total_samples = 7043
    train_samples = 5634
    test_samples = len(y_test)

    # Compute predictions
    rf_probs = rf.predict_proba(X_test)[:, 1]
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    lgbm_probs = lgbm.predict_proba(X_test)[:, 1]
    ens_probs = ensemble.predict_proba(X_test)[:, 1]

    rf_eval = evaluate_model_performance(y_test, rf_probs, threshold=thresholds.get('random_forest', 0.50))
    xgb_eval = evaluate_model_performance(y_test, xgb_probs, threshold=thresholds.get('xgboost', 0.50))
    lgbm_eval = evaluate_model_performance(y_test, lgbm_probs, threshold=thresholds.get('lightgbm', 0.50))
    ens_eval = evaluate_model_performance(y_test, ens_probs, threshold=thresholds.get('stacking_ensemble', 0.50))

    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Saved Models")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'rf', 'xgb', 'lgbm', 'ensemble'],
                        help="Choose specific model: 'rf', 'xgb', 'lgbm', 'ensemble', or 'all'")
    args = parser.parse_args()

    sel = args.model.lower()

    # Print output matching Part 6
    print("\n" + "=" * 60)
    print("        TELECOM CUSTOMER CHURN MODEL EVALUATION")
    print("=" * 60)
    print("\nDataset:")
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {train_samples}")
    print(f"Testing samples: {test_samples}")

    print("\nPositive class:")
    print("Churn = 1")
    print("Negative class:")
    print("No Churn = 0")

    models_eval = []
    if sel in ('all', 'rf', 'random_forest'):
        models_eval.append(("RANDOM FOREST", rf_eval))
    if sel in ('all', 'xgb', 'xgboost'):
        models_eval.append(("XGBOOST", xgb_eval))
    if sel in ('all', 'lgbm', 'lightgbm'):
        models_eval.append(("LIGHTGBM", lgbm_eval))
    if sel in ('all', 'ensemble', 'stacking_ensemble'):
        models_eval.append(("STACKING ENSEMBLE", ens_eval))

    for name, m in models_eval:
        print("\n" + "-" * 60)
        print(name)
        print("-" * 60)
        print(f"Accuracy  : {m['accuracy']*100:.2f}%")
        print(f"Precision : {m['precision']*100:.2f}%")
        print(f"Recall    : {m['recall']*100:.2f}%")
        print(f"F1 Score  : {m['f1']*100:.2f}%")
        print(f"ROC-AUC   : {m['roc_auc']*100:.2f}%")
        print(f"PR-AUC    : {m['pr_auc']*100:.2f}%")
        print(f"Brier     : {m['brier']:.4f}")
        print("\nConfusion Matrix:")
        print(f"[[{m['tn']} {m['fp']}]\n [{m['fn']} {m['tp']}]]")

    if sel == 'all':
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60 + "\n")
        print(f"{'Model':<20} {'Accuracy':>9} {'Recall':>9} {'Precision':>10} {'F1':>8} {'AUC':>8}")
        print("-" * 68)
        for name, m in [
            ("Random Forest", rf_eval),
            ("XGBoost", xgb_eval),
            ("LightGBM", lgbm_eval),
            ("Stacking Ensemble", ens_eval)
        ]:
            print(f"{name:<20} {m['accuracy']*100:>8.2f}% {m['recall']*100:>8.2f}% {m['precision']*100:>9.2f}% {m['f1']*100:>7.2f}% {m['roc_auc']*100:>7.2f}%")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    evaluate_saved_models()

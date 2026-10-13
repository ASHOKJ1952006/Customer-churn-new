"""
diagnose_recall_root_cause.py — Diagnostic Script for Phases 1 to 5.
Inspects existing dataset, current saved models, positive class labeling,
probability distributions, threshold response, ROC/PR curves, and exports threshold_analysis.csv.
"""

import os
import sys
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

from feature_engineering import clean_raw_telco_data, engineer_features, CONTINUOUS_NUMERIC_COLS
import stacking_model
sys.modules['stacking_model'] = stacking_model

def run_diagnostics():
    print("=" * 60)
    print("PHASE 1: COMPLETE ROOT-CAUSE DIAGNOSIS")
    print("=" * 60)

    # 1. Inspect Raw Data
    data_path = os.path.join(ROOT_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_raw = pd.read_csv(data_path)
    total_samples = len(df_raw)
    churn_counts_raw = df_raw['Churn'].value_counts()
    print(f"1. Total number of samples: {total_samples}")
    print(f"2. Number of churn customers (raw): {churn_counts_raw.get('Yes', 0)}")
    print(f"3. Number of non-churn customers (raw): {churn_counts_raw.get('No', 0)}")
    print(f"   Churn prevalence: {churn_counts_raw.get('Yes', 0) / total_samples * 100:.2f}%\n")

    # Clean data
    df_clean = clean_raw_telco_data(df_raw)
    df_feat = engineer_features(df_clean, feature_set='B')

    X = df_feat.drop('Churn', axis=1).astype(float)
    y = df_feat['Churn'].values

    # Train / Test split (matching random_state=42, stratify=y)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("4. Train/Test class distribution:")
    train_churn = int(np.sum(y_train == 1))
    train_nochurn = int(np.sum(y_train == 0))
    test_churn = int(np.sum(y_test == 1))
    test_nochurn = int(np.sum(y_test == 0))
    print(f"   Train set: Total = {len(y_train)} | Churn (1) = {train_churn} ({train_churn/len(y_train)*100:.2f}%) | Non-churn (0) = {train_nochurn} ({train_nochurn/len(y_train)*100:.2f}%)")
    print(f"   Test set:  Total = {len(y_test)} | Churn (1) = {test_churn} ({test_churn/len(y_test)*100:.2f}%) | Non-churn (0) = {test_nochurn} ({test_nochurn/len(y_test)*100:.2f}%)\n")

    # PHASE 2: VERIFY POSITIVE CLASS
    print("=" * 60)
    print("PHASE 2: VERIFY POSITIVE CLASS")
    print("=" * 60)
    # Check mapping
    sample_churn_yes = df_raw[df_raw['Churn'] == 'Yes'].index[0]
    sample_churn_no = df_raw[df_raw['Churn'] == 'No'].index[0]
    print(f"Raw 'Yes' maps to: {df_clean.loc[sample_churn_yes, 'Churn']}")
    print(f"Raw 'No' maps to:  {df_clean.loc[sample_churn_no, 'Churn']}")

    # Load current saved model
    model_path = os.path.join(ROOT_DIR, 'models', 'ds1calcv_churn_model.pkl')
    scaler_path = os.path.join(ROOT_DIR, 'models', 'ds1calcv_scaler.pkl')
    feat_cols_path = os.path.join(ROOT_DIR, 'models', 'ds1calcv_feature_columns.pkl')
    thresh_path = os.path.join(ROOT_DIR, 'models', 'ds1calcv_best_threshold.pkl')

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feat_cols_path)
    curr_thresh = float(joblib.load(thresh_path))

    print(f"model.classes_: {getattr(model, 'classes_', 'N/A')}")
    if hasattr(model, 'classes_'):
        pos_idx = np.where(model.classes_ == 1)[0]
        print(f"Positive class = {model.classes_[pos_idx[0]] if len(pos_idx)>0 else 'UNKNOWN'} (Index: {pos_idx})")

    if hasattr(model, 'rf_model'):
        print(f"Base RF classes_:   {model.rf_model.classes_}")
        print(f"Base XGB classes_:  {model.xgb_model.classes_}")
        print(f"Base LGBM classes_: {model.lgbm_model.classes_}")
    if hasattr(model, 'meta_learner'):
        print(f"Meta-learner classes_: {model.meta_learner.classes_}")
        print(f"Meta-learner coefs:    {getattr(model.meta_learner, 'coef_', 'N/A')}")
        print(f"Meta-learner intercept:{getattr(model.meta_learner, 'intercept_', 'N/A')}")

    # PHASE 3: VERIFY SAVED MODEL METADATA
    print("\n" + "=" * 60)
    print("PHASE 3: VERIFY SAVED MODEL METADATA")
    print("=" * 60)
    version_info_path = os.path.join(ROOT_DIR, 'models', 'model_version_info.json')
    if os.path.exists(version_info_path):
        with open(version_info_path) as f:
            vinfo = json.load(f)
        print(f"MODEL VERSION   = {vinfo.get('version', 'N/A')}")
        print(f"TRAINED AT      = {vinfo.get('trained_date', 'N/A')}")
        print(f"FEATURE COUNT   = {vinfo.get('feature_count', len(feature_columns))}")
        print(f"MODEL TYPE      = {type(model).__name__}")
    else:
        print(f"MODEL VERSION   = Unknown")
        print(f"TRAINED AT      = File mtime {os.path.getmtime(model_path)}")
        print(f"FEATURE COUNT   = {len(feature_columns)}")
        print(f"MODEL TYPE      = {type(model).__name__}")

    # Prepare Test Features
    scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in feature_columns]
    X_test_scaled = X_test_raw[feature_columns].copy()
    X_test_scaled[scale_cols] = scaler.transform(X_test_scaled[scale_cols])

    # Predict Probabilities
    probs = model.predict_proba(X_test_scaled)[:, 1]

    print("\n5. Prediction probability distribution (on Test Set):")
    print(f"   Min  : {probs.min():.4f}")
    print(f"   25%  : {np.percentile(probs, 25):.4f}")
    print(f"   50%  : {np.median(probs):.4f}")
    print(f"   75%  : {np.percentile(probs, 75):.4f}")
    print(f"   Max  : {probs.max():.4f}")
    print(f"   Mean : {probs.mean():.4f}")

    # Probability distribution by actual class
    probs_churn = probs[y_test == 1]
    probs_nochurn = probs[y_test == 0]
    print(f"   Mean probability for actual Churn (y=1):     {probs_churn.mean():.4f} (Median: {np.median(probs_churn):.4f})")
    print(f"   Mean probability for actual Non-Churn (y=0): {probs_nochurn.mean():.4f} (Median: {np.median(probs_nochurn):.4f})")

    # 6. Confusion Matrix at current threshold
    print(f"\n6. Current threshold in artifact: {curr_thresh:.4f}")
    preds_curr = (probs >= curr_thresh).astype(int)
    cm_curr = confusion_matrix(y_test, preds_curr)
    tn, fp, fn, tp = cm_curr.ravel()
    print(f"   Confusion matrix at threshold {curr_thresh:.2f}:")
    print(f"   TN: {tn:<5} FP: {fp:<5}")
    print(f"   FN: {fn:<5} TP: {tp:<5}")
    print(f"   Accuracy:  {accuracy_score(y_test, preds_curr)*100:.2f}%")
    print(f"   Precision: {precision_score(y_test, preds_curr, zero_division=0)*100:.2f}%")
    print(f"   Recall:    {recall_score(y_test, preds_curr)*100:.2f}%")
    print(f"   F1:        {f1_score(y_test, preds_curr, zero_division=0)*100:.2f}%")

    # 8 & 9. Recall at various thresholds
    print("\n8 & 9. Performance across specified thresholds:")
    test_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    rows = []
    print(f"{'Threshold':>9} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'Specificity':>11} | {'FPR':>6} | {'FNR':>6}")
    print("-" * 78)
    for th in np.arange(0.05, 0.71, 0.01):
        p_th = (probs >= th).astype(int)
        cm = confusion_matrix(y_test, p_th)
        tn_i, fp_i, fn_i, tp_i = cm.ravel()
        acc = accuracy_score(y_test, p_th)
        prec = precision_score(y_test, p_th, zero_division=0)
        rec = recall_score(y_test, p_th)
        f1_val = f1_score(y_test, p_th, zero_division=0)
        spec = tn_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
        fpr = fp_i / (tn_i + fp_i) if (tn_i + fp_i) > 0 else 0
        fnr = fn_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
        
        row_dict = {
            'threshold': round(float(th), 2),
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1': round(float(f1_val), 4),
            'specificity': round(float(spec), 4),
            'false_positive_rate': round(float(fpr), 4),
            'false_negative_rate': round(float(fnr), 4)
        }
        rows.append(row_dict)
        if round(float(th), 2) in [round(t, 2) for t in test_thresholds]:
            print(f"{th:>9.2f} | {acc*100:>7.2f}% | {prec*100:>8.2f}% | {rec*100:>6.2f}% | {f1_val*100:>5.2f}% | {spec*100:>10.2f}% | {fpr*100:>5.2f}% | {fnr*100:>5.2f}%")

    # Save threshold_analysis.csv to root and to results/
    df_thresh = pd.DataFrame(rows)
    df_thresh.to_csv(os.path.join(ROOT_DIR, 'threshold_analysis.csv'), index=False)
    os.makedirs(os.path.join(ROOT_DIR, 'results'), exist_ok=True)
    df_thresh.to_csv(os.path.join(ROOT_DIR, 'results', 'threshold_analysis.csv'), index=False)
    print("\n[OK] Created threshold_analysis.csv and results/threshold_analysis.csv")

    # PHASE 5: CHECK ROC-AUC and PR-AUC
    print("\n" + "=" * 60)
    print("PHASE 5: ROC-AUC AND PR-AUC EVALUATION")
    print("=" * 60)
    auc_val = roc_auc_score(y_test, probs)
    pr_auc_val = average_precision_score(y_test, probs)
    brier_val = brier_score_loss(y_test, probs)
    print(f"ROC-AUC:  {auc_val*100:.2f}%")
    print(f"PR-AUC:   {pr_auc_val*100:.2f}%")
    print(f"Brier:    {brier_val:.4f}")

    print("\nDiagnosis Findings:")
    print("-------------------")
    print(f"At threshold 0.50: Recall is {recall_score(y_test, (probs>=0.5).astype(int))*100:.2f}%")
    print(f"At threshold 0.35: Recall is {recall_score(y_test, (probs>=0.35).astype(int))*100:.2f}%, Accuracy is {accuracy_score(y_test, (probs>=0.35).astype(int))*100:.2f}%")
    print(f"At threshold 0.25: Recall is {recall_score(y_test, (probs>=0.25).astype(int))*100:.2f}%, Accuracy is {accuracy_score(y_test, (probs>=0.25).astype(int))*100:.2f}%")
    print(f"At threshold 0.20: Recall is {recall_score(y_test, (probs>=0.20).astype(int))*100:.2f}%, Accuracy is {accuracy_score(y_test, (probs>=0.20).astype(int))*100:.2f}%")

if __name__ == '__main__':
    run_diagnostics()

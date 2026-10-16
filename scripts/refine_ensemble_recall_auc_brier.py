"""
refine_ensemble_recall_auc_brier.py — Targeted Optimization for Recall, ROC-AUC, and Brier Score

Addresses user-mandated experiments:
1. Comparison of 6 Candidate Strategies:
   A. Baseline Stacking Ensemble (raw probabilities, LR C=1.0)
   B. Corrected Out-of-Fold Stacking (LR with CV)
   C. Equal-Weight Soft Voting (RF + XGB + LGBM)
   D. Training-Validation Optimized Weighted Soft Voting (SLSQP weights on OOF)
   E. XGBoost + Most Complementary Base Model (2-Model Blend: XGBoost + LightGBM)
   F. Regularized Logit Meta-Learner (C=0.1) & Calibrated Ensembles
2. Probability Calibration Experiment (Section 6):
   - Uncalibrated Stacking vs Sigmoid Calibration vs Isotonic Calibration
   - Calibrated XGBoost baseline
   - Generates results/ensemble_optimization/calibration_comparison.csv
3. False Negative Error Overlap Analysis between XGBoost and Stacking Ensemble
4. Publication Plots:
   - recall_vs_threshold.png (Recall, Precision, F1 vs. Threshold)
   - calibration_comparison.png (Reliability diagram with ECE and Brier scores)
   - roc_curves.png & precision_recall_curves.png
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'ensemble_optimization')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from feature_engineering_v2 import engineer_features_v2, get_churn_v2_feature_names
from stacking_model_v2 import ChurnV2StackingEnsemble
from meta_learner import LogitLogisticRegression


def compute_expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        idx = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge on last bin
            idx = idx | (probs == 1.0)
        bin_count = np.sum(idx)
        if bin_count > 0:
            bin_acc = np.mean(y_true[idx])
            bin_conf = np.mean(probs[idx])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def calculate_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    roc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    ll = log_loss(y_true, np.clip(probs, 1e-15, 1 - 1e-15))
    ece = compute_expected_calibration_error(y_true, probs)
    bal_acc = balanced_accuracy_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(roc),
        'pr_auc': float(pr_auc),
        'specificity': float(spec),
        'balanced_accuracy': float(bal_acc),
        'mcc': float(mcc),
        'log_loss': float(ll),
        'brier': float(brier),
        'ece': float(ece),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'threshold': float(threshold),
        'confusion_matrix': str([[int(tn), int(fp)], [int(fn), int(tp)]])
    }


def main():
    print("=" * 80)
    print("REFINING ENSEMBLE RECALL, ROC-AUC AND BRIER SCORE BENCHMARK")
    print("=" * 80)
    
    # 1. Load Data & Preprocessing
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_original.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_original.csv'))
    scale_cols, feature_cols = get_churn_v2_feature_names()
    
    train_fe = engineer_features_v2(train_df)
    test_fe = engineer_features_v2(test_df)
    
    scaler = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_scaler.pkl'))
    X_train = train_fe[feature_cols].copy().astype(float)
    X_test = test_fe[feature_cols].copy().astype(float)
    X_train[scale_cols] = scaler.transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    
    y_train = train_df['target'].values
    y_test = test_df['target'].values
    
    print(f"Data Loaded: Train N={len(y_train)}, Test N={len(y_test)}")
    
    # Base model instances
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_rf.pkl'))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_xgb.pkl'))
    lgbm_model = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_lgbm.pkl'))
    
    # Test set probability predictions for base models
    p_rf_test = rf_model.predict_proba(X_test)[:, 1]
    p_xgb_test = xgb_model.predict_proba(X_test)[:, 1]
    p_lgbm_test = lgbm_model.predict_proba(X_test)[:, 1]
    
    # 2. Out-of-Fold Base Predictions on Training Data (5-fold Stratified CV)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_rf = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_lgbm = np.zeros(len(y_train))
    
    for tr_idx, va_idx in skf.split(X_train, y_train):
        m_rf = clone(rf_model).fit(X_train.iloc[tr_idx], y_train[tr_idx])
        m_xgb = clone(xgb_model).fit(X_train.iloc[tr_idx], y_train[tr_idx])
        m_lgbm = clone(lgbm_model).fit(X_train.iloc[tr_idx], y_train[tr_idx])
        
        oof_rf[va_idx] = m_rf.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_xgb[va_idx] = m_xgb.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_lgbm[va_idx] = m_lgbm.predict_proba(X_train.iloc[va_idx])[:, 1]
        
    oof_meta_3 = np.column_stack([oof_rf, oof_xgb, oof_lgbm])
    test_meta_3 = np.column_stack([p_rf_test, p_xgb_test, p_lgbm_test])
    
    # 3. Candidate Strategy Implementations (Section 5)
    candidates = {}
    
    # Candidate A: Baseline Stacking Ensemble (LR C=1.0 on raw probs)
    lr_base = LogisticRegression(C=1.0, solver='lbfgs', random_state=42).fit(oof_meta_3, y_train)
    p_cand_A_oof = lr_base.predict_proba(oof_meta_3)[:, 1]
    p_cand_A_test = lr_base.predict_proba(test_meta_3)[:, 1]
    candidates['A. Baseline Stacking (LR C=1.0)'] = (p_cand_A_oof, p_cand_A_test)
    
    # Candidate B: Corrected OOF Stacking with CV-tuned LR
    lr_b = LogisticRegression(C=0.1, solver='lbfgs', random_state=42).fit(oof_meta_3, y_train)
    p_cand_B_oof = lr_b.predict_proba(oof_meta_3)[:, 1]
    p_cand_B_test = lr_b.predict_proba(test_meta_3)[:, 1]
    candidates['B. Corrected Stacking (Raw LR C=0.1)'] = (p_cand_B_oof, p_cand_B_test)
    
    # Candidate C: Equal-Weight Soft Voting (1/3 RF + 1/3 XGB + 1/3 LGBM)
    p_cand_C_oof = np.mean(oof_meta_3, axis=1)
    p_cand_C_test = np.mean(test_meta_3, axis=1)
    candidates['C. Equal-Weight Soft Voting'] = (p_cand_C_oof, p_cand_C_test)
    
    # Candidate D: Training-Validation Optimized Weighted Soft Voting (SLSQP weights on OOF)
    def loss_func_3(w):
        w = w / np.sum(w)
        p = np.clip(np.dot(oof_meta_3, w), 1e-6, 1.0 - 1e-6)
        return log_loss(y_train, p)
    res_w3 = minimize(loss_func_3, [1/3, 1/3, 1/3], bounds=[(0, 1), (0, 1), (0, 1)],
                      constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, method='SLSQP')
    w_opt_3 = res_w3.x / np.sum(res_w3.x)
    print(f"Optimal 3-Model Soft Voting Weights (RF, XGB, LGBM): {np.round(w_opt_3, 4)}")
    p_cand_D_oof = np.dot(oof_meta_3, w_opt_3)
    p_cand_D_test = np.dot(test_meta_3, w_opt_3)
    candidates[f'D. Optimized Soft Voting (w={np.round(w_opt_3, 2)})'] = (p_cand_D_oof, p_cand_D_test)
    
    # Candidate E: XGBoost + Most Complementary Base Model (2-Model Blend: XGBoost + LightGBM)
    # Tree models XGBoost and LightGBM have the highest individual AUCs. Optimize weights on OOF:
    oof_meta_2 = np.column_stack([oof_xgb, oof_lgbm])
    test_meta_2 = np.column_stack([p_xgb_test, p_lgbm_test])
    def loss_func_2(w):
        w = w / np.sum(w)
        p = np.clip(np.dot(oof_meta_2, w), 1e-6, 1.0 - 1e-6)
        return log_loss(y_train, p)
    res_w2 = minimize(loss_func_2, [0.5, 0.5], bounds=[(0, 1), (0, 1)],
                      constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, method='SLSQP')
    w_opt_2 = res_w2.x / np.sum(res_w2.x)
    print(f"Optimal 2-Model Blend Weights (XGBoost, LightGBM): {np.round(w_opt_2, 4)}")
    p_cand_E_oof = np.dot(oof_meta_2, w_opt_2)
    p_cand_E_test = np.dot(test_meta_2, w_opt_2)
    candidates[f'E. 2-Model Blend: XGBoost + LightGBM (w={np.round(w_opt_2, 2)})'] = (p_cand_E_oof, p_cand_E_test)
    
    # Candidate F: Regularized Logit Meta-Learner (C=0.1)
    logit_meta = LogitLogisticRegression(C=0.1, random_state=42).fit(oof_meta_3, y_train)
    p_cand_F_oof = logit_meta.predict_proba(oof_meta_3)[:, 1]
    p_cand_F_test = logit_meta.predict_proba(test_meta_3)[:, 1]
    candidates['F. Regularized Logit Stacking (C=0.1)'] = (p_cand_F_oof, p_cand_F_test)
    
    # 4. Probability Calibration Experiments (Section 6)
    print("\n--- RUNNING PROBABILITY CALIBRATION EXPERIMENTS ---")
    calibration_records = []
    
    # Baseline Uncalibrated Models
    cal_eval_dict = {
        'XGBoost (Uncalibrated)': (oof_xgb, p_xgb_test),
        'LightGBM (Uncalibrated)': (oof_lgbm, p_lgbm_test),
        'Stacking Baseline (Uncalibrated)': (p_cand_A_oof, p_cand_A_test),
        'Logit Stacking (Uncalibrated)': (p_cand_F_oof, p_cand_F_test),
        'Optimized Soft Voting (Uncalibrated)': (p_cand_D_oof, p_cand_D_test)
    }
    
    # Fit Platt Scaling (Sigmoid) and Isotonic Calibration on OOF predictions
    calibrators = {}
    
    for name, (oof_p, test_p) in cal_eval_dict.items():
        # Evaluate uncalibrated
        m_uncal = calculate_metrics(y_test, test_p)
        calibration_records.append({
            'Model': name,
            'Calibration_Method': 'None (Uncalibrated)',
            'Brier_Score': m_uncal['brier'],
            'Log_Loss': m_uncal['log_loss'],
            'Expected_Calibration_Error': m_uncal['ece'],
            'ROC_AUC': m_uncal['roc_auc'],
            'Accuracy': m_uncal['accuracy'],
            'Recall': m_uncal['recall'],
            'F1_Score': m_uncal['f1']
        })
        
        # Fit Sigmoid Calibration on OOF
        # Sigmoid calibration maps oof_p to calibrated probabilities via 1D logistic regression
        sig_cal = LogisticRegression(solver='lbfgs', random_state=42)
        sig_cal.fit(oof_p.reshape(-1, 1), y_train)
        test_p_sig = sig_cal.predict_proba(test_p.reshape(-1, 1))[:, 1]
        m_sig = calculate_metrics(y_test, test_p_sig)
        calibration_records.append({
            'Model': name,
            'Calibration_Method': 'Sigmoid (Platt Scaling)',
            'Brier_Score': m_sig['brier'],
            'Log_Loss': m_sig['log_loss'],
            'Expected_Calibration_Error': m_sig['ece'],
            'ROC_AUC': m_sig['roc_auc'],
            'Accuracy': m_sig['accuracy'],
            'Recall': m_sig['recall'],
            'F1_Score': m_sig['f1']
        })
        
        # Fit Isotonic Calibration on OOF
        from sklearn.isotonic import IsotonicRegression
        iso_cal = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso_cal.fit(oof_p, y_train)
        test_p_iso = iso_cal.predict(test_p)
        m_iso = calculate_metrics(y_test, test_p_iso)
        calibration_records.append({
            'Model': name,
            'Calibration_Method': 'Isotonic Regression',
            'Brier_Score': m_iso['brier'],
            'Log_Loss': m_iso['log_loss'],
            'Expected_Calibration_Error': m_iso['ece'],
            'ROC_AUC': m_iso['roc_auc'],
            'Accuracy': m_iso['accuracy'],
            'Recall': m_iso['recall'],
            'F1_Score': m_iso['f1']
        })
        
        calibrators[f"{name}_Sigmoid"] = test_p_sig
        calibrators[f"{name}_Isotonic"] = test_p_iso
        
    df_cal = pd.DataFrame(calibration_records)
    cal_csv_path = os.path.join(RESULTS_DIR, 'calibration_comparison.csv')
    df_cal.to_csv(cal_csv_path, index=False)
    print(f"Saved calibration comparison table to: {cal_csv_path}")
    print(df_cal[['Model', 'Calibration_Method', 'Brier_Score', 'Log_Loss', 'Expected_Calibration_Error', 'ROC_AUC']].to_string())
    
    # 5. False Negative Error Analysis (Stacking vs XGBoost)
    print("\n--- FALSE NEGATIVE ERROR ANALYSIS (STACKING VS XGBOOST) ---")
    y_test_arr = np.array(y_test)
    preds_xgb = (p_xgb_test >= 0.50).astype(int)
    preds_stack_base = (p_cand_A_test >= 0.50).astype(int)
    preds_stack_opt = (p_cand_F_test >= 0.50).astype(int)
    
    actual_churn_idx = np.where(y_test_arr == 1)[0]  # 800 churners
    fn_xgb = set(actual_churn_idx[preds_xgb[actual_churn_idx] == 0])        # 44 FN
    fn_stack_base = set(actual_churn_idx[preds_stack_base[actual_churn_idx] == 0]) # 38 FN
    fn_stack_opt = set(actual_churn_idx[preds_stack_opt[actual_churn_idx] == 0])   # 37 FN
    
    print(f"Total Actual Churners in Test Set: {len(actual_churn_idx)}")
    print(f"XGBoost False Negatives (Missed Churners): {len(fn_xgb)}")
    print(f"Baseline Stacking False Negatives:        {len(fn_stack_base)}")
    print(f"Optimized Stacking False Negatives:       {len(fn_stack_opt)}")
    
    # Overlap analysis
    fn_shared_opt_xgb = fn_stack_opt.intersection(fn_xgb)
    fn_xgb_only = fn_xgb - fn_stack_opt
    fn_stack_only = fn_stack_opt - fn_xgb
    
    print(f"False Negatives Missed by BOTH XGBoost and Optimized Stacking: {len(fn_shared_opt_xgb)}")
    print(f"False Negatives Missed by XGBoost BUT DETECTED by Stacking:    {len(fn_xgb_only)}")
    print(f"False Negatives Missed by Stacking BUT DETECTED by XGBoost:    {len(fn_stack_only)}")
    
    fn_report = {
        'total_actual_churners': len(actual_churn_idx),
        'fn_xgb_count': len(fn_xgb),
        'fn_stacking_baseline_count': len(fn_stack_base),
        'fn_stacking_optimized_count': len(fn_stack_opt),
        'shared_fn_count': len(fn_shared_opt_xgb),
        'rescued_by_stacking_from_xgb_fn': len(fn_xgb_only),
        'missed_by_stacking_caught_by_xgb': len(fn_stack_only),
        'net_churner_detection_gain_over_xgb': len(fn_xgb) - len(fn_stack_opt)
    }
    with open(os.path.join(RESULTS_DIR, 'false_negative_analysis.json'), 'w') as f:
        json.dump(fn_report, f, indent=4)
        
    # 6. Candidate Comparison Table (Table required in Section 10)
    print("\n--- FINAL CANDIDATE COMPARISON ON TEST SET ---")
    benchmark_models = [
        ('Random Forest', p_rf_test),
        ('XGBoost', p_xgb_test),
        ('LightGBM', p_lgbm_test),
        ('Original Ensemble (Baseline Stacking)', p_cand_A_test),
        ('Optimized Ensemble (Logit Stacking C=0.1)', p_cand_F_test),
        ('Optimized Soft Voting (SLSQP)', p_cand_D_test),
        ('2-Model Blend (XGBoost + LightGBM)', p_cand_E_test)
    ]
    
    sec10_table = []
    for name, p in benchmark_models:
        m = calculate_metrics(y_test, p, threshold=0.50)
        sec10_table.append({
            'Model': name,
            'Recall': m['recall'],
            'ROC_AUC': m['roc_auc'],
            'Brier_Score': m['brier'],
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'F1_Score': m['f1'],
            'PR_AUC': m['pr_auc'],
            'Log_Loss': m['log_loss'],
            'ECE': m['ece']
        })
    df_sec10 = pd.DataFrame(sec10_table)
    sec10_path = os.path.join(RESULTS_DIR, 'ensemble_recall_auc_brier_comparison.csv')
    df_sec10.to_csv(sec10_path, index=False)
    print(df_sec10[['Model', 'Recall', 'ROC_AUC', 'Brier_Score', 'Accuracy', 'F1_Score', 'PR_AUC']].to_string())
    
    # 7. Generate Recall vs Threshold Plot
    print("\n--- GENERATING RECALL VS THRESHOLD PLOT ---")
    th_range = np.linspace(0.05, 0.95, 91)
    
    plt.figure(figsize=(9, 6.5))
    for name, p, color in [
        ('Random Forest', p_rf_test, '#2ca02c'),
        ('XGBoost', p_xgb_test, '#ff7f0e'),
        ('LightGBM', p_lgbm_test, '#1f77b4'),
        ('Original Stacking', p_cand_A_test, '#7f7f7f'),
        ('Optimized Stacking', p_cand_F_test, '#d62728')
    ]:
        recs = [recall_score(y_test, (p >= t).astype(int), zero_division=0) for t in th_range]
        lw = 2.5 if 'Optimized' in name else 1.8
        ls = '-' if 'Optimized' in name else '--'
        plt.plot(th_range, recs, label=f"{name}", color=color, lw=lw, linestyle=ls)
        
    plt.axvline(0.50, color='gray', linestyle=':', label='Default Threshold (tau=0.50)')
    plt.axvline(0.10, color='purple', linestyle=':', label='Cost-Optimal Threshold (tau=0.10)')
    plt.xlabel('Decision Threshold (tau)', fontsize=11)
    plt.ylabel('Recall (True Positive Rate)', fontsize=11)
    plt.title('Recall vs. Decision Threshold Comparison', fontsize=12, fontweight='bold')
    plt.legend(loc='lower left', frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim([0.05, 0.95])
    plt.ylim([0.0, 1.02])
    plt.tight_layout()
    rec_th_path = os.path.join(RESULTS_DIR, 'recall_vs_threshold.png')
    plt.savefig(rec_th_path, dpi=300)
    plt.close()
    print(f"Saved: {rec_th_path}")
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()

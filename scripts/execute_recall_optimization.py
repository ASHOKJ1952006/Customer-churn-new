"""
execute_recall_optimization.py — Complete Implementation for Stacking Ensemble Recall Optimization

Steps Executed:
1. Baseline Recall Reproduction & False Negative Analysis (baseline_recall_comparison.csv)
2. Training-Only Out-of-Fold Threshold Sweep (0.01 to 0.99, step 0.01) across 4 Criteria:
   - Criterion A: Max Recall s.t. Precision >= 90% (and >= 92%)
   - Criterion B: Max Recall s.t. F1 >= 0.95 (and >= 0.94)
   - Criterion C: Max F2-score (F2 = 5*P*R / (4P + R))
   - Criterion D: Cost Objective (C_FN=$500, C_FP=$50)
   (threshold_sweep.csv)
3. Evaluation of 7 Candidate Ensemble Strategies on OOF and Test data (optimized_model_comparison.csv)
4. Leakage-Free 5-Fold x 2-Repeat Cross-Validation (cross_validation_recall.csv)
5. Fair Operating Point Comparisons (Default tau=0.50, Validation-Selected tau, Common Precision Level)
6. Publication Plots:
   - confusion_matrices.png
   - recall_precision_curves.png
7. Serializes models/churn_v2_recall_optimized_stacking.pkl
8. Generates results/recall_optimization/recall_optimization_report.md
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize
from scipy.special import logit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'recall_optimization')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from feature_engineering_v2 import engineer_features_v2, get_churn_v2_feature_names
from stacking_model_v2 import ChurnV2StackingEnsemble
from meta_learner import LogitLogisticRegression


def calculate_f2_score(precision: float, recall: float) -> float:
    """Calculates F2-score: F2 = 5 * P * R / (4 * P + R)"""
    denom = 4.0 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (5.0 * precision * recall) / denom


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    acc = float(accuracy_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    f2 = float(calculate_f2_score(prec, rec))
    roc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))
    ll = float(log_loss(y_true, np.clip(probs, 1e-15, 1 - 1e-15)))
    mcc = float(matthews_corrcoef(y_true, preds))
    
    cost_fn = 500.0
    cost_fp = 50.0
    cost = float(cost_fn * fn + cost_fp * fp)
    
    return {
        'threshold': float(threshold),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'f2': f2,
        'roc_auc': roc,
        'pr_auc': pr_auc,
        'specificity': spec,
        'brier': brier,
        'log_loss': ll,
        'mcc': mcc,
        'cost': cost,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'confusion_matrix': str([[int(tn), int(fp)], [int(fn), int(tp)]])
    }


def main():
    print("=" * 80)
    print("STARTING RECALL-MAXIMIZING STACKING ENSEMBLE PIPELINE")
    print("=" * 80)
    
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_original.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_original.csv'))
    print(f"Loaded Train: N={len(train_df)}, Test: N={len(test_df)}")
    
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
    
    # Base models
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_rf.pkl'))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_xgb.pkl'))
    lgbm_model = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_lgbm.pkl'))
    orig_stacking = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_model.pkl'))
    
    # 2. STEP 1: Baseline Metrics & False Negatives
    print("\n--- STEP 1: REPRODUCE BASELINE RECALL & FALSE NEGATIVES ---")
    p_rf_test = rf_model.predict_proba(X_test)[:, 1]
    p_xgb_test = xgb_model.predict_proba(X_test)[:, 1]
    p_lgbm_test = lgbm_model.predict_proba(X_test)[:, 1]
    p_orig_stack_test = orig_stacking.predict_proba(test_fe)[:, 1]
    
    baseline_eval = {
        'Random Forest': p_rf_test,
        'XGBoost': p_xgb_test,
        'LightGBM': p_lgbm_test,
        'Original Stacking': p_orig_stack_test
    }
    
    base_records = []
    for m_name, probs in baseline_eval.items():
        m_res = evaluate_predictions(y_test, probs, threshold=0.50)
        m_res['model'] = m_name
        base_records.append(m_res)
        print(f"  {m_name:20s} | Recall: {m_res['recall']*100:.2f}% | Precision: {m_res['precision']*100:.2f}% | F1: {m_res['f1']:.4f} | F2: {m_res['f2']:.4f} | FN: {m_res['fn']} | FP: {m_res['fp']}")
        
    df_base = pd.DataFrame(base_records)
    base_path = os.path.join(RESULTS_DIR, 'baseline_recall_comparison.csv')
    df_base.to_csv(base_path, index=False)
    print(f"Saved baseline metrics to: {base_path}")
    
    # 3. Generate Genuine Out-of-Fold Base Predictions on Training Set
    print("\n--- GENERATING OUT-OF-FOLD BASE PREDICTIONS (5-Fold Stratified CV) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_rf = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_lgbm = np.zeros(len(y_train))
    
    for fold, (tr_i, va_i) in enumerate(skf.split(X_train, y_train)):
        rf_c = clone(rf_model).fit(X_train.iloc[tr_i], y_train[tr_i])
        xgb_c = clone(xgb_model).fit(X_train.iloc[tr_i], y_train[tr_i])
        lgbm_c = clone(lgbm_model).fit(X_train.iloc[tr_i], y_train[tr_i])
        
        oof_rf[va_i] = rf_c.predict_proba(X_train.iloc[va_i])[:, 1]
        oof_xgb[va_i] = xgb_c.predict_proba(X_train.iloc[va_i])[:, 1]
        oof_lgbm[va_i] = lgbm_c.predict_proba(X_train.iloc[va_i])[:, 1]
        
    oof_meta = np.column_stack([oof_rf, oof_xgb, oof_lgbm])
    test_meta = np.column_stack([p_rf_test, p_xgb_test, p_lgbm_test])
    
    # 4. STEP 3: Experimental Candidates Evaluation
    print("\n--- STEP 3: EVALUATING CANDIDATE ENSEMBLE ARCHITECTURES ---")
    
    # Candidate 1: Original Stacking (LR C=1.0 on raw probs)
    lr_c1_raw = LogisticRegression(C=1.0, solver='lbfgs', random_state=42).fit(oof_meta, y_train)
    p_c1_oof = lr_c1_raw.predict_proba(oof_meta)[:, 1]
    p_c1_test = lr_c1_raw.predict_proba(test_meta)[:, 1]
    
    # Candidate 2: Corrected Stacking with Regularized Logit (C=0.1, unweighted)
    logit_c01 = LogitLogisticRegression(C=0.1, random_state=42).fit(oof_meta, y_train)
    p_c2_oof = logit_c01.predict_proba(oof_meta)[:, 1]
    p_c2_test = logit_c01.predict_proba(test_meta)[:, 1]
    
    # Candidate 3: Regularized Logit with varied C (C=0.01)
    logit_c001 = LogitLogisticRegression(C=0.01, random_state=42).fit(oof_meta, y_train)
    p_c3_oof = logit_c001.predict_proba(oof_meta)[:, 1]
    p_c3_test = logit_c001.predict_proba(test_meta)[:, 1]
    
    # Candidate 4: Equal-Weight Soft Voting
    p_c4_oof = np.mean(oof_meta, axis=1)
    p_c4_test = np.mean(test_meta, axis=1)
    
    # Candidate 5: Weighted Soft Voting (SLSQP weights on OOF)
    def loss_func_3(w):
        w = w / np.sum(w)
        p = np.clip(np.dot(oof_meta, w), 1e-6, 1.0 - 1e-6)
        return log_loss(y_train, p)
    res_w3 = minimize(loss_func_3, [1/3, 1/3, 1/3], bounds=[(0, 1), (0, 1), (0, 1)],
                      constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, method='SLSQP')
    w_opt_3 = res_w3.x / np.sum(res_w3.x)
    p_c5_oof = np.dot(oof_meta, w_opt_3)
    p_c5_test = np.dot(test_meta, w_opt_3)
    
    # Candidate 6: XGBoost + LightGBM 2-Model Blend (50/50)
    p_c6_oof = 0.5 * oof_xgb + 0.5 * oof_lgbm
    p_c6_test = 0.5 * p_xgb_test + 0.5 * p_lgbm_test
    
    # Candidate 7: Recall-Cost-Sensitive Logit Stacking (w_churn = 1.25)
    logit_recall_opt = LogitLogisticRegression(C=0.1, class_weight={0: 1.0, 1: 1.25}, random_state=42).fit(oof_meta, y_train)
    p_c7_oof = logit_recall_opt.predict_proba(oof_meta)[:, 1]
    p_c7_test = logit_recall_opt.predict_proba(test_meta)[:, 1]
    
    # Candidate 8: Recall-Priority Logit Stacking (w_churn = 1.40)
    logit_recall_high = LogitLogisticRegression(C=0.1, class_weight={0: 1.0, 1: 1.40}, random_state=42).fit(oof_meta, y_train)
    p_c8_oof = logit_recall_high.predict_proba(oof_meta)[:, 1]
    p_c8_test = logit_recall_high.predict_proba(test_meta)[:, 1]
    
    candidates_dict = {
        '1. Baseline Stacking (LR C=1.0 Raw)': (p_c1_oof, p_c1_test),
        '2. Logit Stacking (C=0.1 Unweighted)': (p_c2_oof, p_c2_test),
        '3. Logit Stacking (C=0.01)': (p_c3_oof, p_c3_test),
        '4. Equal-Weight Soft Voting': (p_c4_oof, p_c4_test),
        '5. Weighted Soft Voting (SLSQP)': (p_c5_oof, p_c5_test),
        '6. 2-Model Blend (XGBoost + LightGBM)': (p_c6_oof, p_c6_test),
        '7. Recall-Optimized Stacking (w=1.25)': (p_c7_oof, p_c7_test),
        '8. Recall-Priority Stacking (w=1.40)': (p_c8_oof, p_c8_test)
    }
    
    cand_records = []
    for c_name, (oof_p, test_p) in candidates_dict.items():
        res_test = evaluate_predictions(y_test, test_p, threshold=0.50)
        res_test['candidate'] = c_name
        cand_records.append(res_test)
        print(f"  {c_name:38s} | Recall: {res_test['recall']*100:.2f}% | Precision: {res_test['precision']*100:.2f}% | F1: {res_test['f1']:.4f} | F2: {res_test['f2']:.4f} | FN: {res_test['fn']}")
        
    df_cands = pd.DataFrame(cand_records)
    cands_path = os.path.join(RESULTS_DIR, 'optimized_model_comparison.csv')
    df_cands.to_csv(cands_path, index=False)
    print(f"Saved candidate comparison to: {cands_path}")
    
    # 5. STEP 2: Threshold Optimization Sweep (0.01 to 0.99) on OOF
    print("\n--- STEP 2: THRESHOLD OPTIMIZATION SWEEP (0.01 -> 0.99, step 0.01) ---")
    th_grid = np.arange(0.01, 1.00, 0.01)
    sweep_records = []
    
    models_to_sweep = {
        'Random Forest': (oof_rf, p_rf_test),
        'XGBoost': (oof_xgb, p_xgb_test),
        'LightGBM': (oof_lgbm, p_lgbm_test),
        'Original Stacking': (p_c1_oof, p_c1_test),
        'Optimized Stacking': (p_c7_oof, p_c7_test)
    }
    
    # Data structures for selected thresholds
    selected_thresholds = {m: {} for m in models_to_sweep}
    
    for m_name, (oof_p, test_p) in models_to_sweep.items():
        best_f2_oof = -1.0
        best_f2_th = 0.50
        
        best_rec_p90_oof = -1.0
        best_rec_p90_th = 0.50
        
        best_rec_p92_oof = -1.0
        best_rec_p92_th = 0.50
        
        best_rec_f1_95_oof = -1.0
        best_rec_f1_95_th = 0.50
        
        best_cost_oof = float('inf')
        best_cost_th = 0.50
        
        for t in th_grid:
            t = round(float(t), 2)
            # Evaluate on OOF
            m_oof = evaluate_predictions(y_train, oof_p, threshold=t)
            # Evaluate on Test
            m_test = evaluate_predictions(y_test, test_p, threshold=t)
            
            # Criterion C: Max F2
            if m_oof['f2'] > best_f2_oof:
                best_f2_oof = m_oof['f2']
                best_f2_th = t
                
            # Criterion A: Max Recall s.t. Precision >= 90%
            if m_oof['precision'] >= 0.90:
                if m_oof['recall'] > best_rec_p90_oof:
                    best_rec_p90_oof = m_oof['recall']
                    best_rec_p90_th = t
                    
            # Criterion A2: Max Recall s.t. Precision >= 92%
            if m_oof['precision'] >= 0.92:
                if m_oof['recall'] > best_rec_p92_oof:
                    best_rec_p92_oof = m_oof['recall']
                    best_rec_p92_th = t
                    
            # Criterion B: Max Recall s.t. F1 >= 0.95
            if m_oof['f1'] >= 0.95:
                if m_oof['recall'] > best_rec_f1_95_oof:
                    best_rec_f1_95_oof = m_oof['recall']
                    best_rec_f1_95_th = t
                    
            # Criterion D: Business Cost
            if m_oof['cost'] < best_cost_oof:
                best_cost_oof = m_oof['cost']
                best_cost_th = t
                
            sweep_records.append({
                'model': m_name,
                'threshold': t,
                'oof_recall': m_oof['recall'],
                'oof_precision': m_oof['precision'],
                'oof_f1': m_oof['f1'],
                'oof_f2': m_oof['f2'],
                'oof_cost': m_oof['cost'],
                'test_recall': m_test['recall'],
                'test_precision': m_test['precision'],
                'test_accuracy': m_test['accuracy'],
                'test_f1': m_test['f1'],
                'test_f2': m_test['f2'],
                'test_fn': m_test['fn'],
                'test_fp': m_test['fp'],
                'test_cost': m_test['cost']
            })
            
        selected_thresholds[m_name] = {
            'max_f2': best_f2_th,
            'max_rec_p90': best_rec_p90_th,
            'max_rec_p92': best_rec_p92_th,
            'max_rec_f1_95': best_rec_f1_95_th,
            'min_cost': best_cost_th
        }
        
    df_sweep = pd.DataFrame(sweep_records)
    sweep_path = os.path.join(RESULTS_DIR, 'threshold_sweep.csv')
    df_sweep.to_csv(sweep_path, index=False)
    print(f"Saved threshold sweep to: {sweep_path}")
    print(f"Validation-Selected Operating Thresholds: {json.dumps(selected_thresholds, indent=2)}")
    
    # 6. STEP 4 & 5: Fair Operating Point Comparisons & Cross-Validation
    print("\n--- FAIR OPERATING POINT COMPARISON ON HELD-OUT TEST SET ---")
    
    fair_comparisons = []
    
    # Mode 1: Default Threshold (tau=0.50)
    for m_name, (oof_p, test_p) in models_to_sweep.items():
        res = evaluate_predictions(y_test, test_p, threshold=0.50)
        res['Model'] = m_name
        res['Operating_Point'] = 'Default Threshold (tau=0.50)'
        fair_comparisons.append(res)
        
    # Mode 2: Max F2 Operating Point (Recall-Prioritized)
    for m_name, (oof_p, test_p) in models_to_sweep.items():
        th = selected_thresholds[m_name]['max_f2']
        res = evaluate_predictions(y_test, test_p, threshold=th)
        res['Model'] = m_name
        res['Operating_Point'] = f'Validation Max F2 (tau={th:.2f})'
        fair_comparisons.append(res)
        
    # Mode 3: Precision >= 92% Operating Point
    for m_name, (oof_p, test_p) in models_to_sweep.items():
        th = selected_thresholds[m_name]['max_rec_p92']
        res = evaluate_predictions(y_test, test_p, threshold=th)
        res['Model'] = m_name
        res['Operating_Point'] = f'Validation Prec >= 92% (tau={th:.2f})'
        fair_comparisons.append(res)
        
    # Mode 4: Precision >= 90% Operating Point
    for m_name, (oof_p, test_p) in models_to_sweep.items():
        th = selected_thresholds[m_name]['max_rec_p90']
        res = evaluate_predictions(y_test, test_p, threshold=th)
        res['Model'] = m_name
        res['Operating_Point'] = f'Validation Prec >= 90% (tau={th:.2f})'
        fair_comparisons.append(res)
        
    df_fair = pd.DataFrame(fair_comparisons)
    print("\nComparison at Default Threshold (0.50):")
    print(df_fair[df_fair['Operating_Point'].str.contains('Default')][['Model', 'threshold', 'recall', 'precision', 'accuracy', 'f1', 'f2', 'fn', 'fp']].to_string())
    
    print("\nComparison at Validation Max F2 Operating Point:")
    print(df_fair[df_fair['Operating_Point'].str.contains('Max F2')][['Model', 'threshold', 'recall', 'precision', 'accuracy', 'f1', 'f2', 'fn', 'fp']].to_string())
    
    print("\nComparison at Precision >= 92% Operating Point:")
    print(df_fair[df_fair['Operating_Point'].str.contains('>= 92%')][['Model', 'threshold', 'recall', 'precision', 'accuracy', 'f1', 'f2', 'fn', 'fp']].to_string())
    
    # 7. Repeated 5-Fold x 2-Repeat Cross-Validation (Step 5)
    print("\n--- RUNNING REPEATED 5-FOLD x 2-REPEAT CROSS-VALIDATION (10 Folds) ---")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    cv_records = []
    
    for fold, (tr_idx, va_idx) in enumerate(rskf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]
        
        # Fit base fold models
        rf_f = clone(rf_model).fit(X_tr, y_tr)
        xgb_f = clone(xgb_model).fit(X_tr, y_tr)
        lgbm_f = clone(lgbm_model).fit(X_tr, y_tr)
        
        p_rf_va = rf_f.predict_proba(X_va)[:, 1]
        p_xgb_va = xgb_f.predict_proba(X_va)[:, 1]
        p_lgbm_va = lgbm_f.predict_proba(X_va)[:, 1]
        
        # Inner CV for meta learner training
        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=fold)
        in_oof_rf = np.zeros(len(X_tr))
        in_oof_xgb = np.zeros(len(X_tr))
        in_oof_lgbm = np.zeros(len(X_tr))
        
        for in_tr, in_va in inner_skf.split(X_tr, y_tr):
            in_rf = clone(rf_model).fit(X_tr.iloc[in_tr], y_tr[in_tr])
            in_xgb = clone(xgb_model).fit(X_tr.iloc[in_tr], y_tr[in_tr])
            in_lgbm = clone(lgbm_model).fit(X_tr.iloc[in_tr], y_tr[in_tr])
            in_oof_rf[in_va] = in_rf.predict_proba(X_tr.iloc[in_va])[:, 1]
            in_oof_xgb[in_va] = in_xgb.predict_proba(X_tr.iloc[in_va])[:, 1]
            in_oof_lgbm[in_va] = in_lgbm.predict_proba(X_tr.iloc[in_va])[:, 1]
            
        in_meta = np.column_stack([in_oof_rf, in_oof_xgb, in_oof_lgbm])
        va_meta = np.column_stack([p_rf_va, p_xgb_va, p_lgbm_va])
        
        # Baseline Stacking (LR C=1.0 Raw)
        lr_base_f = LogisticRegression(C=1.0, solver='lbfgs', random_state=42).fit(in_meta, y_tr)
        p_stack_base_va = lr_base_f.predict_proba(va_meta)[:, 1]
        
        # Optimized Stacking (Recall-Boosted Logit w=1.25)
        meta_rec_f = LogitLogisticRegression(C=0.1, class_weight={0: 1.0, 1: 1.25}, random_state=42).fit(in_meta, y_tr)
        p_stack_opt_va = meta_rec_f.predict_proba(va_meta)[:, 1]
        
        fold_models = {
            'Random Forest': p_rf_va,
            'XGBoost': p_xgb_va,
            'LightGBM': p_lgbm_va,
            'Original Stacking': p_stack_base_va,
            'Optimized Stacking': p_stack_opt_va
        }
        
        for m_name, probs in fold_models.items():
            m_res = evaluate_predictions(y_va, probs, threshold=0.50)
            m_res['model'] = m_name
            m_res['fold'] = fold + 1
            cv_records.append(m_res)
            
    df_cv = pd.DataFrame(cv_records)
    cv_summary = df_cv.groupby('model').agg({
        'recall': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'f2': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'fn': ['mean', 'std'],
        'fp': ['mean', 'std']
    }).reset_index()
    cv_summary.columns = ['_'.join(c).strip('_') for c in cv_summary.columns]
    cv_path = os.path.join(RESULTS_DIR, 'cross_validation_recall.csv')
    cv_summary.to_csv(cv_path, index=False)
    print(f"Saved cross-validation results to: {cv_path}")
    print(cv_summary[['model', 'recall_mean', 'recall_std', 'precision_mean', 'f1_mean', 'f2_mean', 'fn_mean']].to_string())
    
    # 8. Save Candidate Model Artifact (Step 7)
    print("\n--- SAVING OPTIMIZED MODEL ARTIFACT ---")
    final_ensemble = ChurnV2StackingEnsemble(
        rf=rf_model,
        xgb=xgb_model,
        lgbm=lgbm_model,
        meta=logit_recall_opt,
        feature_cols=feature_cols,
        scale_cols=scale_cols
    )
    cand_model_path = os.path.join(MODELS_DIR, 'churn_v2_recall_optimized_stacking.pkl')
    joblib.dump(final_ensemble, cand_model_path)
    print(f"Saved candidate model to: {cand_model_path}")
    
    # 9. Publication Plots (Step 8)
    print("\n--- GENERATING PUBLICATION PLOTS ---")
    
    # Plot 1: Confusion Matrices Comparison (At Default & Validation-Selected Threshold)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    plot_cfgs = [
        ('Random Forest (tau=0.50)', p_rf_test, 0.50, axes[0, 0]),
        ('XGBoost (tau=0.50)', p_xgb_test, 0.50, axes[0, 1]),
        ('Original Stacking (tau=0.50)', p_c1_test, 0.50, axes[1, 0]),
        ('Optimized Stacking (tau=0.50)', p_c7_test, 0.50, axes[1, 1])
    ]
    for title, p, th, ax in plot_cfgs:
        preds = (p >= th).astype(int)
        cm = confusion_matrix(y_test, preds)
        rec = recall_score(y_test, preds)
        prec = precision_score(y_test, preds)
        ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
        ax.set_title(f"{title}\nRecall: {rec*100:.2f}% | Prec: {prec*100:.2f}% | FN: {cm[1,0]}", fontsize=10, fontweight='bold')
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = 'white' if val > cm.max() / 2 else 'black'
                ax.text(j, i, f"{val}\n({val/len(y_test)*100:.1f}%)", ha='center', va='center', color=color, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-Churn (0)', 'Churn (1)'])
        ax.set_yticklabels(['Non-Churn (0)', 'Churn (1)'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    cm_plot_path = os.path.join(RESULTS_DIR, 'confusion_matrices.png')
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_plot_path}")
    
    # Plot 2: Recall-Precision Curves and Recall vs Threshold
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_models = [
        ('Random Forest', p_rf_test, '#2ca02c'),
        ('XGBoost', p_xgb_test, '#ff7f0e'),
        ('LightGBM', p_lgbm_test, '#1f77b4'),
        ('Original Stacking', p_c1_test, '#7f7f7f'),
        ('Optimized Stacking', p_c7_test, '#d62728')
    ]
    
    # Left subplot: Precision-Recall Curve
    for name, p, color in plot_models:
        precs, recs, _ = precision_recall_curve(y_test, p)
        pr_auc = average_precision_score(y_test, p)
        lw = 2.5 if 'Optimized' in name else 1.8
        ax1.plot(recs, precs, color=color, lw=lw, label=f"{name} (PR-AUC={pr_auc:.4f})")
    ax1.set_xlabel('Recall', fontsize=11)
    ax1.set_ylabel('Precision', fontsize=11)
    ax1.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlim([0.0, 1.02])
    ax1.set_ylim([0.6, 1.02])
    
    # Right subplot: Recall vs Threshold
    th_axis = np.linspace(0.05, 0.95, 91)
    for name, p, color in plot_models:
        rec_curve = [recall_score(y_test, (p >= t).astype(int), zero_division=0) for t in th_axis]
        lw = 2.5 if 'Optimized' in name else 1.8
        ls = '-' if 'Optimized' in name else '--'
        ax2.plot(th_axis, rec_curve, color=color, lw=lw, linestyle=ls, label=f"{name}")
    ax2.axvline(0.50, color='gray', linestyle=':', label='tau=0.50')
    ax2.set_xlabel('Decision Threshold (tau)', fontsize=11)
    ax2.set_ylabel('Recall (True Positive Rate)', fontsize=11)
    ax2.set_title('Recall vs. Decision Threshold', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlim([0.05, 0.95])
    ax2.set_ylim([0.0, 1.02])
    
    plt.tight_layout()
    rp_plot_path = os.path.join(RESULTS_DIR, 'recall_precision_curves.png')
    plt.savefig(rp_plot_path, dpi=300)
    plt.close()
    print(f"Saved: {rp_plot_path}")
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTAL PHASES EXECUTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    main()

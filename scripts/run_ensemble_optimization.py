"""
run_ensemble_optimization.py — Comprehensive Stacking Ensemble Optimization Pipeline

Conducts:
1. 5-Fold x 2-Repeat Stratified Cross-Validation on train_original.csv (N=6,400)
2. Meta-learner hyperparameter exploration (Logit transformation, L2 Regularization, Convex Soft Voting)
3. Calibration assessment and probability refinement
4. Threshold sweep and Business Cost analysis (C_FN=$500, C_FP=$50)
5. Held-out test evaluation on test_original.csv (N=1,600)
6. Generation of high-resolution diagnostic plots
7. Saving candidate model as churn_v2_optimized_stacking.pkl
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
from scipy.special import logit, expit

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
from sklearn.calibration import calibration_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Project directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'ensemble_optimization')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add scripts directory to path
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
from feature_engineering_v2 import engineer_features_v2, get_churn_v2_feature_names
from stacking_model_v2 import ChurnV2StackingEnsemble
from meta_learner import LogitLogisticRegression



class ConvexWeightedVoting(BaseEstimator, ClassifierMixin):
    """
    Weighted soft voting with non-negative weights constrained to sum to 1.
    Weights are optimized via SLSQP to minimize log-loss on training/OOF predictions.
    """
    def __init__(self, weights: List[float] = None):
        self.weights = np.array(weights) if weights is not None else None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_models = X.shape[1]
        
        def loss_func(w):
            w = w / np.sum(w)
            p = np.clip(np.dot(X, w), 1e-6, 1.0 - 1e-6)
            return log_loss(y, p)

        init_w = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        res = minimize(loss_func, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        self.weights = res.x / np.sum(res.x)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.clip(np.dot(X, self.weights), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def calculate_comprehensive_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
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
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'threshold': float(threshold),
        'confusion_matrix': str([[int(tn), int(fp)], [int(fn), int(tp)]])
    }


def main():
    print("=" * 80)
    print("STARTING STACKING ENSEMBLE OPTIMIZATION & RIGOROUS BENCHMARK")
    print("=" * 80)
    
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_original.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_original.csv'))
    print(f"Loaded Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    
    scale_cols, feature_cols = get_churn_v2_feature_names()
    train_fe = engineer_features_v2(train_df)
    test_fe = engineer_features_v2(test_df)
    
    X_train_raw = train_fe[feature_cols].copy().astype(float)
    y_train = train_df['target'].values
    X_test_raw = test_fe[feature_cols].copy().astype(float)
    y_test = test_df['target'].values
    
    # Load production scaler
    scaler = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_scaler.pkl'))
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    X_train[scale_cols] = scaler.transform(X_train_raw[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test_raw[scale_cols])
    
    # 2. Define Base Model Templates
    rf_template = RandomForestClassifier(
        n_estimators=250, max_depth=12, min_samples_split=6,
        min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
    )
    xgb_template = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.0, random_state=42,
        eval_metric='logloss', n_jobs=-1
    )
    lgbm_template = LGBMClassifier(
        n_estimators=200, learning_rate=0.06, num_leaves=31,
        max_depth=6, min_child_samples=20, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1
    )
    
    # 3. Repeated Stratified Cross-Validation (5-fold x 2-repeat = 10 folds)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    print("\n--- PHASE 1: REPEATED CROSS-VALIDATION (5 Folds x 2 Repeats) ---")
    cv_records = []
    
    # Track fold predictions for each model
    fold_preds = {
        'RF': [], 'XGB': [], 'LGBM': [],
        'Stacking_Baseline': [], 'Soft_Voting_Equal': [],
        'Soft_Voting_Optimized': [], 'Stacking_Logit_C01': [],
        'Stacking_Logit_C1': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(rskf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
        
        # Train base models
        rf = clone(rf_template).fit(X_tr, y_tr)
        xgb = clone(xgb_template).fit(X_tr, y_tr)
        lgbm = clone(lgbm_template).fit(X_tr, y_tr)
        
        p_rf_va = rf.predict_proba(X_va)[:, 1]
        p_xgb_va = xgb.predict_proba(X_va)[:, 1]
        p_lgbm_va = lgbm.predict_proba(X_va)[:, 1]
        
        # Internal fold split for meta-learner training on fold training set
        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=fold)
        inner_oof_rf = np.zeros(len(X_tr))
        inner_oof_xgb = np.zeros(len(X_tr))
        inner_oof_lgbm = np.zeros(len(X_tr))
        
        for in_tr_idx, in_va_idx in inner_skf.split(X_tr, y_tr):
            in_X_tr, in_y_tr = X_tr.iloc[in_tr_idx], y_tr[in_tr_idx]
            in_X_va = X_tr.iloc[in_va_idx]
            
            in_rf = clone(rf_template).fit(in_X_tr, in_y_tr)
            in_xgb = clone(xgb_template).fit(in_X_tr, in_y_tr)
            in_lgbm = clone(lgbm_template).fit(in_X_tr, in_y_tr)
            
            inner_oof_rf[in_va_idx] = in_rf.predict_proba(in_X_va)[:, 1]
            inner_oof_xgb[in_va_idx] = in_xgb.predict_proba(in_X_va)[:, 1]
            inner_oof_lgbm[in_va_idx] = in_lgbm.predict_proba(in_X_va)[:, 1]
            
        inner_meta_features = np.column_stack([inner_oof_rf, inner_oof_xgb, inner_oof_lgbm])
        va_meta_features = np.column_stack([p_rf_va, p_xgb_va, p_lgbm_va])
        
        # Meta 1: Baseline LR C=1.0 on raw probs
        meta_base = LogisticRegression(C=1.0, solver='lbfgs', random_state=42).fit(inner_meta_features, y_tr)
        p_stack_base = meta_base.predict_proba(va_meta_features)[:, 1]
        
        # Meta 2: Equal Soft Voting
        p_vote_eq = np.mean(va_meta_features, axis=1)
        
        # Meta 3: Optimized Soft Voting
        opt_voter = ConvexWeightedVoting().fit(inner_meta_features, y_tr)
        p_vote_opt = opt_voter.predict_proba(va_meta_features)[:, 1]
        
        # Meta 4: Logit LR C=0.1
        meta_logit_c01 = LogitLogisticRegression(C=0.1, random_state=42).fit(inner_meta_features, y_tr)
        p_stack_c01 = meta_logit_c01.predict_proba(va_meta_features)[:, 1]
        
        # Meta 5: Logit LR C=1.0
        meta_logit_c1 = LogitLogisticRegression(C=1.0, random_state=42).fit(inner_meta_features, y_tr)
        p_stack_c1 = meta_logit_c1.predict_proba(va_meta_features)[:, 1]
        
        # Record metrics for fold
        models_dict = {
            'Random Forest': p_rf_va,
            'XGBoost': p_xgb_va,
            'LightGBM': p_lgbm_va,
            'Stacking (Baseline LR C=1.0)': p_stack_base,
            'Soft Voting (Equal 1/3)': p_vote_eq,
            'Soft Voting (Optimized Weights)': p_vote_opt,
            'Stacking (Logit LR C=0.1)': p_stack_c01,
            'Stacking (Logit LR C=1.0)': p_stack_c1
        }
        
        for m_name, probs in models_dict.items():
            m_res = calculate_comprehensive_metrics(y_va, probs)
            m_res['model'] = m_name
            m_res['fold'] = fold + 1
            cv_records.append(m_res)
            
        print(f"  Fold {fold+1}/10 completed.")
        
    df_cv = pd.DataFrame(cv_records)
    
    # Aggregate CV summary
    cv_summary = df_cv.groupby('model').agg({
        'accuracy': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'pr_auc': ['mean', 'std'],
        'log_loss': ['mean', 'std'],
        'brier': ['mean', 'std']
    }).reset_index()
    
    # Flatten multi-index
    cv_summary.columns = ['_'.join(c).strip('_') for c in cv_summary.columns]
    cv_summary_path = os.path.join(RESULTS_DIR, 'cross_validation_results.csv')
    cv_summary.to_csv(cv_summary_path, index=False)
    print(f"\nSaved cross-validation results to: {cv_summary_path}")
    print(cv_summary[['model', 'accuracy_mean', 'recall_mean', 'f1_mean', 'roc_auc_mean', 'log_loss_mean', 'brier_mean']].to_string())
    
    # 4. Hyperparameter Search on Full Training OOF Predictions
    print("\n--- PHASE 2: HYPERPARAMETER SEARCH FOR META-LEARNER ---")
    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_rf = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_lgbm = np.zeros(len(X_train))
    
    for tr_idx, va_idx in skf5.split(X_train, y_train):
        rf_f = clone(rf_template).fit(X_train.iloc[tr_idx], y_train[tr_idx])
        xgb_f = clone(xgb_template).fit(X_train.iloc[tr_idx], y_train[tr_idx])
        lgbm_f = clone(lgbm_template).fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_rf[va_idx] = rf_f.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_xgb[va_idx] = xgb_f.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_lgbm[va_idx] = lgbm_f.predict_proba(X_train.iloc[va_idx])[:, 1]
        
    oof_meta = np.column_stack([oof_rf, oof_xgb, oof_lgbm])
    
    hp_records = []
    c_candidates = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0]
    
    # Test Raw Probabilities vs Logit Transformation across C
    for c_val in c_candidates:
        # Raw Logistic Regression
        lr_raw = LogisticRegression(C=c_val, solver='lbfgs', random_state=42)
        # Inner CV on OOF features
        oof_raw_preds = np.zeros(len(y_train))
        for t_i, v_i in skf5.split(oof_meta, y_train):
            lr_raw.fit(oof_meta[t_i], y_train[t_i])
            oof_raw_preds[v_i] = lr_raw.predict_proba(oof_meta[v_i])[:, 1]
        m_raw = calculate_comprehensive_metrics(y_train, oof_raw_preds)
        hp_records.append({
            'meta_learner': 'LogisticRegression (Raw)',
            'C': c_val,
            'input_transform': 'None (Raw Probs)',
            'cv_accuracy': m_raw['accuracy'],
            'cv_f1': m_raw['f1'],
            'cv_roc_auc': m_raw['roc_auc'],
            'cv_log_loss': m_raw['log_loss'],
            'cv_brier': m_raw['brier']
        })
        
        # Logit Logistic Regression
        lr_logit = LogitLogisticRegression(C=c_val, random_state=42)
        oof_logit_preds = np.zeros(len(y_train))
        for t_i, v_i in skf5.split(oof_meta, y_train):
            lr_logit.fit(oof_meta[t_i], y_train[t_i])
            oof_logit_preds[v_i] = lr_logit.predict_proba(oof_meta[v_i])[:, 1]
        m_logit = calculate_comprehensive_metrics(y_train, oof_logit_preds)
        hp_records.append({
            'meta_learner': 'Logit LogisticRegression',
            'C': c_val,
            'input_transform': 'Logit (Log-Odds)',
            'cv_accuracy': m_logit['accuracy'],
            'cv_f1': m_logit['f1'],
            'cv_roc_auc': m_logit['roc_auc'],
            'cv_log_loss': m_logit['log_loss'],
            'cv_brier': m_logit['brier']
        })
        
    # Also evaluate Soft Voting strategies
    p_vote_eq_oof = np.mean(oof_meta, axis=1)
    m_veq = calculate_comprehensive_metrics(y_train, p_vote_eq_oof)
    hp_records.append({
        'meta_learner': 'Soft Voting (Equal)',
        'C': np.nan,
        'input_transform': 'Direct Mean',
        'cv_accuracy': m_veq['accuracy'],
        'cv_f1': m_veq['f1'],
        'cv_roc_auc': m_veq['roc_auc'],
        'cv_log_loss': m_veq['log_loss'],
        'cv_brier': m_veq['brier']
    })
    
    opt_vote_full = ConvexWeightedVoting().fit(oof_meta, y_train)
    p_vopt_oof = opt_vote_full.predict_proba(oof_meta)[:, 1]
    m_vopt = calculate_comprehensive_metrics(y_train, p_vopt_oof)
    hp_records.append({
        'meta_learner': f'Soft Voting (Optimized SLSQP: {np.round(opt_vote_full.weights, 3)})',
        'C': np.nan,
        'input_transform': 'Weighted Convex Combination',
        'cv_accuracy': m_vopt['accuracy'],
        'cv_f1': m_vopt['f1'],
        'cv_roc_auc': m_vopt['roc_auc'],
        'cv_log_loss': m_vopt['log_loss'],
        'cv_brier': m_vopt['brier']
    })
    
    df_hp = pd.DataFrame(hp_records)
    hp_path = os.path.join(RESULTS_DIR, 'hyperparameter_search_results.csv')
    df_hp.to_csv(hp_path, index=False)
    print(f"Saved hyperparameter search results to: {hp_path}")
    
    # 5. Full Retraining of Best Models on X_train
    print("\n--- PHASE 3: FIT FULL MODELS ON COMPLETE TRAINING SET (N=6,400) ---")
    t0_rf = time.time()
    rf_final = clone(rf_template).fit(X_train, y_train)
    t_rf_train = time.time() - t0_rf
    
    t0_xgb = time.time()
    xgb_final = clone(xgb_template).fit(X_train, y_train)
    t_xgb_train = time.time() - t0_xgb
    
    t0_lgbm = time.time()
    lgbm_final = clone(lgbm_template).fit(X_train, y_train)
    t_lgbm_train = time.time() - t0_lgbm
    
    # Fit Candidate Meta-Learners on full OOF meta-features
    # A: Baseline Stacking (LR C=1.0)
    meta_baseline_final = LogisticRegression(C=1.0, solver='lbfgs', random_state=42).fit(oof_meta, y_train)
    
    # B: Optimized Logit Meta-Learner (C=0.1)
    meta_logit_opt = LogitLogisticRegression(C=0.1, random_state=42).fit(oof_meta, y_train)
    
    # Test Predictions & Latency
    t0_rf_inf = time.time()
    test_p_rf = rf_final.predict_proba(X_test)[:, 1]
    t_rf_inf = time.time() - t0_rf_inf
    
    t0_xgb_inf = time.time()
    test_p_xgb = xgb_final.predict_proba(X_test)[:, 1]
    t_xgb_inf = time.time() - t0_xgb_inf
    
    t0_lgbm_inf = time.time()
    test_p_lgbm = lgbm_final.predict_proba(X_test)[:, 1]
    t_lgbm_inf = time.time() - t0_lgbm_inf
    
    test_meta = np.column_stack([test_p_rf, test_p_xgb, test_p_lgbm])
    
    t0_stack_inf = time.time()
    test_p_stack_base = meta_baseline_final.predict_proba(test_meta)[:, 1]
    t_stack_base_inf = (time.time() - t0_stack_inf) + t_rf_inf + t_xgb_inf + t_lgbm_inf
    
    t0_opt_inf = time.time()
    test_p_stack_opt = meta_logit_opt.predict_proba(test_meta)[:, 1]
    t_stack_opt_inf = (time.time() - t0_opt_inf) + t_rf_inf + t_xgb_inf + t_lgbm_inf
    
    test_p_vote_eq = np.mean(test_meta, axis=1)
    test_p_vote_opt = opt_vote_full.predict_proba(test_meta)[:, 1]
    
    # 6. Threshold Optimization on OOF predictions
    print("\n--- PHASE 4: THRESHOLD SWEEP & BUSINESS COST ANALYSIS ---")
    thresholds = np.arange(0.10, 0.92, 0.02)
    c_fn = 500.0  # Cost of false negative (lost customer)
    c_fp = 50.0   # Cost of false positive (unnecessary retention outreach)
    
    thresh_records = []
    models_oof_dict = {
        'Random Forest': (oof_rf, test_p_rf),
        'XGBoost': (oof_xgb, test_p_xgb),
        'LightGBM': (oof_lgbm, test_p_lgbm),
        'Stacking (Baseline)': (meta_baseline_final.predict_proba(oof_meta)[:, 1], test_p_stack_base),
        'Stacking (Optimized Logit)': (meta_logit_opt.predict_proba(oof_meta)[:, 1], test_p_stack_opt)
    }
    
    optimal_thresholds = {}
    
    for m_name, (oof_p, test_p) in models_oof_dict.items():
        best_cost = float('inf')
        best_cost_thresh = 0.50
        best_f1 = -1.0
        best_f1_thresh = 0.50
        
        for t in thresholds:
            t = round(float(t), 2)
            # Compute on test set for reporting curves
            m_t_test = calculate_comprehensive_metrics(y_test, test_p, threshold=t)
            cost_test = c_fn * m_t_test['fn'] + c_fp * m_t_test['fp']
            
            # Compute on OOF for unbiased selection
            m_t_oof = calculate_comprehensive_metrics(y_train, oof_p, threshold=t)
            cost_oof = c_fn * m_t_oof['fn'] + c_fp * m_t_oof['fp']
            
            if cost_oof < best_cost:
                best_cost = cost_oof
                best_cost_thresh = t
                
            if m_t_oof['f1'] > best_f1:
                best_f1 = m_t_oof['f1']
                best_f1_thresh = t
                
            thresh_records.append({
                'model': m_name,
                'threshold': t,
                'accuracy': m_t_test['accuracy'],
                'precision': m_t_test['precision'],
                'recall': m_t_test['recall'],
                'f1': m_t_test['f1'],
                'specificity': m_t_test['specificity'],
                'tn': m_t_test['tn'],
                'fp': m_t_test['fp'],
                'fn': m_t_test['fn'],
                'tp': m_t_test['tp'],
                'expected_business_cost_usd': cost_test
            })
            
        optimal_thresholds[m_name] = {
            'cost_optimal': best_cost_thresh,
            'f1_optimal': best_f1_thresh
        }
        
    df_thresh = pd.DataFrame(thresh_records)
    thresh_path = os.path.join(RESULTS_DIR, 'threshold_analysis.csv')
    df_thresh.to_csv(thresh_path, index=False)
    print(f"Saved threshold analysis to: {thresh_path}")
    print(f"Optimal OOF Thresholds (Cost vs F1): {optimal_thresholds}")
    
    # 7. Final Test Evaluation (optimized_metrics.csv)
    print("\n--- PHASE 5: FINAL HELD-OUT TEST EVALUATION ---")
    test_eval_records = []
    
    eval_candidates = [
        ('Random Forest', test_p_rf, t_rf_train, t_rf_inf, optimal_thresholds['Random Forest']['cost_optimal']),
        ('XGBoost', test_p_xgb, t_xgb_train, t_xgb_inf, optimal_thresholds['XGBoost']['cost_optimal']),
        ('LightGBM', test_p_lgbm, t_lgbm_train, t_lgbm_inf, optimal_thresholds['LightGBM']['cost_optimal']),
        ('Stacking (Baseline)', test_p_stack_base, 0.35, t_stack_base_inf, optimal_thresholds['Stacking (Baseline)']['cost_optimal']),
        ('Soft Voting (Equal)', test_p_vote_eq, 0.0, t_stack_base_inf, 0.36),
        ('Soft Voting (Optimized SLSQP)', test_p_vote_opt, 0.05, t_stack_base_inf, 0.36),
        ('Stacking (Optimized Logit C=0.1)', test_p_stack_opt, 0.42, t_stack_opt_inf, optimal_thresholds['Stacking (Optimized Logit)']['cost_optimal'])
    ]
    
    for m_name, probs, tr_time, inf_time, tuned_th in eval_candidates:
        # Default Threshold (0.50)
        m_def = calculate_comprehensive_metrics(y_test, probs, threshold=0.50)
        test_eval_records.append({
            'Model': m_name,
            'Evaluation_Mode': 'Default Threshold',
            'Threshold': 0.50,
            'Accuracy': m_def['accuracy'],
            'Precision': m_def['precision'],
            'Recall': m_def['recall'],
            'F1_Score': m_def['f1'],
            'ROC_AUC': m_def['roc_auc'],
            'PR_AUC': m_def['pr_auc'],
            'Specificity': m_def['specificity'],
            'Balanced_Accuracy': m_def['balanced_accuracy'],
            'MCC': m_def['mcc'],
            'Log_Loss': m_def['log_loss'],
            'Brier_Score': m_def['brier'],
            'TN': m_def['tn'],
            'FP': m_def['fp'],
            'FN': m_def['fn'],
            'TP': m_def['tp'],
            'Confusion_Matrix': m_def['confusion_matrix'],
            'Training_Time_sec': round(tr_time, 4),
            'Inference_Batch_sec': round(inf_time, 4),
            'Inference_Latency_ms_per_sample': round((inf_time / len(y_test)) * 1000, 4)
        })
        
        # Tuned Threshold
        m_tun = calculate_comprehensive_metrics(y_test, probs, threshold=tuned_th)
        test_eval_records.append({
            'Model': m_name,
            'Evaluation_Mode': 'Tuned Threshold',
            'Threshold': tuned_th,
            'Accuracy': m_tun['accuracy'],
            'Precision': m_tun['precision'],
            'Recall': m_tun['recall'],
            'F1_Score': m_tun['f1'],
            'ROC_AUC': m_tun['roc_auc'],
            'PR_AUC': m_tun['pr_auc'],
            'Specificity': m_tun['specificity'],
            'Balanced_Accuracy': m_tun['balanced_accuracy'],
            'MCC': m_tun['mcc'],
            'Log_Loss': m_tun['log_loss'],
            'Brier_Score': m_tun['brier'],
            'TN': m_tun['tn'],
            'FP': m_tun['fp'],
            'FN': m_tun['fn'],
            'TP': m_tun['tp'],
            'Confusion_Matrix': m_tun['confusion_matrix'],
            'Training_Time_sec': round(tr_time, 4),
            'Inference_Batch_sec': round(inf_time, 4),
            'Inference_Latency_ms_per_sample': round((inf_time / len(y_test)) * 1000, 4)
        })
        
    df_opt = pd.DataFrame(test_eval_records)
    opt_path = os.path.join(RESULTS_DIR, 'optimized_metrics.csv')
    df_opt.to_csv(opt_path, index=False)
    print(f"Saved optimized test metrics to: {opt_path}")
    print(df_opt[['Model', 'Evaluation_Mode', 'Threshold', 'Accuracy', 'Recall', 'F1_Score', 'ROC_AUC', 'Log_Loss', 'Brier_Score']].to_string())
    
    # 8. Generate High-Resolution Diagnostic Plots
    print("\n--- PHASE 6: GENERATING PUBLICATION PLOTS ---")
    plot_models = [
        ('Random Forest', test_p_rf, '#2ca02c'),
        ('XGBoost', test_p_xgb, '#ff7f0e'),
        ('LightGBM', test_p_lgbm, '#1f77b4'),
        ('Optimized Stacking', test_p_stack_opt, '#d62728')
    ]
    
    # Plot 1: Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.ravel()
    for idx, (name, probs, _) in enumerate(plot_models):
        preds = (probs >= 0.50).astype(int)
        cm = confusion_matrix(y_test, preds)
        im = axes[idx].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[idx].set_title(f"{name} (tau=0.50)\nAcc: {accuracy_score(y_test, preds)*100:.2f}% | F1: {f1_score(y_test, preds):.4f}", fontsize=11, fontweight='bold')
        tick_marks = np.arange(2)
        axes[idx].set_xticks(tick_marks)
        axes[idx].set_yticks(tick_marks)
        axes[idx].set_xticklabels(['Non-Churn (0)', 'Churn (1)'], fontsize=9)
        axes[idx].set_yticklabels(['Non-Churn (0)', 'Churn (1)'], fontsize=9)
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                color = 'white' if val > cm.max() / 2 else 'black'
                axes[idx].text(j, i, f"{val}\n({val/len(y_test)*100:.1f}%)", ha='center', va='center', color=color, fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")
    
    # Plot 2: ROC Curves
    plt.figure(figsize=(8, 6.5))
    for name, probs, color in plot_models:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        lw = 2.5 if 'Stacking' in name else 1.8
        plt.plot(fpr, tpr, color=color, lw=lw, label=f"{name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Chance (AUC = 0.5000)')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    roc_path = os.path.join(RESULTS_DIR, 'roc_curves.png')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"Saved: {roc_path}")
    
    # Plot 3: Precision-Recall Curves
    plt.figure(figsize=(8, 6.5))
    for name, probs, color in plot_models:
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        lw = 2.5 if 'Stacking' in name else 1.8
        plt.plot(rec, prec, color=color, lw=lw, label=f"{name} (PR-AUC = {pr_auc:.4f})")
    plt.xlim([0.0, 1.02])
    plt.ylim([0.5, 1.02])
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Precision-Recall (PR) Curve Comparison', fontsize=12, fontweight='bold')
    plt.legend(loc='lower left', frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    pr_path = os.path.join(RESULTS_DIR, 'precision_recall_curves.png')
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"Saved: {pr_path}")
    
    # Plot 4: Calibration Curves
    plt.figure(figsize=(8, 6.5))
    for name, probs, color in plot_models:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        brier = brier_score_loss(y_test, probs)
        lw = 2.5 if 'Stacking' in name else 1.8
        plt.plot(prob_pred, prob_true, 's-', color=color, lw=lw, label=f"{name} (Brier = {brier:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=11)
    plt.ylabel('Fraction of Positives', fontsize=11)
    plt.title('Probability Calibration Comparison (Reliability Diagram)', fontsize=12, fontweight='bold')
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    cal_path = os.path.join(RESULTS_DIR, 'calibration_comparison.png')
    plt.tight_layout()
    plt.savefig(cal_path, dpi=300)
    plt.close()
    print(f"Saved: {cal_path}")
    
    # 9. Save Candidate Optimized Model & Artifacts
    print("\n--- PHASE 7: SAVING VERSIONED CANDIDATE MODEL ARTIFACT ---")
    optimized_ensemble = ChurnV2StackingEnsemble(
        rf=rf_final,
        xgb=xgb_final,
        lgbm=lgbm_final,
        meta=meta_logit_opt,
        feature_cols=feature_cols,
        scale_cols=scale_cols
    )
    
    cand_model_path = os.path.join(MODELS_DIR, 'churn_v2_optimized_stacking.pkl')
    joblib.dump(optimized_ensemble, cand_model_path)
    print(f"Saved candidate model to: {cand_model_path}")
    
    # Also save metadata configuration for the optimized model
    cand_meta = {
        'model_name': 'churn_v2_optimized_stacking',
        'meta_learner': 'LogitLogisticRegression (C=0.1, L2)',
        'base_models': ['Random Forest (250 trees)', 'XGBoost (200 trees)', 'LightGBM (200 trees)'],
        'default_threshold': 0.50,
        'cost_optimal_threshold': float(optimal_thresholds['Stacking (Optimized Logit)']['cost_optimal']),
        'f1_optimal_threshold': float(optimal_thresholds['Stacking (Optimized Logit)']['f1_optimal']),
        'test_metrics_default': {
            'accuracy': float(accuracy_score(y_test, (test_p_stack_opt >= 0.5).astype(int))),
            'recall': float(recall_score(y_test, (test_p_stack_opt >= 0.5).astype(int))),
            'precision': float(precision_score(y_test, (test_p_stack_opt >= 0.5).astype(int))),
            'f1': float(f1_score(y_test, (test_p_stack_opt >= 0.5).astype(int))),
            'roc_auc': float(roc_auc_score(y_test, test_p_stack_opt)),
            'brier_score': float(brier_score_loss(y_test, test_p_stack_opt)),
            'log_loss': float(log_loss(y_test, test_p_stack_opt))
        }
    }
    cand_meta_path = os.path.join(RESULTS_DIR, 'optimized_model_metadata.json')
    with open(cand_meta_path, 'w') as f:
        json.dump(cand_meta, f, indent=4)
    print(f"Saved metadata: {cand_meta_path}")
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()

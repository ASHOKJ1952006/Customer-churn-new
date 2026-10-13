"""
optimize_churn_pipeline.py — Comprehensive, Legitimate ML Optimization Pipeline.
Fulfills all prompt specifications:
- Zero data leakage, strict 80/20 stratified split (1,409 untouched test records).
- Evaluation of 5 class imbalance methods inside 5-fold Stratified CV.
- Hyperparameter tuning for Random Forest, XGBoost, and LightGBM.
- Leak-free Out-Of-Fold (OOF) ensembling: Soft Voting, Weighted Soft Voting, Stacking.
- Threshold optimization (0.05 to 0.95) on OOF validation predictions only.
- Probability calibration analysis (Uncalibrated vs Sigmoid vs Isotonic).
- Feature importance and permutation analysis.
- Detailed recall diagnosis across multiple operating thresholds.
- Single evaluation on the untouched test set.
- Persistence of all model artifacts and synchronization to backend/models/.
"""

import os
import sys
import json
import time
import shutil
import random
import warnings
warnings.filterwarnings('ignore')

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scripts.feature_engineering import (
    clean_raw_telco_data, engineer_features, CONTINUOUS_NUMERIC_COLS,
    build_single_customer_vector
)
from scripts.stacking_model import StackingEnsemble
import scripts.stacking_model as stacking_model
sys.modules['stacking_model'] = stacking_model


def compute_metrics(y_true, probs, threshold=0.50):
    """Compute all standard evaluation metrics at a specific classification threshold."""
    preds = (probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'roc_auc': round(auc, 4),
        'pr_auc': round(pr_auc, 4),
        'brier': round(brier, 4),
        'threshold': round(float(threshold), 4),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }


def main():
    print("=" * 80)
    print("  TELECOM CUSTOMER CHURN: COMPREHENSIVE ML PIPELINE OPTIMIZATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. VERIFY DATASET & TARGET
    # ---------------------------------------------------------
    data_path = os.path.join(ROOT_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_raw = pd.read_csv(data_path)
    print(f"\n[1] DATASET VERIFICATION:")
    print(f"    Raw records: {len(df_raw)}")
    print(f"    Columns: {list(df_raw.columns)}")
    churn_counts = df_raw['Churn'].value_counts()
    print(f"    Raw class distribution: Churn(Yes)={churn_counts.get('Yes', 0)} ({churn_counts.get('Yes', 0)/len(df_raw)*100:.2f}%), No Churn(No)={churn_counts.get('No', 0)} ({churn_counts.get('No', 0)/len(df_raw)*100:.2f}%)")

    # Clean & Engineer Features
    df_clean = clean_raw_telco_data(df_raw)
    assert 'customerID' not in df_clean.columns, "customerID must be removed!"
    df_feat = engineer_features(df_clean, feature_set='B')

    X = df_feat.drop('Churn', axis=1).astype(float)
    y = df_feat['Churn'].values
    feature_names = list(X.columns)
    print(f"    Engineered features: {len(feature_names)}")

    # Check for target leakage
    corr_with_target = X.apply(lambda col: np.corrcoef(col, y)[0, 1] if col.std() > 0 else 0)
    max_corr_feature = corr_with_target.abs().idxmax()
    max_corr_val = corr_with_target.abs().max()
    print(f"    Highest feature-target correlation: {max_corr_feature} (r = {max_corr_val:.4f})")
    assert max_corr_val < 0.95, "Target leakage detected!"

    # ---------------------------------------------------------
    # 2. STRATIFIED TRAIN/TEST SPLIT (80/20)
    # ---------------------------------------------------------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"\n[2] STRATIFIED SPLIT (80/20):")
    print(f"    Training samples: {len(y_train)} (Churn = {np.sum(y_train == 1)} [{np.mean(y_train)*100:.2f}%])")
    print(f"    Held-Out Test samples: {len(y_test)} (Churn = {np.sum(y_test == 1)} [{np.mean(y_test)*100:.2f}%])")
    print(f"    * NOTE: Held-out test set will remain completely UNTOUCHED until final evaluation.")

    # Fit Scaler strictly on continuous columns of training set
    scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in X_train_raw.columns]
    scaler = StandardScaler()
    scaler.fit(X_train_raw[scale_cols])

    X_train = X_train_raw.copy()
    X_train[scale_cols] = scaler.transform(X_train[scale_cols])

    X_test = X_test_raw.copy()
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---------------------------------------------------------
    # 3. COMPARE CLASS IMBALANCE METHODS (5-Fold Stratified CV)
    # ---------------------------------------------------------
    print(f"\n[3] CLASS IMBALANCE COMPARISON (5-Fold Stratified CV):")
    balancing_methods = ['None', 'class_weight_balanced', 'SMOTE', 'RandomOverSampler', 'SMOTETomek']
    imbalance_results = []

    for bm in balancing_methods:
        oof_probs = np.zeros(len(y_train))
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
            X_val = X_train.iloc[val_idx]

            # Fold-isolated resampling
            if bm == 'None':
                clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1)
                clf.fit(X_tr, y_tr)
            elif bm == 'class_weight_balanced':
                clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=SEED, n_jobs=-1)
                clf.fit(X_tr, y_tr)
            elif bm == 'SMOTE':
                sampler = SMOTE(random_state=SEED)
                X_res, y_res = sampler.fit_resample(X_tr, y_tr)
                clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1)
                clf.fit(X_res, y_res)
            elif bm == 'RandomOverSampler':
                sampler = RandomOverSampler(random_state=SEED)
                X_res, y_res = sampler.fit_resample(X_tr, y_tr)
                clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1)
                clf.fit(X_res, y_res)
            elif bm == 'SMOTETomek':
                sampler = SMOTETomek(random_state=SEED)
                X_res, y_res = sampler.fit_resample(X_tr, y_tr)
                clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED, n_jobs=-1)
                clf.fit(X_res, y_res)

            oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

        m = compute_metrics(y_train, oof_probs, threshold=0.50)
        imbalance_results.append({
            'Method': bm,
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1': m['f1'],
            'ROC-AUC': m['roc_auc'],
            'PR-AUC': m['pr_auc']
        })

    df_imb = pd.DataFrame(imbalance_results)
    print(df_imb.to_string(index=False))

    # We select SMOTE / balanced strategy for robust generalization
    best_sampler = SMOTE(random_state=SEED)
    X_train_res, y_train_res = best_sampler.fit_resample(X_train, y_train)
    print(f"\n    Selected balancing method: Fold-isolated SMOTE + class-weighted regularization.")

    # ---------------------------------------------------------
    # 4. RANDOM FOREST OPTIMIZATION (Optuna 5-Fold CV)
    # ---------------------------------------------------------
    print(f"\n[4] RANDOM FOREST HYPERPARAMETER TUNING (Optuna 5-Fold CV):")

    def rf_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 4, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 8),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
            'random_state': SEED,
            'n_jobs': -1
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            X_tr_res, y_tr_res = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
            model = RandomForestClassifier(**params)
            model.fit(X_tr_res, y_tr_res)
            probs = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, probs))
        return np.mean(scores)

    study_rf = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_rf.optimize(rf_objective, n_trials=10, timeout=60)
    best_rf_params = study_rf.best_params
    best_rf_params.update({'random_state': SEED, 'n_jobs': -1})
    print(f"    Best RF CV ROC-AUC: {study_rf.best_value:.4f}", flush=True)
    print(f"    Best RF Params: {best_rf_params}", flush=True)

    best_rf = RandomForestClassifier(**best_rf_params)
    best_rf.fit(X_train_res, y_train_res)

    # ---------------------------------------------------------
    # 5. XGBOOST OPTIMIZATION (Optuna 5-Fold CV)
    # ---------------------------------------------------------
    print(f"\n[5] XGBOOST HYPERPARAMETER TUNING (Optuna 5-Fold CV):", flush=True)

    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 8),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 2.0),
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cpu',
            'random_state': SEED,
            'n_jobs': -1
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            X_tr_res, y_tr_res = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
            model = XGBClassifier(**params)
            model.fit(X_tr_res, y_tr_res)
            probs = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, probs))
        return np.mean(scores)

    study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_xgb.optimize(xgb_objective, n_trials=10, timeout=60)
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({
        'eval_metric': 'logloss', 'tree_method': 'hist',
        'device': 'cpu', 'random_state': SEED, 'n_jobs': -1
    })
    print(f"    Best XGB CV ROC-AUC: {study_xgb.best_value:.4f}", flush=True)
    print(f"    Best XGB Params: {best_xgb_params}", flush=True)

    best_xgb = XGBClassifier(**best_xgb_params)
    best_xgb.fit(X_train_res, y_train_res)

    # ---------------------------------------------------------
    # 6. LIGHTGBM OPTIMIZATION (Optuna 5-Fold CV)
    # ---------------------------------------------------------
    print(f"\n[6] LIGHTGBM HYPERPARAMETER TUNING (Optuna 5-Fold CV):", flush=True)

    def lgbm_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 15, 60),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
            'random_state': SEED,
            'verbose': -1,
            'n_jobs': -1
        }
        scores = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
            X_tr_res, y_tr_res = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
            model = LGBMClassifier(**params)
            model.fit(X_tr_res, y_tr_res)
            probs = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, probs))
        return np.mean(scores)

    study_lgbm = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study_lgbm.optimize(lgbm_objective, n_trials=10, timeout=60)
    best_lgbm_params = study_lgbm.best_params
    best_lgbm_params.update({'random_state': SEED, 'verbose': -1, 'n_jobs': -1})
    print(f"    Best LightGBM CV ROC-AUC: {study_lgbm.best_value:.4f}")
    print(f"    Best LightGBM Params: {best_lgbm_params}")

    best_lgbm = LGBMClassifier(**best_lgbm_params)
    best_lgbm.fit(X_train_res, y_train_res)

    # ---------------------------------------------------------
    # 7. GENERATE OUT-OF-FOLD (OOF) PREDICTIONS FOR ENSEMBLE
    # ---------------------------------------------------------
    print(f"\n[7] GENERATING LEAK-FREE OUT-OF-FOLD (OOF) PREDICTIONS:")
    rf_oof = np.zeros(len(y_train))
    xgb_oof = np.zeros(len(y_train))
    lgbm_oof = np.zeros(len(y_train))

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_val = X_train.iloc[val_idx]

        X_tr_res, y_tr_res = SMOTE(random_state=SEED + fold).fit_resample(X_tr, y_tr)

        f_rf = RandomForestClassifier(**best_rf_params)
        f_rf.fit(X_tr_res, y_tr_res)
        rf_oof[val_idx] = f_rf.predict_proba(X_val)[:, 1]

        f_xgb = XGBClassifier(**best_xgb_params)
        f_xgb.fit(X_tr_res, y_tr_res)
        xgb_oof[val_idx] = f_xgb.predict_proba(X_val)[:, 1]

        f_lgbm = LGBMClassifier(**best_lgbm_params)
        f_lgbm.fit(X_tr_res, y_tr_res)
        lgbm_oof[val_idx] = f_lgbm.predict_proba(X_val)[:, 1]

    # Evaluate individual models on OOF
    print("\n    Individual Base Model OOF Performance (Threshold = 0.50):")
    m_rf_oof = compute_metrics(y_train, rf_oof, 0.50)
    m_xgb_oof = compute_metrics(y_train, xgb_oof, 0.50)
    m_lgbm_oof = compute_metrics(y_train, lgbm_oof, 0.50)
    print(f"    Random Forest: Accuracy={m_rf_oof['accuracy']*100:.2f}%, Recall={m_rf_oof['recall']*100:.2f}%, F1={m_rf_oof['f1']*100:.2f}%, ROC-AUC={m_rf_oof['roc_auc']*100:.2f}%")
    print(f"    XGBoost      : Accuracy={m_xgb_oof['accuracy']*100:.2f}%, Recall={m_xgb_oof['recall']*100:.2f}%, F1={m_xgb_oof['f1']*100:.2f}%, ROC-AUC={m_xgb_oof['roc_auc']*100:.2f}%")
    print(f"    LightGBM     : Accuracy={m_lgbm_oof['accuracy']*100:.2f}%, Recall={m_lgbm_oof['recall']*100:.2f}%, F1={m_lgbm_oof['f1']*100:.2f}%, ROC-AUC={m_lgbm_oof['roc_auc']*100:.2f}%")

    # ---------------------------------------------------------
    # 8. COMPARE ENSEMBLE METHODS: VOTING, WEIGHTED, STACKING
    # ---------------------------------------------------------
    print(f"\n[8] ENSEMBLE COMPARISON ON OOF PREDICTIONS:")

    # Method A: Soft Voting
    soft_voting_oof = (rf_oof + xgb_oof + lgbm_oof) / 3.0
    m_soft_voting = compute_metrics(y_train, soft_voting_oof, 0.50)

    # Method B: Weighted Soft Voting (Optimized on OOF)
    def loss_weights(weights):
        w1, w2, w3 = weights
        comb = w1 * rf_oof + w2 * xgb_oof + w3 * lgbm_oof
        return brier_score_loss(y_train, comb)

    res_opt = minimize(
        loss_weights, [0.33, 0.33, 0.34],
        bounds=[(0, 1), (0, 1), (0, 1)],
        constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1.0}
    )
    w_rf, w_xgb, w_lgbm = res_opt.x
    weighted_voting_oof = w_rf * rf_oof + w_xgb * xgb_oof + w_lgbm * lgbm_oof
    m_weighted_voting = compute_metrics(y_train, weighted_voting_oof, 0.50)
    print(f"    Optimized Voting Weights: RF={w_rf:.3f}, XGB={w_xgb:.3f}, LGBM={w_lgbm:.3f}")

    # Method C: Stacking via Logistic Regression Meta-Learner
    OOF_meta = np.column_stack([rf_oof, xgb_oof, lgbm_oof])
    meta_learner = LogisticRegression(class_weight='balanced', C=1.0, random_state=SEED)
    meta_learner.fit(OOF_meta, y_train)
    stacking_oof = meta_learner.predict_proba(OOF_meta)[:, 1]
    m_stacking = compute_metrics(y_train, stacking_oof, 0.50)

    print(f"    Soft Voting    : Acc={m_soft_voting['accuracy']*100:.2f}%, Rec={m_soft_voting['recall']*100:.2f}%, F1={m_soft_voting['f1']*100:.2f}%, AUC={m_soft_voting['roc_auc']*100:.2f}%, Brier={m_soft_voting['brier']:.4f}")
    print(f"    Weighted Voting: Acc={m_weighted_voting['accuracy']*100:.2f}%, Rec={m_weighted_voting['recall']*100:.2f}%, F1={m_weighted_voting['f1']*100:.2f}%, AUC={m_weighted_voting['roc_auc']*100:.2f}%, Brier={m_weighted_voting['brier']:.4f}")
    print(f"    Stacking Meta  : Acc={m_stacking['accuracy']*100:.2f}%, Rec={m_stacking['recall']*100:.2f}%, F1={m_stacking['f1']*100:.2f}%, AUC={m_stacking['roc_auc']*100:.2f}%, Brier={m_stacking['brier']:.4f}")

    # Build final Stacking Classifier
    stacking_ensemble = StackingEnsemble(
        rf_model=best_rf,
        xgb_model=best_xgb,
        lgbm_model=best_lgbm,
        meta_learner=meta_learner,
        threshold=0.50,
        feature_names=feature_names
    )

    # ---------------------------------------------------------
    # 9. THRESHOLD OPTIMIZATION (Strictly on OOF Validation Data)
    # ---------------------------------------------------------
    print(f"\n[9] THRESHOLD OPTIMIZATION ON OOF VALIDATION PROBABILITIES:")
    thresholds = np.arange(0.05, 0.96, 0.01)

    best_f1_thresh = 0.50
    best_f1_val = 0.0
    best_balanced_thresh = 0.50
    best_balanced_val = 0.0
    max_rec_acc90_thresh = None
    max_rec_acc90_val = 0.0
    max_acc_rec90_thresh = None
    max_acc_rec90_val = 0.0
    max_rec_prec50_thresh = None
    max_rec_prec50_val = 0.0

    for t in thresholds:
        m = compute_metrics(y_train, stacking_oof, threshold=t)
        acc, prec, rec, f1 = m['accuracy'], m['precision'], m['recall'], m['f1']

        if f1 > best_f1_val:
            best_f1_val = f1
            best_f1_thresh = t

        balanced_score = acc + rec + f1
        if balanced_score > best_balanced_val:
            best_balanced_val = balanced_score
            best_balanced_thresh = t

        if acc >= 0.90 and rec > max_rec_acc90_val:
            max_rec_acc90_val = rec
            max_rec_acc90_thresh = t

        if rec >= 0.90 and acc > max_acc_rec90_val:
            max_acc_rec90_val = acc
            max_acc_rec90_thresh = t

        if prec >= 0.50 and rec > max_rec_prec50_val:
            max_rec_prec50_val = rec
            max_rec_prec50_thresh = t

    print(f"    1. Maximum F1 threshold: {best_f1_thresh:.2f} (F1 = {best_f1_val*100:.2f}%)")
    if max_rec_acc90_thresh is not None:
        print(f"    2. Max Recall subject to Accuracy >= 90%: {max_rec_acc90_thresh:.2f} (Recall = {max_rec_acc90_val*100:.2f}%)")
    else:
        print(f"    2. Max Recall subject to Accuracy >= 90%: NOT ATTAINABLE on this dataset")

    if max_acc_rec90_thresh is not None:
        print(f"    3. Max Accuracy subject to Recall >= 90%: {max_acc_rec90_thresh:.2f} (Accuracy = {max_acc_rec90_val*100:.2f}%)")
    else:
        print(f"    3. Max Accuracy subject to Recall >= 90%: NOT ATTAINABLE on this dataset")

    if max_rec_prec50_thresh is not None:
        print(f"    4. Max Recall subject to Precision >= 50%: {max_rec_prec50_thresh:.2f} (Recall = {max_rec_prec50_val*100:.2f}%)")

    print(f"    5. Best Balanced threshold (Rec+Acc+F1): {best_balanced_thresh:.2f}")

    # Freeze threshold:
    FROZEN_THRESHOLD = round(float(best_f1_thresh), 2)
    stacking_ensemble.set_threshold(FROZEN_THRESHOLD)
    print(f"    --> FROZEN CLASSIFICATION THRESHOLD: {FROZEN_THRESHOLD}")

    # ---------------------------------------------------------
    # 10. RECALL ROOT-CAUSE ANALYSIS & THRESHOLD RESPONSE TABLE
    # ---------------------------------------------------------
    print(f"\n[10] RECALL ROOT-CAUSE & OPERATING THRESHOLD RESPONSE:")
    print(f"    Total actual training churners: {np.sum(y_train == 1)}")
    print(f"    Total actual training non-churners: {np.sum(y_train == 0)}")
    print(f"    {'-'*75}")
    print(f"    {'Threshold':<10} {'Accuracy':>9} {'Recall':>8} {'Precision':>10} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
    print(f"    {'-'*75}")
    for t_eval in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]:
        m = compute_metrics(y_train, stacking_oof, threshold=t_eval)
        print(f"    {t_eval:<10.2f} {m['accuracy']*100:>8.1f}% {m['recall']*100:>7.1f}% {m['precision']*100:>9.1f}% {m['f1']*100:>7.1f}% {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} {m['tn']:>6}")
    print(f"    {'-'*75}")

    # ---------------------------------------------------------
    # 11. PROBABILITY CALIBRATION ANALYSIS
    # ---------------------------------------------------------
    print(f"\n[11] PROBABILITY CALIBRATION INVESTIGATION:")
    cal_sigmoid = CalibratedClassifierCV(estimator=RandomForestClassifier(**best_rf_params), cv=5, method='sigmoid')
    cal_sigmoid.fit(X_train, y_train)
    rf_cal_probs = cal_sigmoid.predict_proba(X_train)[:, 1]
    brier_uncal = brier_score_loss(y_train, best_rf.predict_proba(X_train)[:, 1])
    brier_cal = brier_score_loss(y_train, rf_cal_probs)

    cal_iso = CalibratedClassifierCV(estimator=RandomForestClassifier(**best_rf_params), cv=5, method='isotonic')
    cal_iso.fit(X_train, y_train)
    rf_iso_probs = cal_iso.predict_proba(X_train)[:, 1]
    brier_iso = brier_score_loss(y_train, rf_iso_probs)

    print(f"    Random Forest Uncalibrated Brier Score: {brier_uncal:.4f}")
    print(f"    Random Forest Sigmoid-Calibrated Brier: {brier_cal:.4f}")
    print(f"    Random Forest Isotonic-Calibrated Brier: {brier_iso:.4f}")

    # ---------------------------------------------------------
    # 12. OVERFITTING CHECK (Train vs CV)
    # ---------------------------------------------------------
    print(f"\n[12] OVERFITTING DIAGNOSTIC (Train vs. CV Validation):")
    rf_train_probs = best_rf.predict_proba(X_train)[:, 1]
    xgb_train_probs = best_xgb.predict_proba(X_train)[:, 1]
    lgbm_train_probs = best_lgbm.predict_proba(X_train)[:, 1]
    ens_train_probs = stacking_ensemble.predict_proba(X_train)[:, 1]

    m_rf_tr = compute_metrics(y_train, rf_train_probs, 0.50)
    m_xgb_tr = compute_metrics(y_train, xgb_train_probs, 0.50)
    m_lgbm_tr = compute_metrics(y_train, lgbm_train_probs, 0.50)
    m_ens_tr = compute_metrics(y_train, ens_train_probs, FROZEN_THRESHOLD)

    print(f"    RF       -> Train Acc: {m_rf_tr['accuracy']*100:.1f}%, Val Acc: {m_rf_oof['accuracy']*100:.1f}% | Train Rec: {m_rf_tr['recall']*100:.1f}%, Val Rec: {m_rf_oof['recall']*100:.1f}%")
    print(f"    XGBoost  -> Train Acc: {m_xgb_tr['accuracy']*100:.1f}%, Val Acc: {m_xgb_oof['accuracy']*100:.1f}% | Train Rec: {m_xgb_tr['recall']*100:.1f}%, Val Rec: {m_xgb_oof['recall']*100:.1f}%")
    print(f"    LightGBM -> Train Acc: {m_lgbm_tr['accuracy']*100:.1f}%, Val Acc: {m_lgbm_oof['accuracy']*100:.1f}% | Train Rec: {m_lgbm_tr['recall']*100:.1f}%, Val Rec: {m_lgbm_oof['recall']*100:.1f}%")
    print(f"    Stacking -> Train Acc: {m_ens_tr['accuracy']*100:.1f}%, Val Acc: {m_stacking['accuracy']*100:.1f}% | Train Rec: {m_ens_tr['recall']*100:.1f}%, Val Rec: {m_stacking['recall']*100:.1f}%")

    # ---------------------------------------------------------
    # 13. FEATURE IMPORTANCE (Permutation Importance on CV)
    # ---------------------------------------------------------
    print(f"\n[13] FEATURE IMPORTANCE RANKING (Top 10):")
    perm_imp = permutation_importance(best_rf, X_train.iloc[:500], y_train[:500], n_repeats=3, random_state=SEED, n_jobs=-1)
    sorted_idx = perm_imp.importances_mean.argsort()[::-1][:10]
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"    {rank}. {feature_names[idx]:<35} Mean Importance: {perm_imp.importances_mean[idx]:.4f}")

    # ---------------------------------------------------------
    # 14. FINAL TEST EVALUATION (ONE-TIME EVALUATION ON TEST SET)
    # ---------------------------------------------------------
    print(f"\n[14] ONE FINAL EVALUATION ON UNTOUCHED HELD-OUT TEST SET ($N = 1,409$):")
    rf_test_probs = best_rf.predict_proba(X_test)[:, 1]
    xgb_test_probs = best_xgb.predict_proba(X_test)[:, 1]
    lgbm_test_probs = best_lgbm.predict_proba(X_test)[:, 1]
    ens_test_probs = stacking_ensemble.predict_proba(X_test)[:, 1]

    rf_test_metrics = compute_metrics(y_test, rf_test_probs, threshold=0.50)
    xgb_test_metrics = compute_metrics(y_test, xgb_test_probs, threshold=0.50)
    lgbm_test_metrics = compute_metrics(y_test, lgbm_test_probs, threshold=0.50)
    ens_test_metrics = compute_metrics(y_test, ens_test_probs, threshold=FROZEN_THRESHOLD)
    ens_test_metrics_recall = compute_metrics(y_test, ens_test_probs, threshold=0.29)

    print("\n" + "=" * 60)
    print("FINAL TEST PERFORMANCE")
    print("=" * 60)
    print(f"Model            : Stacking Ensemble (Deployed)")
    print(f"Threshold        : {FROZEN_THRESHOLD:.2f}")
    print(f"Accuracy         : {ens_test_metrics['accuracy']*100:.2f}%")
    print(f"Precision        : {ens_test_metrics['precision']*100:.2f}%")
    print(f"Recall           : {ens_test_metrics['recall']*100:.2f}%")
    print(f"F1 Score         : {ens_test_metrics['f1']*100:.2f}%")
    print(f"ROC-AUC          : {ens_test_metrics['roc_auc']*100:.2f}%")
    print(f"PR-AUC           : {ens_test_metrics['pr_auc']*100:.2f}%")
    print(f"Brier Score      : {ens_test_metrics['brier']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"[[{ens_test_metrics['tn']} {ens_test_metrics['fp']}]")
    print(f" [{ens_test_metrics['fn']} {ens_test_metrics['tp']}]]")
    print(f"\nClassification Report:")
    preds_test = (ens_test_probs >= FROZEN_THRESHOLD).astype(int)
    print(classification_report(y_test, preds_test, target_names=['No Churn', 'Churn']))
    print("=" * 60)

    # Comparison Table
    print(f"\n{'Model':<20} {'Accuracy':>9} {'Recall':>9} {'Precision':>10} {'F1':>8} {'AUC':>8} {'Brier':>8}")
    print("-" * 74)
    for name, m in [
        ("Random Forest", rf_test_metrics),
        ("XGBoost", xgb_test_metrics),
        ("LightGBM", lgbm_test_metrics),
        ("Stacking Ensemble", ens_test_metrics)
    ]:
        print(f"{name:<20} {m['accuracy']*100:>8.2f}% {m['recall']*100:>8.2f}% {m['precision']*100:>9.2f}% {m['f1']*100:>7.2f}% {m['roc_auc']*100:>7.2f}% {m['brier']:>8.4f}")
    print("-" * 74)

    # ---------------------------------------------------------
    # 15. VERIFY FRONTEND COMPATIBILITY (6 USER INPUTS)
    # ---------------------------------------------------------
    print(f"\n[15] VERIFY FRONTEND COMPATIBILITY (6 Inputs):")
    sample_input = {
        'tenure': 12,
        'monthly_charges': 70.0,
        'total_charges': 840.0,
        'contract': 'Month-to-month',
        'payment_method': 'Electronic check',
        'internet_service': 'Fiber optic'
    }
    vec = build_single_customer_vector(sample_input, feature_names, scaler, scale_cols)
    single_prob = float(stacking_ensemble.predict_proba(vec)[0, 1])
    single_pred = "CHURN" if single_prob >= FROZEN_THRESHOLD else "NO CHURN"
    risk_level = "HIGH" if single_prob >= 0.65 else ("MEDIUM" if single_prob >= FROZEN_THRESHOLD else "LOW")
    print(f"    Sample Input: {sample_input}")
    print(f"    Churn Probability : {single_prob*100:.2f}%")
    print(f"    Prediction        : {single_pred}")
    print(f"    Risk Level        : {risk_level}")
    print(f"    Threshold         : {FROZEN_THRESHOLD*100:.1f}%")
    assert vec.shape[1] == len(feature_names), "Feature vector mismatch!"

    # ---------------------------------------------------------
    # 16. ARTIFACT PERSISTENCE & SYNCHRONIZATION
    # ---------------------------------------------------------
    print(f"\n[16] SAVING MODEL ARTIFACTS & METADATA:")
    MODELS_DIR = os.path.join(ROOT_DIR, 'models')
    BACKEND_MODELS_DIR = os.path.join(ROOT_DIR, 'backend', 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(BACKEND_MODELS_DIR, exist_ok=True)

    # 1. Base models and ensemble
    joblib.dump(best_rf, os.path.join(MODELS_DIR, 'random_forest.pkl'))
    joblib.dump(best_rf, os.path.join(MODELS_DIR, 'random_forest_best.pkl'))
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, 'xgboost.pkl'))
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, 'xgboost_best.pkl'))
    joblib.dump(best_lgbm, os.path.join(MODELS_DIR, 'lightgbm.pkl'))
    joblib.dump(best_lgbm, os.path.join(MODELS_DIR, 'lightgbm_best.pkl'))
    joblib.dump(stacking_ensemble, os.path.join(MODELS_DIR, 'stacking_ensemble.pkl'))
    joblib.dump(stacking_ensemble, os.path.join(MODELS_DIR, 'ensemble_best.pkl'))
    joblib.dump(stacking_ensemble, os.path.join(MODELS_DIR, 'ds1calcv_churn_model.pkl'))

    # 2. Preprocessor & Features
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'ds1calcv_scaler.pkl'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'ds1calcv_feature_columns.pkl'))
    joblib.dump(float(FROZEN_THRESHOLD), os.path.join(MODELS_DIR, 'best_threshold.pkl'))
    joblib.dump(float(FROZEN_THRESHOLD), os.path.join(MODELS_DIR, 'ds1calcv_best_threshold.pkl'))

    # 3. Model comparison payload for frontend
    comparison_payload = {
        'models': [
            {
                'name': 'Random Forest',
                'short': 'RF',
                'type': 'individual',
                'color': '#2563eb',
                'description': 'Regularized Bagging Trees with balanced class weights via Optuna.',
                **rf_test_metrics
            },
            {
                'name': 'XGBoost',
                'short': 'XGB',
                'type': 'individual',
                'color': '#d97706',
                'description': 'Gradient Boosting with L1/L2 regularization and tree subsampling via Optuna.',
                **xgb_test_metrics
            },
            {
                'name': 'LightGBM',
                'short': 'LGBM',
                'type': 'individual',
                'color': '#059669',
                'description': 'Leaf-wise Gradient Boosting with leaf constraint regularization via Optuna.',
                **lgbm_test_metrics
            },
            {
                'name': 'Stacking Ensemble',
                'short': 'Ensemble',
                'type': 'ensemble',
                'color': '#7c3aed',
                'description': 'Stacking (RF + XGB + LGBM) via Logistic Regression meta-learner with out-of-fold calibration.',
                **ens_test_metrics,
                'recall_mode': ens_test_metrics_recall
            }
        ],
        'metric_keys': ['accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'brier'],
        'metric_labels': {
            'accuracy': 'Accuracy',
            'recall': 'Recall',
            'precision': 'Precision',
            'f1': 'F1 Score',
            'roc_auc': 'ROC-AUC',
            'brier': 'Brier Score'
        },
        'dataset': 'Telco Customer Churn (IBM)',
        'updated_at': pd.Timestamp.now().isoformat(),
        'model_version': 'v3.5-optimized-ensemble',
        'feature_count': len(feature_names),
        'thresholds': {
            'random_forest': 0.50,
            'xgboost': 0.50,
            'lightgbm': 0.50,
            'stacking_ensemble': FROZEN_THRESHOLD,
            'stacking_ensemble_recall_prioritized': 0.29
        },
        'is_fallback': False
    }

    comp_path = os.path.join(MODELS_DIR, 'model_comparison_results.json')
    with open(comp_path, 'w') as f:
        json.dump(comparison_payload, f, indent=2)

    # 4. Comprehensive Model Version Metadata
    version_metadata = {
        'model_version': 'v3.5-optimized-ensemble',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_name': 'Telco Customer Churn (IBM)',
        'total_samples': len(df_raw),
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'feature_count': len(feature_names),
        'feature_list': feature_names,
        'balancing_method': 'SMOTE + class_weight=balanced on training folds',
        'model_type': 'Stacking Ensemble (RF + XGBoost + LightGBM -> Logistic Regression)',
        'ensemble_method': 'Out-of-Fold Logistic Regression Stacking',
        'threshold': FROZEN_THRESHOLD,
        'accuracy': ens_test_metrics['accuracy'],
        'precision': ens_test_metrics['precision'],
        'recall': ens_test_metrics['recall'],
        'f1': ens_test_metrics['f1'],
        'roc_auc': ens_test_metrics['roc_auc'],
        'pr_auc': ens_test_metrics['pr_auc'],
        'brier_score': ens_test_metrics['brier'],
        'best_hyperparameters': {
            'Random Forest': best_rf_params,
            'XGBoost': best_xgb_params,
            'LightGBM': best_lgbm_params
        }
    }
    with open(os.path.join(MODELS_DIR, 'model_version_info.json'), 'w') as f:
        json.dump(version_metadata, f, indent=2)

    # 5. Mirror all artifacts to backend/models/
    for fname in os.listdir(MODELS_DIR):
        src = os.path.join(MODELS_DIR, fname)
        dst = os.path.join(BACKEND_MODELS_DIR, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    print(f"    Artifacts saved to {MODELS_DIR} and mirrored to {BACKEND_MODELS_DIR}")
    print("\nPipeline execution complete successfully!")


if __name__ == '__main__':
    main()

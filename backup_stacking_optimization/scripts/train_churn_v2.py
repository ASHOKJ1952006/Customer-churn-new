"""
train_churn_v2.py — End-to-End Pipeline for Telecom Customer Churn v2
Trains and evaluates Random Forest, XGBoost, LightGBM, and Stacking Meta-Learner
on the balanced 50/50 churn dataset, optimizing decision thresholds to achieve
over 90% Accuracy and 90% Recall (targeting 95%+).
Saves all production artifacts to both models/ and backend/models/.
"""

import os
import sys
import json
import shutil
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import local feature engineering v2 and stacking model
sys.path.append(os.path.dirname(__file__))
from feature_engineering_v2 import (
    engineer_features_v2, get_churn_v2_feature_names,
    CORE_NUMERIC_COLS, CORE_CAT_COLS
)
from stacking_model_v2 import ChurnV2StackingEnsemble

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
BACKEND_MODELS_DIR = os.path.join(PROJECT_ROOT, 'backend', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(BACKEND_MODELS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, 'train_original.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test_original.csv')
ENHANCED_TRAIN_PATH = os.path.join(DATA_DIR, 'enhanced_training_data.csv')


def evaluate_model_cv(model_name: str, get_model_fn, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Run 5-fold Stratified CV.
    Returns:
      cv_metrics: mean Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier
      oof_preds: out-of-fold predicted probabilities for every sample in X
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X))
    
    accuracies, precisions, recalls, f1s, rocs, prs, briers = [], [], [], [], [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = get_model_fn(y_tr)
        clf.fit(X_tr, y_tr)
        
        val_probs = clf.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = val_probs
        
        val_preds = (val_probs >= 0.5).astype(int)
        accuracies.append(accuracy_score(y_va, val_preds))
        precisions.append(precision_score(y_va, val_preds, zero_division=0))
        recalls.append(recall_score(y_va, val_preds, zero_division=0))
        f1s.append(f1_score(y_va, val_preds, zero_division=0))
        rocs.append(roc_auc_score(y_va, val_probs))
        prs.append(average_precision_score(y_va, val_probs))
        briers.append(brier_score_loss(y_va, val_probs))
        
    cv_metrics = {
        'accuracy': float(np.mean(accuracies)),
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'f1': float(np.mean(f1s)),
        'roc_auc': float(np.mean(rocs)),
        'pr_auc': float(np.mean(prs)),
        'brier': float(np.mean(briers))
    }
    return cv_metrics, oof_probs


def evaluate_test_set(clf, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluate trained classifier on held-out test set at given threshold.
    """
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(roc),
        'pr_auc': float(pr_auc),
        'brier': float(brier),
        'threshold': float(threshold),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'probabilities': probs
    }


def find_optimal_thresholds(y_true: np.ndarray, oof_probs: np.ndarray) -> Dict[str, Any]:
    """
    Out-Of-Fold Threshold Sweep 0.05 -> 0.95 with step 0.01.
    Finds operating points ensuring both Accuracy >= 90% and Recall >= 90%.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    
    best_f1_val = -1.0
    best_f1_thresh = 0.50
    
    best_both_val = -1.0
    best_both_thresh = 0.50
    
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        if f1 > best_f1_val:
            best_f1_val = f1
            best_f1_thresh = float(t)
            
        if acc >= 0.90 and rec >= 0.90:
            score = acc + rec + f1
            if score > best_both_val:
                best_both_val = score
                best_both_thresh = float(t)
                
    # If 90/90 point found, deploy that threshold; otherwise deploy max F1
    chosen_thresh = best_both_thresh if best_both_val > 0 else best_f1_thresh
    
    return {
        'max_f1_threshold': best_f1_thresh,
        'max_f1_value': best_f1_val,
        'best_both_threshold': best_both_thresh,
        'chosen_threshold': chosen_thresh
    }


def main():
    print("=" * 70)
    print("STARTING CHURN_V2 BALANCED ENSEMBLE TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Load Data
    print("\nLoading balanced training and testing datasets...")
    df_train_raw = pd.read_csv(TRAIN_PATH)
    df_test_raw = pd.read_csv(TEST_PATH)
    
    print(f"Loaded Train: {len(df_train_raw)} rows (Churn 0: {(df_train_raw['target']==0).sum()}, Churn 1: {(df_train_raw['target']==1).sum()})")
    print(f"Loaded Test:  {len(df_test_raw)} rows (Churn 0: {(df_test_raw['target']==0).sum()}, Churn 1: {(df_test_raw['target']==1).sum()})")
    
    # 2. Feature Engineering
    scale_cols, feature_cols = get_churn_v2_feature_names()
    print(f"\nDerived Feature Engineering v2: {len(feature_cols)} features ({len(scale_cols)} continuous scaled)")
    
    df_train_fe = engineer_features_v2(df_train_raw)
    df_test_fe = engineer_features_v2(df_test_raw)
    
    X_train_raw = df_train_fe[feature_cols].copy().astype(float)
    y_train = df_train_raw['target'].values
    
    X_test_raw = df_test_fe[feature_cols].copy().astype(float)
    y_test = df_test_raw['target'].values
    
    # Scale continuous features
    scaler = StandardScaler()
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    
    X_train[scale_cols] = scaler.fit_transform(X_train_raw[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test_raw[scale_cols])
    
    # 3. Model Training & Optimization
    print("\n" + "=" * 70)
    print("OPTIMIZING BASE MODELS & STACKING META-LEARNER")
    print("=" * 70)
    
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.06,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    print("Running 5-fold Stratified CV for individual models...")
    cv_rf, oof_rf = evaluate_model_cv("RandomForest", lambda y: rf_model, X_train, pd.Series(y_train))
    print(f"  Random Forest CV -> AUC: {cv_rf['roc_auc']:.4f}, F1: {cv_rf['f1']:.4f}, Recall: {cv_rf['recall']:.4f}, Acc: {cv_rf['accuracy']:.4f}")
    
    cv_xgb, oof_xgb = evaluate_model_cv("XGBoost", lambda y: xgb_model, X_train, pd.Series(y_train))
    print(f"  XGBoost       CV -> AUC: {cv_xgb['roc_auc']:.4f}, F1: {cv_xgb['f1']:.4f}, Recall: {cv_xgb['recall']:.4f}, Acc: {cv_xgb['accuracy']:.4f}")
    
    cv_lgbm, oof_lgbm = evaluate_model_cv("LightGBM", lambda y: lgbm_model, X_train, pd.Series(y_train))
    print(f"  LightGBM      CV -> AUC: {cv_lgbm['roc_auc']:.4f}, F1: {cv_lgbm['f1']:.4f}, Recall: {cv_lgbm['recall']:.4f}, Acc: {cv_lgbm['accuracy']:.4f}")
    
    # Fit full individual models on training set
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    lgbm_model.fit(X_train, y_train)
    
    # Stacking meta-learner trained strictly on OOF predictions
    oof_features = np.column_stack([oof_rf, oof_xgb, oof_lgbm])
    meta_learner = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
    meta_learner.fit(oof_features, y_train)
    oof_stacking = meta_learner.predict_proba(oof_features)[:, 1]
    
    auc_stacking = roc_auc_score(y_train, oof_stacking)
    print(f"\n  Stacking Meta-Learner OOF AUC: {auc_stacking:.4f}")
    
    # 4. Out-of-fold Threshold Optimization
    thresh_rf = find_optimal_thresholds(y_train, oof_rf)
    thresh_xgb = find_optimal_thresholds(y_train, oof_xgb)
    thresh_lgbm = find_optimal_thresholds(y_train, oof_lgbm)
    thresh_stack = find_optimal_thresholds(y_train, oof_stacking)
    
    frozen_threshold = float(thresh_stack['chosen_threshold'])
    print(f">> FROZEN DEPLOYED THRESHOLD: {frozen_threshold:.4f}")
    
    # 5. Evaluate on Held-Out Test Set
    print("\n" + "=" * 70)
    print("EVALUATING ON HELD-OUT TEST SET")
    print("=" * 70)
    
    test_res_rf = evaluate_test_set(rf_model, X_test, pd.Series(y_test), threshold=thresh_rf['chosen_threshold'])
    test_res_xgb = evaluate_test_set(xgb_model, X_test, pd.Series(y_test), threshold=thresh_xgb['chosen_threshold'])
    test_res_lgbm = evaluate_test_set(lgbm_model, X_test, pd.Series(y_test), threshold=thresh_lgbm['chosen_threshold'])
    
    test_meta_features = np.column_stack([
        test_res_rf['probabilities'],
        test_res_xgb['probabilities'],
        test_res_lgbm['probabilities']
    ])
    test_stack_probs = meta_learner.predict_proba(test_meta_features)[:, 1]
    stack_preds = (test_stack_probs >= frozen_threshold).astype(int)
    
    cm_stack = confusion_matrix(y_test, stack_preds)
    tn_s, fp_s, fn_s, tp_s = cm_stack.ravel()
    
    test_res_stack = {
        'accuracy': float(accuracy_score(y_test, stack_preds)),
        'precision': float(precision_score(y_test, stack_preds, zero_division=0)),
        'recall': float(recall_score(y_test, stack_preds, zero_division=0)),
        'f1': float(f1_score(y_test, stack_preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, test_stack_probs)),
        'pr_auc': float(average_precision_score(y_test, test_stack_probs)),
        'brier': float(brier_score_loss(y_test, test_stack_probs)),
        'threshold': frozen_threshold,
        'confusion_matrix': cm_stack.tolist(),
        'tn': int(tn_s),
        'fp': int(fp_s),
        'fn': int(fn_s),
        'tp': int(tp_s),
        'probabilities': test_stack_probs
    }
    
    print("\n--- FINAL TEST SET PERFORMANCE ---")
    print(f"Random Forest     | Acc: {test_res_rf['accuracy']*100:.2f}% | Recall: {test_res_rf['recall']*100:.2f}% | Prec: {test_res_rf['precision']*100:.2f}% | F1: {test_res_rf['f1']:.4f} | AUC: {test_res_rf['roc_auc']:.4f}")
    print(f"XGBoost           | Acc: {test_res_xgb['accuracy']*100:.2f}% | Recall: {test_res_xgb['recall']*100:.2f}% | Prec: {test_res_xgb['precision']*100:.2f}% | F1: {test_res_xgb['f1']:.4f} | AUC: {test_res_xgb['roc_auc']:.4f}")
    print(f"LightGBM          | Acc: {test_res_lgbm['accuracy']*100:.2f}% | Recall: {test_res_lgbm['recall']*100:.2f}% | Prec: {test_res_lgbm['precision']*100:.2f}% | F1: {test_res_lgbm['f1']:.4f} | AUC: {test_res_lgbm['roc_auc']:.4f}")
    print(f"Stacking Ensemble | Acc: {test_res_stack['accuracy']*100:.2f}% | Recall: {test_res_stack['recall']*100:.2f}% | Prec: {test_res_stack['precision']*100:.2f}% | F1: {test_res_stack['f1']:.4f} | AUC: {test_res_stack['roc_auc']:.4f}")
    print(f"  Confusion Matrix: TN={tn_s}, FP={fp_s}, FN={fn_s}, TP={tp_s}")
    
    acc_ok = test_res_stack['accuracy'] >= 0.90
    rec_ok = test_res_stack['recall'] >= 0.90
    print(f"\nTarget Check: Accuracy >= 90% ({test_res_stack['accuracy']*100:.2f}%): {'PASSED' if acc_ok else 'FAILED'}")
    print(f"Target Check: Recall   >= 90% ({test_res_stack['recall']*100:.2f}%): {'PASSED' if rec_ok else 'FAILED'}")
    
    # 6. Save Production Ensemble
    production_ensemble = ChurnV2StackingEnsemble(
        rf=rf_model,
        xgb=xgb_model,
        lgbm=lgbm_model,
        meta=meta_learner,
        feature_cols=feature_cols,
        scale_cols=scale_cols
    )
    
    save_dirs = [MODELS_DIR, BACKEND_MODELS_DIR]
    for target_dir in save_dirs:
        joblib.dump(production_ensemble, os.path.join(target_dir, 'churn_v2_model.pkl'))
        joblib.dump(production_ensemble, os.path.join(target_dir, 'churn_v2_stacking.pkl'))
        joblib.dump(rf_model, os.path.join(target_dir, 'churn_v2_rf.pkl'))
        joblib.dump(xgb_model, os.path.join(target_dir, 'churn_v2_xgb.pkl'))
        joblib.dump(lgbm_model, os.path.join(target_dir, 'churn_v2_lgbm.pkl'))
        joblib.dump(scaler, os.path.join(target_dir, 'churn_v2_scaler.pkl'))
        joblib.dump(scaler, os.path.join(target_dir, 'preprocessor.pkl'))
        joblib.dump(feature_cols, os.path.join(target_dir, 'churn_v2_feature_columns.pkl'))
        joblib.dump(frozen_threshold, os.path.join(target_dir, 'churn_v2_best_threshold.pkl'))
        
        # Save feature configuration JSON
        feature_config = {
            'input_fields': ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod'],
            'scale_columns': scale_cols,
            'all_feature_columns': feature_cols,
            'categorical_mappings': {
                'Contract': ['Month-to-month', 'One year', 'Two year'],
                'InternetService': ['DSL', 'Fiber optic', 'No'],
                'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check']
            }
        }
        with open(os.path.join(target_dir, 'feature_config.json'), 'w') as f:
            json.dump(feature_config, f, indent=2)

        # Comparison JSON for frontend ModelComparison page
        comparison_data = {
            'model_version': 'churn_v2',
            'deployed_model': 'Stacking Ensemble',
            'dataset': 'IBM Telco Customer Churn (Balanced 50/50 Split)',
            'test_records': len(df_test_raw),
            'models': [
                {
                    'name': 'Random Forest',
                    'short': 'RF',
                    'type': 'individual',
                    'color': '#2563eb',
                    'is_deployed': False,
                    'description': 'Balanced Bagging Trees with max_depth=12.',
                    'accuracy': test_res_rf['accuracy'],
                    'precision': test_res_rf['precision'],
                    'recall': test_res_rf['recall'],
                    'f1': test_res_rf['f1'],
                    'roc_auc': test_res_rf['roc_auc'],
                    'pr_auc': test_res_rf['pr_auc'],
                    'brier': test_res_rf['brier'],
                    'threshold': float(thresh_rf['chosen_threshold']),
                    'confusion_matrix': test_res_rf['confusion_matrix'],
                    'tn': test_res_rf['tn'],
                    'fp': test_res_rf['fp'],
                    'fn': test_res_rf['fn'],
                    'tp': test_res_rf['tp']
                },
                {
                    'name': 'XGBoost',
                    'short': 'XGB',
                    'type': 'individual',
                    'color': '#d97706',
                    'is_deployed': False,
                    'description': 'Gradient Boosting with L1/L2 regularization and tree subsampling.',
                    'accuracy': test_res_xgb['accuracy'],
                    'precision': test_res_xgb['precision'],
                    'recall': test_res_xgb['recall'],
                    'f1': test_res_xgb['f1'],
                    'roc_auc': test_res_xgb['roc_auc'],
                    'pr_auc': test_res_xgb['pr_auc'],
                    'brier': test_res_xgb['brier'],
                    'threshold': float(thresh_xgb['chosen_threshold']),
                    'confusion_matrix': test_res_xgb['confusion_matrix'],
                    'tn': test_res_xgb['tn'],
                    'fp': test_res_xgb['fp'],
                    'fn': test_res_xgb['fn'],
                    'tp': test_res_xgb['tp']
                },
                {
                    'name': 'LightGBM',
                    'short': 'LGBM',
                    'type': 'individual',
                    'color': '#059669',
                    'is_deployed': False,
                    'description': 'Leaf-wise Gradient Boosting with depth constraints.',
                    'accuracy': test_res_lgbm['accuracy'],
                    'precision': test_res_lgbm['precision'],
                    'recall': test_res_lgbm['recall'],
                    'f1': test_res_lgbm['f1'],
                    'roc_auc': test_res_lgbm['roc_auc'],
                    'pr_auc': test_res_lgbm['pr_auc'],
                    'brier': test_res_lgbm['brier'],
                    'threshold': float(thresh_lgbm['chosen_threshold']),
                    'confusion_matrix': test_res_lgbm['confusion_matrix'],
                    'tn': test_res_lgbm['tn'],
                    'fp': test_res_lgbm['fp'],
                    'fn': test_res_lgbm['fn'],
                    'tp': test_res_lgbm['tp']
                },
                {
                    'name': 'Stacking Ensemble',
                    'short': 'Ensemble',
                    'type': 'ensemble',
                    'color': '#7c3aed',
                    'is_deployed': True,
                    'description': 'Stacking (RF + XGB + LGBM) via Logistic Regression meta-learner with calibrated threshold.',
                    'accuracy': test_res_stack['accuracy'],
                    'precision': test_res_stack['precision'],
                    'recall': test_res_stack['recall'],
                    'f1': test_res_stack['f1'],
                    'roc_auc': test_res_stack['roc_auc'],
                    'pr_auc': test_res_stack['pr_auc'],
                    'brier': test_res_stack['brier'],
                    'threshold': frozen_threshold,
                    'confusion_matrix': test_res_stack['confusion_matrix'],
                    'tn': test_res_stack['tn'],
                    'fp': test_res_stack['fp'],
                    'fn': test_res_stack['fn'],
                    'tp': test_res_stack['tp']
                }
            ],
            'metric_keys': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'brier'],
            'metric_labels': {
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1': 'F1 Score',
                'roc_auc': 'ROC-AUC',
                'brier': 'Brier Score'
            },
            'generated_at': pd.Timestamp.now().isoformat(),
            'is_fallback': False
        }
        with open(os.path.join(target_dir, 'model_comparison_results.json'), 'w') as f:
            json.dump(comparison_data, f, indent=2)

        # Version info JSON
        version_info = {
            'version': 'churn_v2',
            'deployed_model': 'Stacking Ensemble',
            'dataset': 'IBM Telco Customer Churn (Balanced 50/50 Split)',
            'features': len(feature_cols),
            'threshold': frozen_threshold,
            'models': ['Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble'],
            'status': 'active',
            'accuracy': test_res_stack['accuracy'],
            'recall': test_res_stack['recall'],
            'f1': test_res_stack['f1'],
            'brier': test_res_stack['brier']
        }
        with open(os.path.join(target_dir, 'model_version_info.json'), 'w') as f:
            json.dump(version_info, f, indent=2)

    print("\nAll production artifacts successfully saved to models/ and backend/models/!")
    print("=" * 70)


if __name__ == '__main__':
    main()

"""
phase2_rigorous_evaluation.py — Leakage-Safe Repeated Stratified K-Fold Cross-Validation.

EXCLUSIVELY evaluates balanced_telco_churn.csv.
- Preprocessing and scaling fitted strictly on training folds.
- No validation data leaks into feature transformations, scaling, or model fitting.
- Evaluates 8 canonical models + Stacking Ensemble across repeated Stratified K-Fold CV.
- Reports Mean ± Standard Deviation for Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Brier Score.
- Saves all per-fold results, summary CSVs, confusion matrices, and JSON logs to results/.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'balanced_telco_churn.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. LEAKAGE-FREE FEATURE ENGINEERING FUNCTION
# -----------------------------------------------------------------------------
CORE_NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CORE_CAT_COLS = ['Contract', 'InternetService', 'PaymentMethod']

DERIVED_NUMERIC_COLS = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'AverageMonthlySpend', 'ChargeIncrease', 'ChargeIncreaseRatio',
    'TenureSquared', 'TenureLog',
    'tenure_x_MonthlyCharges', 'tenure_x_IsMonthToMonth', 'MonthlyCharges_x_IsMonthToMonth'
]

BINARY_FLAG_COLS = [
    'IsNewCustomer', 'IsLongTermCustomer', 'IsMonthToMonth', 'IsLongTermContract',
    'IsElectronicCheck', 'HighRiskCombo', 'VeryHighRisk',
    'FiberOptic_x_MonthToMonth', 'FiberOptic_x_ElectronicCheck'
]

ONE_HOT_COLS = [
    'Contract_One year', 'Contract_Two year',
    'InternetService_Fiber optic', 'InternetService_No',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

ALL_FEATURE_COLS = DERIVED_NUMERIC_COLS + BINARY_FLAG_COLS + ONE_HOT_COLS
assert len(ALL_FEATURE_COLS) == 27, f"Expected 27 features, got {len(ALL_FEATURE_COLS)}"


def derive_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Row-independent deterministic feature transformations."""
    df = df_raw.copy()
    
    # Coerce numeric
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0.0).clip(lower=0.0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0).clip(lower=0.0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0).clip(lower=0.0)
    
    zero_tc = (df['TotalCharges'] == 0) & (df['tenure'] > 0)
    df.loc[zero_tc, 'TotalCharges'] = df.loc[zero_tc, 'tenure'] * df.loc[zero_tc, 'MonthlyCharges']
    
    df['Contract'] = df['Contract'].astype(str).str.strip()
    df['InternetService'] = df['InternetService'].astype(str).str.strip()
    df['PaymentMethod'] = df['PaymentMethod'].astype(str).str.strip()
    
    # Derived Spend & Tenure Features
    df['AverageMonthlySpend'] = df['TotalCharges'] / np.maximum(df['tenure'], 1.0)
    df['ChargeIncrease'] = df['MonthlyCharges'] - df['AverageMonthlySpend']
    df['ChargeIncreaseRatio'] = df['ChargeIncrease'] / np.maximum(df['AverageMonthlySpend'], 1e-6)
    df['TenureSquared'] = df['tenure'] ** 2
    df['TenureLog'] = np.log1p(df['tenure'])
    
    # Binary indicators
    df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
    df['IsLongTermCustomer'] = (df['tenure'] >= 48).astype(int)
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['IsLongTermContract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)
    df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['HighRiskCombo'] = ((df['IsMonthToMonth'] == 1) & (df['IsElectronicCheck'] == 1)).astype(int)
    df['VeryHighRisk'] = ((df['HighRiskCombo'] == 1) & (df['tenure'] <= 12)).astype(int)
    
    # Interactions
    df['tenure_x_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['tenure_x_IsMonthToMonth'] = df['tenure'] * df['IsMonthToMonth']
    df['MonthlyCharges_x_IsMonthToMonth'] = df['MonthlyCharges'] * df['IsMonthToMonth']
    df['FiberOptic_x_MonthToMonth'] = ((df['InternetService'] == 'Fiber optic') & (df['IsMonthToMonth'] == 1)).astype(int)
    df['FiberOptic_x_ElectronicCheck'] = ((df['InternetService'] == 'Fiber optic') & (df['IsElectronicCheck'] == 1)).astype(int)
    
    # Dummy encoding
    df['Contract_One year'] = (df['Contract'] == 'One year').astype(int)
    df['Contract_Two year'] = (df['Contract'] == 'Two year').astype(int)
    df['InternetService_Fiber optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['InternetService_No'] = (df['InternetService'] == 'No').astype(int)
    df['PaymentMethod_Credit card (automatic)'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(int)
    df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['PaymentMethod_Mailed check'] = (df['PaymentMethod'] == 'Mailed check').astype(int)
    
    return df[ALL_FEATURE_COLS].astype(float)


# -----------------------------------------------------------------------------
# 2. MODEL FACTORIES
# -----------------------------------------------------------------------------
def get_models(seed: int = 42) -> Dict[str, Any]:
    return {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=seed, solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=seed
        ),
        'SVM (RBF)': SVC(
            C=1.0, kernel='rbf', probability=True, random_state=seed
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=6, min_samples_leaf=2,
            random_state=seed, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.85, random_state=seed, eval_metric='logloss', n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=150, max_depth=6, num_leaves=31, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85, random_state=seed, verbose=-1, n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=150, depth=6, learning_rate=0.08, random_seed=seed,
            verbose=0, thread_count=-1
        ),
        'MLPClassifier': MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01,
            max_iter=300, random_state=seed, early_stopping=True, n_iter_no_change=15
        )
    }


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_preds = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': float(accuracy_score(y_true, y_preds)),
        'precision': float(precision_score(y_true, y_preds, zero_division=0)),
        'recall': float(recall_score(y_true, y_preds, zero_division=0)),
        'f1': float(f1_score(y_true, y_preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_probs)),
        'pr_auc': float(average_precision_score(y_true, y_probs)),
        'brier': float(brier_score_loss(y_true, y_probs)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }


# -----------------------------------------------------------------------------
# 3. REPEATED STRATIFIED K-FOLD PIPELINE
# -----------------------------------------------------------------------------
def run_phase2_rigorous_evaluation(n_splits: int = 5, n_repeats: int = 2, random_state: int = 42):
    print("=" * 80)
    print("PHASE 2: LEAKAGE-SAFE REPEATED STRATIFIED K-FOLD EVALUATION")
    print(f"Dataset: balanced_telco_churn.csv ONLY ({n_splits}-Fold x {n_repeats}-Repeat CV)")
    print("=" * 80)
    
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"
    df_raw = pd.read_csv(DATA_PATH)
    
    print(f"Loaded records: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    print(f"Target distribution:\n{df_raw['target'].value_counts().to_dict()}")
    
    # 6 Production inputs
    X_raw_df = df_raw[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']]
    y = df_raw['target'].values
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    model_names = [
        'Logistic Regression', 'Decision Tree', 'SVM (RBF)',
        'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost',
        'MLPClassifier', 'Stacking Ensemble'
    ]
    
    fold_results = {m: [] for m in model_names}
    confusion_matrices_total = {m: np.zeros((2, 2), dtype=int) for m in model_names}
    
    total_folds = n_splits * n_repeats
    start_time = time.time()
    
    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_raw_df, y), 1):
        fold_start = time.time()
        print(f"\n--- Running Fold {fold_idx}/{total_folds} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")
        
        # 1. SPLIT FIRST (LEAKAGE SAFE)
        df_train_raw = X_raw_df.iloc[train_idx].copy()
        df_val_raw = X_raw_df.iloc[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 2. FEATURE DERIVATION
        X_train_df = derive_features(df_train_raw)
        X_val_df = derive_features(df_val_raw)
        
        # 3. FIT PREPROCESSING ONLY ON TRAINING FOLD
        scaler = StandardScaler()
        X_train_scaled = X_train_df.copy()
        X_val_scaled = X_val_df.copy()
        
        X_train_scaled[DERIVED_NUMERIC_COLS] = scaler.fit_transform(X_train_df[DERIVED_NUMERIC_COLS])
        X_val_scaled[DERIVED_NUMERIC_COLS] = scaler.transform(X_val_df[DERIVED_NUMERIC_COLS])
        
        # 4. INSTANTIATE MODELS
        models = get_models(seed=random_state + fold_idx)
        val_probs_dict = {}
        
        # Train and evaluate 8 base models
        for m_name, clf in models.items():
            t0 = time.time()
            clf.fit(X_train_scaled, y_train)
            probs_val = clf.predict_proba(X_val_scaled)[:, 1]
            val_probs_dict[m_name] = probs_val
            
            m_metrics = evaluate_predictions(y_val, probs_val, threshold=0.50)
            m_metrics['fold'] = fold_idx
            m_metrics['model'] = m_name
            fold_results[m_name].append(m_metrics)
            
            cm = np.array([[m_metrics['tn'], m_metrics['fp']], [m_metrics['fn'], m_metrics['tp']]])
            confusion_matrices_total[m_name] += cm
            
            print(f"  {m_name:20s} | Acc: {m_metrics['accuracy']*100:5.2f}% | Rec: {m_metrics['recall']*100:5.2f}% | Prec: {m_metrics['precision']*100:5.2f}% | AUC: {m_metrics['roc_auc']*100:5.2f}% ({time.time()-t0:.1f}s)")
            
        # 5. STACKING ENSEMBLE (Trained strictly with inner 5-fold OOF on training fold)
        # Inner CV on X_train_scaled to prevent meta-learner overfitting
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + fold_idx)
        inner_oof_rf = np.zeros(len(X_train_scaled))
        inner_oof_xgb = np.zeros(len(X_train_scaled))
        inner_oof_lgbm = np.zeros(len(X_train_scaled))
        
        for in_tr_idx, in_va_idx in inner_cv.split(X_train_scaled, y_train):
            X_in_tr, y_in_tr = X_train_scaled.iloc[in_tr_idx], y_train[in_tr_idx]
            X_in_va = X_train_scaled.iloc[in_va_idx]
            
            in_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)
            in_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=random_state, eval_metric='logloss', n_jobs=-1)
            in_lgbm = LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=25, learning_rate=0.08, random_state=random_state, verbose=-1, n_jobs=-1)
            
            in_rf.fit(X_in_tr, y_in_tr)
            in_xgb.fit(X_in_tr, y_in_tr)
            in_lgbm.fit(X_in_tr, y_in_tr)
            
            inner_oof_rf[in_va_idx] = in_rf.predict_proba(X_in_va)[:, 1]
            inner_oof_xgb[in_va_idx] = in_xgb.predict_proba(X_in_va)[:, 1]
            inner_oof_lgbm[in_va_idx] = in_lgbm.predict_proba(X_in_va)[:, 1]
            
        # Fit meta-learner strictly on inner OOF predictions
        meta_X_train = np.column_stack([inner_oof_rf, inner_oof_xgb, inner_oof_lgbm])
        meta_learner = LogisticRegression(C=1.0, random_state=random_state)
        meta_learner.fit(meta_X_train, y_train)
        
        # Meta validation predictions from full base learners
        meta_X_val = np.column_stack([
            val_probs_dict['Random Forest'],
            val_probs_dict['XGBoost'],
            val_probs_dict['LightGBM']
        ])
        probs_stack_val = meta_learner.predict_proba(meta_X_val)[:, 1]
        
        stack_metrics = evaluate_predictions(y_val, probs_stack_val, threshold=0.50)
        stack_metrics['fold'] = fold_idx
        stack_metrics['model'] = 'Stacking Ensemble'
        fold_results['Stacking Ensemble'].append(stack_metrics)
        
        cm_stack = np.array([[stack_metrics['tn'], stack_metrics['fp']], [stack_metrics['fn'], stack_metrics['tp']]])
        confusion_matrices_total['Stacking Ensemble'] += cm_stack
        
        print(f"  {'Stacking Ensemble':20s} | Acc: {stack_metrics['accuracy']*100:5.2f}% | Rec: {stack_metrics['recall']*100:5.2f}% | Prec: {stack_metrics['precision']*100:5.2f}% | AUC: {stack_metrics['roc_auc']*100:5.2f}%")
        print(f"Fold {fold_idx} completed in {time.time()-fold_start:.1f}s")

    print("\n" + "=" * 80)
    print("PHASE 2 EVALUATION COMPLETED: AGGREGATING FINAL METRICS")
    print("=" * 80)
    
    # Flatten per-fold results to DataFrame
    all_fold_records = []
    for m_name, res_list in fold_results.items():
        all_fold_records.extend(res_list)
        
    df_folds = pd.DataFrame(all_fold_records)
    folds_csv_path = os.path.join(RESULTS_DIR, 'phase2_repeated_cv_folds.csv')
    df_folds.to_csv(folds_csv_path, index=False)
    print(f"Saved all per-fold metric records to: {folds_csv_path}")
    
    # Calculate Mean ± Std
    summary_rows = []
    for m_name in model_names:
        sub = df_folds[df_folds['model'] == m_name]
        summary_rows.append({
            'Model': m_name,
            'Accuracy': f"{sub['accuracy'].mean()*100:.2f}% ± {sub['accuracy'].std()*100:.2f}%",
            'Precision': f"{sub['precision'].mean()*100:.2f}% ± {sub['precision'].std()*100:.2f}%",
            'Recall': f"{sub['recall'].mean()*100:.2f}% ± {sub['recall'].std()*100:.2f}%",
            'F1': f"{sub['f1'].mean()*100:.2f}% ± {sub['f1'].std()*100:.2f}%",
            'ROC-AUC': f"{sub['roc_auc'].mean()*100:.2f}% ± {sub['roc_auc'].std()*100:.2f}%",
            'PR-AUC': f"{sub['pr_auc'].mean()*100:.2f}% ± {sub['pr_auc'].std()*100:.2f}%",
            'Brier Score': f"{sub['brier'].mean():.4f} ± {sub['brier'].std():.4f}",
            'acc_mean': sub['accuracy'].mean(),
            'acc_std': sub['accuracy'].std(),
            'prec_mean': sub['precision'].mean(),
            'prec_std': sub['precision'].std(),
            'rec_mean': sub['recall'].mean(),
            'rec_std': sub['recall'].std(),
            'f1_mean': sub['f1'].mean(),
            'f1_std': sub['f1'].std(),
            'roc_mean': sub['roc_auc'].mean(),
            'roc_std': sub['roc_auc'].std(),
            'pr_mean': sub['pr_auc'].mean(),
            'pr_std': sub['pr_auc'].std(),
            'brier_mean': sub['brier'].mean(),
            'brier_std': sub['brier'].std(),
            'total_cm': confusion_matrices_total[m_name].tolist()
        })
        
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(RESULTS_DIR, 'phase2_repeated_cv_summary.csv')
    df_summary[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC', 'Brier Score']].to_csv(summary_csv_path, index=False)
    print(f"Saved summary metrics table to: {summary_csv_path}")
    
    summary_json_path = os.path.join(RESULTS_DIR, 'phase2_repeated_cv_summary.json')
    with open(summary_json_path, 'w') as jf:
        json.dump(summary_rows, jf, indent=2)
    print(f"Saved JSON summary to: {summary_json_path}")
    
    # Print formatted comparative table
    print("\n" + "=" * 115)
    print(f"{'Model':22s} | {'Accuracy':15s} | {'Precision':15s} | {'Recall':15s} | {'F1':15s} | {'ROC-AUC':15s} | {'Brier Score':15s}")
    print("-" * 115)
    for r in summary_rows:
        print(f"{r['Model']:22s} | {r['Accuracy']:15s} | {r['Precision']:15s} | {r['Recall']:15s} | {r['F1']:15s} | {r['ROC-AUC']:15s} | {r['Brier Score']:15s}")
    print("=" * 115)
    print(f"Total Phase 2 execution time: {time.time()-start_time:.1f} seconds")
    
    return summary_rows


if __name__ == '__main__':
    run_phase2_rigorous_evaluation(n_splits=5, n_repeats=2, random_state=42)

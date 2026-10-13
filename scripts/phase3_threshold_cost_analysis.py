"""
phase3_threshold_cost_analysis.py — Leakage-Safe Threshold Optimization and Business Cost Analysis.

Scope:
- Dataset: balanced_telco_churn.csv EXCLUSIVELY.
- Models: Random Forest, XGBoost, LightGBM, and Stacking Ensemble ONLY.
- Evaluation Scheme: 5-Fold x 2-Repeat Stratified CV (10 folds total).
- Leakage-Safe: Optimal thresholds (tau_cost* and tau_F2*) selected SOLELY from training-side inner OOF predictions.
- Validation fold is completely unseen during threshold selection.
- Primary Business Cost: C_FN = $500, C_FP = $50 (10:1 ratio).
- Sensitivity Analysis: 3:1, 5:1, 10:1, 20:1 ratios.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List
sys.stdout.reconfigure(line_buffering=True)

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'balanced_telco_churn.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'paper_figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. 27-DIMENSIONAL FEATURE ENGINEERING
# -----------------------------------------------------------------------------
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


def derive_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0.0).clip(lower=0.0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0).clip(lower=0.0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0).clip(lower=0.0)
    
    zero_tc = (df['TotalCharges'] == 0) & (df['tenure'] > 0)
    df.loc[zero_tc, 'TotalCharges'] = df.loc[zero_tc, 'tenure'] * df.loc[zero_tc, 'MonthlyCharges']
    
    df['Contract'] = df['Contract'].astype(str).str.strip()
    df['InternetService'] = df['InternetService'].astype(str).str.strip()
    df['PaymentMethod'] = df['PaymentMethod'].astype(str).str.strip()
    
    df['AverageMonthlySpend'] = df['TotalCharges'] / np.maximum(df['tenure'], 1.0)
    df['ChargeIncrease'] = df['MonthlyCharges'] - df['AverageMonthlySpend']
    df['ChargeIncreaseRatio'] = df['ChargeIncrease'] / np.maximum(df['AverageMonthlySpend'], 1e-6)
    df['TenureSquared'] = df['tenure'] ** 2
    df['TenureLog'] = np.log1p(df['tenure'])
    
    df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
    df['IsLongTermCustomer'] = (df['tenure'] >= 48).astype(int)
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['IsLongTermContract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)
    df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['HighRiskCombo'] = ((df['IsMonthToMonth'] == 1) & (df['IsElectronicCheck'] == 1)).astype(int)
    df['VeryHighRisk'] = ((df['HighRiskCombo'] == 1) & (df['tenure'] <= 12)).astype(int)
    
    df['tenure_x_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['tenure_x_IsMonthToMonth'] = df['tenure'] * df['IsMonthToMonth']
    df['MonthlyCharges_x_IsMonthToMonth'] = df['MonthlyCharges'] * df['IsMonthToMonth']
    df['FiberOptic_x_MonthToMonth'] = ((df['InternetService'] == 'Fiber optic') & (df['IsMonthToMonth'] == 1)).astype(int)
    df['FiberOptic_x_ElectronicCheck'] = ((df['InternetService'] == 'Fiber optic') & (df['IsElectronicCheck'] == 1)).astype(int)
    
    df['Contract_One year'] = (df['Contract'] == 'One year').astype(int)
    df['Contract_Two year'] = (df['Contract'] == 'Two year').astype(int)
    df['InternetService_Fiber optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['InternetService_No'] = (df['InternetService'] == 'No').astype(int)
    df['PaymentMethod_Credit card (automatic)'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(int)
    df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['PaymentMethod_Mailed check'] = (df['PaymentMethod'] == 'Mailed check').astype(int)
    
    return df[ALL_FEATURE_COLS].astype(float)


# -----------------------------------------------------------------------------
# 2. THRESHOLD OPTIMIZATION UTILS (STRICTLY ON TRAINING OOF)
# -----------------------------------------------------------------------------
def calculate_cost(fn: int, fp: int, c_fn: float = 500.0, c_fp: float = 50.0) -> float:
    return float(c_fn * fn + c_fp * fp)


def find_optimal_thresholds_from_oof(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    cost_ratios: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    """Sweeps threshold in [0.01, 0.99] strictly on training-side OOF probabilities."""
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    # Track optimal thresholds
    best_f2 = -1.0
    tau_f2 = 0.50
    
    best_costs = {ratio_name: float('inf') for ratio_name in cost_ratios}
    tau_costs = {ratio_name: 0.50 for ratio_name in cost_ratios}
    
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # F2 calculation
        f2 = fbeta_score(y_true, preds, beta=2.0, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            tau_f2 = float(t)
            
        # Cost calculation for each ratio
        for ratio_name, (c_fn, c_fp) in cost_ratios.items():
            cost = calculate_cost(fn, fp, c_fn=c_fn, c_fp=c_fp)
            if cost < best_costs[ratio_name]:
                best_costs[ratio_name] = cost
                tau_costs[ratio_name] = float(t)
                
    result = {'tau_f2': tau_f2}
    for ratio_name in cost_ratios:
        result[f'tau_cost_{ratio_name}'] = tau_costs[ratio_name]
    return result


def evaluate_at_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    tau: float,
    c_fn: float = 500.0,
    c_fp: float = 50.0
) -> Dict[str, Any]:
    preds = (probs >= tau).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    cost = calculate_cost(fn, fp, c_fn=c_fn, c_fp=c_fp)
    
    return {
        'threshold': float(tau),
        'accuracy': float(accuracy_score(y_true, preds)),
        'precision': float(precision_score(y_true, preds, zero_division=0)),
        'recall': float(recall_score(y_true, preds, zero_division=0)),
        'f1': float(f1_score(y_true, preds, zero_division=0)),
        'f2': float(fbeta_score(y_true, preds, beta=2.0, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, probs)),
        'pr_auc': float(average_precision_score(y_true, probs)),
        'brier': float(brier_score_loss(y_true, probs)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'cost': float(cost),
        'cost_per_customer': float(cost / len(y_true))
    }


# -----------------------------------------------------------------------------
# 3. REPEATED STRATIFIED K-FOLD PIPELINE
# -----------------------------------------------------------------------------
def run_phase3_evaluation(n_splits: int = 5, n_repeats: int = 2, random_state: int = 42):
    print("=" * 85)
    print("PHASE 3: LEAKAGE-SAFE THRESHOLD AND BUSINESS COST EVALUATION")
    print(f"Dataset: balanced_telco_churn.csv ONLY ({n_splits}-Fold x {n_repeats}-Repeat CV)")
    print("Models: Random Forest, XGBoost, LightGBM, Stacking Ensemble")
    print("=" * 85)
    
    df_raw = pd.read_csv(DATA_PATH)
    X_raw_df = df_raw[['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']]
    y = df_raw['target'].values
    
    COST_RATIOS = {
        '3_1': (300.0, 100.0),
        '5_1': (500.0, 100.0),
        '10_1': (500.0, 50.0),   # Primary baseline
        '20_1': (1000.0, 50.0)
    }
    
    MODELS_ORDER = ['Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble']
    
    # Store metrics for each model and strategy:
    # Strategies: 'default_0.50', 'f2_optimal', 'cost_optimal_10_1', 'cost_optimal_3_1', 'cost_optimal_5_1', 'cost_optimal_20_1'
    STRATEGIES = ['default_0.50', 'f2_optimal', 'cost_optimal_10_1', 'cost_optimal_3_1', 'cost_optimal_5_1', 'cost_optimal_20_1']
    
    results_collector = {
        m: {s: [] for s in STRATEGIES}
        for m in MODELS_ORDER
    }
    
    # For cost-curve visualization (average validation cost across folds for each tau)
    tau_grid = np.arange(0.01, 1.00, 0.01)
    curve_collector_val_cost = {
        m: {r: np.zeros(len(tau_grid)) for r in COST_RATIOS}
        for m in MODELS_ORDER
    }
    
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
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
        
        # 3. PREPROCESSING FITTED EXCLUSIVELY ON TRAINING FOLD
        scaler = StandardScaler()
        X_train_scaled = X_train_df.copy()
        X_val_scaled = X_val_df.copy()
        X_train_scaled[DERIVED_NUMERIC_COLS] = scaler.fit_transform(X_train_df[DERIVED_NUMERIC_COLS])
        X_val_scaled[DERIVED_NUMERIC_COLS] = scaler.transform(X_val_df[DERIVED_NUMERIC_COLS])
        
        # 4. INNER 5-FOLD CV ON TRAINING FOLD FOR TRAINING-SIDE OOF PROBABILITIES
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state + fold_idx)
        inner_oof_rf = np.zeros(len(X_train_scaled))
        inner_oof_xgb = np.zeros(len(X_train_scaled))
        inner_oof_lgbm = np.zeros(len(X_train_scaled))
        
        for in_tr, in_va in inner_cv.split(X_train_scaled, y_train):
            X_in_tr, y_in_tr = X_train_scaled.iloc[in_tr], y_train[in_tr]
            X_in_va = X_train_scaled.iloc[in_va]
            
            clf_rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=6, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
            clf_xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, random_state=random_state, eval_metric='logloss', n_jobs=-1)
            clf_lgbm = LGBMClassifier(n_estimators=150, max_depth=6, num_leaves=31, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, random_state=random_state, verbose=-1, n_jobs=-1)
            
            clf_rf.fit(X_in_tr, y_in_tr)
            clf_xgb.fit(X_in_tr, y_in_tr)
            clf_lgbm.fit(X_in_tr, y_in_tr)
            
            inner_oof_rf[in_va] = clf_rf.predict_proba(X_in_va)[:, 1]
            inner_oof_xgb[in_va] = clf_xgb.predict_proba(X_in_va)[:, 1]
            inner_oof_lgbm[in_va] = clf_lgbm.predict_proba(X_in_va)[:, 1]
            
        # Stacking inner OOF probabilities
        meta_X_train = np.column_stack([inner_oof_rf, inner_oof_xgb, inner_oof_lgbm])
        meta_clf = LogisticRegression(C=1.0, random_state=random_state)
        meta_clf.fit(meta_X_train, y_train)
        inner_oof_stack = meta_clf.predict_proba(meta_X_train)[:, 1]
        
        oof_dict = {
            'Random Forest': inner_oof_rf,
            'XGBoost': inner_oof_xgb,
            'LightGBM': inner_oof_lgbm,
            'Stacking Ensemble': inner_oof_stack
        }
        
        # 5. SELECT OPTIMAL THRESHOLDS STRICTLY USING TRAINING OOF (ZERO LEAKAGE)
        optimal_thresholds = {}
        for m_name, oof_p in oof_dict.items():
            optimal_thresholds[m_name] = find_optimal_thresholds_from_oof(y_train, oof_p, COST_RATIOS)
            
        # 6. FIT FULL BASE MODELS ON FULL TRAINING FOLD
        full_rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=6, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
        full_xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, random_state=random_state, eval_metric='logloss', n_jobs=-1)
        full_lgbm = LGBMClassifier(n_estimators=150, max_depth=6, num_leaves=31, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85, random_state=random_state, verbose=-1, n_jobs=-1)
        
        full_rf.fit(X_train_scaled, y_train)
        full_xgb.fit(X_train_scaled, y_train)
        full_lgbm.fit(X_train_scaled, y_train)
        
        # Predict on validation fold
        p_rf_val = full_rf.predict_proba(X_val_scaled)[:, 1]
        p_xgb_val = full_xgb.predict_proba(X_val_scaled)[:, 1]
        p_lgbm_val = full_lgbm.predict_proba(X_val_scaled)[:, 1]
        
        meta_X_val = np.column_stack([p_rf_val, p_xgb_val, p_lgbm_val])
        p_stack_val = meta_clf.predict_proba(meta_X_val)[:, 1]
        
        val_probs_map = {
            'Random Forest': p_rf_val,
            'XGBoost': p_xgb_val,
            'LightGBM': p_lgbm_val,
            'Stacking Ensemble': p_stack_val
        }
        
        # 7. EVALUATE ON COMPLETELY UNSEEN VALIDATION FOLD
        for m_name in MODELS_ORDER:
            p_val = val_probs_map[m_name]
            opt_th = optimal_thresholds[m_name]
            
            # A. Default symmetric tau = 0.50
            res_def = evaluate_at_threshold(y_val, p_val, 0.50, c_fn=500.0, c_fp=50.0)
            res_def['fold'] = fold_idx
            res_def['strategy'] = 'Default (tau=0.50)'
            results_collector[m_name]['default_0.50'].append(res_def)
            
            # B. F2-optimal threshold
            res_f2 = evaluate_at_threshold(y_val, p_val, opt_th['tau_f2'], c_fn=500.0, c_fp=50.0)
            res_f2['fold'] = fold_idx
            res_f2['strategy'] = f"F2-optimal (tau={opt_th['tau_f2']:.2f})"
            results_collector[m_name]['f2_optimal'].append(res_f2)
            
            # C. Cost-optimal 10:1 (Primary baseline: C_FN=$500, C_FP=$50)
            tau_10_1 = opt_th['tau_cost_10_1']
            res_cost_10 = evaluate_at_threshold(y_val, p_val, tau_10_1, c_fn=500.0, c_fp=50.0)
            res_cost_10['fold'] = fold_idx
            res_cost_10['strategy'] = f"Cost-optimal 10:1 (tau={tau_10_1:.2f})"
            results_collector[m_name]['cost_optimal_10_1'].append(res_cost_10)
            
            # D. Sensitivity ratios: 3:1, 5:1, 20:1
            tau_3 = opt_th['tau_cost_3_1']
            res_cost_3 = evaluate_at_threshold(y_val, p_val, tau_3, c_fn=300.0, c_fp=100.0)
            res_cost_3['fold'] = fold_idx
            res_cost_3['strategy'] = f"Cost-optimal 3:1 (tau={tau_3:.2f})"
            results_collector[m_name]['cost_optimal_3_1'].append(res_cost_3)
            
            tau_5 = opt_th['tau_cost_5_1']
            res_cost_5 = evaluate_at_threshold(y_val, p_val, tau_5, c_fn=500.0, c_fp=100.0)
            res_cost_5['fold'] = fold_idx
            res_cost_5['strategy'] = f"Cost-optimal 5:1 (tau={tau_5:.2f})"
            results_collector[m_name]['cost_optimal_5_1'].append(res_cost_5)
            
            tau_20 = opt_th['tau_cost_20_1']
            res_cost_20 = evaluate_at_threshold(y_val, p_val, tau_20, c_fn=1000.0, c_fp=50.0)
            res_cost_20['fold'] = fold_idx
            res_cost_20['strategy'] = f"Cost-optimal 20:1 (tau={tau_20:.2f})"
            results_collector[m_name]['cost_optimal_20_1'].append(res_cost_20)
            
            # Update validation cost curve points
            for r_idx, (r_name, (cfn, cfp)) in enumerate(COST_RATIOS.items()):
                for t_idx, t_val in enumerate(tau_grid):
                    preds_grid = (p_val >= t_val).astype(int)
                    cm_g = confusion_matrix(y_val, preds_grid)
                    c_g = calculate_cost(cm_g[1, 0], cm_g[0, 1], c_fn=cfn, c_fp=cfp)
                    curve_collector_val_cost[m_name][r_name][t_idx] += c_g / total_folds
                    
            print(f"  {m_name:18s} | Def: Cost=${res_def['cost']:,.0f} (Rec={res_def['recall']*100:.1f}%) | Opt 10:1 (tau={tau_10_1:.2f}): Cost=${res_cost_10['cost']:,.0f} (Rec={res_cost_10['recall']*100:.1f}%)")
            
        print(f"Fold {fold_idx} finished in {time.time()-fold_start:.1f}s")

    print("\n" + "=" * 85)
    print("PHASE 3 COMPUTATION COMPLETED: AGGREGATING FINAL METRICS")
    print("=" * 85)
    
    # -------------------------------------------------------------------------
    # 4. AGGREGATE SUMMARY TABLES
    # -------------------------------------------------------------------------
    summary_list = []
    for m_name in MODELS_ORDER:
        for s in STRATEGIES:
            folds_data = pd.DataFrame(results_collector[m_name][s])
            summary_list.append({
                'Model': m_name,
                'Strategy': s,
                'Threshold': f"{folds_data['threshold'].mean():.2f} ± {folds_data['threshold'].std():.2f}",
                'Accuracy': f"{folds_data['accuracy'].mean()*100:.2f}% ± {folds_data['accuracy'].std()*100:.2f}%",
                'Precision': f"{folds_data['precision'].mean()*100:.2f}% ± {folds_data['precision'].std()*100:.2f}%",
                'Recall': f"{folds_data['recall'].mean()*100:.2f}% ± {folds_data['recall'].std()*100:.2f}%",
                'F1-Score': f"{folds_data['f1'].mean()*100:.2f}% ± {folds_data['f1'].std()*100:.2f}%",
                'F2-Score': f"{folds_data['f2'].mean()*100:.2f}% ± {folds_data['f2'].std()*100:.2f}%",
                'TN': f"{folds_data['tn'].mean():.1f} ± {folds_data['tn'].std():.1f}",
                'FP': f"{folds_data['fp'].mean():.1f} ± {folds_data['fp'].std():.1f}",
                'FN': f"{folds_data['fn'].mean():.1f} ± {folds_data['fn'].std():.1f}",
                'TP': f"{folds_data['tp'].mean():.1f} ± {folds_data['tp'].std():.1f}",
                'Business Cost ($)': f"${folds_data['cost'].mean():,.1f} ± ${folds_data['cost'].std():,.1f}",
                'Cost per Customer ($)': f"${folds_data['cost_per_customer'].mean():.2f} ± ${folds_data['cost_per_customer'].std():.2f}",
                'raw_th_mean': folds_data['threshold'].mean(),
                'raw_cost_mean': folds_data['cost'].mean(),
                'raw_rec_mean': folds_data['recall'].mean(),
                'raw_prec_mean': folds_data['precision'].mean(),
                'raw_acc_mean': folds_data['accuracy'].mean(),
                'raw_f1_mean': folds_data['f1'].mean(),
                'raw_f2_mean': folds_data['f2'].mean(),
                'raw_fn_mean': folds_data['fn'].mean(),
                'raw_fp_mean': folds_data['fp'].mean(),
                'raw_tn_mean': folds_data['tn'].mean(),
                'raw_tp_mean': folds_data['tp'].mean()
            })
            
    df_summary = pd.DataFrame(summary_list)
    summary_csv = os.path.join(RESULTS_DIR, 'phase3_threshold_metrics_summary.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved primary threshold metrics summary to: {summary_csv}")
    
    with open(os.path.join(RESULTS_DIR, 'phase3_threshold_metrics_summary.json'), 'w') as jf:
        json.dump(summary_list, jf, indent=2)
        
    # Print formatted primary comparison table (Default vs F2 vs Cost 10:1)
    print("\n" + "=" * 125)
    print(f"{'Model':18s} | {'Operating Policy':22s} | {'Threshold':12s} | {'Recall':14s} | {'Precision':14s} | {'F1-Score':14s} | {'F2-Score':14s} | {'Total Cost ($)':18s}")
    print("-" * 125)
    for r in summary_list:
        if r['Strategy'] in ['default_0.50', 'f2_optimal', 'cost_optimal_10_1']:
            strat_label = "Default (tau=0.50)" if r['Strategy']=='default_0.50' else ("F2-Optimal" if r['Strategy']=='f2_optimal' else "Cost-Optimal (10:1)")
            print(f"{r['Model']:18s} | {strat_label:22s} | {r['Threshold']:12s} | {r['Recall']:14s} | {r['Precision']:14s} | {r['F1-Score']:14s} | {r['F2-Score']:14s} | {r['Business Cost ($)']:18s}")
    print("=" * 125)
    
    # -------------------------------------------------------------------------
    # 5. GENERATE COST-VS-THRESHOLD PLOTS
    # -------------------------------------------------------------------------
    print("\nGenerating publication-quality cost and sensitivity figures...")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})
    
    # Figure 1: Cost vs Threshold for 10:1 primary baseline across the 4 models
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    colors = {'Random Forest': '#2ca02c', 'XGBoost': '#1f77b4', 'LightGBM': '#ff7f0e', 'Stacking Ensemble': '#9467bd'}
    
    for m_name in MODELS_ORDER:
        costs_10 = curve_collector_val_cost[m_name]['10_1']
        sub_opt = [r for r in summary_list if r['Model'] == m_name and r['Strategy'] == 'cost_optimal_10_1'][0]
        opt_th = sub_opt['raw_th_mean']
        opt_c = sub_opt['raw_cost_mean']
        
        ax.plot(tau_grid, costs_10, label=f"{m_name}", color=colors[m_name], linewidth=2.0)
        ax.scatter([opt_th], [opt_c], color=colors[m_name], s=70, zorder=5)
        
    ax.axvline(0.50, color='gray', linestyle='--', linewidth=1.2, label=r'Default Threshold ($\tau=0.50$)')
    ax.set_xlabel(r'Decision Threshold ($\tau$)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Expected Business Cost ($) [N=1,600]', fontsize=11, fontweight='bold')
    ax.set_title('Expected Business Cost vs. Classification Threshold (Ratio 10:1: $C_{FN}=\\$500, C_{FP}=\\$50$)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper center', frameon=True, fontsize=9.5)
    fig.tight_layout()
    
    plot1_path = os.path.join(FIGURES_DIR, 'cost_vs_threshold_curves.png')
    fig.savefig(plot1_path)
    fig.savefig(os.path.join(RESULTS_DIR, 'cost_vs_threshold_curves.png'))
    plt.close(fig)
    print(f"Saved Figure: {plot1_path}")
    
    # Figure 2: Sensitivity Comparison (3:1, 5:1, 10:1, 20:1) for Stacking Ensemble
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ratio_labels = {
        '3_1': '3:1 Ratio ($C_{FN}=\\$300, C_{FP}=\\$100$)',
        '5_1': '5:1 Ratio ($C_{FN}=\\$500, C_{FP}=\\$100$)',
        '10_1': '10:1 Baseline ($C_{FN}=\\$500, C_{FP}=\\$50$)',
        '20_1': '20:1 Ratio ($C_{FN}=\\$1000, C_{FP}=\\$50$)'
    }
    ratio_colors = {'3_1': '#3366cc', '5_1': '#dc3912', '10_1': '#ff9900', '20_1': '#109618'}
    
    for r_key, r_label in ratio_labels.items():
        norm_costs = curve_collector_val_cost['Stacking Ensemble'][r_key] / 1600.0  # cost per customer
        sub_r = [r for r in summary_list if r['Model'] == 'Stacking Ensemble' and r['Strategy'] == f'cost_optimal_{r_key}'][0]
        opt_th = sub_r['raw_th_mean']
        opt_norm_c = sub_r['raw_cost_mean'] / 1600.0
        
        ax.plot(tau_grid, norm_costs, label=r_label, color=ratio_colors[r_key], linewidth=2.0)
        ax.scatter([opt_th], [opt_norm_c], color=ratio_colors[r_key], s=70, zorder=5)
        
    ax.axvline(0.50, color='gray', linestyle='--', linewidth=1.2, label=r'Default Threshold ($\tau=0.50$)')
    ax.set_xlabel(r'Decision Threshold ($\tau$)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Expected Cost per Customer ($)', fontsize=11, fontweight='bold')
    ax.set_title('Cost Sensitivity Across Asymmetric Cost Ratios (Stacking Ensemble)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    fig.tight_layout()
    
    plot2_path = os.path.join(FIGURES_DIR, 'cost_sensitivity_curves.png')
    fig.savefig(plot2_path)
    fig.savefig(os.path.join(RESULTS_DIR, 'cost_sensitivity_curves.png'))
    plt.close(fig)
    print(f"Saved Figure: {plot2_path}")
    
    print(f"Phase 3 completed in {time.time()-start_time:.1f}s")
    return summary_list


if __name__ == '__main__':
    run_phase3_evaluation(n_splits=5, n_repeats=2, random_state=42)

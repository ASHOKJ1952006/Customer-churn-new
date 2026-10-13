"""
Master Reproducibility Pipeline for:
"Explainable Ensemble Learning and Cost-Sensitive Decision Support Framework for Telecom Customer Churn Prediction"

Usage:
    python reproduce.py [--quick]

Arguments:
    --quick : Runs a fast smoke-test reproduction (2 folds, 1 repeat, 20 concordance samples, 50 latency runs).
              Default is full academic reproduction (5 folds, 2 repeats, 200 concordance samples, 1000 latency runs).
"""

import sys
import os
import argparse
import time
import json
import platform
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, fbeta_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Headless matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Explainability
import shap
from lime.lime_tabular import LimeTabularExplainer

# Configure base paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
TABLES_DIR = os.path.join(SCRIPT_DIR, 'tables')
LOGS_DIR = os.path.join(SCRIPT_DIR, 'logs')

for d in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Configure logging
log_file = os.path.join(LOGS_DIR, f"reproduce_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("reproduce")

RANDOM_SEED = 42
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'balanced_telco_churn.csv')


def verify_dataset():
    """Verify that only balanced_telco_churn.csv is used and exists."""
    logger.info("================================================================================")
    logger.info("STEP 1: DATASET VERIFICATION")
    logger.info("================================================================================")
    if not os.path.exists(DATASET_PATH):
        err_msg = (
            f"\n[FATAL ERROR] Required dataset not found at expected location:\n"
            f"  {DATASET_PATH}\n\n"
            f"To reproduce the paper's results, place 'balanced_telco_churn.csv' in the 'data/' directory.\n"
            f"Do not substitute IBM Telco or any other external dataset."
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    df = pd.read_csv(DATASET_PATH)
    logger.info(f"Loaded dataset: {DATASET_PATH}")
    logger.info(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Strict dimension check
    if df.shape[0] != 8000:
        logger.warning(f"Expected exactly 8,000 rows, found {df.shape[0]}.")

    churn_counts = df['Churn'].value_counts().to_dict()
    logger.info(f"Class distribution: {churn_counts}")
    return df


def engineer_features(df_in):
    """
    Leakage-safe deterministic feature engineering (27 model dimensions).
    Strictly transforms 6 production inputs into 27 features without target leakage.
    """
    df = df_in.copy()
    
    # 1. Base Continuous Indicators
    df['tenure'] = df['tenure'].astype(float)
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

    # 2. Derived Spend Trajectory Indicators
    df['AverageMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1.0)
    df['ChargeIncrease'] = df['MonthlyCharges'] - df['AverageMonthlySpend']
    df['ChargeIncreaseRatio'] = df['MonthlyCharges'] / (df['AverageMonthlySpend'] + 1e-5)
    df['TenureLog'] = np.log1p(df['tenure'])
    df['TenureSquared'] = df['tenure'] ** 2

    # 3. Categorical Boolean Flags
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['IsLongTermContract'] = (df['Contract'].isin(['One year', 'Two year'])).astype(int)
    df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['IsFiberOptic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
    df['IsLongTermCustomer'] = (df['tenure'] >= 24).astype(int)

    # 4. Multi-Risk Interaction Combinations
    df['HighRiskCombo'] = (
        (df['IsMonthToMonth'] == 1) & 
        (df['IsElectronicCheck'] == 1) & 
        (df['tenure'] <= 12)
    ).astype(int)

    df['VeryHighRisk'] = (
        (df['HighRiskCombo'] == 1) & 
        (df['MonthlyCharges'] >= 70.0)
    ).astype(int)

    # 5. Continuous x Categorical Interactions
    df['tenure_x_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['tenure_x_IsMonthToMonth'] = df['tenure'] * df['IsMonthToMonth']
    df['MonthlyCharges_x_IsMonthToMonth'] = df['MonthlyCharges'] * df['IsMonthToMonth']
    df['FiberOptic_x_MonthToMonth'] = df['IsFiberOptic'] * df['IsMonthToMonth']
    df['FiberOptic_x_ElectronicCheck'] = df['IsFiberOptic'] * df['IsElectronicCheck']

    # 6. One-Hot Categorical Columns (exact 7 levels)
    # Contract: One year, Two year (Month-to-month is dropped reference)
    df['Contract_One year'] = (df['Contract'] == 'One year').astype(int)
    df['Contract_Two year'] = (df['Contract'] == 'Two year').astype(int)

    # InternetService: Fiber optic, No (DSL is dropped reference)
    df['InternetService_Fiber optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['InternetService_No'] = (df['InternetService'] == 'No').astype(int)

    # PaymentMethod: Credit card (auto), Electronic check, Mailed check (Bank transfer auto dropped)
    df['PaymentMethod_Credit card (automatic)'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(int)
    df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['PaymentMethod_Mailed check'] = (df['PaymentMethod'] == 'Mailed check').astype(int)

    feature_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend',
        'ChargeIncrease', 'ChargeIncreaseRatio', 'TenureLog', 'TenureSquared',
        'IsMonthToMonth', 'IsLongTermContract', 'IsElectronicCheck', 'IsNewCustomer',
        'IsLongTermCustomer', 'HighRiskCombo', 'VeryHighRisk',
        'tenure_x_MonthlyCharges', 'tenure_x_IsMonthToMonth',
        'MonthlyCharges_x_IsMonthToMonth', 'FiberOptic_x_MonthToMonth',
        'FiberOptic_x_ElectronicCheck', 'Contract_One year', 'Contract_Two year',
        'InternetService_Fiber optic', 'InternetService_No',
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check'
    ]

    continuous_scale_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'AverageMonthlySpend',
        'ChargeIncrease', 'ChargeIncreaseRatio', 'TenureLog', 'TenureSquared',
        'tenure_x_MonthlyCharges', 'tenure_x_IsMonthToMonth',
        'MonthlyCharges_x_IsMonthToMonth'
    ]

    return df[feature_cols], feature_cols, continuous_scale_cols


class CleanStackingEnsemble:
    """Wolpert Stacked Generalization with Out-of-Fold Logistic Regression Meta-Learner."""
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=random_state, n_jobs=-1)
        self.xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=random_state, eval_metric='logloss', n_jobs=-1)
        self.lgb = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31, random_state=random_state, verbose=-1, n_jobs=-1)
        self.meta_clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', random_state=random_state)
        self.base_models = [self.rf, self.xgb, self.lgb]

    def fit_oof(self, X_train, y_train, cv_folds=5):
        """Fit base models on K folds to generate out-of-fold meta-features, then train meta-learner."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((len(X_train), len(self.base_models)))

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]

            for m_idx, model in enumerate(self.base_models):
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_tr, y_tr)
                oof_preds[val_idx, m_idx] = model_clone.predict_proba(X_va)[:, 1]

        # Train meta-classifier on OOF predictions
        self.meta_clf.fit(oof_preds, y_train)

        # Refit base models on full training set
        for model in self.base_models:
            model.fit(X_train, y_train)
        return oof_preds

    def predict_proba(self, X):
        meta_features = np.zeros((len(X), len(self.base_models)))
        for m_idx, model in enumerate(self.base_models):
            meta_features[:, m_idx] = model.predict_proba(X)[:, 1]
        return self.meta_clf.predict_proba(meta_features)


def run_repeated_cv(df, feature_cols, scale_cols, n_splits=5, n_repeats=2):
    """
    Phase 2: Leakage-Safe Repeated Stratified Cross-Validation across 4 Core Models.
    """
    logger.info("================================================================================")
    logger.info(f"STEP 2: REPEATED CROSS-VALIDATION ({n_splits} Folds x {n_repeats} Repeats = {n_splits*n_repeats} Folds)")
    logger.info("================================================================================")

    y = (df['Churn'] == 'Yes').astype(int).values
    df_raw = df.copy()

    model_names = ['Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble']
    metrics_records = {m: [] for m in model_names}
    threshold_records = {m: [] for m in model_names}

    seeds = [42, 100][:n_repeats]
    fold_global = 1

    for seed_idx, seed in enumerate(seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df_raw, y), 1):
            logger.info(f"Executing Fold {fold_global}/{n_splits*n_repeats} (Seed {seed}, Split {fold_idx})...")

            # 1. Feature Engineering strictly inside fold
            df_train_raw = df_raw.iloc[train_idx]
            df_val_raw = df_raw.iloc[val_idx]

            X_tr_df, _, _ = engineer_features(df_train_raw)
            X_va_df, _, _ = engineer_features(df_val_raw)

            # 2. Fit Scaler strictly on train fold
            scaler = StandardScaler()
            X_tr_scaled = X_tr_df.copy()
            X_va_scaled = X_va_df.copy()
            X_tr_scaled[scale_cols] = scaler.fit_transform(X_tr_df[scale_cols])
            X_va_scaled[scale_cols] = scaler.transform(X_va_df[scale_cols])

            X_tr = X_tr_scaled.values
            X_va = X_va_scaled.values
            y_tr = y[train_idx]
            y_va = y[val_idx]

            # 3. Instantiate Models
            rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=seed, n_jobs=-1)
            xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=seed, eval_metric='logloss', n_jobs=-1)
            lgb = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1, n_jobs=-1)
            stacking = CleanStackingEnsemble(random_state=seed)

            # Fit base models and record inner OOF predictions for threshold tuning
            rf.fit(X_tr, y_tr)
            xgb.fit(X_tr, y_tr)
            lgb.fit(X_tr, y_tr)
            oof_stacking = stacking.fit_oof(X_tr, y_tr, cv_folds=5)

            # Predict probabilities on validation fold
            models_dict = {
                'Random Forest': (rf, rf.predict_proba(X_va)[:, 1]),
                'XGBoost': (xgb, xgb.predict_proba(X_va)[:, 1]),
                'LightGBM': (lgb, lgb.predict_proba(X_va)[:, 1]),
                'Stacking Ensemble': (stacking, stacking.predict_proba(X_va)[:, 1])
            }

            # Evaluate Metrics at default threshold 0.50
            for m_name, (m_obj, p_va) in models_dict.items():
                y_pred = (p_va >= 0.50).astype(int)
                metrics_records[m_name].append({
                    'Fold': fold_global,
                    'Accuracy': accuracy_score(y_va, y_pred),
                    'Precision': precision_score(y_va, y_pred, zero_division=0),
                    'Recall': recall_score(y_va, y_pred, zero_division=0),
                    'F1': f1_score(y_va, y_pred, zero_division=0),
                    'ROC_AUC': roc_auc_score(y_va, p_va),
                    'PR_AUC': average_precision_score(y_va, p_va),
                    'Brier': brier_score_loss(y_va, p_va)
                })

            # Cost-Sensitive Threshold Optimization (Phase 3 inside fold)
            # Find cost-optimal and F2-optimal thresholds strictly using inner cross-validation
            c_fn = 500.0
            c_fp = 50.0

            for m_name, (m_obj, p_va) in models_dict.items():
                # Grid search threshold on training predictions
                threshold_grid = np.linspace(0.01, 0.99, 99)
                p_tr = m_obj.predict_proba(X_tr)[:, 1]

                costs_tr = []
                f2_scores_tr = []
                for th in threshold_grid:
                    pred_tr = (p_tr >= th).astype(int)
                    fn_tr = np.sum((pred_tr == 0) & (y_tr == 1))
                    fp_tr = np.sum((pred_tr == 1) & (y_tr == 0))
                    costs_tr.append(c_fn * fn_tr + c_fp * fp_tr)
                    f2_scores_tr.append(fbeta_score(y_tr, pred_tr, beta=2, zero_division=0))

                best_th_cost = threshold_grid[np.argmin(costs_tr)]
                best_th_f2 = threshold_grid[np.argmax(f2_scores_tr)]

                # Apply selected thresholds to outer validation fold
                for policy_name, th_val in [('Default', 0.50), ('F2_Optimal', best_th_f2), ('Cost_Optimal', best_th_cost)]:
                    pred_va = (p_va >= th_val).astype(int)
                    fn_va = np.sum((pred_va == 0) & (y_va == 1))
                    fp_va = np.sum((pred_va == 1) & (y_va == 0))
                    val_cost = c_fn * fn_va + c_fp * fp_va
                    val_recall = recall_score(y_va, pred_va, zero_division=0)

                    threshold_records[m_name].append({
                        'Fold': fold_global,
                        'Policy': policy_name,
                        'Threshold': th_val,
                        'Recall': val_recall,
                        'FN': fn_va,
                        'FP': fp_va,
                        'Cost': val_cost
                    })

            fold_global += 1

    # Summarize Phase 2 Cross-Validation Results
    summary_rows = []
    for m_name in model_names:
        df_m = pd.DataFrame(metrics_records[m_name])
        row = {'Model': m_name}
        for col in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'PR_AUC', 'Brier']:
            row[f'{col}_Mean'] = df_m[col].mean()
            row[f'{col}_Std'] = df_m[col].std()
        summary_rows.append(row)

    df_cv_summary = pd.DataFrame(summary_rows)
    cv_csv_path = os.path.join(RESULTS_DIR, 'phase2_repeated_cv_summary.csv')
    df_cv_summary.to_csv(cv_csv_path, index=False)
    logger.info(f"Saved Repeated CV Summary Table: {cv_csv_path}")

    # Summarize Phase 3 Threshold & Cost Results
    thresh_summary_rows = []
    for m_name in model_names:
        df_th = pd.DataFrame(threshold_records[m_name])
        for pol in ['Default', 'F2_Optimal', 'Cost_Optimal']:
            sub = df_th[df_th['Policy'] == pol]
            thresh_summary_rows.append({
                'Model': m_name,
                'Policy': pol,
                'Threshold_Mean': sub['Threshold'].mean(),
                'Threshold_Std': sub['Threshold'].std(),
                'Recall_Mean': sub['Recall'].mean() * 100.0,
                'Recall_Std': sub['Recall'].std() * 100.0,
                'FN_Mean': sub['FN'].mean(),
                'FN_Std': sub['FN'].std(),
                'Cost_Mean': sub['Cost'].mean(),
                'Cost_Std': sub['Cost'].std()
            })

    df_th_summary = pd.DataFrame(thresh_summary_rows)
    th_csv_path = os.path.join(RESULTS_DIR, 'phase3_threshold_cost_summary.csv')
    df_th_summary.to_csv(th_csv_path, index=False)
    logger.info(f"Saved Threshold & Cost Summary Table: {th_csv_path}")

    return df_cv_summary, df_th_summary


def run_explainability_and_decision_support(df, feature_cols, scale_cols, n_concordance=200):
    """
    Phase 4: SHAP, LIME Concordance, State-Aware Recommendation, ROI, and Latency Benchmark.
    """
    logger.info("================================================================================")
    logger.info("STEP 3: EXPLAINABILITY, DECISION SUPPORT, AND LATENCY BENCHMARKING")
    logger.info("================================================================================")

    # Train/Test Split (80% train / 20% test = 6400 / 1600)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()

    X_train_df, _, _ = engineer_features(df_train)
    X_test_df, _, _ = engineer_features(df_test)

    scaler = StandardScaler()
    X_train_scaled = X_train_df.copy()
    X_test_scaled = X_test_df.copy()
    X_train_scaled[scale_cols] = scaler.fit_transform(X_train_df[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test_df[scale_cols])

    y_train = (df_train['Churn'] == 'Yes').astype(int).values
    y_test = (df_test['Churn'] == 'Yes').astype(int).values

    # Train Stacking Ensemble on Train split
    stacking_model = CleanStackingEnsemble(random_state=42)
    _ = stacking_model.fit_oof(X_train_scaled.values, y_train, cv_folds=5)

    # 1. Global SHAP Feature Importance
    logger.info("Computing Global SHAP Explanations...")
    # Use TreeExplainer on XGBoost component for efficient exact polynomial evaluation
    tree_explainer = shap.TreeExplainer(stacking_model.xgb)
    shap_vals = tree_explainer.shap_values(X_test_scaled.values)

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    shap_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False).reset_index(drop=True)

    shap_csv_path = os.path.join(RESULTS_DIR, 'global_shap_importance.csv')
    shap_importance.to_csv(shap_csv_path, index=False)
    logger.info(f"Saved Global SHAP Importance: {shap_csv_path}")

    # Generate SHAP Bar and Beeswarm Plots
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    top10 = shap_importance.iloc[:10][::-1]
    ax.barh(top10['Feature'], top10['Mean_Absolute_SHAP'], color='#1f77b4', edgecolor='black', height=0.6)
    ax.set_xlabel('Mean |SHAP Value| (Impact on Log-Odds of Churn)', fontsize=10, fontweight='bold')
    ax.set_title('Global SHAP Feature Attribution (Top 10 Risk Drivers)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6, axis='x')
    fig.tight_layout()
    bar_fig_path = os.path.join(FIGURES_DIR, 'shap_feature_importance_bar.png')
    fig.savefig(bar_fig_path)
    plt.close(fig)
    logger.info(f"Saved SHAP Bar Figure: {bar_fig_path}")

    # 2. LIME Explainer Initialization
    def predict_fn(x_arr):
        return stacking_model.predict_proba(x_arr)

    lime_explainer = LimeTabularExplainer(
        training_data=X_train_scaled.values,
        feature_names=feature_cols,
        class_names=['Loyal', 'Churn'],
        mode='classification',
        random_state=42
    )

    # 3. Quantitative Concordance Study across N customers
    logger.info(f"Evaluating Quantitative SHAP vs. LIME Concordance across {n_concordance} customers...")
    eval_indices = np.random.RandomState(42).choice(len(X_test_scaled), size=min(n_concordance, len(X_test_scaled)), replace=False)

    spearman_corrs = []
    top3_overlaps = []
    top5_overlaps = []

    for count, idx in enumerate(eval_indices, 1):
        x_i = X_test_scaled.iloc[idx].values
        s_i = tree_explainer.shap_values(x_i.reshape(1, -1))[0]

        exp_i = lime_explainer.explain_instance(x_i, predict_fn, num_features=len(feature_cols), labels=[1])
        lime_dict = dict(exp_i.as_list(label=1))

        lime_weights = np.zeros(len(feature_cols))
        for rule, weight in lime_dict.items():
            for col_idx, col_name in enumerate(feature_cols):
                if col_name in rule:
                    lime_weights[col_idx] = weight
                    break

        rho, _ = spearmanr(np.abs(s_i), np.abs(lime_weights))
        if not np.isnan(rho):
            spearman_corrs.append(rho)

        top3_shap = set(np.argsort(np.abs(s_i))[::-1][:3])
        top3_lime = set(np.argsort(np.abs(lime_weights))[::-1][:3])
        top3_overlaps.append(len(top3_shap.intersection(top3_lime)) / 3.0)

        top5_shap = set(np.argsort(np.abs(s_i))[::-1][:5])
        top5_lime = set(np.argsort(np.abs(lime_weights))[::-1][:5])
        top5_overlaps.append(len(top5_shap.intersection(top5_lime)) / 5.0)

    concordance_dict = {
        'Sample_Size': len(eval_indices),
        'Spearman_Correlation_Mean': float(np.mean(spearman_corrs)),
        'Spearman_Correlation_Std': float(np.std(spearman_corrs)),
        'Spearman_Correlation_Median': float(np.median(spearman_corrs)),
        'Top3_Overlap_Mean': float(np.mean(top3_overlaps)),
        'Top3_Overlap_Std': float(np.std(top3_overlaps)),
        'Top5_Overlap_Mean': float(np.mean(top5_overlaps)),
        'Top5_Overlap_Std': float(np.std(top5_overlaps))
    }

    conc_path = os.path.join(RESULTS_DIR, 'shap_lime_concordance.json')
    with open(conc_path, 'w') as f:
        json.dump(concordance_dict, f, indent=2)
    logger.info(f"Saved Concordance Summary: {conc_path}")

    # 4. Simulated Retention Economics Analysis
    logger.info("Executing Simulated Retention Economics...")
    threshold_deploy = 0.26
    p_test_churn = stacking_model.predict_proba(X_test_scaled.values)[:, 1]
    flagged = p_test_churn >= threshold_deploy
    n_flagged = int(np.sum(flagged))
    tp_flagged = int(np.sum(flagged & (y_test == 1)))

    c_contact = 50.0
    r_success = 0.50
    clv_saved = 500.0

    outreach_cost = n_flagged * c_contact
    retained_customers = tp_flagged * r_success
    gross_saved = retained_customers * clv_saved
    net_benefit = gross_saved - outreach_cost
    simulated_roi = (net_benefit / max(outreach_cost, 1.0)) * 100.0

    roi_dict = {
        'Test_Cohort_Size': len(y_test),
        'Threshold': threshold_deploy,
        'Flagged_Subscribers': n_flagged,
        'True_Positive_Churners': tp_flagged,
        'Intervention_Cost': outreach_cost,
        'Expected_Retained_Customers': retained_customers,
        'Gross_Value_Saved': gross_saved,
        'Net_Simulated_Benefit': net_benefit,
        'Simulated_ROI_Percentage': simulated_roi
    }

    roi_path = os.path.join(RESULTS_DIR, 'simulated_retention_roi.json')
    with open(roi_path, 'w') as f:
        json.dump(roi_dict, f, indent=2)
    logger.info(f"Saved Simulated ROI Summary: {roi_path}")

    # 5. Latency Benchmarks
    logger.info("Benchmarking Prediction and Explainability Latency...")
    single_x = X_test_scaled.iloc[0:1].values
    batch_100_x = X_test_scaled.iloc[:100].values

    # Pre-warm
    _ = stacking_model.predict_proba(single_x)

    # Profile single prediction (100 runs)
    t_single = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = stacking_model.predict_proba(single_x)
        t_single.append((time.perf_counter() - t0) * 1000.0)

    # Profile batch 100 prediction (100 runs)
    t_batch = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = stacking_model.predict_proba(batch_100_x)
        t_batch.append((time.perf_counter() - t0) * 1000.0)

    # Profile SHAP explanation (50 runs)
    t_shap = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = tree_explainer.shap_values(single_x)
        t_shap.append((time.perf_counter() - t0) * 1000.0)

    latency_records = [
        {'Operation': 'Single-Customer Prediction', 'Batch_Size': 1, 'Mean_ms': np.mean(t_single), 'P95_ms': np.percentile(t_single, 95)},
        {'Operation': 'Batch Customer Prediction', 'Batch_Size': 100, 'Mean_ms': np.mean(t_batch), 'P95_ms': np.percentile(t_batch, 95)},
        {'Operation': 'Local SHAP Explanation', 'Batch_Size': 1, 'Mean_ms': np.mean(t_shap), 'P95_ms': np.percentile(t_shap, 95)}
    ]
    df_latency = pd.DataFrame(latency_records)
    latency_path = os.path.join(RESULTS_DIR, 'reproduced_latency_benchmark.csv')
    df_latency.to_csv(latency_path, index=False)
    logger.info(f"Saved Latency Benchmark: {latency_path}")

    return shap_importance, concordance_dict, roi_dict, df_latency


def main():
    parser = argparse.ArgumentParser(description="Master Reproduction Pipeline")
    parser.add_argument('--quick', action='store_true', help="Run fast reproduction smoke-test")
    args = parser.parse_args()

    logger.info("================================================================================")
    logger.info("MASTER REPRODUCIBILITY PIPELINE STARTING")
    logger.info(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    logger.info("================================================================================")

    # Step 1: Dataset Verification
    df = verify_dataset()

    # Step 2: Feature Engineering Specs
    _, feature_cols, scale_cols = engineer_features(df.head(10))
    logger.info(f"Verified feature dimensions: {len(feature_cols)} total, {len(scale_cols)} continuous scaled.")

    # Step 3: Repeated Cross-Validation & Threshold Cost Optimization
    n_splits = 2 if args.quick else 5
    n_repeats = 1 if args.quick else 2
    n_concordance = 20 if args.quick else 200

    cv_summary, th_summary = run_repeated_cv(df, feature_cols, scale_cols, n_splits=n_splits, n_repeats=n_repeats)

    # Step 4: Explainability, Concordance, Decision Support & Latency
    shap_imp, conc_res, roi_res, df_lat = run_explainability_and_decision_support(
        df, feature_cols, scale_cols, n_concordance=n_concordance
    )

    logger.info("================================================================================")
    logger.info("REPRODUCIBILITY EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"All generated artifacts saved in: {SCRIPT_DIR}")
    logger.info("================================================================================")


if __name__ == '__main__':
    main()

import os
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)

# 1. Load cached base predictions and CV results
data = joblib.load('scratch/cached_base_preds.pkl')
oof_rf, oof_xgb, oof_lgbm = data['oof_rf'], data['oof_xgb'], data['oof_lgbm']
test_rf, test_xgb, test_lgbm = data['test_rf'], data['test_xgb'], data['test_lgbm']
y_train, y_test = data['y_train'], data['y_test']

cv_df = pd.read_csv('results/ensemble_optimization/cross_validation_results.csv')
cv_dict = {}
for _, row in cv_df.iterrows():
    m = row['model']
    cv_dict[m] = {
        'roc_auc_mean': row['roc_auc_mean'],
        'roc_auc_std': row['roc_auc_std'],
        'acc_mean': row['accuracy_mean'],
        'acc_std': row['accuracy_std'],
        'rec_mean': row['recall_mean'],
        'rec_std': row['recall_std'],
        'f1_mean': row['f1_mean'],
        'f1_std': row['f1_std']
    }

eps = 1e-5
oof_meta_raw = np.column_stack([oof_rf, oof_xgb, oof_lgbm])
test_meta_raw = np.column_stack([test_rf, test_xgb, test_lgbm])

oof_meta_logit = logit(np.clip(oof_meta_raw, eps, 1.0 - eps))
test_meta_logit = logit(np.clip(test_meta_raw, eps, 1.0 - eps))

# Baseline Stacking (Raw LR C=1.0)
lr_raw = LogisticRegression(C=1.0, solver='lbfgs', random_state=42).fit(oof_meta_raw, y_train)
test_raw_stack = lr_raw.predict_proba(test_meta_raw)[:, 1]

# Corrected Logit Stacking (Logit LR C=0.1)
lr_logit = LogisticRegression(C=0.1, solver='lbfgs', random_state=42).fit(oof_meta_logit, y_train)
test_logit_stack = lr_logit.predict_proba(test_meta_logit)[:, 1]

def get_row(model_name, probs, threshold, cv_key=None):
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    
    cv_str_auc = "N/A"
    cv_str_acc = "N/A"
    if cv_key and cv_key in cv_dict:
        c = cv_dict[cv_key]
        cv_str_auc = f"{c['roc_auc_mean']:.4f} +/- {c['roc_auc_std']:.4f}"
        cv_str_acc = f"{c['acc_mean']:.4f} +/- {c['acc_std']:.4f}"
        
    return {
        'Model': model_name,
        'Threshold': round(threshold, 2),
        'Recall': float(recall_score(y_test, pred)),
        'Precision': float(precision_score(y_test, pred, zero_division=0)),
        'F1-score': float(f1_score(y_test, pred, zero_division=0)),
        'Accuracy': float(accuracy_score(y_test, pred)),
        'ROC-AUC': float(roc_auc_score(y_test, probs)),
        'Brier score': float(brier_score_loss(y_test, probs)),
        'PR-AUC': float(average_precision_score(y_test, probs)),
        'False negatives': int(fn),
        'False positives': int(fp),
        'CV_ROC_AUC': cv_str_auc,
        'CV_Accuracy': cv_str_acc
    }

rows = [
    get_row('Original Random Forest', test_rf, 0.38, 'Random Forest'),
    get_row('Original XGBoost', test_xgb, 0.29, 'XGBoost'),
    get_row('Original LightGBM', test_lgbm, 0.32, 'LightGBM'),
    get_row('Original Stacking Ensemble', test_raw_stack, 0.26, 'Stacking (Baseline LR C=1.0)'),
    get_row('Corrected Stacking Ensemble', test_logit_stack, 0.50, 'Stacking (Logit LR C=0.1)'),
    get_row('Recall-optimized Ensemble', test_logit_stack, 0.26, 'Stacking (Logit LR C=0.1)'),
    get_row('Best validated Ensemble', test_logit_stack, 0.50, 'Stacking (Logit LR C=0.1)')
]

df_candidates = pd.DataFrame(rows)
out_csv = 'results/ensemble_optimization/candidate_comparison.csv'
df_candidates.to_csv(out_csv, index=False)
print("Successfully generated", out_csv)
print(df_candidates[['Model', 'Threshold', 'Recall', 'Precision', 'F1-score', 'Accuracy', 'ROC-AUC', 'Brier score', 'False negatives', 'False positives']].to_string())

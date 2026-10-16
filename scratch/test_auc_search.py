import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, brier_score_loss, recall_score, precision_score, f1_score, accuracy_score
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression

data = joblib.load('scratch/cached_base_preds.pkl')
oof_rf, oof_xgb, oof_lgbm = data['oof_rf'], data['oof_xgb'], data['oof_lgbm']
test_rf, test_xgb, test_lgbm = data['test_rf'], data['test_xgb'], data['test_lgbm']
y_train, y_test = data['y_train'], data['y_test']
X_train, X_test = data['X_train'], data['X_test']

# Let's explore soft voting weights grid search and continuous optimization
# directly maximizing AUC or minimizing rank loss on OOF
best_oof_auc = 0
best_w = None
best_test_auc = 0

for w_rf in np.linspace(0, 0.4, 21):
    for w_xgb in np.linspace(0.4, 1.0, 31):
        w_lgb = 1.0 - w_rf - w_xgb
        if w_lgb < 0:
            continue
        w = np.array([w_rf, w_xgb, w_lgb])
        p_oof = w_rf * oof_rf + w_xgb * oof_xgb + w_lgb * oof_lgbm
        auc_oof = roc_auc_score(y_train, p_oof)
        if auc_oof > best_oof_auc:
            best_oof_auc = auc_oof
            best_w = w
            p_test = w_rf * test_rf + w_xgb * test_xgb + w_lgb * test_lgbm
            best_test_auc = roc_auc_score(y_test, p_test)

print(f"Optimal OOF Soft Voting weights: {best_w.round(4)} | OOF AUC: {best_oof_auc:.6f} | Test AUC: {best_test_auc:.6f}")

# What about blending XGBoost and LGBM on logit scale?
for w_xgb in np.linspace(0.0, 1.0, 101):
    w_lgb = 1.0 - w_xgb
    p_oof = expit(w_xgb * logit(np.clip(oof_xgb, 1e-5, 1-1e-5)) + w_lgb * logit(np.clip(oof_lgbm, 1e-5, 1-1e-5)))
    auc_oof = roc_auc_score(y_train, p_oof)
    p_test = expit(w_xgb * logit(np.clip(test_xgb, 1e-5, 1-1e-5)) + w_lgb * logit(np.clip(test_lgbm, 1e-5, 1-1e-5)))
    auc_test = roc_auc_score(y_test, p_test)
    if auc_oof > best_oof_auc:
        print(f"Logit blend w_xgb={w_xgb:.3f} | OOF AUC: {auc_oof:.6f} | Test AUC: {auc_test:.6f}")

# Let's check XGBoost test AUC alone:
print("XGBoost alone test AUC:", roc_auc_score(y_test, test_xgb))
print("LGBM alone test AUC:", roc_auc_score(y_test, test_lgbm))
print("RF alone test AUC:", roc_auc_score(y_test, test_rf))

# Let's check combinations with other models or hyperparameters:
# What if we examine the base models themselves?
# In train_churn_v2.py:
# RF: n_estimators=250, max_depth=12, min_samples_split=6, min_samples_leaf=2
# XGB: n_estimators=200, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, min_child_weight=3, gamma=0.1, reg_alpha=0.05, reg_lambda=1.0
# LGBM: n_estimators=200, learning_rate=0.06, num_leaves=31, max_depth=6, min_child_samples=20, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1.0

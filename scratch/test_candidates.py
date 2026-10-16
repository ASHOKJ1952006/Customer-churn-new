import os
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix
)

# Load cached data
data = joblib.load('scratch/cached_base_preds.pkl')
oof_rf, oof_xgb, oof_lgbm = data['oof_rf'], data['oof_xgb'], data['oof_lgbm']
test_rf, test_xgb, test_lgbm = data['test_rf'], data['test_xgb'], data['test_lgbm']
y_train, y_test = data['y_train'], data['y_test']

oof_meta_raw = np.column_stack([oof_rf, oof_xgb, oof_lgbm])
test_meta_raw = np.column_stack([test_rf, test_xgb, test_lgbm])

# Logit transformed meta-features
eps = 1e-5
oof_meta_logit = logit(np.clip(oof_meta_raw, eps, 1.0 - eps))
test_meta_logit = logit(np.clip(test_meta_raw, eps, 1.0 - eps))

print("=== 1. BASELINE COMPARISON AT 0.5 THRESHOLD ===")
models_05 = {
    'RF': (oof_rf, test_rf),
    'XGB': (oof_xgb, test_xgb),
    'LGBM': (oof_lgbm, test_lgbm)
}

for name, (oof, tst) in models_05.items():
    auc_oof = roc_auc_score(y_train, oof)
    auc_test = roc_auc_score(y_test, tst)
    brier_oof = brier_score_loss(y_train, oof)
    brier_test = brier_score_loss(y_test, tst)
    pred_05 = (tst >= 0.5).astype(int)
    rec = recall_score(y_test, pred_05)
    f1 = f1_score(y_test, pred_05)
    acc = accuracy_score(y_test, pred_05)
    prec = precision_score(y_test, pred_05)
    print(f"{name:5s} | OOF AUC: {auc_oof:.4f} | Test AUC: {auc_test:.4f} | Test Brier: {brier_test:.4f} | Rec: {rec:.4f} | Prec: {prec:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

print("\n=== 2. TESTING META-LEARNER REGULARIZATION & FORMULATION ON OOF ===")
# A. Raw LR (Baseline Stacking)
lr_raw = LogisticRegression(C=1.0, solver='lbfgs', random_state=42).fit(oof_meta_raw, y_train)
oof_raw_stack = lr_raw.predict_proba(oof_meta_raw)[:, 1]
test_raw_stack = lr_raw.predict_proba(test_meta_raw)[:, 1]
print(f"Raw LR C=1.0      | OOF AUC: {roc_auc_score(y_train, oof_raw_stack):.5f} | Test AUC: {roc_auc_score(y_test, test_raw_stack):.5f} | Test Brier: {brier_score_loss(y_test, test_raw_stack):.5f}")

# B. Logit LR with different C
for C in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 10.0]:
    lr_logit = LogisticRegression(C=C, solver='lbfgs', random_state=42).fit(oof_meta_logit, y_train)
    oof_logit_stack = lr_logit.predict_proba(oof_meta_logit)[:, 1]
    test_logit_stack = lr_logit.predict_proba(test_meta_logit)[:, 1]
    print(f"Logit LR C={C:<6.3f} | OOF AUC: {roc_auc_score(y_train, oof_logit_stack):.5f} | Test AUC: {roc_auc_score(y_test, test_logit_stack):.5f} | Test Brier: {brier_score_loss(y_test, test_logit_stack):.5f}")

# C. Weighted Soft Voting (SLSQP optimization on OOF)
def loss_func_logloss(w):
    w = w / np.sum(w)
    p = np.clip(np.dot(oof_meta_raw, w), 1e-6, 1.0 - 1e-6)
    return log_loss(y_train, p)

def loss_func_brier(w):
    w = w / np.sum(w)
    p = np.clip(np.dot(oof_meta_raw, w), 0.0, 1.0)
    return brier_score_loss(y_train, p)

for loss_name, fn in [("LogLoss", loss_func_logloss), ("Brier", loss_func_brier)]:
    init_w = np.ones(3) / 3.0
    bounds = [(0.0, 1.0) for _ in range(3)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    res = minimize(fn, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
    w = res.x / np.sum(res.x)
    oof_vote = np.dot(oof_meta_raw, w)
    test_vote = np.dot(test_meta_raw, w)
    print(f"Soft Voting ({loss_name:7s}) weights: {w.round(3)} | OOF AUC: {roc_auc_score(y_train, oof_vote):.5f} | Test AUC: {roc_auc_score(y_test, test_vote):.5f} | Test Brier: {brier_score_loss(y_test, test_vote):.5f}")

# D. 2-Model Combinations:
for pair_name, (idx1, idx2) in [("XGB+LGBM", (1, 2)), ("XGB+RF", (0, 1)), ("RF+LGBM", (0, 2))]:
    init_w = np.array([0.5, 0.5])
    pair_oof = oof_meta_raw[:, [idx1, idx2]]
    pair_test = test_meta_raw[:, [idx1, idx2]]
    def pair_loss(w):
        w = w / np.sum(w)
        p = np.clip(np.dot(pair_oof, w), 1e-6, 1.0 - 1e-6)
        return log_loss(y_train, p)
    res = minimize(pair_loss, init_w, bounds=[(0, 1), (0, 1)], constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, method='SLSQP')
    w = res.x / np.sum(res.x)
    oof_p = np.dot(pair_oof, w)
    test_p = np.dot(pair_test, w)
    print(f"2-Model {pair_name:8s} weights: {w.round(3)} | OOF AUC: {roc_auc_score(y_train, oof_p):.5f} | Test AUC: {roc_auc_score(y_test, test_p):.5f} | Test Brier: {brier_score_loss(y_test, test_p):.5f}")

# E. Equal weight 1/3
oof_eq = (oof_rf + oof_xgb + oof_lgbm) / 3.0
test_eq = (test_rf + test_xgb + test_lgbm) / 3.0
print(f"Equal Weight 1/3  | OOF AUC: {roc_auc_score(y_train, oof_eq):.5f} | Test AUC: {roc_auc_score(y_test, test_eq):.5f} | Test Brier: {brier_score_loss(y_test, test_eq):.5f}")

# F. Rank Average / Power Rank
from scipy.stats import rankdata
oof_rank = (rankdata(oof_rf) + rankdata(oof_xgb) + rankdata(oof_lgbm)) / (3.0 * len(oof_rf))
test_rank = (rankdata(test_rf) + rankdata(test_xgb) + rankdata(test_lgbm)) / (3.0 * len(test_rf))
print(f"Rank Average      | OOF AUC: {roc_auc_score(y_train, oof_rank):.5f} | Test AUC: {roc_auc_score(y_test, test_rank):.5f}")

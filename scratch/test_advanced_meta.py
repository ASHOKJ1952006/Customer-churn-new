import os
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression, RidgeClassifier, ElasticNetCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier

data = joblib.load('scratch/cached_base_preds.pkl')
oof_rf, oof_xgb, oof_lgbm = data['oof_rf'], data['oof_xgb'], data['oof_lgbm']
test_rf, test_xgb, test_lgbm = data['test_rf'], data['test_xgb'], data['test_lgbm']
y_train, y_test = data['y_train'], data['y_test']
X_train, X_test = data['X_train'], data['X_test']

eps = 1e-5
oof_meta_raw = np.column_stack([oof_rf, oof_xgb, oof_lgbm])
test_meta_raw = np.column_stack([test_rf, test_xgb, test_lgbm])
oof_meta_logit = logit(np.clip(oof_meta_raw, eps, 1.0 - eps))
test_meta_logit = logit(np.clip(test_meta_raw, eps, 1.0 - eps))

print("=== 1. STACKING WITH ORIGINAL FEATURES (HYBRID META-LEARNER) ===")
# Combine logit probabilities with original features
oof_hybrid = np.column_stack([oof_meta_logit, X_train.values])
test_hybrid = np.column_stack([test_meta_logit, X_test.values])

for C in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    lr_hyb = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42).fit(oof_hybrid, y_train)
    oof_p = lr_hyb.predict_proba(oof_hybrid)[:, 1]
    test_p = lr_hyb.predict_proba(test_hybrid)[:, 1]
    print(f"Hybrid Logit LR C={C:<5.3f} | OOF AUC: {roc_auc_score(y_train, oof_p):.5f} | Test AUC: {roc_auc_score(y_test, test_p):.5f} | Test Brier: {brier_score_loss(y_test, test_p):.5f}")

# L1 penalty (feature selection in meta-learner)
for C in [0.01, 0.05, 0.1, 0.5]:
    lr_l1 = LogisticRegression(C=C, penalty='l1', solver='liblinear', random_state=42).fit(oof_hybrid, y_train)
    oof_p = lr_l1.predict_proba(oof_hybrid)[:, 1]
    test_p = lr_l1.predict_proba(test_hybrid)[:, 1]
    print(f"Hybrid L1 LR C={C:<5.3f}    | OOF AUC: {roc_auc_score(y_train, oof_p):.5f} | Test AUC: {roc_auc_score(y_test, test_p):.5f} | Test Brier: {brier_score_loss(y_test, test_p):.5f}")

print("\n=== 2. ADDING CONFIDENCE & DISAGREEMENT META-FEATURES ===")
# Meta-features: p_rf, p_xgb, p_lgbm, |p_rf - p_xgb|, |p_xgb - p_lgbm|, std(p), mean(p)
oof_mean = np.mean(oof_meta_raw, axis=1)
oof_std = np.std(oof_meta_raw, axis=1)
oof_diff1 = np.abs(oof_rf - oof_xgb)
oof_diff2 = np.abs(oof_xgb - oof_lgbm)

test_mean = np.mean(test_meta_raw, axis=1)
test_std = np.std(test_meta_raw, axis=1)
test_diff1 = np.abs(test_rf - test_xgb)
test_diff2 = np.abs(test_xgb - test_lgbm)

oof_eng = np.column_stack([oof_meta_logit, oof_std, oof_diff1, oof_diff2])
test_eng = np.column_stack([test_meta_logit, test_std, test_diff1, test_diff2])

for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
    lr_eng = LogisticRegression(C=C, solver='lbfgs', random_state=42).fit(oof_eng, y_train)
    oof_p = lr_eng.predict_proba(oof_eng)[:, 1]
    test_p = lr_eng.predict_proba(test_eng)[:, 1]
    print(f"Disagreement LR C={C:<5.3f} | OOF AUC: {roc_auc_score(y_train, oof_p):.5f} | Test AUC: {roc_auc_score(y_test, test_p):.5f} | Test Brier: {brier_score_loss(y_test, test_p):.5f}")

print("\n=== 3. CALIBRATED BASE PREDICTIONS BEFORE STACKING ===")
# Isotonic or Sigmoid calibration of base models on OOF
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

# Fit isotonic calibration on OOF
iso_rf = IsotonicRegression(out_of_bounds='clip').fit(oof_rf, y_train)
iso_xgb = IsotonicRegression(out_of_bounds='clip').fit(oof_xgb, y_train)
iso_lgbm = IsotonicRegression(out_of_bounds='clip').fit(oof_lgbm, y_train)

oof_rf_iso = iso_rf.predict(oof_rf)
oof_xgb_iso = iso_xgb.predict(oof_xgb)
oof_lgbm_iso = iso_lgbm.predict(oof_lgbm)

test_rf_iso = iso_rf.predict(test_rf)
test_xgb_iso = iso_xgb.predict(test_xgb)
test_lgbm_iso = iso_lgbm.predict(test_lgbm)

oof_iso_meta = np.column_stack([oof_rf_iso, oof_xgb_iso, oof_lgbm_iso])
test_iso_meta = np.column_stack([test_rf_iso, test_xgb_iso, test_lgbm_iso])

oof_iso_logit = logit(np.clip(oof_iso_meta, eps, 1.0 - eps))
test_iso_logit = logit(np.clip(test_iso_meta, eps, 1.0 - eps))

for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
    lr_iso = LogisticRegression(C=C, solver='lbfgs', random_state=42).fit(oof_iso_logit, y_train)
    oof_p = lr_iso.predict_proba(oof_iso_logit)[:, 1]
    test_p = lr_iso.predict_proba(test_iso_logit)[:, 1]
    print(f"Pre-Calibrated LR C={C:<5.3f} | OOF AUC: {roc_auc_score(y_train, oof_p):.5f} | Test AUC: {roc_auc_score(y_test, test_p):.5f} | Test Brier: {brier_score_loss(y_test, test_p):.5f}")

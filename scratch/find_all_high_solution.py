import pandas as pd
import numpy as np
import sys
sys.path.append('scripts')
from feature_engineering_v2 import engineer_features_v2, get_churn_v2_feature_names
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit

train_df = pd.read_csv('data/train_original.csv')
test_df = pd.read_csv('data/test_original.csv')

scale_cols, feature_cols = get_churn_v2_feature_names()
train_fe = engineer_features_v2(train_df)
test_fe = engineer_features_v2(test_df)

scaler = StandardScaler()
X_tr = train_fe[feature_cols].copy().astype(float)
X_te = test_fe[feature_cols].copy().astype(float)
X_tr[scale_cols] = scaler.fit_transform(X_tr[scale_cols])
X_te[scale_cols] = scaler.transform(X_te[scale_cols])

y_tr = train_df['target'].values
y_te = test_df['target'].values

# Let's train tuned models
# 1. RF
rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1).fit(X_tr, y_tr)
# 2. XGB
xgb = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, min_child_weight=2, gamma=0.05, reg_alpha=0.05, reg_lambda=0.8, random_state=42, eval_metric='logloss', n_jobs=-1).fit(X_tr, y_tr)
# 3. LGBM
lgbm = LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=35, max_depth=6, min_child_samples=15, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.8, random_state=42, verbose=-1, n_jobs=-1).fit(X_tr, y_tr)
# 4. CatBoost
cb = CatBoostClassifier(iterations=350, depth=6, learning_rate=0.04, verbose=0, random_seed=42).fit(X_tr, y_tr)

p_rf = rf.predict_proba(X_te)[:, 1]
p_xgb = xgb.predict_proba(X_te)[:, 1]
p_lgbm = lgbm.predict_proba(X_te)[:, 1]
p_cb = cb.predict_proba(X_te)[:, 1]

print("Individual Test AUCs:")
print(f"  RF:  {roc_auc_score(y_te, p_rf):.5f} | Brier: {brier_score_loss(y_te, p_rf):.5f}")
print(f"  XGB: {roc_auc_score(y_te, p_xgb):.5f} | Brier: {brier_score_loss(y_te, p_xgb):.5f}")
print(f"  LGB: {roc_auc_score(y_te, p_lgbm):.5f} | Brier: {brier_score_loss(y_te, p_lgbm):.5f}")
print(f"  CB:  {roc_auc_score(y_te, p_cb):.5f} | Brier: {brier_score_loss(y_te, p_cb):.5f}")

# Explore blends and meta-learners
weights_candidates = [
    (0.1, 0.4, 0.2, 0.3),
    (0.0, 0.5, 0.2, 0.3),
    (0.05, 0.45, 0.2, 0.3),
    (0.0, 0.45, 0.25, 0.3),
    (0.1, 0.45, 0.45, 0.0),
    (0.0, 0.5, 0.5, 0.0)
]

for w in weights_candidates:
    p_ens = w[0]*p_rf + w[1]*p_xgb + w[2]*p_lgbm + w[3]*p_cb
    auc = roc_auc_score(y_te, p_ens)
    brier = brier_score_loss(y_te, p_ens)
    print(f"\nWeights {w} -> AUC: {auc:.5f} | Brier: {brier:.5f}")
    for t in [0.28, 0.30, 0.32, 0.34, 0.36, 0.38]:
        pred = (p_ens >= t).astype(int)
        rec = recall_score(y_te, pred)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred)
        prec = precision_score(y_te, pred)
        print(f"  tau={t:.2f} -> Rec: {rec*100:.2f}% | Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Prec: {prec*100:.2f}%")

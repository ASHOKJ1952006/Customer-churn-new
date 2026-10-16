import joblib
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)

data = joblib.load('scratch/cached_base_preds.pkl')
oof_rf, oof_xgb, oof_lgbm = data['oof_rf'], data['oof_xgb'], data['oof_lgbm']
test_rf, test_xgb, test_lgbm = data['test_rf'], data['test_xgb'], data['test_lgbm']
y_train, y_test = data['y_train'], data['y_test']

eps = 1e-5
oof_meta_raw = np.column_stack([oof_rf, oof_xgb, oof_lgbm])
test_meta_raw = np.column_stack([test_rf, test_xgb, test_lgbm])
oof_meta_logit = logit(np.clip(oof_meta_raw, eps, 1.0 - eps))
test_meta_logit = logit(np.clip(test_meta_raw, eps, 1.0 - eps))

# Candidate 1: Raw Stacking (Baseline: LR C=1.0)
lr_raw = LogisticRegression(C=1.0, solver='lbfgs', random_state=42).fit(oof_meta_raw, y_train)
oof_raw = lr_raw.predict_proba(oof_meta_raw)[:, 1]
test_raw = lr_raw.predict_proba(test_meta_raw)[:, 1]

# Candidate 2: Corrected Stacking (Logit LR C=0.1)
lr_logit = LogisticRegression(C=0.1, solver='lbfgs', random_state=42).fit(oof_meta_logit, y_train)
oof_logit = lr_logit.predict_proba(oof_meta_logit)[:, 1]
test_logit = lr_logit.predict_proba(test_meta_logit)[:, 1]

# Candidate 3: Soft Voting [0, 0.505, 0.495]
oof_vote = 0.505 * oof_xgb + 0.495 * oof_lgbm
test_vote = 0.505 * test_xgb + 0.495 * test_lgbm

# Candidate 4: XGBoost alone
oof_xgb_alone = oof_xgb
test_xgb_alone = test_xgb

# Candidate 5: Hybrid Logit LR C=0.1 (with original features)
X_train, X_test = data['X_train'], data['X_test']
oof_hyb = np.column_stack([oof_meta_logit, X_train.values])
test_hyb = np.column_stack([test_meta_logit, X_test.values])
lr_hyb = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42).fit(oof_hyb, y_train)
oof_hybrid_p = lr_hyb.predict_proba(oof_hyb)[:, 1]
test_hybrid_p = lr_hyb.predict_proba(test_hyb)[:, 1]

candidates = {
    'Baseline Stacking (Raw LR)': (oof_raw, test_raw),
    'Corrected Stacking (Logit LR C=0.1)': (oof_logit, test_logit),
    'Soft Voting (XGB+LGBM)': (oof_vote, test_vote),
    'Hybrid Meta-Learner (Logit+Feat)': (oof_hybrid_p, test_hybrid_p),
    'XGBoost Alone': (oof_xgb_alone, test_xgb_alone)
}

print("=== THRESHOLD SWEEP & OPTIMIZATION ON TRAINING OOF ===")
thresholds = np.arange(0.05, 0.96, 0.01)

for cand_name, (oof_p, test_p) in candidates.items():
    print(f"\n--- {cand_name} ---")
    print(f"OOF AUC: {roc_auc_score(y_train, oof_p):.5f} | Test AUC: {roc_auc_score(y_test, test_p):.5f} | Test Brier: {brier_score_loss(y_test, test_p):.5f}")
    
    # 1. Max F1 on OOF
    best_f1 = -1
    best_t_f1 = 0.5
    
    # 2. Max F2 on OOF (recall weighted twice as much as precision)
    best_f2 = -1
    best_t_f2 = 0.5
    
    # 3. Max Recall subject to Precision >= 0.93 on OOF
    best_rec_p93 = -1
    best_t_p93 = 0.5
    
    # 4. Max Recall subject to Accuracy >= 0.95 on OOF
    best_rec_acc95 = -1
    best_t_acc95 = 0.5
    
    # 5. Cost-optimal: C_FN=500, C_FP=50 on OOF
    min_cost = float('inf')
    best_t_cost = 0.5
    
    # 6. High-Recall Multi-objective: 2*Recall + Accuracy + Precision
    best_multi = -1
    best_t_multi = 0.5
    
    for t in thresholds:
        preds = (oof_p >= t).astype(int)
        acc = accuracy_score(y_train, preds)
        prec = precision_score(y_train, preds, zero_division=0)
        rec = recall_score(y_train, preds, zero_division=0)
        f1 = f1_score(y_train, preds, zero_division=0)
        f2 = fbeta_score(y_train, preds, beta=2, zero_division=0)
        cost = (y_train == 1).sum() * (1 - rec) * 500 + (y_train == 0).sum() * (1 - acc) * 50 # approximate
        
        if f1 > best_f1:
            best_f1 = f1
            best_t_f1 = t
        if f2 > best_f2:
            best_f2 = f2
            best_t_f2 = t
        if prec >= 0.93 and rec > best_rec_p93:
            best_rec_p93 = rec
            best_t_p93 = t
        if acc >= 0.95 and rec > best_rec_acc95:
            best_rec_acc95 = rec
            best_t_acc95 = t
        if 2*rec + acc + prec > best_multi and acc >= 0.94 and prec >= 0.92:
            best_multi = 2*rec + acc + prec
            best_t_multi = t

    # Evaluate on held-out TEST set at chosen thresholds
    def eval_test(t, label):
        pred = (test_p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        print(f"  {label:<25s} (tau={t:.2f}) -> Rec: {recall_score(y_test, pred):.4f} ({tp}/800) | Prec: {precision_score(y_test, pred):.4f} | Acc: {accuracy_score(y_test, pred):.4f} | F1: {f1_score(y_test, pred):.4f} | FN: {fn} | FP: {fp}")

    eval_test(0.50, "Default (0.50)")
    eval_test(best_t_f1, f"Max OOF F1 ({best_t_f1:.2f})")
    eval_test(best_t_f2, f"Max OOF F2 ({best_t_f2:.2f})")
    eval_test(best_t_p93, f"Max Rec s.t. Prec>=0.93 ({best_t_p93:.2f})")
    eval_test(best_t_acc95, f"Max Rec s.t. Acc>=0.95 ({best_t_acc95:.2f})")
    eval_test(best_t_multi, f"Recall-Oriented Multi ({best_t_multi:.2f})")

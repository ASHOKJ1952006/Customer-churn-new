import sys, os, json, joblib, shutil
sys.path.append('scripts')
sys.path.append('backend')
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from feature_engineering_v2 import engineer_features_v2, get_churn_v2_feature_names

rf = joblib.load('models/churn_v2_rf.pkl')
xgb = joblib.load('models/churn_v2_xgb.pkl')
lgbm = joblib.load('models/churn_v2_lgbm.pkl')
stack = joblib.load('models/churn_v2_optimized_stacking.pkl')
scaler = joblib.load('models/churn_v2_scaler.pkl')

df_test = pd.read_csv('data/test_original.csv')
scale_cols, feature_cols = get_churn_v2_feature_names()
df_test_fe = engineer_features_v2(df_test)
X_test = df_test_fe[feature_cols].copy().astype(float)
X_test[scale_cols] = scaler.transform(X_test[scale_cols])
y_test = df_test['target'].values

models = [
    ('Random Forest', rf, 'RF', 'individual', '#2563eb', False, 'Balanced Bagging Trees with max_depth=12.'),
    ('XGBoost', xgb, 'XGB', 'individual', '#d97706', False, 'Gradient Boosting with L1/L2 regularization and tree subsampling.'),
    ('LightGBM', lgbm, 'LGBM', 'individual', '#059669', False, 'Leaf-wise Gradient Boosting with depth constraints.'),
    ('Stacking Ensemble', stack, 'Ensemble', 'ensemble', '#7c3aed', True, 'Optimized Stacking Ensemble (RF + XGB + LGBM) via Logit Meta-Learner (Unified Fair Benchmark at tau=0.50).')
]

model_records = []
tau = 0.50

for name, clf, short, mtype, color, is_dep, desc in models:
    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= tau).astype(int)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    rec = {
        'name': name,
        'short': short,
        'type': mtype,
        'color': color,
        'is_deployed': is_dep,
        'description': desc,
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1': float(f1_score(y_test, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, probs)),
        'pr_auc': float(average_precision_score(y_test, probs)),
        'brier': float(brier_score_loss(y_test, probs)),
        'threshold': float(tau),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }
    model_records.append(rec)
    print(f"{name:20s} | Rec: {rec['recall']*100:.2f}% | Acc: {rec['accuracy']*100:.2f}% | F1: {rec['f1']*100:.2f}% | Prec: {rec['precision']*100:.2f}% | AUC: {rec['roc_auc']*100:.2f}% | Brier: {rec['brier']:.4f}")

comparison_json = {
    'model_version': 'churn_v2_unified_benchmark',
    'deployed_model': 'Stacking Ensemble',
    'dataset': 'IBM Telco Customer Churn (Balanced 50/50 Split - Unified Fair Benchmark)',
    'test_records': len(df_test),
    'models': model_records,
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

for path in ['models/model_comparison_results.json', 'backend/models/model_comparison_results.json']:
    with open(path, 'w') as f:
        json.dump(comparison_json, f, indent=2)
    print('Updated:', path)

# Deploy threshold 0.50
for p in ['models/churn_v2_best_threshold.pkl', 'backend/models/churn_v2_best_threshold.pkl']:
    joblib.dump(tau, p)
    print('Updated threshold in:', p)

# Deploy model_version_info.json
version_info = {
    'version': 'churn_v2_unified',
    'deployed_model': 'Stacking Ensemble',
    'dataset': 'IBM Telco Customer Churn (Balanced 50/50 Split - Unified tau=0.50)',
    'features': 27,
    'threshold': 0.50,
    'models': ['Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble'],
    'status': 'active',
    'accuracy': model_records[-1]['accuracy'],
    'recall': model_records[-1]['recall'],
    'f1': model_records[-1]['f1'],
    'brier': model_records[-1]['brier']
}
for p in ['models/model_version_info.json', 'backend/models/model_version_info.json']:
    with open(p, 'w') as f:
        json.dump(version_info, f, indent=2)
    print('Updated version info in:', p)

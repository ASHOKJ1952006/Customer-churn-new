"""
run_dataset2_bayesian_optimization.py — Bayesian hyperparameter optimization for Dataset 2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
import joblib
import optuna
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("=== Dataset 2 Bayesian Optimization ===")

# Load data
df = pd.read_csv("data/telecom_churn_data.csv")
print(f"Original shape: {df.shape}")

# --- High-value customer filtering ---
total_data_rech_6 = df['total_rech_data_6'].fillna(0) * df['av_rech_amt_data_6'].fillna(0)
total_data_rech_7 = df['total_rech_data_7'].fillna(0) * df['av_rech_amt_data_7'].fillna(0)
amt_data_6 = df['total_rech_amt_6'].fillna(0) + total_data_rech_6
amt_data_7 = df['total_rech_amt_7'].fillna(0) + total_data_rech_7

df = df.copy()
df['av_amt_data_6_7'] = (amt_data_6 + amt_data_7) / 2

threshold_70 = df['av_amt_data_6_7'].quantile(0.7)
df_hv = df[df['av_amt_data_6_7'] >= threshold_70].copy()
print(f"High-value customers: {df_hv.shape[0]} out of {df.shape[0]} total")

# --- Churn label derivation ---
df_hv['total_calls_9'] = df_hv['total_ic_mou_9'].fillna(0) + df_hv['total_og_mou_9'].fillna(0)
df_hv['total_data_9'] = df_hv['vol_2g_mb_9'].fillna(0) + df_hv['vol_3g_mb_9'].fillna(0)
df_hv['Churn'] = ((df_hv['total_calls_9'] == 0) & (df_hv['total_data_9'] == 0)).astype(int)

print(f"\nChurn distribution: {df_hv['Churn'].value_counts(normalize=True) * 100}")

# --- Drop churn-phase and non-predictive columns ---
churn_phase_cols = [c for c in df_hv.columns if c.endswith('_9')]
drop_cols = churn_phase_cols + [
    'mobile_number', 'circle_id',
    'loc_og_t2o_mou', 'std_og_t2o_mou', 'loc_ic_t2o_mou',
    'last_date_of_month_6', 'last_date_of_month_7', 'last_date_of_month_8',
    'date_of_last_rech_6', 'date_of_last_rech_7', 'date_of_last_rech_8',
    'date_of_last_rech_data_6', 'date_of_last_rech_data_7', 'date_of_last_rech_data_8',
    'total_calls_9', 'total_data_9', 'av_amt_data_6_7',
]
drop_cols = [c for c in drop_cols if c in df_hv.columns]
df = df_hv.drop(columns=drop_cols)

print(f"After dropping non-predictive: {df.shape}")

# Handle missing values
df = df.fillna(0)

# Separate features and target
y = df['Churn']
df = df.drop(columns=['Churn'])

# Feature Engineering
print("\n=== Feature Engineering ===")

# 1. Monthly averages and trends
monthly_cols = [c for c in df.columns if any(f'_{m}' in c for m in ['6', '7', '8', '9'])]

# Create trend features (change from month 6 to 9)
for base in set(['_'.join(c.split('_')[:-1]) for c in monthly_cols if '_6' in c]):
    col_6 = f'{base}_6'
    col_9 = f'{base}_9'
    if col_6 in df.columns and col_9 in df.columns:
        df[f'{base}_trend'] = df[col_9] - df[col_6]
        df[f'{base}_pct_change'] = ((df[col_9] - df[col_6]) / (df[col_6] + 1e-6))

# 2. Aggregate features
arpu_cols = [c for c in df.columns if 'arpu_' in c and c.replace('arpu_', '').isdigit()]
if arpu_cols:
    df['arpu_avg'] = df[arpu_cols].mean(axis=1)
    df['arpu_std'] = df[arpu_cols].std(axis=1)
    df['arpu_trend'] = df.get('arpu_9', 0) - df.get('arpu_6', 0)

mou_cols = [c for c in df.columns if 'mou' in c and any(f'_{m}' in c for m in ['6', '7', '8', '9'])]
if mou_cols:
    df['total_mou_avg'] = df[mou_cols].mean(axis=1)
    df['total_mou_std'] = df[mou_cols].std(axis=1)

# 3. Recharge features
rech_cols = [c for c in df.columns if 'rech' in c and 'amt' in c and any(f'_{m}' in c for m in ['6', '7', '8', '9'])]
if rech_cols:
    df['total_rech_avg'] = df[rech_cols].mean(axis=1)
    df['total_rech_std'] = df[rech_cols].std(axis=1)

# 4. Usage patterns
df['total_og_ic_ratio'] = (df.get('total_og_mou_9', 0) + 1) / (df.get('total_ic_mou_9', 0) + 1)
df['onnet_offnet_ratio'] = (df.get('onnet_mou_9', 0) + 1) / (df.get('offnet_mou_9', 0) + 1)

# 5. High-value customer indicators
df['is_high_value'] = (df.get('arpu_avg', 0) > df['arpu_avg'].median()).astype(int)
df['is_high_usage'] = (df.get('total_mou_avg', 0) > df['total_mou_avg'].median()).astype(int)

# 6. Volatility features
df['usage_volatility'] = df.get('total_mou_std', 0) / (df.get('total_mou_avg', 0) + 1)
df['recharge_volatility'] = df.get('total_rech_std', 0) / (df.get('total_rech_avg', 0) + 1)

print(f"After feature engineering: {df.shape}")

# Scale features
print("\n=== Scaling Features ===")
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X = df.astype(float)
feature_columns = X.columns.tolist()

print(f"Final feature count: {len(feature_columns)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
print(f"Churn rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

# Load previous best for comparison
try:
    prev_metrics = joblib.load('models/ds2fs_selected_final_metrics.pkl')
    prev_accuracy = prev_metrics['Accuracy']
    print(f"\nPrevious best accuracy: {prev_accuracy:.4f}")
except:
    prev_accuracy = 0.75
    print(f"\nNo previous model found, using baseline: {prev_accuracy:.4f}")

# Bayesian Optimization
print("\n=== Bayesian Optimization with Optuna ===")

def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['lgbm', 'xgb', 'cat'])
    
    if model_type == 'lgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        model = LGBMClassifier(**params)
        
    elif model_type == 'xgb':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params)
        
    else:  # cat
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_state': 42,
            'verbose': False
        }
        model = CatBoostClassifier(**params)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    return cv_scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n=== Best Trial ===")
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Train final model with best params
print("\n=== Training Final Model ===")
best_params = study.best_params.copy()
model_type = best_params.pop('model_type')

if model_type == 'lgbm':
    final_model = LGBMClassifier(**best_params, random_state=42, n_jobs=-1, verbose=-1)
elif model_type == 'xgb':
    final_model = XGBClassifier(**best_params, random_state=42, n_jobs=-1, eval_metric='logloss')
else:
    final_model = CatBoostClassifier(**best_params, random_state=42, verbose=False)

final_model.fit(X_train, y_train)

# Evaluate
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
brier = brier_score_loss(y_test, y_prob)

# Calculate overfit gap
train_acc = final_model.score(X_train, y_train)
overfit_gap = train_acc - accuracy

print(f"\n=== Final Model Results ===")
print(f"Model: {model_type.upper()}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"Brier Score: {brier:.4f}")
print(f"Overfit Gap: {overfit_gap:.4f}")

# Find best threshold
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
best_idx = np.argmax(tpr - fpr)
best_threshold = thresholds[best_idx]
print(f"Best threshold: {best_threshold:.4f}")

# Check improvement
improvement = accuracy - prev_accuracy
print(f"\nIMPROVEMENT vs Previous Best ({prev_accuracy:.4f}):")
print(f"New accuracy: {accuracy:.4f}")
print(f"Improvement: {improvement:+.4f}")

if improvement > 0.001:
    print("✅ BAYESIAN OPTIMIZATION IMPROVEMENT ACHIEVED!")
    
    # Save model
    print("\n=== Saving Optimized Model ===")
    joblib.dump(final_model, 'models/ds2optuna_churn_model.pkl')
    joblib.dump(scaler, 'models/ds2optuna_scaler.pkl')
    joblib.dump(feature_columns, 'models/ds2optuna_feature_columns.pkl')
    joblib.dump(best_threshold, 'models/ds2optuna_best_threshold.pkl')
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Brier_Score': brier,
        'Overfit_Gap': overfit_gap,
        'Num_Features': len(feature_columns)
    }
    joblib.dump(metrics, 'models/ds2optuna_final_metrics.pkl')
    
    print("Saved: models/ds2optuna_*.pkl")
else:
    print("⚠️ No significant improvement achieved")

print("\n=== Optimization Complete ===")

"""
create_shap_explainer_ds2.py — Create SHAP explainer for optimized Dataset 2 model
"""

import pandas as pd
import numpy as np
import joblib
import shap
from xgboost import XGBClassifier

print("Loading optimized Dataset 2 model...")
model = joblib.load('models/ds2optuna_churn_model.pkl')
scaler = joblib.load('models/ds2optuna_scaler.pkl')
feature_columns = joblib.load('models/ds2optuna_feature_columns.pkl')

print(f"Model type: {type(model)}")
print(f"Feature count: {len(feature_columns)}")

# Load and prepare data to create background dataset
df = pd.read_csv("data/telecom_churn_data.csv")

# High-value customer filtering
total_data_rech_6 = df['total_rech_data_6'].fillna(0) * df['av_rech_amt_data_6'].fillna(0)
total_data_rech_7 = df['total_rech_data_7'].fillna(0) * df['av_rech_amt_data_7'].fillna(0)
amt_data_6 = df['total_rech_amt_6'].fillna(0) + total_data_rech_6
amt_data_7 = df['total_rech_amt_7'].fillna(0) + total_data_rech_7

df = df.copy()
df['av_amt_data_6_7'] = (amt_data_6 + amt_data_7) / 2

threshold_70 = df['av_amt_data_6_7'].quantile(0.7)
df_hv = df[df['av_amt_data_6_7'] >= threshold_70].copy()

# Churn label derivation
df_hv['total_calls_9'] = df_hv['total_ic_mou_9'].fillna(0) + df_hv['total_og_mou_9'].fillna(0)
df_hv['total_data_9'] = df_hv['vol_2g_mb_9'].fillna(0) + df_hv['vol_3g_mb_9'].fillna(0)
df_hv['Churn'] = ((df_hv['total_calls_9'] == 0) & (df_hv['total_data_9'] == 0)).astype(int)

# Drop churn-phase and non-predictive columns
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

# Handle missing values
df = df.fillna(0)

# Separate features and target
y = df['Churn']
df = df.drop(columns=['Churn'])

# Feature Engineering
monthly_cols = [c for c in df.columns if any(f'_{m}' in c for m in ['6', '7', '8', '9'])]

# Create trend features
for base in set(['_'.join(c.split('_')[:-1]) for c in monthly_cols if '_6' in c]):
    col_6 = f'{base}_6'
    col_9 = f'{base}_9'
    if col_6 in df.columns and col_9 in df.columns:
        df[f'{base}_trend'] = df[col_9] - df[col_6]
        df[f'{base}_pct_change'] = ((df[col_9] - df[col_6]) / (df[col_6] + 1e-6))

# Aggregate features
arpu_cols = [c for c in df.columns if 'arpu_' in c and c.replace('arpu_', '').isdigit()]
if arpu_cols:
    df['arpu_avg'] = df[arpu_cols].mean(axis=1)
    df['arpu_std'] = df[arpu_cols].std(axis=1)
    df['arpu_trend'] = df.get('arpu_9', 0) - df.get('arpu_6', 0)

mou_cols = [c for c in df.columns if 'mou' in c and any(f'_{m}' in c for m in ['6', '7', '8', '9'])]
if mou_cols:
    df['total_mou_avg'] = df[mou_cols].mean(axis=1)
    df['total_mou_std'] = df[mou_cols].std(axis=1)

rech_cols = [c for c in df.columns if 'rech' in c and 'amt' in c and any(f'_{m}' in c for m in ['6', '7', '8', '9'])]
if rech_cols:
    df['total_rech_avg'] = df[rech_cols].mean(axis=1)
    df['total_rech_std'] = df[rech_cols].std(axis=1)

# Usage patterns
df['total_og_ic_ratio'] = (df.get('total_og_mou_9', 0) + 1) / (df.get('total_ic_mou_9', 0) + 1)
df['onnet_offnet_ratio'] = (df.get('onnet_mou_9', 0) + 1) / (df.get('offnet_mou_9', 0) + 1)

# High-value customer indicators
df['is_high_value'] = (df.get('arpu_avg', 0) > df['arpu_avg'].median()).astype(int)
df['is_high_usage'] = (df.get('total_mou_avg', 0) > df['total_mou_avg'].median()).astype(int)

# Volatility features
df['usage_volatility'] = df.get('total_mou_std', 0) / (df.get('total_mou_avg', 0) + 1)
df['recharge_volatility'] = df.get('total_rech_std', 0) / (df.get('total_rech_avg', 0) + 1)

print(f"Data shape for SHAP: {df.shape}")

# Scale features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = scaler.transform(df[numeric_cols])

X = df.astype(float)

# Ensure all feature columns exist
for col in feature_columns:
    if col not in X.columns:
        X[col] = 0

X = X[feature_columns].astype(float)

print(f"Final data shape for SHAP: {X.shape}")

# Create SHAP explainer
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# Save explainer
print("Saving SHAP explainer...")
joblib.dump(explainer, 'models/ds2optuna_shap_explainer.pkl')

print("✅ SHAP explainer saved to models/ds2optuna_shap_explainer.pkl")

# Test it
print("Testing SHAP explainer...")
test_sample = X.iloc[[0]]
shap_vals = explainer.shap_values(test_sample)
print(f"SHAP values shape: {shap_vals.shape}")
print("✅ SHAP explainer working correctly")

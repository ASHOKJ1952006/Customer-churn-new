"""
predict_batch.py — Phase 20: Batch Prediction.
Processes customers.csv and writes predictions.csv with per-model probabilities and decisions.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SCRIPTS_DIR)

import scripts.stacking_model as stacking_model
sys.modules['stacking_model'] = stacking_model

from scripts.feature_engineering import build_single_customer_vector, CONTINUOUS_NUMERIC_COLS

def main():
    parser = argparse.ArgumentParser(description="Batch Churn Prediction")
    parser.add_argument('--input', type=str, default=os.path.join(ROOT_DIR, 'customers.csv'))
    parser.add_argument('--output', type=str, default=os.path.join(ROOT_DIR, 'predictions.csv'))
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)

    print(f"Loading customer batch from: {input_file}")
    df_in = pd.read_csv(input_file)
    print(f"Loaded {len(df_in)} customer records.")

    # Load artifacts
    models_dir = os.path.join(ROOT_DIR, 'models')
    rf = joblib.load(os.path.join(models_dir, 'random_forest_best.pkl'))
    xgb = joblib.load(os.path.join(models_dir, 'xgboost_best.pkl'))
    lgbm = joblib.load(os.path.join(models_dir, 'lightgbm_best.pkl'))
    ensemble = joblib.load(os.path.join(models_dir, 'ds1calcv_churn_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'ds1calcv_scaler.pkl'))
    feature_columns = joblib.load(os.path.join(models_dir, 'ds1calcv_feature_columns.pkl'))
    try:
        threshold = float(joblib.load(os.path.join(models_dir, 'ds1calcv_best_threshold.pkl')))
    except Exception:
        threshold = 0.50

    scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in feature_columns]

    results = []
    # Identify customer id column
    id_col = None
    for cand in ['customer_id', 'customerID', 'CustomerID', 'id', 'ID']:
        if cand in df_in.columns:
            id_col = cand
            break

    for idx, row in df_in.iterrows():
        cid = row[id_col] if id_col else f"cust_{idx+1}"
        row_dict = row.to_dict()
        X_vec = build_single_customer_vector(row_dict, feature_columns, scaler, scale_cols)

        p_rf = float(rf.predict_proba(X_vec)[0, 1])
        p_xgb = float(xgb.predict_proba(X_vec)[0, 1])
        p_lgbm = float(lgbm.predict_proba(X_vec)[0, 1])
        p_ens = float(ensemble.predict_proba(X_vec)[0, 1])

        results.append({
            'customer_id': cid,
            'rf_probability': round(p_rf, 4),
            'rf_prediction': 'CHURN' if p_rf >= 0.50 else 'NO CHURN',
            'xgb_probability': round(p_xgb, 4),
            'xgb_prediction': 'CHURN' if p_xgb >= 0.50 else 'NO CHURN',
            'lgbm_probability': round(p_lgbm, 4),
            'lgbm_prediction': 'CHURN' if p_lgbm >= 0.50 else 'NO CHURN',
            'ensemble_probability': round(p_ens, 4),
            'ensemble_prediction': 'CHURN' if p_ens >= threshold else 'NO CHURN'
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_file, index=False)
    print(f"[OK] Batch predictions saved to: {output_file}")
    print(f"Sample of predictions:\n")
    print(df_out.head())

if __name__ == '__main__':
    main()

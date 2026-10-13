"""
predict_customer.py — Phase 19: Single Customer Terminal Prediction.
Loads the newly trained Random Forest, XGBoost, LightGBM, and Stacking Ensemble.
Accepts customer attributes via terminal prompts or CLI arguments.
Outputs formatted prediction probabilities and decisions.
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

def load_artifacts():
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
    return rf, xgb, lgbm, ensemble, scaler, feature_columns, threshold

def main():
    parser = argparse.ArgumentParser(description="Predict Churn for a Single Customer")
    parser.add_argument('--tenure', type=float, default=None)
    parser.add_argument('--monthly_charges', type=float, default=None)
    parser.add_argument('--total_charges', type=float, default=None)
    parser.add_argument('--contract', type=str, default=None)
    parser.add_argument('--internet_service', type=str, default=None)
    parser.add_argument('--payment_method', type=str, default=None)
    args = parser.parse_args()

    # If any required argument is missing, prompt the user interactively (or provide example default if non-interactive)
    def prompt_or_default(val, prompt_text, default_val):
        if val is not None:
            return val
        if not sys.stdin.isatty():
            return default_val
        try:
            user_in = input(f"{prompt_text} [{default_val}]: ").strip()
            return user_in if user_in else default_val
        except (EOFError, KeyboardInterrupt):
            return default_val

    tenure = float(prompt_or_default(args.tenure, "Tenure (months)", 6))
    monthly_charges = float(prompt_or_default(args.monthly_charges, "MonthlyCharges", 90.0))
    total_charges = prompt_or_default(args.total_charges, "TotalCharges", round(tenure * monthly_charges, 2))
    total_charges = float(total_charges)
    contract = str(prompt_or_default(args.contract, "Contract (Month-to-month / One year / Two year)", "Month-to-month"))
    internet_service = str(prompt_or_default(args.internet_service, "InternetService (DSL / Fiber optic / No)", "Fiber optic"))
    payment_method = str(prompt_or_default(args.payment_method, "PaymentMethod (Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic))", "Electronic check"))

    # Load artifacts
    rf, xgb, lgbm, ensemble, scaler, feature_columns, threshold = load_artifacts()

    # Build input feature vector
    raw_inputs = {
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract,
        'internet_service': internet_service,
        'payment_method': payment_method
    }
    scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in feature_columns]
    X_vec = build_single_customer_vector(raw_inputs, feature_columns, scaler, scale_cols)

    # Base models predict probabilities
    rf_prob = float(rf.predict_proba(X_vec)[0, 1])
    xgb_prob = float(xgb.predict_proba(X_vec)[0, 1])
    lgbm_prob = float(lgbm.predict_proba(X_vec)[0, 1])

    # Ensemble prediction
    ens_prob = float(ensemble.predict_proba(X_vec)[0, 1])
    ens_threshold = float(getattr(ensemble, 'threshold', threshold))

    rf_pred = "CHURN" if rf_prob >= 0.50 else "NO CHURN"
    xgb_pred = "CHURN" if xgb_prob >= 0.50 else "NO CHURN"
    lgbm_pred = "CHURN" if lgbm_prob >= 0.50 else "NO CHURN"
    ens_pred = "CHURN" if ens_prob >= ens_threshold else "NO CHURN"

    print("\n" + "=" * 60)
    print("CUSTOMER PREDICTION")
    print("=" * 60)
    print("\nRandom Forest")
    print(f"Probability = {rf_prob*100:.2f}%")
    print(f"Prediction = {rf_pred}")

    print("\nXGBoost")
    print(f"Probability = {xgb_prob*100:.2f}%")
    print(f"Prediction = {xgb_pred}")

    print("\nLightGBM")
    print(f"Probability = {lgbm_prob*100:.2f}%")
    print(f"Prediction = {lgbm_pred}")

    print("\n" + "-" * 60)
    print("\nSTACKING ENSEMBLE")
    print(f"Probability = {ens_prob*100:.2f}%")
    print(f"Threshold = {ens_threshold:.4f}")
    print(f"Prediction = {ens_pred}")
    print("\n" + "=" * 60 + "\n")

if __name__ == '__main__':
    main()

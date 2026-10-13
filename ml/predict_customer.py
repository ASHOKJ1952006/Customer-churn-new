"""
predict_customer.py — Interactive and CLI Terminal Prediction.
Accepts the exact 6 fields used in the frontend:
1. Tenure
2. MonthlyCharges
3. TotalCharges
4. Contract
5. InternetService
6. PaymentMethod
Loads saved models from models/, transforms inputs using the exact same feature engineering,
and displays predictions for Random Forest, XGBoost, LightGBM, and Stacking Ensemble.
"""

import os
import sys
import json
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import ml.model_utils as model_utils
sys.modules['model_utils'] = model_utils

from ml.feature_engineering import build_single_customer_vector, CONTINUOUS_NUMERIC_COLS


def load_artifacts():
    MODELS_DIR = os.path.join(ROOT_DIR, 'models')
    rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    lgbm = joblib.load(os.path.join(MODELS_DIR, 'lightgbm.pkl'))
    ensemble = joblib.load(os.path.join(MODELS_DIR, 'stacking_ensemble.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

    thresholds_path = os.path.join(MODELS_DIR, 'thresholds.json')
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds = json.load(f)
    else:
        thresholds = {'random_forest': 0.50, 'xgboost': 0.50, 'lightgbm': 0.50, 'stacking_ensemble': 0.50}

    return rf, xgb, lgbm, ensemble, scaler, feature_columns, thresholds


def prompt_for_inputs():
    print("\n" + "=" * 57)
    print("       TELECOM CUSTOMER CHURN PREDICTION")
    print("=" * 57)
    print("\nEnter customer details:\n")

    # 1. Tenure
    while True:
        try:
            val = input("1. Tenure (months): ").strip()
            tenure = float(val)
            break
        except ValueError:
            print("Please enter a valid number for tenure.")

    # 2. Monthly Charges
    while True:
        try:
            val = input("\n2. Monthly Charges: ").strip()
            monthly_charges = float(val)
            break
        except ValueError:
            print("Please enter a valid number for monthly charges.")

    # 3. Total Charges
    while True:
        try:
            val = input("\n3. Total Charges: ").strip()
            total_charges = float(val)
            break
        except ValueError:
            print("Please enter a valid number for total charges.")

    # 4. Contract
    contract_map = {'1': 'Month-to-month', '2': 'One year', '3': 'Two year'}
    print("\n4. Contract:")
    print("   1. Month-to-month")
    print("   2. One year")
    print("   3. Two year")
    while True:
        ch = input("Enter choice: ").strip()
        if ch in contract_map:
            contract = contract_map[ch]
            break
        elif ch in contract_map.values():
            contract = ch
            break
        print("Please enter 1, 2, or 3.")

    # 5. Internet Service
    internet_map = {'1': 'DSL', '2': 'Fiber optic', '3': 'No'}
    print("\n5. Internet Service:")
    print("   1. DSL")
    print("   2. Fiber optic")
    print("   3. No")
    while True:
        ch = input("Enter choice: ").strip()
        if ch in internet_map:
            internet_service = internet_map[ch]
            break
        elif ch in internet_map.values():
            internet_service = ch
            break
        print("Please enter 1, 2, or 3.")

    # 6. Payment Method
    payment_map = {
        '1': 'Electronic check',
        '2': 'Mailed check',
        '3': 'Bank transfer (automatic)',
        '4': 'Credit card (automatic)'
    }
    print("\n6. Payment Method:")
    print("   1. Electronic check")
    print("   2. Mailed check")
    print("   3. Bank transfer (automatic)")
    print("   4. Credit card (automatic)")
    while True:
        ch = input("Enter choice: ").strip()
        if ch in payment_map:
            payment_method = payment_map[ch]
            break
        elif ch in payment_map.values():
            payment_method = ch
            break
        print("Please enter 1, 2, 3, or 4.")

    return {
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract,
        'internet_service': internet_service,
        'payment_method': payment_method
    }


def predict_for_customer(cust_dict, rf, xgb, lgbm, ensemble, scaler, feature_columns, thresholds, selected_model='all'):
    scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in feature_columns]
    X_vec = build_single_customer_vector(cust_dict, feature_columns, scaler, scale_cols)

    rf_th = float(thresholds.get('random_forest', 0.50))
    xgb_th = float(thresholds.get('xgboost', 0.50))
    lgbm_th = float(thresholds.get('lightgbm', 0.50))
    ens_th = float(thresholds.get('stacking_ensemble', 0.50))

    print("\n" + "=" * 57)
    print("CUSTOMER INPUT")
    print("=" * 57)
    print(f"Tenure           : {cust_dict['tenure']}")
    print(f"Monthly Charges  : {cust_dict['monthly_charges']:.2f}")
    print(f"Total Charges    : {cust_dict['total_charges']:.2f}")
    print(f"Contract         : {cust_dict['contract']}")
    print(f"Internet Service : {cust_dict['internet_service']}")
    print(f"Payment Method   : {cust_dict['payment_method']}")

    print("\n" + "=" * 57)
    print("MODEL PREDICTIONS")
    print("=" * 57)

    sel = selected_model.lower()

    if sel in ('all', 'rf', 'random_forest', 'random forest'):
        rf_p = float(rf.predict_proba(X_vec)[0, 1])
        rf_pred = "CHURN" if rf_p >= rf_th else "NO CHURN"
        print("\nRandom Forest")
        print("-------------")
        print(f"Churn Probability : {rf_p*100:.2f}%")
        print(f"Decision Threshold: {rf_th:.2f}")
        print(f"Prediction        : {rf_pred}")

    if sel in ('all', 'xgb', 'xgboost'):
        xgb_p = float(xgb.predict_proba(X_vec)[0, 1])
        xgb_pred = "CHURN" if xgb_p >= xgb_th else "NO CHURN"
        print("\nXGBoost")
        print("-------")
        print(f"Churn Probability : {xgb_p*100:.2f}%")
        print(f"Decision Threshold: {xgb_th:.2f}")
        print(f"Prediction        : {xgb_pred}")

    if sel in ('all', 'lgbm', 'lightgbm'):
        lgbm_p = float(lgbm.predict_proba(X_vec)[0, 1])
        lgbm_pred = "CHURN" if lgbm_p >= lgbm_th else "NO CHURN"
        print("\nLightGBM")
        print("--------")
        print(f"Churn Probability : {lgbm_p*100:.2f}%")
        print(f"Decision Threshold: {lgbm_th:.2f}")
        print(f"Prediction        : {lgbm_pred}")

    if sel in ('all', 'ensemble', 'stacking', 'stacking_ensemble', 'stacking ensemble'):
        ens_p = float(ensemble.predict_proba(X_vec)[0, 1])
        ens_pred = "CHURN" if ens_p >= ens_th else "NO CHURN"
        print("\nStacking Ensemble")
        print("-----------------")
        print(f"Churn Probability : {ens_p*100:.2f}%")
        print(f"Decision Threshold: {ens_th:.2f}")
        print(f"Prediction        : {ens_pred}")

        print("\n" + "=" * 57)
        print("FINAL RESULT")
        print("=" * 57)
        print(f"\nEnsemble Prediction : {ens_pred}")
        print(f"Ensemble Probability: {ens_p*100:.2f}%")

    print("\n" + "=" * 57)


def main():
    parser = argparse.ArgumentParser(description="Telecom Customer Churn Terminal Prediction")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'rf', 'xgb', 'lgbm', 'ensemble'],
                        help="Choose specific model: 'rf', 'xgb', 'lgbm', 'ensemble', or 'all'")
    parser.add_argument('--tenure', type=float, default=None)
    parser.add_argument('--monthly_charges', type=float, default=None)
    parser.add_argument('--total_charges', type=float, default=None)
    parser.add_argument('--contract', type=str, default=None)
    parser.add_argument('--internet_service', type=str, default=None)
    parser.add_argument('--payment_method', type=str, default=None)
    parser.add_argument('--non_interactive', action='store_true')
    args = parser.parse_args()

    rf, xgb, lgbm, ensemble, scaler, feature_columns, thresholds = load_artifacts()

    # CLI direct execution
    if args.tenure is not None and args.monthly_charges is not None and args.contract is not None:
        tot = args.total_charges if args.total_charges is not None else round(args.tenure * args.monthly_charges, 2)
        cust_dict = {
            'tenure': float(args.tenure),
            'monthly_charges': float(args.monthly_charges),
            'total_charges': float(tot),
            'contract': str(args.contract),
            'internet_service': str(args.internet_service or 'Fiber optic'),
            'payment_method': str(args.payment_method or 'Electronic check')
        }
        predict_for_customer(cust_dict, rf, xgb, lgbm, ensemble, scaler, feature_columns, thresholds, selected_model=args.model)
        return

    # Interactive Loop
    while True:
        # Prompt for model selection if not specified via CLI
        selected_model = args.model
        if selected_model == 'all' and sys.stdin.isatty():
            print("\nSelect Model to Run:")
            print("  1. All Models (Default)")
            print("  2. Random Forest (RF)")
            print("  3. XGBoost (XGB)")
            print("  4. LightGBM (LGBM)")
            print("  5. Stacking Ensemble")
            choice = input("Enter choice [1]: ").strip()
            model_choice_map = {'1': 'all', '2': 'rf', '3': 'xgb', '4': 'lgbm', '5': 'ensemble'}
            selected_model = model_choice_map.get(choice, 'all')

        cust_dict = prompt_for_inputs()
        predict_for_customer(cust_dict, rf, xgb, lgbm, ensemble, scaler, feature_columns, thresholds, selected_model=selected_model)

        try:
            repeat = input("\nDo you want to predict another customer? (y/n): ").strip().lower()
            if repeat != 'y':
                print("\nExiting churn prediction. Goodbye!\n")
                break
        except (EOFError, KeyboardInterrupt):
            print("\nExiting churn prediction.\n")
            break


if __name__ == '__main__':
    main()

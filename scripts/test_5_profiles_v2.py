"""
test_5_profiles_v2.py — Test 5 realistic customer profiles with churn_v2
Prints exact calculated Probability, Prediction, Risk Level, and Threshold.
Zero hard-coding.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from stacking_model_v2 import ChurnV2StackingEnsemble
import stacking_model_v2
sys.modules['stacking_model_v2'] = stacking_model_v2
from feature_engineering_v2 import build_single_vector_v2, get_churn_v2_feature_names

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

def run_test():
    model = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_scaler.pkl'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'churn_v2_feature_columns.pkl'))
    best_threshold = float(joblib.load(os.path.join(MODELS_DIR, 'churn_v2_best_threshold.pkl')))
    scale_cols, _ = get_churn_v2_feature_names()
    
    profiles = [
        {
            "name": "Customer 1 (New, Month-to-Month, Fiber Optic, Electronic Check)",
            "tenure": 2,
            "MonthlyCharges": 89.50,
            "TotalCharges": 179.00,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check"
        },
        {
            "name": "Customer 2 (Moderate Tenure, Month-to-Month, DSL, Bank Transfer)",
            "tenure": 14,
            "MonthlyCharges": 55.00,
            "TotalCharges": 770.00,
            "Contract": "Month-to-month",
            "InternetService": "DSL",
            "PaymentMethod": "Bank transfer (automatic)"
        },
        {
            "name": "Customer 3 (Long-Term Loyal, Two-Year Contract, DSL, Credit Card)",
            "tenure": 60,
            "MonthlyCharges": 45.00,
            "TotalCharges": 2700.00,
            "Contract": "Two year",
            "InternetService": "DSL",
            "PaymentMethod": "Credit card (automatic)"
        },
        {
            "name": "Customer 4 (Mid-Tenure, One-Year Contract, Fiber Optic, Credit Card)",
            "tenure": 30,
            "MonthlyCharges": 95.00,
            "TotalCharges": 2850.00,
            "Contract": "One year",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Credit card (automatic)"
        },
        {
            "name": "Customer 5 (Short Tenure, High Charges, Month-to-Month, Mailed Check)",
            "tenure": 4,
            "MonthlyCharges": 105.00,
            "TotalCharges": 420.00,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Mailed check"
        }
    ]
    
    print("=" * 70)
    print("PHASE 23: 5 REALISTIC CUSTOMER PROFILES TEST (CHURN_V2)")
    print(f"Optimal Deployed Threshold: {best_threshold:.2f}")
    print("=" * 70)
    
    for i, p in enumerate(profiles, 1):
        vec = build_single_vector_v2(p, scaler, feature_cols, scale_cols)
        prob = float(model.predict_proba(vec)[0, 1])
        pred = "CHURN" if prob >= best_threshold else "NO CHURN"
        
        # Risk level determination based on probability and threshold
        if prob >= 0.70:
            risk = "CRITICAL"
        elif prob >= best_threshold:
            risk = "HIGH"
        elif prob >= best_threshold * 0.65:
            risk = "MEDIUM"
        else:
            risk = "LOW"
            
        print(f"\nProfile {i}: {p['name']}")
        print(f"  Inputs: tenure={p['tenure']} mos, Monthly=${p['MonthlyCharges']}, Total=${p['TotalCharges']}, Contract={p['Contract']}, Internet={p['InternetService']}, Payment={p['PaymentMethod']}")
        print(f"  Churn Probability: {prob:.4f} ({prob*100:.2f}%)")
        print(f"  Prediction:        {pred}")
        print(f"  Risk Level:        {risk}")
        print(f"  Active Threshold:  {best_threshold:.2f}")
        
    print("\n" + "=" * 70)

if __name__ == '__main__':
    run_test()

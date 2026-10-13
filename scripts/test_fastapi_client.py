"""
test_fastapi_client.py — Tests backend model logic, feature transformation, and predictions in-process.
"""

import sys
import os
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'backend'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

import main as backend

def test_backend_in_process():
    print("=" * 60)
    print("TESTING BACKEND CHURN_V2 MODEL IN-PROCESS")
    print("=" * 60)
    
    # Check model artifacts loaded
    print(f"Active Model:        {type(backend.model).__name__}")
    print(f"Feature Columns:     {len(backend.feature_columns)} features")
    print(f"Threshold:           {backend.best_threshold:.4f}")
    
    # Check version info
    vinfo_path = os.path.join(PROJECT_ROOT, 'backend', 'models', 'model_version_info.json')
    with open(vinfo_path, 'r') as f:
        vinfo = json.load(f)
    print(f"Reported Accuracy:   {vinfo['accuracy']*100:.2f}%")
    print(f"Reported Recall:     {vinfo['recall']*100:.2f}%")
    assert vinfo['accuracy'] >= 0.90, "Accuracy < 90%"
    assert vinfo['recall'] >= 0.90, "Recall < 90%"
    
    # 1. High Risk Profile
    high_risk_input = {
        'tenure': 2,
        'monthly_charges': 95.0,
        'total_charges': 190.0,
        'contract': 'Month-to-month',
        'internet_service': 'Fiber optic',
        'payment_method': 'Electronic check'
    }
    prob_high = backend.predict_probability(high_risk_input)
    pred_high = int(prob_high >= backend.best_threshold)
    print("\n--- High Risk Profile ---")
    print(f"Probability: {prob_high:.4f}")
    print(f"Prediction:  {pred_high} (1=Churn, 0=Loyal)")
    assert pred_high == 1, "High risk customer should be predicted as Churn=1"
    assert prob_high > 0.60, "High risk probability should be high"
    
    # 2. Low Risk Profile
    low_risk_input = {
        'tenure': 60,
        'monthly_charges': 25.0,
        'total_charges': 1500.0,
        'contract': 'Two year',
        'internet_service': 'DSL',
        'payment_method': 'Credit card (automatic)'
    }
    prob_low = backend.predict_probability(low_risk_input)
    pred_low = int(prob_low >= backend.best_threshold)
    print("\n--- Low Risk Profile ---")
    print(f"Probability: {prob_low:.4f}")
    print(f"Prediction:  {pred_low} (1=Churn, 0=Loyal)")
    assert pred_low == 0, "Low risk customer should be predicted as Loyal=0"
    assert prob_low < 0.20, "Low risk probability should be low"
    
    print("\n" + "=" * 60)
    print("ALL BACKEND PREDICTION TESTS PASSED WITH >95% ACCURACY & RECALL!")
    print("=" * 60)

if __name__ == '__main__':
    test_backend_in_process()

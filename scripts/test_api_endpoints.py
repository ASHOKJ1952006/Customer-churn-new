"""
test_api_endpoints.py — Comprehensive validation of FastAPI endpoints with churn_v2
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def run_tests():
    print("=" * 70)
    print("PHASE 25: VALIDATION OF FASTAPI ENDPOINTS WITH CHURN_V2")
    print("=" * 70)
    
    # 1. Root
    res = client.get("/")
    assert res.status_code == 200, f"Root failed: {res.status_code}"
    print(f"1. GET / -> Status {res.status_code}: {res.json()}")
    
    # 2. Model Info
    res = client.get("/api/model-info")
    assert res.status_code == 200, f"Model info failed: {res.status_code}"
    info = res.json()
    print(f"\n2. GET /api/model-info -> Status {res.status_code}:")
    print(f"   Version: {info.get('version')}")
    print(f"   Deployed Model: {info.get('deployed_model')}")
    print(f"   Threshold: {info.get('threshold')}")
    print(f"   Accuracy: {info.get('accuracy')}")
    print(f"   Recall: {info.get('recall')}")
    
    # 3. Model Comparison
    res = client.get("/api/model-comparison")
    assert res.status_code == 200, f"Model comparison failed: {res.status_code}"
    comp = res.json()
    print(f"\n3. GET /api/model-comparison -> Status {res.status_code}:")
    print(f"   Model Version: {comp.get('model_version')}")
    print(f"   Deployed Model: {comp.get('deployed_model')}")
    for m in comp.get('models', []):
        print(f"   - {m['name']} ({'DEPLOYED' if m.get('is_deployed') else 'Individual'}): Acc={m['accuracy']*100:.2f}%, Rec={m['recall']*100:.2f}%, F1={m['f1']:.4f}, AUC={m['roc_auc']:.4f}")
        
    # 4. Single Prediction with SHAP & LIME
    single_payload = {
        "dataset": "ds1",
        "tenure": 2,
        "monthly_charges": 89.50,
        "total_charges": 179.00,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check"
    }
    res = client.post("/api/predict/single", json=single_payload)
    assert res.status_code == 200, f"Single prediction failed: {res.status_code}, {res.text}"
    pred_data = res.json()
    print(f"\n4. POST /api/predict/single -> Status {res.status_code}:")
    print(f"   Probability: {pred_data.get('probability'):.4f}")
    print(f"   Prediction: {pred_data.get('prediction')} (1=Churn, 0=No Churn)")
    print(f"   Risk Level: {pred_data.get('risk_level')}")
    print(f"   Threshold: {pred_data.get('threshold')}")
    print(f"   SHAP Available: {pred_data.get('shap', {}).get('available')}")
    print(f"   LIME Available: {pred_data.get('lime', {}).get('available')}")
    top_factors = pred_data.get('major_churn_drivers', []) or pred_data.get('top_factors', [])
    print(f"   Top Churn Drivers ({len(top_factors)}): {[f.get('factor') or f.get('name') for f in top_factors[:3]]}")
    print(f"   Recommended Action: {pred_data.get('recommended_action')}")
    
    # 5. What-if Simulator
    whatif_payload = {
        "dataset": "ds1",
        "base_profile": single_payload,
        "factor": "contract"
    }
    res = client.post("/api/simulate/what-if", json=whatif_payload)
    assert res.status_code == 200, f"What-if simulation failed: {res.status_code}, {res.text}"
    whatif_data = res.json()
    print(f"\n5. POST /api/simulate/what-if -> Status {res.status_code}:")
    print(f"   Baseline Probability: {whatif_data.get('baseline_probability'):.4f}")
    for s in whatif_data.get('scenarios', []):
        print(f"   Scenario '{s['scenario_label']}': Probability = {s['probability']:.4f}")
    print(f"   Best Option: {whatif_data.get('best_option')} ({whatif_data.get('best_probability'):.4f})")
    print(f"   Recommendation: {whatif_data.get('recommendation')}")
    
    print("\n" + "=" * 70)
    print("ALL API ENDPOINTS VALIDATED SUCCESSFULLY WITH CHURN_V2!")
    print("=" * 70)

if __name__ == '__main__':
    run_tests()

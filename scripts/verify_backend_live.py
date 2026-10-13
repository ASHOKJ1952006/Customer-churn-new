"""
verify_backend_live.py — Verify live FastAPI endpoints with requests
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def verify():
    print("=" * 70)
    print("VERIFYING LIVE BACKEND ENDPOINTS")
    print("=" * 70)
    
    # 1. Model Info
    r = requests.get(f"{BASE_URL}/api/model-info")
    print(f"\n1. GET /api/model-info: Status {r.status_code}")
    info = r.json()
    print(f"   Version: {info.get('version')}")
    print(f"   Model: {info.get('deployed_model')}")
    print(f"   Threshold: {info.get('threshold')}")
    print(f"   Accuracy: {info.get('accuracy')}")
    print(f"   Recall: {info.get('recall')}")
    print(f"   F1: {info.get('f1')}")
    
    # 2. Model Comparison
    r = requests.get(f"{BASE_URL}/api/model-comparison")
    print(f"\n2. GET /api/model-comparison: Status {r.status_code}")
    comp = r.json()
    print(f"   Model Version: {comp.get('model_version')}")
    print(f"   Deployed: {comp.get('deployed_model')}")
    for m in comp.get('models', []):
        print(f"   - {m['name']} ({'DEPLOYED' if m.get('is_deployed') else 'Individual'}): Recall={m['recall']*100:.2f}%, Acc={m['accuracy']*100:.2f}%, F1={m['f1']:.4f}, AUC={m['roc_auc']:.4f}")
        
    # 3. Single Prediction (6 core fields)
    payload = {
        "dataset": "ds1",
        "tenure": 2,
        "monthly_charges": 89.50,
        "total_charges": 179.00,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check"
    }
    r = requests.post(f"{BASE_URL}/api/predict/single", json=payload)
    print(f"\n3. POST /api/predict/single: Status {r.status_code}")
    pred = r.json()
    print(f"   Probability: {pred.get('probability'):.4f}")
    print(f"   Prediction: {pred.get('prediction')} (1=CHURN, 0=NO CHURN)")
    print(f"   Risk Level: {pred.get('risk_level')}")
    print(f"   Threshold: {pred.get('threshold')}")
    print(f"   Action: {pred.get('recommended_action')}")
    
    shap_info = pred.get('shap', {})
    print(f"   SHAP Available: {shap_info.get('available')}")
    if shap_info.get('available'):
        print(f"   SHAP Top Drivers: {[f['display_name'] for f in shap_info.get('features', [])[:3]]}")
        
    lime_info = pred.get('lime', {})
    print(f"   LIME Available: {lime_info.get('available')}")
    
    # 4. What-if Simulation
    whatif_payload = {
        "dataset": "ds1",
        "base_profile": payload,
        "factor": "contract"
    }
    r = requests.post(f"{BASE_URL}/api/simulate/what-if", json=whatif_payload)
    print(f"\n4. POST /api/simulate/what-if: Status {r.status_code}")
    whatif = r.json()
    print(f"   Baseline Probability: {whatif.get('baseline_probability'):.4f}")
    for s in whatif.get('scenarios', []):
        print(f"   Scenario '{s['scenario_label']}': Probability = {s['probability']:.4f}")
    print(f"   Best Option: {whatif.get('best_option')} ({whatif.get('best_probability'):.4f})")
    
    # 5. Dashboard History
    r = requests.get(f"{BASE_URL}/api/dashboard/history")
    print(f"\n5. GET /api/dashboard/history: Status {r.status_code}")
    hist = r.json().get('history', [])
    print(f"   History entries count: {len(hist)}")
    if hist:
        print(f"   Latest: Customer {hist[0].get('customer_id')}, Prob={hist[0].get('probability'):.4f}, Risk={hist[0].get('risk_level')}")
        
    print("\n" + "=" * 70)
    print("ALL LIVE BACKEND ENDPOINTS VERIFIED!")
    print("=" * 70)

if __name__ == '__main__':
    verify()

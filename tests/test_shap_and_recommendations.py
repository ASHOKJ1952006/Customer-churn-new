"""
test_shap_and_recommendations.py — Comprehensive validation test for SHAP and Recommendations
Tests 3 distinct customer profiles, verifies SHAP validity, sign conventions, and dynamic recommendations.
"""

import requests
import json
import math

BASE_URL = "http://localhost:8000"

def test_health():
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print("Health check passed.")

def validate_shap_structure(shap_data, expected_feature_count=None):
    assert shap_data.get("available") is True, f"SHAP marked unavailable! Error: {shap_data.get('error')}"
    features = shap_data.get("features", [])
    if expected_feature_count is not None:
        assert len(features) == expected_feature_count, f"Feature count mismatch: expected {expected_feature_count}, got {len(features)}"
    else:
        assert len(features) > 0, "Features list should not be empty"
    
    for f in features:
        val = f.get("shap_value")
        assert val is not None, f"Missing shap_value for {f}"
        assert not math.isnan(val), f"NaN in shap_value for {f}"
        assert not math.isinf(val), f"Inf in shap_value for {f}"
        
        # Check sign convention
        if val > 0:
            assert f["impact"] == "risk", f"Expected risk impact for positive SHAP {val} on {f['feature']}"
            assert f["direction"] == "increases_churn", f"Expected increases_churn for positive SHAP {val} on {f['feature']}"
        elif val < 0:
            assert f["impact"] == "protective", f"Expected protective impact for negative SHAP {val} on {f['feature']}"
            assert f["direction"] == "decreases_churn", f"Expected decreases_churn for negative SHAP {val} on {f['feature']}"
        
        assert f["importance"] == round(abs(val), 4), f"Importance should be abs(shap_value)"
        assert f["display_name"], f"Display name should not be empty for {f['feature']}"

def run_tests():
    test_health()

    # --- CUSTOMER 1: High Risk Profile ---
    c1 = {
        "dataset": "ds1",
        "tenure": 2,
        "monthly_charges": 95.0,
        "total_charges": 190.0,
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic"
    }
    r1 = requests.post(f"{BASE_URL}/api/predict/single", json=c1)
    assert r1.status_code == 200, f"Customer 1 failed: {r1.text}"
    data1 = r1.json()
    print("\n--- CUSTOMER 1 (High Risk) ---")
    print(f"Probability: {data1['probability']:.4f} | Risk Level: {data1['risk_level']}")
    print(f"Top Risk Drivers: {[f['display_name'] + ' (' + str(f['shap_value']) + ')' for f in data1['top_risk_factors'][:3]]}")
    print(f"Primary Action: {data1['recommended_action']}")
    validate_shap_structure(data1["shap"])
    assert data1["probability"] > 0.4, "Customer 1 should have high churn probability"
    assert len(data1["top_risk_factors"]) > 0, "Customer 1 should have positive risk factors"

    # --- CUSTOMER 2: Low Risk Profile ---
    c2 = {
        "dataset": "ds1",
        "tenure": 65,
        "monthly_charges": 45.0,
        "total_charges": 2925.0,
        "contract": "Two year",
        "payment_method": "Credit card (automatic)",
        "internet_service": "DSL"
    }
    r2 = requests.post(f"{BASE_URL}/api/predict/single", json=c2)
    assert r2.status_code == 200, f"Customer 2 failed: {r2.text}"
    data2 = r2.json()
    print("\n--- CUSTOMER 2 (Low Risk) ---")
    print(f"Probability: {data2['probability']:.4f} | Risk Level: {data2['risk_level']}")
    print(f"Top Protective Drivers: {[f['display_name'] + ' (' + str(f['shap_value']) + ')' for f in data2['top_protective_factors'][:3]]}")
    print(f"Primary Action: {data2['recommended_action']}")
    validate_shap_structure(data2["shap"])
    assert data2["probability"] < data1["probability"], "Customer 2 should have lower churn probability than Customer 1"
    assert len(data2["top_protective_factors"]) > 0, "Customer 2 should have strong protective factors"

    # --- CUSTOMER 3: Different Risk Profile ---
    c3 = {
        "dataset": "ds1",
        "tenure": 10,
        "monthly_charges": 85.0,
        "total_charges": 850.0,
        "contract": "Month-to-month",
        "payment_method": "Credit card (automatic)",
        "internet_service": "Fiber optic"
    }
    r3 = requests.post(f"{BASE_URL}/api/predict/single", json=c3)
    assert r3.status_code == 200, f"Customer 3 failed: {r3.text}"
    data3 = r3.json()
    print("\n--- CUSTOMER 3 (Different Profile) ---")
    print(f"Probability: {data3['probability']:.4f} | Risk Level: {data3['risk_level']}")
    print(f"Top Risk Drivers: {[f['display_name'] + ' (' + str(f['shap_value']) + ')' for f in data3['top_risk_factors'][:3]]}")
    print(f"Primary Action: {data3['recommended_action']}")
    validate_shap_structure(data3["shap"])

    # DYNAMIC RECOMMENDATION CHECK:
    # Customer 1 vs Customer 2 vs Customer 3 actions MUST NOT be identical!
    assert data1["recommended_action"] != data2["recommended_action"], "Customer 1 and 2 must have different recommendations"
    print("\nCustomer 1 & 2 have distinct recommendations: PASS")
    assert data2["recommended_action"] != data3["recommended_action"], "Customer 2 and 3 must have different recommendations"
    print("Customer 2 & 3 have distinct recommendations: PASS")

    # --- WHAT-IF SIMULATOR TEST ---
    whatif_payload = {
        "dataset": "ds1",
        "base_profile": c1,
        "factor": "contract"
    }
    r_whatif = requests.post(f"{BASE_URL}/api/whatif/simulate", json=whatif_payload)
    assert r_whatif.status_code == 200, f"What-if simulation failed: {r_whatif.text}"
    data_whatif = r_whatif.json()
    print("\n--- WHAT-IF SIMULATION (Contract) ---")
    print(f"Baseline Probability: {data_whatif['baseline_probability']:.4f}")
    print(f"Scenarios: {data_whatif['scenarios']}")
    print(f"Best Option: {data_whatif['best_option']} ({data_whatif['best_probability']:.4f})")
    print(f"What-If Recommendation: {data_whatif['recommendation']}")
    assert len(data_whatif["scenarios"]) == 3, "Should have 3 contract scenarios"
    assert data_whatif["best_probability"] < data_whatif["baseline_probability"], "Upgrading contract should lower churn probability"

    print("\nALL SANITY CHECKS AND VALIDATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()

"""
test_reduced_api.py - Test the backend API with reduced feature set (6 fields)
"""

import requests
import json

API_URL = 'http://localhost:8000'

# Test data with only 6 fields (reduced from 20)
test_data = {
    'dataset': 'ds1',
    'tenure': 12,
    'monthly_charges': 70.0,
    'total_charges': 1000.0,
    'contract': 'Month-to-month',
    'payment_method': 'Electronic check',
    'internet_service': 'DSL'
}

print("="*70)
print("TESTING REDUCED FEATURE API (6 fields)")
print("="*70)
print(f"\nInput data (6 fields):")
print(json.dumps(test_data, indent=2))

try:
    response = requests.post(f'{API_URL}/api/predict/single', json=test_data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS - Prediction received")
        print(f"\nChurn Probability: {result.get('probability', 0):.4f}")
        print(f"Risk Level: {result.get('risk_level', 'Unknown')}")
        print(f"Recommended Action: {result.get('recommended_action', 'N/A')}")
        print(f"Survival Months: {result.get('survival_months', 0):.2f}")
        print(f"Threshold: {result.get('threshold', 0):.4f}")
        
        print(f"\nTop Factors:")
        for factor in result.get('top_factors', []):
            print(f"  - {factor.get('name', 'Unknown')}: {factor.get('value', 0):.4f}")
    else:
        print(f"\n❌ ERROR - Status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.Timeout:
    print("\n❌ ERROR - Request timed out")
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR - Could not connect to backend. Is it running?")
except Exception as e:
    print(f"\n❌ ERROR - {str(e)}")

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print("\nNOTE: If you changed the model loading in main.py, you need to RESTART the backend server for changes to take effect.")

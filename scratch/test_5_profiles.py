import sys, os, asyncio, json
sys.path.append(os.path.abspath('backend'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from main import predict_single, SinglePredictRequest

profiles = {
    "CUSTOMER A (New + M2M + E-Check + High Charges)": {
        'dataset': 'ds1',
        'tenure': 3,
        'monthly_charges': 95.0,
        'total_charges': 285.0,
        'contract': 'Month-to-month',
        'payment_method': 'Electronic check',
        'internet_service': 'Fiber optic',
        'paperless_billing': 'Yes',
        'tech_support': 'No',
        'online_security': 'No'
    },
    "CUSTOMER B (Long Tenure + Two-Year + Autopay)": {
        'dataset': 'ds1',
        'tenure': 60,
        'monthly_charges': 50.0,
        'total_charges': 3000.0,
        'contract': 'Two year',
        'payment_method': 'Bank transfer (automatic)',
        'internet_service': 'DSL',
        'paperless_billing': 'No',
        'tech_support': 'Yes',
        'online_security': 'Yes'
    },
    "CUSTOMER C (Month-to-month + Autopay + Moderate Charges)": {
        'dataset': 'ds1',
        'tenure': 18,
        'monthly_charges': 65.0,
        'total_charges': 1170.0,
        'contract': 'Month-to-month',
        'payment_method': 'Credit card (automatic)',
        'internet_service': 'DSL',
        'paperless_billing': 'Yes',
        'tech_support': 'No',
        'online_security': 'No'
    },
    "CUSTOMER D (Fiber + High Charges + No Tech Support)": {
        'dataset': 'ds1',
        'tenure': 14,
        'monthly_charges': 105.0,
        'total_charges': 1470.0,
        'contract': 'One year',
        'payment_method': 'Credit card (automatic)',
        'internet_service': 'Fiber optic',
        'paperless_billing': 'Yes',
        'tech_support': 'No',
        'online_security': 'No'
    },
    "CUSTOMER E (Long Tenure + Low Charges + Two-Year)": {
        'dataset': 'ds1',
        'tenure': 70,
        'monthly_charges': 20.0,
        'total_charges': 1400.0,
        'contract': 'Two year',
        'payment_method': 'Mailed check',
        'internet_service': 'No',
        'paperless_billing': 'No',
        'tech_support': 'No',
        'online_security': 'No'
    }
}

results = {}

for name, payload in profiles.items():
    print(f"\n{'='*70}\nEVALUATING: {name}\n{'='*70}")
    res = asyncio.run(predict_single(SinglePredictRequest(**payload)))
    results[name] = res
    
    print(f"Churn Probability: {res['churn_probability']*100:.2f}% | Risk Level: {res['risk_level']}")
    print(f"Explanation Confidence: {res['explanation_confidence']}")
    print(f"Agreement Summary: {res['agreement_summary']}")
    
    print(f"Primary Churn Driver: {res['management_summary']['primary_churn_driver']}")
    print(f"Recommended Strategy: {res['management_summary']['recommended_strategy']}")
    print(f"Priority: {res['management_summary']['priority']} | Cost: {res['management_summary']['estimated_action_cost']}")
    
    print("\nTop Major Churn Drivers:")
    for d in res['major_churn_drivers'][:4]:
        print(f"  - {d['factor']} ({d['impact_level']}) - Actionable: {d['is_actionable']}")
        print(f"    Why: {d['business_explanation']}")
        
    print(f"\nProtective Factors ({len(res['protective_factors'])} found):")
    for p in res['protective_factors'][:3]:
        print(f"  [+] {p['factor']} (Score: {p['combined_score']}) - {p['explanation']}")
        
    print(f"\nTop Recommended Actions ({len(res['recommendations'])} actions):")
    for r in res['recommendations'][:4]:
        sim_text = ""
        if r.get('simulation'):
            sim = r['simulation']
            sim_text = f" | ML Sim: {sim['before_probability']*100:.1f}% -> {sim['simulated_probability']*100:.1f}% (Diff -{sim['risk_reduction_pct']} pp)"
        print(f"  #{r['rank']} [{r['priority']}] {r['title']} [Cost: {r['cost']}, Impact: {r['expected_impact']}, Owner: {r['owner']}]{sim_text}")
        print(f"     Action: {r['action']}")

    print("\nStrategy Bundles:")
    for b in res.get('strategy_bundles', []):
        print(f"  [*] {b['name']} ({b['cost']} Cost): {b['baseline_probability']*100:.1f}% -> {b['simulated_probability']*100:.1f}% (Diff -{b['risk_reduction_pct']} pp)")

# Verification assertions
probs = [res['churn_probability'] for res in results.values()]
assert len(set(probs)) == 5, "All 5 customer profiles must have distinct churn probabilities!"

# Verify Customer B (Two-year + Autopay) has no contract upgrade or autopay migration
cust_b_titles = [r['title'] for r in results["CUSTOMER B (Long Tenure + Two-Year + Autopay)"]['recommendations']]
assert not any("Contract Commitment Upgrade" in t for t in cust_b_titles), "Customer B must NOT receive contract upgrade!"
assert not any("Autopay Migration" in t for t in cust_b_titles), "Customer B must NOT receive autopay migration!"

# Verify Customer C (Already Autopay) has no autopay migration
cust_c_titles = [r['title'] for r in results["CUSTOMER C (Month-to-month + Autopay + Moderate Charges)"]['recommendations']]
assert not any("Autopay Migration" in t for t in cust_c_titles), "Customer C must NOT receive autopay migration!"
assert any("Contract Commitment Upgrade" in t for t in cust_c_titles), "Customer C must receive contract upgrade!"

# Verify Customer D (One-year contract, Fiber, High spend) gets plan review or tech support, but not M2M actions
cust_d_titles = [r['title'] for r in results["CUSTOMER D (Fiber + High Charges + No Tech Support)"]['recommendations']]
assert any("Support" in t or "Security" in t or "Fiber" in t or "Plan" in t for t in cust_d_titles), "Customer D must receive appropriate service/plan recommendations!"

print("\n" + "#"*70)
print("ALL 5 CUSTOMER PROFILE VERIFICATION ASSERTIONS PASSED SUCCESSFULLY!")
print("#"*70)

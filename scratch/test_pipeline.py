import sys, json, os, asyncio
sys.path.append(os.path.abspath('backend'))
from main import predict_single, SinglePredictRequest

print("--- TESTING CUSTOMER A (High Risk / New M2M E-Check) ---")
payload_a = {
    'dataset': 'ds1',
    'tenure': 4,
    'monthly_charges': 95.0,
    'total_charges': 380.0,
    'contract': 'Month-to-month',
    'payment_method': 'Electronic check',
    'internet_service': 'Fiber optic',
    'paperless_billing': 'Yes',
    'tech_support': 'No',
    'online_security': 'No'
}
data_a = asyncio.run(predict_single(SinglePredictRequest(**payload_a)))
print("Probability:", round(data_a['churn_probability'], 4))
print("Risk Level:", data_a['risk_level'])
print("Explanation Confidence:", data_a['explanation_confidence'])
print("Primary Driver:", data_a['management_summary']['primary_churn_driver'])
print("Recommended Strategy:", data_a['management_summary']['recommended_strategy'])
print("Major Churn Drivers:")
for d in data_a['major_churn_drivers']:
    print(f"  - {d['factor']} ({d['impact_level']}) | Actionable: {d['is_actionable']}")
print("Recommendations:")
for r in data_a['recommendations']:
    sim = f" [Sim: {r['simulation']['before_probability']*100:.1f}% -> {r['simulation']['simulated_probability']*100:.1f}% (-{r['simulation']['risk_reduction_pct']} pp)]" if r.get('simulation') else ""
    print(f"  #{r['rank']} [{r['priority']}] {r['title']} (Cost: {r['cost']}, Impact: {r['expected_impact']}, Owner: {r['owner']}, Time: {r['timeframe']}){sim}")
print("Strategy Bundles:")
for b in data_a['strategy_bundles']:
    print(f"  * {b['name']} ({b['cost']} Cost): {b['baseline_probability']*100:.1f}% -> {b['simulated_probability']*100:.1f}% (-{b['risk_reduction_pct']} pp)")

print("\n--- TESTING CUSTOMER B (Loyal / Low Risk 2-Year Autopay) ---")
payload_b = {
    'dataset': 'ds1',
    'tenure': 55,
    'monthly_charges': 45.0,
    'total_charges': 2475.0,
    'contract': 'Two year',
    'payment_method': 'Bank transfer (automatic)',
    'internet_service': 'DSL',
    'paperless_billing': 'No',
    'tech_support': 'Yes',
    'online_security': 'Yes'
}
data_b = asyncio.run(predict_single(SinglePredictRequest(**payload_b)))
print("Probability:", round(data_b['churn_probability'], 4))
print("Risk Level:", data_b['risk_level'])
print("Protective Factors Count:", len(data_b['protective_factors']))
for p in data_b['protective_factors'][:3]:
    print(f"  + {p['factor']} (Score: {p['combined_score']})")
print("Recommendations (Verify NO contract upgrade or autopay migration!):")
for r in data_b['recommendations']:
    print(f"  #{r['rank']} [{r['priority']}] {r['title']}")
# Assert no contract upgrade for customer B
action_titles = [r['title'] for r in data_b['recommendations']]
assert not any('Contract Commitment Upgrade' in t for t in action_titles), "Customer B should NOT receive contract upgrade!"
assert not any('Autopay Migration' in t for t in action_titles), "Customer B should NOT receive autopay migration!"
print("TEST PASSED: Customer B correctly pruned unnecessary recommendations!")

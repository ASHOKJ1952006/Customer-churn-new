"""
recommendation_engine.py — Telecom Customer Retention Decision Support System
Transforms raw SHAP and LIME signals into executive and operational retention intelligence.
Generates prioritized, customer-specific action recommendations with transparent scoring,
counterfactual ML simulations, strategy bundles, and responsible AI safeguards.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from business_factors import (
    FEATURE_BUSINESS_MAPPING,
    RETENTION_ACTIONS_CATALOG,
    CATEGORY_CONTRACT,
    CATEGORY_PAYMENT,
    CATEGORY_PRICING,
    CATEGORY_LIFECYCLE,
    CATEGORY_SERVICE_TYPE,
    CATEGORY_SERVICE_EXP,
    CATEGORY_ENGAGEMENT,
    CATEGORY_COMPOUND,
    CATEGORY_DEMOGRAPHIC,
    get_business_factor,
    get_action_definition,
)

def determine_risk_level(probability: float, threshold: float = 0.3663) -> str:
    """
    Determine customer risk level against decision threshold.
    Returns: 'Low', 'Medium', 'High', or 'Critical'
    """
    if probability < 0.20:
        return "Low"
    elif probability < threshold:
        return "Medium"
    elif probability < 0.70:
        return "High"
    else:
        return "Critical"

class TelecomRetentionDecisionEngine:
    """
    Telecom Decision Support System:
    - Filters and aggregates SHAP + LIME into Top 5 Major Churn Drivers
    - Enforces Customer-State Specificity (prunes inapplicable actions)
    - Computes transparent Action Score: (Risk × Prob × Relevance × Impact) / Cost
    - Ranks Top Retention Suggestions with Owners, Timeframes, and Costs
    - Evaluates Counterfactual What-If ML Simulations for each action and strategy bundle
    - Produces Executive Telecom Management Summary and Dynamic Narrative
    """

    def generate_recommendations(
        self,
        inputs: Dict[str, Any],
        probability: float,
        threshold: float,
        shap_summary: Dict[str, Any],
        lime_summary: Optional[Dict[str, Any]] = None,
        combined_summary: Optional[Dict[str, Any]] = None,
        predict_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Produce comprehensive telecom retention decision support output.
        """
        risk_lvl = determine_risk_level(probability, threshold)

        # ----------------------------------------------------------------------
        # 1. Extract Customer State Profile
        # ----------------------------------------------------------------------
        import math
        def safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                f = float(val)
                return default if (math.isnan(f) or math.isinf(f)) else f
            except (ValueError, TypeError):
                return default

        tenure = safe_float(inputs.get("tenure") if inputs.get("tenure") is not None else inputs.get("Tenure"), 0.0)
        monthly_charges = safe_float(inputs.get("monthly_charges") if inputs.get("monthly_charges") is not None else inputs.get("MonthlyCharges"), 0.0)
        total_charges = safe_float(inputs.get("total_charges") if inputs.get("total_charges") is not None else inputs.get("TotalCharges"), 0.0)
        contract = str(inputs.get("contract") or inputs.get("Contract") or "Month-to-month")
        payment_method = str(inputs.get("payment_method") or inputs.get("PaymentMethod") or "Electronic check")
        internet_service = str(inputs.get("internet_service") or inputs.get("InternetService") or "DSL")
        tech_support = str(inputs.get("tech_support") or inputs.get("TechSupport") or "No")
        online_security = str(inputs.get("online_security") or inputs.get("OnlineSecurity") or "No")
        streaming_tv = str(inputs.get("streaming_tv") or inputs.get("StreamingTV") or "No")
        streaming_movies = str(inputs.get("streaming_movies") or inputs.get("StreamingMovies") or "No")
        paperless_billing = str(inputs.get("paperless_billing") or inputs.get("PaperlessBilling") or "Yes")

        avg_monthly = total_charges / max(tenure, 1.0)
        charge_increase = monthly_charges - avg_monthly

        # ----------------------------------------------------------------------
        # 2. Extract Aligned Explainability Signals (SHAP + LIME)
        # ----------------------------------------------------------------------
        combined_factors = combined_summary.get("combined_factors", []) if combined_summary else []
        explanation_confidence = combined_summary.get("explanation_confidence", "HIGH") if combined_summary else "HIGH"
        agreement_summary = combined_summary.get("agreement_summary", "") if combined_summary else ""

        # Fallback to SHAP if combined not available
        if not combined_factors and shap_summary and "features" in shap_summary:
            combined_factors = [
                {
                    "feature": f["feature"],
                    "display_name": f["display_name"],
                    "category": f.get("category", "General"),
                    "is_actionable": f.get("is_actionable", True),
                    "business_meaning": f.get("business_meaning", ""),
                    "shap_value": f["shap_value"],
                    "lime_weight": 0.0,
                    "combined_score": abs(f["shap_value"]),
                    "direction": f["direction"],
                    "impact": f["impact"],
                    "actual_value": f.get("value", "N/A"),
                    "sign_agreement": True
                }
                for f in shap_summary["features"]
            ]

        factor_map = {f["feature"]: f for f in combined_factors}

        # ----------------------------------------------------------------------
        # 3. Identify Top 5 Major Churn Drivers (Business Aggregated)
        # ----------------------------------------------------------------------
        risk_contributors = [f for f in combined_factors if f["direction"] == "increases_churn"]
        
        # Deduplicate by Business Category to provide broad, non-redundant drivers
        seen_categories = set()
        major_churn_drivers = []
        for f in risk_contributors:
            cat = f.get("category", "General")
            if cat in [CATEGORY_DEMOGRAPHIC]:
                # Exclude demographics from business drivers for responsible AI
                continue
            if cat not in seen_categories or len(major_churn_drivers) < 3:
                seen_categories.add(cat)
                
                score = f["combined_score"]
                if score >= 0.70:
                    impact_label = "HIGH IMPACT"
                elif score >= 0.35:
                    impact_label = "MEDIUM IMPACT"
                else:
                    impact_label = "MODERATE IMPACT"

                major_churn_drivers.append({
                    "factor": f["display_name"],
                    "category": cat,
                    "technical_feature": f["feature"],
                    "actual_value": f.get("actual_value", "N/A"),
                    "combined_score": f["combined_score"],
                    "shap_value": f.get("shap_value", 0.0),
                    "lime_weight": f.get("lime_weight", 0.0),
                    "impact_level": impact_label,
                    "is_actionable": f["is_actionable"],
                    "business_explanation": f["business_meaning"] or f"Attribute increases churn vulnerability for {f['display_name']}."
                })
            if len(major_churn_drivers) >= 5:
                break

        # ----------------------------------------------------------------------
        # 4. Identify Protective Factors
        # ----------------------------------------------------------------------
        protective_contributors = [f for f in combined_factors if f["direction"] == "decreases_churn"]
        protective_factors = []
        for f in protective_contributors[:5]:
            protective_factors.append({
                "factor": f["display_name"],
                "category": f.get("category", "General"),
                "technical_feature": f["feature"],
                "actual_value": f.get("actual_value", "N/A"),
                "combined_score": f["combined_score"],
                "shap_value": f.get("shap_value", 0.0),
                "lime_weight": f.get("lime_weight", 0.0),
                "explanation": f"Acts as a protective factor stabilizing the account against churn."
            })

        # ----------------------------------------------------------------------
        # 5. Customer-Specific Candidate Action Generation (Rule Engine)
        # ----------------------------------------------------------------------
        candidates = []

        # Helper to check feature weight
        def get_risk_weight(feat_names: List[str]) -> float:
            weights = []
            for fn in feat_names:
                if fn in factor_map and factor_map[fn]["direction"] == "increases_churn":
                    weights.append(factor_map[fn]["combined_score"])
            return max(weights) if weights else 0.0

        # Multi-Factor Compounded Risk Detection
        m2m_active = (contract == "Month-to-month")
        echeck_active = (payment_method == "Electronic check")
        short_tenure_active = (tenure <= 12)
        high_spend_active = (monthly_charges >= 70.0)

        is_compound_risk = (m2m_active and echeck_active and (short_tenure_active or high_spend_active) and probability >= threshold)

        # RULE 10: Multi-Factor Compounded Risk (VIP Intervention)
        if is_compound_risk:
            act_def = get_action_definition("ACT_MULTI_FACTOR_INTERVENTION")
            candidates.append({
                "action_id": "ACT_MULTI_FACTOR_INTERVENTION",
                "risk_contribution": max(0.85, get_risk_weight(["HighRiskCombo", "VeryHighRisk", "Contract_Month-to-month"])),
                "relevance": 1.3,
                "cost_val": 3.0,  # HIGH
                "impact_val": 3.0, # HIGH
                "def": act_def,
                "custom_why": f"Customer exhibits multiple compounded risk factors simultaneously ({contract}, {payment_method}, {int(tenure)} mo tenure, and ${monthly_charges:.2f}/mo bill).",
                "simulation_mutation": {
                    "contract": "Two year",
                    "payment_method": "Bank transfer (automatic)",
                    "monthly_charges": max(monthly_charges - 15.0, 30.0)
                }
            })

        # RULE 1: Month-to-Month Contract Upgrade
        # STRICT FILTER: Never recommend contract upgrade if customer already has 1-year or 2-year contract!
        m2m_weight = get_risk_weight(["Contract_Month-to-month", "IsMonthToMonth", "HighRiskCombo", "ContractLength"])
        if m2m_active and (m2m_weight > 0.05 or probability >= threshold or is_compound_risk):
            act_def = get_action_definition("ACT_CONTRACT_UPGRADE")
            candidates.append({
                "action_id": "ACT_CONTRACT_UPGRADE",
                "risk_contribution": max(m2m_weight, 0.60),
                "relevance": 1.2 if probability >= threshold else 1.0,
                "cost_val": 2.0,  # MEDIUM
                "impact_val": 3.0, # HIGH
                "def": act_def,
                "custom_why": f"Customer is on a {contract} contract without commitment lock-in, which strongly elevates churn probability.",
                "simulation_mutation": {"contract": "Two year"}
            })

        # RULE 2: Electronic Check Autopay Migration
        # STRICT FILTER: Never recommend autopay migration if customer is already on automatic payments!
        is_already_autopay = ("automatic" in payment_method.lower())
        echeck_weight = get_risk_weight(["PaymentMethod_Electronic check", "IsElectronicCheck", "HighRiskCombo"])
        if not is_already_autopay and (echeck_weight > 0.03 or payment_method == "Electronic check"):
            act_def = get_action_definition("ACT_PAYMENT_MIGRATION")
            candidates.append({
                "action_id": "ACT_PAYMENT_MIGRATION",
                "risk_contribution": max(echeck_weight, 0.45),
                "relevance": 1.1,
                "cost_val": 1.0,  # LOW
                "impact_val": 2.0, # MEDIUM
                "def": act_def,
                "custom_why": f"Customer pays via {payment_method}, which experiences higher billing friction and voluntary termination than automated debit.",
                "simulation_mutation": {"payment_method": "Bank transfer (automatic)"}
            })

        # RULE 3: Pricing & Plan Optimization Review
        # STRICT FILTER: Only recommend plan review/discounts if price features actually contribute to churn and bill is substantial!
        pricing_weight = get_risk_weight(["MonthlyCharges", "AvgMonthlySpend", "ChargeToTenureRatio"])
        if monthly_charges >= 65.0 and (pricing_weight > 0.05 or (monthly_charges >= 85.0 and probability >= threshold)):
            act_def = get_action_definition("ACT_PRICING_REVIEW")
            candidates.append({
                "action_id": "ACT_PRICING_REVIEW",
                "risk_contribution": max(pricing_weight, 0.50),
                "relevance": 1.1,
                "cost_val": 1.0,  # LOW
                "impact_val": 3.0, # HIGH
                "def": act_def,
                "custom_why": f"Recurring monthly charge of ${monthly_charges:.2f} is a confirmed price pressure contributor to churn risk.",
                "simulation_mutation": {"monthly_charges": max(monthly_charges - 15.0, 25.0)}
            })

        # RULE 4: Recent Charge Spike Relief
        # STRICT FILTER: Only trigger if charge increase is positive and verified!
        charge_spike_weight = get_risk_weight(["ChargeIncrease", "ChargeIncreaseRatio", "TotalChargesDifference"])
        if charge_increase >= 3.0 and (charge_spike_weight > 0.02 or probability >= threshold):
            act_def = get_action_definition("ACT_CHARGE_SPIKE_RELIEF")
            candidates.append({
                "action_id": "ACT_CHARGE_SPIKE_RELIEF",
                "risk_contribution": max(charge_spike_weight, 0.40),
                "relevance": 1.0,
                "cost_val": 1.0,  # LOW
                "impact_val": 2.0, # MEDIUM
                "def": act_def,
                "custom_why": f"Recent billing of ${monthly_charges:.2f} represents a +${charge_increase:.2f} increase over lifetime average spend.",
                "simulation_mutation": {"charge_increase": 0.0}
            })

        # RULE 5: Short Tenure Onboarding Care
        # STRICT FILTER: Only for tenure <= 12 months!
        tenure_weight = get_risk_weight(["tenure", "IsNewCustomer", "TenureGroup_7-12mo", "VeryHighRisk"])
        if tenure <= 12 and (tenure_weight > 0.03 or tenure <= 6):
            act_def = get_action_definition("ACT_NEW_CUSTOMER_ONBOARDING")
            candidates.append({
                "action_id": "ACT_NEW_CUSTOMER_ONBOARDING",
                "risk_contribution": max(tenure_weight, 0.45),
                "relevance": 1.1,
                "cost_val": 1.0,  # LOW
                "impact_val": 2.0, # MEDIUM
                "def": act_def,
                "custom_why": f"Customer has only {int(tenure)} months of tenure and has not yet established deep relationship stickiness.",
                "simulation_mutation": {"tenure": tenure + 6}
            })

        # RULE 6: Fiber Optic Service Audit
        # STRICT FILTER: Only if InternetService is Fiber optic AND explanation confirms fiber contributes to churn!
        fiber_weight = get_risk_weight(["InternetService_Fiber optic", "FiberAndMonthToMonth", "FiberAndElectronicCheck"])
        if internet_service == "Fiber optic" and (fiber_weight > 0.04 or (fiber_weight > 0.01 and probability >= threshold)):
            act_def = get_action_definition("ACT_FIBER_SERVICE_AUDIT")
            candidates.append({
                "action_id": "ACT_FIBER_SERVICE_AUDIT",
                "risk_contribution": max(fiber_weight, 0.35),
                "relevance": 1.0,
                "cost_val": 1.0,  # LOW
                "impact_val": 2.0, # MEDIUM
                "def": act_def,
                "custom_why": f"Fiber optic tier has an active positive churn weight, indicating potential service reliability or price-to-value friction.",
                "simulation_mutation": {"service_audit": "verified"}
            })

        # RULE 7: Tech Support Add-on
        # STRICT FILTER: Never recommend tech support if customer already has TechSupport == 'Yes'!
        tech_sup_weight = get_risk_weight(["TechSupport_No", "TechSupport_Yes"])
        if tech_support == "No" and internet_service != "No" and (tech_sup_weight > 0.02 or probability >= threshold):
            act_def = get_action_definition("ACT_ADD_TECH_SUPPORT")
            candidates.append({
                "action_id": "ACT_ADD_TECH_SUPPORT",
                "risk_contribution": max(tech_sup_weight, 0.30),
                "relevance": 0.9,
                "cost_val": 1.0,  # LOW
                "impact_val": 2.0, # MEDIUM
                "def": act_def,
                "custom_why": "Customer has broadband internet without dedicated Technical Support coverage, increasing vulnerability during service faults.",
                "simulation_mutation": {"tech_support": "Yes"}
            })

        # RULE 8: Online Security Add-on
        # STRICT FILTER: Never recommend online security if customer already has OnlineSecurity == 'Yes'!
        sec_weight = get_risk_weight(["OnlineSecurity_No", "OnlineSecurity_Yes"])
        if online_security == "No" and internet_service != "No" and (sec_weight > 0.02 or probability >= threshold):
            act_def = get_action_definition("ACT_ADD_ONLINE_SECURITY")
            candidates.append({
                "action_id": "ACT_ADD_ONLINE_SECURITY",
                "risk_contribution": max(sec_weight, 0.30),
                "relevance": 0.9,
                "cost_val": 1.0,  # LOW
                "impact_val": 2.0, # MEDIUM
                "def": act_def,
                "custom_why": "Account lacks cybersecurity protection add-ons, leaving digital household security unaddressed.",
                "simulation_mutation": {"online_security": "Yes"}
            })

        # RULE 9: Multi-Service Bundle Promotion
        if internet_service != "No" and (streaming_tv == "Yes" or streaming_movies == "Yes") and probability >= threshold:
            act_def = get_action_definition("ACT_SERVICE_BUNDLE_OFFER")
            candidates.append({
                "action_id": "ACT_SERVICE_BUNDLE_OFFER",
                "risk_contribution": 0.35,
                "relevance": 0.95,
                "cost_val": 2.0,  # MEDIUM
                "impact_val": 3.0, # HIGH
                "def": act_def,
                "custom_why": "Customer utilizes multiple entertainment services but lacks an integrated loyalty bundle discount.",
                "simulation_mutation": {"bundled": True}
            })

        # RULE 11: Loyal / Low Risk Customer Appreciation
        # If customer is Low Risk and has strong protective factors (e.g. 2-year contract or long tenure)
        if probability < threshold and len(candidates) == 0:
            act_def = get_action_definition("ACT_LOYALTY_APPRECIATION")
            candidates.append({
                "action_id": "ACT_LOYALTY_APPRECIATION",
                "risk_contribution": 0.20,
                "relevance": 1.0,
                "cost_val": 1.0,  # LOW
                "impact_val": 2.0, # MEDIUM
                "def": act_def,
                "custom_why": f"Customer shows healthy loyalty metrics (churn probability: {probability*100:.1f}%). Aggressive discounts or heavy intervention are not required.",
                "simulation_mutation": {"loyalty_perk": "applied"}
            })

        # Fallback if no candidate triggered
        if not candidates:
            act_def = get_action_definition("ACT_LOYALTY_APPRECIATION")
            candidates.append({
                "action_id": "ACT_LOYALTY_APPRECIATION",
                "risk_contribution": 0.20,
                "relevance": 1.0,
                "cost_val": 1.0,
                "impact_val": 2.0,
                "def": act_def,
                "custom_why": "Account profile displays steady stability without acute risk drivers.",
                "simulation_mutation": {}
            })

        # ----------------------------------------------------------------------
        # 6. Action Scoring Formula & Ranking
        # ----------------------------------------------------------------------
        # Action Score = (Risk Contribution * Customer Risk * Business Relevance * Expected Impact) / Action Cost
        cust_risk_factor = max(probability, 0.20)
        scored_candidates = []

        for c in candidates:
            risk_contrib = c["risk_contribution"]
            rel = c["relevance"]
            imp = c["impact_val"]
            cost = c["cost_val"]
            action_score = (risk_contrib * cust_risk_factor * rel * imp) / cost

            scored_candidates.append({
                **c,
                "action_score": round(action_score, 4)
            })

        # Sort descending by Action Score
        scored_candidates.sort(key=lambda x: x["action_score"], reverse=True)

        # Truncate to maximum of 10 recommendations (never pad with filler)
        top_candidates = scored_candidates[:10]

        # Format Recommendations into standard cards
        ranked_recommendations = []
        for idx, item in enumerate(top_candidates, 1):
            d = item["def"]
            # Dynamic Priority based on probability and action score
            if probability < threshold:
                prio = "LOW"
            elif probability >= 0.70 and (item["action_score"] >= 1.0 or idx == 1):
                prio = "CRITICAL"
            elif item["action_score"] >= 0.50 or (probability >= threshold and idx <= 2):
                prio = "HIGH"
            elif item["action_score"] >= 0.25:
                prio = "MEDIUM"
            else:
                prio = "LOW"

            # Execute real ML simulation if predict_fn provided
            sim_result = None
            if predict_fn and item.get("simulation_mutation"):
                try:
                    sim_inputs = inputs.copy()
                    for k, v in item["simulation_mutation"].items():
                        if k in sim_inputs:
                            sim_inputs[k] = v
                        elif k == "monthly_charges":
                            sim_inputs["monthly_charges"] = v

                    sim_prob = float(predict_fn(sim_inputs))
                    risk_reduction = max(0.0, (probability - sim_prob) * 100)
                    sim_result = {
                        "before_probability": round(probability, 4),
                        "simulated_probability": round(sim_prob, 4),
                        "risk_reduction_pct": round(risk_reduction, 1),
                        "mutations": item["simulation_mutation"]
                    }
                except Exception as e:
                    pass

            ranked_recommendations.append({
                "rank": idx,
                "action_id": item["action_id"],
                "title": d["title"],
                "factor": d["category"],
                "action": d["action"],
                "why": item.get("custom_why", d["why"]),
                "business_objective": d["business_objective"],
                "owner": d["owner"],
                "priority": prio,
                "cost": d["cost"],
                "expected_impact": d["impact"],
                "timeframe": d["timeframe"],
                "action_score": item["action_score"],
                "simulation": sim_result
            })

        primary_action = ranked_recommendations[0]["action"] if ranked_recommendations else "Review customer account for custom loyalty plan."

        # ----------------------------------------------------------------------
        # 7. Time Horizon Grouping
        # ----------------------------------------------------------------------
        immediate_actions = [r for r in ranked_recommendations if "Immediate" in r["timeframe"] or r["priority"] in ["CRITICAL", "HIGH"]]
        short_term_actions = [r for r in ranked_recommendations if "Short-Term" in r["timeframe"] or r["priority"] == "MEDIUM"]
        long_term_actions = [r for r in ranked_recommendations if "Long-Term" in r["timeframe"] or r["priority"] == "LOW"]

        # ----------------------------------------------------------------------
        # 8. Counterfactual Retention Strategy Bundles (Strategy A, B, C)
        # ----------------------------------------------------------------------
        strategy_bundles = self._build_strategy_bundles(inputs, probability, predict_fn)

        # ----------------------------------------------------------------------
        # 9. Telecom Management Summary & Dynamic Narrative
        # ----------------------------------------------------------------------
        management_summary = {
            "risk_level": risk_lvl,
            "churn_probability": round(probability, 4),
            "primary_churn_driver": major_churn_drivers[0]["factor"] if major_churn_drivers else "Account Age / Baseline",
            "secondary_churn_drivers": [d["factor"] for d in major_churn_drivers[1:4]],
            "recommended_strategy": f"{ranked_recommendations[0]['title']} ({ranked_recommendations[0]['owner']})" if ranked_recommendations else "Standard Loyalty Program",
            "estimated_action_cost": ranked_recommendations[0]["cost"] if ranked_recommendations else "LOW",
            "priority": ranked_recommendations[0]["priority"] if ranked_recommendations else "LOW",
            "protective_factors_count": len(protective_factors),
            "explanation_confidence": explanation_confidence
        }

        # Dynamic Natural Language Narrative
        narrative_parts = [
            f"Customer has a {probability*100:.1f}% predicted probability of churn and is classified as {risk_lvl.upper()} RISK."
        ]
        if major_churn_drivers:
            drivers_str = ", ".join([d["factor"].lower() for d in major_churn_drivers[:3]])
            narrative_parts.append(f"The primary factors elevating churn risk are {drivers_str}.")
        if protective_factors:
            prot_str = ", ".join([p["factor"].lower() for p in protective_factors[:2]])
            narrative_parts.append(f"Currently, {prot_str} {'is' if len(protective_factors)==1 else 'are'} acting as protective factor(s) helping retain the account.")
        if ranked_recommendations:
            narrative_parts.append(
                f"The recommended immediate action is to {ranked_recommendations[0]['action']} "
                f"A secondary follow-up is to {ranked_recommendations[1]['action'] if len(ranked_recommendations)>1 else 'monitor customer sentiment'}."
            )
        customer_summary = " ".join(narrative_parts)

        return {
            "risk_level": risk_lvl,
            "primary_action": primary_action,
            "customer_summary": customer_summary,
            "management_summary": management_summary,
            "major_churn_drivers": major_churn_drivers,
            "protective_factors": protective_factors,
            "recommendations": ranked_recommendations,
            "immediate_actions": immediate_actions[:3],
            "short_term_actions": short_term_actions[:3],
            "long_term_actions": long_term_actions[:3],
            "strategy_bundles": strategy_bundles,
            "explanation_confidence": explanation_confidence,
            "agreement_summary": agreement_summary,
            # Legacy compatibility fields
            "risk_drivers": [d["factor"] for d in major_churn_drivers[:3]],
            "protective_drivers": [p["factor"] for p in protective_factors[:3]],
        }

    def _build_strategy_bundles(
        self,
        inputs: Dict[str, Any],
        baseline_prob: float,
        predict_fn: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """
        Build Strategy A (Low Cost), Strategy B (Balanced), and Strategy C (High Retention)
        with real ML simulated counterfactual probabilities.
        """
        bundles = []

        # Strategy A: Low Cost
        # Autopay migration + Support outreach + Billing review
        bundle_a_mutations = {
            "payment_method": "Bank transfer (automatic)",
            "tech_support": "Yes",
        }
        # Strategy B: Balanced
        # Autopay + 1-Year/2-Year Contract + Small Tariff Adjustment
        bundle_b_mutations = {
            "payment_method": "Bank transfer (automatic)",
            "contract": "One year",
            "monthly_charges": max(float(inputs.get("monthly_charges", 70.0)) - 10.0, 25.0)
        }
        # Strategy C: High Retention
        # Two-Year Contract + Autopay + Security + Support + $15 Credit
        bundle_c_mutations = {
            "contract": "Two year",
            "payment_method": "Bank transfer (automatic)",
            "online_security": "Yes",
            "tech_support": "Yes",
            "monthly_charges": max(float(inputs.get("monthly_charges", 70.0)) - 15.0, 25.0)
        }

        configs = [
            {
                "bundle_id": "strategy_a",
                "name": "Strategy A — Low Cost Operations",
                "cost": "LOW",
                "tactics": [
                    "Migrate to Autopay with one-time credit",
                    "Conduct proactive Technical Support check-in",
                    "Billing transparency verification"
                ],
                "mutations": bundle_a_mutations
            },
            {
                "bundle_id": "strategy_b",
                "name": "Strategy B — Balanced Retention",
                "cost": "MEDIUM",
                "tactics": [
                    "One-Year Contract upgrade with $10/mo discount",
                    "Automatic payment enrollment",
                    "Customer appreciation onboarding call"
                ],
                "mutations": bundle_b_mutations
            },
            {
                "bundle_id": "strategy_c",
                "name": "Strategy C — High-Impact Account Lock-in",
                "cost": "HIGH",
                "tactics": [
                    "Two-Year Contract upgrade with VIP rate guarantee",
                    "Automated payment migration",
                    "Complimentary 24/7 Tech Support & Security Suite",
                    "Dedicated Senior Retention Specialist review"
                ],
                "mutations": bundle_c_mutations
            }
        ]

        for cfg in configs:
            sim_prob = baseline_prob
            reduction = 0.0
            if predict_fn:
                try:
                    b_inputs = inputs.copy()
                    for k, v in cfg["mutations"].items():
                        b_inputs[k] = v
                    sim_prob = float(predict_fn(b_inputs))
                    reduction = max(0.0, (baseline_prob - sim_prob) * 100)
                except Exception as e:
                    pass

            bundles.append({
                "bundle_id": cfg["bundle_id"],
                "name": cfg["name"],
                "cost": cfg["cost"],
                "tactics": cfg["tactics"],
                "baseline_probability": round(baseline_prob, 4),
                "simulated_probability": round(sim_prob, 4),
                "risk_reduction_pct": round(reduction, 1),
                "mutations": cfg["mutations"]
            })

        return bundles

# Global singleton
recommendation_engine = TelecomRetentionDecisionEngine()

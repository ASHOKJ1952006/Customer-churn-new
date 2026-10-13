"""
shap_service.py — Robust SHAP explanation service with multi-tier fallback
Provides feature name alignment, proper positive/negative direction mapping,
and graceful fallback strategies for complex ensemble and calibrated models.
"""

import logging
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Optional, Tuple
from business_factors import get_business_factor

logger = logging.getLogger("shap_service")

# Mapping of raw model column names to readable display names
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    # Contract & Billing
    'PaymentMethod_Electronic check': 'Electronic Check',
    'PaymentMethod_Mailed check': 'Mailed Check',
    'PaymentMethod_Credit card (automatic)': 'Credit Card (Autopay)',
    'PaymentMethod_Bank transfer (automatic)': 'Bank Transfer (Autopay)',
    'Contract_Month-to-month': 'Month-to-Month Contract',
    'Contract_One year': 'One-Year Contract',
    'Contract_Two year': 'Two-Year Contract',
    'PaperlessBilling': 'Paperless Billing',
    'MonthlyCharges': 'Monthly Charges',
    'TotalCharges': 'Total Charges',
    
    # Internet & Services
    'InternetService_Fiber optic': 'Fiber Optic Internet',
    'InternetService_DSL': 'DSL Internet',
    'InternetService_No': 'No Internet Service',
    'PhoneService': 'Phone Service',
    'MultipleLines_Yes': 'Multiple Phone Lines',
    'MultipleLines_No phone service': 'No Phone Service',
    'OnlineSecurity_Yes': 'Online Security Add-on',
    'OnlineSecurity_No internet service': 'No Internet Service',
    'OnlineBackup_Yes': 'Online Backup Add-on',
    'OnlineBackup_No internet service': 'No Internet Service',
    'DeviceProtection_Yes': 'Device Protection Add-on',
    'DeviceProtection_No internet service': 'No Internet Service',
    'TechSupport_Yes': 'Tech Support Add-on',
    'TechSupport_No internet service': 'No Internet Service',
    'StreamingTV_Yes': 'Streaming TV Service',
    'StreamingTV_No internet service': 'No Internet Service',
    'StreamingMovies_Yes': 'Streaming Movies Service',
    'StreamingMovies_No internet service': 'No Internet Service',
    
    # Customer Demographics & Profile
    'tenure': 'Tenure',
    'gender': 'Gender',
    'SeniorCitizen': 'Senior Citizen Status',
    'Partner': 'Has Partner',
    'Dependents': 'Has Dependents',
    
    # Engineered features
    'ChargeIncrease': 'Recent Charge Increase',
    'ChargeIncreaseRatio': 'Charge Increase Ratio',
    'AvgMonthlySpend': 'Average Monthly Spend',
    'MonthlyChargesPerTenure': 'Monthly Spend per Tenure',
    'TotalChargesPerTenure': 'Total Spend per Tenure',
    'ChargeToTenureRatio': 'Charge to Tenure Ratio',
    'TotalChargesDifference': 'Total Charges Difference',
    'TenureSquared': 'Tenure Squared',
    'TenureLog': 'Tenure (Log Scale)',
    'NumServices': 'Number of Services',
    'ContractLength': 'Contract Duration (Months)',
    'IsMonthToMonth': 'Month-to-Month Contract',
    'IsLongTermContract': 'Long-Term Contract',
    'IsElectronicCheck': 'Electronic Check Payment',
    'IsAutomaticPayment': 'Automatic Payment Enabled',
    'HasInternet': 'Has Internet Service',
    'FiberAndMonthToMonth': 'Fiber Optic + Month-to-Month',
    'FiberAndElectronicCheck': 'Fiber Optic + Electronic Check',
    'MonthToMonthAndElectronicCheck': 'M2M + Electronic Check',
    'HighRiskCombo': 'High-Risk Profile (M2M + E-Check)',
    'VeryHighRisk': 'Very High Risk (New + High Risk Profile)',
    'HighChargesAndMonthToMonth': 'High Charges + Month-to-Month',
    'IsNewCustomer': 'New Customer Status',
    'IsLoyalCustomer': 'Loyal Customer Status',
    
    # Tenure buckets
    'TenureGroup_7-12mo': 'Tenure (7-12 Months)',
    'TenureGroup_13-24mo': 'Tenure (13-24 Months)',
    'TenureGroup_25-48mo': 'Tenure (25-48 Months)',
    'TenureGroup_49-72mo': 'Tenure (49-72 Months)',
    'TenureBucket_7-12': 'Tenure (7-12 Months)',
    'TenureBucket_13-24': 'Tenure (13-24 Months)',
    'TenureBucket_25-48': 'Tenure (25-48 Months)',
    'TenureBucket_49+': 'Tenure (49+ Months)',
}

def get_display_name(feature_name: str) -> str:
    """Return friendly display name for a feature column."""
    if feature_name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature_name]
    # Fallback cleanup for underscores and generic naming
    cleaned = feature_name.replace('_', ' ').strip()
    return cleaned.title() if cleaned else feature_name


def get_actual_feature_value(feature_name: str, input_df: pd.DataFrame, raw_inputs: Optional[Dict[str, Any]] = None) -> Any:
    """Retrieve readable actual value for a feature."""
    if raw_inputs:
        if feature_name == 'tenure' and 'tenure' in raw_inputs:
            return f"{raw_inputs['tenure']} mos"
        if feature_name == 'MonthlyCharges' and 'monthly_charges' in raw_inputs:
            return f"${raw_inputs['monthly_charges']:.2f}"
        if feature_name == 'TotalCharges' and 'total_charges' in raw_inputs:
            return f"${raw_inputs['total_charges']:.2f}"
        if feature_name.startswith('Contract_'):
            expected = feature_name.replace('Contract_', '')
            actual = raw_inputs.get('contract', '')
            return "Yes" if actual == expected else "No"
        if feature_name.startswith('PaymentMethod_'):
            expected = feature_name.replace('PaymentMethod_', '')
            actual = raw_inputs.get('payment_method', '')
            return "Yes" if actual == expected else "No"
        if feature_name.startswith('InternetService_'):
            expected = feature_name.replace('InternetService_', '')
            actual = raw_inputs.get('internet_service', '')
            return "Yes" if actual == expected else "No"
        if feature_name == 'HighRiskCombo':
            m2m = raw_inputs.get('contract') == 'Month-to-month'
            echeck = raw_inputs.get('payment_method') == 'Electronic check'
            return "Yes" if (m2m and echeck) else "No"
        if feature_name == 'ChargeIncrease':
            tenure = max(float(raw_inputs.get('tenure', 1)), 1.0)
            avg = float(raw_inputs.get('total_charges', 0)) / tenure
            inc = float(raw_inputs.get('monthly_charges', 0)) - avg
            return f"${inc:+.2f}"
            
    if feature_name in input_df.columns:
        val = input_df[feature_name].iloc[0]
        if isinstance(val, (int, np.integer)) or val in (0.0, 1.0):
            return "Yes" if val == 1 else "No"
        if isinstance(val, (float, np.floating)):
            return f"{val:.2f}"
        return str(val)
    return "N/A"


class RobustSHAPExplainer:
    """
    Computes SHAP explanations with a robust multi-tier fallback architecture:
      - Tier 1: Precomputed explainer if provided and valid.
      - Tier 2 (Method A): TreeExplainer on model if tree-based.
      - Tier 3 (Method B): KernelExplainer with predict_fn wrapping model.predict_proba(X)[:, 1].
      - Tier 4 (Method C): TreeExplainer on the strongest underlying tree estimator from the stacked ensemble fold.
    """

    def __init__(self):
        self._cached_kernel_explainers: Dict[str, shap.KernelExplainer] = {}
        self._cached_tree_explainers: Dict[str, shap.TreeExplainer] = {}

    def extract_shap_vector(self, shap_result: Any, n_features: int) -> Tuple[np.ndarray, Optional[float]]:
        """
        Normalize various SHAP return shapes to a 1D array of class 1 (churn) SHAP values.
        """
        base_val = None
        if isinstance(shap_result, list):
            # Binary classifier: list of [class0_values, class1_values]
            if len(shap_result) > 1:
                arr = np.asarray(shap_result[1])
            else:
                arr = np.asarray(shap_result[0])
            if arr.ndim > 1:
                vec = arr[0]
            else:
                vec = arr
        elif hasattr(shap_result, 'values'):
            # shap.Explanation object
            vals = shap_result.values
            if hasattr(shap_result, 'base_values'):
                bv = shap_result.base_values
                base_val = float(bv[0, 1]) if (hasattr(bv, 'ndim') and bv.ndim > 1) else float(bv[0] if hasattr(bv, '__len__') else bv)
            if vals.ndim == 3:
                vec = vals[0, :, 1]
            elif vals.ndim == 2:
                vec = vals[0]
            else:
                vec = vals
        else:
            arr = np.asarray(shap_result)
            if arr.ndim == 3:
                # Shape (1, n_features, 2)
                vec = arr[0, :, 1]
            elif arr.ndim == 2:
                # Shape (1, n_features)
                vec = arr[0]
            else:
                vec = arr

        vec = np.asarray(vec, dtype=float)
        if len(vec) != n_features:
            raise ValueError(f"Extracted SHAP vector length {len(vec)} does not match feature count {n_features}")
        return vec, base_val

    def explain(
        self,
        model: Any,
        input_df: pd.DataFrame,
        feature_columns: List[str],
        precomputed_explainer: Optional[Any] = None,
        raw_inputs: Optional[Dict[str, Any]] = None,
        dataset_name: str = "ds1"
    ) -> Dict[str, Any]:
        """
        Compute aligned, directional SHAP explanation with fallbacks.
        """
        n_features = len(feature_columns)
        # Ensure DataFrame has exact columns
        if not isinstance(input_df, pd.DataFrame):
            input_df = pd.DataFrame(input_df, columns=feature_columns)
        else:
            input_df = input_df[feature_columns]

        shap_values: Optional[np.ndarray] = None
        base_value: float = 0.5
        explainer_type: str = "Unknown"
        method_used: str = "none"

        # --- Tier 1: Try precomputed explainer if available ---
        if precomputed_explainer is not None:
            try:
                res = precomputed_explainer.shap_values(input_df)
                vec, bv = self.extract_shap_vector(res, n_features)
                if not np.any(np.isnan(vec)) and not np.any(np.isinf(vec)):
                    shap_values = vec
                    explainer_type = type(precomputed_explainer).__name__
                    if bv is not None:
                        base_value = float(bv)
                    elif hasattr(precomputed_explainer, 'expected_value'):
                        ev = precomputed_explainer.expected_value
                        base_value = float(ev[1] if (hasattr(ev, '__len__') and len(ev) > 1) else (ev[0] if hasattr(ev, '__len__') else ev))
                    method_used = f"Precomputed ({explainer_type})"
            except Exception as e:
                logger.warning(f"Precomputed explainer failed: {e}. Falling back to dynamic strategies.")

        # --- Tier 2: Method A - Direct TreeExplainer if tree-based model ---
        if shap_values is None:
            model_name = type(model).__name__.lower()
            if any(t in model_name for t in ['xgb', 'lgbm', 'lightgbm', 'forest', 'tree', 'catboost']):
                try:
                    tree_exp = shap.TreeExplainer(model)
                    res = tree_exp.shap_values(input_df)
                    vec, bv = self.extract_shap_vector(res, n_features)
                    if not np.any(np.isnan(vec)) and not np.any(np.isinf(vec)):
                        shap_values = vec
                        explainer_type = "TreeExplainer"
                        if hasattr(tree_exp, 'expected_value'):
                            ev = tree_exp.expected_value
                            base_value = float(ev[1] if (hasattr(ev, '__len__') and len(ev) > 1) else (ev[0] if hasattr(ev, '__len__') else ev))
                        method_used = "Method A (Direct TreeExplainer)"
                except Exception as e:
                    logger.info(f"Method A TreeExplainer not directly applicable: {e}")

        # --- Tier 3: Method B - KernelExplainer on predict_proba wrapper ---
        if shap_values is None:
            try:
                cache_key = f"kernel_{dataset_name}"
                if cache_key not in self._cached_kernel_explainers:
                    # Construct a representative small background baseline
                    bg = pd.DataFrame(np.zeros((5, n_features)), columns=feature_columns)
                    
                    def predict_fn(X_arr):
                        if not isinstance(X_arr, pd.DataFrame):
                            X_df = pd.DataFrame(X_arr, columns=feature_columns)
                        else:
                            X_df = X_arr
                        preds = model.predict_proba(X_df)
                        return preds[:, 1]

                    kernel_exp = shap.KernelExplainer(predict_fn, bg)
                    self._cached_kernel_explainers[cache_key] = kernel_exp
                else:
                    kernel_exp = self._cached_kernel_explainers[cache_key]

                res = kernel_exp.shap_values(input_df, nsamples=100)
                vec, bv = self.extract_shap_vector(res, n_features)
                if not np.any(np.isnan(vec)) and not np.any(np.isinf(vec)):
                    shap_values = vec
                    explainer_type = "KernelExplainer (Probability Wrapper)"
                    ev = kernel_exp.expected_value
                    base_value = float(ev[1] if (hasattr(ev, '__len__') and len(ev) > 1) else (ev[0] if hasattr(ev, '__len__') else ev))
                    method_used = "Method B (KernelExplainer Predict Proba Wrapper)"
            except Exception as e:
                logger.warning(f"Method B KernelExplainer failed: {e}")

        # --- Tier 4: Method C - Explain underlying tree estimator in ensemble ---
        if shap_values is None:
            try:
                sub_estimator = None
                sub_est_name = "Underlying Tree Estimator"
                
                # Check for CalibratedClassifierCV
                if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
                    cal0 = model.calibrated_classifiers_[0]
                    base = getattr(cal0, 'estimator', getattr(cal0, 'base_estimator', None))
                    if hasattr(base, 'named_steps') and 'stack' in base.named_steps:
                        stack = base.named_steps['stack']
                        if hasattr(stack, 'named_estimators_'):
                            # Choose LGBM or RF
                            if 'lgbm' in stack.named_estimators_:
                                sub_estimator = stack.named_estimators_['lgbm']
                                sub_est_name = "Fold-0 Stacking LightGBM"
                            elif 'rf' in stack.named_estimators_:
                                sub_estimator = stack.named_estimators_['rf']
                                sub_est_name = "Fold-0 Stacking Random Forest"
                    elif base is not None:
                        sub_estimator = base
                        sub_est_name = f"Fold-0 {type(base).__name__}"

                if sub_estimator is None and hasattr(model, 'named_estimators_'):
                    if 'lgbm' in model.named_estimators_:
                        sub_estimator = model.named_estimators_['lgbm']
                        sub_est_name = "Stacking LightGBM Base Model"
                    elif 'rf' in model.named_estimators_:
                        sub_estimator = model.named_estimators_['rf']
                        sub_est_name = "Stacking Random Forest Base Model"
                    elif 'xgb' in model.named_estimators_:
                        sub_estimator = model.named_estimators_['xgb']
                        sub_est_name = "Stacking XGBoost Base Model"

                if sub_estimator is not None:
                    tree_exp = shap.TreeExplainer(sub_estimator)
                    res = tree_exp.shap_values(input_df)
                    vec, bv = self.extract_shap_vector(res, n_features)
                    if not np.any(np.isnan(vec)) and not np.any(np.isinf(vec)):
                        shap_values = vec
                        explainer_type = f"TreeExplainer ({sub_est_name})"
                        if hasattr(tree_exp, 'expected_value'):
                            ev = tree_exp.expected_value
                            base_value = float(ev[1] if (hasattr(ev, '__len__') and len(ev) > 1) else (ev[0] if hasattr(ev, '__len__') else ev))
                        method_used = f"Method C (TreeExplainer on {sub_est_name})"
            except Exception as e:
                logger.exception("Method C TreeExplainer failed on underlying estimator")

        # --- Check if all strategies failed ---
        if shap_values is None:
            logger.error("All SHAP explanation strategies failed.")
            return {
                "available": False,
                "error": "Explanation temporarily unavailable across all calculation strategies.",
                "explainer_type": "None",
                "base_value": 0.0,
                "features": [],
                "top_risk_factors": [],
                "top_protective_factors": []
            }

        # --- Map SHAP values to structured output ---
        features_list = []
        top_risk_factors = []
        top_protective_factors = []

        for idx, col in enumerate(feature_columns):
            val = float(shap_values[idx])
            bf = get_business_factor(col)
            display_name = bf.get("business_name") or get_display_name(col)
            actual_val = get_actual_feature_value(col, input_df, raw_inputs)
            
            # Correct sign convention:
            # Positive SHAP: pushes prediction toward CHURN (Risk)
            # Negative SHAP: pushes prediction toward NOT CHURN (Protective)
            is_risk = val > 0
            direction = "increases_churn" if is_risk else "decreases_churn"
            impact = "risk" if is_risk else "protective"
            importance = abs(val)

            feat_obj = {
                "feature": col,
                "display_name": display_name,
                "category": bf.get("category", "General"),
                "is_actionable": bf.get("is_actionable", True),
                "business_meaning": bf.get("meaning", ""),
                "value": actual_val,
                "shap_value": round(val, 4),
                "importance": round(importance, 4),
                "impact": impact,
                "direction": direction,
            }
            features_list.append(feat_obj)

            if is_risk and importance > 0.0001:
                top_risk_factors.append({
                    "feature": col,
                    "display_name": display_name,
                    "category": bf.get("category", "General"),
                    "is_actionable": bf.get("is_actionable", True),
                    "shap_value": round(val, 4),
                    "value": actual_val,
                    "importance": round(importance, 4)
                })
            elif not is_risk and importance > 0.0001:
                top_protective_factors.append({
                    "feature": col,
                    "display_name": display_name,
                    "category": bf.get("category", "General"),
                    "is_actionable": bf.get("is_actionable", True),
                    "shap_value": round(val, 4),
                    "value": actual_val,
                    "importance": round(importance, 4)
                })

        # Sort factors by importance (magnitude) descending
        features_list.sort(key=lambda x: x["importance"], reverse=True)
        top_risk_factors.sort(key=lambda x: x["importance"], reverse=True)
        top_protective_factors.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "available": True,
            "explainer_type": explainer_type,
            "method_used": method_used,
            "base_value": round(float(base_value), 4),
            "features": features_list,
            "top_risk_factors": top_risk_factors[:5],
            "top_protective_factors": top_protective_factors[:5]
        }

def combine_shap_and_lime(
    shap_summary: Dict[str, Any],
    lime_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Combines SHAP and LIME explanations via normalized contribution scoring.
    Does NOT blindly average raw values because they operate in different explanation spaces.
    
    Formula:
      norm_shap = |shap_val| / max(|shap_all|)
      norm_lime = |lime_val| / max(|lime_all|)
      sign_agreement = 1.0 if sign(shap) == sign(lime) else 0.0
      combined_score = 0.5 * norm_shap + 0.5 * norm_lime + 0.2 * sign_agreement
    """
    shap_feats = {f["feature"]: f for f in shap_summary.get("features", [])}
    lime_feats = {f["feature"]: f for f in lime_summary.get("features", [])}
    
    if not shap_feats and not lime_feats:
        return {
            "combined_factors": [],
            "explanation_confidence": "LOW",
            "concordance_rate": 0.0,
            "agreement_summary": "Neither SHAP nor LIME explanations were available."
        }

    max_shap = max([abs(f.get("shap_value", 0.0)) for f in shap_feats.values()] or [1.0])
    max_lime = max([abs(f.get("lime_weight", 0.0)) for f in lime_feats.values()] or [1.0])
    if max_shap == 0: max_shap = 1.0
    if max_lime == 0: max_lime = 1.0

    all_keys = set(shap_feats.keys()).union(set(lime_feats.keys()))
    combined_list = []
    concordance_matches = 0
    top_driver_comparisons = 0

    for feat_key in all_keys:
        s_obj = shap_feats.get(feat_key, {})
        l_obj = lime_feats.get(feat_key, {})
        
        s_val = float(s_obj.get("shap_value", 0.0))
        l_val = float(l_obj.get("lime_weight", 0.0))
        
        norm_s = abs(s_val) / max_shap
        norm_l = abs(l_val) / max_lime
        
        # Agreement evaluation
        same_sign = (s_val > 0 and l_val > 0) or (s_val < 0 and l_val < 0)
        agreement_score = 1.0 if same_sign else 0.0
        
        combined_importance = (0.5 * norm_s) + (0.5 * norm_l) + (0.2 * agreement_score)
        direction = "increases_churn" if (s_val > 0 or (s_val == 0 and l_val > 0)) else "decreases_churn"
        
        bf = get_business_factor(feat_key)
        display_name = s_obj.get("display_name") or l_obj.get("display_name") or bf.get("business_name", feat_key)
        
        combined_list.append({
            "feature": feat_key,
            "display_name": display_name,
            "category": bf.get("category", "General"),
            "is_actionable": bf.get("is_actionable", True),
            "business_meaning": bf.get("meaning", ""),
            "shap_value": round(s_val, 4),
            "lime_weight": round(l_val, 4),
            "normalized_shap": round(norm_s, 4),
            "normalized_lime": round(norm_l, 4),
            "sign_agreement": same_sign,
            "combined_score": round(combined_importance, 4),
            "direction": direction,
            "impact": "risk" if direction == "increases_churn" else "protective",
            "actual_value": s_obj.get("value", "N/A")
        })

    # Sort by combined score descending
    combined_list.sort(key=lambda x: x["combined_score"], reverse=True)

    # Check concordance on top 10 factors
    top_check = combined_list[:10]
    for item in top_check:
        top_driver_comparisons += 1
        if item["sign_agreement"]:
            concordance_matches += 1

    concordance_rate = (concordance_matches / max(top_driver_comparisons, 1))

    if concordance_rate >= 0.70:
        confidence = "HIGH"
        agreement_summary = f"High Confidence ({int(concordance_rate*100)}% Agreement): Both SHAP and LIME strongly agree on the primary factors driving churn."
    elif concordance_rate >= 0.45:
        confidence = "MEDIUM"
        agreement_summary = f"Medium Confidence ({int(concordance_rate*100)}% Agreement): SHAP and LIME agree on core drivers with slight local ranking variance."
    else:
        confidence = "LOW"
        agreement_summary = f"Low Confidence ({int(concordance_rate*100)}% Agreement): SHAP and LIME show differing local feature weights; evaluate carefully."

    return {
        "combined_factors": combined_list,
        "explanation_confidence": confidence,
        "concordance_rate": round(concordance_rate, 2),
        "agreement_summary": agreement_summary,
        "top_combined_risk": [f for f in combined_list if f["direction"] == "increases_churn"][:5],
        "top_combined_protective": [f for f in combined_list if f["direction"] == "decreases_churn"][:5]
    }

# Global singleton
robust_explainer = RobustSHAPExplainer()


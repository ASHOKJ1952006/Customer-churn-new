"""
main.py — FastAPI backend for Churn Prediction API
Wraps existing model logic from app.py without rewriting prediction logic
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import joblib
import shap
from xgboost import XGBClassifier
import uuid
import os
from datetime import datetime
from database import ChurnDatabase
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from scripts.stacking_model import StackingEnsemble
import scripts.stacking_model as stacking_model
sys.modules['stacking_model'] = stacking_model
from shap_explainer import SHAPToTextExplainer
from shap_service import robust_explainer, combine_shap_and_lime
from lime_service import lime_service
from recommendation_engine import recommendation_engine, determine_risk_level
from business_factors import get_business_factor
from feature_engineering import build_single_customer_vector, engineer_features, CONTINUOUS_NUMERIC_COLS
import logging

logger = logging.getLogger("main")

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:63221", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = ChurnDatabase()

def load_model_file(filename: str):
    """Safely locate and load serialized model artifacts across candidate directories."""
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), 'models', filename),
        os.path.join(os.path.dirname(__file__), '..', 'models', filename),
        os.path.join('models', filename),
        os.path.join('..', 'models', filename),
        filename
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError(f"Could not locate model file {filename} in candidate paths: {candidate_paths}")

# --- Load Dataset 1 models (Telco, Calibrated CV v3.0 - Reloaded) ---
ds1_model = load_model_file('ds1calcv_churn_model.pkl')
ds1_scaler = load_model_file('ds1calcv_scaler.pkl')
ds1_feature_columns = load_model_file('ds1calcv_feature_columns.pkl')
ds1_best_threshold = load_model_file('ds1calcv_best_threshold.pkl')
try:
    ds1_explainer = load_model_file('ds1calcv_shap_explainer.pkl')
except Exception as e:
    logger.warning(f"Could not load ds1calcv_shap_explainer.pkl: {e}. Fallback explainer will be used.")
    ds1_explainer = None

# --- Load Dataset 2 models (Telecom, Optimized) ---
ds2_model = load_model_file('ds2optuna_churn_model.pkl')
ds2_scaler = load_model_file('ds2optuna_scaler.pkl')
ds2_feature_columns = load_model_file('ds2optuna_feature_columns.pkl')
ds2_best_threshold = load_model_file('ds2optuna_best_threshold.pkl')
try:
    ds2_explainer = load_model_file('ds2optuna_shap_explainer.pkl')
except Exception as e:
    logger.warning(f"Could not load ds2optuna_shap_explainer.pkl: {e}. Fallback explainer will be used.")
    ds2_explainer = None

# --- Load Cox survival model for time-to-churn prediction ---
try:
    cox_model = load_model_file('survival_cox_model.pkl')
except Exception:
    cox_model = None

cox_features = [
    'MonthlyCharges', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'PaperlessBilling', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Electronic check', 'InternetService_Fiber optic',
    'InternetService_No', 'OnlineSecurity_Yes', 'TechSupport_Yes', 'HighRiskCombo'
]

# Global variables for active model (will be set per request)
model = ds1_model
scaler = ds1_scaler
feature_columns = ds1_feature_columns
best_threshold = ds1_best_threshold
explainer = ds1_explainer
current_dataset = "ds1"

# Force GPU-trained models to CPU for inference stability
def force_cpu_on_xgb(model):
    """Recursively force XGBClassifier to CPU, handling nested structures."""
    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            force_cpu_on_xgb(est)
    if hasattr(model, 'named_steps'):
        for step_name, step in model.named_steps.items():
            force_cpu_on_xgb(step)
    if hasattr(model, 'calibrated_classifiers_'):
        for cal in model.calibrated_classifiers_:
            force_cpu_on_xgb(cal.estimator)
    if isinstance(model, XGBClassifier):
        model.set_params(device='cpu')

force_cpu_on_xgb(ds1_model)
force_cpu_on_xgb(ds2_model)

ACTION_RULES = {
    'Contract': 'Offer discount to upgrade to a longer-term contract for better lock-in.',
    'HighRiskCombo': 'High-risk profile (month-to-month + electronic check) — prioritize proactive outreach.',
    'InternetService_Fiber optic': 'Investigate fiber service quality or offer a fiber plan discount.',
    'PaymentMethod_Electronic check': 'Encourage switch to auto-debit/UPI with a small incentive.',
    'PaperlessBilling': 'Offer a loyalty reward tied to billing preference.',
    'ChargeIncrease': 'Recent bill increase detected — consider a loyalty discount to offset it.',
    'MonthlyCharges': 'Offer a bundled discount to reduce effective monthly cost.',
    'IsNewCustomer': 'New customer — assign to onboarding/early retention outreach program.',
    'tenure': 'Assign to early-tenure onboarding/retention outreach program.',
    'NumServices': 'Low service engagement — offer a bundle to increase platform investment.',
    'OnlineSecurity': 'Bundle free online security add-on for 3 months.',
    'TechSupport': 'Proactively offer free tech support session.',
}

def top_factors_to_action(top_factors):
    for feature in top_factors:
        for key, action in ACTION_RULES.items():
            if key in feature:
                return action
    return 'Review customer profile for a tailored retention offer.'

def set_active_dataset(dataset: str):
    """Set active model based on dataset selection"""
    global model, scaler, feature_columns, best_threshold, explainer, current_dataset
    if dataset == "ds1":
        model = ds1_model
        scaler = ds1_scaler
        feature_columns = ds1_feature_columns
        best_threshold = ds1_best_threshold
        explainer = ds1_explainer
        current_dataset = "ds1"
    else:
        model = ds2_model
        scaler = ds2_scaler
        feature_columns = ds2_feature_columns
        best_threshold = ds2_best_threshold
        explainer = ds2_explainer
        current_dataset = "ds2"

def risk_label(prob):
    return determine_risk_level(prob, best_threshold)

def build_feature_vector(inputs):
    """Build feature vector from inputs using consistent feature_engineering module."""
    if current_dataset == "ds1":
        scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in feature_columns]
        return build_single_customer_vector(inputs, feature_columns, scaler, scale_cols)
    else:
        # Dataset 2 feature building (simplified for demo)
        raw = {}
        for col in feature_columns:
            if col in inputs:
                raw[col] = inputs[col]
            elif '_6' in col:
                base = col.replace('_6', '')
                if f'{base}_6' in inputs:
                    raw[col] = inputs[f'{base}_6']
                else:
                    raw[col] = 0
            elif '_7' in col:
                base = col.replace('_7', '')
                if f'{base}_7' in inputs:
                    raw[col] = inputs[f'{base}_7']
                else:
                    raw[col] = 0
            elif '_8' in col:
                base = col.replace('_8', '')
                if f'{base}_8' in inputs:
                    raw[col] = inputs[f'{base}_8']
                else:
                    raw[col] = 0
            elif 'trend' in col:
                base = col.replace('_trend', '')
                if f'{base}_8' in inputs and f'{base}_6' in inputs:
                    raw[col] = inputs[f'{base}_8'] - inputs[f'{base}_6']
                else:
                    raw[col] = 0
            elif 'pct_change' in col:
                base = col.replace('_pct_change', '')
                if f'{base}_8' in inputs and f'{base}_6' in inputs:
                    raw[col] = (inputs[f'{base}_8'] - inputs[f'{base}_6']) / (inputs[f'{base}_6'] + 1e-6)
                else:
                    raw[col] = 0
            elif 'avg' in col:
                base = col.replace('_avg', '')
                vals = [inputs.get(f'{base}_6', 0), inputs.get(f'{base}_7', 0), inputs.get(f'{base}_8', 0)]
                raw[col] = sum(vals) / len(vals) if vals else 0
            elif 'std' in col:
                base = col.replace('_std', '')
                vals = [inputs.get(f'{base}_6', 0), inputs.get(f'{base}_7', 0), inputs.get(f'{base}_8', 0)]
                raw[col] = np.std(vals) if vals else 0
            else:
                raw[col] = 0
        
        input_df = pd.DataFrame([raw])[feature_columns].astype(float)
        if hasattr(scaler, 'feature_names_in_'):
            scale_cols = list(scaler.feature_names_in_)
            scale_cols = [col for col in scale_cols if col in input_df.columns]
            input_df[scale_cols] = scaler.transform(input_df[scale_cols])
        return input_df

def predict_probability(inputs):
    """Predict churn probability from raw inputs."""
    X = build_feature_vector(inputs)
    prob = model.predict_proba(X)[0, 1]
    return prob

def predict_median_survival(inputs):
    """Predict median survival time using Cox model (complementary to classification)."""
    # Simplified Cox prediction for demo
    # In production, this would use the full Cox model logic
    return 12.0  # Placeholder - median survival in months

def preprocess_raw_dataframe(raw_df):
    """Preprocess raw dataframe for batch prediction - ported from app.py"""
    df = raw_df.copy()
    
    # Handle missing values
    df = df.fillna(0)
    
    # For Dataset 1, apply similar encoding as build_feature_vector
    if current_dataset == "ds1":
        customer_ids = df.get('customerID', [f"cust_{i}" for i in range(len(df))]).tolist()
        df_encoded = engineer_features(df, feature_set='B')
        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0.0
        df_encoded = df_encoded[feature_columns].astype(float)
        scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in df_encoded.columns]
        if hasattr(scaler, 'transform') and scale_cols:
            df_encoded[scale_cols] = scaler.transform(df_encoded[scale_cols])
        return df_encoded, customer_ids
    else:
        # Dataset 2 preprocessing (simplified)
        customer_ids = df.get('mobile_number', [f"cust_{i}" for i in range(len(df))]).tolist()
        return df[feature_columns].astype(float), customer_ids

# Pydantic models for request/response
class SinglePredictRequest(BaseModel):
    dataset: str = "ds1"
    tenure: int = 12
    monthly_charges: float = 70.0
    total_charges: float = 1000.0
    contract: str = "Month-to-month"
    payment_method: str = "Electronic check"
    paperless_billing: str = "Yes"
    phone_service: str = "Yes"
    multiple_lines: str = "No"
    internet_service: str = "DSL"
    online_security: str = "No"
    online_backup: str = "No"
    device_protection: str = "No"
    tech_support: str = "No"
    streaming_tv: str = "No"
    streaming_movies: str = "No"
    gender: str = "Male"
    partner: str = "No"
    dependents: str = "No"
    senior_citizen: Union[str, int, bool] = "No"
    # Dataset 2 specific fields
    arpu_6: Optional[float] = None
    arpu_7: Optional[float] = None
    arpu_8: Optional[float] = None
    total_og_mou_6: Optional[float] = None
    total_og_mou_7: Optional[float] = None
    total_og_mou_8: Optional[float] = None
    total_ic_mou_6: Optional[float] = None
    total_ic_mou_7: Optional[float] = None
    total_ic_mou_8: Optional[float] = None
    total_rech_amt_6: Optional[float] = None
    total_rech_amt_7: Optional[float] = None
    total_rech_amt_8: Optional[float] = None

class WhatIfRequest(BaseModel):
    dataset: str = "ds1"
    base_profile: SinglePredictRequest
    factor: str = "contract"

class RiskQueueUpdate(BaseModel):
    status: str
    notes: Optional[str] = None

# API Routes

@app.get("/")
async def root():
    return {"message": "Churn Prediction API", "version": "1.0.0"}

@app.post("/api/predict/single")
async def predict_single(request: SinglePredictRequest):
    """Predict churn for a single customer with robust SHAP and dynamic recommendations"""
    try:
        # Set active dataset
        set_active_dataset(request.dataset)
        
        # Convert request to dict (using model_dump for Pydantic V2)
        inputs = request.model_dump()
        
        # Predict probability
        prob = float(predict_probability(inputs))
        input_df = build_feature_vector(inputs)
        
        # Generate robust SHAP explanation
        shap_res = robust_explainer.explain(
            model=model,
            input_df=input_df,
            feature_columns=feature_columns,
            precomputed_explainer=explainer,
            raw_inputs=inputs,
            dataset_name=request.dataset
        )

        # Generate robust LIME explanation
        lime_res = lime_service.explain(
            model=model,
            input_df=input_df,
            feature_columns=feature_columns,
            dataset_name=request.dataset,
            scaler=scaler,
            raw_inputs=inputs,
            num_samples=300
        )

        # Combine SHAP and LIME signals via normalized contribution scoring
        combined_res = combine_shap_and_lime(shap_res, lime_res)
        
        # Generate dynamic recommendations with real counterfactual simulations
        rec_res = recommendation_engine.generate_recommendations(
            inputs=inputs,
            probability=prob,
            threshold=float(best_threshold),
            shap_summary=shap_res,
            lime_summary=lime_res,
            combined_summary=combined_res,
            predict_fn=predict_probability
        )
        
        # Prepare legacy top factors for existing frontend components
        if shap_res.get("available") and shap_res.get("features"):
            top_factors_list = [
                {"name": f["display_name"], "value": float(f["shap_value"])}
                for f in shap_res["features"][:5]
            ]
        else:
            top_factors_list = [{"name": "Explanation temporarily unavailable", "value": 0.0}]
            
        # Plain English explanation
        if shap_res.get("available") and shap_res.get("features"):
            text_explainer = SHAPToTextExplainer()
            shap_vals_arr = np.array([f["shap_value"] for f in shap_res["features"]])
            explanation = text_explainer.generate_explanation(
                shap_vals_arr,
                [f["feature"] for f in shap_res["features"]],
                feature_values=None,
                top_n=3
            )
        else:
            explanation = "SHAP explanation temporarily unavailable."
        
        # Primary recommended action
        action = rec_res["primary_action"]
        
        # Get survival prediction
        survival = predict_median_survival(inputs)
        
        # Save to database with extended decision support fields
        try:
            top_drivers = rec_res.get("major_churn_drivers", [])
            recs = rec_res.get("recommendations", [])
            db.add_single_prediction(
                customer_id=f"single_{uuid.uuid4().hex[:8]}",
                churn_probability=prob,
                risk_level=rec_res["risk_level"],
                top_factors=[f["factor"] for f in top_drivers[:3]] if top_drivers else [f["name"] for f in top_factors_list],
                recommended_action=action,
                input_features=inputs,
                prediction=1 if prob >= float(best_threshold) else 0,
                top_factor_1=top_drivers[0]["factor"] if len(top_drivers) > 0 else None,
                top_factor_2=top_drivers[1]["factor"] if len(top_drivers) > 1 else None,
                top_factor_3=top_drivers[2]["factor"] if len(top_drivers) > 2 else None,
                recommended_action_1=recs[0]["action"] if len(recs) > 0 else action,
                recommended_action_2=recs[1]["action"] if len(recs) > 1 else None,
                recommended_action_3=recs[2]["action"] if len(recs) > 2 else None,
                recommendation_priority=rec_res.get("management_summary", {}).get("priority", "HIGH"),
                recommendation_confidence=rec_res.get("explanation_confidence", "HIGH")
            )
        except Exception as db_error:
            logger.warning(f"Database error (non-critical): {db_error}")
        
        return {
            # Core prediction metrics
            "prediction": 1 if prob >= float(best_threshold) else 0,
            "probability": prob,
            "churn_probability": prob,
            "risk_level": rec_res["risk_level"],
            "threshold": float(best_threshold),
            
            # Telecom Business Decision Support Layer
            "management_summary": rec_res.get("management_summary", {}),
            "customer_summary": rec_res.get("customer_summary", ""),
            "major_factors": rec_res.get("major_churn_drivers", []),
            "major_churn_drivers": rec_res.get("major_churn_drivers", []),
            "protective_factors": rec_res.get("protective_factors", []),
            "recommendations": rec_res.get("recommendations", []),
            "immediate_actions": rec_res.get("immediate_actions", []),
            "short_term_actions": rec_res.get("short_term_actions", []),
            "long_term_actions": rec_res.get("long_term_actions", []),
            "strategy_bundles": rec_res.get("strategy_bundles", []),
            "explanation_confidence": rec_res.get("explanation_confidence", "HIGH"),
            "agreement_summary": rec_res.get("agreement_summary", ""),
            
            # Explainability models
            "shap": shap_res,
            "lime": lime_res,
            "combined_factors": combined_res.get("combined_factors", []),
            
            # Legacy fields for existing frontend consumers
            "top_factors": top_factors_list,
            "recommended_action": action,
            "survival_months": survival,
            "explanation": explanation,
            "top_risk_factors": shap_res.get("top_risk_factors", []),
            "top_protective_factors": shap_res.get("top_protective_factors", []),
            "risk_drivers": rec_res.get("risk_drivers", []),
            "protective_drivers": rec_res.get("protective_drivers", [])
        }
    except Exception as e:
        import traceback
        logger.exception(f"Error in predict_single: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...), dataset: str = Form("ds1")):
    """Predict churn for batch CSV upload"""
    try:
        # Set active dataset
        set_active_dataset(dataset)
        
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Preprocess
        X_batch, customer_ids = preprocess_raw_dataframe(df)
        
        # Predict
        probs = model.predict_proba(X_batch)[:, 1]
        
        # Get SHAP values
        if explainer:
            shap_vals_batch_result = explainer.shap_values(X_batch)
            if isinstance(shap_vals_batch_result, list):
                shap_vals_batch = shap_vals_batch_result[1] if len(shap_vals_batch_result) > 1 else shap_vals_batch_result[0]
            elif len(shap_vals_batch_result.shape) == 3:
                shap_vals_batch = shap_vals_batch_result[:, :, 1]
            else:
                shap_vals_batch = shap_vals_batch_result
        else:
            shap_vals_batch = None
        
        # Process results
        results = []
        for i in range(len(X_batch)):
            if shap_vals_batch is not None:
                row_shap = pd.Series(shap_vals_batch[i], index=feature_columns)
                top3 = row_shap.sort_values(ascending=False).head(3).index.tolist()
            else:
                top3 = ["SHAP unavailable"]
            
            prob = probs[i]
            risk = risk_label(prob)
            action = top_factors_to_action(top3)
            
            results.append({
                "customer_id": customer_ids[i] if i < len(customer_ids) else f"cust_{i}",
                "probability": float(prob),
                "risk_level": risk,
                "top_factors": top3,
                "recommended_action": action
            })
        
        # Organizational Aggregations (Requirement 24)
        driver_counter = {}
        strategy_counter = {}
        
        for r in results:
            # Map technical feature names to business categories
            for feat in r["top_factors"]:
                bf = get_business_factor(feat)
                bname = bf.get("business_name", feat.replace('_', ' ').title())
                driver_counter[bname] = driver_counter.get(bname, 0) + 1
            
            # Categorize retention campaigns
            act = r["recommended_action"].lower()
            if "contract" in act or "term" in act:
                strat = "Contract Commitment Upgrade Campaign"
            elif "electronic" in act or "check" in act or "auto" in act or "payment" in act:
                strat = "Autopay & Digital Payment Migration"
            elif "charge" in act or "bill" in act or "price" in act:
                strat = "Pricing Transparency & Tariff Review Campaign"
            elif "new" in act or "onboard" in act or "tenure" in act:
                strat = "Early-Tenure Onboarding & Care Outreach"
            elif "support" in act:
                strat = "Technical Support Assistance Outreach"
            elif "fiber" in act:
                strat = "Fiber Quality-of-Service Audit Campaign"
            else:
                strat = "Customer Value & Loyalty Outreach"
            
            strategy_counter[strat] = strategy_counter.get(strat, 0) + 1
            
        total_custs = max(len(results), 1)
        top_org_drivers = [
            {"driver": k, "count": v, "percentage": round((v / total_custs) * 100, 1)}
            for k, v in sorted(driver_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        top_org_strategies = [
            {"strategy": k, "count": v, "percentage": round((v / total_custs) * 100, 1)}
            for k, v in sorted(strategy_counter.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        # Calculate summary
        high_count = sum(1 for r in results if r["risk_level"] in ["High", "Critical"])
        medium_count = sum(1 for r in results if r["risk_level"] == "Medium")
        low_count = sum(1 for r in results if r["risk_level"] == "Low")
        
        # Generate batch ID
        batch_id = str(uuid.uuid4())
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"batch_results_{batch_id}.csv", index=False)
        
        # Save to database
        try:
            db.add_batch_predictions(batch_id, [
                {
                    'customer_id': r["customer_id"],
                    'churn_probability': r["probability"],
                    'risk_level': r["risk_level"],
                    'top_factors': r["top_factors"],
                    'recommended_action': r["recommended_action"]
                }
                for r in results
            ])
        except Exception as db_error:
            print(f"Database error (non-critical): {db_error}")
        
        return {
            "batch_id": batch_id,
            "summary": {
                "total": len(results),
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
                "top_organizational_drivers": top_org_drivers,
                "top_retention_strategies": top_org_strategies
            },
            "results": results
        }
    except Exception as e:
        import traceback
        print(f"Error in predict_batch: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/batch/download/{batch_id}")
async def download_batch(batch_id: str):
    """Download batch results CSV"""
    file_path = f"batch_results_{batch_id}.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Batch not found")
    return FileResponse(file_path, filename=f"batch_results_{batch_id}.csv")

@app.post("/api/whatif/simulate")
@app.post("/api/simulate/what-if")
async def whatif_simulate(request: WhatIfRequest):
    """What-if simulation for different factor values with dynamic recommendations"""
    try:
        # Set active dataset
        set_active_dataset(request.dataset)
        
        base_inputs = request.base_profile.dict()
        factor = request.factor
        
        # Get baseline probability
        baseline_prob = predict_probability(base_inputs)
        
        # Define scenarios based on factor
        scenarios = []
        if factor == "contract":
            values = ["Month-to-month", "One year", "Two year"]
        elif factor == "payment_method":
            values = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        elif factor == "internet_service":
            values = ["DSL", "Fiber optic", "No"]
        else:
            values = ["Yes", "No"]
        
        for value in values:
            test_inputs = base_inputs.copy()
            test_inputs[factor] = value
            prob = float(predict_probability(test_inputs))
            scenarios.append({
                "scenario_label": value,
                "probability": prob
            })
        
        # Find best option
        best_scenario = min(scenarios, key=lambda x: x["probability"])
        
        # Generate dynamic recommendation for best scenario
        best_inputs = base_inputs.copy()
        best_inputs[factor] = best_scenario["scenario_label"]
        best_df = build_feature_vector(best_inputs)
        
        best_shap = robust_explainer.explain(
            model=model,
            input_df=best_df,
            feature_columns=feature_columns,
            precomputed_explainer=explainer,
            raw_inputs=best_inputs,
            dataset_name=request.dataset
        )
        
        best_rec = recommendation_engine.generate_recommendations(
            inputs=best_inputs,
            probability=best_scenario["probability"],
            threshold=float(best_threshold),
            shap_summary=best_shap
        )
        
        best_action = f"Changing {factor.replace('_', ' ')} to '{best_scenario['scenario_label']}' lowers churn probability from {baseline_prob*100:.1f}% to {best_scenario['probability']*100:.1f}%. {best_rec['primary_action']}"
        
        return {
            "baseline_probability": float(baseline_prob),
            "scenarios": scenarios,
            "best_option": best_scenario["scenario_label"],
            "best_probability": best_scenario["probability"],
            "recommendation": best_action,
            "recommendations": best_rec.get("recommendations", [])
        }
    except Exception as e:
        logger.exception(f"Error in whatif_simulate: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/history")
async def get_history():
    """Get prediction history from database (both single and batch)"""
    try:
        # Get single predictions
        single_df = db.get_single_predictions(limit=100)
        single_records = single_df.to_dict('records') if not single_df.empty else []
        
        # Get batch predictions
        batch_df = db.get_batch_predictions(limit=100)
        batch_records = batch_df.to_dict('records') if not batch_df.empty else []
        
        # Combine and sort by timestamp
        all_records = []
        for record in single_records:
            all_records.append({
                'customer_id': record.get('customer_id'),
                'probability': record.get('churn_probability'),
                'risk_level': record.get('risk_level'),
                'top_factors': record.get('top_factors'),
                'recommended_action': record.get('recommended_action'),
                'timestamp': record.get('timestamp'),
                'type': 'single'
            })
        
        for record in batch_records:
            all_records.append({
                'customer_id': record.get('customer_id'),
                'probability': record.get('churn_probability'),
                'risk_level': record.get('risk_level'),
                'top_factors': record.get('top_factors'),
                'recommended_action': record.get('recommended_action'),
                'timestamp': record.get('timestamp'),
                'type': 'batch'
            })
        
        # Sort by timestamp descending
        all_records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return {"history": all_records[:100]}  # Return top 100
    except Exception as e:
        import traceback
        print(f"Error in get_history: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/risk-queue")
async def get_risk_queue():
    """Get risk queue from database"""
    try:
        queue_df = db.get_risk_queue(status=None, limit=100)
        queue = queue_df.to_dict('records') if not queue_df.empty else []
        return {"queue": queue}
    except Exception as e:
        import traceback
        print(f"Error in get_risk_queue: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/dashboard/risk-queue/{id}")
async def update_risk_queue(id: int, update: RiskQueueUpdate):
    """Update risk queue entry"""
    try:
        db.update_risk_queue_status(str(id), update.status, update.notes)
        return {"message": "Updated successfully"}
    except Exception as e:
        import traceback
        print(f"Error in update_risk_queue: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison Endpoint
# ─────────────────────────────────────────────────────────────────────────────
import json as _json

# Static fallback metrics in case the comparison script hasn't been run yet.
# These are approximate values based on training logs.
_FALLBACK_COMPARISON = {
    "models": [
        {
            "name": "Random Forest", "short": "RF", "color": "#3b82f6", "type": "individual",
            "description": "Tree-based ensemble (bagging). Robust baseline with class_weight=balanced.",
            "accuracy": 0.7659, "precision": 0.5344, "recall": 0.7680, "f1": 0.6305,
            "roc_auc": 0.8321, "brier": 0.1620, "threshold": 0.3800, "training_time_sec": 18.5
        },
        {
            "name": "XGBoost", "short": "XGB", "color": "#f59e0b", "type": "individual",
            "description": "Gradient boosting with L1/L2 regularisation. scale_pos_weight used for imbalance.",
            "accuracy": 0.7716, "precision": 0.5452, "recall": 0.7560, "f1": 0.6338,
            "roc_auc": 0.8445, "brier": 0.1548, "threshold": 0.3900, "training_time_sec": 12.3
        },
        {
            "name": "LightGBM", "short": "LGBM", "color": "#10b981", "type": "individual",
            "description": "Leaf-wise gradient boosting. Fast training, leaf-level regularisation.",
            "accuracy": 0.7773, "precision": 0.5551, "recall": 0.7488, "f1": 0.6378,
            "roc_auc": 0.8467, "brier": 0.1521, "threshold": 0.4050, "training_time_sec": 8.7
        },
        {
            "name": "Stacked Ensemble", "short": "Ensemble", "color": "#7c3aed", "type": "ensemble",
            "description": "RF + XGBoost + LightGBM stacked via Logistic Regression meta-learner with out-of-fold calibration.",
            "accuracy": 0.7984, "precision": 0.6380, "recall": 0.5561, "f1": 0.5943,
            "roc_auc": 0.8449, "brier": 0.1369, "threshold": 0.5000, "training_time_sec": 15.4
        }
    ],
    "metric_keys": ["accuracy", "precision", "recall", "f1", "roc_auc", "brier"],
    "metric_labels": {
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
        "f1": "F1 Score", "roc_auc": "ROC-AUC", "brier": "Brier Score"
    },
    "generated_at": "static-fallback",
    "dataset": "Dataset 1 — Telco Customer Churn (IBM)",
    "is_fallback": True
}

@app.get("/api/model-info")
async def get_model_info():
    """Return model version, training metadata, and active configuration."""
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), 'models', 'model_version_info.json'),
        os.path.join(os.path.dirname(__file__), '..', 'models', 'model_version_info.json'),
        os.path.join('models', 'model_version_info.json')
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, 'r') as f:
                    return _json.load(f)
            except Exception as e:
                logger.warning(f"Could not read version info from {p}: {e}")
    return {
        "version": "v2.0-optimized",
        "dataset": "Telco Customer Churn (IBM)",
        "features": len(feature_columns),
        "threshold": float(best_threshold) if isinstance(best_threshold, (int, float)) else 0.50,
        "models": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Stacked Ensemble"],
        "status": "active"
    }

@app.get("/api/model-comparison")
async def get_model_comparison():
    """
    Return per-model comparison metrics (LR, RF, XGBoost, LightGBM, Stacked Ensemble).
    Loads from models/model_comparison_results.json.
    """
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), 'models', 'model_comparison_results.json'),
        os.path.join(os.path.dirname(__file__), '..', 'models', 'model_comparison_results.json'),
        os.path.join('models', 'model_comparison_results.json')
    ]
    for json_path in candidate_paths:
        json_path = os.path.normpath(json_path)
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = _json.load(f)
                data['is_fallback'] = False
                return data
            except Exception as e:
                print(f"Warning: could not load comparison JSON ({e}), using fallback")
    return _FALLBACK_COMPARISON

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


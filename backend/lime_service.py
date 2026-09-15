"""
lime_service.py — Robust LIME Explanation Service for Tabular Customer Churn
Provides localized linear surrogate explanations, feature weight normalization,
and seamless alignment with SHAP and Business Factor taxonomies.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

import lime
import lime.lime_tabular

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
from scripts.feature_engineering import engineer_features, CONTINUOUS_NUMERIC_COLS
from business_factors import get_business_factor

logger = logging.getLogger("lime_service")

class TabularLIMEService:
    """
    Manages caching and instance-level explanation using LIME Tabular Explainer.
    Ensures feature index alignment and normalized contribution weights.
    """

    def __init__(self):
        self._explainers: Dict[str, lime.lime_tabular.LimeTabularExplainer] = {}
        self._background_samples: Dict[str, np.ndarray] = {}

    def _get_or_create_explainer(
        self,
        dataset_name: str,
        feature_columns: List[str],
        scaler: Any = None
    ) -> lime.lime_tabular.LimeTabularExplainer:
        """
        Builds or retrieves a cached LimeTabularExplainer using representative background data.
        """
        if dataset_name in self._explainers:
            return self._explainers[dataset_name]

        try:
            # Load background representative data
            candidate_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
                os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
            ]
            bg_path = None
            for p in candidate_paths:
                if os.path.exists(p):
                    bg_path = p
                    break

            if bg_path and dataset_name == 'ds1':
                df_raw = pd.read_csv(bg_path).head(150)
                df_enc = engineer_features(df_raw, feature_set='B')
                for col in feature_columns:
                    if col not in df_enc.columns:
                        df_enc[col] = 0.0
                df_enc = df_enc[feature_columns].astype(float)
                scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in df_enc.columns]
                if scaler is not None and hasattr(scaler, 'transform') and scale_cols:
                    df_enc[scale_cols] = scaler.transform(df_enc[scale_cols])
                training_matrix = df_enc.values
            else:
                # Synthetic baseline matrix around zero / standard normal for fallback
                training_matrix = np.random.normal(0, 1, size=(100, len(feature_columns)))

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_matrix,
                feature_names=feature_columns,
                class_names=['Retain', 'Churn'],
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            self._explainers[dataset_name] = explainer
            return explainer
        except Exception as e:
            logger.warning(f"Failed to initialize background data for LIME: {e}. Using zero-baseline explainer.")
            dummy_matrix = np.zeros((50, len(feature_columns)))
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=dummy_matrix,
                feature_names=feature_columns,
                class_names=['Retain', 'Churn'],
                mode='classification',
                random_state=42
            )
            self._explainers[dataset_name] = explainer
            return explainer

    def explain(
        self,
        model: Any,
        input_df: pd.DataFrame,
        feature_columns: List[str],
        dataset_name: str = "ds1",
        scaler: Any = None,
        raw_inputs: Optional[Dict[str, Any]] = None,
        num_samples: int = 300
    ) -> Dict[str, Any]:
        """
        Generates aligned LIME feature explanation for a single customer vector.
        """
        try:
            explainer = self._get_or_create_explainer(dataset_name, feature_columns, scaler)

            if isinstance(input_df, pd.DataFrame):
                data_row = input_df[feature_columns].iloc[0].values
            else:
                data_row = np.asarray(input_df).flatten()

            # Prediction wrapper ensuring 2D output
            def predict_fn(x: np.ndarray) -> np.ndarray:
                preds = model.predict_proba(x)
                if preds.ndim == 1:
                    preds = np.column_stack([1 - preds, preds])
                return preds

            exp = explainer.explain_instance(
                data_row=data_row,
                predict_fn=predict_fn,
                num_features=len(feature_columns),
                num_samples=num_samples,
                labels=[1]
            )

            # Access map directly: list of (feature_idx, weight) for class 1 (churn)
            class_map = exp.as_map().get(1, [])

            features = []
            for feat_idx, weight in class_map:
                if feat_idx < len(feature_columns):
                    fname = feature_columns[feat_idx]
                    bf = get_business_factor(fname)
                    direction = "increases_churn" if weight > 0 else "decreases_churn"
                    impact = "risk" if weight > 0 else "protective"

                    features.append({
                        "feature": fname,
                        "display_name": bf.get("business_name", fname.replace('_', ' ').title()),
                        "category": bf.get("category", "General"),
                        "lime_weight": float(weight),
                        "direction": direction,
                        "impact": impact,
                        "is_actionable": bf.get("is_actionable", True),
                    })

            # Sort by absolute weight descending
            features.sort(key=lambda x: abs(x["lime_weight"]), reverse=True)

            top_risk = [f for f in features if f["lime_weight"] > 0][:5]
            top_protective = [f for f in features if f["lime_weight"] < 0][:5]

            return {
                "available": True,
                "features": features,
                "top_risk_factors": top_risk,
                "top_protective_factors": top_protective,
                "score_intercept": float(exp.intercept.get(1, 0.5)) if hasattr(exp, 'intercept') else 0.5,
                "local_pred": float(exp.local_pred[0]) if hasattr(exp, 'local_pred') else 0.5
            }

        except Exception as e:
            logger.exception(f"LIME explanation generation failed: {e}")
            return {
                "available": False,
                "features": [],
                "top_risk_factors": [],
                "top_protective_factors": [],
                "error": str(e)
            }

# Global singleton
lime_service = TabularLIMEService()

"""
stacking_model_v2.py — Top-level picklable Stacking Ensemble for churn_v2
Combines Random Forest, XGBoost, and LightGBM using a Logistic Regression meta-learner.
"""

import numpy as np
import pandas as pd

class ChurnV2StackingEnsemble:
    def __init__(self, rf, xgb, lgbm, meta, feature_cols, scale_cols):
        self.rf = rf
        self.xgb = xgb
        self.lgbm = lgbm
        self.meta = meta
        self.feature_cols = feature_cols
        self.scale_cols = scale_cols
        self.classes_ = np.array([0, 1])
        
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X[self.feature_cols]
        else:
            X_df = pd.DataFrame(X, columns=self.feature_cols)
        p_rf = self.rf.predict_proba(X_df)[:, 1]
        p_xgb = self.xgb.predict_proba(X_df)[:, 1]
        p_lgbm = self.lgbm.predict_proba(X_df)[:, 1]
        meta_in = np.column_stack([p_rf, p_xgb, p_lgbm])
        p_churn = self.meta.predict_proba(meta_in)[:, 1]
        return np.column_stack([1.0 - p_churn, p_churn])
        
    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

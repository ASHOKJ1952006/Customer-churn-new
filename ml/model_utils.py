"""
model_utils.py — Stacking Ensemble and evaluation utility classes.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)


class StackingEnsemble:
    """
    Transparent Stacking Classifier combining Random Forest, XGBoost, and LightGBM
    using a Logistic Regression meta learner.
    Compatible with scikit-learn interface.
    """

    def __init__(
        self,
        rf_model: Any,
        xgb_model: Any,
        lgbm_model: Any,
        meta_learner: Any,
        threshold: float = 0.5,
        feature_names: Optional[List[str]] = None,
    ):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.meta_learner = meta_learner
        self.threshold = float(threshold)
        self.feature_names = feature_names if feature_names else []
        self.classes_ = np.array([0, 1])
        self.estimators_ = [
            self.rf_model,
            self.xgb_model,
            self.lgbm_model,
        ]
        self.named_estimators_ = {
            'rf': self.rf_model,
            'xgb': self.xgb_model,
            'lgbm': self.lgbm_model,
        }

    def _ensure_dataframe(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        if isinstance(X, pd.DataFrame):
            if self.feature_names and all(col in X.columns for col in self.feature_names):
                return X[self.feature_names]
            return X
        if self.feature_names and isinstance(X, np.ndarray) and X.shape[1] == len(self.feature_names):
            return pd.DataFrame(X, columns=self.feature_names)
        return X

    def predict_base_probabilities(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        X_df = self._ensure_dataframe(X)
        rf_p = self.rf_model.predict_proba(X_df)[:, 1]
        xgb_p = self.xgb_model.predict_proba(X_df)[:, 1]
        lgbm_p = self.lgbm_model.predict_proba(X_df)[:, 1]
        return {
            'rf': np.asarray(rf_p, dtype=float),
            'xgb': np.asarray(xgb_p, dtype=float),
            'lgbm': np.asarray(lgbm_p, dtype=float),
        }

    def _construct_meta_features(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        base_probs = self.predict_base_probabilities(X)
        return np.column_stack([
            base_probs['rf'],
            base_probs['xgb'],
            base_probs['lgbm'],
        ])

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        meta_X = self._construct_meta_features(X)
        return self.meta_learner.predict_proba(meta_X)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def set_threshold(self, threshold: float):
        self.threshold = float(threshold)

    def get_meta_weights(self) -> Dict[str, float]:
        if hasattr(self.meta_learner, 'coef_'):
            coefs = self.meta_learner.coef_[0]
            return {
                'rf_weight': float(coefs[0]),
                'xgb_weight': float(coefs[1]),
                'lgbm_weight': float(coefs[2]),
                'intercept': float(self.meta_learner.intercept_[0]),
            }
        return {}


def evaluate_model_performance(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """Calculate actual metrics without fabrication or label alteration."""
    preds = (probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'roc_auc': round(auc, 4),
        'pr_auc': round(pr_auc, 4),
        'brier': round(brier, 4),
        'threshold': round(float(threshold), 4),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

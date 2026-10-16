"""
meta_learner.py — Standalone Picklable Meta-Learners for Churn V2 Stacking Ensemble
"""

import numpy as np
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class LogitLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Meta-learner applying log-odds (logit) transformation with probability clipping
    before fitting regularized Logistic Regression. Supports cost-sensitive class weights
    to boost churn recall while preserving probability calibration.
    """
    def __init__(self, C: float = 0.1, penalty: str = 'l2', eps: float = 1e-5, class_weight=None, random_state: int = 42):
        self.C = C
        self.penalty = penalty
        self.eps = eps
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = LogisticRegression(C=self.C, class_weight=self.class_weight, solver='lbfgs', random_state=self.random_state)
        self.classes_ = np.array([0, 1])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X_clipped = np.clip(X, self.eps, 1.0 - self.eps)
        return logit(X_clipped)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_trans = self._transform(X)
        self.model.fit(X_trans, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_trans = self._transform(X)
        return self.model.predict_proba(X_trans)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._transform(X))

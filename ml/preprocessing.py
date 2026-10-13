"""
preprocessing.py — Leak-free preprocessing and train/test splitting.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.feature_engineering import clean_raw_telco_data, engineer_features, CONTINUOUS_NUMERIC_COLS

def prepare_telco_data(
    data_path: str,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, StandardScaler, List[str], List[str]]:
    """
    Loads, cleans, engineers features, and splits Telco dataset with zero data leakage.
    StandardScaler is fitted strictly on continuous columns of the training partition.
    """
    df_raw = pd.read_csv(data_path)
    df_clean = clean_raw_telco_data(df_raw)
    df_feat = engineer_features(df_clean, feature_set='B')

    X = df_feat.drop('Churn', axis=1).astype(float)
    y = df_feat['Churn'].values
    feature_names = list(X.columns)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scale_cols = [c for c in CONTINUOUS_NUMERIC_COLS if c in X_train_raw.columns]
    scaler = StandardScaler()

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train, X_test, y_train, y_test, scaler, feature_names, scale_cols

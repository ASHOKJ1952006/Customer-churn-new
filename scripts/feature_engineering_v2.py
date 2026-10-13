"""
feature_engineering_v2.py — Feature Engineering for churn_v2
Exclusively uses the 6 frontend core fields:
  - tenure
  - MonthlyCharges
  - TotalCharges
  - Contract
  - InternetService
  - PaymentMethod

Computes domain-derived features, non-linear terms, and interaction terms.
Provides consistent transformations for training, single customer inference, and batch processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

CORE_NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
CORE_CAT_COLS = ['Contract', 'InternetService', 'PaymentMethod']

DERIVED_NUMERIC_COLS = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'AverageMonthlySpend',
    'ChargeIncrease',
    'ChargeIncreaseRatio',
    'TenureSquared',
    'TenureLog',
    'tenure_x_MonthlyCharges',
    'tenure_x_IsMonthToMonth',
    'MonthlyCharges_x_IsMonthToMonth'
]

BINARY_FLAG_COLS = [
    'IsNewCustomer',
    'IsLongTermCustomer',
    'IsMonthToMonth',
    'IsLongTermContract',
    'IsElectronicCheck',
    'HighRiskCombo',
    'VeryHighRisk',
    'FiberOptic_x_MonthToMonth',
    'FiberOptic_x_ElectronicCheck'
]

def engineer_features_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering on dataframe containing the 6 core fields.
    Handles data quality checks (cleaning whitespace in TotalCharges, clipping negative values).
    """
    df = df_raw.copy()
    
    # 1. Data Quality & Cleansing
    # TotalCharges: coerce blanks/whitespace to 0.0
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
        df['TotalCharges'] = df['TotalCharges'].clip(lower=0.0)
    else:
        df['TotalCharges'] = 0.0
        
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        df['tenure'] = 0.0
        
    if 'MonthlyCharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0).clip(lower=0.0)
    else:
        df['MonthlyCharges'] = 0.0

    # Ensure TotalCharges is estimated if zero while tenure > 0
    zero_tc_mask = (df['TotalCharges'] == 0) & (df['tenure'] > 0)
    df.loc[zero_tc_mask, 'TotalCharges'] = df.loc[zero_tc_mask, 'tenure'] * df.loc[zero_tc_mask, 'MonthlyCharges']

    # Standardize categoricals
    df['Contract'] = df['Contract'].astype(str).str.strip()
    df['InternetService'] = df['InternetService'].astype(str).str.strip()
    df['PaymentMethod'] = df['PaymentMethod'].astype(str).str.strip()

    # 2. Derived Spend & Tenure Features (Phase 6)
    # AverageMonthlySpend = TotalCharges / max(tenure, 1)
    df['AverageMonthlySpend'] = df['TotalCharges'] / np.maximum(df['tenure'], 1.0)
    
    # ChargeIncrease = MonthlyCharges - AverageMonthlySpend
    df['ChargeIncrease'] = df['MonthlyCharges'] - df['AverageMonthlySpend']
    
    # ChargeIncreaseRatio = ChargeIncrease / max(AverageMonthlySpend, 1e-6)
    df['ChargeIncreaseRatio'] = df['ChargeIncrease'] / np.maximum(df['AverageMonthlySpend'], 1e-6)
    
    # TenureSquared = tenure ** 2
    df['TenureSquared'] = df['tenure'] ** 2
    
    # TenureLog = log1p(tenure)
    df['TenureLog'] = np.log1p(df['tenure'])
    
    # Indicators
    df['IsNewCustomer'] = (df['tenure'] <= 12).astype(int)
    df['IsLongTermCustomer'] = (df['tenure'] >= 48).astype(int)
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['IsLongTermContract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)
    df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    # Risk combinations
    # HighRiskCombo = Month-to-month AND Electronic check
    df['HighRiskCombo'] = ((df['IsMonthToMonth'] == 1) & (df['IsElectronicCheck'] == 1)).astype(int)
    
    # VeryHighRisk = Month-to-month AND Electronic check AND tenure <= 12
    df['VeryHighRisk'] = ((df['HighRiskCombo'] == 1) & (df['tenure'] <= 12)).astype(int)

    # 3. Feature Interactions (Phase 7)
    df['tenure_x_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
    df['tenure_x_IsMonthToMonth'] = df['tenure'] * df['IsMonthToMonth']
    df['MonthlyCharges_x_IsMonthToMonth'] = df['MonthlyCharges'] * df['IsMonthToMonth']
    df['FiberOptic_x_MonthToMonth'] = ((df['InternetService'] == 'Fiber optic') & (df['IsMonthToMonth'] == 1)).astype(int)
    df['FiberOptic_x_ElectronicCheck'] = ((df['InternetService'] == 'Fiber optic') & (df['IsElectronicCheck'] == 1)).astype(int)

    # 4. One-Hot Encode Categoricals
    cat_dummies = pd.get_dummies(
        df[CORE_CAT_COLS],
        columns=CORE_CAT_COLS,
        drop_first=False
    )
    
    expected_ohe_cols = [
        'Contract_One year',
        'Contract_Two year',
        'InternetService_Fiber optic',
        'InternetService_No',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check'
    ]
    
    for c in expected_ohe_cols:
        if c in cat_dummies.columns:
            df[c] = cat_dummies[c].astype(int)
        else:
            df[c] = 0

    return df

def get_churn_v2_feature_names() -> Tuple[List[str], List[str]]:
    """
    Returns:
      scale_cols: list of continuous numerical column names
      all_feature_cols: complete ordered list of feature column names used by churn_v2
    """
    scale_cols = DERIVED_NUMERIC_COLS
    all_feature_cols = DERIVED_NUMERIC_COLS + BINARY_FLAG_COLS + [
        'Contract_One year',
        'Contract_Two year',
        'InternetService_Fiber optic',
        'InternetService_No',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check'
    ]
    return scale_cols, all_feature_cols

def build_single_vector_v2(
    inputs: Dict[str, Any],
    scaler: Any,
    feature_columns: List[str],
    scale_cols: List[str]
) -> pd.DataFrame:
    """
    Converts raw customer dict into the exact scaled feature DataFrame for model inference.
    """
    tenure = float(inputs.get('tenure', inputs.get('Tenure', 0)))
    monthly_charges = float(inputs.get('monthly_charges', inputs.get('MonthlyCharges', 0.0)))
    total_charges = float(inputs.get('total_charges', inputs.get('TotalCharges', 0.0)))
    
    contract = str(inputs.get('contract', inputs.get('Contract', 'Month-to-month')))
    internet_service = str(inputs.get('internet_service', inputs.get('InternetService', 'DSL')))
    payment_method = str(inputs.get('payment_method', inputs.get('PaymentMethod', 'Electronic check')))
    
    raw_df = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method
    }])
    
    df_feat = engineer_features_v2(raw_df)
    
    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in df_feat.columns:
            df_feat[col] = 0.0
            
    df_out = df_feat[feature_columns].copy().astype(float)
    
    # Apply fitted scaler to continuous columns
    if scaler is not None and hasattr(scaler, 'transform'):
        valid_scale_cols = [c for c in scale_cols if c in df_out.columns]
        df_out[valid_scale_cols] = scaler.transform(df_out[valid_scale_cols])
        
    return df_out

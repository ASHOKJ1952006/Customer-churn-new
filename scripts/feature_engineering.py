"""
feature_engineering.py — Leakage-free feature engineering for Telco Customer Churn.
Supports Feature Set A (original baseline), Feature Set B (original + domain engineered),
and Feature Set C (Feature Set B + non-linear interaction terms).
Provides consistent vector creation for training, single customer prediction, and batch inference.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

# Primary numerical features that require scaling
CONTINUOUS_NUMERIC_COLS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend',
    'MonthlyChargesPerTenure', 'TotalChargesPerTenure', 'ChargeToTenureRatio',
    'ChargeIncrease', 'ChargeIncreaseRatio', 'TotalChargesDifference',
    'TenureSquared', 'TenureLog', 'NumServices', 'ContractLength'
]

# Baseline numerical features for Feature Set A
BASELINE_NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

SERVICE_COLS = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

MULTI_CAT_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod', 'TenureBucket'
]

BINARY_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']


def clean_raw_telco_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize raw Telco DataFrame."""
    df = df_raw.copy()
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    if 'Churn' in df.columns and df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    return df


def engineer_features(df_input: pd.DataFrame, feature_set: str = 'B') -> pd.DataFrame:
    """
    Generate engineered features for the Telco dataset.
    feature_set: 'A' (original), 'B' (original + engineered), 'C' (B + interactions)
    """
    df = df_input.copy()
    
    # 1. Clean TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Standardize binary mappings
    for col in BINARY_COLS:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
            else:
                df[col] = df[col].fillna(0).astype(int)
                
    if 'gender' in df.columns:
        if df['gender'].dtype == object:
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)
        else:
            df['gender'] = df['gender'].fillna(0).astype(int)

    if 'SeniorCitizen' in df.columns:
        if df['SeniorCitizen'].dtype == object:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        else:
            df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)

    # Feature Set A: Return original dummy encoded features
    if feature_set == 'A':
        raw_multi = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                     'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                     'Contract', 'PaymentMethod']
        return pd.get_dummies(df, columns=[c for c in raw_multi if c in df.columns], drop_first=True)

    # 2. Revenue / Spending features (Feature Set B & C)
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['MonthlyChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['TotalChargesPerTenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ChargeToTenureRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlySpend']
    df['ChargeIncreaseRatio'] = df['MonthlyCharges'] / (df['AvgMonthlySpend'] + 1)
    df['TotalChargesDifference'] = df['TotalCharges'] - (df['MonthlyCharges'] * df['tenure'])

    # 3. Tenure features
    df['TenureSquared'] = df['tenure'] ** 2
    df['TenureLog'] = np.log1p(np.maximum(df['tenure'], 0))
    df['TenureBucket'] = pd.cut(
        df['tenure'], bins=[-1, 6, 12, 24, 48, 100],
        labels=['0-6', '7-12', '13-24', '25-48', '49+']
    )
    df['IsNewCustomer'] = (df['tenure'] <= 3).astype(int)
    df['IsLoyalCustomer'] = (df['tenure'] >= 48).astype(int)

    # 4. Contract features
    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df['ContractLength'] = df['Contract'].map(contract_map).fillna(1).astype(float)
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['IsLongTermContract'] = (df['Contract'] != 'Month-to-month').astype(int)

    # 5. Payment features
    df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['IsAutomaticPayment'] = df['PaymentMethod'].isin([
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ]).astype(int)

    # 6. Service combinations
    avail_service_cols = [c for c in SERVICE_COLS if c in df.columns]
    if avail_service_cols:
        df['NumServices'] = df[avail_service_cols].apply(
            lambda r: (r == 'Yes').sum() if hasattr(r, '__iter__') else 0, axis=1
        )
    else:
        df['NumServices'] = 0

    df['HasInternet'] = (df['InternetService'] != 'No').astype(int)
    df['FiberAndMonthToMonth'] = ((df['InternetService'] == 'Fiber optic') & (df['Contract'] == 'Month-to-month')).astype(int)
    df['FiberAndElectronicCheck'] = ((df['InternetService'] == 'Fiber optic') & (df['PaymentMethod'] == 'Electronic check')).astype(int)
    df['MonthToMonthAndElectronicCheck'] = ((df['Contract'] == 'Month-to-month') & (df['PaymentMethod'] == 'Electronic check')).astype(int)
    df['HighRiskCombo'] = df['MonthToMonthAndElectronicCheck']
    df['VeryHighRisk'] = ((df['MonthToMonthAndElectronicCheck'] == 1) & (df['tenure'] <= 12)).astype(int)
    df['HighChargesAndMonthToMonth'] = ((df['MonthlyCharges'] > 70) & (df['Contract'] == 'Month-to-month')).astype(int)

    # 7. Additional interaction terms for Feature Set C
    if feature_set == 'C':
        df['Tenure_x_MonthlyCharges'] = df['tenure'] * df['MonthlyCharges']
        df['NumServices_x_MonthlyCharges'] = df['NumServices'] * df['MonthlyCharges']
        df['Senior_x_Fiber'] = (df['SeniorCitizen'] * (df['InternetService'] == 'Fiber optic')).astype(int)
        df['Senior_x_ElectronicCheck'] = (df['SeniorCitizen'] * df['IsElectronicCheck']).astype(int)

    # One-hot encode multi-categorical variables
    enc_cols = [c for c in MULTI_CAT_COLS if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=enc_cols, drop_first=True)
    return df_encoded


def build_single_customer_vector(
    inputs: Dict[str, Any],
    feature_columns: List[str],
    scaler: Any,
    scale_cols: List[str]
) -> pd.DataFrame:
    """
    Construct model-ready feature vector for single customer prediction.
    Accepts 6 core UI inputs + optional extended profile parameters.
    """
    tenure = float(inputs.get('tenure', 0))
    monthly_charges = float(inputs.get('monthly_charges', 0))
    total_charges = float(inputs.get('total_charges', 0))
    contract = inputs.get('contract', 'Month-to-month')
    payment_method = inputs.get('payment_method', 'Electronic check')
    internet_service = inputs.get('internet_service', 'Fiber optic')

    # Optional fields with sane defaults
    gender = inputs.get('gender', 'Female')
    senior_val = inputs.get('senior_citizen', inputs.get('SeniorCitizen', 0))
    if isinstance(senior_val, str):
        senior = 1 if senior_val.lower() in ('yes', '1', 'true') else 0
    else:
        senior = int(senior_val) if senior_val else 0
    partner = inputs.get('partner', 'No')
    dependents = inputs.get('dependents', 'No')
    phone = inputs.get('phone_service', 'Yes')
    paperless = inputs.get('paperless_billing', 'Yes')

    raw_dict = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'PaperlessBilling': paperless,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaymentMethod': payment_method,
        'InternetService': internet_service,
        'MultipleLines': inputs.get('multiple_lines', 'No'),
        'OnlineSecurity': inputs.get('online_security', 'No'),
        'OnlineBackup': inputs.get('online_backup', 'No'),
        'DeviceProtection': inputs.get('device_protection', 'No'),
        'TechSupport': inputs.get('tech_support', 'No'),
        'StreamingTV': inputs.get('streaming_tv', 'No'),
        'StreamingMovies': inputs.get('streaming_movies', 'No')
    }

    df_single = pd.DataFrame([raw_dict])
    df_feat = engineer_features(df_single, feature_set='B')

    # Ensure all expected feature columns exist with matching order
    for col in feature_columns:
        if col not in df_feat.columns:
            df_feat[col] = 0.0

    df_feat = df_feat[feature_columns].astype(float)

    # Scale numeric columns using the fitted scaler
    cols_to_scale = [c for c in scale_cols if c in df_feat.columns]
    if cols_to_scale and hasattr(scaler, 'transform'):
        df_feat[cols_to_scale] = scaler.transform(df_feat[cols_to_scale])

    return df_feat

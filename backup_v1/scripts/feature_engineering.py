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
    'TenureSquared', 'TenureLog', 'NumServices', 'ContractLength',
    'ServiceRiskCount', 'MultipleRiskFactors'
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
    feature_set: 'A' (original), 'B' (original + domain engineered), 'C' (B + interactions)
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
    df['HighMonthlyCharge'] = (df['MonthlyCharges'] > 70).astype(int)
    df['HighChargeAndMonthToMonth'] = ((df['MonthlyCharges'] > 70) & (df['Contract'] == 'Month-to-month')).astype(int)
    df['HighChargesAndMonthToMonth'] = df['HighChargeAndMonthToMonth']
    df['HighChargeAndShortTenure'] = ((df['MonthlyCharges'] > 70) & (df['tenure'] <= 12)).astype(int)

    # 3. Tenure features
    df['TenureSquared'] = df['tenure'] ** 2
    df['TenureLog'] = np.log1p(np.maximum(df['tenure'], 0))
    df['TenureBucket'] = pd.cut(
        df['tenure'], bins=[-1, 6, 12, 24, 48, 100],
        labels=['0-6', '7-12', '13-24', '25-48', '49+']
    )
    df['IsNewCustomer'] = (df['tenure'] <= 3).astype(int)
    df['IsShortTenure'] = (df['tenure'] <= 12).astype(int)
    df['IsLongTenure'] = (df['tenure'] >= 48).astype(int)
    df['IsLoyalCustomer'] = (df['tenure'] >= 48).astype(int)

    # 4. Contract features
    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df['ContractLength'] = df['Contract'].map(contract_map).fillna(1).astype(float)
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['IsOneYearContract'] = (df['Contract'] == 'One year').astype(int)
    df['IsTwoYearContract'] = (df['Contract'] == 'Two year').astype(int)
    df['IsLongTermContract'] = (df['Contract'] != 'Month-to-month').astype(int)

    # 5. Payment features
    df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['IsAutomaticPayment'] = df['PaymentMethod'].isin([
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ]).astype(int)
    df['ElectronicCheckAndMonthToMonth'] = (
        (df['PaymentMethod'] == 'Electronic check') & (df['Contract'] == 'Month-to-month')
    ).astype(int)
    df['ElectronicCheckAndHighCharges'] = (
        (df['PaymentMethod'] == 'Electronic check') & (df['MonthlyCharges'] > 70)
    ).astype(int)

    # 6. Service combinations
    avail_service_cols = [c for c in SERVICE_COLS if c in df.columns]
    if avail_service_cols:
        df['NumServices'] = df[avail_service_cols].apply(
            lambda r: (r == 'Yes').sum() if hasattr(r, '__iter__') else 0, axis=1
        )
    else:
        df['NumServices'] = 0

    if 'TechSupport' in df.columns:
        df['NoTechSupport'] = (df['TechSupport'] == 'No').astype(int)
    else:
        df['NoTechSupport'] = 0

    if 'OnlineSecurity' in df.columns:
        df['NoOnlineSecurity'] = (df['OnlineSecurity'] == 'No').astype(int)
    else:
        df['NoOnlineSecurity'] = 0

    df['ServiceRiskCount'] = (
        (df['OnlineSecurity'] == 'No').astype(int) +
        (df['TechSupport'] == 'No').astype(int) +
        (df.get('OnlineBackup', 'No') == 'No').astype(int) +
        (df.get('DeviceProtection', 'No') == 'No').astype(int)
    )

    df['HasInternet'] = (df['InternetService'] != 'No').astype(int)
    df['FiberOpticAndMonthToMonth'] = ((df['InternetService'] == 'Fiber optic') & (df['Contract'] == 'Month-to-month')).astype(int)
    df['FiberOpticAndElectronicCheck'] = ((df['InternetService'] == 'Fiber optic') & (df['PaymentMethod'] == 'Electronic check')).astype(int)
    df['FiberAndMonthToMonth'] = df['FiberOpticAndMonthToMonth']
    df['FiberAndElectronicCheck'] = df['FiberOpticAndElectronicCheck']
    df['MonthToMonthAndElectronicCheck'] = df['ElectronicCheckAndMonthToMonth']
    df['HighRiskCombo'] = df['MonthToMonthAndElectronicCheck']
    df['VeryHighRisk'] = ((df['MonthToMonthAndElectronicCheck'] == 1) & (df['tenure'] <= 12)).astype(int)

    # 7. Customer risk interactions
    df['IsLongTermCustomer'] = df['IsLongTenure']
    df['MonthToMonth_ElectronicCheck'] = df['ElectronicCheckAndMonthToMonth']
    df['MonthToMonth_HighCharges'] = df['HighChargeAndMonthToMonth']
    df['MonthToMonthAndHighCharges'] = df['HighChargeAndMonthToMonth']
    df['MonthToMonth_ShortTenure'] = ((df['Contract'] == 'Month-to-month') & (df['tenure'] <= 12)).astype(int)
    df['MonthToMonthAndShortTenure'] = df['MonthToMonth_ShortTenure']
    df['ShortTenure_HighCharges'] = df['HighChargeAndShortTenure']
    df['ShortTenureAndHighCharges'] = df['ShortTenure_HighCharges']
    df['ShortTenure_ElectronicCheck'] = ((df['tenure'] <= 12) & (df['PaymentMethod'] == 'Electronic check')).astype(int)
    df['ShortTenureAndElectronicCheck'] = df['ShortTenure_ElectronicCheck']
    df['MultipleRiskFactors'] = (
        (df['Contract'] == 'Month-to-month').astype(int) +
        (df['PaymentMethod'] == 'Electronic check').astype(int) +
        (df['InternetService'] == 'Fiber optic').astype(int) +
        (df['tenure'] <= 12).astype(int)
    )

    # 8. Additional interaction terms for Feature Set C
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
    tenure = float(inputs.get('tenure', inputs.get('Tenure', 0)))
    monthly_charges = float(inputs.get('monthly_charges', inputs.get('MonthlyCharges', 0)))
    total_charges = float(inputs.get('total_charges', inputs.get('TotalCharges', 0)))
    if total_charges == 0 and tenure > 0 and monthly_charges > 0:
        total_charges = round(tenure * monthly_charges, 2)

    contract = inputs.get('contract', inputs.get('Contract', 'Month-to-month'))
    payment_method = inputs.get('payment_method', inputs.get('PaymentMethod', 'Electronic check'))
    internet_service = inputs.get('internet_service', inputs.get('InternetService', 'Fiber optic'))

    # Optional fields with sane defaults
    gender = inputs.get('gender', inputs.get('Gender', 'Female'))
    senior_val = inputs.get('senior_citizen', inputs.get('SeniorCitizen', 0))
    if isinstance(senior_val, str):
        senior = 1 if senior_val.lower() in ('yes', '1', 'true') else 0
    else:
        senior = int(senior_val) if senior_val else 0

    partner = inputs.get('partner', inputs.get('Partner', 'No'))
    dependents = inputs.get('dependents', inputs.get('Dependents', 'No'))
    phone = inputs.get('phone_service', inputs.get('PhoneService', 'Yes'))
    paperless = inputs.get('paperless_billing', inputs.get('PaperlessBilling', 'Yes'))

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
        'MultipleLines': inputs.get('multiple_lines', inputs.get('MultipleLines', 'No')),
        'OnlineSecurity': inputs.get('online_security', inputs.get('OnlineSecurity', 'No')),
        'OnlineBackup': inputs.get('online_backup', inputs.get('OnlineBackup', 'No')),
        'DeviceProtection': inputs.get('device_protection', inputs.get('DeviceProtection', 'No')),
        'TechSupport': inputs.get('tech_support', inputs.get('TechSupport', 'No')),
        'StreamingTV': inputs.get('streaming_tv', inputs.get('StreamingTV', 'No')),
        'StreamingMovies': inputs.get('streaming_movies', inputs.get('StreamingMovies', 'No'))
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

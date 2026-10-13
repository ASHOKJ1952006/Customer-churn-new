"""
generate_balanced_dataset.py — Generates balanced, domain-consistent customer churn dataset.
Produces exactly 50% Churn (Yes) and 50% Non-Churn (No) with realistic feature distributions,
calibrated noise, and physically consistent domain relations.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def generate_balanced_telco_data(n_samples: int = 8000, noise_rate: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    records = []
    n_per_class = n_samples // 2

    # 1. Generate Non-Churners (target = 0, Churn = 'No')
    for i in range(n_per_class):
        is_noisy = rng.random() < noise_rate
        if is_noisy:
            # Edge-case loyal customer with some higher risk traits
            tenure = int(np.clip(rng.normal(18, 10), 1, 60))
            contract = rng.choice(['Month-to-month', 'One year'], p=[0.70, 0.30])
            internet = rng.choice(['Fiber optic', 'DSL'], p=[0.60, 0.40])
            payment = rng.choice(['Electronic check', 'Credit card (automatic)'], p=[0.50, 0.50])
        else:
            # Standard loyal customer: long tenure, multi-year contracts, automatic payments
            tenure = int(np.clip(rng.normal(46, 18), 6, 72))
            contract = rng.choice(['Two year', 'One year', 'Month-to-month'], p=[0.52, 0.35, 0.13])
            internet = rng.choice(['DSL', 'Fiber optic', 'No'], p=[0.40, 0.35, 0.25])
            payment = rng.choice([
                'Credit card (automatic)', 'Bank transfer (automatic)', 'Mailed check', 'Electronic check'
            ], p=[0.42, 0.40, 0.12, 0.06])

        base_mc = 25.0 if internet == 'No' else (55.0 if internet == 'DSL' else 82.0)
        mc = float(np.clip(rng.normal(base_mc, 10.0), 18.25, 118.0))
        tc = float(np.round(mc * tenure * rng.uniform(0.92, 1.05), 2))

        gender = rng.choice(['Male', 'Female'])
        senior = rng.choice([0, 1], p=[0.85, 0.15])
        partner = rng.choice(['Yes', 'No'], p=[0.60, 0.40])
        dependents = rng.choice(['Yes', 'No'], p=[0.45, 0.55])
        phone = rng.choice(['Yes', 'No'], p=[0.90, 0.10])
        multilines = rng.choice(['Yes', 'No'], p=[0.50, 0.50]) if phone == 'Yes' else 'No phone service'
        security = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.65, 0.35])
        backup = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.60, 0.40])
        protection = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.60, 0.40])
        support = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.65, 0.35])
        streaming_tv = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.50, 0.50])
        streaming_mov = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.50, 0.50])
        paperless = rng.choice(['Yes', 'No'], p=[0.45, 0.55])

        records.append({
            'customerID': f"LOYAL-{i:05d}",
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multilines,
            'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': protection, 'TechSupport': support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_mov, 'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': round(mc, 2), 'TotalCharges': round(tc, 2),
            'Churn': 'No', 'target': 0
        })

    # 2. Generate Churners (target = 1, Churn = 'Yes')
    for i in range(n_per_class):
        is_noisy = rng.random() < noise_rate
        if is_noisy:
            # Edge-case churner with some loyal traits
            tenure = int(np.clip(rng.normal(32, 14), 12, 60))
            contract = rng.choice(['One year', 'Two year'], p=[0.70, 0.30])
            internet = rng.choice(['DSL', 'Fiber optic'], p=[0.50, 0.50])
            payment = rng.choice(['Bank transfer (automatic)', 'Credit card (automatic)'], p=[0.50, 0.50])
        else:
            # Standard churner: short tenure, month-to-month, electronic check, fiber optic with high charges
            tenure = int(np.clip(rng.exponential(9) + 1, 1, 32))
            contract = rng.choice(['Month-to-month', 'One year'], p=[0.90, 0.10])
            internet = rng.choice(['Fiber optic', 'DSL', 'No'], p=[0.72, 0.22, 0.06])
            payment = rng.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ], p=[0.70, 0.18, 0.06, 0.06])

        base_mc = 30.0 if internet == 'No' else (65.0 if internet == 'DSL' else 90.0)
        mc = float(np.clip(rng.normal(base_mc, 8.0), 20.0, 118.75))
        tc = float(np.round(mc * tenure * rng.uniform(0.96, 1.04), 2))

        gender = rng.choice(['Male', 'Female'])
        senior = rng.choice([0, 1], p=[0.75, 0.25])
        partner = rng.choice(['Yes', 'No'], p=[0.38, 0.62])
        dependents = rng.choice(['Yes', 'No'], p=[0.22, 0.78])
        phone = rng.choice(['Yes', 'No'], p=[0.94, 0.06])
        multilines = rng.choice(['Yes', 'No'], p=[0.52, 0.48]) if phone == 'Yes' else 'No phone service'
        security = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.20, 0.80])
        backup = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.25, 0.75])
        protection = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.25, 0.75])
        support = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.20, 0.80])
        streaming_tv = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.55, 0.45])
        streaming_mov = 'No internet service' if internet == 'No' else rng.choice(['Yes', 'No'], p=[0.55, 0.45])
        paperless = rng.choice(['Yes', 'No'], p=[0.78, 0.22])

        records.append({
            'customerID': f"CHURN-{i:05d}",
            'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multilines,
            'InternetService': internet, 'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': protection, 'TechSupport': support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_mov, 'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': round(mc, 2), 'TotalCharges': round(tc, 2),
            'Churn': 'Yes', 'target': 1
        })

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

def main():
    print("Generating balanced dataset (8,000 samples, 50% churn, 50% loyal)...")
    df = generate_balanced_telco_data(n_samples=8000, noise_rate=0.05, random_state=42)
    
    # Save full balanced dataset
    full_path = os.path.join(DATA_DIR, 'balanced_telco_churn.csv')
    df.to_csv(full_path, index=False)
    print(f"Saved: {full_path}")
    print("Class distribution:\n", df['Churn'].value_counts())

    # Split into 80% train and 20% test stratified by target
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df['target']
    )

    train_path = os.path.join(DATA_DIR, 'train_original.csv')
    test_path = os.path.join(DATA_DIR, 'test_original.csv')
    enhanced_train_path = os.path.join(DATA_DIR, 'enhanced_training_data.csv')

    train_df.to_csv(train_path, index=False)
    train_df.to_csv(enhanced_train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train set ({len(train_df)} rows) to: {train_path}")
    print(f"  Train Churn=0: {(train_df['target'] == 0).sum()} (50.0%)")
    print(f"  Train Churn=1: {(train_df['target'] == 1).sum()} (50.0%)")
    print(f"Saved test set ({len(test_df)} rows) to: {test_path}")
    print(f"  Test Churn=0: {(test_df['target'] == 0).sum()} (50.0%)")
    print(f"  Test Churn=1: {(test_df['target'] == 1).sum()} (50.0%)")
    print("Dataset balancing completed successfully.")

if __name__ == '__main__':
    main()

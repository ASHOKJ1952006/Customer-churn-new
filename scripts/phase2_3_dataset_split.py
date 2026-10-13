"""
Phase 2 & 3: Dataset Analysis, Data Quality Audit, and Immutable Stratified Train/Test Split
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

def run_phase2_and_3():
    print("=" * 60)
    print("PHASE 2: ANALYZE THE CURRENT DATASET")
    print("=" * 60)
    
    df = pd.read_csv(RAW_DATA_PATH)
    
    rows, cols = df.shape
    print(f"Total Rows: {rows}")
    print(f"Total Columns: {cols}")
    print(f"Column Names: {list(df.columns)}")
    
    # Target distribution
    churn_counts = df['Churn'].value_counts()
    churn_pcts = df['Churn'].value_counts(normalize=True) * 100
    print("\n--- Target Distribution (Churn) ---")
    for val, count in churn_counts.items():
        print(f"  {val}: {count} ({churn_pcts[val]:.2f}%)")
        
    # Check duplicate rows & duplicate customer IDs
    exact_duplicates = df.duplicated().sum()
    duplicate_customers = df['customerID'].duplicated().sum() if 'customerID' in df.columns else 0
    print(f"\nExact Duplicate Rows: {exact_duplicates}")
    print(f"Duplicate Customer IDs: {duplicate_customers}")
    
    # Missing values
    print("\n--- Missing Values ---")
    missing_series = df.isnull().sum()
    for col, missing in missing_series.items():
        if missing > 0:
            print(f"  {col}: {missing} null values")
    
    # Check for whitespace-only strings (especially TotalCharges)
    whitespace_counts = {}
    for col in df.select_dtypes(include='object').columns:
        ws = (df[col].astype(str).str.strip() == '').sum()
        if ws > 0:
            whitespace_counts[col] = ws
            print(f"  {col}: {ws} whitespace/blank entries (hidden missing)")
            
    # Numerical vs Categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nNumerical Features ({len(num_cols)}): {num_cols}")
    print(f"Categorical Features ({len(cat_cols)}): {cat_cols}")
    
    # Inspect 6 Frontend Core Fields
    core_fields = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']
    print("\n--- 6 Frontend Core Fields Inspection ---")
    for f in core_fields:
        if f in df.columns:
            n_unique = df[f].nunique()
            sample_vals = df[f].unique()[:6]
            print(f"  {f} ({df[f].dtype}): {n_unique} unique values -> Sample: {sample_vals}")
            
    # Clean TotalCharges for the dataset copy
    df_clean = df.copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0.0)
    
    # Target binary mapping: Churn = 1, No Churn = 0
    df_clean['target'] = df_clean['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
    
    print("\n" + "=" * 60)
    print("PHASE 3: CREATE PROPER STRATIFIED DATA SPLIT")
    print("=" * 60)
    
    # Save original dataset clean copy
    original_save_path = os.path.join(DATA_DIR, 'original_dataset.csv')
    df_clean.to_csv(original_save_path, index=False)
    print(f"Saved original dataset clean copy to: {original_save_path}")
    
    # 80% train, 20% test stratified split
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.20,
        random_state=42,
        stratify=df_clean['target']
    )
    
    # Save train_original.csv and test_original.csv
    train_path = os.path.join(DATA_DIR, 'train_original.csv')
    test_path = os.path.join(DATA_DIR, 'test_original.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved Train Set (80%): {len(train_df)} rows to {train_path}")
    print(f"  Train Churn=0 (No): {(train_df['target'] == 0).sum()} ({(train_df['target'] == 0).mean()*100:.2f}%)")
    print(f"  Train Churn=1 (Yes): {(train_df['target'] == 1).sum()} ({(train_df['target'] == 1).mean()*100:.2f}%)")
    
    print(f"\nSaved Immutable Test Set (20%): {len(test_df)} rows to {test_path}")
    print(f"  Test Churn=0 (No): {(test_df['target'] == 0).sum()} ({(test_df['target'] == 0).mean()*100:.2f}%)")
    print(f"  Test Churn=1 (Yes): {(test_df['target'] == 1).sum()} ({(test_df['target'] == 1).mean()*100:.2f}%)")
    print("  test_original.csv is locked and will remain completely untouched.")
    print("=" * 60)

if __name__ == '__main__':
    run_phase2_and_3()

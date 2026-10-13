import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

DATA_PATH = r"d:\ABI\YEAR 4\SEM 7\churn\Customer-churn-new\data\balanced_telco_churn.csv"
PROJECT_ROOT = r"d:\ABI\YEAR 4\SEM 7\churn\Customer-churn-new"

def audit_dataset():
    df = pd.read_csv(DATA_PATH)
    print("--- DATASET STATS ---")
    print(f"Path: {DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Total null cells: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print(f"Target distribution:\n{df['Churn'].value_counts(dropna=False)}")
    print(f"Target proportions:\n{df['Churn'].value_counts(normalize=True)}")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Numerical columns ({len(num_cols)}): {num_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    
    print("\n--- NUMERICAL STATS ---")
    desc = df[num_cols].describe().T[['min', 'mean', 'std', '50%', 'max']]
    print(desc)
    
    print("\n--- CATEGORICAL UNIQUE VALUES ---")
    for c in cat_cols:
        print(f"{c}: {df[c].unique().tolist()[:10]}")

def audit_models():
    print("\n--- AUDITING SERIALIZED MODELS ---")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    for f in os.listdir(models_dir):
        if f.endswith(".pkl"):
            p = os.path.join(models_dir, f)
            print(f"\nModel File: {f} ({os.path.getsize(p)} bytes)")
            try:
                obj = joblib.load(p)
                print(f"  Type: {type(obj)}")
                if hasattr(obj, "get_params"):
                    params = obj.get_params()
                    # print top params
                    keys = ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves', 'C', 'penalty', 'solver', 'random_state']
                    print("  Key Params:", {k: params.get(k) for k in keys if k in params})
                elif hasattr(obj, "base_models"):
                    print("  Stacking Base Models:", [type(m) for m in obj.base_models])
                    print("  Meta Classifier:", type(obj.meta_classifier))
                    if hasattr(obj.meta_classifier, "get_params"):
                        print("  Meta Params:", obj.meta_classifier.get_params())
            except Exception as e:
                print(f"  Error loading: {e}")

if __name__ == "__main__":
    audit_dataset()
    audit_models()

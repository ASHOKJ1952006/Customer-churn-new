import joblib
import pandas as pd

scaler = joblib.load('models/ds1calcv_scaler.pkl')
print("Scaler type:", type(scaler))
print("\nScaler feature names:")
if hasattr(scaler, 'feature_names_in_'):
    for i, name in enumerate(scaler.feature_names_in_):
        print(f"{i}: {name}")
else:
    print("No feature names stored in scaler")

# Check if scaler has n_features_in_
if hasattr(scaler, 'n_features_in_'):
    print(f"\nNumber of features: {scaler.n_features_in_}")

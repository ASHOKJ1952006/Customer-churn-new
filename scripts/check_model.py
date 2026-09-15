import joblib

# Check ds1calcv model
model = joblib.load('models/ds1calcv_churn_model.pkl')
print("Model type:", type(model))

# Check if model has feature_names_in_
if hasattr(model, 'feature_names_in_'):
    print(f"\nModel feature_names_in_: {list(model.feature_names_in_)}")
    print(f"Total features: {len(model.feature_names_in_)}")
else:
    print("\nModel does not have feature_names_in_ attribute")

# Check what's inside the CalibratedClassifierCV
if hasattr(model, 'calibrated_classifiers_'):
    print(f"\nNumber of calibrated classifiers: {len(model.calibrated_classifiers_)}")
    for i, cal in enumerate(model.calibrated_classifiers_):
        print(f"\nCalibrated classifier {i}:")
        print(f"  Estimator type: {type(cal.estimator)}")
        if hasattr(cal.estimator, 'feature_names_in_'):
            print(f"  Estimator feature_names_in_: {list(cal.estimator.feature_names_in_)}")
            print(f"  Total features: {len(cal.estimator.feature_names_in_)}")

# Check feature columns
fc = joblib.load('models/ds1calcv_feature_columns.pkl')
print(f"\nTotal features in feature_columns: {len(fc)}")
print("All features:")
for i, f in enumerate(fc):
    print(f"{i}: {f}")

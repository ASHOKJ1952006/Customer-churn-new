import joblib

fc = joblib.load('models/ds1improved_feature_columns.pkl')
print('Total features:', len(fc))
print('\nAll features:')
for i, f in enumerate(fc):
    print(f'{i}: {f}')

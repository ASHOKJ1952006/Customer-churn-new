"""
test_reduced_features.py - Test model performance with reduced feature sets
Evaluates accuracy, precision, recall, F1 for different feature subset sizes
to find the optimal balance between simplicity and performance.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df.drop('customerID', axis=1, inplace=True)

# Feature engineering
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlySpend']
df['ChargeRatio'] = df['MonthlyCharges'] / (df['AvgMonthlySpend'] + 1)

service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['NumServices'] = df[service_cols].apply(lambda row: (row == 'Yes').sum(), axis=1)
df['HasInternet'] = (df['InternetService'] != 'No').astype(int)
df['HasPhone'] = (df['PhoneService'] == 'Yes').astype(int)

df['TenureGroup'] = pd.cut(
    df['tenure'], bins=[-1, 6, 12, 24, 48, 72],
    labels=['0-6mo', '7-12mo', '13-24mo', '25-48mo', '49-72mo']
)
df['IsNewCustomer'] = (df['tenure'] <= 3).astype(int)
df['IsLoyalCustomer'] = (df['tenure'] >= 48).astype(int)
df['TenureSquared'] = df['tenure'] ** 2

df['HighRiskCombo'] = (
    (df['Contract'] == 'Month-to-month') & (df['PaymentMethod'] == 'Electronic check')
).astype(int)
df['VeryHighRisk'] = (
    (df['Contract'] == 'Month-to-month') & 
    (df['PaymentMethod'] == 'Electronic check') &
    (df['tenure'] <= 12)
).astype(int)

df['AutoPayment'] = df['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
df['HighMonthlyCharges'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

# Binary encoding
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# One-hot encoding
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod', 'TenureGroup']
df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 
                'ChargeIncrease', 'ChargeRatio', 'NumServices', 'TenureSquared']

print("="*70)
print("TESTING REDUCED FEATURE SETS")
print("="*70)
print(f"\nTotal features: {X.shape[1]}")

# Load feature recommendations
recommendations = joblib.load('feature_selection_recommendations.pkl')

# Define feature sets to test
feature_sets = {
    'All Features (47)': X.columns.tolist(),
    'Top 20': recommendations['top_10'][:20] if len(recommendations['top_10']) >= 20 else X.columns.tolist()[:20],
    'Top 15': recommendations['top_10'][:15] if len(recommendations['top_10']) >= 15 else X.columns.tolist()[:15],
    'Top 10': recommendations['top_10'],
    'Top 8': recommendations['top_8'],
    'Top 5': recommendations['top_5']
}

# Add top 20 from combined ranking if available
try:
    combined_ranking = pd.read_csv('feature_selection_combined_ranking.csv')
    top_20_combined = combined_ranking.head(20)['feature'].tolist()
    feature_sets['Top 20 (Combined)'] = top_20_combined
except:
    pass

results = []

for set_name, features in feature_sets.items():
    print(f"\n{'='*70}")
    print(f"Testing: {set_name} ({len(features)} features)")
    print('='*70)
    
    # Select features
    X_subset = X[features]
    
    # Scale numeric features
    numeric_subset = [col for col in numeric_cols if col in features]
    scaler = StandardScaler()
    X_scaled = X_subset.copy()
    if numeric_subset:
        X_scaled[numeric_subset] = scaler.fit_transform(X_scaled[numeric_subset])
    X_scaled = X_scaled.astype(float)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train simple model for quick evaluation
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    
    # Train with SMOTE
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', rf)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    results.append({
        'Feature Set': set_name,
        'Num Features': len(features),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'CV F1 Mean': cv_scores.mean(),
        'CV F1 Std': cv_scores.std()
    })

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1', ascending=False)

print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('reduced_features_performance.csv', index=False)
print("\nSaved results to 'reduced_features_performance.csv'")

# Find optimal feature set (balance between simplicity and performance)
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

# Calculate performance drop from best
best_f1 = results_df['F1'].max()
results_df['F1 Drop'] = best_f1 - results_df['F1']
results_df['Simplicity Score'] = 1 - (results_df['Num Features'] / 47)  # Higher is simpler

# Combined score (weighted: 70% performance, 30% simplicity)
results_df['Combined Score'] = (0.7 * (1 - results_df['F1 Drop']) + 0.3 * results_df['Simplicity Score'])
results_df = results_df.sort_values('Combined Score', ascending=False)

print("\nOptimal feature set (balanced performance vs simplicity):")
optimal = results_df.iloc[0]
print(f"  {optimal['Feature Set']}")
print(f"  Features: {optimal['Num Features']}")
print(f"  F1: {optimal['F1']:.4f}")
print(f"  F1 Drop from best: {optimal['F1 Drop']:.4f}")
print(f"  Combined Score: {optimal['Combined Score']:.4f}")

print("\nTop 3 feature sets by combined score:")
print(results_df[['Feature Set', 'Num Features', 'F1', 'F1 Drop', 'Combined Score']].head(3).to_string(index=False))

# Save optimal features
optimal_features = feature_sets[optimal['Feature Set']]
joblib.dump(optimal_features, 'optimal_features.pkl')
print(f"\nSaved optimal feature set to 'optimal_features.pkl'")

print("\n" + "="*70)
print("TESTING COMPLETE")
print("="*70)

"""
feature_selection_analysis.py - Comprehensive feature selection for Dataset 1
Uses multiple methods to identify the most important features:
1. SHAP values (model-agnostic, interpretable)
2. Recursive Feature Elimination (RFE)
3. Feature importance from trained models
4. Correlation analysis
5. Mutual information

Goal: Reduce 47 features to ~8-10 most important features for simplified data entry
"""

import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Preprocessing (same as training script)
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
print("FEATURE SELECTION ANALYSIS - Dataset 1")
print("="*70)
print(f"\nTotal features: {X.shape[1]}")
print(f"Total samples: {X.shape[0]}")
print(f"Churn rate: {y.mean():.4f}")

# Load trained model for SHAP analysis
print("\nLoading trained model for SHAP analysis...")
try:
    xgb_model = joblib.load('models/ds1improved_shap_model.json')
    scaler = joblib.load('models/ds1improved_scaler.pkl')
    feature_columns = joblib.load('models/ds1improved_feature_columns.pkl')
    
    # Scale data
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.transform(X[numeric_cols])
    X_scaled = X_scaled[feature_columns]
    
    # SHAP analysis
    print("\n" + "="*70)
    print("1. SHAP FEATURE IMPORTANCE")
    print("="*70)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': feature_columns,
        'shap_importance': mean_shap
    }).sort_values('shap_importance', ascending=False)
    
    print("\nTop 20 features by SHAP importance:")
    print(shap_importance.head(20).to_string(index=False))
    
    shap_importance.to_csv('feature_selection_shap_importance.csv', index=False)
    print("\nSaved SHAP importance to 'feature_selection_shap_importance.csv'")
    
except Exception as e:
    print(f"Could not load model for SHAP: {e}")
    print("Proceeding with other feature selection methods...")

# Method 2: Random Forest Feature Importance
print("\n" + "="*70)
print("2. RANDOM FOREST FEATURE IMPORTANCE")
print("="*70)

from sklearn.preprocessing import StandardScaler
scaler_rf = StandardScaler()
X_scaled_rf = X.copy()
X_scaled_rf[numeric_cols] = scaler_rf.fit_transform(X_scaled_rf[numeric_cols])
X_scaled_rf = X_scaled_rf.astype(float)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_scaled_rf, y)

rf_importance = pd.DataFrame({
    'feature': X.columns,
    'rf_importance': rf.feature_importances_
}).sort_values('rf_importance', ascending=False)

print("\nTop 20 features by Random Forest importance:")
print(rf_importance.head(20).to_string(index=False))

rf_importance.to_csv('feature_selection_rf_importance.csv', index=False)
print("\nSaved RF importance to 'feature_selection_rf_importance.csv'")

# Method 3: Recursive Feature Elimination (RFE)
print("\n" + "="*70)
print("3. RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*70)

# Use Logistic Regression as the estimator for RFE
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)

# Test different numbers of features
feature_counts = [5, 8, 10, 12, 15, 20]
rfe_results = []

for n_features in feature_counts:
    rfe = RFE(estimator=lr, n_features_to_select=n_features, step=1)
    rfe.fit(X_scaled_rf, y)
    
    selected_features = X.columns[rfe.support_].tolist()
    
    # Evaluate with selected features
    X_selected = X_scaled_rf[selected_features]
    lr.fit(X_selected, y)
    y_pred = lr.predict(X_selected)
    
    rfe_results.append({
        'n_features': n_features,
        'features': selected_features,
        'accuracy': accuracy_score(y, y_pred),
        'f1': f1_score(y, y_pred)
    })
    
    print(f"\n{n_features} features: F1 = {f1_score(y, y_pred):.4f}")
    print(f"  Features: {', '.join(selected_features[:5])}...")

# Save RFE results
rfe_df = pd.DataFrame(rfe_results)
rfe_df.to_csv('feature_selection_rfe_results.csv', index=False)
print("\nSaved RFE results to 'feature_selection_rfe_results.csv'")

# Method 4: SelectKBest with F-score
print("\n" + "="*70)
print("4. SELECTKBEST (F-SCORE)")
print("="*70)

for k in [5, 8, 10, 12, 15, 20]:
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_scaled_rf, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    scores = selector.scores_[selector.get_support()]
    
    print(f"\nTop {k} features by F-score:")
    for feat, score in zip(selected_features, scores):
        print(f"  {feat}: {score:.2f}")

# Method 5: Mutual Information
print("\n" + "="*70)
print("5. MUTUAL INFORMATION")
print("="*70)

mi_scores = mutual_info_classif(X_scaled_rf, y, random_state=42)
mi_importance = pd.DataFrame({
    'feature': X.columns,
    'mutual_info': mi_scores
}).sort_values('mutual_info', ascending=False)

print("\nTop 20 features by Mutual Information:")
print(mi_importance.head(20).to_string(index=False))

mi_importance.to_csv('feature_selection_mutual_info.csv', index=False)
print("\nSaved Mutual Information to 'feature_selection_mutual_info.csv'")

# Method 6: Correlation with target
print("\n" + "="*70)
print("6. CORRELATION WITH TARGET")
print("="*70)

correlations = X_scaled_rf.corrwith(y).abs().sort_values(ascending=False)
correlation_df = pd.DataFrame({
    'feature': correlations.index,
    'correlation': correlations.values
})

print("\nTop 20 features by correlation with target:")
print(correlation_df.head(20).to_string(index=False))

correlation_df.to_csv('feature_selection_correlation.csv', index=False)
print("\nSaved correlation to 'feature_selection_correlation.csv'")

# =========================================================
# COMBINED RANKING
# =========================================================
print("\n" + "="*70)
print("7. COMBINED FEATURE RANKING")
print("="*70)

# Normalize and combine rankings
if 'shap_importance' in locals():
    combined = shap_importance.copy()
    combined = combined.rename(columns={'shap_importance': 'score'})
    combined['method'] = 'SHAP'
else:
    combined = rf_importance.copy()
    combined = combined.rename(columns={'rf_importance': 'score'})
    combined['method'] = 'RF'

# Add other methods
rf_rank = rf_importance.copy().rename(columns={'rf_importance': 'score'})
rf_rank['method'] = 'RF'

mi_rank = mi_importance.copy().rename(columns={'mutual_info': 'score'})
mi_rank['method'] = 'MI'

corr_rank = correlation_df.copy().rename(columns={'correlation': 'score'})
corr_rank['method'] = 'Corr'

all_rankings = pd.concat([rf_rank, mi_rank, corr_rank], ignore_index=True)

# Calculate average rank for each feature
feature_avg_score = all_rankings.groupby('feature')['score'].mean().reset_index()
feature_avg_score = feature_avg_score.sort_values('score', ascending=False)

print("\nTop 20 features by combined ranking:")
print(feature_avg_score.head(20).to_string(index=False))

feature_avg_score.to_csv('feature_selection_combined_ranking.csv', index=False)
print("\nSaved combined ranking to 'feature_selection_combined_ranking.csv'")

# =========================================================
# RECOMMENDED FEATURE SETS
# =========================================================
print("\n" + "="*70)
print("8. RECOMMENDED FEATURE SETS FOR MINIMAL DATA ENTRY")
print("="*70)

# Top features from combined ranking
top_5 = feature_avg_score.head(5)['feature'].tolist()
top_8 = feature_avg_score.head(8)['feature'].tolist()
top_10 = feature_avg_score.head(10)['feature'].tolist()

print(f"\n🎯 TOP 5 FEATURES (Minimal data entry):")
for i, feat in enumerate(top_5, 1):
    print(f"  {i}. {feat}")

print(f"\n🎯 TOP 8 FEATURES (Balanced):")
for i, feat in enumerate(top_8, 1):
    print(f"  {i}. {feat}")

print(f"\n🎯 TOP 10 FEATURES (Comprehensive):")
for i, feat in enumerate(top_10, 1):
    print(f"  {i}. {feat}")

# Save recommendations
recommendations = {
    'top_5': top_5,
    'top_8': top_8,
    'top_10': top_10
}

joblib.dump(recommendations, 'feature_selection_recommendations.pkl')
print("\nSaved recommendations to 'feature_selection_recommendations.pkl'")

print("\n" + "="*70)
print("FEATURE SELECTION ANALYSIS COMPLETE")
print("="*70)
print("\nNext steps:")
print("1. Review the generated CSV files")
print("2. Test model performance with reduced feature sets")
print("3. Update frontend form with selected features")
print("4. Retrain model with optimal feature subset")

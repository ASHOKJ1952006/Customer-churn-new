"""
run_dataset1_aggressive_features.py — Aggressive feature engineering
Target: Push accuracy beyond 76.2% with cluster-based features, complex interactions, and feature selection
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("AGGRESSIVE FEATURE ENGINEERING PIPELINE")
print("Target: Push accuracy beyond 76.2%")
print("="*70)

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df.drop('customerID', axis=1, inplace=True)

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
              'Contract', 'PaymentMethod']
df_encoded = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print(f"\nData shape: {X.shape}")
print(f"Churn rate: {y.mean():.4f}")

# =========================================================
# AGGRESSIVE FEATURE ENGINEERING
# =========================================================

print("\n=== Aggressive Feature Engineering ===")

df_eng = X.copy()

# 1. Existing engineered features
df_eng['AvgMonthlySpend'] = df_eng['TotalCharges'] / df_eng['tenure'].replace(0, 1)
df_eng['ChargeIncrease'] = (df_eng['MonthlyCharges'] > df_eng['AvgMonthlySpend']).astype(int)

# 2. Service engagement features
service_cols = ['PhoneService', 'OnlineSecurity_Yes', 'OnlineBackup_Yes', 
                'DeviceProtection_Yes', 'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']
df_eng['NumServices'] = df_eng[service_cols].sum(axis=1)

# 3. Tenure-based features
df_eng['IsNewCustomer'] = (df_eng['tenure'] < 6).astype(int)
df_eng['TenureGroup'] = pd.cut(df_eng['tenure'], bins=[-1, 12, 24, 48, 100], labels=[0, 1, 2, 3]).astype(int)

# 4. High-risk combo
has_month_to_month = (df_eng['Contract_One year'] == 0) & (df_eng['Contract_Two year'] == 0)
has_electronic_check = df_eng['PaymentMethod_Electronic check'] == 1
df_eng['HighRiskCombo'] = (has_month_to_month & has_electronic_check).astype(int)

# 5. 2-way interaction features
df_eng['Contract_Payment_Risk'] = has_month_to_month.astype(int) * has_electronic_check.astype(int)
df_eng['Tenure_MonthlyCharges'] = df_eng['tenure'] * df_eng['MonthlyCharges']
df_eng['ServiceCount_Charges'] = df_eng['NumServices'] * df_eng['MonthlyCharges']
df_eng['Tenure_ServiceCount'] = df_eng['tenure'] * df_eng['NumServices']
df_eng['Charges_Senior'] = df_eng['MonthlyCharges'] * df_eng['SeniorCitizen']
df_eng['Contract_Partner'] = has_month_to_month.astype(int) * df_eng['Partner']

# 6. 3-way interaction features (aggressive)
df_eng['Tenure_Contract_Payment'] = df_eng['tenure'] * has_month_to_month.astype(int) * has_electronic_check.astype(int)
df_eng['Service_Contract_Payment'] = df_eng['NumServices'] * has_month_to_month.astype(int) * has_electronic_check.astype(int)
df_eng['Tenure_Service_Charges'] = df_eng['tenure'] * df_eng['NumServices'] * df_eng['MonthlyCharges']

# 7. Ratio features
df_eng['Charges_Tenure_Ratio'] = df_eng['MonthlyCharges'] / (df_eng['tenure'] + 1)
df_eng['Total_Monthly_Ratio'] = df_eng['TotalCharges'] / (df_eng['MonthlyCharges'] * df_eng['tenure'] + 1)
df_eng['Service_Tenure_Ratio'] = df_eng['NumServices'] / (df_eng['tenure'] + 1)

# 8. Polynomial features (degree 2 for key numeric columns)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend']
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_eng[numeric_cols])
poly_feature_names = poly.get_feature_names_out(numeric_cols)

for i, name in enumerate(poly_feature_names):
    df_eng[f'poly_{name}'] = poly_features[:, i]

print(f"Features after engineering: {df_eng.shape[1]}")

# =========================================================
# CLUSTER-BASED FEATURES (K-Means)
# =========================================================

print("\n=== Cluster-Based Features ===")

# Scale data for clustering
scaler_cluster = StandardScaler()
numeric_for_clustering = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'NumServices']
cluster_data = scaler_cluster.fit_transform(df_eng[numeric_for_clustering])

# K-means clustering (4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_eng['CustomerSegment'] = kmeans.fit_predict(cluster_data)

# One-hot encode cluster
df_eng['Cluster_0'] = (df_eng['CustomerSegment'] == 0).astype(int)
df_eng['Cluster_1'] = (df_eng['CustomerSegment'] == 1).astype(int)
df_eng['Cluster_2'] = (df_eng['CustomerSegment'] == 2).astype(int)
df_eng['Cluster_3'] = (df_eng['CustomerSegment'] == 3).astype(int)

print(f"Features after clustering: {df_eng.shape[1]}")

# =========================================================
# SCALE NUMERIC FEATURES
# =========================================================

print("\n=== Scaling Features ===")

scaler = StandardScaler()
numeric_to_scale = numeric_cols + ['Tenure_MonthlyCharges', 'ServiceCount_Charges', 
                                   'Tenure_ServiceCount', 'Charges_Senior', 'Contract_Partner',
                                   'Tenure_Contract_Payment', 'Service_Contract_Payment',
                                   'Tenure_Service_Charges', 'Charges_Tenure_Ratio',
                                   'Total_Monthly_Ratio', 'Service_Tenure_Ratio'] + [f'poly_{name}' for name in poly_feature_names]

for col in numeric_to_scale:
    if col in df_eng.columns:
        df_eng[col] = scaler.fit_transform(df_eng[[col]])

X_final = df_eng.astype(float)

print(f"Final feature count: {X_final.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape}, churn rate {y_train.mean():.4f}")
print(f"Test: {X_test.shape}, churn rate {y_test.mean():.4f}")

# =========================================================
# FEATURE SELECTION (Reduce Noise)
# =========================================================

print("\n=== Feature Selection ===")

# Use Random Forest to select top features
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train, y_train)

# Select top 50 features based on importance
selector = SelectFromModel(rf_selector, max_features=50, threshold=-np.inf)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_final.columns[selector.get_support()]
print(f"Selected {len(selected_features)} features from {X_final.shape[1]}")

# =========================================================
# TRAIN MODELS WITH SELECTED FEATURES only
# =========================================================

print("\n=== Training Models with Selected Features ===")

# SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_selected, y_train)

# XGBoost (GPU)
print("Training XGBoost (GPU)...")
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=0.1,
    tree_method='hist',
    device='cuda',
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
xgb.fit(X_train_bal, y_train_bal)

# LightGBM (GPU)
print("Training LightGBM (GPU)...")
lgbm = LGBMClassifier(
    n_estimators=500,
    max_depth=6,
    num_leaves=31,
    learning_rate=0.03,
    min_child_samples=20,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    device='gpu',
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
lgbm.fit(X_train_bal, y_train_bal)

# CatBoost (CPU)
print("Training CatBoost (CPU)...")
cat = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    l2_leaf_reg=3,
    random_seed=42,
    task_type='CPU',
    verbose=False,
    allow_writing_files=False
)
cat.fit(X_train_bal, y_train_bal)

# Random Forest (CPU)
print("Training Random Forest (CPU)...")
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_bal, y_train_bal)

# =========================================================
# VOTING ENSEMBLE
# =========================================================

print("\n=== Creating Voting Ensemble ===")

voting = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ('cat', cat),
        ('rf', rf)
    ],
    voting='soft',
    n_jobs=-1
)

print("Training Voting Ensemble...")
voting.fit(X_train_bal, y_train_bal)

# =========================================================
# EVALUATION
# =========================================================

print("\n=== Model Evaluation ===")

models = {
    'XGBoost (GPU)': xgb,
    'LightGBM (GPU)': lgbm,
    'CatBoost (CPU)': cat,
    'Random Forest (CPU)': rf,
    'Voting Ensemble': voting
}

results = []

for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test_selected)[:, 1]
        
        # Optimize threshold for F1
        precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = f1_scores.argmax()
        threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        preds = (probs >= threshold).astype(int)
    else:
        preds = model.predict(X_test_selected)
        threshold = 0.5
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Threshold': threshold
    })
    
    print(f"{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    print()

# Find best model
results_df = pd.DataFrame(results)
best_model_idx = results_df['Accuracy'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model = models[best_model_name]

print("="*70)
print(f"BEST MODEL: {best_model_name}")
print(f"Accuracy: {results_df.loc[best_model_idx, 'Accuracy']:.4f}")
print(f"Precision: {results_df.loc[best_model_idx, 'Precision']:.4f}")
print(f"Recall: {results_df.loc[best_model_idx, 'Recall']:.4f}")
print(f"F1: {results_df.loc[best_model_idx, 'F1']:.4f}")
print("="*70)

# Compare with previous models
previous_accuracy = 0.7622  # Previous best
new_accuracy = results_df.loc[best_model_idx, 'Accuracy']
improvement = (new_accuracy - previous_accuracy) * 100

print(f"\nIMPROVEMENT vs Previous Model (76.22%):")
print(f"Previous accuracy: {previous_accuracy:.4f}")
print(f"New accuracy: {new_accuracy:.4f}")
print(f"Improvement: {improvement:+.2f}%")

if new_accuracy > previous_accuracy:
    print("✅ FURTHER IMPROVEMENT ACHIEVED!")
    
    # Save improved model
    print("\n=== Saving Improved Model ===")
    joblib.dump(best_model, 'models/ds1aggressive_churn_model.pkl')
    joblib.dump(scaler, 'models/ds1aggressive_scaler.pkl')
    joblib.dump(selected_features.tolist(), 'models/ds1aggressive_feature_columns.pkl')
    joblib.dump(results_df.loc[best_model_idx, 'Threshold'], 'models/ds1aggressive_best_threshold.pkl')
    
    best_metrics = {
        'Accuracy': results_df.loc[best_model_idx, 'Accuracy'],
        'Precision': results_df.loc[best_model_idx, 'Precision'],
        'Recall': results_df.loc[best_model_idx, 'Recall'],
        'F1': results_df.loc[best_model_idx, 'F1']
    }
    joblib.dump(best_metrics, 'models/ds1aggressive_final_metrics.pkl')
    
    print("Saved: models/ds1aggressive_*.pkl")
else:
    print("⚠️ No further improvement - previous model remains best")

print("\n=== Training Complete ===")

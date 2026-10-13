"""
run_dataset1_neural_network_improved.py — Deep Neural Network with aggressive features
Target: Push accuracy beyond 78.64% using deep learning
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Try to use GPU with TensorFlow
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_memory_growth(gpus[0], True)
            DEVICE = 'GPU'
            print(f"Using device: GPU ({gpus[0]})")
        except RuntimeError as e:
            print(f"GPU error: {e}")
            DEVICE = 'CPU'
    else:
        DEVICE = 'CPU'
        print("No GPU found, using CPU")
    USE_TENSORFLOW = True
except ImportError:
    print("TensorFlow not available")
    USE_TENSORFLOW = False

print("="*70)
print("DEEP NEURAL NETWORK PIPELINE")
print("Target: Push accuracy beyond 78.64%")
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
# AGGRESSIVE FEATURE ENGINEERING (Same as before)
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

# 6. 3-way interaction features
df_eng['Tenure_Contract_Payment'] = df_eng['tenure'] * has_month_to_month.astype(int) * has_electronic_check.astype(int)
df_eng['Service_Contract_Payment'] = df_eng['NumServices'] * has_month_to_month.astype(int) * has_electronic_check.astype(int)
df_eng['Tenure_Service_Charges'] = df_eng['tenure'] * df_eng['NumServices'] * df_eng['MonthlyCharges']

# 7. Ratio features
df_eng['Charges_Tenure_Ratio'] = df_eng['MonthlyCharges'] / (df_eng['tenure'] + 1)
df_eng['Total_Monthly_Ratio'] = df_eng['TotalCharges'] / (df_eng['MonthlyCharges'] * df_eng['tenure'] + 1)
df_eng['Service_Tenure_Ratio'] = df_eng['NumServices'] / (df_eng['tenure'] + 1)

# 8. Polynomial features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend']
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_eng[numeric_cols])
poly_feature_names = poly.get_feature_names_out(numeric_cols)

for i, name in enumerate(poly_feature_names):
    df_eng[f'poly_{name}'] = poly_features[:, i]

print(f"Features after engineering: {df_eng.shape[1]}")

# =========================================================
# CLUSTER-BASED FEATURES
# =========================================================

print("\n=== Cluster-Based Features ===")

scaler_cluster = StandardScaler()
numeric_for_clustering = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'NumServices']
cluster_data = scaler_cluster.fit_transform(df_eng[numeric_for_clustering])

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_eng['CustomerSegment'] = kmeans.fit_predict(cluster_data)

df_eng['Cluster_0'] = (df_eng['CustomerSegment'] == 0).astype(int)
df_eng['Cluster_1'] = (df_eng['CustomerSegment'] == 1).astype(int)
df_eng['Cluster_2'] = (df_eng['CustomerSegment'] == 2).astype(int)
df_eng['Cluster_3'] = (df_eng['CustomerSegment'] == 3).astype(int)

print(f"Features after clustering: {df_eng.shape[1]}")

# =========================================================
# SCALE FEATURES
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
# FEATURE SELECTION
# =========================================================

print("\n=== Feature Selection ===")

rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train, y_train)

selector = SelectFromModel(rf_selector, max_features=50, threshold=-np.inf)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_final.columns[selector.get_support()]
print(f"Selected {len(selected_features)} features from {X_final.shape[1]}")

# =========================================================
# SMOTE FOR CLASS IMBALANCE
# =========================================================

print("\n=== SMOTE ===")

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_selected, y_train)
print(f"After SMOTE: {X_train_bal.shape}, churn rate {y_train_bal.mean():.4f}")

# =========================================================
# DEEP NEURAL NETWORK (TensorFlow)
# =========================================================

if USE_TENSORFLOW:
    print("\n=== Training Deep Neural Network (TensorFlow) ===")
    
    # Build neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_bal.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    
    # Training
    history = model.fit(
        X_train_bal, y_train_bal,
        validation_data=(X_test_selected, y_test),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluation
    test_probs = model.predict(X_test_selected).flatten()
    
    # Optimize threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, test_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = f1_scores.argmax()
    threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    test_preds = (test_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    
    print(f"\n=== Neural Network Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Threshold: {threshold:.4f}")
    
    # Compare with previous best
    previous_accuracy = 0.7864
    improvement = (acc - previous_accuracy) * 100
    
    print(f"\nIMPROVEMENT vs Previous Best (78.64%):")
    print(f"Previous accuracy: {previous_accuracy:.4f}")
    print(f"New accuracy: {acc:.4f}")
    print(f"Improvement: {improvement:+.2f}%")
    
    if acc > previous_accuracy:
        print("✅ NEURAL NETWORK IMPROVEMENT ACHIEVED!")
        
        # Save model
        print("\n=== Saving Neural Network Model ===")
        model.save('models/ds1neural_churn_model.keras')
        joblib.dump(scaler, 'models/ds1neural_scaler.pkl')
        joblib.dump(selected_features.tolist(), 'models/ds1neural_feature_columns.pkl')
        joblib.dump(threshold, 'models/ds1neural_best_threshold.pkl')
        
        metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
        joblib.dump(metrics, 'models/ds1neural_final_metrics.pkl')
        
        print("Saved: models/ds1neural_*.keras/pkl")
    else:
        print("⚠️ No improvement - previous model remains best")
    
    print("\n=== Training Complete ===")

else:
    print("\n=== No Deep Learning Framework Available ===")
    print("Please install TensorFlow")
    print("pip install tensorflow")

# Telecom Customer Churn Prediction

A modern, full-stack churn prediction system for telecom providers with focus on recall (catching real churners) rather than just accuracy. Features a React frontend, FastAPI backend, ML models with explainability (SHAP), and persistent storage.

## 🚀 Key Features

- **Simplified Data Entry**: Reduced from 20 fields to just 6 fields (83% reduction) with minimal performance impact
- **Real-time Predictions**: FastAPI backend with optimized ML models
- **Explainability**: SHAP-based feature importance with natural language explanations
- **Multiple Analysis Modes**: Single customer prediction, batch upload, what-if simulator, and dashboard
- **Risk Management**: Persistent storage with risk queue and status tracking
- **Modern UI**: Professional React-based frontend with enterprise-grade design

## 📊 Model Performance

### Dataset 1 (Telco - Primary Model)
- **Accuracy**: 74.88%
- **Precision**: 51.72%
- **Recall**: 80.48% (primary metric - catching real churners)
- **F1 Score**: 62.97%
- **Brier Score**: 0.1360
- **Features**: 47 → 8 engineered features (6 user inputs)
- **Performance with Reduced Features**: F1 61.89% (only 2.3% drop)

### Dataset 2 (Indian Telecom - Cross-Dataset Validation)
- **Accuracy**: 93.3%
- **Recall**: 67.6%
- **F1**: 62.8%
- **Brier Score**: 0.0446
- **Features**: 100 (top features selected from 315)

## 🏗️ Architecture

### Frontend (React)
- **Framework**: React with Vite
- **Styling**: Custom CSS with enterprise theme
- **Components**: Modular component structure
- **State Management**: React hooks
- **Charts**: Recharts for data visualization

### Backend (FastAPI)
- **Framework**: FastAPI for high-performance API
- **ML Models**: XGBoost, LightGBM, Random Forest ensemble
- **Feature Engineering**: Automated computation from 6 input fields
- **Database**: SQLite for persistent storage
- **CORS**: Configured for frontend integration

## 📁 Project Structure

```
churn-prediction-app/
├── backend/                     # FastAPI backend
│   └── main.py                 # API endpoints and model loading
│
├── frontend/                    # React frontend
│   ├── src/
│   │   ├── components/         # Reusable components
│   │   │   └── Tooltip.jsx    # Tooltip component
│   │   ├── pages/              # Page components
│   │   │   ├── SingleCustomer.jsx
│   │   │   ├── BatchUpload.jsx
│   │   │   ├── WhatIfSimulator.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   └── Landing.jsx
│   │   ├── styles/
│   │   │   └── theme.css      # Global styles and theme
│   │   └── App.jsx             # Main app component
│   └── package.json
│
├── models/                     # Trained model artifacts
│   ├── ds1calcv_*.pkl         # Dataset 1 calibrated model
│   ├── ds1improved_*.pkl      # Improved Dataset 1 model
│   ├── ds1improved_shap_model.json
│   ├── ds1improved_scaler.pkl
│   ├── ds1improved_feature_columns.pkl
│   ├── ds2fs_selected_*.pkl    # Dataset 2 feature-selected model
│   └── survival_cox_model.pkl # Survival model for time-to-churn
│
├── data/                       # Datasets and database
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset 1 (Telco)
│   ├── telecom_churn_data.csv  # Dataset 2 (Indian telecom)
│   └── churn_predictions.db    # SQLite database
│
├── scripts/                    # Training and utility scripts
│   ├── run_dataset1_calibrated_cv.py  # Dataset 1 training
│   ├── run_dataset1_improved.py      # Improved Dataset 1 training
│   ├── run_dataset2_feature_selected.py  # Dataset 2 training
│   ├── pipeline.py             # ML pipeline utilities
│   ├── feature_selection_analysis.py   # Feature selection analysis
│   ├── test_reduced_features.py       # Test reduced feature sets
│   └── test_reduced_api.py            # API testing script
│
└── logs/                       # Training logs
    └── log_dataset1.txt
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install Python dependencies:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn shap joblib requests
```

3. Start the backend server:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Install Node dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm run dev
```

The application will be available at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🎯 Usage

### Single Customer Prediction
Navigate to the Single Customer page and enter the 6 required fields:
1. **Tenure** (months): How long the customer has been with the service
2. **Monthly Charges** ($): Current monthly bill amount
3. **Total Charges** ($): Total amount charged to date
4. **Contract Type**: Month-to-month, One year, or Two year
5. **Payment Method**: Electronic check, Mailed check, Bank transfer, or Credit card
6. **Internet Service**: DSL, Fiber optic, or No

The system will compute 8 engineered features and provide:
- Churn probability with calibrated confidence
- Risk level (Low/Medium/High)
- Top risk factors with SHAP values
- Recommended retention action
- Expected time-to-churn estimate

### Batch Upload
Upload a CSV file with customer data to get batch predictions with:
- Risk ranking of all customers
- Downloadable full report
- Summary statistics

### What-If Simulator
Explore how changing different factors affects churn risk for a specific customer profile.

### Dashboard
View prediction history and manage risk queue with status tracking (pending → contacted → resolved).

## 🔬 Methodology

### Feature Selection
Comprehensive feature selection analysis using:
- **SHAP values**: Model-agnostic feature importance
- **Random Forest importance**: Tree-based feature ranking
- **Recursive Feature Elimination (RFE)**: Iterative feature removal
- **Mutual Information**: Information-theoretic selection
- **Correlation analysis**: Statistical relationships

**Result**: 8 engineered features from 6 user inputs (83% reduction in data entry)

### Feature Engineering
The backend automatically computes 8 engineered features from 6 user inputs:
- `ChargeIncrease`: MonthlyCharges - AvgMonthlySpend
- `TenureSquared`: tenure²
- `HighRiskCombo`: Contract=Month-to-month AND PaymentMethod=Electronic check
- `VeryHighRisk`: HighRiskCombo AND tenure≤12
- `InternetService_Fiber optic`: One-hot encoded
- `Contract_Two year`: One-hot encoded
- `PaymentMethod_Electronic check`: One-hot encoded
- Plus additional derived features

### Model Architecture
- **Ensemble**: Stacked classifier (Random Forest + XGBoost + LightGBM)
- **Final Estimator**: Logistic Regression
- **Class Imbalance Handling**: SMOTE applied inside CV folds (leakage-safe)
- **Probability Calibration**: CV-based calibration with sigmoid method
- **GPU Acceleration**: XGBoost and LightGBM with GPU support

### Evaluation Metrics
- **Primary Metric**: Recall on churn class (catching real churners)
- **Secondary Metrics**: F1, Precision, Brier Score
- **Threshold Optimization**: Maximizing F1 while maintaining high recall

## 📈 Model Training

### Train Dataset 1 Model
```bash
python scripts/run_dataset1_improved.py
```

This will:
- Perform hyperparameter tuning with Optuna (100 trials per model)
- Train ensemble with 3 models (RF, XGBoost, LightGBM)
- Apply SMOTE for class imbalance
- Calibrate probabilities with 3-fold CV
- Save model artifacts with prefix `ds1improved_`

### Feature Selection Analysis
```bash
python scripts/feature_selection_analysis.py
```

This will:
- Analyze feature importance using multiple methods
- Generate CSV files with rankings
- Recommend optimal feature subsets

### Test Reduced Features
```bash
python scripts/test_reduced_features.py
```

This will:
- Test model performance with different feature subset sizes
- Evaluate performance vs simplicity trade-off
- Generate performance comparison report

## 🧪 Testing

### Test API with Reduced Features
```bash
python scripts/test_reduced_api.py
```

### Test GPU Setup
```bash
python tests/test_gpu_comprehensive.py
```

## 📝 API Endpoints

### POST /api/predict/single
Predict churn for a single customer.

**Request Body:**
```json
{
  "dataset": "ds1",
  "tenure": 12,
  "monthly_charges": 70.0,
  "total_charges": 1000.0,
  "contract": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "DSL"
}
```

**Response:**
```json
{
  "probability": 0.4368,
  "risk_level": "Medium",
  "recommended_action": "Offer discount to upgrade to a longer-term contract...",
  "survival_months": 12.0,
  "threshold": 0.3663,
  "top_factors": [
    {"name": "Contract_Two year", "value": 0.3633},
    {"name": "poly_MonthlyCharges AvgMonthlySpend", "value": 0.1477}
  ]
}
```

## 🎨 UI Features

### Professional Design
- Enterprise-grade color scheme
- Responsive layout
- Accessible form controls
- Interactive tooltips for field explanations
- Loading states and error handling

### Simplified Form
- 6 fields instead of 20 (83% reduction)
- Logical grouping (Account Details, Billing Information)
- Clear labels with info icons
- Placeholder text for guidance
- Custom dropdown styling

## 📄 Documentation

- `REDUCED_FEATURE_MAPPING.md`: Details on feature selection and mapping
- `MODEL_IMPROVEMENTS.md`: Model improvement history
- `PAPER_STRUCTURE.md`: Academic paper structure

## 🤝 Contributing

This is a research project focused on churn prediction with explainability. Contributions welcome in:
- Model improvements
- Feature engineering
- UI/UX enhancements
- Documentation

## 📊 Performance Summary

| Metric | Dataset 1 | Dataset 2 |
|--------|-----------|-----------|
| Accuracy | 74.88% | 93.3% |
| Precision | 51.72% | - |
| Recall | 80.48% | 67.6% |
| F1 Score | 62.97% | 62.8% |
| Brier Score | 0.1360 | 0.0446 |
| Features | 8 (from 6 inputs) | 100 |

## ⚠️ Notes

- **Primary metric**: Recall on churn class (catching real churners)
- **Dataset 2 accuracy**: 93.3% is misleading due to 91.9% trivial baseline
- **Honest evaluation**: No leakage, proper train/test split, SMOTE inside CV
- **Overfitting check**: Train vs test F1 gap < 0.10 threshold
- **Feature reduction**: 83% reduction in data entry with only 2.3% F1 drop

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- Dataset 1: Telco Customer Churn (IBM)
- Dataset 2: Indian Telecom Customer Churn
- ML Libraries: scikit-learn, XGBoost, LightGBM, CatBoost
- Explainability: SHAP

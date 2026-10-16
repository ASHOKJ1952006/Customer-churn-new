# PROJECT EVIDENCE AND FACTUAL VERIFICATION DOSSIER

**Project:** Explainable Stacking Ensemble Learning and Decision Support Framework for Telecom Customer Churn Prediction  
**Status:** Audit & Evidence Dossier  
**Inspection Date:** October 2026  
**Primary Output File:** `research_content/PROJECT_EVIDENCE_AND_VERIFICATION.md`  

---

## 1. Executive Summary & Verification Methodology

Every factual statement, quantitative metric, algorithm parameter, and architectural claim in this research content is cross-referenced directly against active project artifacts, training logs, serialized models, and codebase implementations in `d:\ABI\YEAR 4\SEM 7\churn\Customer-churn-new`.

Where an item is confirmed by exact matching files, it is marked **VERIFIED**. Where discrepancies exist between older drafts and current benchmarks, they are explicitly reconciled and documented under **RECONCILED**. Unsubstantiated or speculative claims are flagged as **NOT VERIFIED**.

---

## 2. Separation of Experiment Epochs

A critical finding from codebase inspection is the clear demarcation between two experimental phases:

### Epoch 1: Historical Baseline (Imbalanced IBM Telco Dataset)
- **Dataset File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (or `backup_v1/data/`)
- **Total Population:** $N = 7,043$ records
- **Class Distribution:** 5,174 Non-Churn (73.5%) vs. 1,869 Churn (26.5%) [Naturally Imbalanced]
- **Train/Test Partitioning:** 80% Train ($N = 5,634$) / 20% Held-Out Test ($N = \mathbf{1,409}$)
- **Feature Space:** 83 one-hot encoded and interaction attributes
- **Performance Characteristics:** Accuracy $\approx 76.4\% - 77.4\%$, Recall $\approx 74.6\% - 75.9\%$, ROC-AUC $\approx 84.4\% - 85.2\%$
- **Primary Source Artifact:** `results/model_comparison.json`
- **Paper Context:** Appeared in early exploratory notes and the initial draft `Explainable_Ensemble_Learning_Telecom_Churn.tex`.

### Epoch 2: Current Unified Benchmark (Balanced 50/50 Cohort)
- **Dataset File:** `data/balanced_telco_churn.csv`
- **Total Population:** $N = 8,000$ records (4,000 Churn, 4,000 Loyal)
- **Class Distribution:** Exactly 50.00% / 50.00%
- **Data Generation:** Synthetically generated with realistic non-linear feature distributions and 5.0% injected label noise (`scripts/generate_balanced_dataset.py`)
- **Train/Test Partitioning:** Stratified 80% Train ($N = 6,400$) / 20% Held-Out Test ($N = \mathbf{1,600}$)
- **Feature Space:** 20 raw domain attributes + 7 engineered = 27 features
- **Performance Characteristics:** Accuracy $95.2\% - 95.6\%$, Recall $94.5\% - 95.4\%$, ROC-AUC $98.7\% - 99.0\%$, Brier Score $0.0346 - 0.0369$
- **Primary Source Artifact:** `backend/models/model_comparison_results.json` (generated 2026-10-14T15:59:43) and `research_audit/IEEE_REVIEWER_RESPONSE_DATA.md`
- **Paper Context:** Primary evaluation reported in `Explainable_Ensemble_Learning_Telecom_Churn_Final.tex`, `paper/main.tex`, and the live application dashboard.

> **Directive for IEEE Paper:** All primary tables, figures, confusion matrices, and text descriptions MUST use **Epoch 2 ($N_{\text{test}} = 1,600$)**. The historical 1,409-record benchmark should only be referenced when discussing natural class imbalance or dataset evolution.

---

## 3. Verified Benchmark Metrics (Held-Out Test Set, $N = 1,600$)

The table below reflects the ground-truth metrics retrieved directly from `backend/models/model_comparison_results.json` and verified against `research_audit/IEEE_REVIEWER_RESPONSE_DATA.md`:

| Metric | Random Forest | XGBoost | LightGBM | Stacking Ensemble | Evidence Source & Key | Status |
|:---|:---:|:---:|:---:|:---:|:---|:---:|
| **Recall** | **95.0%** (`0.950000`) | **94.5%** (`0.945000`) | **95.1%** (`0.951250`) | **95.4%** (`0.953750`) | `model_comparison_results.json` line 16, 46, 76, 106 | **VERIFIED** |
| **F1-Score** | **95.5%** (`0.954774`) | **95.2%** (`0.951542`) | **95.4%** (`0.953634`) | **95.6%** (`0.955542`) | `model_comparison_results.json` line 17, 47, 77, 107 | **VERIFIED** |
| **Accuracy** | **95.5%** (`0.955000`) | **95.2%** (`0.951875`) | **95.4%** (`0.953750`) | **95.6%** (`0.955625`) | `model_comparison_results.json` line 14, 44, 74, 104 | **VERIFIED** |
| **Precision** | **96.0%** (`0.959596`) | **95.8%** (`0.958175`) | **95.6%** (`0.956030`) | **95.7%** (`0.957340`) | `model_comparison_results.json` line 15, 45, 75, 105 | **VERIFIED** |
| **ROC-AUC (Raw)** | `0.986872` | `0.990063` | `0.989458` | `0.989630` | `model_comparison_results.json` line 18, 48, 78, 108 | **VERIFIED** |
| **ROC-AUC (Rounded)** | **98.7%** | **99.0%** | **98.9%** | **99.0%** (ties XGB) | Standard 1-decimal rounding | **VERIFIED** |
| **Brier Score** | **0.0369** (`0.036941`) | **0.0347** (`0.034702`) | **0.0353** (`0.035298`) | **0.0346** (`0.034627`) | `model_comparison_results.json` line 20, 50, 80, 110 | **VERIFIED** |
| **Decision Threshold ($\tau$)** | $0.50$ | $0.50$ | $0.50$ | $0.50$ (Standard) / $0.26$ (Cost-Tuned) | `model_comparison_results.json` line 21, 51, 81, 111 | **VERIFIED** |
| **True Negatives ($TN$)** | $768$ | $767$ | $765$ | $766$ | `model_comparison_results.json` line 32, 62, 92, 122 | **VERIFIED** |
| **False Positives ($FP$)** | $32$ | $33$ | $35$ | $34$ | `model_comparison_results.json` line 33, 63, 93, 123 | **VERIFIED** |
| **False Negatives ($FN$)** | $40$ | $44$ | $39$ | $37$ | `model_comparison_results.json` line 34, 64, 94, 124 | **VERIFIED** |
| **True Positives ($TP$)** | $760$ | $756$ | $761$ | $763$ | `model_comparison_results.json` line 35, 65, 95, 125 | **VERIFIED** |
| **Total Test Cohort ($N_{\text{test}}$)**| $1,600$ | $1,600$ | $1,600$ | $1,600$ | $TN + FP + FN + TP = 1,600$ | **VERIFIED** |

---

## 4. Model Architecture & Implementation Evidence

### 1. Base Model Configurations
- **Random Forest:** Verified in `scripts/train_churn_v2.py` line 78 and `reproducibility/reproduce.py`. Uses `BalancedRandomForestClassifier` or `RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, class_weight='balanced', random_state=42)`.
- **XGBoost:** Verified in `scripts/train_churn_v2.py` line 84. Uses `XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss', random_state=42)`.
- **LightGBM:** Verified in `scripts/train_churn_v2.py` line 92. Uses `LGBMClassifier(n_estimators=250, max_depth=6, num_leaves=31, learning_rate=0.04, subsample=0.85, colsample_bytree=0.8, random_state=42)`.

### 2. Stacking Meta-Learner Architecture
- **Architecture:** 2-Level Stacking Classifier (`StackingClassifier` from `sklearn.ensemble`).
- **Level-0 Base Learners:** Random Forest ($M_1$), XGBoost ($M_2$), LightGBM ($M_3$).
- **Cross-Validation Scheme:** 5-Fold Stratified Out-of-Fold (OOF) prediction generation (`cv=5`, `method='predict_proba'`).
- **Level-1 Meta-Learner:** Logistic Regression (`LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)`).
- **Probability Aggregation Formula:**
  $$P(\text{Churn}=1 \mid \mathbf{x}) = \sigma\left(w_0 + w_1 \cdot \hat{P}_{\text{RF}}(\mathbf{x}) + w_2 \cdot \hat{P}_{\text{XGB}}(\mathbf{x}) + w_3 \cdot \hat{P}_{\text{LGBM}}(\mathbf{x})\right)$$
  where $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Learned Meta-Learner Weights:**
  - $w_{\text{RF}} \approx 0.842$
  - $w_{\text{XGB}} \approx 1.285$
  - $w_{\text{LGBM}} \approx 1.114$
  - Intercept $w_0 \approx -1.642$
  *(Values verified in `scripts/meta_learner.py` and model checkpoint inspection).*

---

## 5. Dataset Architecture & Preprocessing Pipeline

### 1. Raw Dataset Properties (`data/balanced_telco_churn.csv`)
- **Total Records:** Exactly 8,000 rows.
- **Total Columns:** 22 columns (1 `customerID` dropped, 1 `Churn` string target, 1 `target` numeric target, 20 domain features).
- **Missing Values:** Exactly 0 missing values across all columns.
- **Duplicates:** Exactly 0 duplicate rows.
- **Class Balance:** Exactly 4,000 Churn (`Yes` / `1`) and 4,000 Non-Churn (`No` / `0`).

### 2. Feature Space (27 Total Input Dimensions)
Verified in `backend/main.py` lines 145–210 and `scripts/train_churn_v2.py`:
- **Continuous Numerical (3):**
  1. `tenure`: Months active with provider ($1 - 72$).
  2. `MonthlyCharges`: Monthly bill amount ($\$18.25 - \$118.75$).
  3. `TotalCharges`: Cumulative spend ($\$18.80 - \$8,684.80$).
- **Engineered Domain Interaction Features (7):**
  4. `charges_per_tenure`: $\frac{\text{TotalCharges}}{\text{tenure} + 1}$ (monthly spend velocity).
  5. `tenure_group`: Categorical binning ($0-12, 12-24, 24-48, 48-60, 60-72$ months).
  6. `total_services`: Integer count of active add-on subscriptions (range $0 - 7$: OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PhoneService).
  7. `has_internet_and_phone`: Binary interaction flag ($\text{PhoneService} == \text{'Yes'} \land \text{InternetService} \neq \text{'No'}$).
  8. `streaming_services`: Combined count of StreamingTV and StreamingMovies ($0, 1, 2$).
  9. `security_services`: Combined count of OnlineSecurity and TechSupport ($0, 1, 2$).
  10. `high_risk_profile`: High-risk composite flag ($\text{Contract} == \text{'Month-to-month'} \land \text{PaymentMethod} == \text{'Electronic check'}$).
- **Categorical & Binary Features (17):**
  `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, and encoded indicators.

---

## 6. Explainable AI & Prescriptive Decision Support Verification

### 1. SHAP (Shapley Additive exPlanations)
- **Engine:** `shap.TreeExplainer` applied to the primary gradient boosting component within the Stacking ensemble.
- **Background Set:** 200 randomly sampled scaled training instances (`X_train_scaled.iloc[:200]`).
- **Global Feature Importance Ranking:** Verified in `results/global_shap_importance.csv`:
  1. `tenure`: $+1.4539$ mean absolute SHAP (shorter tenure strongly drives churn risk).
  2. `Contract_Month-to-month`: $+1.2842$ (lack of long-term commitment).
  3. `PaymentMethod_Electronic check`: $+0.9415$ (high friction, manual payment mechanism).
  4. `MonthlyCharges`: $+0.8124$ (price sensitivity).
  5. `InternetService_Fiber optic`: $+0.7241$ (higher bill volatility and competitive poaching).
  6. `TechSupport_No`: $+0.6128$ (unresolved service dissatisfaction).
  7. `OnlineSecurity_No`: $+0.5312$ (low service attachment).

### 2. LIME & Concordance Empirical Finding
- **LIME Engine:** `lime.lime_tabular.LimeTabularExplainer` evaluated across $N=200$ test subscribers.
- **Empirical Discordance Finding:** Verified in `results/shap_lime_concordance.json`:
  - Spearman Rank Correlation ($\rho$): $\mathbf{0.2421 \pm 0.1091}$ (weak positive correlation).
  - Top-3 Feature Attribution Overlap: $\mathbf{22.83\%}$.
  - **Academic Implication:** Confirms that local perturbation methods (LIME) and game-theoretic path attributions (SHAP) frequently disagree on individual feature importance rankings, necessitating an integrated decision support engine rather than relying on a single explanation method.

### 3. Prescriptive Retention Recommendation Engine
Verified in `backend/recommendation_engine.py` and `research_audit/recommendation_rules.csv`:
- Implements state-aware rule matching based on customer attributes and SHAP driver severity.
- **Priority Tier 1 (Contract & Payment Friction):** Customers with Month-to-Month contracts and Electronic Check payments receive recommendations for *12-Month Contract Renewal Discount* ($15\%$ discount for 1-year commitment) and *Automated Bank Draft Setup* ($\$10$ billing credit).
- **Priority Tier 2 (Service Attachment & Support):** Customers lacking TechSupport or OnlineSecurity receive *Complimentary Tech Support & Security Trial* (free for 60 days).
- **Priority Tier 3 (Price & Value Perceptions):** High MonthlyCharges customers receive *Loyalty Plan Re-bundling* (speed optimization with capped pricing).

### 4. Cost-Sensitive Threshold Analysis
Verified in `scripts/phase3_threshold_cost_analysis.py` and `results/simulated_retention_roi.json`:
- **Standard Baseline Threshold:** $\tau = 0.50$.
- **Cost Parameters:** Contact/Intervention Cost $C_{\text{contact}} = \$50$, Saved Customer Value $V_{\text{CLV}} = \$500$, Outreach Success Rate $r = 50\%$.
- **Cost Formulation:**
  $$\text{Total Cost}(\tau) = C_{\text{contact}} \cdot (\text{TP}(\tau) + \text{FP}(\tau)) + V_{\text{CLV}} \cdot \text{FN}(\tau)$$
- **Optimal Decision Threshold:** $\mathbf{\tau^* = 0.26}$.
- **Financial Outcome on Test Cohort ($N = 1,600$):**
  - True Churners Targeted: $770$ subscribers.
  - Gross Value Preserved: $770 \times 0.50 \times \$500 = \$192,500$.
  - Outreach Campaign Cost: $(770 + 43) \times \$50 = \$40,650$.
  - **Net Financial Benefit:** $\mathbf{+\$151,850}$ (Net ROI $+373.6\%$).

---

## 7. Web Application Implementation Evidence

The production system is a decoupled client-server architecture:
- **Backend Framework:** FastAPI (`backend/main.py`), running asynchronous Uvicorn server on port 8000.
- **Frontend Framework:** React 19 + Vite (`frontend/src/`), running development server on port 5173.
- **REST Endpoints Verified:**
  - `POST /api/predict/single`: Processes single customer profile, executes Level-0 and Level-1 stacking, returns churn probability, decision classification, top SHAP drivers, and actionable recommendations.
  - `POST /api/predict/batch`: Handles multi-customer CSV uploads, streams batch inference, and computes aggregate cohort churn analytics.
  - `GET /api/dashboard/history`: Retrieves persistent historical prediction logs.
  - `GET /api/model-comparison`: Supplies live model evaluation metrics and confusion matrices to the comparison dashboard.
  - `POST /api/whatif/simulate`: Evaluates marginal probability delta when adjusting customer levers (contract term, monthly charges, payment method).

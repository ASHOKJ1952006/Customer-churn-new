# Phase 1: Comprehensive Data & Feature Specification Audit

**Status:** Completed  
**Dataset Analyzed:** `data/balanced_telco_churn.csv` (Synthetic Benchmark) & `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (Real IBM Telco Benchmark)  
**Evaluator:** Senior ML Research Engineer & IEEE Paper Editor  

---

## 1. Formal Data Audit (Items A – R)

### A. Dataset Name
*   **Primary Experimental Dataset:** `balanced_telco_churn.csv` (Domain-consistent balanced synthetic telecommunication churn benchmark).
*   **Real Public Validation Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Canonical IBM Telco Customer Churn dataset).

### B. Dataset Path
*   Synthetic Dataset: `d:\ABI\YEAR 4\SEM 7\churn\Customer-churn-new\data\balanced_telco_churn.csv`
*   Real IBM Telco Dataset: `d:\ABI\YEAR 4\SEM 7\churn\Customer-churn-new\data\WA_Fn-UseC_-Telco-Customer-Churn.csv`
*   Train Split (Synthetic): `d:\ABI\YEAR 4\SEM 7\churn\Customer-churn-new\data\train_original.csv`
*   Test Split (Synthetic): `d:\ABI\YEAR 4\SEM 7\churn\Customer-churn-new\data\test_original.csv`

### C. Number of Rows
*   Synthetic Dataset: Exactly **8,000** customer account records.
*   Synthetic Train Split: **6,400** records (80.0%).
*   Synthetic Test Split: **1,600** records (20.0%).
*   Real IBM Telco Dataset: **7,043** records.

### D. Number of Columns
*   Synthetic Dataset: **22 columns** (20 predictor attributes + `Churn` string label + `target` integer label).
*   Real IBM Telco Dataset: **21 columns** (20 predictor attributes + `Churn` string label).

### E. Target Column
*   `target`: Binary integer indicator where $1 = \text{Churn}$ (positive class) and $0 = \text{Loyal / Non-Churn}$ (negative class).
*   `Churn`: String equivalent (`'Yes'` for 1, `'No'` for 0).

### F. Class Distribution
*   **Synthetic Dataset (`balanced_telco_churn.csv`):** Exactly 50.0% / 50.0% balance:
    *   Class 0 (Loyal): 4,000 instances (50.0%)
    *   Class 1 (Churn): 4,000 instances (50.0%)
*   **Synthetic Train Split (`train_original.csv`):**
    *   Class 0: 3,200 instances (50.0%)
    *   Class 1: 3,200 instances (50.0%)
*   **Synthetic Test Split (`test_original.csv`):**
    *   Class 0: 800 instances (50.0%)
    *   Class 1: 800 instances (50.0%)
*   **Real IBM Telco Dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`):**
    *   Class 0 (Loyal): 5,174 instances (73.46%)
    *   Class 1 (Churn): 1,869 instances (26.54%)

### G. Missing Values
*   Synthetic Dataset: **0 null / NaN cells** across all 22 columns.
*   Real IBM Telco Dataset: 11 blank / whitespace strings in `TotalCharges` (for accounts with `tenure == 0`), coerced to `0.0`.

### H. Raw Feature List (20 Tabular Predictors in CSV)
1. `customerID` (Account Identifier, dropped prior to model input)
2. `gender` (Categorical: Male, Female)
3. `SeniorCitizen` (Binary numeric: 0, 1)
4. `Partner` (Binary categorical: Yes, No)
5. `Dependents` (Binary categorical: Yes, No)
6. `tenure` (Numeric integer: months with carrier)
7. `PhoneService` (Binary categorical: Yes, No)
8. `MultipleLines` (Categorical: Yes, No, No phone service)
9. `InternetService` (Categorical: DSL, Fiber optic, No)
10. `OnlineSecurity` (Categorical: Yes, No, No internet service)
11. `OnlineBackup` (Categorical: Yes, No, No internet service)
12. `DeviceProtection` (Categorical: Yes, No, No internet service)
13. `TechSupport` (Categorical: Yes, No, No internet service)
14. `StreamingTV` (Categorical: Yes, No, No internet service)
15. `StreamingMovies` (Categorical: Yes, No, No internet service)
16. `Contract` (Categorical: Month-to-month, One year, Two year)
17. `PaperlessBilling` (Binary categorical: Yes, No)
18. `PaymentMethod` (Categorical: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
19. `MonthlyCharges` (Continuous float: monthly bill amount in USD)
20. `TotalCharges` (Continuous float: cumulative lifetime amount billed in USD)

### I. Six Production Input Fields
The operational production system, user interface form, and REST API intake strictly **6 raw fields** to minimize customer entry latency and prevent data collection burden:
1. `tenure` (Numeric: integer months, 0–72)
2. `MonthlyCharges` (Numeric: monthly bill in USD)
3. `TotalCharges` (Numeric: lifetime bill in USD)
4. `Contract` (Categorical: 'Month-to-month', 'One year', 'Two year')
5. `InternetService` (Categorical: 'DSL', 'Fiber optic', 'No')
6. `PaymentMethod` (Categorical: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')

### J. Engineered Feature List (Derived from the 6 Inputs)
From the 6 production inputs, the deterministic feature engineering engine computes:
*   *Spend Dynamics:*
    *   `AverageMonthlySpend`: $\text{TotalCharges} / \max(\text{tenure}, 1)$
    *   `ChargeIncrease`: $\text{MonthlyCharges} - \text{AverageMonthlySpend}$ (bill shock detector)
    *   `ChargeIncreaseRatio`: $\text{ChargeIncrease} / \max(\text{AverageMonthlySpend}, 10^{-6})$
*   *Non-linear Transformations:*
    *   `TenureSquared`: $\text{tenure}^2$ (diminishing return on loyalty)
    *   `TenureLog`: $\ln(1 + \text{tenure})$ (stabilizes early-tenure variance)
*   *Continuous Multiplicative Interactions:*
    *   `tenure_x_MonthlyCharges`: $\text{tenure} \times \text{MonthlyCharges}$
    *   `tenure_x_IsMonthToMonth`: $\text{tenure} \times \mathbb{I}(\text{Contract} = \text{M2M})$
    *   `MonthlyCharges_x_IsMonthToMonth`: $\text{MonthlyCharges} \times \mathbb{I}(\text{Contract} = \text{M2M})$
*   *Binary Risk Flags:*
    *   `IsNewCustomer`: $\mathbb{I}(\text{tenure} \le 12)$
    *   `IsLongTermCustomer`: $\mathbb{I}(\text{tenure} \ge 48)$
    *   `IsMonthToMonth`: $\mathbb{I}(\text{Contract} = \text{'Month-to-month'})$
    *   `IsLongTermContract`: $\mathbb{I}(\text{Contract} \in \{\text{'One year'}, \text{'Two year'}\})$
    *   `IsElectronicCheck`: $\mathbb{I}(\text{PaymentMethod} = \text{'Electronic check'})$
    *   `HighRiskCombo`: $\mathbb{I}(\text{IsMonthToMonth} = 1 \land \text{IsElectronicCheck} = 1)$
    *   `VeryHighRisk`: $\mathbb{I}(\text{HighRiskCombo} = 1 \land \text{tenure} \le 12)$
    *   `FiberOptic_x_MonthToMonth`: $\mathbb{I}(\text{InternetService} = \text{'Fiber optic'} \land \text{IsMonthToMonth} = 1)$
    *   `FiberOptic_x_ElectronicCheck`: $\mathbb{I}(\text{InternetService} = \text{'Fiber optic'} \land \text{IsElectronicCheck} = 1)$

### K. Categorical Feature List (Before One-Hot Encoding)
The 3 categorical attributes among the 6 production inputs:
1. `Contract` (3 unique levels: 'Month-to-month', 'One year', 'Two year')
2. `InternetService` (3 unique levels: 'DSL', 'Fiber optic', 'No')
3. `PaymentMethod` (4 unique levels: 'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')

### L. Numerical Feature List (Continuous Columns Requiring Scaling)
Exactly **11 continuous features** are normalized via `StandardScaler` (verified in `models/churn_v2_scaler.pkl`):
1. `tenure`
2. `MonthlyCharges`
3. `TotalCharges`
4. `AverageMonthlySpend`
5. `ChargeIncrease`
6. `ChargeIncreaseRatio`
7. `TenureSquared`
8. `TenureLog`
9. `tenure_x_MonthlyCharges`
10. `tenure_x_IsMonthToMonth`
11. `MonthlyCharges_x_IsMonthToMonth`

### M. One-Hot Encoded Feature Count
The pipeline applies one-hot dummy encoding using reference baselines, retaining **7 dummy indicators**:
1. `Contract_One year`
2. `Contract_Two year`
   *(Reference omitted: `Contract_Month-to-month`, captured by `IsMonthToMonth`)*
3. `InternetService_Fiber optic`
4. `InternetService_No`
   *(Reference omitted: `InternetService_DSL`)*
5. `PaymentMethod_Credit card (automatic)`
6. `PaymentMethod_Electronic check`
7. `PaymentMethod_Mailed check`
   *(Reference omitted: `PaymentMethod_Bank transfer (automatic)`)*

### N. Final Transformed Feature Count
**EXACTLY 27 DIMENSIONS**.  
Formulation:
$$\text{Total Transformed Dimensions} = \underbrace{11}_{\text{Continuous Scaled}} + \underbrace{9}_{\text{Binary Risk Flags}} + \underbrace{7}_{\text{One-Hot Dummies}} = 27$$
Confirmed by:
*   `models/churn_v2_feature_columns.pkl`: exactly 27 entries.
*   `models/churn_v2_model.pkl`: `n_features_in_ = 27` for RF, XGBoost, and LightGBM.
*   `models/churn_v2_scaler.pkl`: `n_features_in_ = 11`.

### O. Train / Test Split
*   Stratified train/test partition: **80% Train ($N=6,400$)** and **20% Test ($N=1,600$)**, stratified on `target` with `random_state = 42`.
*   Class counts in Train: 3,200 churners (50.0%), 3,200 loyal accounts (50.0%).
*   Class counts in Test: 800 churners (50.0%), 800 loyal accounts (50.0%).

### P. Balancing Method and When It Was Applied
*   **Balancing Procedure:** Synthesis-level exact 50/50 generation.
*   **Timing:** Applied *globally* during synthetic data generation (`generate_balanced_telco_data(n_samples=8000)` generates 4,000 loyal records and 4,000 churn records before partitioning).
*   **Scientific Caveat:** Because balancing was applied prior to splitting, both train and test sets have 50.0% churn prevalence. Real telecom churn is typically 1.5%–3.0% monthly and ~20%–26% annually. Evaluating solely on a 50/50 test set artificially inflates Precision and PR-AUC.

### Q. Potential Leakage Risks
1.  **Imputation & Feature Derivation:** Fully row-independent mathematical derivations (e.g. `TotalCharges / tenure`); zero inter-row data leakage.
2.  **StandardScaler Fitting:** In `train_churn_v2.py`, `StandardScaler` was fit on `X_train` and applied to `X_test`. However, during 5-fold CV in that script, scaling was applied to `X_train` prior to CV fold iteration rather than inside each CV fold. For strict leakage-free evaluation in Phase 2, scaling must be encapsulated inside an `sklearn.pipeline.Pipeline` or `ColumnTransformer`.
3.  **Threshold Optimization:** The threshold was swept over out-of-fold (OOF) cross-validation predictions on the training set, avoiding direct test-set leakage.

### R. Exact Dataset Used for Each Reported Metric
1.  **Paper Table I, III, IV, V, VI, VII and Figs. 2, 3, 4, 6, 7:**
    *   Evaluated strictly on the **synthetic test set (`test_original.csv`, $N=1,600$)**.
    *   Models evaluated at individual tuned operating points: Stacking ($\tau^*=0.26$), XGBoost ($\tau^*=0.29$), LightGBM ($\tau^*=0.32$), Random Forest ($\tau^*=0.38$).
2.  **Earlier/Baseline Evaluation (`results/model_comparison.json`):**
    *   Evaluated on the **real IBM Telco test set ($N=1,409$, 26.5% churn prevalence)** at default symmetric threshold $\tau=0.50$.
    *   Yielded Accuracy 76.2%–77.4% and F1 61.6%–64.0%.
    *   *Critical Requirement:* These two sets of numbers must never be conflated; the paper must clearly distinguish synthetic benchmark results from real-world IBM Telco validation results.

---

## 2. Empirical Verification Results (Phase 1 Benchmarks)

### 2.1 Synthetic Label Determinism & 5% Noise Ceiling Analysis
*   A simple single Decision Tree (depth=5) trained on **only 3 core features** (`tenure`, `IsMonthToMonth`, `IsElectronicCheck`) achieves **92.84% accuracy**.
*   A Decision Tree on all 27 features achieves **94.51% accuracy**.
*   `generate_balanced_dataset.py` explicitly injects `noise_rate = 0.05` (5.0% trait swapping for both loyal and churn accounts).
*   **Conclusion:** The synthetic dataset has a theoretical Bayes error rate of $\approx 5.0\%$, creating an artificial noise ceiling of $\approx 95.0\%$. The models' performance (94.9%–95.4%) confirms they are hitting this synthetic ceiling.

### 2.2 Synthetic Test Set Performance: Tuned vs. Default $\tau=0.50$ ($N=1,600$)

| Model | Operating $\tau$ | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.38 | 95.38% | 94.59% | 96.25% | 95.42% | 98.69% | 98.31% | 0.0369 |
| *(RF default)* | 0.50 | 95.50% | 95.96% | 95.00% | 95.48% | 98.69% | 98.31% | 0.0369 |
| **XGBoost** | 0.29 | 94.88% | 92.54% | **97.62%** | 95.01% | **99.01%** | **98.90%** | **0.0347** |
| *(XGB default)* | 0.50 | 95.19% | 95.82% | 94.50% | 95.15% | 99.01% | 98.90% | 0.0347 |
| **LightGBM** | 0.32 | 95.00% | 93.37% | 96.88% | 95.09% | 98.95% | 98.82% | 0.0353 |
| *(LGBM default)* | 0.50 | 95.38% | 95.60% | 95.12% | 95.36% | 98.95% | 98.82% | 0.0353 |
| **Stacking Ensemble** | 0.26 | **95.44%** | 94.71% | 96.25% | **95.47%** | 98.94% | 98.80% | 0.0362 |
| *(Stack default)* | 0.50 | **95.44%** | **95.61%** | 95.25% | 95.43% | 98.94% | 98.80% | 0.0362 |

### 2.3 Evaluation at Realistic Churn Prevalences (Stacking Ensemble at $\tau=0.26$)
Evaluated across 50 repeated stratified subsamples of the test set:
*   **At 50.0% Prevalence (Balanced Test Set):** Accuracy 95.44%, Precision 94.71%, Recall 96.25%, F1 95.47%, PR-AUC 98.80%
*   **At 26.5% Prevalence (Real IBM Telco Baseline):** Accuracy 95.11% $\pm$ 0.27%, Precision 86.63% $\pm$ 0.12%, Recall 96.44% $\pm$ 1.03%, F1 91.27% $\pm$ 0.53%, PR-AUC 96.94% $\pm$ 0.32%
*   **At 15.0% Prevalence (Annual Industry Attrition):** Accuracy 94.87% $\pm$ 0.21%, Precision 75.93% $\pm$ 0.27%, Recall 96.23% $\pm$ 1.40%, F1 84.88% $\pm$ 0.71%, PR-AUC 94.29% $\pm$ 0.74%
*   **At 5.0% Prevalence (Quarterly Low-Risk Cohort):** Accuracy 94.70% $\pm$ 0.18%, Precision 48.41% $\pm$ 0.94%, Recall 96.14% $\pm$ 3.53%, F1 64.39% $\pm$ 1.62%, PR-AUC 85.66% $\pm$ 2.76%

*Insight:* While Recall remains robust ($\approx 96.1\%$) across all prevalence levels, Precision drops from 94.7% at balanced prevalence to 48.4% at 5% prevalence, demonstrating why fixed operational thresholds must be calibrated against real-world base rates.

### 2.4 Cross-Dataset Evaluation on Real IBM Telco Data ($N=7,043$)
Evaluating the synthetic-trained models directly on the real IBM Telco dataset under identical 6-to-27 feature mapping:

| Model | Operating $\tau$ | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.38 | 69.49% | 45.02% | **67.74%** | **54.09%** | **76.48%** | **54.36%** | **0.2192** |
| *(RF default)* | 0.50 | 71.69% | 47.50% | 63.46% | 54.33% | 76.48% | 54.36% | 0.2192 |
| **XGBoost** | 0.29 | 71.11% | 46.58% | 60.57% | 52.66% | 75.66% | 53.65% | 0.2216 |
| *(XGB default)* | 0.50 | **73.41%** | **49.90%** | 54.57% | 52.13% | 75.66% | 53.65% | 0.2216 |
| **LightGBM** | 0.32 | 71.15% | 46.45% | 57.14% | 51.25% | 73.73% | 51.94% | 0.2323 |
| *(LGBM default)* | 0.50 | 72.80% | 48.85% | 53.34% | 51.00% | 73.73% | 51.94% | 0.2323 |
| **Stacking Ensemble** | 0.26 | 71.42% | 46.98% | 60.03% | 52.71% | 76.19% | 53.95% | 0.2286 |
| *(Stack default)* | 0.50 | 73.32% | 49.76% | 55.70% | 52.56% | 76.19% | 53.95% | 0.2286 |

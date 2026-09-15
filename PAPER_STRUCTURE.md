# Paper Structure: Customer Churn Prediction for Indian Telecom

## Abstract
- Problem: Customer churn prediction for Indian telecom providers
- Approach: Stacked ensemble with SMOTE, feature engineering, calibration
- Key contribution: Recall-focused evaluation (not just accuracy), dual explainability (SHAP + LIME)
- Results: Dataset 1 (Telco): Recall 0.810, Accuracy 0.750, F1 0.633; Dataset 2 (Indian): Recall 0.676, Accuracy 0.933, F1 0.628
- Significance: Practical retention-focused model with working demo application

## 1. Introduction
- Background: Telecom churn as critical business problem
- Indian telecom context: Airtel/Jio/Vi competitive landscape
- Gap in literature: Most papers focus on accuracy, not recall; lack of explainability
- Research questions:
  1. Can stacked ensembles with proper SMOTE improve recall over baseline?
  2. Does behavioral feature engineering help on time-series data?
  3. Can calibration improve probability reliability without sacrificing recall?
- Contributions:
  - Recall-focused methodology
  - Leakage-safe SMOTE implementation
  - Behavioral feature engineering for time-series data
  - CV-based calibration preserving recall
  - Dual explainability (SHAP + LIME)
  - Working demo application with persistent storage

## 2. Related Work
- Base paper: Maan & Maan (2023) - plain XGBoost, 96.2% accuracy (likely inflated)
- Comparison with other churn prediction papers
- Ensemble methods in churn prediction
- SMOTE applications in imbalanced classification
- Explainability in ML (SHAP, LIME)
- Gap: Most papers don't address recall, calibration, or provide working demos

## 3. Methodology

### 3.1 Datasets
- **Dataset 1**: Telco-Customer-Churn (Kaggle/IBM)
  - 7,043 rows, reframed for Indian telecom context
  - Features: tenure, contract type, payment method, services, demographics
  - Churn rate: ~26.5%
  - Limitations: Static snapshot, no time-series structure

- **Dataset 2**: High-Value Customer Churn (Indian/SE Asian telecom)
  - 99,999 raw rows → 30,001 high-value customers (top 30% by recharge)
  - 4-month usage history (months 6-8 active, month 9 churn phase)
  - Churn label: Zero usage in month 9
  - Churn rate: 8.14% (severe imbalance)
  - Features: Call minutes, data usage, recharge amounts across months
  - Caution: 91.9% trivial baseline (always predict no churn)

### 3.2 Data Preprocessing
- Stratified 80/20 train/test split
- Missing value handling
- Feature scaling (StandardScaler)
- **Critical**: All preprocessing AFTER train/test split to prevent leakage

### 3.3 Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique)
- **Leakage-safe implementation**: SMOTE inside CV folds via imblearn Pipeline
- Earlier bug: SMOTE applied before CV → inflated F1 from 0.58 to 0.855 (fake)
- Current approach: SMOTE per-fold only on training data

### 3.4 Feature Engineering
- **Dataset 1** (6 engineered features):
  - AvgMonthlySpend: Average monthly charges
  - ChargeIncrease: Recent bill increase flag
  - NumServices: Total services subscribed
  - TenureGroup: Tenure buckets
  - HighRiskCombo: Month-to-month + electronic check interaction
  - IsNewCustomer: New customer flag (< 6 months tenure)
  - Impact: Recall improved 0.794 → 0.821

- **Dataset 2** (behavioral change features):
  - CallUsageDrop_7to8: Call usage decline from month 7 to 8
  - DataUsageDrop_7to8: Data usage decline from month 7 to 8
  - RechargeConsistency: Recharge amount stability (std/mean ratio)
  - WentQuiet: Zero usage in month 8 (extending fb_user_8 signal)
  - Trend features: Month 8 - Month 6 for all base features
  - Impact: Recall improved 0.613 → 0.6865

### 3.5 Model Architecture
- **Base learners**:
  - Random Forest (CPU, n_jobs=-1)
  - XGBoost (GPU-accelerated, tree_method='hist', device='cuda')
  - LightGBM (CPU, per GPU guidelines)
- **Meta-learner**: Logistic Regression
- **Stacking**: 5-fold CV, SMOTE inside each fold
- **GPU setup**: NVIDIA RTX 5050, CUDA 13.2, 1.90x speedup confirmed

### 3.6 Hyperparameter Tuning
- RandomizedSearchCV (scikit-learn)
- Optuna (TPE sampler)
- Compared honestly, kept whichever generalized better
- **Dataset 2 overfitting episode**: Unconstrained Optuna → severe overfitting
- Fix: Tightened Optuna search space bounds

### 3.7 Probability Calibration
- Problem: Raw stacked ensemble probabilities overconfident
  - Predicted 75% churn probability → only 41% actual churn rate
- Solution: CV-based calibration (CalibratedClassifierCV)
- **Critical**: Calibration per-fold, NOT on held-out split (cost recall)
- Result: Brier score improved 0.1554 → 0.1366, recall preserved (0.821 → 0.810)

### 3.8 Threshold Optimization
- Precision-recall curve to find optimal threshold
- Not default 0.5
- F1-maximizing threshold selected
- Applied consistently across all evaluations

### 3.9 Explainability
- **SHAP** (SHapley Additive exPlanations):
  - Global feature importance
  - Local explanation per customer
  - TreeExplainer for tree-based models
- **LIME** (Local Interpretable Model-agnostic Explanations):
  - Local approximations
  - Cross-validated against SHAP for agreement
- **Actionable recommendations**: Rule-based layer mapping SHAP output to retention actions

### 3.10 Evaluation Metrics
- **Primary**: Recall on churn class (catching real churners)
- Secondary: Accuracy, Precision, F1, Brier score (calibration quality)
- **Overfitting check**: Train vs test F1 gap on original unbalanced distribution
- **Honest comparison**: Never compare SMOTE-balanced train against unbalanced test

## 4. Results

### 4.1 Dataset 1 (Telco) Results

| Approach | Accuracy | Precision | Recall | F1 | Brier Score |
|----------|----------|-----------|--------|-----|-------------|
| Baseline (plain XGBoost) | 0.746 | 0.519 | 0.794 | 0.629 | 0.1554 |
| + SMOTE (leakage-safe) | 0.750 | 0.519 | 0.794 | 0.633 | N/A |
| + Stacked Ensemble | 0.751 | 0.518 | 0.794 | 0.634 | N/A |
| + Feature Engineering | 0.750 | 0.519 | **0.821** | 0.633 | N/A |
| + CV Calibration | **0.750** | **0.519** | **0.810** | **0.633** | **0.1366** |

**Key findings**:
- Feature engineering improved recall by 2.7 percentage points
- Calibration improved Brier score by 0.0188 with negligible recall change
- Overfitting gap: ~0.05 (healthy)
- HighRiskCombo ranked #2 in feature importance

### 4.2 Dataset 2 (Indian Telecom) Results

| Approach | Accuracy | Precision | Recall | F1 | Brier Score | Features | Overfit Gap |
|----------|----------|-----------|--------|-----|-------------|----------|-------------|
| Baseline (trivial) | 0.919 | N/A | 0.000 | N/A | N/A | - | - |
| Regularized Ensemble | 0.943 | 0.657 | 0.613 | 0.634 | N/A | 315 | 0.1082 |
| + Feature Engineering | 0.933 | 0.5736 | **0.6865** | 0.625 | N/A | 315 | 0.1082 |
| + Feature Selection (top 100) | **0.933** | **0.586** | **0.676** | **0.628** | **0.0446** | 100 | 0.0895 |

**Key findings**:
- Behavioral feature engineering improved recall by 7.35 percentage points
- Feature selection reduced overfitting gap from 0.1082 to 0.0895 (below 0.10 threshold)
- Final model uses top 100 features by importance
- **Ablation note**: The 315-feature version is retained in this table to demonstrate why feature selection was necessary (overfitting gap > 0.10), but is not the final deployed model
- **Caution**: 91.9% trivial baseline makes accuracy misleading

### 4.3 Ablation Study

| Experiment | Recall | Decision |
|------------|--------|----------|
| Baseline | 0.794 | - |
| + CatBoost (4th learner) | 0.751 | Rejected (recall drop) |
| + Feature Engineering | 0.821 | Adopted |
| + Calibration | 0.810 | Adopted (better calibration) |

### 4.4 Cross-Dataset Comparison

| Metric | Dataset 1 | Dataset 2 | Notes |
|--------|-----------|-----------|-------|
| Accuracy | 0.750 | 0.933 | Dataset 2 inflated by imbalance |
| Recall | 0.810 | 0.676 | Dataset 1 better for churn detection |
| Precision | 0.519 | 0.586 | Dataset 2 higher precision |
| F1 | 0.633 | 0.628 | Comparable |
| Brier Score | 0.1366 | 0.0446 | Dataset 2 better calibrated |
| Churn rate | 26.5% | 8.14% | Different class distributions |

**Interpretation**: Dataset 2's higher accuracy is due to severe class imbalance, not better model performance. Recall is the more honest comparison. Dataset 1 achieves higher recall (81% vs 67.6%), making it more effective for churn detection despite lower accuracy.

## 5. Discussion

### 5.1 Accuracy vs Recall Trade-off
- Base paper's 96.2% accuracy likely inflated by leakage
- Honest ceiling for Dataset 1: ~78-82% accuracy
- **Recall is the business-relevant metric**: Missing a real churner costs more than a false alarm
- Our approach prioritizes recall (0.810) over accuracy (0.750)

### 5.2 Dataset 2's Trivial Baseline Problem
- 91.9% accuracy by always predicting no churn
- Our 93.3% accuracy is only 1.4% improvement over trivial
- **Accuracy alone is misleading** for imbalanced datasets
- Recall (68.65%) is the honest performance measure

### 5.3 Feature Engineering Impact
- Dataset 1: Static features (HighRiskCombo) helped
- Dataset 2: Behavioral change features (usage drops, consistency) helped
- **Time-series structure enables richer features** (Dataset 2 advantage)

### 5.4 Calibration Value
- Raw probabilities were overconfident
- Calibration improved Brier score without sacrificing recall
- **Better probability estimates = better business decisions**

### 5.5 GPU Acceleration
- XGBoost with NVIDIA RTX 5050: 1.90x speedup
- **Only XGBoost benefits**: LightGBM/CatBoost on CPU (setup risk vs benefit)
- **Critical**: Force GPU-trained models to CPU for inference (Streamlit stability)

### 5.6 Limitations
- Dataset 1: Static snapshot, no time-series structure
- Dataset 2: Severe class imbalance, derived churn label
- Both datasets: Limited geographic scope
- Model complexity: Stacking requires more computational resources

### 5.7 Fairness Audit Findings

A fairness audit was conducted across demographic groups (gender and senior citizen status) on Dataset 1 using the final calibrated model. The audit revealed a concerning disparity for senior citizens:

**Gender Analysis** (No concerning disparities):
- Disparate impact ratio: 1.0023 (meets 80% rule)
- Recall ratio (Female/Male): 0.9728
- All fairness ratios within acceptable range

**Senior Citizen Analysis** (Bias detected):
- Disparate impact ratio: 0.4794 (violates 80% rule)
- Recall for seniors: 87.8% vs. 67.6% for non-seniors (20 percentage point difference)
- False positive rate for seniors: 45.2% vs. 19.4% for non-seniors (25.7 percentage point difference)
- The model is more likely to predict seniors as churners

**Trade-off interpretation**: The higher recall for seniors means the model catches more actual senior churners (positive), but the higher false positive rate means more non-churning seniors are flagged for retention offers (negative). This disparate impact (0.4794) fails the 80% rule commonly used in fairness assessments.

**Limitation**: This finding is documented as a limitation worth flagging, not a fully diagnosed root cause. The disparity may stem from feature correlations (e.g., tenure patterns, service usage) rather than direct bias. Future work should include dedicated fairness-aware re-training or per-group threshold studies to address this systematically. The current model remains unchanged for the primary results reported in this paper.

### 5.8 Alternative Deployment Considerations: Cost-Sensitive Threshold

The primary results reported in this paper use an F1-maximizing threshold (0.2428) selected via precision-recall curve analysis. However, an alternative deployment lens considers business costs directly: the cost of missing a churner (false negative) versus the cost of a false alarm (false positive).

**Assumed cost parameters** (for illustration only):
- False negative cost: $500 (lost customer lifetime value)
- False positive cost: $50 (retention offer sent to non-churner)
- Cost ratio (FN/FP): 10x
- True positive cost: $50 (retention offer sent to churner who stays)
- True negative cost: $0 (no action, customer stays)

**Threshold comparison**:

| Metric | F1-Optimal (0.2428) | Cost-Optimal (0.0951) | Change |
|--------|---------------------|----------------------|--------|
| Accuracy | 0.7658 | 0.6210 | -0.1448 |
| Precision | 0.5438 | 0.4085 | -0.1353 |
| **Recall** | 0.7299 | **0.9545** | +0.2246 |
| F1 | 0.6233 | 0.5721 | -0.0512 |
| **Expected Cost** | $75,600 | **$52,200** | -$23,400 |

**Operational trade-off**: The cost-optimal threshold (0.0951) catches nearly all churners (95.5% recall) but at the cost of contacting roughly half of all non-churning customers (49% false positive rate). This is a significant operational burden: retention teams would need to contact twice as many customers compared to the F1-optimal threshold.

**Limitations**: The cost parameters above are assumptions, not established facts. A real deployment would need these ratios validated against actual business data (e.g., customer lifetime value, retention offer costs, conversion rates) before adopting the cost-optimal threshold operationally. The sensitivity analysis shows that higher cost ratios favor even lower thresholds, but the operational trade-off remains.

**Conclusion**: The cost-sensitive analysis is presented as an alternate lens for business decision-making, not a replacement for the primary F1-optimal threshold used in the live demo and primary results. The choice between thresholds depends on business priorities: maximize churn detection (cost-optimal) vs. balance precision and operational efficiency (F1-optimal).

## 6. Demo Application

### 6.1 Features
- **Tab 1**: Single customer prediction with SHAP explanations
- **Tab 2**: Batch CSV upload with risk-ranked downloadable report
- **Tab 3**: What-if simulator (change one factor, see risk change)
- **Tab 4**: Dashboard with persistent SQLite storage
  - Prediction history
  - Risk queue for follow-up
  - Batch upload history
  - Data management

### 6.2 Technical Implementation
- Streamlit framework
- SQLite-backed persistence (ChurnDatabase class)
- GPU-trained models forced to CPU for inference stability
- SHAP integration for explainability
- Actionable recommendations layer

## 7. Conclusion and Future Work

### 7.1 Contributions
- Recall-focused methodology (not accuracy-chasing)
- Leakage-safe SMOTE implementation
- Behavioral feature engineering for time-series data
- CV-based calibration preserving recall
- Dual explainability (SHAP + LIME)
- Working demo application with persistent storage

### 7.2 Future Work
- **Survival analysis**: Model time-to-churn instead of binary churn
- **Uplift modeling**: Model response to retention offers
- **Cost-sensitive threshold selection**: Business objective-driven thresholds
- **Fairness audit**: Check bias across demographic groups
- **Natural-language explanations**: Convert SHAP to plain English

### 7.3 Practical Impact
- Model ready for deployment in telecom retention teams
- Dashboard enables risk queue management
- Explainability builds trust with business stakeholders
- Recall-focused approach catches more real churners

## References
- Maan & Maan (2023) - Base paper
- XGBoost documentation
- SMOTE paper (Chawla et al., 2002)
- SHAP paper (Lundberg & Lee, 2017)
- LIME paper (Ribeiro et al., 2016)
- Calibration methods (Platt, 1999; Zadrozny & Elkan, 2001)

## Appendix
- A: Full hyperparameter configurations
- B: Feature importance rankings
- C: Calibration curves
- D: SHAP summary plots
- E: GPU setup guidelines
- F: Database schema

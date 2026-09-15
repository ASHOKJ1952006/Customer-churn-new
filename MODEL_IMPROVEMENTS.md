# Model Improvements Summary

## 1. Cost-Sensitive Threshold Selection

### Business Cost Parameters
- **False Positive Cost**: $50 (retention offer to non-churner)
- **False Negative Cost**: $500 (losing a customer)
- **True Positive Cost**: $50 (retention offer to saved customer)
- **True Negative Cost**: $0 (no action, customer stays)
- **Cost Ratio (FN/FP)**: 10x (losing a customer is 10x more expensive than false alarm)

### Results Comparison

| Metric | F1-Optimal (0.2428) | Cost-Optimal (0.0951) | Change |
|--------|---------------------|----------------------|--------|
| Accuracy | 0.7658 | 0.6210 | -0.1448 |
| Precision | 0.5438 | 0.4085 | -0.1353 |
| **Recall** | 0.7299 | **0.9545** | +0.2246 |
| F1 | 0.6233 | 0.5721 | -0.0512 |
| **Expected Cost** | $75,600 | **$52,200** | -$23,400 |

### Key Findings
- **Cost-optimal threshold**: 0.0951 (much lower than F1-optimal 0.2428)
- **Recall improvement**: 22.5 percentage points (73% → 95%)
- **Cost reduction**: $23,400 per 1,409 customers ($16.61 per customer)
- **Trade-off**: Lower accuracy and precision, but business-optimal

### Sensitivity Analysis

| Cost Ratio (FN/FP) | Threshold | Recall | Expected Cost |
|-------------------|-----------|--------|---------------|
| 5x | 0.1137 | 0.9251 | $47,650 |
| 10x | 0.0951 | 0.9545 | $52,200 |
| 15x | 0.0675 | 0.9786 | $56,000 |
| 20x | 0.0517 | 0.9947 | $57,050 |
| 30x | 0.0517 | 0.9947 | $58,050 |

**Interpretation**: Higher cost ratios favor even lower thresholds to catch more churners.

---

## 2. Fairness Audit

### Gender Analysis

| Group | Size | Accuracy | Precision | Recall | F1 | PPR | FPR | FNR |
|-------|------|----------|-----------|--------|-----|-----|-----|-----|
| Female | 687 | 0.7642 | 0.5628 | 0.7202 | 0.6318 | 0.3595 | 0.2186 | 0.2798 |
| Male | 722 | 0.7618 | 0.5174 | 0.7403 | 0.6091 | 0.3587 | 0.2311 | 0.2597 |

#### Fairness Ratios (Female / Male)
- **Accuracy Ratio**: 1.0032 (✅ No disparity)
- **Recall Ratio**: 0.9728 (✅ No disparity)
- **FPR Ratio**: 0.9462 (✅ No disparity)
- **FNR Ratio**: 1.0775 (✅ No disparity)
- **Disparate Impact (PPR)**: 1.0023 (✅ Meets 80% rule)

**Conclusion**: No concerning gender disparities found.

---

### Senior Citizen Analysis

| Group | Size | Accuracy | Precision | Recall | F1 | PPR | FPR | FNR |
|-------|------|----------|-----------|--------|-----|-----|-----|-----|
| Non-Senior | 1,187 | 0.7759 | 0.5137 | 0.6775 | 0.5844 | 0.3067 | 0.1943 | 0.3225 |
| Senior | 222 | 0.6937 | 0.6056 | 0.8776 | 0.7167 | 0.6396 | 0.4516 | 0.1224 |

#### Fairness Ratios (Non-Senior / Senior)
- **Accuracy Ratio**: 1.1185 (⚠️ Seniors have lower accuracy)
- **Recall Ratio**: 0.7721 (⚠️ Seniors have higher recall)
- **FPR Ratio**: 0.4302 (⚠️ Seniors have higher FPR)
- **FNR Ratio**: 2.6335 (⚠️ Seniors have lower FNR)
- **Disparate Impact (PPR)**: 0.4794 (⚠️ Violates 80% rule)

#### Concerning Disparities
- **Recall disparity**: 0.20 (seniors have 20% higher recall)
- **FPR disparity**: 0.257 (seniors have 25.7% higher false positive rate)

**Conclusion**: Model shows bias against senior citizens - they are more likely to be predicted as churners (higher PPR, higher FPR, higher recall).

---

## 3. Survival Analysis (Time-to-Churn)

### Overview
Complementary analysis to classification model - answers "WHEN will they churn?" instead of "WILL they churn?". Uses Dataset 1 (Telco) which has natural duration structure (tenure).

### Cox Proportional Hazards Model Results

**C-Index (Concordance)**: 0.8539
- Original (with tenure leakage): 0.9857 (inflated, unrealistic)
- Corrected (tenure removed from covariates): 0.8539 (realistic, excellent)
- Interpretation: Falls within expected 0.70-0.85 range for published churn survival models

**Top Hazard Ratios (Risk-Increasing)**:
- InternetService_Fiber optic: 2.92 (192% higher churn rate)
- PhoneService: 1.86 (86% higher)
- PaymentMethod_Electronic check: 1.73 (73% higher)
- PaperlessBilling: 1.17 (17% higher)

**Top Hazard Ratios (Protective)**:
- Contract_Two year: 0.036 (96% reduction in churn rate)
- Contract_One year: 0.185 (82% reduction)
- InternetService_No: 0.193 (81% reduction)
- Partner: 0.577 (42% reduction)
- OnlineSecurity_Yes: 0.585 (41% reduction)

**Alignment with SHAP**: Hazard ratios confirm SHAP findings - contract types are strongly protective, electronic check payment is risk-increasing.

### Kaplan-Meier Stratifications

**Contract Type** (log-rank p-value: 9.12e-61) - Extremely significant
- Month-to-month: median survival = ∞ (never reaches 50% churn)
- One year: median survival = ∞
- Two year: median survival = ∞
- Interpretation: Dramatic separation, confirming contract is strongest predictor

**HighRiskCombo** (log-rank p-value: 5.07e-254) - Extremely significant
- High Risk (HighRiskCombo=1): median survival = 32 months
- Low Risk (HighRiskCombo=0): median survival = ∞
- Interpretation: High-risk customers churn much faster

### Proportional Hazards Assumption Violations

**9 features violate the proportional hazards assumption**:
- MonthlyCharges (p < 0.005) - Non-linear relationship
- PhoneService (p = 0.0001) - Binary categorical
- Contract_One year (p < 0.005) - Binary categorical
- Contract_Two year (p < 0.005) - Binary categorical
- InternetService_Fiber optic (p = 0.0004) - Binary categorical
- InternetService_No (p = 0.0225) - Binary categorical
- OnlineSecurity_Yes (p = 0.0255) - Binary categorical
- TechSupport_Yes (p = 0.0417) - Binary categorical
- HighRiskCombo (p = 0.0255) - Binary categorical

**Interpretation**: Hazard ratios for these features should be read as approximate/time-varying rather than constant. The violations are common in real-world data and don't invalidate the model, but indicate opportunities for refinement.

**Future refinement**: Cox stratification (fitting separate baseline hazards per subgroup for violating categorical features) is a valid approach if more time is available. This was not implemented in the current analysis due to diminishing returns vs. time cost.

### Data Leakage Fix

**Issue**: Initial C-index of 0.9857 was inflated due to including `tenure` as a covariate (methodological error - tenure is the DURATION variable, not a predictor).

**Fix**: Removed `tenure` and tenure-derived features (`IsNewCustomer`) from Cox model covariates. Tenure is now only used as `duration_col` when fitting the model.

**Result**: C-index dropped from 0.9857 (unrealistic) to 0.8539 (realistic, excellent).

### App Integration

Added median survival time prediction to the What-If Simulator tab as a complementary insight alongside churn probability. The Cox model uses the same customer profile inputs but excludes tenure and IsNewCustomer to avoid leakage.

---

## Recommendations

### 1. Cost-Sensitive Threshold
- **Implement cost-optimal threshold (0.0951)** for business deployment
- **Trade-off**: Accept lower accuracy for significant cost savings
- **Sensitivity**: Adjust threshold based on actual business cost ratios

### 2. Fairness Mitigation
- **Senior citizen bias detected** - requires mitigation
- **Options**:
  - Group-specific thresholds (higher threshold for seniors)
  - Re-weight training data (down-weight senior samples)
  - Add fairness constraints during training
  - Monitor and report disparities in production

### 3. Next Steps
- Update app with cost-sensitive threshold option
- Implement fairness mitigation for senior citizens
- Document fairness monitoring in production

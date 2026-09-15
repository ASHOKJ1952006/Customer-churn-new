# Reduced Feature Mapping - Dataset 1

## Goal
Reduce data entry from ~20 fields to minimal set while maintaining model performance.

## ✅ COMPLETED

## Optimal Feature Set: 8 Engineered Features
Based on comprehensive feature selection analysis (SHAP, RF, RFE, Mutual Information, Correlation):

1. **ChargeIncrease** - MonthlyCharges - AvgMonthlySpend
2. **tenure** - Customer tenure in months
3. **HighRiskCombo** - Contract=Month-to-month AND PaymentMethod=Electronic check
4. **TenureSquared** - tenure²
5. **VeryHighRisk** - Contract=Month-to-month AND PaymentMethod=Electronic check AND tenure≤12
6. **InternetService_Fiber optic** - One-hot encoded
7. **Contract_Two year** - One-hot encoded
8. **PaymentMethod_Electronic check** - One-hot encoded

## Performance
- **F1 Score**: 0.6189 (vs 0.6334 with all features)
- **Performance Drop**: Only 2.3% reduction
- **Simplicity**: 83% reduction in features (47 → 8)
- **Combined Score**: 0.9387 (optimal balance)

## Required User Input Fields (6 fields)
The 8 engineered features can be computed from just 6 basic input fields:

| Field | Type | Options | Notes |
|-------|------|---------|-------|
| **tenure** | Number | 0-72 months | How long customer has been with service |
| **MonthlyCharges** | Number | Currency | Current monthly bill amount |
| **TotalCharges** | Number | Currency | Total amount charged to date |
| **Contract** | Select | Month-to-month, One year, Two year | Contract type |
| **PaymentMethod** | Select | Electronic check, Mailed check, Bank transfer, Credit card | How customer pays |
| **InternetService** | Select | No, DSL, Fiber optic | Internet service type |

## Feature Engineering Logic (Backend)
The backend computes the 8 engineered features from the 6 user inputs:

```python
# From 6 user inputs, compute 8 features:
ChargeIncrease = MonthlyCharges - (TotalCharges / (tenure + 1))
TenureSquared = tenure ** 2
HighRiskCombo = 1 if (Contract == 'Month-to-month' and PaymentMethod == 'Electronic check') else 0
VeryHighRisk = 1 if (Contract == 'Month-to-month' and PaymentMethod == 'Electronic check' and tenure <= 12) else 0
InternetService_Fiber optic = 1 if InternetService == 'Fiber optic' else 0
Contract_Two year = 1 if Contract == 'Two year' else 0
PaymentMethod_Electronic check = 1 if PaymentMethod == 'Electronic check' else 0
```

## Benefits
- **83% reduction** in data entry fields (20 → 6)
- **Minimal performance loss** (2.3% F1 drop)
- **Faster data entry** for users
- **Lower error rate** with fewer fields
- **Better user experience**

## Implementation Status
- ✅ Feature selection analysis complete
- ✅ Performance testing complete
- ✅ Frontend form updated to 6 fields
- ✅ Backend updated to compute engineered features
- ✅ API tested and working correctly

## Files Modified
- `frontend/src/pages/SingleCustomer.jsx` - Reduced form to 6 fields
- `backend/main.py` - Updated `build_feature_vector()` to compute features from 6 inputs
- `scripts/feature_selection_analysis.py` - Feature selection analysis script
- `scripts/test_reduced_features.py` - Performance testing script
- `scripts/test_reduced_api.py` - API testing script

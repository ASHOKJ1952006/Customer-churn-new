# Project Audit: Codebase, Pipelines, Metrics, and Paper Discrepancies

**Date:** September 2026  
**Auditor:** Senior ML Research Engineer & IEEE Paper Editor  
**Paper Under Review:** *Explainable Stacking Ensemble Learning and Decision Support Framework for Telecom Customer Churn Prediction*  
**Target Venue:** IEEE Conference (standard 6–8 page two-column format)

---

## 1. Inventory of Files

### 1.1 Data Files (`/data`)
*   `data/balanced_telco_churn.csv`: 8,000 rows, 22 columns (20 predictors + `Churn` + `target`). Exactly 4,000 churners and 4,000 non-churners. Generated synthetically by `scripts/generate_balanced_dataset.py` with 5% label/trait noise.
*   `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`: 7,043 rows, 21 columns. The canonical public IBM Telco Customer Churn dataset (churn prevalence: 26.54%, 1,869 churners, 5,174 non-churners).
*   `data/train_original.csv` / `data/enhanced_training_data.csv`: 6,400 rows (80% stratified train split of synthetic data).
*   `data/test_original.csv`: 1,600 rows (20% stratified test split of synthetic data).
*   `data/churn_predictions.db`: SQLite database storing logged customer records and API predictions.
*   `data/final_cross_dataset_comparison.csv`: Brief summary comparing dataset metrics.

### 1.2 Training & ML Core (`/ml`, `/scripts`, `/models`)
*   `scripts/generate_balanced_dataset.py`: Synthetic generator script using parameterized normal/exponential distributions and 5% noise injection.
*   `scripts/feature_engineering_v2.py`: The active 27-feature engineering pipeline derived from 6 raw user inputs (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`). Generates 11 numeric features, 9 binary flags, and 7 one-hot categories.
*   `scripts/stacking_model_v2.py`: Custom wrapper class `ChurnV2StackingEnsemble` combining RF, XGBoost, and LightGBM with Logistic Regression meta-learner.
*   `scripts/train_churn_v2.py`: Full training pipeline on synthetic train set. Fits RF, XGBoost, LightGBM; computes 5-fold out-of-fold (OOF) probabilities; fits Logistic Regression meta-learner; sweeps OOF thresholds targeting Acc $\ge$ 90% and Recall $\ge$ 90%; evaluates on `test_original.csv`.
*   `scripts/evaluate_models.py`: CLI evaluation script computing confusion matrices, ROC/PR curves, and threshold sweeps.
*   `scripts/cost_sensitive_threshold.py`: Standalone script for cost analysis with asymmetric penalties ($C_{FN}$, $C_{FP}$).
*   `ml/feature_engineering.py`: Alternative 47-feature engineering pipeline with 16 continuous and multi-categorical features.
*   `ml/train_models.py`: Earlier training pipeline comparing models on the real IBM dataset.
*   `models/churn_v2_*.pkl`: Active production model artifacts (RF, XGB, LGBM, Stacking, Scaler, threshold `0.26`).
*   `models/model_version_info.json`: Production metadata specifying threshold `0.260`, accuracy `0.9544`, recall `0.9625`, F1 `0.9547`, Brier `0.0362`.

### 1.3 Backend & Decision Support (`/backend`)
*   `backend/main.py`: FastAPI server exposing endpoints for single inference (`/predict`), batch inference (`/batch-predict`), what-if counterfactual simulation (`/what-if`), SHAP (`/shap`), LIME (`/lime`), and recommendations (`/recommendations`).
*   `backend/recommendation_engine.py`: `TelecomRetentionDecisionEngine` evaluating customer profile, SHAP values, and LIME weights against business rules. Prunes inapplicable interventions, computes action score $\frac{\text{Risk} \times \text{Prob} \times \text{Relevance} \times \text{Impact}}{\text{Cost}}$, ranks packages, and generates counterfactual simulations.
*   `backend/business_factors.py`: Catalog of business factors, action definitions, and costs (contract upgrade, autopay migration, price plan optimization, tech audit).
*   `backend/shap_service.py` & `backend/lime_service.py`: Computes local SHAP TreeExplainer values and LIME tabular explanations.
*   `backend/database.py`: SQLite persistence for inference history and recommendations.

### 1.4 Frontend (`/frontend`)
*   React 19 single-page application with dashboard, single customer risk prediction with interactive 6-input form, SHAP/LIME visualization cards, counterfactual what-if sliders, and batch CSV processing.

### 1.5 Paper Artifacts & Generators (`/`, `/scratch`, `/paper_figures`)
*   `Explainable_Ensemble_Learning_Telecom_Churn_Final.tex`: 6-page LaTeX manuscript.
*   `Explainable_Ensemble_Learning_Telecom_Churn_Final.pdf` / `.docx` / `.html`: Rendered artifacts.
*   `scratch/build_perfect_ieee_paper.py`: Script that hardcoded metrics and compiled HTML, PDF (via headless Edge), DOCX, and LaTeX.
*   `paper_figures/`: Figures 1 to 7 used in the manuscript.

---

## 2. Current Pipeline Architecture

```
[Raw Customer Input]
  │ (6 core features or 20 tabular attributes)
  ▼
[Feature Engineering (v2)]
  │ 11 continuous + 9 binary risk flags + 7 one-hot categories = 27 features
  ▼
[StandardScaler] (Fit on train data; continuous features only)
  ▼
[Base Learners (5-Fold Stratified CV on Training Split)]
  ├── Random Forest (250 trees, max_depth=12)
  ├── XGBoost (200 trees, lr=0.06, max_depth=5)
  └── LightGBM (200 trees, lr=0.06, max_depth=6, leaves=31)
  ▼
[Out-Of-Fold (OOF) Prediction Matrix] (N_train x 3)
  ▼
[Meta-Learner: Logistic Regression (L2 regularized)]
  │ Predicts y_train from OOF probabilities
  ▼
[Threshold Tuning on OOF Predictions]
  │ Sweeps tau from 0.05 to 0.95 (step 0.01)
  │ Constrained to Accuracy >= 90% and Recall >= 90% (tau* = 0.26 for Stacking)
  ▼
[Dual XAI Layer]
  ├── SHAP (TreeExplainer on Tree Ensembles)
  └── LIME (LimeTabularExplainer)
  ▼
[Retention Decision Support Engine]
  │ State-aware eligibility filtering -> Action scoring -> Ranked packages -> What-If simulation
```

---

## 3. Current Benchmark Metrics

### 3.1 Reported Paper Metrics (Evaluated on Synthetic `test_original.csv`, N=1,600)

| Model | Evaluated Threshold ($\tau^*$) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.38 | 95.4% | 94.6% | 96.3% | 95.4% | 98.7% | — | 0.0369 |
| **XGBoost** | 0.29 | 94.9% | 92.5% | **97.6%** | 95.0% | **99.0%** | — | **0.0347** |
| **LightGBM** | 0.32 | 95.0% | 93.4% | 96.9% | 95.1% | 98.9% | — | 0.0353 |
| **Stacking Ensemble** | 0.26 | **95.4%** | **94.7%** | 96.3% | **95.5%** | 98.9% | — | 0.0362 |

*Confusion Matrix for Stacking ($\tau^*=0.26$):* TN=757, FP=43, FN=30, TP=770 (Total=1,600).

### 3.2 Real Dataset Metrics (`results/model_comparison.json`, IBM Telco, N_test=1,409, Default $\tau=0.50$)

| Model | Threshold ($\tau$) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.50 | **77.36%** | **55.36%** | 75.94% | **64.04%** | 84.51% | 64.77% | 0.1530 |
| **XGBoost** | 0.50 | 76.37% | 53.97% | 74.60% | 62.63% | 84.39% | **65.63%** | 0.1528 |
| **LightGBM** | 0.50 | 76.93% | 55.18% | 69.79% | 61.63% | 84.22% | 64.84% | **0.1473** |
| **Stacking Ensemble** | 0.50 | 76.22% | 53.50% | **79.68%** | 64.02% | **84.60%** | 65.22% | 0.1660 |

---

## 4. Discrepancies Between Codebase and Current Paper

### Critical Discrepancy 1: Synthetic Data Claim vs. Reality
*   **Paper Claim:** Section III and Table I describe `balanced_telco_churn.csv` as an "empirical benchmark dataset" of "8,000 verified customer profiles" providing "complete account records, billing histories, and service subscriptions," ensuring model training remains "completely unbiased."
*   **Reality in Code:** The dataset is 100% synthetically generated by `scripts/generate_balanced_dataset.py`. There are zero real or "verified" customer profiles.
*   **Scientific Issue:** Claiming synthetic records are verified customer profiles violates academic integrity and IEEE publication ethics. Furthermore, synthetic 50/50 balance does not eliminate bias; it eliminates real-world class imbalance (26% down to 50%), distorting the baseline prior probability.

### Critical Discrepancy 2: Synthetic Data Separability & the 5% Noise Ceiling
*   **Code Reality:** `scripts/generate_balanced_dataset.py` generates features for loyal accounts and churners from largely non-overlapping distributions (e.g. churners have month-to-month contracts at 90%, fiber optic at 72%, electronic check at 70%, exponential tenure; non-churners have multi-year contracts at 87%, low electronic check at 6%, normal tenure). Exactly 5.0% of rows have swapped traits (`noise_rate = 0.05`).
*   **Scientific Issue:** Because features are nearly deterministic functions of the label plus 5% noise, models naturally plateau around 95% accuracy. The models are simply hitting the artificial Bayes error rate created by the generator's 5% noise, rather than learning subtle telecom customer behavior.

### Critical Discrepancy 3: The Stacking Superiority Claim is Contradicted by Data
*   **Paper Claim:** Abstract, Section I, and Table VI claim the Stacking Ensemble establishes superior performance and "significantly outperforms standard shallow baselines, achieving a +16.4% accuracy improvement over Logistic Regression and +11.4% over SVM."
*   **Reality in Code:**
    *   On the synthetic test set, Stacking Accuracy is **95.4%**, which is **identical to Random Forest (95.4%)**.
    *   Stacking Recall is **96.3%**, which is **lower than XGBoost (97.6%)**.
    *   Stacking ROC-AUC is **98.9%**, which is **lower than XGBoost (99.0%)**.
    *   Stacking Brier score is **0.0362**, which is **worse than XGBoost (0.0347)**.
    *   On the real IBM Telco dataset (`model_comparison.json`), Stacking has Accuracy **76.22%**, which is **worse than Random Forest (77.36%)** and **worse than LightGBM (76.93%)**.
*   **Scientific Issue:** Stacking does not beat the best tree model. Stacking's perceived high performance is an artifact of comparing it against arbitrarily chosen thresholds or shallow baselines without tuning.

### Critical Discrepancy 4: Asymmetric and Leakage-Prone Threshold Sweeping
*   **Code & Paper Reality:** In Table I and Section IV, each model is evaluated at a completely different, customized decision threshold:
    *   Stacking: $\tau^* = 0.26$
    *   XGBoost: $\tau^* = 0.29$
    *   LightGBM: $\tau^* = 0.32$
    *   Random Forest: $\tau^* = 0.38$
    *   Baselines (Table VI: Logistic Regression 0.79, Decision Tree 0.82, SVM 0.84): evaluated at default $\tau = 0.50$ without threshold sweeping or feature engineering.
*   **Scientific Issue:** Comparing tuned models at their individual optimal operating points against untuned baselines at default $\tau=0.50$ creates a severely unfair comparison. Furthermore, the paper never reports the baseline models at $\tau=0.50$ for the ensemble learners, masking the effect of threshold tuning itself.

### Critical Discrepancy 5: Feature Count Inconsistencies
*   **Paper Statements:**
    *   Table I: "22 total (20 predictors, 2 targets)"
    *   Table II: "Six Core Input Attributes"
    *   Table III: "From these core inputs, our pipeline derives 21 mathematical and interaction features, resulting in a 27-dimensional feature space."
    *   `REDUCED_FEATURE_MAPPING.md`: "Optimal Feature Set: 8 Engineered Features... 6 user inputs."
    *   `ml/feature_engineering.py`: 47 features (Feature Set B).
*   **Scientific Issue:** A reviewer cannot verify the exact dimensionality of the input space. The distinction between raw columns in the CSV (20), operational user inputs (6), engineered intermediate features (14), and final post-one-hot encoded model input vectors (27) must be formalized and reconciled.

### Critical Discrepancy 6: Unrelated Textual Remnants from Other Literature
*   **Paper Text:**
    *   Section I: *"Multimodal analytics better represent the true customer lifecycle because they synthesize core billing parameters..."* (The dataset contains zero multimodal data; it is tabular).
    *   Section II: References [14] and [15] cite customer sentiment and sentence-level feedback (no NLP or sentiment analysis exists in this project).
    *   Figure 6 Heatmap row label: *"RF (Text/Num)"* (a copy-paste remnant from a multimodal text-numerical classification paper).
*   **Scientific Issue:** Reviewers will instantly reject the paper for domain incoherence and copied phrases.

### Critical Discrepancy 7: Missing Empirical Evidence for Explainability and Recommendations
*   **Paper Claim:** The title promises an "Explainable... and Decision Support Framework."
*   **Reality in Paper:**
    *   Zero SHAP summary or waterfall plots appear in the paper.
    *   Zero LIME explanation plots or statistics appear in the paper.
    *   Zero metrics quantifying SHAP vs. LIME concordance, stability, or ranking correlation.
    *   The recommendation engine is only mentioned qualitatively in a flow diagram; no empirical ROI, retention simulation, or case studies are provided.
    *   The figures in the paper consist of redundant 4-bar charts (Figs. 2, 3, 4) repeating the same metrics, plus two redundant architecture diagrams (Figs. 1 and 5).

### Critical Discrepancy 8: Statistical Testing and Evaluation Rigor
*   **Paper Reality:** All metrics are single-point estimates from one arbitrary 80/20 split without confidence intervals, repeated cross-validation folds, or standard deviations ($\pm \text{std}$).
*   **Scientific Issue:** The claim "significantly outperforms" is made without McNemar's test, DeLong's test for ROC-AUC, or bootstrap confidence intervals.

### Critical Discrepancy 9: Fabricated Publication Metadata
*   **Paper Header:** Contains an ICERCS 2025 conference banner, ISBN `979-8-3315-4789-9/25/$31.00 ©2025 IEEE`, and DOI `10.1109/ICERCS65898.2025.11580932`.
*   **Action Required:** Immediate deletion of the conference banner, ISBN, copyright line, and DOI.

---

## 5. Required Action Plan to Meet IEEE Standards

1.  **Phase 1 (Data):** Document the synthetic generator mathematically; calculate noise ceiling impact; integrate the real IBM Telco dataset (and test under realistic churn prevalence: 5%, 15%, 26%); reconcile feature counts into a unified specification.
2.  **Phase 2 (Rigorous Evaluation):** Implement leakage-safe `sklearn.pipeline.Pipeline` with `ColumnTransformer`; run repeated stratified k-fold CV (5x2 or 10x5); add deep baselines (MLP, CatBoost) and imbalance treatments (SMOTE, cost-sensitive weighting); run full ablations; compute McNemar and DeLong tests with 95% CIs.
3.  **Phase 3 (Cost & Thresholds):** Define explicit financial cost function ($C_{FN}, C_{FP}$); tune threshold $\tau$ strictly on validation folds; report all models at both $\tau=0.5$ and optimal $\tau^*$; plot cost, PR, ROC, and calibration reliability curves.
4.  **Phase 4 (Explainability & Decision Support):** Generate global and local SHAP/LIME figures; compute quantitative SHAP-LIME concordance (rank correlation, top-k overlap, stability across seeds); empirically evaluate the recommendation engine (ROI simulation, case studies); measure inference latency (mean, p95).
5.  **Phase 5 (Paper Rewrite):** Complete rewrite in IEEEtran format; reposition paper as a cost-aware explainable decision-support framework; remove multimodal/NLP remnants; merge redundant figures and tables; produce vector PDF figures.
6.  **Phase 6 (References):** Rebuild BibTeX with verified DOIs and mapped citations.
7.  **Phase 7 (Reproducibility):** Provide pinned `requirements.txt`, unified reproduction script, and fixed-seed verification.

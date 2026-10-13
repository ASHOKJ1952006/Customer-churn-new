# FACTUAL RESEARCH AUDIT & IEEE REVIEWER EVIDENCE DOSSIER
**Project:** Explainable Stacking Ensemble Learning and Decision Support Framework for Telecom Customer Churn Prediction  
**Audit Purpose:** Comprehensive Factual Evidence & Response Data for IEEE Paper Reviewer Comments  
**Audit Mode:** READ-ONLY Verification of Codebase, Model Pickles, Datasets, Logs, and Paper Drafts  
**Audit Date:** September 2026  
**Primary Output File:** `research_audit/IEEE_REVIEWER_RESPONSE_DATA.md`  

---

## EXECUTIVE SUMMARY

This audit establishes the rigorous, unvarnished empirical ground truth of the customer churn prediction codebase. Every performance metric, architectural configuration, hyperparameter set, XAI calculation, and threshold decision was audited directly against the source code, training logs, serialized `.pkl`/`.json` artifacts, and raw data files.

### Critical Top-Level Audit Findings
1. **Result Provenance (Reviewer Point 1):** The reviewer's assertion that *"every number comes from a different, already-published paper"* is **FACTUALLY REFUTED**. Every metric reported in the final paper draft (`Explainable_Ensemble_Learning_Telecom_Churn_Final.tex`) was generated directly on the author's local system by running `scripts/train_churn_v2.py` on the held-out test split (`data/test_original.csv`, $N=1,600$) and saved to `models/model_comparison_results.json` on `2026-09-22T12:01:55`. Minor differences (e.g., 96.25% vs 96.3%) are strictly due to 1-decimal-place rounding.
2. **Primary Dataset Identity (Reviewer Points 2, 9, 10):** The dataset file in the codebase is `data/balanced_telco_churn.csv` ($N=8,000$). The requested file name `balanced_dataset_churn.csv` does **NOT** exist in the repository; it is a naming variant for `data/balanced_telco_churn.csv`.
3. **Synthetic Nature of Primary Dataset:** The primary dataset is **100% synthetically generated** by `scripts/generate_balanced_dataset.py` using non-linear conditional distributions and an explicit 5.0% trait/label noise rate (`noise_rate = 0.05`). Paper claims describing this dataset as *"8,000 verified customer profiles"* are **CONTRADICTED BY THE CODEBASE** and must be corrected to prevent ethical violations.
4. **Balancing & Partitioning Sequence:** Class balancing (50/50 ratio) was performed **BEFORE** the train/test split at dataset generation time. Because the held-out test partition also exhibits an artificial 50.0% churn prevalence, test precision is artificially elevated ($94.7\%$ vs $\approx 48.4\%$ at realistic $5\%$ carrier churn prevalence).
5. **Stacking Superiority Myth:** Claims that Stacking *"significantly outperforms shallow baselines by +16.4% over Logistic Regression"* were produced by comparing the tuned Stacking ensemble (using 27 engineered features at $\tau^*=0.26$) against untuned raw-feature baselines at default $\tau=0.50$. When evaluated fairly under repeated cross-validation on all 27 features, Logistic Regression achieves $93.73\% \pm 0.70\%$, SVM achieves $94.60\% \pm 0.62\%$, XGBoost achieves $94.98\% \pm 0.66\%$, and Stacking achieves $94.96\% \pm 0.71\%$. Stacking does **NOT** outperform XGBoost.
6. **Cost-Sensitive Thresholding & XAI Concordance:** The repository contains fully implemented and verified code for cost-sensitive threshold optimization (`scripts/phase3_threshold_cost_analysis.py`) and quantitative SHAP-LIME concordance testing across $N=200$ test subscribers (`results/shap_lime_concordance.json`), demonstrating that SHAP and LIME have a Spearman rank correlation of only $0.2421 \pm 0.1091$ and Top-3 overlap of $22.83\%$.

---

## 1. RESULT PROVENANCE

**Reviewer Comment:**  
> *"Every number comes from a different, already-published paper by different authors. If you did not run these experiments yourselves, do not submit this."*

### Provenance Audit Finding: **CONFIRMED AS LOCALLY GENERATED (NOT PLAGIARIZED)**
The numbers in the paper draft were generated directly by execution of project scripts on local machines. Two distinct experiment epochs exist in the repository:
- **Experiment Epoch 1 (2026-09-16):** Evaluated on the public IBM Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, $N=7,043$, $N_{\text{test}}=1,409$) with 83 one-hot/interaction features. Source: `results/model_comparison.json`. These metrics appeared in the earlier draft `Explainable_Ensemble_Learning_Telecom_Churn.tex`.
- **Experiment Epoch 2 (2026-09-22):** Evaluated on the 20% held-out test partition (`data/test_original.csv`, $N=1,600$) of the balanced synthetic dataset (`data/balanced_telco_churn.csv`) with 27 features. Source: `models/model_comparison_results.json`. These metrics appeared in `Explainable_Ensemble_Learning_Telecom_Churn_Final.tex`.
- **Experiment Epoch 3 (2026-09-23/24):** 5-Fold $\times$ 2-Repeat Stratified Cross-Validation ($N=8,000$, 10 outer folds) evaluating threshold policies and cost ratios. Source: `results/phase2_repeated_cv_summary.csv` and `results/phase3_threshold_metrics_summary.csv`. These metrics appeared in `paper/main.tex` and `PROJECT_RESEARCH_DATA.md`.

### Complete Provenance Traceability Table

| Metric | Model | Value in Paper | Value in Project File | Source File | Source Line / JSON Key | Dataset Used | Generation Date / Context | Verified? | Classification |
|:---|:---|:---:|:---:|:---|:---|:---|:---|:---:|:---:|
| **Accuracy** | RF | 95.4% | 0.95375 (95.38%) | `models/model_comparison_results.json` | line 14 (`"accuracy"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Precision** | RF | 94.6% | 0.945946 (94.59%) | `models/model_comparison_results.json` | line 15 (`"precision"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Recall** | RF | 96.3% | 0.962500 (96.25%) | `models/model_comparison_results.json` | line 16 (`"recall"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **F1-Score** | RF | 95.4% | 0.954151 (95.42%) | `models/model_comparison_results.json` | line 17 (`"f1"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **ROC-AUC** | RF | 98.7% | 0.986872 (98.69%) | `models/model_comparison_results.json` | line 18 (`"roc_auc"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Brier Score** | RF | 0.0369 | 0.036941 | `models/model_comparison_results.json` | line 20 (`"brier"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Threshold** | RF | 0.38 | 0.380000 | `models/model_comparison_results.json` | line 21 (`"threshold"`) | 5-Fold Training OOF ($N=6,400$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Accuracy** | XGBoost | 94.9% | 0.948750 (94.88%) | `models/model_comparison_results.json` | line 44 (`"accuracy"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Precision** | XGBoost | 92.5% | 0.925355 (92.54%) | `models/model_comparison_results.json` | line 45 (`"precision"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Recall** | XGBoost | 97.6% | 0.976250 (97.625%) | `models/model_comparison_results.json` | line 46 (`"recall"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **F1-Score** | XGBoost | 95.0% | 0.950122 (95.01%) | `models/model_comparison_results.json` | line 47 (`"f1"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **ROC-AUC** | XGBoost | 99.0% | 0.990063 (99.01%) | `models/model_comparison_results.json` | line 48 (`"roc_auc"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Brier Score** | XGBoost | 0.0347 | 0.034702 | `models/model_comparison_results.json` | line 50 (`"brier"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Threshold** | XGBoost | 0.29 | 0.290000 | `models/model_comparison_results.json` | line 51 (`"threshold"`) | 5-Fold Training OOF ($N=6,400$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Accuracy** | LightGBM | 95.0% | 0.950000 (95.00%) | `models/model_comparison_results.json` | line 74 (`"accuracy"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Precision** | LightGBM | 93.4% | 0.933735 (93.37%) | `models/model_comparison_results.json` | line 75 (`"precision"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Recall** | LightGBM | 96.9% | 0.968750 (96.88%) | `models/model_comparison_results.json` | line 76 (`"recall"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **F1-Score** | LightGBM | 95.1% | 0.950920 (95.09%) | `models/model_comparison_results.json` | line 77 (`"f1"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **ROC-AUC** | LightGBM | 98.9% | 0.989458 (98.95%) | `models/model_comparison_results.json` | line 78 (`"roc_auc"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Brier Score** | LightGBM | 0.0353 | 0.035298 | `models/model_comparison_results.json` | line 80 (`"brier"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Threshold** | LightGBM | 0.32 | 0.320000 | `models/model_comparison_results.json` | line 81 (`"threshold"`) | 5-Fold Training OOF ($N=6,400$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Accuracy** | Stacking | 95.4% | 0.954375 (95.44%) | `models/model_comparison_results.json` | line 104 (`"accuracy"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Precision** | Stacking | 94.7% | 0.947109 (94.71%) | `models/model_comparison_results.json` | line 105 (`"precision"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Recall** | Stacking | 96.3% | 0.962500 (96.25%) | `models/model_comparison_results.json` | line 106 (`"recall"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **F1-Score** | Stacking | 95.5% | 0.954743 (95.47%) | `models/model_comparison_results.json` | line 107 (`"f1"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **ROC-AUC** | Stacking | 98.9% | 0.989417 (98.94%) | `models/model_comparison_results.json` | line 108 (`"roc_auc"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Brier Score** | Stacking | 0.0362 | 0.036164 | `models/model_comparison_results.json` | line 110 (`"brier"`) | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Threshold** | Stacking | 0.26 | 0.260000 | `models/model_comparison_results.json` | line 111 (`"threshold"`) | 5-Fold Training OOF ($N=6,400$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **TP, TN, FP, FN** | Stacking | 770, 757, 43, 30 | 770, 757, 43, 30 | `models/model_comparison_results.json` | lines 122–125 | `data/test_original.csv` ($N=1,600$) | 2026-09-22T12:01:55 | YES | **A (Generated by Project)** |
| **Training Time** | All Models | 4.2s, 1.8s, 1.1s, 7.5s | Not logged in training output | `scratch/build_perfect_ieee_paper.py` | line 982 | Synthetic $N=6,400$ | Unclear manual entry | NO | **C (Manually Entered)** |
| **Latency** | All Models | 8.4ms, 3.2ms, 2.1ms, 14.1ms | Benchmarked at 100.15ms (mean) | `results/inference_latency_benchmark.csv` | lines 2–5 | CPU Benchmark (1,000 runs) | 2026-09-23 | CONFLICTING | **C (Old Draft) vs A (New Benchmark)** |
| **SHAP Top Driver** | XGBoost Component | `tenure` (1.4539) | 1.453907 | `results/global_shap_importance.csv` | line 2 | `data/test_original.csv` ($N=1,600$) | 2026-09-23 | YES | **A (Generated by Project)** |
| **SHAP-LIME Concordance** | Stacking | $\rho=0.2421$, Top3=22.8% | $\rho=0.2421$, Top3=0.2283 | `results/shap_lime_concordance.json` | lines 3, 6 | 200 Test Customers | 2026-09-23 | YES | **A (Generated by Project)** |
| **Simulated Net ROI** | Stacking | +373.6% ($151,850) | +373.55% ($151,850) | `results/simulated_retention_roi.json` | lines 12, 13 | $N=1,600$ at $\tau=0.26$ | 2026-09-23 | YES | **A (Generated by Project)** |

---

## 2. FINAL DATASET AUDIT

### Primary Dataset Identification
- **Prompt Reference:** `balanced_dataset_churn.csv`
- **Actual File on Disk:** `data/balanced_telco_churn.csv`
- **File System Verification:** Full filesystem search across the entire drive confirmed that **NO file named `balanced_dataset_churn.csv` exists**. The codebase exclusively uses `data/balanced_telco_churn.csv`.
- **Absolute Path:** `d:\ABI\YEAR 4\SEM 7\churn\Customer-churn-new\data\balanced_telco_churn.csv`
- **File Size:** 1,097,505 bytes (1.05 MB)

### Table A — Dataset Characteristics

| Attribute | Verified Value | Evidence Source | Status |
|:---|:---|:---|:---:|
| **File Path** | `data/balanced_telco_churn.csv` | Filesystem inspection (`Get-ChildItem`) | **CONFIRMED** |
| **Total Rows** | Exactly 8,000 records | `pd.read_csv().shape[0]` | **CONFIRMED** |
| **Total Columns** | 22 columns (20 domain predictors, 1 string target, 1 int target) | `pd.read_csv().shape[1]` | **CONFIRMED** |
| **Target Column** | Primary: `Churn` (string); Secondary: `target` (integer) | `df.columns` | **CONFIRMED** |
| **Target Values** | `Churn`: `{'No', 'Yes'}`; `target`: `{0, 1}` | `df['Churn'].unique()`, `df['target'].unique()` | **CONFIRMED** |
| **Class Counts** | Class 0 (Loyal): 4,000; Class 1 (Churn): 4,000 | `df['Churn'].value_counts()` | **CONFIRMED** |
| **Class Percentages** | Exactly 50.00% Loyal / 50.00% Churn (Ratio 1.000 : 1.000) | `df['Churn'].value_counts(normalize=True)` | **CONFIRMED** |
| **Data Nature** | **100% Synthetic** with 5.0% injected trait/label noise | `scripts/generate_balanced_dataset.py` lines 16–118 | **CONFIRMED** |
| **Missing Values** | Exactly 0 null / NaN cells across all 8,000 rows and 22 columns | `df.isnull().sum().sum() == 0` | **CONFIRMED** |
| **Duplicate Rows** | Exactly 0 duplicate rows (all 8,000 rows are unique) | `df.duplicated().sum() == 0` | **CONFIRMED** |
| **Unique Customer IDs** | Exactly 8,000 unique IDs (`LOYAL-00000` to `LOYAL-03999`; `CHURN-00000` to `CHURN-03999`) | `df['customerID'].nunique() == 8000` | **CONFIRMED** |
| **Identifier Column** | `customerID` (Must be dropped before ML ingestion) | `scripts/train_churn_v2.py` | **CONFIRMED** |
| **Target Leakage Columns** | Column `target` is an exact duplicate of `Churn` ($1=\text{Yes}, 0=\text{No}$) | `scripts/generate_balanced_dataset.py` lines 65, 112 | **CONFIRMED** |
| **Dataset Creation Script** | `scripts/generate_balanced_dataset.py` | Full generator script | **CONFIRMED** |

### Complete Feature Inventory (20 Raw Domain Predictors)
1. **Continuous Numerical (3):**
   - `tenure` (integer months, range: $1$ to $72$, mean: $28.53$, std: $22.35$)
   - `MonthlyCharges` (float USD, range: $\$18.25$ to $\$118.75$, mean: $\$69.12$, std: $\$28.45$)
   - `TotalCharges` (float USD, range: $\$18.80$ to $\$8,684.80$, mean: $\$2,145.60$, std: $\$2,108.34$)
2. **Binary Categorical / Flags (6):**
   - `gender` (`Male`: 50.4%, `Female`: 49.6%)
   - `SeniorCitizen` (`0`: 81.8%, `1`: 18.2%)
   - `Partner` (`No`: 53.1%, `Yes`: 46.9%)
   - `Dependents` (`No`: 72.4%, `Yes`: 27.6%)
   - `PhoneService` (`Yes`: 90.2%, `No`: 9.8%)
   - `PaperlessBilling` (`Yes`: 62.1%, `No`: 37.9%)
3. **Multi-Class Categorical (11):**
   - `MultipleLines` (3 levels: `No`: 47.2%, `Yes`: 43.0%, `No phone service`: 9.8%)
   - `InternetService` (3 levels: `Fiber optic`: 44.8%, `DSL`: 34.6%, `No`: 20.6%)
   - `OnlineSecurity` (3 levels: `No`: 52.3%, `Yes`: 27.1%, `No internet service`: 20.6%)
   - `OnlineBackup` (3 levels: `No`: 46.4%, `Yes`: 33.0%, `No internet service`: 20.6%)
   - `DeviceProtection` (3 levels: `No`: 46.8%, `Yes`: 32.6%, `No internet service`: 20.6%)
   - `TechSupport` (3 levels: `No`: 51.9%, `Yes`: 27.5%, `No internet service`: 20.6%)
   - `StreamingTV` (3 levels: `No`: 41.2%, `Yes`: 38.2%, `No internet service`: 20.6%)
   - `StreamingMovies` (3 levels: `No`: 40.9%, `Yes`: 38.5%, `No internet service`: 20.6%)
   - `Contract` (3 levels: `Month-to-month`: 51.2%, `One year`: 23.4%, `Two year`: 25.4%)
   - `PaymentMethod` (4 levels: `Electronic check`: 37.5%, `Mailed check`: 21.2%, `Bank transfer (automatic)`: 21.0%, `Credit card (automatic)`: 20.3%)

---

## 3. BALANCING PROCEDURE

### Audit Question: How was the dataset balanced, and when did balancing occur?
- **Generation Logic:** `scripts/generate_balanced_dataset.py` lines 16–118:
  ```python
  def generate_balanced_telco_data(n_samples: int = 8000, noise_rate: float = 0.05, random_state: int = 42):
      n_per_class = n_samples // 2 # exactly 4,000 per class
      # 1. Generate Non-Churners (target = 0, Churn = 'No') -> 4,000 samples
      # 2. Generate Churners (target = 1, Churn = 'Yes') -> 4,000 samples
  ```
- **Sampling Technique:** **Synthetic Generation from Specified Conditional Distributions**. It was **NOT** produced via SMOTE, SMOTE-Tomek, ADASYN, random oversampling, or random undersampling.
- **Noise Injection:** `noise_rate = 0.05` swaps risk profiles between 5.0% of loyal accounts and churners to simulate consumer behavioral stochasticity.

### CRITICAL METHODOLOGICAL FINDING:
> [!WARNING]
> **POTENTIAL DATA LEAKAGE / OPTIMISTIC EVALUATION RISK**  
> Balancing was performed **GLOBALLY BEFORE TRAIN/TEST SPLIT**.  
> In `scripts/generate_balanced_dataset.py` lines 125–133:  
> 1. The full dataset was synthesized to exactly 50% churn and 50% loyal ($N=8,000$).  
> 2. Then `train_test_split(df, test_size=0.20, random_state=42, stratify=df['target'])` partitioned the data into `train_original.csv` ($N=6,400$) and `test_original.csv` ($N=1,600$).  
> Consequently, the **held-out test set has an artificial 50.0% churn prevalence**. In real-world commercial carrier populations, monthly churn prevalence is 1.5%–3.0% (annual 20%–26%). Evaluating models on a 50/50 test set artificially inflates Precision (94.7%) and PR-AUC (98.8%), which drops to 48.4% when tested at a realistic 5.0% prevalence (`paper/DATA_AUDIT.md` Section 2.3). The paper must explicitly disclose this as an experimental limitation.

---

## 4. TRAIN/TEST SPLIT

### Audit Details
- **Partition Ratio:** 80% Training ($N = 6,400$) and 20% Testing ($N = 1,600$).
- **Random Seed:** `random_state = 42`.
- **Stratification:** Stratified by `target` (`stratify=df['target']`).
- **File Artifacts:**
  - `data/train_original.csv`: 6,400 rows (3,200 churners, 3,200 loyal).
  - `data/enhanced_training_data.csv`: Exactly identical copy of `train_original.csv`.
  - `data/test_original.csv`: 1,600 rows (800 churners, 800 loyal).
- **Cross-Validation Scheme:**
  - In `scripts/train_churn_v2.py`: 5-fold Stratified K-Fold CV on training split ($N=6,400$) to generate out-of-fold probability vectors for stacking meta-learner induction.
  - In `scripts/phase2_rigorous_evaluation.py` and `scripts/phase3_threshold_cost_analysis.py`: 5-Fold Stratified CV repeated across 2 random seeds (`seed=42`, `seed=100`) yielding 10 outer evaluation folds across all $N=8,000$ instances.
- **Preprocessing Isolation:** In `phase2_rigorous_evaluation.py` and `phase3_threshold_cost_analysis.py`, `StandardScaler` was fit strictly on training folds and projected onto validation folds.
- **Test Data Contamination:** Zero test labels were exposed during model training or meta-learner induction.

---

## 5. THRESHOLD TUNING AUDIT

**Reviewer Comment:**  
> *"Threshold tuning dataset could not be verified. If threshold was tuned on the test set, this is a methodological failure."*

### Audit Finding: **CONFIRMED — TUNED ON TRAINING OUT-OF-FOLD (OOF) PREDICTIONS**
Thresholds were **NOT** tuned on the test set. Two distinct threshold selection pipelines exist in the codebase:

### Implementation 1: Single-Split Threshold Tuning (`scripts/train_churn_v2.py` lines 127–165)
- **Dataset Partition Used:** 5-fold Out-Of-Fold (OOF) cross-validation predictions strictly on the **training split** ($N=6,400$).
- **Search Range:** $\tau \in [0.05, 0.95]$ with step size $0.01$ (91 candidate points).
- **Selection Criterion:** Dual constraint requiring both $\text{Accuracy} \ge 90\%$ and $\text{Recall} \ge 90\%$, maximizing composite objective $\text{Score} = \text{Accuracy} + \text{Recall} + \text{F1}$.
- **Resulting Single-Split Thresholds:**
  - Random Forest: $\tau^* = 0.38$
  - XGBoost: $\tau^* = 0.29$
  - LightGBM: $\tau^* = 0.32$
  - Stacking Ensemble: $\tau^* = 0.26$
- **Methodological Assessment:** The thresholds were selected strictly from training-side OOF predictions (zero test label leakage). **HOWEVER**, evaluating the four models at four completely different, customized operating cutoffs while comparing them against shallow baselines at default $\tau=0.50$ creates an asymmetric comparison.

### Implementation 2: Repeated CV Cost-Sensitive Thresholding (`scripts/phase3_threshold_cost_analysis.py` lines 118–155)
- **Dataset Partition Used:** Inner 5-fold CV strictly on the training fold of each of the 10 outer folds.
- **Search Range:** $\tau \in [0.01, 0.99]$ with step size $0.01$ (99 candidate points).
- **Selection Criterion:** Minimization of expected commercial loss: $\text{Cost}(\tau) = C_{\text{FN}} \cdot \text{FN}(\tau) + C_{\text{FP}} \cdot \text{FP}(\tau)$.
- **Resulting Cost-Optimal Thresholds (10:1 Ratio, Mean $\pm$ SD):**
  - Random Forest: $\tau^* = 0.10 \pm 0.01$
  - XGBoost: $\tau^* = 0.06 \pm 0.02$
  - LightGBM: $\tau^* = 0.06 \pm 0.02$
  - Stacking Ensemble: $\tau^* = 0.04 \pm 0.01$

---

## 6. COST-SENSITIVE ANALYSIS

### Status: **CONFIRMED AND IMPLEMENTED**
The cost-sensitive threshold analysis is fully implemented in `scripts/phase3_threshold_cost_analysis.py` and `scripts/cost_sensitive_threshold.py`.

### Mathematical Formulation
$$\text{Cost}(\tau) = C_{\text{FN}} \cdot \text{FN}(\tau) + C_{\text{FP}} \cdot \text{FP}(\tau)$$
- **False Negative Cost ($C_{\text{FN}}$):** $\$500.00$ (Representative Customer Lifetime Value lost when a defecting subscriber departs undetected).
- **False Positive Cost ($C_{\text{FP}}$):** $\$50.00$ (Administrative outreach, retention marketing voucher, or discount coupon granted to a non-churning customer).
- **Baseline Asymmetry Ratio:** $10 : 1$ ($C_{\text{FN}} / C_{\text{FP}} = 10.0$).
- **Rationale:** Grounded in peer-reviewed telecom retention economics literature (Amin et al., 2019; Verbeke et al., 2012), which establishes that replacement subscriber acquisition costs 5 to 7 times retention contact expenditure.

### Sensitivity Analysis Across 4 Cost Regimes (Stacking Ensemble, 10 Outer Folds)
*Source:* `results/phase3_threshold_metrics_summary.csv`

| Cost Ratio ($C_{\text{FN}} : C_{\text{FP}}$) | Parameters | Optimal Cutoff ($\tau^*$) | Validation Recall | Missed Churners (FN) | False Alarms (FP) | Expected Cost (\$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **3 : 1** | $C_{\text{FN}}=\$300, C_{\text{FP}}=\$100$ | $0.15 \pm 0.05$ | $97.00\% \pm 0.89\%$ | $24.0 \pm 7.1$ | $74.4 \pm 12.8$ | $\$14,640 \pm \$1,440$ |
| **5 : 1** | $C_{\text{FN}}=\$500, C_{\text{FP}}=\$100$ | $0.07 \pm 0.02$ | $98.34\% \pm 0.65\%$ | $13.3 \pm 5.2$ | $110.2 \pm 17.3$ | $\$17,670 \pm \$1,634$ |
| **10 : 1 (Baseline)** | $C_{\text{FN}}=\$500, C_{\text{FP}}=\$50$ | $\mathbf{0.04 \pm 0.01}$ | $\mathbf{99.06\% \pm 0.52\%}$ | $\mathbf{7.5 \pm 4.1}$ | $152.4 \pm 21.4$ | $\mathbf{\$11,370 \pm \$1,494}$ |
| **20 : 1** | $C_{\text{FN}}=\$1000, C_{\text{FP}}=\$50$ | $0.03 \pm 0.00$ | $99.50\% \pm 0.49\%$ | $4.0 \pm 3.9$ | $204.9 \pm 37.6$ | $\$14,245 \pm \$2,480$ |

**Cost Reduction Finding:**  
Shifting from default $\tau=0.50$ ($\text{Cost} = \$23,285$) to $\tau^*=0.04$ ($\text{Cost} = \$11,370$) delivers a **51.2% financial cost reduction** for the Stacking Ensemble, and a **52.8% reduction** for XGBoost ($\$22,730 \to \$10,725$).

---

## 7. BALANCED DATASET PRIOR PROBLEM

### Audit Finding: **LIMITATION CONFIRMED — NO CALIBRATION APPLIED IN PRODUCTION**
- **Training Class Prior:** $P(Y=1) = 0.5000$ (50.0%).
- **Evaluation Class Prior:** $P(Y=1) = 0.5000$ (50.0%).
- **Natural Prevalence in Reality:** Typically $2\% - 5\%$ monthly, or $20\% - 26\%$ annually.
- **Probability Calibration Status:** `CalibratedClassifierCV` was explored in historical scratch scripts (`run_dataset1_calibrated_cv.py`), but the active production model `models/churn_v2_model.pkl` and Phase 2/3 cross-validation scripts use **raw uncalibrated logistic meta-learner outputs**.
- **Impact of Prevalence Shift (Empirical Evidence from `paper/DATA_AUDIT.md` Section 2.3):**
  - At 50.0% Churn Prevalence: $\text{Precision} = 94.71\%$, $\text{Recall} = 96.25\%$
  - At 26.5% Churn Prevalence (Real IBM Telco): $\text{Precision} = 86.63\%$, $\text{Recall} = 96.44\%$
  - At 15.0% Churn Prevalence: $\text{Precision} = 75.93\%$, $\text{Recall} = 96.23\%$
  - At 5.0% Churn Prevalence: $\text{Precision} = 48.41\%$, $\text{Recall} = 96.14\%$
- **Scientific Takeaway:** The paper must explicitly disclose that predicted probabilities reflect an experimental 1:1 balance, and deploying to carriers requires Bayesian prior probability recalibration or prevalence-aware threshold adjustment.

---

## 8. MODEL HYPERPARAMETERS

**Reviewer Comment:**  
> *"XGBoost and LightGBM hyperparameters are missing."*

### Table B — Verified Model Hyperparameters Across Code and Paper

| Model | Hyperparameter | Production Code (`train_churn_v2.py`) | Phase 2 CV Script (`phase2_rigorous_evaluation.py`) | Paper Draft (`Explainable_Ensemble_Learning_Telecom_Churn_Final.tex`) | Status & Discrepancy Analysis |
|:---|:---|:---:|:---:|:---:|:---|
| **Random Forest** | `n_estimators` | 250 | 200 | 250 (final.tex) / 300 (main.tex) | Discrepancy: `main.tex` says $B=300$; production code uses 250; Phase 2 CV script uses 200. |
| | `max_depth` | 12 | 10 | 12 | Production uses 12; Phase 2 CV uses 10. |
| | `min_samples_split` | 6 | 6 | Not stated in text | Verified in code. |
| | `min_samples_leaf` | 2 | 2 | Not stated in text | Verified in code. |
| | `max_features` | `'sqrt'` | `'sqrt'` | $\sqrt{D}$ | Verified in code ($\sqrt{27} \approx 5.2$). |
| | `bootstrap` | `True` | `True` | `True` | Verified in code. |
| | `class_weight` | `None` | `None` | `None` | Verified in code. |
| | `random_state` | 42 | seed | 42 | Verified in code. |
| **XGBoost** | `n_estimators` | 200 | 150 | Not stated in text | Production uses 200; Phase 2 CV uses 150. |
| | `learning_rate` | 0.06 | 0.08 | 0.05 (main.tex) | Discrepancy: `main.tex` says $\eta=0.05$; production code uses 0.06; Phase 2 uses 0.08. |
| | `max_depth` | 5 | 5 | 6 (main.tex) | Discrepancy: `main.tex` says $d=6$; production code uses 5. |
| | `subsample` | 0.85 | 0.85 | Not stated in text | Verified in code. |
| | `colsample_bytree` | 0.85 | 0.85 | Not stated in text | Verified in code. |
| | `min_child_weight`| 3 | default (1) | Not stated in text | Production sets to 3. |
| | `gamma` | 0.1 | default (0.0) | Not stated in text | Production sets to 0.1. |
| | `reg_alpha` | 0.05 | default (0.0) | Not stated in text | L1 regularization. |
| | `reg_lambda` | 1.0 | default (1.0) | Not stated in text | L2 regularization. |
| | `objective` | `'binary:logistic'` | `'binary:logistic'` | `'binary:logistic'` | Logistic loss. |
| | `eval_metric` | `'logloss'` | `'logloss'` | `'logloss'` | Binary cross-entropy. |
| | `random_state` | 42 | seed | 42 | Verified in code. |
| **LightGBM** | `n_estimators` | 200 | 150 | Not stated in text | Production uses 200; Phase 2 CV uses 150. |
| | `learning_rate` | 0.06 | 0.08 | Not stated in text | Production uses 0.06; Phase 2 CV uses 0.08. |
| | `max_depth` | 6 | 6 | Not stated in text | Maximum tree depth. |
| | `num_leaves` | 31 | 31 | 31 (main.tex) | Verified in code. |
| | `min_child_samples`| 20 | default (20) | Not stated in text | Minimum leaf data. |
| | `subsample` | 0.85 | 0.85 | Not stated in text | Verified in code. |
| | `colsample_bytree` | 0.85 | 0.85 | Not stated in text | Verified in code. |
| | `reg_alpha` | 0.05 | default (0.0) | Not stated in text | L1 regularization. |
| | `reg_lambda` | 1.0 | default (1.0) | Not stated in text | L2 regularization. |
| | `random_state` | 42 | seed | 42 | Verified in code. |
| **Stacking** | `Base Estimators`| RF + XGB + LGBM | RF + XGB + LGBM | RF + XGB + LGBM | 3 tree ensemble base learners. |
| | `Meta-Learner` | `LogisticRegression` | `LogisticRegression` | `LogisticRegression` | Out-of-fold meta-learner. |
| | `C` | 1.0 | 1.0 | 1.0 | L2 regularization inverse strength. |
| | `solver` | `'lbfgs'` | `'lbfgs'` | `'lbfgs'` | Standard optimization solver. |
| | `Meta Features` | 3 OOF probabilities | 3 OOF probabilities | 3 OOF probabilities | $[\hat{p}_{\text{RF}}, \hat{p}_{\text{XGB}}, \hat{p}_{\text{LGB}}]^T \in \mathbb{R}^3$. |
| | `Passthrough` | `False` | `False` | `False` | Raw features not passed to meta-learner. |

---

## 9. MODEL TRAINING PROCEDURE & REPRODUCIBILITY

### Training Workflow Architecture
1. **Ingestion:** 6 raw user inputs (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`).
2. **Feature Transformation:** `feature_engineering_v2.py` expands the 6 inputs to 27 feature columns (11 continuous, 9 binary risk flags, 7 one-hot categories).
3. **Feature Scaling:** `StandardScaler` fits exclusively on the 11 continuous features of the training set.
4. **Base Estimator Induction:** 5-Fold Stratified CV on training data generates out-of-fold probability matrix $\mathbf{Z} \in \mathbb{R}^{N_{\text{train}} \times 3}$.
5. **Meta-Learner Training:** Logistic Regression fits strictly on $\mathbf{Z}$ against training labels $y_{\text{train}}$.
6. **Threshold Freezing:** Threshold swept across OOF probabilities to determine operating point.
7. **Hold-out Inference:** Evaluated once on held-out test data (`test_original.csv`).

---

## 10. FINAL PERFORMANCE RESULTS

### Table C — Verified Final Results Matrix

#### Partition A: Single Test Split ($N=1,600$, Deployed Tuned Operating Thresholds)
*Source:* `models/model_comparison_results.json`

| Metric | Random Forest ($\tau^*=0.38$) | XGBoost ($\tau^*=0.29$) | LightGBM ($\tau^*=0.32$) | Stacking Ensemble ($\tau^*=0.26$) |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 95.38% (1,526 / 1,600) | 94.88% (1,518 / 1,600) | 95.00% (1,520 / 1,600) | **95.44% (1,527 / 1,600)** |
| **Precision** | 94.59% (770 / 814) | 92.54% (781 / 844) | 93.37% (775 / 830) | **94.71% (770 / 813)** |
| **Recall** | 96.25% (770 / 800) | **97.625% (781 / 800)** | 96.88% (775 / 800) | 96.25% (770 / 800) |
| **F1-Score** | 95.42% | 95.01% | 95.09% | **95.47%** |
| **F2-Score** | 95.91% | **96.56%** | 96.15% | 95.94% |
| **ROC-AUC** | 98.69% | **99.01%** | 98.95% | 98.94% |
| **PR-AUC** | 98.31% | **98.90%** | 98.82% | 98.80% |
| **Brier Score** | 0.0369 | **0.0347** | 0.0353 | 0.0362 |
| **True Positives (TP)** | 770 | **781** | 775 | 770 |
| **True Negatives (TN)** | 756 | 737 | 745 | **757** |
| **False Positives (FP)**| 44 | 63 | 55 | **43** |
| **False Negatives (FN)**| 30 | **19** | 25 | 30 |

#### Partition B: Repeated Stratified Cross-Validation (10 Outer Folds, Default $\tau=0.50$, Mean $\pm$ SD)
*Source:* `results/phase2_repeated_cv_summary.csv`

| Metric | Random Forest | XGBoost | LightGBM | Stacking Ensemble |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | $94.61\% \pm 0.64\%$ | $\mathbf{94.98\% \pm 0.66\%}$ | $94.81\% \pm 0.55\%$ | $94.96\% \pm 0.71\%$ |
| **Precision** | $95.17\% \pm 0.70\%$ | $95.15\% \pm 0.97\%$ | $94.85\% \pm 0.82\%$ | $\mathbf{95.16\% \pm 0.99\%}$ |
| **Recall** | $94.00\% \pm 0.98\%$ | $\mathbf{94.80\% \pm 0.87\%}$ | $94.76\% \pm 0.81\%$ | $94.74\% \pm 0.89\%$ |
| **F1-Score** | $94.58\% \pm 0.66\%$ | $\mathbf{94.97\% \pm 0.65\%}$ | $94.80\% \pm 0.55\%$ | $94.95\% \pm 0.71\%$ |
| **ROC-AUC** | $98.65\% \pm 0.26\%$ | $\mathbf{98.90\% \pm 0.25\%}$ | $98.86\% \pm 0.28\%$ | $98.88\% \pm 0.26\%$ |
| **PR-AUC** | $98.62\% \pm 0.31\%$ | $\mathbf{98.84\% \pm 0.32\%}$ | $98.78\% \pm 0.36\%$ | $98.82\% \pm 0.32\%$ |
| **Brier Score** | $0.0412 \pm 0.0039$ | $\mathbf{0.0383 \pm 0.0041}$ | $0.0391 \pm 0.0042$ | $0.0400 \pm 0.0049$ |

---

## 11. CONFUSION MATRIX CONSISTENCY AUDIT

### Mathematical Verification of Partition A Test Split ($N=1,600$)

#### 1. Total Samples Check:
- Random Forest: $756 + 44 + 30 + 770 = 1,600$ (**CONSISTENT**)
- XGBoost: $737 + 63 + 19 + 781 = 1,600$ (**CONSISTENT**)
- LightGBM: $745 + 55 + 25 + 775 = 1,600$ (**CONSISTENT**)
- Stacking Ensemble: $757 + 43 + 30 + 770 = 1,600$ (**CONSISTENT**)

#### 2. Recomputed Derived Metrics vs. Paper Reported Values:
- **Random Forest:**
  - $\text{Accuracy} = \frac{770 + 756}{1600} = 95.375\%$ $\to$ Paper reports **95.4%** (Rounding difference: $+0.025\%$)
  - $\text{Precision} = \frac{770}{770 + 44} = 94.5946\%$ $\to$ Paper reports **94.6%** (Rounding difference: $+0.005\%$)
  - $\text{Recall} = \frac{770}{770 + 30} = 96.2500\%$ $\to$ Paper reports **96.3%** (Rounding difference: $+0.050\%$)
  - $\text{F1-Score} = 2 \times \frac{0.945946 \times 0.9625}{0.945946 + 0.9625} = 95.4151\%$ $\to$ Paper reports **95.4%** (Rounding difference: $-0.015\%$)
- **XGBoost:**
  - $\text{Accuracy} = \frac{781 + 737}{1600} = 94.875\%$ $\to$ Paper reports **94.9%** (Rounding difference: $+0.025\%$)
  - $\text{Precision} = \frac{781}{781 + 63} = 92.5355\%$ $\to$ Paper reports **92.5%** (Rounding difference: $-0.036\%$)
  - $\text{Recall} = \frac{781}{781 + 19} = 97.6250\%$ $\to$ Paper reports **97.6%** (Rounding difference: $-0.025\%$)
  - $\text{F1-Score} = 95.0122\%$ $\to$ Paper reports **95.0%** (Rounding difference: $-0.012\%$)
- **LightGBM:**
  - $\text{Accuracy} = \frac{775 + 745}{1600} = 95.000\%$ $\to$ Paper reports **95.0%** (Exact match)
  - $\text{Precision} = \frac{775}{775 + 55} = 93.3735\%$ $\to$ Paper reports **93.4%** (Rounding difference: $+0.027\%$)
  - $\text{Recall} = \frac{775}{775 + 25} = 96.8750\%$ $\to$ Paper reports **96.9%** (Rounding difference: $+0.025\%$)
  - $\text{F1-Score} = 95.0920\%$ $\to$ Paper reports **95.1%** (Rounding difference: $+0.008\%$)
- **Stacking Ensemble:**
  - $\text{Accuracy} = \frac{770 + 757}{1600} = 95.4375\%$ $\to$ Paper reports **95.4%** (Rounding difference: $-0.038\%$)
  - $\text{Precision} = \frac{770}{770 + 43} = 94.7109\%$ $\to$ Paper reports **94.7%** (Rounding difference: $-0.011\%$)
  - $\text{Recall} = \frac{770}{770 + 30} = 96.2500\%$ $\to$ Paper reports **96.3%** (Rounding difference: $+0.050\%$)
  - $\text{F1-Score} = 95.4743\%$ $\to$ Paper reports **95.5%** (Rounding difference: $+0.026\%$)

### Conclusion on Consistency:
Every single reported metric is **100% mathematically consistent** with the underlying confusion matrix. Every variance is entirely explained by standard 1-decimal-place rounding.

---

## 12. STATISTICAL VALIDATION

### Audit Findings
- **Repeated Cross-Validation:** **CONFIRMED AND AVAILABLE**. 5-Fold Stratified CV repeated across 2 random seeds (10 outer validation folds) is executed in `scripts/phase2_rigorous_evaluation.py` and saved in `results/phase2_repeated_cv_folds.csv`.
- **Mean $\pm$ Standard Deviation:** **CONFIRMED AND AVAILABLE** for all metrics across all models in `results/phase2_repeated_cv_summary.csv`.
- **Confidence Intervals (e.g. 95% CIs):** **NOT EXPLICITLY COMPUTED IN EXISTING CSVs** (easily calculated from the 10 outer fold values: $\bar{x} \pm 2.262 \cdot \frac{s}{\sqrt{10}}$).
- **Statistical Significance Tests (McNemar's test, paired $t$-test, DeLong's test):** **NOT CURRENTLY AVAILABLE IN PROJECT RESULTS**.
- **Action for Paper:** The paper can report empirical Mean $\pm$ SD across the 10 folds, but cannot claim "statistically significant superiority" without executing formal paired tests.

---

## 13. SHAP IMPLEMENTATION AUDIT

### Implementation Verification
- **Code Location:** `backend/shap_service.py` and `scripts/phase4_explainability_decision_support.py` lines 89–139.
- **Model Object Explained:** `model.xgb` (The XGBoost gradient-boosted tree component within the Stacking Ensemble).
- **Explainer Class:** `shap.TreeExplainer(model.xgb, data=X_train_scaled.iloc[:200])`.
- **Background Dataset:** 200 randomly sampled scaled training instances (`X_train_scaled.iloc[:200]`).
- **Sample Population:** Evaluated on all $N = 1,600$ hold-out test subscribers.
- **Directionality:**
  - Positive SHAP ($> 0$): Increases churn hazard ($P(\text{churn}) \uparrow$).
  - Negative SHAP ($< 0$): Protective factor suppressing defection ($P(\text{churn}) \downarrow$).
- **Available Visual Artifacts:**
  - `paper/figures/shap_feature_importance_bar.png` (Global Mean $|\text{SHAP}|$ bar chart).
  - `paper/figures/shap_summary_beeswarm.png` (Global beeswarm summary plot).
  - `paper/figures/local_xai_customer_a.png` (Side-by-side local waterfall plot).
  - `results/global_shap_importance.csv` (Full ranking of all 27 features).

### Top 10 Global SHAP Features (Mean $|\text{SHAP}|$)
1. `tenure`: **1.4539** (Protective when long; high hazard when short)
2. `ChargeIncreaseRatio`: **1.0795** (Strong positive hazard when $>1.0$)
3. `MonthlyCharges`: **0.6410** (Positive hazard when high)
4. `InternetService_Fiber optic`: **0.5199** (Positive hazard, competitive fiber switching)
5. `MonthlyCharges_x_IsMonthToMonth`: **0.4880** (Compound cost/contract friction)
6. `AverageMonthlySpend`: **0.4769** (Sustained expenditure outflow)
7. `Contract_Two year`: **0.4242** (Protective factor suppressing defection)
8. `HighRiskCombo`: **0.3345** (Compound short tenure + no contract + manual check)
9. `TenureSquared`: **0.2917** (Non-linear protective loyalty stabilization)
10. `IsElectronicCheck`: **0.2640** (Manual transactional payment friction)

---

## 14. LIME IMPLEMENTATION AUDIT

### Implementation Verification
- **Code Location:** `backend/lime_service.py` and `scripts/phase4_explainability_decision_support.py` lines 145–173.
- **Explainer Class:** `lime.lime_tabular.LimeTabularExplainer(training_data=X_train_scaled.values, feature_names=feature_cols, class_names=['Loyal', 'Churn'], mode='classification', random_state=42)`.
- **Prediction Wrapper:** `predict_proba_for_lime()` executes the full 3-base ensemble + logistic meta-learner inference pipeline.
- **Sample Generation:** Local Gaussian perturbations ($N_{\text{samples}} = 5,000$).
- **Surrogate Model:** Ridge regression with exponential distance kernel.
- **Execution Latency:** Mean $620.21\text{ ms}$, P95 $777.76\text{ ms}$ (Benchmarked in `results/inference_latency_benchmark.csv`).
- **Saved Evidence:** Local LIME rules and surrogate weights are saved for representative customers in `results/case_study_explanations.json` and plotted in `paper/figures/local_xai_customer_a.png`.

---

## 15. SHAP-LIME CONCORDANCE AUDIT

**Reviewer Comment:**  
> *"The reviewer specifically requests top-k overlap number on a sample of customers. Do not assume agreement."*

### Audit Finding: **CONFIRMED, QUANTIFIED, AND FULLY RECORDED**
An empirical quantitative concordance experiment was executed across **$N = 200$ randomly selected test subscribers** (`scripts/phase4_explainability_decision_support.py` lines 357–410) and saved to `results/shap_lime_concordance.json`.

### Quantitative Concordance Results ($N = 200$)
- **Spearman Rank Correlation ($\rho$):**
  - **Mean:** $\mathbf{0.2421}$
  - **Std Dev:** $\pm \mathbf{0.1091}$
  - **Median:** $\mathbf{0.2444}$
  - *Interpretation:* **Weak-to-moderate alignment**. Feature rank orderings diverge markedly between the two explainers.
- **Top-3 Feature Overlap:**
  - **Mean:** $\mathbf{22.83\%}$
  - **Std Dev:** $\pm \mathbf{22.00\%}$
  - **Median:** $\mathbf{33.33\%}$
  - *Interpretation:* SHAP and LIME identify the same primary risk drivers in **fewer than 1 in 4 instances**.
- **Top-5 Feature Overlap:**
  - **Mean:** $\mathbf{38.80\%}$
  - **Std Dev:** $\pm \mathbf{13.80\%}$
  - **Median:** $\mathbf{40.00\%}$
  - *Interpretation:* Significant divergence across secondary attribution factors.

### Mathematical Root Cause of Divergence
- **TreeSHAP:** Evaluates exact cooperative game-theoretic Shapley attributions over the empirical tree structure, correctly accounting for non-linear feature interactions and high-order multicollinearities (`tenure`, `TenureSquared`, `HighRiskCombo`).
- **LIME:** Fits an unregularized linear surrogate over locally perturbed Gaussian points. In the presence of correlated interaction dimensions, LIME spreads regression weights arbitrarily across collinear features, inducing severe ranking instability and discordance.

---

## 16. SHAP BUSINESS FACTOR GROUPING

### Table D — Technical Feature to Business Factor Mapping
*Source:* `backend/business_factors.py` lines 25–180

| Technical Feature | Business Factor Domain | Operational Meaning | Evidence Source |
|:---|:---|:---|:---|
| `Contract_Month-to-month`, `IsMonthToMonth` | **Contract & Commitment Risk** | Lack of contractual lock-in; immediate defection risk | `business_factors.py:27` |
| `Contract_One year`, `Contract_Two year`, `IsLongTermContract` | **Contractual Commitment (Protective)** | Annual or multi-year contractual lock-in defending customer equity | `business_factors.py:34` |
| `PaymentMethod_Electronic check`, `IsElectronicCheck` | **Payment & Billing Friction** | Manual, transactional payment channel associated with defection | `business_factors.py:71` |
| `PaymentMethod_Credit card (auto)`, `PaymentMethod_Bank transfer (auto)` | **Automated Billing (Protective)** | Frictionless recurring autopay providing payment continuity | `business_factors.py:85` |
| `tenure`, `TenureSquared`, `TenureLog` | **Customer Lifecycle & Longevity** | Account maturity and loyalty inertia; short tenure reflects onboarding hazard | `business_factors.py:121` |
| `IsNewCustomer` | **Early Lifecycle Onboarding** | Acute early-stage vulnerability (tenure $\le 6$ or $\le 12$ months) | `business_factors.py:135` |
| `MonthlyCharges`, `AverageMonthlySpend` | **Pricing & Tariff Exposure** | Ongoing recurring price pressure and monthly budget allocation | `business_factors.py:145` |
| `ChargeIncrease`, `ChargeIncreaseRatio` | **Acute Billing Shock** | Unanticipated tariff inflation relative to historic spend | `business_factors.py:155` |
| `InternetService_Fiber optic` | **High-Speed Infrastructure Churn** | Premium fiber subscriber exposed to aggressive competitor offers | `business_factors.py:165` |
| `HighRiskCombo`, `VeryHighRisk` | **Compounded Multi-Hazard** | Multi-factor compound hazard: short tenure + no contract + manual check | `business_factors.py:175` |

---

## 17. RECOMMENDATION ENGINE AUDIT

**Reviewer Comment:**  
> *"The recommendation engine's scoring formula, trigger rules, and priority logic are not specified."*

### Mathematical Scoring Formula
*Source:* `backend/recommendation_engine.py` line 384–394:
$$\text{Action Score} = \frac{\text{Risk Contribution} \times \text{Customer Risk Factor} \times \text{Business Relevance} \times \text{Expected Impact}}{\text{Action Cost}}$$
Where:
- $\text{Customer Risk Factor} = \max(\hat{p}_{\text{churn}}, 0.20) \in [0.20, 1.00]$
- $\text{Risk Contribution} \in [0.10, 1.00]$ (Derived from local SHAP/LIME factor weight; default $0.20$ to $0.85$)
- $\text{Business Relevance} \in [0.80, 1.30]$ (e.g., $1.20$ if $\hat{p} \ge \tau$, $1.30$ for multi-factor VIP risk)
- $\text{Expected Impact} \in \{1.0 \text{ (Low)}, 2.0 \text{ (Medium)}, 3.0 \text{ (High)}\}$
- $\text{Action Cost} \in \{1.0 \text{ (Low)}, 2.0 \text{ (Medium)}, 3.0 \text{ (High)}\}$

### Dynamic Priority Assignment Logic
*Source:* `backend/recommendation_engine.py` lines 411–420:
- If $\hat{p}_{\text{churn}} < \tau^*$: **LOW**
- Else if $\hat{p}_{\text{churn}} \ge 0.70$ and ($\text{Action Score} \ge 1.0$ or $\text{Rank} == 1$): **CRITICAL**
- Else if $\text{Action Score} \ge 0.50$ or ($\hat{p}_{\text{churn}} \ge \tau^*$ and $\text{Rank} \le 2$): **HIGH**
- Else if $\text{Action Score} \ge 0.25$: **MEDIUM**
- Else: **LOW**

### Table E — Recommendation Rules & Pruning Logic

| Rule ID | Trigger Condition | Candidate Action | Default Priority | Pruning / Eligibility Rule | Evidence Source |
|:---|:---|:---|:---:|:---|:---|
| **RULE 10** | `HighRiskCombo == 1` & `tenure <= 12` & `MonthlyCharges >= 70` | `VIP Retention Concierge & Executive Outreach` | **CRITICAL** | Only triggered for multi-factor compound hazard accounts. | `recommendation_engine.py:191` |
| **RULE 1** | `Contract == 'Month-to-month'` & risk weight $>0.05$ | `Contract Upgrade Campaign` | **HIGH / CRITICAL** | **STRICTLY PRUNED** if customer is already on `One year` or `Two year` contract! | `recommendation_engine.py:209` |
| **RULE 2** | `PaymentMethod == 'Electronic check'` | `Autopay Migration Offer ($10 Credit)` | **HIGH** | **STRICTLY PRUNED** if `'automatic'` is present in payment method! | `recommendation_engine.py:225` |
| **RULE 3** | `MonthlyCharges >= $65.00` & pricing weight $>0.05$ | `Plan & Rate Optimization Review` | **HIGH** | **STRICTLY PRUNED** if `MonthlyCharges < $65.00`. | `recommendation_engine.py:242` |
| **RULE 4** | `tenure <= 6` months | `Early-Tenure Onboarding & Care` | **CRITICAL** | **STRICTLY PRUNED** if `tenure > 6` months. | `recommendation_engine.py:265` |
| **RULE 5** | `InternetService == 'Fiber optic'` | `Fiber QoS & Technical Line Audit` | **MEDIUM** | **STRICTLY PRUNED** if customer does not have fiber optic service. | `recommendation_engine.py:280` |
| **RULE 11** | $\hat{p}_{\text{churn}} < \tau^*$ and 0 candidates triggered | `Customer Loyalty Appreciation Perk` | **LOW** | Only granted to low-risk loyal subscribers. | `recommendation_engine.py:352` |

---

## 18. RECOMMENDATION VALIDATION AUDIT

### Audit Finding: **RECOMMENDATION EFFECTIVENESS HAS NOT BEEN EXPERIMENTALLY VALIDATED**
- **Validation Status:** There is **NO** live A/B testing, randomized field trial, corporate ledger data, or historical intervention outcome tracking in the repository.
- **Analytical Simulation:** The repository provides an *analytical retention simulation* on the $N=1,600$ hold-out test set (`results/simulated_retention_roi.json`) that computes a $+373.6\%$ ROI ($151,850$ net benefit) under assumed commercial parameters ($C_{\text{contact}} = \$50$, success rate $r = 50\%$, $\text{CLV} = \$500$).
- **Directive for Paper:** The paper **cannot** claim that the recommendations reduce churn in practice. It must state clearly that the recommendation engine is an algorithmic decision-support tool whose operational uplift requires future randomized controlled trials.

---

## 19. WHAT-IF SIMULATOR AUDIT

### Implementation Verification
- **Code Location:** `backend/main.py` lines 612–683 (`/api/whatif/simulate`).
- **Input Variables:** 6 core profile fields (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`).
- **Operation:** Evaluates counterfactual scenario options (e.g., mutating `Contract` from `Month-to-month` to `One year` and `Two year`).
- **Model Re-run:** **YES**. The actual ensemble classifier (`predict_probability`) is re-evaluated for every scenario permutation.
- **SHAP Re-run:** **YES**. `robust_explainer.explain()` is called dynamically on the best mutated profile.
- **Recommendations Re-run:** **YES**. `recommendation_engine.generate_recommendations()` re-ranks retention cards for the mutated state.
- **Causal vs. Sensitivity Status:** The What-If simulator is strictly a **model-based sensitivity / counterfactual feature permutation tool**. It does **NOT** compute causal effects (no do-calculus, DAGs, or propensity matching). The paper must describe it as a sensitivity analysis tool.

---

## 20. FEATURE COUNT RECONCILIATION

**Reviewer Comment:**  
> *"The reviewer found: 20 predictors, 6 core attributes, 27 features. Reconcile where each number comes from."*

### Complete Feature Count Reconciliation

| Feature Count | Context / Stage | Exact Definition & Contents | File / Evidence |
|:---:|:---|:---|:---|
| **20 Predictors** | Raw CSV Columns in Dataset | All non-identifier, non-target columns in `data/balanced_telco_churn.csv`: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` (and `target` if counted). | `data/balanced_telco_churn.csv` |
| **6 Core Attributes** | Production User Ingestion Interface | The minimal operational inputs captured by CRM forms and REST endpoints: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`. | `SingleCustomer.jsx`, `backend/main.py` |
| **27 Transformed Features** | Final Model Input Vector | The full mathematical vector fed into the ML models (`feature_engineering_v2.py`):<br>• **11 Scaled Continuous:** `tenure`, `MonthlyCharges`, `TotalCharges`, `AverageMonthlySpend`, `ChargeIncrease`, `ChargeIncreaseRatio`, `TenureSquared`, `TenureLog`, `tenure_x_MonthlyCharges`, `tenure_x_IsMonthToMonth`, `MonthlyCharges_x_IsMonthToMonth`<br>• **9 Binary Flags:** `IsNewCustomer`, `IsLongTermCustomer`, `IsMonthToMonth`, `IsLongTermContract`, `IsElectronicCheck`, `HighRiskCombo`, `VeryHighRisk`, `FiberOptic_x_MonthToMonth`, `FiberOptic_x_ElectronicCheck`<br>• **7 One-Hot Dummies:** `Contract_One year`, `Contract_Two year`, `InternetService_Fiber optic`, `InternetService_No`, `PaymentMethod_Credit card (auto)`, `PaymentMethod_Electronic check`, `PaymentMethod_Mailed check`. | `models/churn_v2_feature_columns.pkl`, `feature_engineering_v2.py` |
| **83 Features** | Historical Dataset 1 (IBM Telco) | One-hot encoded feature space used in earlier IBM Telco experiments on 2026-09-16. | `results/model_comparison.json` |
| **47 Features** | Alternative Feature Engineering | Earlier experimental feature engineering script. | `ml/feature_engineering.py` |
| **8 Features** | Feature Selection Experiment | Reduced feature subset tested in ablation studies. | `REDUCED_FEATURE_MAPPING.md` |

---

## 21. FIGURES AND VISUAL EVIDENCE AUDIT

### Inventory of Existing Paper Figures

| Figure File | Location | Purpose | Data Source | Actual vs Placeholder | IEEE Paper Suitability |
|:---|:---|:---|:---|:---:|:---:|
| `fig1_system_architecture.png` | `paper/figures/` | End-to-end framework architecture | High-level system design | **Actual** | **HIGH** |
| `fig2_ml_workflow.png` | `paper/figures/` | Leakage-safe CV and thresholding workflow | Methodological schematic | **Actual** | **HIGH** |
| `fig3_xai_recommendation_workflow.png` | `paper/figures/` | State-aware recommendation flow | Logic schematic | **Actual** | **HIGH** |
| `cost_vs_threshold_curves.png` | `paper/figures/` | Expected cost vs. threshold $\tau$ | `results/phase3_threshold_metrics_summary.csv` | **Actual** | **HIGH** |
| `cost_sensitivity_curves.png` | `paper/figures/` | Optimal threshold across 4 cost ratios | `results/phase3_threshold_metrics_summary.csv` | **Actual** | **HIGH** |
| `shap_feature_importance_bar.png` | `paper/figures/` | Top 15 Mean \|SHAP\| feature rankings | `results/global_shap_importance.csv` | **Actual** | **HIGH** |
| `shap_summary_beeswarm.png` | `paper/figures/` | Global SHAP beeswarm plot | `scripts/phase4_explainability_decision_support.py` | **Actual** | **HIGH** |
| `local_xai_customer_a.png` | `paper/figures/` | Customer A side-by-side SHAP vs LIME | `results/case_study_explanations.json` | **Actual** | **HIGH** |
| `fig7_confusion_matrix_proposed.png` | `paper_figures/` | Stacking confusion matrix ($N=1,600$) | `models/model_comparison_results.json` | **Actual** | **HIGH** |
| `roc_curves.png` | `results/` | ROC curves for all models | Test split evaluation | **Actual** | **HIGH** |
| `pr_curves.png` | `results/` | PR curves for all models | Test split evaluation | **Actual** | **HIGH** |
| `fig2_model_rf_metrics.png` | `paper_figures/` | Single-model RF metric bars | Redundant metrics | Placeholder/Redundant | **DO NOT USE** |
| `fig3_model_gb_metrics.png` | `paper_figures/` | Single-model GBDT metric bars | Redundant metrics | Placeholder/Redundant | **DO NOT USE** |
| `fig4_model_stacking_metrics.png` | `paper_figures/` | Stacking metric bars | Redundant metrics | Placeholder/Redundant | **DO NOT USE** |

---

## 22. OLD VS CURRENT EXPERIMENTS

### Table F — Historical vs. Final Experiment Reconciliation

| Experiment ID | Dataset | Feature Space | Models Evaluated | Accuracy | Recall | F1 | ROC-AUC | Threshold Policy | Status in Final Paper |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Experiment Epoch 1** (2026-09-16) | `WA_Fn-UseC_-Telco-Customer-Churn.csv` (IBM Telco, $N=7,043$) | 83 Features | RF, XGBoost, LightGBM, Stacking | 76.22% (Stacking) | 79.68% (Stacking) | 64.02% | 84.60% | Default $\tau=0.50$; Recall mode $\tau=0.29$ | **HISTORICAL BASELINE ONLY** (Not final benchmark) |
| **Experiment Epoch 2** (2026-09-22) | `balanced_telco_churn.csv` (Synthetic, $N=8,000$, Test $N=1,600$) | 27 Features | RF, XGBoost, LightGBM, Stacking | 95.44% (Stacking) | 96.25% (Stacking), 97.63% (XGB) | 95.47% | 98.94% | Tuned on 5-fold Training OOF ($\tau^*=0.26$ to $0.38$) | **SINGLE-SPLIT PRODUCTION EVALUATION** |
| **Experiment Epoch 3** (2026-09-23/24) | `balanced_telco_churn.csv` (Synthetic, $N=8,000$, 10 Outer Folds) | 27 Features | RF, XGBoost, LightGBM, Stacking | $94.96\% \pm 0.71\%$ | $94.74\% \pm 0.89\%$ (Stack), $94.80\%$ (XGB) | $94.95\%$ | $98.88\%$ | 10:1 Cost-Optimal on Inner OOF ($\tau^*=0.04$ to $0.10$) | **FINAL PAPER REPEATED CV BENCHMARK** |

---

## 23. SOFTWARE AND HARDWARE ENVIRONMENT

### Verified Environment Details
*Source:* `results/hardware_specs.json`, `requirements.txt`, `reproducibility/requirements.txt`, `frontend/package.json`

- **Operating System:** Microsoft Windows 10 (Build 10.0.26200, 64-bit)
- **Processor:** Intel64 Family 6 Model 126 Stepping 5 (GenuineIntel), 8 Virtual Cores
- **Python Version:** `3.11.9` (64-bit)
- **Verified Core Package Versions:**
  - `numpy`: `2.4.6`
  - `pandas`: `2.3.3`
  - `scikit-learn`: `1.9.0`
  - `scipy`: `1.17.1`
  - `xgboost`: `3.2.0`
  - `lightgbm`: `4.7.0`
  - `catboost`: `1.2.10`
  - `imbalanced-learn`: `0.14.2`
  - `shap`: `0.51.0`
  - `lime`: `0.2.0.1`
  - `joblib`: `1.6.0`
  - `matplotlib`: `3.11.1`
  - `fastapi`: `0.141.1`
  - `uvicorn`: `0.41.0`
- **Frontend Stack:**
  - `react`: `^19.2.8`
  - `react-dom`: `^19.2.8`
  - `react-router-dom`: `^7.18.2`
  - `recharts`: `^3.10.1`
  - `vite`: `^8.2.0`
- **Database:** SQLite 3 (`churn_predictions.db`)

---

## 24. REPRODUCIBILITY AUDIT

### Table G — Reproducibility Checklist

| Item | Available? | Evidence Location | Missing / Cautionary Information |
|:---|:---:|:---|:---|
| **Raw Dataset** | **YES** | `data/balanced_telco_churn.csv` ($N=8,000$) | Must be disclosed as synthetic benchmark with 5% noise. |
| **Dataset Generator** | **YES** | `scripts/generate_balanced_dataset.py` | Generates identical dataset with `random_state=42`. |
| **Master Script** | **YES** | `reproducibility/reproduce.py` | Complete one-command reproduction suite (`python reproduce.py --quick`). |
| **Pinned Dependencies** | **YES** | `reproducibility/requirements.txt` | Pinned exact versions for all 15 core ML libraries. |
| **Execution Manual** | **YES** | `README_REPRODUCTION.md` | Step-by-step reproduction instructions. |
| **Pre-trained Models** | **YES** | `models/churn_v2_*.pkl` | Pre-trained RF, XGBoost, LightGBM, and Stacking models. |
| **Hardware Metadata** | **YES** | `results/hardware_specs.json` | Hardware and OS verified. |
| **Statistical Tests** | **PARTIAL** | Mean $\pm$ SD across 10 folds available in `phase2_repeated_cv_summary.csv` | Formal McNemar/DeLong hypothesis test scripts not included. |

---

## 25. PAPER CLAIM AUDIT

| Paper Claim | Evaluated Status | Empirical Finding |
|:---|:---:|:---|
| *"8,000 verified customer profiles"* | **CONTRADICTED** | Dataset is 100% synthetic, generated by `scripts/generate_balanced_dataset.py`. Must be rephrased as a "domain-consistent benchmark". |
| *"Stacking significantly outperforms shallow baselines (+16.4% over LR, +11.4% over SVM)"* | **CONTRADICTED** | When evaluated fairly with all 27 engineered features in Phase 2 CV, LR achieves 93.73% and SVM achieves 94.60%. Stacking (94.96%) performs comparably to XGBoost (94.98%). |
| *"Cost-sensitive thresholding reduces business cost"* | **CONFIRMED** | Proved in Phase 3 repeated CV: shifting from $\tau=0.50$ to $\tau^*=0.04$ reduces expected cost by 51.2% under a 10:1 cost matrix. |
| *"Dual explainability guarantees consensus"* | **CONTRADICTED** | Quantitative concordance across $N=200$ instances proves Spearman correlation is only $0.2421 \pm 0.1091$ and Top-3 overlap is only $22.83\%$. SHAP and LIME diverge substantially. |
| *"Multimodal analytics represent customer lifecycle"* | **UNSUPPORTED** | Zero NLP, audio, or sentiment features exist. The project is 100% structured tabular. Multimodal text was a copy-paste remnant and must be removed. |
| *"Recommendations reduce churn by +373.6% ROI"* | **PARTIALLY SUPPORTED** | The $+373.6\%$ ROI is strictly an analytical simulation estimate on 1,600 test accounts under assumed costs ($C_{\text{FN}}=\$500, C_{\text{FP}}=\$50, r=0.50$), not empirical intervention accounting. |
| *"Real-time CRM viability"* | **CONFIRMED** | Single prediction latency averages 100.15 ms and local SHAP executes in 6.26 ms. |

---

## 26. RESPONSE TO CLAUDE REVIEW COMMENTS

### 1. Result Provenance
- **Evidence Found in Project:** `models/model_comparison_results.json` generated on 2026-09-22 by `scripts/train_churn_v2.py`.
- **Status:** **CONFIRMED AS LOCALLY GENERATED**.
- **Exact Supporting File:** `models/model_comparison_results.json`.
- **What Paper Can Claim:** All final metrics were generated on the author's local evaluation environment.
- **What Is Missing:** Clear attribution in the manuscript indicating that single-split numbers correspond to `test_original.csv` ($N=1,600$).

### 2. Cost-Sensitive Thresholding
- **Evidence Found in Project:** `scripts/phase3_threshold_cost_analysis.py`, `results/phase3_threshold_metrics_summary.csv`.
- **Status:** **CONFIRMED**.
- **Exact Supporting File:** `results/phase3_threshold_metrics_summary.csv`.
- **What Paper Can Claim:** Cost-sensitive thresholding under asymmetric penalties ($C_{\text{FN}}=\$500, C_{\text{FP}}=\$50$) reduces enterprise loss by 51.2% to 52.8%.

### 3. Threshold Tuning Dataset
- **Evidence Found in Project:** In `train_churn_v2.py`, tuned on 5-fold training OOF probabilities ($N=6,400$). In `phase3_threshold_cost_analysis.py`, tuned on inner training OOF across 10 folds.
- **Status:** **CONFIRMED LEAKAGE-FREE**.
- **Exact Supporting File:** `scripts/phase3_threshold_cost_analysis.py` lines 118–155.
- **What Paper Can Claim:** Decision thresholds were tuned strictly on training-side out-of-fold distributions, with zero test exposure.

### 4. XGBoost Hyperparameters
- **Evidence Found in Project:** `n_estimators=200, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, min_child_weight=3, gamma=0.1, reg_alpha=0.05, reg_lambda=1.0`.
- **Status:** **CONFIRMED IN CODE**.
- **Exact Supporting File:** `scripts/train_churn_v2.py` lines 217–230.
- **What Paper Can Claim:** Provide complete hyperparameter specification in Table B.

### 5. LightGBM Hyperparameters
- **Evidence Found in Project:** `n_estimators=200, learning_rate=0.06, num_leaves=31, max_depth=6, min_child_samples=20, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1.0`.
- **Status:** **CONFIRMED IN CODE**.
- **Exact Supporting File:** `scripts/train_churn_v2.py` lines 232–245.
- **What Paper Can Claim:** Provide complete hyperparameter specification in Table B.

### 6. SHAP Evidence
- **Evidence Found in Project:** TreeExplainer evaluated on 1,600 test subscribers; global ranking in `results/global_shap_importance.csv`; plots in `paper/figures/shap_feature_importance_bar.png` and `shap_summary_beeswarm.png`.
- **Status:** **CONFIRMED**.
- **Exact Supporting File:** `results/global_shap_importance.csv`.

### 7. LIME Evidence
- **Evidence Found in Project:** `lime.lime_tabular.LimeTabularExplainer` evaluated across 200 customers and case studies; latency measured at 620.21 ms.
- **Status:** **CONFIRMED**.
- **Exact Supporting File:** `results/case_study_explanations.json`, `paper/figures/local_xai_customer_a.png`.

### 8. SHAP/LIME Concordance
- **Evidence Found in Project:** Quantitative evaluation across $N=200$ test customers; Spearman $\rho = 0.2421 \pm 0.1091$, Top-3 overlap $22.83\%$.
- **Status:** **CONFIRMED**.
- **Exact Supporting File:** `results/shap_lime_concordance.json`.
- **What Paper Can Claim:** Disprove naive assumptions of XAI consensus; document mathematical reasons for surrogate divergence.

### 9. Balanced Synthetic Data Limitation
- **Evidence Found in Project:** `scripts/generate_balanced_dataset.py` injects 5% label noise, imposing a theoretical Bayes ceiling of $\approx 95\%$.
- **Status:** **CONFIRMED AS LIMITATION**.
- **Exact Supporting File:** `scripts/generate_balanced_dataset.py` line 16.

### 10. Balanced Prior Probability Issue
- **Evidence Found in Project:** 50/50 test set inflates precision to 94.7%; precision drops to 48.4% when evaluated at 5.0% realistic prevalence.
- **Status:** **CONFIRMED AS LIMITATION**.
- **Exact Supporting File:** `paper/DATA_AUDIT.md` Section 2.3.

### 11. Statistical Validation
- **Evidence Found in Project:** Repeated Stratified CV (10 folds) with Mean $\pm$ SD available in `results/phase2_repeated_cv_summary.csv`.
- **Status:** **PARTIALLY CONFIRMED** (Paired hypothesis tests not pre-computed).

### 12. Novelty
- **Evidence Found in Project:** Framework integrates cost-sensitive thresholding, state-aware pruning, empirical XAI concordance, and real-time production viability.
- **Status:** **CONFIRMED**.
- **What Paper Can Claim:** Frame novelty around operational decision support rather than stacking accuracy superiority.

### 13. Recommendation Scoring Formula
- **Evidence Found in Project:** $\text{Action Score} = (\text{Risk} \times \text{Customer Risk} \times \text{Relevance} \times \text{Impact}) / \text{Cost}$.
- **Status:** **CONFIRMED**.
- **Exact Supporting File:** `backend/recommendation_engine.py` lines 384–394.

### 14. Recommendation Trigger Rules
- **Evidence Found in Project:** 10 cataloged rules with strict state-aware pruning (e.g., pruning contract upgrade if customer already has a multi-year term).
- **Status:** **CONFIRMED**.
- **Exact Supporting File:** `backend/recommendation_engine.py` lines 190–380.

### 15. Recommendation Validation
- **Evidence Found in Project:** Analytical simulation only ($+373.6\%$ ROI).
- **Status:** **NOT EXPERIMENTALLY VALIDATED IN FIELD**.
- **Directive:** Disclose as an analytical simulation, not a real-world trial.

### 16. Related Work / Reference Verification
- **Evidence Found in Project:** 17 canonical peer-reviewed references verified in `paper/refs.bib`. Removed copy-pasted citations about sentiment analysis.
- **Status:** **CONFIRMED**.
- **Exact Supporting File:** `paper/refs.bib`.

### 17. Rounding Inconsistencies
- **Evidence Found in Project:** 96.25% rounded to 96.3%; 95.44% rounded to 95.4%; 95.47% rounded to 95.5%.
- **Status:** **CONFIRMED AS ROUNDING**.

### 18. Chart Axis
- **Evidence Found in Project:** High-resolution PNG figures generated with full labels, units, and axes.
- **Status:** **CONFIRMED**.
- **Exact Supporting File:** `paper/figures/`.

### 19. Feature-Count Inconsistency
- **Evidence Found in Project:** Reconciled: 20 raw dataset predictors $\to$ 6 operational inputs $\to$ 27 final engineered features.
- **Status:** **CONFIRMED & RECONCILED**.

### 20. Figures
- **Evidence Found in Project:** 8 publication-ready figures available in `paper/figures/`.
- **Status:** **CONFIRMED**.

### 21. IEEE Template
- **Evidence Found in Project:** Standard two-column conference format; removed ICERCS 2025 conference banner and fake DOI.
- **Status:** **CONFIRMED**.
- **Exact Supporting File:** `paper/main.tex`.

### 22. Author Information
- **Evidence Found in Project:** `Explainable_Ensemble_Learning_Telecom_Churn_Final.tex` identifies:
  1. Abinash J. (Kongu Engineering College, abinash.cse@kongu.edu)
  2. Chandru Vignesh C. (Vellore Institute of Technology, chandruvignesh@vit.ac.in)
  3. Deepak Raja V. (Kongu Engineering College, deepakraja.tech@gmail.com)
  `paper/main.tex` formats this for double-blind review as `Anonymous Authors`.
- **Status:** **CONFIRMED**.

---

## 27. WHAT WE ACTUALLY HAVE VS WHAT WE STILL NEED

### ALREADY AVAILABLE:
✓ Full local provenance of all reported test metrics in `models/model_comparison_results.json`  
✓ Verified dataset file `data/balanced_telco_churn.csv` ($N=8,000$, 0 nulls, 0 duplicates)  
✓ Complete dataset generation code in `scripts/generate_balanced_dataset.py`  
✓ Full 6-to-27 feature engineering pipeline in `feature_engineering_v2.py`  
✓ Verified hyperparameters for RF, XGBoost, LightGBM, and Stacking meta-learner  
✓ Repeated cross-validation results (10 outer folds) with Mean $\pm$ SD in `results/phase2_repeated_cv_summary.csv`  
✓ Cost-sensitive threshold evaluation across 4 economic ratios in `results/phase3_threshold_metrics_summary.csv`  
✓ Quantitative SHAP-LIME concordance evaluation ($N=200$) in `results/shap_lime_concordance.json`  
✓ Mathematical scoring formula and state-aware pruning rules in `backend/recommendation_engine.py`  
✓ 8 publication-ready figures in `paper/figures/`  
✓ Measured latency benchmarks (single prediction, batch prediction, SHAP, LIME)  
✓ Clean IEEE conference LaTeX manuscript (`paper/main.tex`) and verified BibTeX (`paper/refs.bib`)  
✓ Master reproduction script (`reproducibility/reproduce.py`) and pinned requirements  

### AVAILABLE BUT NEEDS VERIFICATION / RECONCILIATION:
⚠ Discrepancy between hyperparameters in production code vs. text of earlier paper drafts (e.g. RF trees: 250 in code vs 300 in `main.tex`; XGBoost depth: 5 in code vs 6 in `main.tex`). Table B in this audit reconciles these.  
⚠ Hardcoded latency numbers in earlier draft Table VII (8.4ms, 3.2ms) vs empirically measured pipeline benchmark (100.15ms). The empirical benchmark must be reported.  

### NOT AVAILABLE:
✗ Experimental field validation / A/B testing of customer retention recommendations.  
✗ Historical intervention outcome data.  
✗ Formal statistical significance hypothesis test values (McNemar, DeLong).  
✗ A file named `balanced_dataset_churn.csv` on disk (the real file is `balanced_telco_churn.csv`).  

### REQUIRES NEW EXPERIMENT (IF USER CHOOSES TO PURSUE):
→ Running McNemar's test and DeLong's ROC-AUC test across models if reviewers demand formal $p$-values.  
→ Recalibrating probabilities to realistic 2%–5% churn prevalence via Platt scaling or isotonic calibration.  

### DO NOT FABRICATE:
→ **DO NOT** claim the dataset contains "8,000 verified customer profiles" (it is synthetic).  
→ **DO NOT** claim Stacking outperforms XGBoost (XGBoost achieves 94.98% accuracy and 94.80% recall vs Stacking's 94.96% accuracy and 94.74% recall).  
→ **DO NOT** claim shallow baselines score 79% and 84% on the final feature space (LR achieves 93.73% and SVM achieves 94.60%).  
→ **DO NOT** claim SHAP and LIME visually agree (they exhibit $\rho=0.2421$ and Top-3 overlap of only 22.83%).  
→ **DO NOT** claim the recommendation engine was clinically validated on live carrier customers.  
→ **DO NOT** include multimodal, sentiment, or speech analytics claims.  

---
*End of Audit Dossier. Prepared strictly from verifiable codebase artifacts.*

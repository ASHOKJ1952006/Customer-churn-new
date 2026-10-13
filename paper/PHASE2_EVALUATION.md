# Phase 2: Rigorous Leakage-Safe Evaluation Report

**Mandatory Scope Declaration:**  
> **"Phase 2 rigorous evaluation was conducted exclusively on `balanced_telco_churn.csv`."**  
> No external datasets (such as IBM Telco `WA_Fn-UseC_-Telco-Customer-Churn.csv`, Cell2Cell, or Orange) were mixed, merged, or used for cross-validation in this phase.

---

## 1. Experimental Setup & Protocol

| Parameter | Protocol Specification |
|:---|:---|
| **Evaluated Dataset** | `data/balanced_telco_churn.csv` |
| **Dataset Dimensions** | 8,000 customer records $\times$ 22 columns (20 tabular features + `Churn` + `target`) |
| **Target Distribution** | Exactly 4,000 Loyal (Class 0, 50.0%) and 4,000 Churn (Class 1, 50.0%) |
| **Missing Values** | 0 null cells across all 8,000 records |
| **Cross-Validation Strategy** | Repeated Stratified K-Fold Cross-Validation (`RepeatedStratifiedKFold`) |
| **Number of Folds ($k$)** | 5 folds per repeat |
| **Number of Repeats ($r$)** | 2 repeats (10 independent outer validation folds total) |
| **Fold Partition Size** | Train Fold: 6,400 samples (50.0% Churn), Validation Fold: 1,600 samples (50.0% Churn) |
| **Resampling Strategy** | No synthetic oversampling (SMOTE) applied; dataset is natively balanced at 50/50 |
| **Random Seeds** | Base seed = 42; fold-specific seeds $42 + \text{fold\_idx}$ for model initialization |
| **Decision Threshold** | Standard symmetric threshold $\tau = 0.50$ across all models for strict parity |

---

## 2. Leakage-Safe Pipeline Architecture

To prevent optimistic bias and data snooping, the pipeline enforces strict isolation:
1. **Split-First Protocol:** Partitioning into train ($N=6,400$) and validation ($N=1,600$) occurs prior to any parameter estimation.
2. **Deterministic Feature Engineering:** The 6 core operational inputs (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`) are transformed into the canonical 27-dimensional feature space using row-independent mathematical formulas.
3. **Training-Fold-Only Scaling:** `StandardScaler` is fitted **exclusively on the training fold** of each split. The fitted transformation parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are applied out-of-sample to the validation fold. Zero validation statistics leak into scaling.
4. **Nested Out-of-Fold Stacking:** The Stacking Ensemble meta-learner (`LogisticRegression`) is trained on out-of-fold probability vectors generated strictly via an **inner 5-fold cross-validation loop** executed entirely inside the training fold. The meta-learner never observes validation fold labels during training.

---

## 3. Model Hyperparameter Specifications

1. **Logistic Regression:** $C=1.0$, `solver='lbfgs'`, `max_iter=1000`, `random_state=seed`
2. **Decision Tree:** `max_depth=6`, `min_samples_split=10`, `min_samples_leaf=5`, `criterion='gini'`, `random_state=seed`
3. **SVM (RBF):** $C=1.0$, `kernel='rbf'`, `probability=True`, `gamma='scale'`, `random_state=seed`
4. **Random Forest:** `n_estimators=200`, `max_depth=10`, `min_samples_split=6`, `min_samples_leaf=2`, `max_features='sqrt'`, `random_state=seed`, `n_jobs=-1`
5. **XGBoost:** `n_estimators=150`, `max_depth=5`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `eval_metric='logloss'`, `random_state=seed`, `n_jobs=-1`
6. **LightGBM:** `n_estimators=150`, `max_depth=6`, `num_leaves=31`, `learning_rate=0.08`, `subsample=0.85`, `colsample_bytree=0.85`, `verbose=-1`, `random_state=seed`, `n_jobs=-1`
7. **CatBoost:** `iterations=150`, `depth=6`, `learning_rate=0.08`, `thread_count=-1`, `verbose=0`, `random_seed=seed`
8. **MLPClassifier (Neural Baseline):** `hidden_layer_sizes=(64, 32)`, `activation='relu'`, `alpha=0.01`, `max_iter=300`, `early_stopping=True`, `n_iter_no_change=15`, `random_state=seed`
9. **Stacking Ensemble:** Base learners (RF, XGB, LGBM) combined via inner 5-fold OOF probability vectors into a regularized `LogisticRegression` ($C=1.0$) meta-learner.

---

## 4. Phase 2 Repeated Cross-Validation Results

The table below presents the empirical evaluation across all 10 validation folds. All metrics are reported as **$\text{Mean} \pm \text{Standard Deviation}$** at $\tau = 0.50$. Recall is reported prominently to reflect churn detection prioritization.

| Model | Recall (Sensitivity) | Accuracy | Precision | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 93.83% $\pm$ 0.82% | 93.73% $\pm$ 0.70% | 93.65% $\pm$ 1.00% | 93.73% $\pm$ 0.69% | 98.09% $\pm$ 0.28% | 97.98% $\pm$ 0.30% | 0.0492 $\pm$ 0.0044 |
| **Decision Tree** | 92.64% $\pm$ 1.22% | 93.73% $\pm$ 0.71% | 94.75% $\pm$ 1.88% | 93.66% $\pm$ 0.65% | 97.41% $\pm$ 0.42% | 96.48% $\pm$ 0.62% | 0.0496 $\pm$ 0.0042 |
| **SVM (RBF)** | 93.86% $\pm$ 0.91% | 94.60% $\pm$ 0.62% | 95.27% $\pm$ 0.69% | 94.56% $\pm$ 0.64% | 98.09% $\pm$ 0.32% | 98.06% $\pm$ 0.42% | 0.0448 $\pm$ 0.0049 |
| **Random Forest** | 94.00% $\pm$ 0.98% | 94.61% $\pm$ 0.64% | 95.17% $\pm$ 0.70% | 94.58% $\pm$ 0.66% | 98.65% $\pm$ 0.26% | 98.62% $\pm$ 0.31% | 0.0412 $\pm$ 0.0039 |
| **MLPClassifier (Neural)** | 93.66% $\pm$ 1.10% | 94.53% $\pm$ 0.73% | 95.33% $\pm$ 0.88% | 94.48% $\pm$ 0.75% | 98.62% $\pm$ 0.23% | 98.58% $\pm$ 0.27% | 0.0426 $\pm$ 0.0046 |
| **LightGBM** | 94.76% $\pm$ 0.81% | 94.81% $\pm$ 0.55% | 94.85% $\pm$ 0.82% | 94.80% $\pm$ 0.55% | 98.86% $\pm$ 0.28% | 98.78% $\pm$ 0.36% | 0.0391 $\pm$ 0.0042 |
| **CatBoost** | 94.78% $\pm$ 0.87% | 94.96% $\pm$ 0.64% | 95.14% $\pm$ 0.95% | 94.95% $\pm$ 0.64% | 98.94% $\pm$ 0.25% | 98.89% $\pm$ 0.31% | 0.0376 $\pm$ 0.0043 |
| **Stacking Ensemble** | 94.74% $\pm$ 0.89% | 94.96% $\pm$ 0.71% | 95.16% $\pm$ 0.99% | 94.95% $\pm$ 0.71% | 98.88% $\pm$ 0.26% | 98.82% $\pm$ 0.32% | 0.0400 $\pm$ 0.0049 |
| **XGBoost** | 94.80% $\pm$ 0.87% | 94.98% $\pm$ 0.66% | 95.15% $\pm$ 0.97% | 94.97% $\pm$ 0.65% | 98.90% $\pm$ 0.25% | 98.84% $\pm$ 0.32% | 0.0383 $\pm$ 0.0041 |

*Note: Models are sorted by increasing F1-score.*

---

## 5. Cumulative Confusion Matrices Across All 10 Folds ($N_{\text{total}} = 16,000$)

Each sample in `balanced_telco_churn.csv` ($N=8,000$) is evaluated exactly twice across the 2 repeats of 5-fold CV (8,000 actual non-churners and 8,000 actual churners evaluated out-of-fold):

| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | Total Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 7,490 | 510 | 494 | 7,506 | 16,000 |
| **Decision Tree** | 7,585 | 415 | 589 | 7,411 | 16,000 |
| **SVM (RBF)** | 7,627 | 373 | 491 | 7,509 | 16,000 |
| **Random Forest** | 7,618 | 382 | 480 | 7,520 | 16,000 |
| **MLPClassifier** | 7,632 | 368 | 507 | 7,493 | 16,000 |
| **LightGBM** | 7,588 | 412 | 419 | 7,581 | 16,000 |
| **CatBoost** | 7,612 | 388 | 418 | 7,582 | 16,000 |
| **Stacking Ensemble** | 7,614 | 386 | 421 | 7,579 | 16,000 |
| **XGBoost** | 7,613 | 387 | 416 | 7,584 | 16,000 |

---

## 6. Comparison: Existing Single-Split Results vs. New Repeated-CV Results

To ensure transparency, we distinguish the project's two evaluation protocols:

### Protocol A: Previous Single Train/Test Split Experiment
*   **Methodology:** Single static 80/20 train/test split ($N_{\text{train}}=6,400$, $N_{\text{test}}=1,600$).
*   **Threshold Selection:** Each model was evaluated at an individually tuned threshold ($\tau^* = 0.26$ for Stacking, $0.29$ for XGB, $0.32$ for LGBM, $0.38$ for RF) selected to maximize recall subject to $\text{Acc} \ge 90\%$.
*   **Reported Test Set Metrics:**
    *   Stacking ($\tau^*=0.26$): Accuracy 95.4%, Recall 96.3%, Precision 94.7%, F1 95.5%, ROC-AUC 98.9%, Brier 0.0362
    *   XGBoost ($\tau^*=0.29$): Accuracy 94.9%, Recall 97.6%, Precision 92.5%, F1 95.0%, ROC-AUC 99.0%, Brier 0.0347
    *   Random Forest ($\tau^*=0.38$): Accuracy 95.4%, Recall 96.3%, Precision 94.6%, F1 95.4%, ROC-AUC 98.7%, Brier 0.0369
*   **Shallow Baselines in Old Paper Table VI:** Stated as Logistic Regression 0.79, Decision Tree 0.82, SVM 0.84.

### Protocol B: New Phase 2 Repeated Stratified K-Fold Protocol
*   **Methodology:** Leakage-safe 5-Fold $\times$ 2-Repeat CV across all 8,000 instances; preprocessing fitted strictly inside each training fold; symmetric baseline threshold $\tau = 0.50$ applied uniformly across all models.
*   **Empirical Findings:**
    *   **The Baseline Gap Disproved:** When shallow models (Logistic Regression, Decision Tree, SVM) receive the exact same 27-dimensional feature engineering and proper scaling, Logistic Regression attains **93.73% $\pm$ 0.70%** (not 0.79) and SVM attains **94.60% $\pm$ 0.62%** (not 0.84). The claim of a "+16.4% improvement" in the old manuscript was an artifact of comparing feature-engineered tree ensembles against unengineered, untuned baselines.
    *   **The Stacking Equivalence:** Stacking Ensemble accuracy (**94.96% $\pm$ 0.71%**) is statistically indistinguishable from CatBoost (**94.96% $\pm$ 0.64%**) and XGBoost (**94.98% $\pm$ 0.66%**). XGBoost yields slightly fewer false negatives (FN=416 vs 421) and CatBoost delivers the lowest Brier calibration error (0.0376 vs 0.0400).
    *   **Noise Ceiling Validation:** All non-linear models (RF, XGB, LGBM, CatBoost, MLP, Stacking) plateau within the narrow band of **94.5% to 95.0% accuracy**, aligning precisely with the theoretical 5% noise ceiling ($\approx 95.0\%$) built into `generate_balanced_dataset.py`.

---

## 7. Leakage Checks & Methodological Limitations

1. **Leakage Verification:**
   * Scaler parameters ($\mu, \sigma$) were verified to be computed exclusively on `X_train` within each fold; `X_val` was transformed out-of-sample.
   * Inner OOF cross-validation for stacking ensured the meta-learner received decorrelated probability predictions.
2. **Important Limitations to Report in Paper:**
   * **Synthetic Bayes Ceiling:** Because `balanced_telco_churn.csv` incorporates 5.0% injected trait noise, no model can surpass $\approx 95.0\%$ generalization accuracy without overfitting to noise.
   * **Equal Model Capability:** Advanced gradient boosters (XGBoost, CatBoost) perform equivalently to the heterogeneous stacking meta-learner. The primary value proposition of the pipeline lies in its **decision-support layer, threshold tuning, and dual explainability (SHAP/LIME)** rather than stacking performance gains.

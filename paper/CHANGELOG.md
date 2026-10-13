# Paper & Pipeline Changelog

This document logs all changes made across project phases, tracking before-and-after numbers, methodology adjustments, and paper repositioning.

---

## [Phase 0: Project Audit] - September 2026
*   Conducted deep audit of codebase, data pipelines, backend recommendation engine, and paper files.
*   Generated `paper/AUDIT.md` detailing pipeline structure, metric sources, and 9 critical code-paper discrepancies.
*   Generated `paper/OPEN_ISSUES.md` tracking unverified claims and citation risks.

---

## [Phase 1: Data & Feature Specification Audit] - September 2026
*   Produced [`paper/DATA_AUDIT.md`](file:///d:/ABI/YEAR%204/SEM%207/churn/Customer-churn-new/paper/DATA_AUDIT.md) providing full itemized answers A through R.
*   Produced [`paper/DATASET_SECTION_DRAFT.md`](file:///d:/ABI/YEAR%204/SEM%207/churn/Customer-churn-new/paper/DATASET_SECTION_DRAFT.md) ready for IEEE conference insertion.
*   Verified exact serialized feature dimensions across code and model files: strictly 6 raw inputs $\to$ 27 model dimensions (11 scaled continuous + 9 binary risk flags + 7 one-hot categories).
*   Quantified synthetic noise ceiling: a simple Decision Tree on only 3 features (`tenure`, `IsMonthToMonth`, `IsElectronicCheck`) attains 92.84% accuracy, and on all 27 features attains 94.51%, directly reflecting the 5.0% injected label noise floor ($\approx 95.0\%$ Bayes ceiling).
*   Evaluated models at realistic churn prevalences (5%, 15%, 26.5%): Precision drops from 94.71% at 50% prevalence down to 48.41% at 5% prevalence, demonstrating necessity of prevalence-aware thresholding.
*   Evaluated current production models on real IBM Telco dataset ($N=7,043$): Accuracy drops to 69.5%–73.4% and ROC-AUC to 73.7%–76.5%, demonstrating real-world domain shift.

---

## [Phase 2: Rigorous Leakage-Safe Evaluation] - September 2026
*   **Dataset:** Exclusively `balanced_telco_churn.csv` ($N=8,000$). Zero external data mixing.
*   **CV Scheme:** Repeated Stratified K-Fold (5 Folds $\times$ 2 Repeats = 10 outer validation folds).
*   **Leakage Control:** All transformations (feature engineering, scaling) fit strictly on training folds.
*   **Models:** Evaluated 8 models + Stacking Ensemble. Disproved shallow baseline gap (LR reaches 93.73%, SVM reaches 94.60%).

---

## [Phase 3: Threshold and Cost Evaluation] - September 2026

### Mandatory Protocol
*   **Dataset:** Exclusively `balanced_telco_churn.csv` ($N=8,000$).
*   **Models Evaluated:** Random Forest, XGBoost, LightGBM, Stacking Ensemble (restricted strictly to these 4 models).
*   **Threshold Selection Protocol:** $\tau_{\text{cost}}^*$ and $\tau_{F_2}^*$ chosen strictly from training-side inner OOF predictions (zero validation data used during selection).
*   **Cost Parameters:** Primary baseline $C_{FN} = \$500, C_{FP} = \$50$ (10:1 ratio); sensitivity evaluated at 3:1, 5:1, 10:1, 20:1.

### Key Numerical Findings (Primary 10:1 Baseline: $C_{FN}=\$500, C_{FP}=\$50$)

| Model Architecture | Default Policy ($\tau=0.50$) | Cost-Optimized Policy ($\tau_{\text{cost}}^*$) | Business Cost Reduction | Missed Churners ($FN$) |
|:---|:---:|:---:|:---:|:---:|
| **Random Forest** | $\$26,295 \pm \$3,869$ ($\tau=0.50$) | $\mathbf{\$13,355 \pm \$1,193}$ ($\tau^*=0.10 \pm 0.01$) | **49.2% reduction** | Dropped from $48.7 \to 8.2$ |
| **XGBoost** | $\$22,730 \pm \$3,467$ ($\tau=0.50$) | $\mathbf{\$10,725 \pm \$1,079}$ ($\tau^*=0.06 \pm 0.02$) | **52.8% reduction** | Dropped from $41.6 \to 5.5$ |
| **LightGBM** | $\$22,600 \pm \$2,970$ ($\tau=0.50$) | $\mathbf{\$10,705 \pm \$1,659}$ ($\tau^*=0.06 \pm 0.02$) | **52.6% reduction** | Dropped from $41.2 \to 5.3$ |
| **Stacking Ensemble** | $\$23,285 \pm \$3,267$ ($\tau=0.50$) | $\mathbf{\$11,370 \pm \$1,494}$ ($\tau^*=0.04 \pm 0.01$) | **51.2% reduction** | Dropped from $42.8 \to 7.5$ |

### Sensitivity Trends (Stacking Ensemble)
*   **3:1 Ratio ($C_{FN}=\$300, C_{FP}=\$100$):** $\tau^* = 0.15 \pm 0.05$, Recall = $97.00\%$, Expected Cost = $\$14,640$
*   **5:1 Ratio ($C_{FN}=\$500, C_{FP}=\$100$):** $\tau^* = 0.07 \pm 0.02$, Recall = $98.34\%$, Expected Cost = $\$17,670$
*   **10:1 Ratio ($C_{FN}=\$500, C_{FP}=\$50$):** $\tau^* = 0.04 \pm 0.01$, Recall = $99.06\%$, Expected Cost = $\$11,370$
*   **20:1 Ratio ($C_{FN}=\$1000, C_{FP}=\$50$):** $\tau^* = 0.03 \pm 0.00$, Recall = $99.50\%$, Expected Cost = $\$14,245$
*   *Monotonic Law:* As the penalty of customer loss escalates relative to retention contact cost, the operating threshold shifts monotonically lower, increasing sensitivity to near-complete defecting subscriber capture ($99.5\%$).

---

## [Phase 4: Explainability, Concordance, Decision Support & Latency] - September 2026

### Mandatory Protocol
*   **Dataset:** Exclusively `balanced_telco_churn.csv` ($N = 8,000$).
*   **Model Architecture:** Serialized Stacking Ensemble (`ChurnV2StackingEnsemble`) with 27 features and deployed threshold $\tau = 0.26$.
*   **XAI Methods:** Tree/Kernel SHAP (Lundberg & Lee) and Tabular LIME (Ribeiro et al.).
*   **Concordance Sample:** $N = 200$ test customer instances evaluated quantitatively.
*   **Decision Support:** Tested on existing `backend/recommendation_engine.py` with state-awareness pruning.

### Key Numerical Findings
1.  **Global SHAP Attribution (Top Drivers):**
    *   `tenure`: Mean $|\text{SHAP}| = 1.4539$
    *   `ChargeIncreaseRatio`: Mean $|\text{SHAP}| = 1.0795$
    *   `MonthlyCharges`: Mean $|\text{SHAP}| = 0.6410$
    *   `InternetService_Fiber optic`: Mean $|\text{SHAP}| = 0.5199$
    *   `MonthlyCharges_x_IsMonthToMonth`: Mean $|\text{SHAP}| = 0.4880$
2.  **Quantitative SHAP vs. LIME Concordance ($N=200$):**
    *   **Spearman Rank Correlation ($\rho$):** $0.2421 \pm 0.1091$ (Median: $0.2444$)
    *   **Top-3 Feature Overlap:** $22.83\% \pm 22.00\%$
    *   **Top-5 Feature Overlap:** $38.80\% \pm 13.80\%$
    *   *Takeaway:* Disproved claims of visual "agreement"; feature rankings diverge significantly due to linear surrogate projection over correlated interaction dimensions.
3.  **Simulated Retention Economics (Test Cohort $N=1,600$ at $\tau=0.26$):**
    *   Flagged: 813 accounts ($50.8\%$)
    *   Intervention Cost: $\$40,650$ ($813 \times \$50$)
    *   Expected Retained Customers: $385.0$ ($770 \text{ TP} \times 0.50$)
    *   Gross Value Saved: $\$192,500$ ($385 \times \$500$)
    *   Net Simulated Benefit: $\$151,850$ (ROI: $+373.6\%$)
4.  **Measured Latency Benchmark:**
    *   **Single-Customer Prediction:** Mean $100.15\text{ ms}$, P95 $115.21\text{ ms}$ (1,000 runs)
    *   **Batch Customer Prediction (100):** Mean $124.41\text{ ms}$, P95 $167.30\text{ ms}$ (1,000 runs)
    *   **Local SHAP Explanation:** Mean $6.26\text{ ms}$, P95 $8.59\text{ ms}$ (200 runs)
    *   **Local LIME Explanation:** Mean $620.21\text{ ms}$, P95 $777.76\text{ ms}$ (100 runs)

---

## [Phase 6: References Verification and Citation Audit] - September 2026
*   Conducted exhaustive citation audit of `paper/main.tex` and `paper/refs.bib`.
*   Verified that every in-text citation corresponds to an exact BibTeX entry, and every BibTeX entry is actively cited in the manuscript.
*   Audited all 17 canonical peer-reviewed references: Breiman (2001), Chen & Guestrin (2016), Ke et al. (2017), Wolpert (1992), Lundberg & Lee (2017), Ribeiro et al. (2016), Elkan (2001), Verbeke et al. (2012), Coussement et al. (2017), Amin et al. (2019), Chawla et al. (2002), Brier (1950), Davis & Goadrich (2006), Ullah et al. (2019), Ullo et al. (2021), Albuquerque et al. (2022), Aggarwal & Mahajan (2024).
*   Eliminated all temporary placeholders, unverified DOIs, and broken citation numbers.
*   Ensured complete compliance with IEEE conference reference formatting standards.

---

## [Phase 7: Master Reproducibility Package] - September 2026
*   Constructed complete, self-contained replication package in `reproducibility/`.
*   Generated `reproducibility/requirements.txt` with pinned versions matching the verified runtime environment (Python 3.11.9, scikit-learn 1.9.0, xgboost 3.2.0, lightgbm 4.7.0, shap 0.51.0, lime 0.2.0.1, pandas 2.3.3, numpy 2.4.6).
*   Built master entry script `reproducibility/reproduce.py` with both `--quick` smoke-test mode and full academic reproduction mode.
*   Verified that `reproduce.py` strictly mandates `balanced_telco_churn.csv` ($N=8,000$) and enforces leakage-safe fold-by-fold feature transformations and threshold optimization.
*   Generated reproduction outputs in `reproducibility/results/`, `reproducibility/figures/`, and `reproducibility/logs/`.
*   Authored comprehensive `README_REPRODUCTION.md` providing exact environment installation and reproduction instructions.



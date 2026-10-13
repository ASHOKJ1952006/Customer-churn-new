# Master Reproducibility Guide

This repository contains the complete, self-contained replication package for the research manuscript:  
**"Explainable Ensemble Learning and Cost-Sensitive Decision Support Framework for Telecom Customer Churn Prediction"**

All code, pipelines, and evaluation routines are strictly leakage-safe and deterministic.

---

## 1. Directory Structure

```
├── data/
│   └── balanced_telco_churn.csv        # Mandatory benchmark dataset (N=8,000)
├── paper/
│   ├── main.tex                        # Master IEEEtran conference manuscript
│   ├── refs.bib                        # Canonical peer-reviewed BibTeX database
│   ├── figures/                        # Exact figures referenced in main.tex
│   └── tables/                         # Exact LaTeX table sources
├── reproducibility/
│   ├── reproduce.py                    # Master reproduction entry point
│   ├── requirements.txt                # Pinned dependency environment
│   ├── results/                        # Generated metric CSVs and JSONs
│   ├── figures/                        # Generated SHAP and cost figures
│   ├── tables/                         # Generated summary tables
│   └── logs/                           # Execution logs with timestamps
├── results/                            # Original empirical results from Phases 2-4
└── README_REPRODUCTION.md              # This documentation
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.11.x (tested on Python 3.11.9, 64-bit)
- Windows, Linux, or macOS

### Step A: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step B: Install Pinned Dependencies
```bash
pip install --upgrade pip
pip install -r reproducibility/requirements.txt
```

---

## 3. Dataset Placement

The pipeline is configured to operate **exclusively** on the domain-consistent benchmark:
- File: `data/balanced_telco_churn.csv`
- Size: Exactly 8,000 subscriber rows $\times$ 22 columns (4,000 Churn / 4,000 Loyal).
- **Rule:** Do not replace with IBM Telco, Cell2Cell, or any other external dataset. If missing, `reproduce.py` will abort with an explicit error.

---

## 4. Running the Reproduction Pipeline

### Fast Smoke-Test Execution
To verify pipeline integrity, feature transformations, model fitting, and export formatting in under 60 seconds:
```bash
python reproducibility/reproduce.py --quick
```
*(Runs 2 outer folds $\times$ 1 repeat, 20 concordance samples, and 100 latency iterations).*

### Full Academic Reproduction
To reproduce the complete empirical metrics reported in the paper (10 outer folds, $N=200$ concordance evaluations, 1,000 latency benchmark iterations):
```bash
python reproducibility/reproduce.py
```

Execution will log progress directly to the console and simultaneously write a persistent log to `reproducibility/logs/reproduce_<timestamp>.log`.

---

## 5. Mapping Generated Outputs to Paper Tables and Figures

| Paper Element | Description | Source Output File |
|:---|:---|:---|
| **Table I** | 5-Fold $\times$ 2-Repeat Cross-Validation (Acc, Prec, Rec, F1, AUC, PR-AUC, Brier) | `reproducibility/results/phase2_repeated_cv_summary.csv` |
| **Table II** | Cost-Sensitive Threshold Optimization ($C_{\text{FN}}=\$500, C_{\text{FP}}=\$50$) | `reproducibility/results/phase3_threshold_cost_summary.csv` |
| **Table III** | Quantitative SHAP vs. LIME Concordance ($N=200$ test subscribers) | `reproducibility/results/shap_lime_concordance.json` |
| **Table IV** | End-to-End Latency Benchmarks (Single, Batch-100, SHAP, LIME) | `reproducibility/results/reproduced_latency_benchmark.csv` |
| **Fig. 4 & 5** | Global SHAP Feature Attributions (Bar & Beeswarm Plots) | `reproducibility/figures/shap_feature_importance_bar.png` |
| **Section VIII.F**| Simulated Retention Economics & ROI (+373.6% ROI) | `reproducibility/results/simulated_retention_roi.json` |

---

## 6. Determinism and Hardware Notes

- **Random Seeds:** Random seeds are fixed to `RANDOM_SEED = 42` and `100`.
- **Multithreading:** GBDT libraries (`xgboost`, `lightgbm`, `scikit-learn`) use multithreaded OpenMP execution. Slight numerical variations ($\pm 0.05\%$) may occur across different CPU microarchitectures or thread schedulers.
- **Explainability:** SHAP values are evaluated using `shap.TreeExplainer` on the gradient-boosted component for exact polynomial complexity, while LIME generates 5,000 perturbed instances per query point.

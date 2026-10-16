# REFERENCE PDF SECTION ANALYSIS & STRUCTURAL MAPPING

**Reference Document:** *A Micro-Level Spatiotemporal Multi-Modal Satellite & Soil Data Fusion Framework for Precision Crop Decision Support* (Project Work II Phase I Report, October 2026, Department of CSE, Kongu Engineering College).  
**Target Project:** *Explainable Stacking Ensemble Learning and Decision Support Framework for Telecom Customer Churn Prediction*  
**Date:** October 2026  

---

## 1. Complete Outline of the Reference Document

The reference PDF is a 68-page engineering project work report structured into preliminary sections, 8 chapters, an appendix, and references:

```
PRELIMINARY PAGES
├── Title Page (p. i)
├── Bonafide Certificate (p. ii)
├── Declaration (p. iii)
├── Abstract (p. iv)
├── Acknowledgement (p. v)
├── Table of Contents (pp. vi–ix)
├── List of Tables (p. x)
├── List of Figures (p. xi)
└── List of Abbreviations (pp. xii–xiii)

CHAPTER 1: INTRODUCTION (pp. 1–4)
├── 1.1 Importance of Crop Recommendation (p. 2)
├── 1.2 Objectives (pp. 2–3)
└── 1.3 Scope of the Project (pp. 3–4)

CHAPTER 2: LITERATURE SURVEY (pp. 5–8)
├── 2.1 Machine Learning for Crop Recommendation (p. 5)
├── 2.2 Remote Sensing in Agriculture (p. 6)
├── 2.3 Soil Health and Land Suitability Assessment (p. 6)
├── 2.4 Ensemble Learning and Explainable AI (pp. 6–7)
│   └── Table 2.1: Literature Review (pp. 7–8)
└── 2.5 Research Gap (p. 8)

CHAPTER 3: EXISTING SYSTEM (pp. 9–11)
├── Overview and Description of Existing Methods (p. 9)
├── Feature Limitations & Disparate Knowledge Bases (pp. 9–10)
├── Regional Generalization & Class Imbalance Challenges (p. 10)
└── Lack of Interpretability & Decision Support (p. 11)

CHAPTER 4: REQUIREMENT SPECIFICATION (pp. 12–17)
├── 4.1 Hardware Requirements (pp. 12–13)
│   └── Table 4.1: Hardware Requirements (p. 13)
├── 4.2 Software Requirements (pp. 13–14)
│   └── Table 4.2: Software Requirements (p. 14)
└── 4.3 Software Description (pp. 15–17)
    ├── 4.3.1 Google Colab (p. 15)
    ├── 4.3.2 Python (p. 15)
    ├── 4.3.3 Pandas and NumPy (p. 15)
    ├── 4.3.4 Google Earth Engine (p. 16)
    ├── 4.3.5 Scikit-learn (p. 16)
    ├── 4.3.6 XGBoost (p. 16)
    ├── 4.3.7 CatBoost (p. 17)
    ├── 4.3.8 Matplotlib and Seaborn (p. 17)
    └── 4.3.9 QGIS (p. 17)

CHAPTER 5: PROPOSED SYSTEM (pp. 18–37)
├── 5.1 Proposed Work (p. 18)
├── 5.2 System Architecture (pp. 18–21)
│   ├── 5.2.1 Data Collection (p. 19)
│   ├── 5.2.2 Data Preprocessing and Spatial Alignment (p. 19)
│   ├── 5.2.3 Feature Construction (p. 19)
│   ├── 5.2.4 Multimodal Data Integration (p. 19)
│   ├── 5.2.5 Model Training (p. 19)
│   │   └── Figure 5.1: System Architecture of the Proposed System (p. 20)
│   ├── 5.2.6 Prediction and Probability Fusion (p. 20)
│   ├── 5.2.7 Crop Ranking (p. 21)
│   └── 5.2.8 Evaluation and Output (p. 21)
├── 5.3 Dataset and Feature Preparation (pp. 21–26)
│   ├── 5.3.1 Dataset Description (p. 21)
│   ├── 5.3.2 Satellite Data (p. 22) [Equations 5.1, 5.2]
│   ├── 5.3.3 Terrain and Soil Features (p. 23)
│   │   └── Figure 5.2: Geospatial data processing and Spatial coverage (p. 23)
│   ├── 5.3.4 Feature Categories (p. 24)
│   │   └── Table 5.1: Feature Categories Used in AgriFusion (p. 24)
│   ├── 5.3.5 Data Cleaning and Preparation (pp. 24–25)
│   └── 5.3.6 Class Imbalance (pp. 25–26)
├── 5.4 Models Used (pp. 26–27)
│   ├── 5.4.1 K-Nearest Neighbors (p. 26)
│   ├── 5.4.2 Random Forest (p. 26)
│   ├── 5.4.3 XGBoost (p. 27)
│   └── 5.4.4 CatBoost (p. 27)
├── 5.5 Training and Evaluation Workflow (pp. 27–28)
│   └── Table 5.2: Dataset Split (p. 28)
├── 5.6 Crop Recommendation and Probability-Based Ranking (pp. 29–30)
│   ├── 5.6.1 Probability Fusion (pp. 29–30) [Equations 5.3, 5.4, 5.5]
│   └── 5.6.2 Generation of Ranked Recommendations (p. 30)
├── 5.7 Performance Evaluation (pp. 30–34)
│   ├── 5.7.1 Top-1 Accuracy (pp. 30–31)
│   ├── 5.7.2 Top-3 and Top-5 Accuracy (p. 31) [Equation 5.6]
│   ├── 5.7.3 Mean Reciprocal Rank (MRR) (pp. 31–32) [Equation 5.7]
│   ├── 5.7.4 Normalized Discounted Cumulative Gain at 5 (NDCG@5) (pp. 32–33) [Equations 5.8, 5.9]
│   ├── 5.7.5 Multiclass ROC-AUC (pp. 33–34) [Equations 5.10, 5.11, 5.12]
│   └── 5.7.6 Evaluation Summary (p. 34)
└── 5.8 Prediction and User Interface (pp. 34–37)
    ├── Figure 5.3: Map Location Picker (p. 35)
    ├── Figure 5.4: Crop Recommendation Results with confidence score (p. 36)
    ├── Figure 5.5: Real-Time Satellite & Weather Monitoring Dashboard (p. 36)
    ├── Figure 5.6: Decision Dashboard with geographical/temporal filters (p. 37)
    └── Figure 5.7: Crop Pattern Analysis (p. 37)

CHAPTER 6: RESULTS AND DISCUSSION (pp. 38–44)
├── 6.1 Dataset Description (pp. 38–39)
│   └── Table 6.1: Summary of the AgriFusion Dataset (p. 38)
├── 6.2 Experimental Setup (p. 39)
├── 6.3 Comparative Study of Machine Learning Models (pp. 40–41)
│   ├── Table 6.2: Comparative Performance of the Evaluated Models (p. 40)
│   └── 6.3.1 Training-Time Analysis (p. 41)
├── 6.4 Ranked Recommendation Results (pp. 41–42)
│   ├── Table 6.3: Ranking Metrics Reported in AgriFusion (p. 41)
│   ├── Figure 6.1: Comparative ROC-AUC Performance (p. 42)
│   └── Figure 6.2: Normalized Confusion Matrix (p. 43)
└── 6.5 Discussion (pp. 43–44)

CHAPTER 7: CONCLUSION AND FUTURE WORK (p. 45)
├── Summary of Findings
└── Future Directions

APPENDIX 1: CODING (pp. 46–53)
└── Full Python Backend Implementation (`main.py` inference pipeline)

REFERENCES (p. 54)
└── Formal citations [1]–[8]

CHAPTER 8: MAPPING OF PROJECT GOALS TO SDG (p. 55)
└── United Nations Sustainable Development Goal 2 (Zero Hunger)
```

---

## 2. Topic-by-Topic Comparative Mapping

The table below explains what each section in the reference paper covers and how it is rigorously mapped to our **Telecom Customer Churn Prediction & Decision Support** project:

| Reference Chapter / Section | Reference Paper Coverage (AgriFusion) | Telecom Churn Project Equivalent | Adaptation Rationale & Domain Shift |
|:---|:---|:---|:---|
| **Title & Frontmatter** | Multimodal Satellite & Soil Data Fusion for Precision Crop Decision Support | *Explainable Stacking Ensemble Learning and Decision Support Framework for Telecom Customer Churn Prediction* | Replaces agronomic crop recommendation with telecom customer attrition mitigation and retention intelligence. |
| **Abstract** | Combines Sentinel-1/2, SRTM, and Soil Health Cards with KNN, RF, XGBoost, CatBoost for ranking 41 crops. | Combines subscriber tenure, contracts, billing, services, and charges with Random Forest, XGBoost, LightGBM, and Stacking Ensemble, augmented with SHAP/LIME and retention recommendations. | Condenses binary churn classification, cost-sensitive thresholding, and XAI concordance into an IEEE-standard abstract. |
| **1.1 Importance / Motivation** | Agronomic crop selection impacts agricultural yield, soil health, and regional food security. | Churn severely damages telecom recurring revenue, customer lifetime value (CLV), and acquisition cost efficiency. | Telecom subscriber acquisition cost ($CAC \approx \$300-\$500$) is $5\times$ higher than retention cost ($C_{\text{contact}} \approx \$50$). |
| **1.2 Objectives** | Integrate satellite, soil, and historical crop data; train 4 classifiers; rank crops; build decision dashboard. | 1) Build 27-feature pipeline; 2) Benchmark RF, XGBoost, LightGBM, Stacking; 3) Optimize cost threshold ($\tau^*=0.26$); 4) Provide SHAP/LIME attribution; 5) Deliver operational web DSS. | Mirrors the exact end-to-end scope from raw data processing to machine learning and human-in-the-loop decision support. |
| **1.3 Scope** | 376,906 records across Tamil Nadu from 2018–2024; 41 crop categories; geospatial prediction. | 8,000 subscriber cohort (6,400 train / 1,600 held-out test); 20 base domain attributes; 4 machine learning models; single, batch, and simulator inference. | Strictly bounds our evaluation to the verified 80/20 benchmark and four models. |
| **2.1–2.4 Literature Review** | Soil-based ML, remote sensing (Sentinel 1/2), land suitability GIS, ensemble voting, SHAP for crops. | Traditional ML churn baselines (LR, SVM, DT), tree ensembles (RF, XGBoost, LightGBM), stacking meta-learning, cost-sensitive thresholding, XAI in customer analytics. | Maps agronomic literature to empirical telecom churn literature, retaining seminal foundations (Breiman 2001, Chen & Guestrin 2016, Lundberg & Lee 2017). |
| **Table 2.1: Literature Survey Table** | Comparison of Rouse (1974), Breiman (2001), Drusch (2012), Chen & Guestrin (2016), Lundberg & Lee (2017), etc. | Structured comparison of seminal and recent churn/XAI studies (Verbeke 2012, Vafeiadis 2015, Chen 2016, Lundberg 2017, Ribeiro 2016, Stripling 2018). | Preserves the 4-column schema: Author/Year, Forum, Proposed Methodology, Findings/Limitations. |
| **2.5 Research Gap** | Disconnect between remote sensing and soil testing; lack of probabilistic ranking. | Black-box churn models output raw probabilities without local driver attribution, financial cost alignment, or actionable campaign guidance. | Identifies our 3-part contribution: Predictive Power + Local Explainability + Prescriptive Action Rules. |
| **Chapter 3: Existing System** | Tabular NPK systems lacking spatial context; single-crop hard outputs; class imbalance vulnerabilities. | Standard binary churn classifiers using default $\tau=0.50$, ignoring asymmetric error costs ($Cost(FN) \gg Cost(FP)$), lacking operational explainability. | Details the failure modes of legacy churn detection systems in active telecom operations. |
| **Chapter 4: Requirements** | Python, GEE, Colab, Scikit-learn, XGBoost, CatBoost, QGIS; 16 GB RAM, T4 GPU. | Python 3.11, FastAPI, React 19, Vite, Scikit-learn, XGBoost, LightGBM, SHAP, LIME; Intel Core i7 / 16 GB RAM. | Documents actual software stack verified in `backend/requirements.txt` and `frontend/package.json`. |
| **5.1–5.2 System Architecture** | Multimodal data ingestion $\to$ preprocessing $\to$ 4-model inference $\to$ probability fusion $\to$ ranked crops $\to$ UI. | Customer Ingestion $\to$ Feature Transformer $\to$ Base Learners $\to$ Logistic Meta-Learner $\to$ Threshold $\to$ SHAP/LIME $\to$ Rule Engine $\to$ Dashboard. | Maps Figure 5.1 architecture directly to the churn system workflow. |
| **5.3 Dataset & Preprocessing** | 376,906 rows (Sentinel + SRTM + SHC); 41 features; NPK index; duplicate/null handling; 70:15:15 split. | 8,000 rows (`balanced_telco_churn.csv`); 20 domain features + 7 engineered = 27 features; 80/20 split (6,400 train / 1,600 test). | Accurately documents missing value imputation, scaling, label encoding, and interaction features. |
| **5.4 Models Evaluated** | KNN, Random Forest, XGBoost, CatBoost. | Random Forest, XGBoost, LightGBM, Stacking Ensemble (with Logistic Regression meta-learner). | Replaces KNN/CatBoost with LightGBM and Stacking Ensemble, per verified project scope. |
| **5.5 Workflow & Split** | 70% Train, 15% Validation, 15% Test. | Stratified 80% Train ($N=6,400$), 20% Held-Out Test ($N=1,600$) with 5-fold cross-validation. | Documents strict zero-leakage hold-out test protocol. |
| **5.6 Recommendation / Ranking** | Weighted probability soft-voting fusion $\sum w_m P_m(c \mid x)$ to rank top crops. | Logistic meta-learner probability output $\sigma(w_0 + \sum w_m z_m)$ passed to cost-sensitive threshold $\tau^*$ and prescriptive rules. | Adapts multi-crop ranking equations to binary ensemble stacking and rule-driven retention strategy assignment. |
| **5.7 Evaluation Metrics** | Top-1/3/5 Accuracy, MRR, NDCG@5, Multiclass ROC-AUC. | Recall, Precision, Accuracy, F1-Score, ROC-AUC, Brier Calibration Score, Confusion Matrix. | Adapts ranking metrics to binary classification and probabilistic calibration measures. |
| **5.8 UI & Decision Support** | Map picker, crop confidence cards, satellite moisture viewer, crop rotation advisor. | Single-Customer Explainability, Batch CSV Inference, Model Comparison Benchmark, What-If Sensitivity Simulator. | Matches our live React dashboard pages ([SingleCustomer.jsx](file:///d:/ABI/YEAR%204/SEM%207/churn/Customer-churn-new/frontend/src/pages/SingleCustomer.jsx), [BatchUpload.jsx](file:///d:/ABI/YEAR%204/SEM%207/churn/Customer-churn-new/frontend/src/pages/BatchUpload.jsx), [ModelComparison.jsx](file:///d:/ABI/YEAR%204/SEM%207/churn/Customer-churn-new/frontend/src/pages/ModelComparison.jsx), [WhatIfSimulator.jsx](file:///d:/ABI/YEAR%204/SEM%207/churn/Customer-churn-new/frontend/src/pages/WhatIfSimulator.jsx)). |
| **6.1–6.3 Benchmark Results** | Accuracy, Precision, Recall, F1, ROC-AUC, Training Time across 6 models + voting. | Recall (95.0%–95.4%), F1 (95.2%–95.6%), Accuracy (95.2%–95.6%), Precision (95.6%–96.0%), ROC-AUC (98.7%–99.0%), Brier (0.0346–0.0369). | Evaluates Random Forest, XGBoost, LightGBM, and Stacking Ensemble on $N=1,600$ test subscribers. |
| **6.4 Confusion Matrix / Analysis** | 41-class confusion matrix, ROC-AUC curve. | Binary confusion matrix ($TN=766, FP=34, FN=37, TP=763$), SHAP summary beeswarm, cost-threshold curve. | Validates true positives, false alarms, and missed churners. |
| **6.5 Discussion** | Trade-offs in crop accuracy vs ROC-AUC; class imbalance; agronomic validation needs. | Precision vs. Recall trade-off; Brier calibration benefit; financial ROI ($+\$151,850$ net gain); SHAP-LIME concordance divergence ($\rho=0.2421$). | Explains why Stacking Ensemble leads in Recall (95.4%) and Brier calibration (0.0346), despite RF leading in Precision (96.0%). |
| **Chapter 7: Conclusion & Future** | Summary of AgriFusion, deep learning future work (ANN/DNN), field agronomy validation. | Summary of Explainable Stacking Framework, prospective A/B campaign testing, drift monitoring, naturally imbalanced cohort validation. | Rigorous conclusion with confirmed operational strengths and transparent threats to validity. |
| **Appendix: Code** | `main.py` crop health / recommendation inference script. | Production backend inference engine (`backend/main.py`, `backend/shap_service.py`, `backend/recommendation_engine.py`). | Documents complete end-to-end production prediction, SHAP waterfall, and rule generation logic. |
| **Chapter 8: SDG Mapping** | UN SDG 2: Zero Hunger. | UN SDG 9 (Industry, Innovation & Infrastructure) & SDG 12 (Responsible Resource Management). | Connects algorithmic retention efficiency to sustainable digital infrastructure and reduced wasteful churn acquisition expenditure. |

---

## 3. Reference Topics That Do Not Transfer Directly

To maintain academic rigor and domain honesty, several elements of the reference PDF cannot and must not be imported into the telecom churn paper:
1. **Remote Sensing Optical/Radar Indices (NDVI, NDWI, VV, VH Polarizations):** Telecom customer churn has no connection to physical ground reflectance or microwave backscatter. These are replaced by telecom operational metrics (Monthly Charges, Tenure, Contract Type, Payment Method, Service Counts).
2. **Terrain Topography (SRTM Elevation, Slope, Aspect):** Replaced by subscriber lifecycle and financial ratios (Charges-to-Tenure Ratio, Average Monthly Spend, Contract Maturity).
3. **Multiclass Crop Ranking (Top-1, Top-3, Top-5, MRR, NDCG@5):** Telecom churn is fundamentally a **binary classification problem** ($\text{Churn} \in \{0, 1\}$) followed by continuous probability calibration and financial thresholding. Ranking metrics are replaced by **Brier calibration score, Precision-Recall AUC, Cost-Sensitive Net Financial Benefit, and SHAP-LIME Concordance ($\rho$)**.
4. **CatBoost & KNN:** KNN and CatBoost were included in the reference crop report but are **not part of the telecom churn four-model benchmark** (Random Forest, XGBoost, LightGBM, Stacking Ensemble).

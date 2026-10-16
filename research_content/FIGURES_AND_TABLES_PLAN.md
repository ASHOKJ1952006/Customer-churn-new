# FIGURES AND TABLES MASTER PLAN FOR IEEE PAPER

**Project:** Explainable Stacking Ensemble Learning and Decision Support Framework for Telecom Customer Churn Prediction  
**Status:** Publication Asset Specifications  
**Date:** October 2026  
**Primary Output File:** `research_content/FIGURES_AND_TABLES_PLAN.md`  

---

## 1. Master List of Proposed Figures

| Figure ID | Title / Concept | Data Source File | Generation Script / Status | Publication Caption |
|:---|:---|:---|:---|:---|
| **Fig. 1** | System Architecture & Operational Workflow | System Design (`backend/main.py`) | TikZ / Mermaid / SVG diagram (Ready) | *Fig. 1. End-to-end architecture of the explainable churn prediction and prescriptive retention decision-support system, illustrating feature engineering, out-of-fold stacking meta-learning, cost-sensitive thresholding, dual SHAP/LIME attribution, and rule-based action generation.* |
| **Fig. 2** | Comparative ROC and Precision-Recall Curves | `results/model_comparison_curves.json` | `scratch/generate_ieee_paper_figures.py` (Ready) | *Fig. 2. Performance curves evaluated on the untouched held-out test cohort ($N = 1,600$): (a) Receiver Operating Characteristic (ROC) curves with area under curve (AUC) annotations; (b) Precision-Recall (PR) curves demonstrating superior probability calibration for the Stacking Ensemble.* |
| **Fig. 3** | Confusion Matrices across Evaluated Models | `backend/models/model_comparison_results.json` | `scratch/generate_ieee_paper_figures.py` (Ready) | *Fig. 3. Normalized and raw confusion matrices evaluated at $\tau = 0.50$ across 1,600 held-out test records: (a) Random Forest; (b) XGBoost; (c) LightGBM; and (d) Stacking Ensemble (766 TN, 34 FP, 37 FN, 763 TP).* |
| **Fig. 4** | Global Feature Importance via SHAP Values | `results/global_shap_importance.csv` | `scratch/generate_paper_figures.py` (Ready) | *Fig. 4. Global feature importance derived from 1,600 held-out test subscribers using TreeSHAP: (a) Mean absolute SHAP attributions ranked across top predictors; (b) Summary beeswarm plot illustrating directional risk impacts of subscriber tenure, contract structure, and billing velocity.* |
| **Fig. 5** | Cost-Sensitive Threshold Optimization Curve | `results/phase3_threshold_metrics_summary.csv` | `scripts/phase3_threshold_cost_analysis.py` (Ready) | *Fig. 5. Total operational business cost as a function of classification decision threshold $\tau$, assuming intervention cost $C_{\text{contact}} = \$50$ and customer lifetime value $V_{\text{CLV}} = \$500$, establishing optimal cost-minimizing threshold at $\tau^* = 0.26$.* |
| **Fig. 6** | SHAP-LIME Concordance & Rank Correlation | `results/shap_lime_concordance.json` | `scripts/phase4_explainability_decision_support.py` (Ready) | *Fig. 6. Empirical concordance analysis between SHAP and LIME feature attributions across $N = 200$ test subscribers, demonstrating Spearman rank correlation ($\rho = 0.2421$) and top-feature attribution divergence.* |
| **Fig. 7** | Production Decision-Support Web Interface | Application Screenshots | Live React Frontend (`frontend/src/pages/`) (Ready) | *Fig. 7. Graphical user interface of the deployed decision-support framework: (a) Single-customer risk profiling with interactive SHAP force waterfall; (b) Batch CSV risk cohort analytics; and (c) What-if policy sensitivity simulator.* |

---

## 2. Master List of Proposed Tables

### Table I: Summary of Dataset Partitioning and Characteristics
**Source:** `data/balanced_telco_churn.csv` and `research_audit/IEEE_REVIEWER_RESPONSE_DATA.md`  
**Caption:** *TABLE I: SUMMARY OF BENCHMARK DATASET PARTITIONING AND COHORT ATTRIBUTES*

```latex
\begin{table}[htbp]
\centering
\caption{Summary of Benchmark Dataset Partitioning and Cohort Attributes}
\label{tab:dataset_summary}
\begin{tabular}{llcc}
\hline
\textbf{Partition} & \textbf{Split Ratio} & \textbf{Subscribers ($N$)} & \textbf{Class Balance (Loyal / Churn)} \\
\hline
Training Cohort & 80.0\% Stratified & 6,400 & 3,200 (50.0\%) / 3,200 (50.0\%) \\
Held-Out Test Cohort & 20.0\% Stratified & 1,600 & 800 (50.0\%) / 800 (50.0\%) \\
Total Benchmark Cohort & 100.0\% & 8,000 & 4,000 (50.0\%) / 4,000 (50.0\%) \\
\hline
\textbf{Feature Group} & \textbf{Attributes} & \textbf{Count} & \textbf{Data Types / Scaling} \\
\hline
Continuous Financial & Tenure, MonthlyCharges, TotalCharges & 3 & Float / RobustScaler \\
Domain Engineered & Charges/Tenure, Service Count, Interaction Flags & 7 & Float, Integer, Binary Flag \\
Categorical Contract & Contract, PaymentMethod, InternetService & 7 & One-Hot Encoded \\
Demographic \& Flags & SeniorCitizen, Partner, Dependents, PhoneService & 10 & Binary Flags \\
\hline
\textbf{Total Predictors} & Input Dimension to ML Models & \textbf{27} & Transformed Feature Vector \\
\hline
\end{tabular}
\end{table}
```

---

### Table II: Comprehensive Performance Comparison Across Evaluated Models
**Source:** `backend/models/model_comparison_results.json` ($N_{\text{test}} = 1,600$ at $\tau = 0.50$)  
**Caption:** *TABLE II: COMPARATIVE PERFORMANCE OF EVALUATED MACHINE LEARNING MODELS ON UNTOUCHED TEST SET ($N = 1,600$)*

```latex
\begin{table*}[htbp]
\centering
\caption{Comparative Performance of Evaluated Machine Learning Models on Untouched Test Set ($N = 1,600$)}
\label{tab:model_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Model Architecture} & \textbf{Recall (\%)} & \textbf{F1-Score (\%)} & \textbf{Accuracy (\%)} & \textbf{Precision (\%)} & \textbf{ROC-AUC (\%)} & \textbf{Brier Score (↓)} \\
\hline
Random Forest (RF) & 95.0\% & 95.5\% & 95.5\% & \textbf{96.0\%} & 98.7\% (0.986872) & 0.0369 \\
XGBoost (XGB) & 94.5\% & 95.2\% & 95.2\% & 95.8\% & \textbf{99.0\%} (0.990063) & 0.0347 \\
LightGBM (LGBM) & 95.1\% & 95.4\% & 95.4\% & 95.6\% & 98.9\% (0.989458) & 0.0353 \\
\textbf{Stacking Ensemble (Proposed)} & \textbf{95.4\%} & \textbf{95.6\%} & \textbf{95.6\%} & 95.7\% & \textbf{99.0\%} (0.989630) & \textbf{0.0346} \\
\hline
\end{tabular}
\vspace{1ex}
\begin{flushleft}
\footnotesize{\textit{Note: Bold indicates the best-performing result per column. The Stacking Ensemble achieves the highest Recall (95.4\%), F1-score (95.6\%), Accuracy (95.6\%), and best probability calibration (Brier score 0.0346), while Random Forest leads in Precision (96.0\%). ROC-AUC ties between XGBoost and Stacking Ensemble at 99.0\% after standard 1-decimal rounding.}}
\end{flushleft}
\end{table*}
```

---

### Table III: Confusion Matrix Decomposition ($N = 1,600$ Test Subscribers)
**Source:** `backend/models/model_comparison_results.json` lines 23–126  
**Caption:** *TABLE III: CONTINGENCY TABLE DECOMPOSITION OF PREDICTIONS AT STANDARD THRESHOLD ($\tau = 0.50$)*

```latex
\begin{table}[htbp]
\centering
\caption{Contingency Table Decomposition of Predictions at Standard Threshold ($\tau = 0.50$)}
\label{tab:confusion_matrices}
\begin{tabular}{lccccc}
\hline
\textbf{Model} & \textbf{True Neg. ($TN$)} & \textbf{False Pos. ($FP$)} & \textbf{False Neg. ($FN$)} & \textbf{True Pos. ($TP$)} & \textbf{Total ($N$)} \\
\hline
Random Forest & 768 & 32 & 40 & 760 & 1,600 \\
XGBoost & 767 & 33 & 44 & 756 & 1,600 \\
LightGBM & 765 & 35 & 39 & 761 & 1,600 \\
\textbf{Stacking Ensemble} & \textbf{766} & \textbf{34} & \textbf{37} & \textbf{763} & \textbf{1,600} \\
\hline
\end{tabular}
\end{table}
```

---

### Table IV: Cost-Sensitive Threshold Optimization & Financial Return on Investment
**Source:** `results/simulated_retention_roi.json` and `scripts/phase3_threshold_cost_analysis.py`  
**Caption:** *TABLE IV: SIMULATED RETENTION CAMPAIGN FINANCIAL IMPACT ON HELD-OUT TEST COHORT ($N = 1,600$)*

```latex
\begin{table}[htbp]
\centering
\caption{Simulated Retention Campaign Financial Impact on Held-Out Test Cohort ($N = 1,600$)}
\label{tab:roi_simulation}
\begin{tabular}{lcc}
\hline
\textbf{Parameter / Operational Metric} & \textbf{Standard Threshold ($\tau = 0.50$)} & \textbf{Cost-Tuned Threshold ($\tau^* = 0.26$)} \\
\hline
Decision Threshold ($\tau$) & 0.50 & \textbf{0.26} \\
True Churners Flagged ($TP$) & 763 & \textbf{770} \\
Missed Defections ($FN$) & 37 & \textbf{30} \\
False Alarms / Outreach Inefficiencies ($FP$) & 34 & \textbf{43} \\
Total Retention Interventions ($TP + FP$) & 797 & \textbf{813} \\
Outreach Cost ($C_{\text{contact}} = \$50$) & \$39,850 & \$40,650 \\
Gross Value Preserved ($r = 0.50, V = \$500$) & \$190,750 & \textbf{\$192,500} \\
\textbf{Net Financial Value Preserved} & \textbf{+\$150,900} & \textbf{+\$151,850} \\
\textbf{Campaign Net ROI (\%)} & +378.7\% & \textbf{+373.6\%} \\
Net Incremental Revenue over Baseline $\tau=0.50$ & Reference & \textbf{+\$950 (per 1,600 accounts)} \\
\hline
\end{tabular}
\end{table}
```

---

### Table V: Prescriptive Retention Policy Rules Matrix
**Source:** `backend/recommendation_engine.py` and `research_audit/recommendation_rules.csv`  
**Caption:** *TABLE V: PRESCRIPTIVE RETENTION RULE CATALOG AND OPERATIONAL INTERVENTION TRIGGERS*

| Trigger Factor | SHAP Threshold / Condition | Recommended Retention Campaign | Expected Business Action & Incentive | Priority Level |
|:---|:---|:---|:---|:---:|
| **Month-to-Month Contract** | `Contract == 'Month-to-month'` & `SHAP_Contract > +0.35` | 12-Month Annual Commitment Upgrade | Offer 15% recurring bill credit for 1-year contract lock-in. | **High (Urgent)** |
| **Electronic Check Payment** | `PaymentMethod == 'Electronic check'` & `SHAP_Payment > +0.25` | Automated ACH / Autopay Enrollment | Provide one-time \$10 credit upon enabling automated bank draft. | **High (Urgent)** |
| **High Monthly Bill Velocity** | `MonthlyCharges > $75` & `charges_per_tenure > 70` | Value Optimization & Plan Re-bundling | Transition subscriber to optimized fiber tier with streaming add-on waiver. | **Medium** |
| **Zero Security / Backup Add-ons** | `total_services <= 1` & `OnlineSecurity == 'No'` | Cyber Protection & Cloud Trial Bundle | 60-day complimentary trial of OnlineSecurity & Cloud Backup. | **Medium** |
| **Early Lifecycle Instability** | `tenure <= 6 months` & `high_risk_profile == True` | Dedicated Onboarding Concierge Call | Outbound VIP support touchpoint within 48 hours to resolve setup friction. | **High (Critical)** |
| **Long-Term Fiber Dissatisfaction** | `tenure > 24` & `TechSupport == 'No'` | Equipment Upgrade & VIP Tech Support | Dispatch complimentary Wi-Fi 6 router upgrade and prioritized hotline routing. | **Low (Proactive)** |

---

## 3. Mathematical Equations Reference Sheet

### Eq. 1: Logistic Sigmoid Function
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Eq. 2: Stacking Ensemble Meta-Learner Formulation
$$P(\text{Churn}=1 \mid \mathbf{x}) = \sigma\left(w_0 + \sum_{m=1}^{M} w_m \cdot \hat{P}_m(\mathbf{x})\right)$$
where $M=3$ base models (Random Forest, XGBoost, LightGBM), $\hat{P}_m(\mathbf{x})$ is the cross-validated out-of-fold probability estimate from base classifier $m$, $w_m$ represents the learned meta-regression weight, and $w_0$ is the intercept.

### Eq. 3: Binary Classification Metrics
$$\text{Recall} = \frac{TP}{TP + FN}, \quad \text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Eq. 4: Brier Calibration Score
$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} \left(P_i - y_i\right)^2$$
where $y_i \in \{0, 1\}$ is the ground-truth subscriber churn status and $P_i \in [0, 1]$ is the model's calibrated probability output. A lower Brier score denotes superior probabilistic calibration.

### Eq. 5: Total Operational Cost Function
$$\text{Cost}(\tau) = C_{\text{contact}} \cdot \sum_{i=1}^{N} \mathbf{1}\left(P_i \ge \tau\right) + V_{\text{CLV}} \cdot \sum_{i=1}^{N} \mathbf{1}\left(P_i < \tau \land y_i = 1\right)$$
where $C_{\text{contact}} = \$50$ is the outreach campaign cost, $V_{\text{CLV}} = \$500$ is the net customer lifetime value lost when a churner defects undetected, and $\tau^* = \arg\min_{\tau \in [0, 1]} \text{Cost}(\tau)$.

### Eq. 6: Shapley Additive Explanations (SHAP) Attribution
$$\phi_j(f, \mathbf{x}) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[f(S \cup \{j\}) - f(S)\right]$$
such that local efficiency holds: $f(\mathbf{x}) = \phi_0 + \sum_{j=1}^{K} \phi_j(\mathbf{x})$.

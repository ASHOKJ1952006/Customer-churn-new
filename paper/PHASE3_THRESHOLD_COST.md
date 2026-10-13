# Phase 3: Leakage-Safe Threshold Optimization and Business Cost Analysis Report

**Mandatory Scope Declaration:**  
> **"Phase 3 threshold and business-cost evaluation was conducted exclusively on `balanced_telco_churn.csv` across Random Forest, XGBoost, LightGBM, and the Stacking Ensemble."**  
> In accordance with project instructions, no external datasets were used, and the model comparison is restricted strictly to these four architectures.

---

## 1. Methodology to Prevent Threshold-Selection Leakage

To prevent optimistic threshold overfitting and data snooping, every threshold is chosen strictly out-of-sample:
1. **Outer Repeated CV Partition:** The dataset ($N=8,000$) is divided into train ($N=6,400$) and validation ($N=1,600$) using Repeated Stratified K-Fold (5 folds $\times$ 2 repeats = 10 outer folds).
2. **Training-Side Inner OOF Predictions:** An inner 5-fold cross-validation loop is executed *strictly within each training fold*. This yields un-leaked out-of-fold probability vectors ($\mathbf{p}_{\text{oof}} \in [0, 1]^{6,400}$) for Random Forest, XGBoost, LightGBM, and Stacking.
3. **Threshold Selection on Training OOF Only:**
   * **Cost-Minimizing Threshold ($\tau_{\text{cost}}^*$):** Selected by minimizing the assumed business cost on the inner training OOF predictions:
     $$\tau_{\text{cost}}^* = \arg\min_{\tau \in [0.01, 0.99]} \left[ C_{FN} \cdot FN(\tau) + C_{FP} \cdot FP(\tau) \right]$$
   * **$F_2$-Optimal Threshold ($\tau_{F_2}^*$):** Selected by maximizing $F_2$ (which weights recall twice as heavily as precision) on the training OOF predictions:
     $$\tau_{F_2}^* = \arg\max_{\tau \in [0.01, 0.99]} \frac{5 \cdot \text{Precision}(\tau) \cdot \text{Recall}(\tau)}{4 \cdot \text{Precision}(\tau) + \text{Recall}(\tau)}$$
4. **Out-of-Sample Application:** The optimal thresholds selected from the training fold are applied to the **completely unseen outer validation fold ($N=1,600$)**. The validation fold is never used to search, select, or guide threshold values.

---

## 2. Primary Business Cost Model ($C_{FN}:C_{FP} = 10:1$)

### Business Context & Assumptions
*   **False Negative Penalty ($C_{FN} = \$500$):** Reflects the estimated average customer lifetime value (LTV) lost when an at-risk subscriber cancels unnoticed.
*   **False Positive Penalty ($C_{FP} = \$50$):** Reflects the administrative cost of a proactive retention offer, discount, or customer outreach delivered to a loyal subscriber who was not actually at risk.
*   *Note:* These parameters are experimental baseline assumptions for cost-sensitive analysis, not empirical company accounting records.

$$\text{Business Cost} = 500 \cdot FN + 50 \cdot FP$$

---

## 3. Primary Evaluation Results: Default vs. $F_2$-Optimal vs. Cost-Optimal

Metrics are reported as **$\text{Mean} \pm \text{Standard Deviation}$** across all 10 unseen outer validation folds ($N=1,600$ per fold):

| Model Architecture | Threshold ($\tau$) | Recall (\%) | Precision (\%) | Accuracy (\%) | F1-Score (\%) | F2-Score (\%) | Missed Churners ($FN$) | False Alarms ($FP$) | Expected Business Cost ($) | Cost / Customer ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | | | | | | | | | | |
| *Default* | $0.50 \pm 0.00$ | $93.91 \pm 0.96$ | $95.08 \pm 0.76$ | $94.53 \pm 0.62$ | $94.49 \pm 0.63$ | $94.14 \pm 0.80$ | $48.7 \pm 7.7$ | $38.9 \pm 6.3$ | $\$26,295 \pm \$3,869$ | $\$16.43 \pm \$2.42$ |
| *$F_2$-Optimal* | $0.24 \pm 0.03$ | $97.06 \pm 0.79$ | $89.89 \pm 1.24$ | $93.06 \pm 0.70$ | $93.33 \pm 0.63$ | $95.53 \pm 0.57$ | $23.5 \pm 6.3$ | $87.5 \pm 12.2$ | $\$16,125 \pm \$2,961$ | $\$10.08 \pm \$1.85$ |
| *Cost-Optimal (10:1)* | $0.10 \pm 0.01$ | $\mathbf{98.97 \pm 0.43}$ | $81.08 \pm 1.43$ | $87.92 \pm 0.98$ | $89.13 \pm 0.75$ | $94.78 \pm 0.27$ | $\mathbf{8.2 \pm 3.5}$ | $185.1 \pm 18.2$ | $\mathbf{\$13,355 \pm \$1,193}$ | $\mathbf{\$8.35 \pm \$0.75}$ |
| **XGBoost** | | | | | | | | | | |
| *Default* | $0.50 \pm 0.00$ | $94.80 \pm 0.88$ | $95.16 \pm 0.80$ | $94.99 \pm 0.56$ | $94.98 \pm 0.56$ | $94.87 \pm 0.70$ | $41.6 \pm 7.0$ | $38.6 \pm 6.8$ | $\$22,730 \pm \$3,467$ | $\$14.21 \pm \$2.17$ |
| *$F_2$-Optimal* | $0.15 \pm 0.03$ | $98.14 \pm 0.76$ | $88.44 \pm 1.66$ | $92.63 \pm 0.88$ | $93.02 \pm 0.75$ | $96.02 \pm 0.44$ | $14.9 \pm 6.1$ | $103.0 \pm 17.5$ | $\$12,600 \pm \$2,518$ | $\$7.88 \pm \$1.57$ |
| *Cost-Optimal (10:1)* | $0.06 \pm 0.02$ | $\mathbf{99.31 \pm 0.46}$ | $83.33 \pm 1.97$ | $89.69 \pm 1.21$ | $90.61 \pm 0.98$ | $95.63 \pm 0.28$ | $\mathbf{5.5 \pm 3.7}$ | $159.5 \pm 22.4$ | $\mathbf{\$10,725 \pm \$1,079}$ | $\mathbf{\$6.70 \pm \$0.67}$ |
| **LightGBM** | | | | | | | | | | |
| *Default* | $0.50 \pm 0.00$ | $94.85 \pm 0.77$ | $95.00 \pm 0.76$ | $94.92 \pm 0.44$ | $94.92 \pm 0.44$ | $94.88 \pm 0.58$ | $41.2 \pm 6.2$ | $40.0 \pm 6.5$ | $\$22,600 \pm \$2,970$ | $\$14.12 \pm \$1.86$ |
| *$F_2$-Optimal* | $0.17 \pm 0.05$ | $98.24 \pm 0.64$ | $88.79 \pm 1.79$ | $92.89 \pm 0.96$ | $93.26 \pm 0.84$ | $96.18 \pm 0.40$ | $14.1 \pm 5.1$ | $99.6 \pm 18.2$ | $\$12,030 \pm \$2,074$ | $\$7.52 \pm \$1.30$ |
| *Cost-Optimal (10:1)* | $0.06 \pm 0.02$ | $\mathbf{99.34 \pm 0.56}$ | $83.22 \pm 2.51$ | $89.60 \pm 1.68$ | $90.54 \pm 1.35$ | $95.62 \pm 0.51$ | $\mathbf{5.3 \pm 4.4}$ | $161.1 \pm 29.7$ | $\mathbf{\$10,705 \pm \$1,659}$ | $\mathbf{\$6.69 \pm \$1.04}$ |
| **Stacking Ensemble** | | | | | | | | | | |
| *Default* | $0.50 \pm 0.00$ | $94.65 \pm 0.82$ | $95.27 \pm 0.92$ | $94.97 \pm 0.60$ | $94.95 \pm 0.60$ | $94.77 \pm 0.67$ | $42.8 \pm 6.6$ | $37.7 \pm 7.7$ | $\$23,285 \pm \$3,267$ | $\$14.55 \pm \$2.04$ |
| *$F_2$-Optimal* | $0.09 \pm 0.01$ | $97.89 \pm 0.64$ | $89.32 \pm 0.86$ | $93.09 \pm 0.51$ | $93.41 \pm 0.47$ | $96.04 \pm 0.47$ | $16.9 \pm 5.1$ | $93.7 \pm 8.5$ | $\$13,135 \pm \$2,430$ | $\$8.21 \pm \$1.52$ |
| *Cost-Optimal (10:1)* | $0.04 \pm 0.01$ | $\mathbf{99.06 \pm 0.52}$ | $83.91 \pm 1.85$ | $90.01 \pm 1.16$ | $90.85 \pm 0.94$ | $95.60 \pm 0.34$ | $\mathbf{7.5 \pm 4.1}$ | $152.4 \pm 21.4$ | $\mathbf{\$11,370 \pm \$1,494}$ | $\mathbf{\$7.11 \pm \$0.93}$ |

---

## 4. Key Financial Takeaways (Default vs. Cost-Optimized)

1. **Halving the Enterprise Business Cost:**
   * For the Stacking Ensemble, optimizing $\tau$ cuts total expected cost from **$\$23,285$ down to $\$11,370$** per 1,600 accounts (a **51.2% cost reduction**).
   * For XGBoost, cost drops from **$\$22,730$ down to $\$10,725$** (a **52.8% cost reduction**).
   * For LightGBM, cost drops from **$\$22,600$ down to $\$10,705$** (a **52.6% cost reduction**).
   * For Random Forest, cost drops from **$\$26,295$ down to $\$13,355$** (a **49.2% cost reduction**).
2. **Eliminating Missed Churners (Drastic False Negative Suppression):**
   * Stacking Ensemble: False negatives collapse from **$42.8 \pm 6.6$ down to $7.5 \pm 4.1$** accounts.
   * XGBoost: False negatives collapse from **$41.6 \pm 7.0$ down to $5.5 \pm 3.7$** accounts.
   * LightGBM: False negatives collapse from **$41.2 \pm 6.2$ down to $5.3 \pm 4.4$** accounts.
   * Recall rises from $\approx 94.7\%$ to **$99.1\%–99.3\%$**.
3. **The Controlled Trade-Off:** The cost-minimizing threshold shifts operational sensitivity to accept $\approx 150–185$ false alarms ($FP$) per 1,600 customers in exchange for saving $\approx 35$ defecting customers whose individual attrition costs $\$500$.

---

## 5. Cost Sensitivity Analysis Across Asymmetric Cost Ratios

Evaluating optimal operating thresholds as the business cost ratio scales from 3:1 to 20:1:

| Model Architecture | Ratio ($C_{FN}:C_{FP}$) | Assumed Costs ($C_{FN}, C_{FP}$) | Optimal $\tau^*$ | Recall (\%) | Precision (\%) | Accuracy (\%) | Expected Total Cost ($) | Cost / Customer ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stacking Ensemble** | 3 : 1 | $\$300, \$100$ | $0.15 \pm 0.05$ | $97.00 \pm 0.89$ | $91.28 \pm 1.29$ | $93.85 \pm 0.55$ | $\$14,640 \pm \$1,440$ | $\$9.15 \pm \$0.90$ |
| | 5 : 1 | $\$500, \$100$ | $0.07 \pm 0.02$ | $98.34 \pm 0.65$ | $87.75 \pm 1.63$ | $92.28 \pm 0.85$ | $\$17,670 \pm \$1,634$ | $\$11.04 \pm \$1.02$ |
| | **10 : 1 (Base)** | $\mathbf{\$500, \$50}$ | $\mathbf{0.04 \pm 0.01}$ | $\mathbf{99.06 \pm 0.52}$ | $\mathbf{83.91 \pm 1.85}$ | $\mathbf{90.01 \pm 1.16}$ | $\mathbf{\$11,370 \pm \$1,494}$ | $\mathbf{\$7.11 \pm \$0.93}$ |
| | 20 : 1 | $\$1000, \$50$ | $0.03 \pm 0.00$ | $99.50 \pm 0.49$ | $79.64 \pm 3.03$ | $86.94 \pm 2.14$ | $\$14,245 \pm \$2,480$ | $\$8.90 \pm \$1.55$ |
| **XGBoost** | 3 : 1 | $\$300, \$100$ | $0.22 \pm 0.04$ | $97.35 \pm 0.68$ | $90.43 \pm 1.39$ | $93.51 \pm 0.71$ | $\$14,620 \pm \$1,410$ | $\$9.14 \pm \$0.88$ |
| | 5 : 1 | $\$500, \$100$ | $0.15 \pm 0.03$ | $98.34 \pm 0.66$ | $88.07 \pm 1.54$ | $92.49 \pm 0.83$ | $\$17,340 \pm \$1,964$ | $\$10.84 \pm \$1.23$ |
| | **10 : 1 (Base)** | $\mathbf{\$500, \$50}$ | $\mathbf{0.06 \pm 0.02}$ | $\mathbf{99.31 \pm 0.46}$ | $\mathbf{83.33 \pm 1.97}$ | $\mathbf{89.69 \pm 1.21}$ | $\mathbf{\$10,725 \pm \$1,079}$ | $\mathbf{\$6.70 \pm \$0.67}$ |
| | 20 : 1 | $\$1000, \$50$ | $0.03 \pm 0.01$ | $99.70 \pm 0.24$ | $78.79 \pm 2.20$ | $86.39 \pm 1.77$ | $\$13,170 \pm \$1,899$ | $\$8.23 \pm \$1.19$ |
| **LightGBM** | 3 : 1 | $\$300, \$100$ | $0.22 \pm 0.05$ | $97.47 \pm 0.87$ | $90.37 \pm 1.15$ | $93.53 \pm 0.57$ | $\$14,390 \pm \$1,684$ | $\$8.99 \pm \$1.05$ |
| | 5 : 1 | $\$500, \$100$ | $0.11 \pm 0.03$ | $98.75 \pm 0.52$ | $86.60 \pm 1.86$ | $91.71 \pm 1.10$ | $\$17,270 \pm \$1,665$ | $\$10.79 \pm \$1.04$ |
| | **10 : 1 (Base)** | $\mathbf{\$500, \$50}$ | $\mathbf{0.06 \pm 0.02}$ | $\mathbf{99.34 \pm 0.56}$ | $\mathbf{83.22 \pm 2.51}$ | $\mathbf{89.60 \pm 1.68}$ | $\mathbf{\$10,705 \pm \$1,659}$ | $\mathbf{\$6.69 \pm \$1.04}$ |
| | 20 : 1 | $\$1000, \$50$ | $0.02 \pm 0.01$ | $99.79 \pm 0.18$ | $78.95 \pm 1.83$ | $86.56 \pm 1.46$ | $\$12,370 \pm \$1,586$ | $\$7.73 \pm \$0.99$ |
| **Random Forest** | 3 : 1 | $\$300, \$100$ | $0.29 \pm 0.03$ | $96.48 \pm 0.98$ | $91.33 \pm 1.17$ | $93.65 \pm 0.58$ | $\$15,800 \pm \$1,928$ | $\$9.88 \pm \$1.20$ |
| | 5 : 1 | $\$500, \$100$ | $0.19 \pm 0.03$ | $97.74 \pm 0.65$ | $87.78 \pm 1.74$ | $92.04 \pm 0.99$ | $\$19,970 \pm \$2,129$ | $\$12.48 \pm \$1.33$ |
| | **10 : 1 (Base)** | $\mathbf{\$500, \$50}$ | $\mathbf{0.10 \pm 0.01}$ | $\mathbf{98.97 \pm 0.43}$ | $\mathbf{81.08 \pm 1.43}$ | $\mathbf{87.92 \pm 0.98}$ | $\mathbf{\$13,355 \pm \$1,193}$ | $\mathbf{\$8.35 \pm \$0.75}$ |
| | 20 : 1 | $\$1000, \$50$ | $0.06 \pm 0.01$ | $99.51 \pm 0.33$ | $77.31 \pm 1.88$ | $85.12 \pm 1.53$ | $\$15,610 \pm \$2,041$ | $\$9.76 \pm \$1.28$ |

*Sensitivity Finding:* As the relative cost of losing a customer increases from 3:1 to 20:1, the optimal operating threshold monotonically decreases from $\approx 0.22$ down to $\approx 0.03$. Consequently, Recall monotonically increases from $\approx 97\%$ up to $99.7\%$.

---

## 6. Generated Publication Figures

1. **Figure 1 (`paper_figures/cost_vs_threshold_curves.png`):** Shows expected business cost on the y-axis versus classification threshold $\tau \in [0.01, 0.99]$ on the x-axis for Random Forest, XGBoost, LightGBM, and Stacking Ensemble under the 10:1 primary baseline, marking the minimum cost point $\tau^*$ and default $\tau=0.50$.
2. **Figure 2 (`paper_figures/cost_sensitivity_curves.png`):** Shows normalized cost per customer across 3:1, 5:1, 10:1, and 20:1 ratios for the Stacking Ensemble, illustrating optimal operating point shifts under varying business cost assumptions.

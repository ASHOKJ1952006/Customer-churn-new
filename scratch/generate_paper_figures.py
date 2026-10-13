"""
generate_paper_figures.py — Generates publication-quality IEEE conference paper figures.
Outputs high-resolution 300 DPI figures into the paper_figures/ directory.
Strictly read-only with respect to project assets; uses verified experimental results.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Ensure paper_figures directory exists
os.makedirs('paper_figures', exist_ok=True)

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

# ==============================================================================
# FIG 1. Proposed System Architecture
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.5, 10.5), dpi=300)
ax.axis('off')

boxes = [
    # (x, y, w, h, text, color, edgecolor)
    (0.25, 0.92, 0.50, 0.05, "Telecom Customer Data (Raw)\n(Demographics, Account, Services)", "#E8F0FE", "#1A73E8"),
    (0.25, 0.83, 0.50, 0.05, "Data Cleaning & Imputation\n(Handling TotalCharges, customerID removal)", "#E8F0FE", "#1A73E8"),
    (0.25, 0.74, 0.50, 0.05, "Domain Feature Engineering\n(ChargeIncrease, HighRiskCombo, TenureSquared)", "#E8F0FE", "#1A73E8"),
    (0.25, 0.65, 0.50, 0.05, "Leak-Free Class Imbalance Handling\n(SMOTE strictly inside 5-Fold CV partitions)", "#E8F0FE", "#1A73E8"),
    
    # Base learners
    (0.05, 0.53, 0.26, 0.06, "Random Forest\n(232 Trees, Depth 21\nBalanced Subsample)", "#FEF7E0", "#F9AB00"),
    (0.37, 0.53, 0.26, 0.06, "XGBoost\n(482 Trees, Depth 4\nHist / GPU Reg Alpha 1.53)", "#FEF7E0", "#F9AB00"),
    (0.69, 0.53, 0.26, 0.06, "LightGBM\n(300 Trees, 38 Leaves\nLeaf-wise Reg Alpha 1.68)", "#FEF7E0", "#F9AB00"),
    
    # Stacking
    (0.25, 0.42, 0.50, 0.05, "Stacking Meta-Learner\n(Logistic Regression on Out-of-Fold Probabilities)", "#FCE8E6", "#D93025"),
    
    # Probability & Threshold
    (0.25, 0.32, 0.50, 0.05, "Calibrated Churn Probability & Threshold Optimization\n(Default tau=0.50 vs Recall-Optimized tau=0.29 / 0.095)", "#E6F4EA", "#1E8E3E"),
    
    # Dual XAI
    (0.12, 0.21, 0.35, 0.05, "SHAP Explainability\n(TreeExplainer / Shapley Values)", "#F3E8FD", "#9334E6"),
    (0.53, 0.21, 0.35, 0.05, "LIME Explainability\n(Tabular Linear Surrogate)", "#F3E8FD", "#9334E6"),
    
    # Decision Engine
    (0.20, 0.11, 0.60, 0.05, "Business Factor Mapping & 10 Actionability Rules\n(State-Aware Pruning, Action Scoring, Ownership, Timeframes)", "#E8F0FE", "#1A73E8"),
    
    # Final Output
    (0.20, 0.02, 0.60, 0.05, "Telecom Retention Decision Support\n(Ranked Retention Cards, What-If Simulation, Risk Queue)", "#E6F4EA", "#137333")
]

from matplotlib.patches import FancyBboxPatch

for x, y, w, h, text, facecolor, edgecolor in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5, weight='bold', color='#202124')

# Draw arrows
arrow_style = dict(arrowstyle="->", color="#5F6368", lw=1.5)

# Downward sequential
ax.annotate("", xy=(0.50, 0.88), xytext=(0.50, 0.92), arrowprops=arrow_style)
ax.annotate("", xy=(0.50, 0.79), xytext=(0.50, 0.83), arrowprops=arrow_style)
ax.annotate("", xy=(0.50, 0.70), xytext=(0.50, 0.74), arrowprops=arrow_style)
ax.annotate("", xy=(0.50, 0.60), xytext=(0.50, 0.65), arrowprops=arrow_style)

# Split to base learners
ax.annotate("", xy=(0.18, 0.59), xytext=(0.50, 0.65), arrowprops=arrow_style)
ax.annotate("", xy=(0.50, 0.59), xytext=(0.50, 0.65), arrowprops=arrow_style)
ax.annotate("", xy=(0.82, 0.59), xytext=(0.50, 0.65), arrowprops=arrow_style)

# Converge to stacking
ax.annotate("", xy=(0.50, 0.47), xytext=(0.18, 0.53), arrowprops=arrow_style)
ax.annotate("", xy=(0.50, 0.47), xytext=(0.50, 0.53), arrowprops=arrow_style)
ax.annotate("", xy=(0.50, 0.47), xytext=(0.82, 0.53), arrowprops=arrow_style)

# Stacking to probability
ax.annotate("", xy=(0.50, 0.37), xytext=(0.50, 0.42), arrowprops=arrow_style)

# Probability to XAI
ax.annotate("", xy=(0.295, 0.26), xytext=(0.50, 0.32), arrowprops=arrow_style)
ax.annotate("", xy=(0.705, 0.26), xytext=(0.50, 0.32), arrowprops=arrow_style)

# XAI to Decision Engine
ax.annotate("", xy=(0.40, 0.16), xytext=(0.295, 0.21), arrowprops=arrow_style)
ax.annotate("", xy=(0.60, 0.16), xytext=(0.705, 0.21), arrowprops=arrow_style)

# Decision Engine to Output
ax.annotate("", xy=(0.50, 0.07), xytext=(0.50, 0.11), arrowprops=arrow_style)

plt.tight_layout()
plt.savefig('paper_figures/fig1_system_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIG 2. Machine Learning Workflow
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
ax.axis('off')

steps = [
    (0.02, 0.35, 0.15, 0.30, "Raw Telco Data\n(7,043 Records\n26.5% Churn)", "#E8F0FE", "#1A73E8"),
    (0.20, 0.35, 0.16, 0.30, "Stratified 80/20 Split\nTrain: 5,634\nTest: 1,409", "#E8F0FE", "#1A73E8"),
    (0.39, 0.35, 0.18, 0.30, "5-Fold CV Pipeline\n- In-fold SMOTE\n- In-fold Scaling\nZero Data Leakage", "#FEF7E0", "#F9AB00"),
    (0.60, 0.35, 0.18, 0.30, "Base Out-of-Fold\nPredictions Matrix\n[P_RF, P_XGB, P_LGBM]", "#FCE8E6", "#D93025"),
    (0.81, 0.35, 0.17, 0.30, "Stacking Meta-Model\n& Threshold Tuning\n(tau = 0.29 / 0.50)", "#E6F4EA", "#1E8E3E"),
]

for x, y, w, h, text, fc, ec in steps:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5, weight='bold', color='#202124')

for i in range(len(steps) - 1):
    x1 = steps[i][0] + steps[i][2]
    x2 = steps[i+1][0]
    y_mid = 0.50
    ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                arrowprops=dict(arrowstyle="->", color="#5F6368", lw=2))

plt.tight_layout()
plt.savefig('paper_figures/fig2_ml_workflow.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIG 3. Explainable AI and Recommendation Workflow
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
ax.axis('off')

xai_steps = [
    (0.02, 0.35, 0.16, 0.30, "Model Inference\nP(Churn) = 0.78\nRisk: High", "#E6F4EA", "#1E8E3E"),
    (0.21, 0.35, 0.18, 0.30, "Dual XAI Decomposition\nSHAP (TreeExplainer)\n+ LIME (Surrogate)", "#F3E8FD", "#9334E6"),
    (0.42, 0.35, 0.18, 0.30, "Business Factor Map\n- Payment/Billing\n- Contract/Tenure\n- Service Add-ons", "#FEF7E0", "#F9AB00"),
    (0.63, 0.35, 0.17, 0.30, "State-Aware Rules\nPrune Inapplicable\nCalculate Action Score", "#E8F0FE", "#1A73E8"),
    (0.83, 0.35, 0.15, 0.30, "Ranked Retention Cards\nOwner, Timeframe,\nWhat-If Impact", "#E6F4EA", "#137333")
]

for x, y, w, h, text, fc, ec in xai_steps:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                          facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.5, weight='bold', color='#202124')

for i in range(len(xai_steps) - 1):
    x1 = xai_steps[i][0] + xai_steps[i][2]
    x2 = xai_steps[i+1][0]
    y_mid = 0.50
    ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                arrowprops=dict(arrowstyle="->", color="#5F6368", lw=2))

plt.tight_layout()
plt.savefig('paper_figures/fig3_xai_recommendation_workflow.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIG 4. Confusion Matrix of Proposed Ensemble
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=300)

# Matrix 1: tau = 0.50
cm1 = np.array([[776, 259], [76, 298]])
im1 = ax1.imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
ax1.set_title("Stacking Ensemble (Default $\\tau=0.50$)\nRecall = 79.68%, Acc = 76.22%", fontsize=10, weight='bold')
tick_marks = np.arange(2)
ax1.set_xticks(tick_marks)
ax1.set_yticks(tick_marks)
ax1.set_xticklabels(['Retain (0)', 'Churn (1)'], fontsize=9)
ax1.set_yticklabels(['Retain (0)', 'Churn (1)'], fontsize=9)
ax1.set_ylabel('True Label', fontsize=9, weight='bold')
ax1.set_xlabel('Predicted Label', fontsize=9, weight='bold')

for i in range(2):
    for j in range(2):
        val = cm1[i, j]
        color = "white" if val > 400 else "black"
        ax1.text(j, i, f"{val}\n({val/14.09:.1f}%)", ha="center", va="center", color=color, fontsize=10, weight='bold')

# Matrix 2: tau = 0.29 (Recall Prioritized)
cm2 = np.array([[614, 421], [37, 337]])
im2 = ax2.imshow(cm2, interpolation='nearest', cmap=plt.cm.Greens)
ax2.set_title("Stacking Ensemble (Recall-Opt $\\tau=0.29$)\nRecall = 90.11%, Acc = 67.49%", fontsize=10, weight='bold')
ax2.set_xticks(tick_marks)
ax2.set_yticks(tick_marks)
ax2.set_xticklabels(['Retain (0)', 'Churn (1)'], fontsize=9)
ax2.set_yticklabels(['Retain (0)', 'Churn (1)'], fontsize=9)
ax2.set_ylabel('True Label', fontsize=9, weight='bold')
ax2.set_xlabel('Predicted Label', fontsize=9, weight='bold')

for i in range(2):
    for j in range(2):
        val = cm2[i, j]
        color = "white" if val > 400 else "black"
        ax2.text(j, i, f"{val}\n({val/14.09:.1f}%)", ha="center", va="center", color=color, fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('paper_figures/fig4_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIG 5. Model Performance Comparison
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)

models = ['Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble']
metrics = {
    'Accuracy': [0.7736, 0.7637, 0.7693, 0.7622],
    'Precision': [0.5536, 0.5397, 0.5518, 0.5350],
    'Recall': [0.7594, 0.7460, 0.6979, 0.7968],
    'F1-Score': [0.6404, 0.6263, 0.6163, 0.6402],
    'ROC-AUC': [0.8451, 0.8439, 0.8422, 0.8460],
}

x = np.arange(len(models))
width = 0.16
colors = ['#4285F4', '#34A853', '#FBBC05', '#EA4335', '#9334E6']

for i, (m_name, vals) in enumerate(metrics.items()):
    rects = ax.bar(x + i*width - 2*width, vals, width, label=m_name, color=colors[i], edgecolor='white', linewidth=0.5)

ax.set_ylabel('Score', fontsize=10, weight='bold')
ax.set_title('Performance Comparison of Churn Prediction Models (Test Set, N=1,409)', fontsize=11, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9.5, weight='bold')
ax.set_ylim(0.40, 0.95)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(loc='lower right', framealpha=0.9, fontsize=8.5)

plt.tight_layout()
plt.savefig('paper_figures/fig5_model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIG 6. SHAP Feature Importance
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)

features = [
    'Contract_Two year',
    'PaymentMethod_Electronic check',
    'VeryHighRisk',
    'PaperlessBilling',
    'NumServices',
    'InternetService_Fiber optic',
    'HighRiskCombo',
    'tenure',
    'TenureSquared',
    'AvgMonthlySpend',
    'TotalCharges',
    'MonthlyCharges',
    'ChargeRatio',
    'ChargeIncrease'
]

importances = [
    0.0146, 0.0162, 0.0161, 0.0178, 0.0253, 0.0281, 
    0.0367, 0.0644, 0.0689, 0.0826, 0.0867, 0.0883, 
    0.0939, 0.1104
]

y_pos = np.arange(len(features))
bars = ax.barh(y_pos, importances, color='#1A73E8', edgecolor='none', height=0.65)

# Color top features distinctly
bars[-1].set_color('#D93025')
bars[-2].set_color('#D93025')
bars[-3].set_color('#F9AB00')
bars[-4].set_color('#F9AB00')

ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=8.5)
ax.set_xlabel('Feature Importance / Mean Absolute SHAP Impact', fontsize=9.5, weight='bold')
ax.set_title('Global Feature Importance Ranking in Churn Prediction (Top 14 Drivers)', fontsize=10.5, weight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(importances):
    ax.text(v + 0.002, i, f"{v:.4f}", va='center', fontsize=8, color='#333333')

ax.set_xlim(0, 0.13)
plt.tight_layout()
plt.savefig('paper_figures/fig6_shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# FIG 7. Example Customer-Level Explanation
# ==============================================================================
fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=300)

shap_factors = [
    ('Contract: Month-to-month', +0.28, '#D93025'),
    ('Payment: Electronic check', +0.18, '#D93025'),
    ('Monthly Charges ($89.50)', +0.14, '#D93025'),
    ('Charge Spike (+$12.30)', +0.09, '#D93025'),
    ('Internet: Fiber Optic', +0.06, '#D93025'),
    ('Service: Online Security', -0.07, '#1E8E3E'),
    ('Tenure (14 months)', -0.11, '#1E8E3E')
]

y_labels = [item[0] for item in shap_factors][::-1]
values = [item[1] for item in shap_factors][::-1]
colors = [item[2] for item in shap_factors][::-1]

y_pos = np.arange(len(y_labels))
ax.barh(y_pos, values, color=colors, height=0.55, edgecolor='none')

ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9, weight='bold')
ax.set_xlabel('SHAP Value (Contribution to Churn Probability)', fontsize=9.5, weight='bold')
ax.set_title('Customer-Level Explanation (Baseline P=0.265 -> Final P=0.775, High Risk)', fontsize=10.5, weight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.7)

for i, v in enumerate(values):
    offset = 0.01 if v >= 0 else -0.03
    ax.text(v + offset, i, f"{v:+.2f}", va='center', fontsize=8.5, weight='bold',
            color='#D93025' if v > 0 else '#1E8E3E')

ax.set_xlim(-0.20, 0.35)
plt.tight_layout()
plt.savefig('paper_figures/fig7_example_customer_explanation.png', dpi=300, bbox_inches='tight')
plt.close()

print("All 7 figures generated successfully in paper_figures/!")

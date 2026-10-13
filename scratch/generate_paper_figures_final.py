"""
generate_paper_figures_final.py — Generates publication-quality IEEE conference figures
for the CURRENT balanced_telco_churn.csv experiment and primary source of truth metrics.
Figures 1 to 7 at 300 DPI.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Output directory
FIGURES_DIR = 'paper_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

# Primary Source of Truth Data
models = ['Random Forest', 'XGBoost', 'LightGBM', 'Stacking Ensemble']
model_colors = ['#2563eb', '#d97706', '#059669', '#7c3aed']
recall_vals = [96.3, 97.6, 96.9, 96.3]
accuracy_vals = [95.4, 94.9, 95.0, 95.4]
f1_vals = [95.4, 95.0, 95.1, 95.5]
roc_auc_vals = [98.7, 99.0, 98.9, 98.9]
precision_vals = [94.6, 92.5, 93.4, 94.7]
threshold_vals = [0.38, 0.29, 0.32, 0.26]
brier_vals = [0.0369, 0.0347, 0.0353, 0.0362]

# ==============================================================================
# FIG 1. Overall System Architecture
# ==============================================================================
def generate_fig1():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.0, 11.5), dpi=300)
    ax.axis('off')

    boxes = [
        # (x, y, w, h, text, facecolor, edgecolor)
        (0.20, 0.92, 0.60, 0.05, "Customer Input Interface\n(Tenure, MonthlyCharges, TotalCharges, Contract, InternetService, PaymentMethod)", "#E8F0FE", "#1A73E8"),
        (0.20, 0.83, 0.60, 0.05, "Data Cleaning & Preprocessing\n(Missing Value Coercion, Scaler Imputation, Quality Assurances)", "#E8F0FE", "#1A73E8"),
        (0.20, 0.74, 0.60, 0.05, "Domain Feature Engineering (27 Features)\n(Spend Ratios, Non-linear Terms, HighRiskCombo, One-Hot Encodings)", "#E8F0FE", "#1A73E8"),
        (0.20, 0.65, 0.60, 0.05, "Balanced Dataset (balanced_telco_churn.csv)\n(N=8,000, 50% Churn / 50% Non-Churn, 80/20 Stratified Split)", "#E8F0FE", "#1A73E8"),
        
        # Base models
        (0.04, 0.53, 0.28, 0.065, "Random Forest\n(250 Trees, max_depth=12\nBagged Decision Trees)", "#FEF7E0", "#F9AB00"),
        (0.36, 0.53, 0.28, 0.065, "XGBoost\n(200 Trees, lr=0.06, depth=5\nGradient Boosting with L1/L2)", "#FEF7E0", "#F9AB00"),
        (0.68, 0.53, 0.28, 0.065, "LightGBM\n(200 Trees, lr=0.06, 31 leaves\nLeaf-wise Tree Growth)", "#FEF7E0", "#F9AB00"),
        
        # Stacking
        (0.20, 0.42, 0.60, 0.055, "Stacking Ensemble Meta-Learner\n(Logistic Regression on 5-Fold Out-of-Fold Probability Vectors)", "#FCE8E6", "#D93025"),
        
        # Threshold Optimization
        (0.20, 0.32, 0.60, 0.055, "Threshold Optimization & Calibration\n(RF: 0.38, XGB: 0.29, LGBM: 0.32, Stacking: 0.26)", "#E6F4EA", "#1E8E3E"),
        
        # Dual Explainability
        (0.10, 0.21, 0.38, 0.055, "SHAP Explainability\n(TreeExplainer & Kernel Fallback\nDirectional Attribution & Baseline)", "#F3E8FD", "#9334E6"),
        (0.52, 0.21, 0.38, 0.055, "LIME Explainability\n(Tabular Linear Surrogates\nLocal Instance Explainability)", "#F3E8FD", "#9334E6"),
        
        # Retention Decision Engine
        (0.15, 0.11, 0.70, 0.055, "Retention Recommendation Engine\n(State-Aware Rule Pruning, Action Scoring, Owner/Timeframe/Cost Mapping)", "#E8F0FE", "#1A73E8"),
        
        # Delivery Layer
        (0.15, 0.02, 0.70, 0.055, "Enterprise Full-Stack Application\n(FastAPI REST Endpoints, React 19 UI, What-If Simulator, SQLite DB)", "#E6F4EA", "#137333")
    ]

    for x, y, w, h, text, facecolor, edgecolor in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                              facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=8.8, weight='bold', color='#202124')

    arrow_style = dict(arrowstyle="->", color="#4B5563", lw=1.6)

    # Downward connectors
    ax.annotate("", xy=(0.50, 0.88), xytext=(0.50, 0.92), arrowprops=arrow_style)
    ax.annotate("", xy=(0.50, 0.79), xytext=(0.50, 0.83), arrowprops=arrow_style)
    ax.annotate("", xy=(0.50, 0.70), xytext=(0.50, 0.74), arrowprops=arrow_style)
    ax.annotate("", xy=(0.50, 0.60), xytext=(0.50, 0.65), arrowprops=arrow_style)

    # Fan-out to base models
    ax.annotate("", xy=(0.18, 0.60), xytext=(0.50, 0.65), arrowprops=arrow_style)
    ax.annotate("", xy=(0.50, 0.60), xytext=(0.50, 0.65), arrowprops=arrow_style)
    ax.annotate("", xy=(0.82, 0.60), xytext=(0.50, 0.65), arrowprops=arrow_style)

    # Fan-in to stacking
    ax.annotate("", xy=(0.50, 0.48), xytext=(0.18, 0.53), arrowprops=arrow_style)
    ax.annotate("", xy=(0.50, 0.48), xytext=(0.50, 0.53), arrowprops=arrow_style)
    ax.annotate("", xy=(0.50, 0.48), xytext=(0.82, 0.53), arrowprops=arrow_style)

    # Stacking to threshold
    ax.annotate("", xy=(0.50, 0.38), xytext=(0.50, 0.42), arrowprops=arrow_style)

    # Threshold to XAI
    ax.annotate("", xy=(0.29, 0.27), xytext=(0.50, 0.32), arrowprops=arrow_style)
    ax.annotate("", xy=(0.71, 0.27), xytext=(0.50, 0.32), arrowprops=arrow_style)

    # XAI to Engine
    ax.annotate("", xy=(0.38, 0.17), xytext=(0.29, 0.21), arrowprops=arrow_style)
    ax.annotate("", xy=(0.62, 0.17), xytext=(0.71, 0.21), arrowprops=arrow_style)

    # Engine to Delivery
    ax.annotate("", xy=(0.50, 0.08), xytext=(0.50, 0.11), arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_system_architecture.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 1 generated successfully.")

# ==============================================================================
# FIG 2. ML Pipeline Workflow
# ==============================================================================
def generate_fig2():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=300)
    ax.axis('off')

    stages = [
        ("1. Dataset Synthesis", "balanced_telco_churn.csv\n8,000 samples (50/50)\nDomain distributions\n5% Calibrated Noise", "#EBF5FB", "#2E86C1"),
        ("2. Stratified Partitioning", "80% Training (6,400)\n20% Held-Out Test (1,600)\nStrict target stratification\nNo cross-split leakage", "#EAF2F8", "#3498DB"),
        ("3. 5-Fold Stratified CV", "Train Base Learners\nExtract Out-of-Fold Probs\nRF, XGBoost, LightGBM\nStandardScaler fitting", "#FEF9E7", "#F1C40F"),
        ("4. Meta-Learner Stacking", "Logistic Regression Meta\nTrained on OOF Probs\nEnsemble Probability\nZero Information Leakage", "#FDEDEC", "#E74C3C"),
        ("5. Threshold Sweep", "Threshold Grid 0.05-0.95\nRecall & Accuracy Targets\nRF: 0.38 | XGB: 0.29\nLGBM: 0.32 | Stack: 0.26", "#EAFAF1", "#2ECC71"),
        ("6. Final Test Evaluation", "1,600 Held-Out Samples\nEnsemble Recall: 96.3%\nAccuracy: 95.4% | F1: 95.5%\nROC-AUC: 98.9%", "#F4ECF7", "#8E44AD")
    ]

    w, h = 0.14, 0.72
    y_pos = 0.15

    for i, (title, desc, bg, edge) in enumerate(stages):
        x_pos = 0.02 + i * 0.165
        rect = FancyBboxPatch((x_pos, y_pos), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                              facecolor=bg, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_pos + w/2, y_pos + h - 0.12, title, ha='center', va='center', fontsize=8.5, weight='bold', color=edge)
        ax.plot([x_pos + 0.01, x_pos + w - 0.01], [y_pos + h - 0.22, y_pos + h - 0.22], color=edge, lw=1.0)
        ax.text(x_pos + w/2, y_pos + 0.26, desc, ha='center', va='center', fontsize=7.5, color='#2C3E50', multialignment='center')

        if i < len(stages) - 1:
            ax.annotate("", xy=(x_pos + w + 0.023, y_pos + h/2), xytext=(x_pos + w + 0.002, y_pos + h/2),
                        arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=2.0))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_ml_pipeline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 2 generated successfully.")

# ==============================================================================
# FIG 3. Model Performance Comparison (Grouped Bar Chart)
# ==============================================================================
def generate_fig3():
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=300)
    
    metrics = ['Recall', 'Accuracy', 'F1-Score', 'Precision', 'ROC-AUC']
    x = np.arange(len(metrics))
    width = 0.18

    # Data matrix
    data_rf = [96.3, 95.4, 95.4, 94.6, 98.7]
    data_xgb = [97.6, 94.9, 95.0, 92.5, 99.0]
    data_lgbm = [96.9, 95.0, 95.1, 93.4, 98.9]
    data_stack = [96.3, 95.4, 95.5, 94.7, 98.9]

    rects1 = ax.bar(x - 1.5*width, data_rf, width, label='Random Forest (τ=0.38)', color='#2563eb', alpha=0.9)
    rects2 = ax.bar(x - 0.5*width, data_xgb, width, label='XGBoost (τ=0.29)', color='#d97706', alpha=0.9)
    rects3 = ax.bar(x + 0.5*width, data_lgbm, width, label='LightGBM (τ=0.32)', color='#059669', alpha=0.9)
    rects4 = ax.bar(x + 1.5*width, data_stack, width, label='Stacking Ensemble (τ=0.26)', color='#7c3aed', alpha=0.9)

    ax.set_ylabel('Score (%)', fontsize=10, weight='bold')
    ax.set_title('Primary Model Performance Comparison Across Core Evaluation Metrics', fontsize=11, weight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9.5, weight='bold')
    ax.set_ylim(88, 101)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.95)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.0, weight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 3 generated successfully.")

# ==============================================================================
# FIG 4. Accuracy Comparison
# ==============================================================================
def generate_fig4():
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    
    bars = ax.bar(models, accuracy_vals, color=model_colors, width=0.55, edgecolor='#333333', linewidth=0.8)
    ax.set_ylabel('Accuracy (%)', fontsize=10, weight='bold')
    ax.set_title('Test Set Classification Accuracy Across Models', fontsize=11, weight='bold', pad=10)
    ax.set_ylim(92, 97.5)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.0, weight='bold')

    ax.axhline(95.0, color='#666666', linestyle=':', lw=1.0, alpha=0.8, label='95.0% Baseline Benchmark')
    ax.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 4 generated successfully.")

# ==============================================================================
# FIG 5. Recall Comparison
# ==============================================================================
def generate_fig5():
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    
    bars = ax.bar(models, recall_vals, color=model_colors, width=0.55, edgecolor='#333333', linewidth=0.8)
    ax.set_ylabel('Recall (%)', fontsize=10, weight='bold')
    ax.set_title('Test Set Churn Recall (Sensitivity) Across Models', fontsize=11, weight='bold', pad=10)
    ax.set_ylim(94, 99.5)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.0, weight='bold')

    ax.axhline(96.0, color='#666666', linestyle=':', lw=1.0, alpha=0.8, label='96.0% Target Recall Level')
    ax.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_recall_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 5 generated successfully.")

# ==============================================================================
# FIG 6. ROC-AUC Comparison
# ==============================================================================
def generate_fig6():
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
    
    bars = ax.bar(models, roc_auc_vals, color=model_colors, width=0.55, edgecolor='#333333', linewidth=0.8)
    ax.set_ylabel('ROC-AUC (%)', fontsize=10, weight='bold')
    ax.set_title('Area Under ROC Curve (ROC-AUC) Across Models', fontsize=11, weight='bold', pad=10)
    ax.set_ylim(97, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.0, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_roc_auc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 6 generated successfully.")

# ==============================================================================
# FIG 7. Explainable AI / SHAP-LIME Workflow
# ==============================================================================
def generate_fig7():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(9.0, 5.6), dpi=300)
    ax.axis('off')

    boxes = [
        (0.04, 0.72, 0.42, 0.22, "SHAP Local Attribution\n- TreeExplainer / KernelExplainer\n- Game-theoretic Shapley values\n- Additive feature attribution\n- Directional sign & magnitude", "#F5EEF8", "#8E44AD"),
        (0.54, 0.72, 0.42, 0.22, "LIME Local Surrogates\n- Tabular linear interpretable model\n- Local perturbation sampling\n- Exponential kernel weighting\n- Local feature slope estimates", "#E8F8F5", "#16A085"),
        
        (0.24, 0.42, 0.52, 0.20, "Dual-XAI Alignment & Verification\n- Sign agreement verification\n- Normalized importance rank\n- Confidence score computation\n- Elimination of artifact discrepancies", "#FEF9E7", "#F39C12"),
        
        (0.04, 0.08, 0.42, 0.24, "State-Aware Rule Engine\n- Validates customer account status\n- Excludes impossible actions (e.g. M2M renewal on 2-year contract)\n- Evaluates multi-factor risk", "#EBF5FB", "#2980B9"),
        (0.54, 0.08, 0.42, 0.24, "Retention Action Scoring & Output\n- Score = (Risk × Prob × Rel × Imp) / Cost\n- Action Owner, Timeframe & Budget\n- Counterfactual What-If ML re-scoring\n- Executive retention summary", "#EAFAF1", "#27AE60")
    ]

    for x, y, w, h, text, bg, edge in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                              facecolor=bg, edgecolor=edge, linewidth=1.5)
        ax.add_patch(rect)
        lines = text.split('\n')
        title = lines[0]
        body = '\n'.join(lines[1:])
        ax.text(x + w/2, y + h - 0.045, title, ha='center', va='center', fontsize=8.8, weight='bold', color=edge)
        ax.plot([x + 0.02, x + w - 0.02], [y + h - 0.08, y + h - 0.08], color=edge, lw=1.0)
        ax.text(x + w/2, y + 0.08, body, ha='center', va='center', fontsize=7.6, color='#2C3E50')

    arrow_style = dict(arrowstyle="->", color="#34495E", lw=1.8)
    
    # SHAP & LIME down to Alignment
    ax.annotate("", xy=(0.38, 0.62), xytext=(0.25, 0.72), arrowprops=arrow_style)
    ax.annotate("", xy=(0.62, 0.62), xytext=(0.75, 0.72), arrowprops=arrow_style)

    # Alignment down to Rule Engine and Action Scoring
    ax.annotate("", xy=(0.25, 0.32), xytext=(0.40, 0.42), arrowprops=arrow_style)
    ax.annotate("", xy=(0.75, 0.32), xytext=(0.60, 0.42), arrowprops=arrow_style)

    # Rule Engine to Action Scoring
    ax.annotate("", xy=(0.54, 0.20), xytext=(0.46, 0.20), arrowprops=arrow_style)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_xai_recommendation_workflow.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 7 generated successfully.")

if __name__ == '__main__':
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    print("All 7 figures generated successfully at 300 DPI.")

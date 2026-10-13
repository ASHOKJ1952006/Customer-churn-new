# -*- coding: utf-8 -*-
"""
generate_ieee_paper_figures.py — Generates exact single-column IEEE conference figures
matching the reference layout:
  Fig 1: Architecture Diagram (Clean single-column flowchart)
  Fig 2: Evaluation Metrics Values for Random Forest Model (4-bar chart)
  Fig 3: Evaluation Metrics Values for Gradient Boosted Models (XGBoost & LightGBM)
  Fig 4: Evaluation Metrics Values for Proposed Stacking Model (4-bar chart)
  Fig 5: Working Flow Diagram of Decision Support System (Flowchart)
  Fig 6: Heatmap for All Algorithms
  Fig 7: Confusion Matrix for Proposed Stacking Model (2x2 matrix)

All figures generated at 300 DPI, precisely sized for IEEE single column (~3.45 in width).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

FIGURES_DIR = 'paper_figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

# ==============================================================================
# FIG 1. Architecture Diagram (Single-column vertical layout)
# ==============================================================================
def generate_fig1():
    fig, ax = plt.subplots(figsize=(3.45, 4.4), dpi=300)
    ax.axis('off')

    # Draw rounded boxes
    boxes = [
        # (x, y, w, h, text, bg, border)
        (0.06, 0.88, 0.88, 0.09, "Customer Account & Billing Data\n(6 Core Raw Attributes)", "#EBF5FB", "#2980B9"),
        (0.06, 0.74, 0.88, 0.09, "Preprocessing & Domain Feature Eng.\n(Missing Value Coercion, 27 Features)", "#EBF5FB", "#2980B9"),
        (0.06, 0.60, 0.88, 0.09, "Base Classifiers (5-Fold Stratified OOF)\n(Random Forest, XGBoost, LightGBM)", "#FEF9E7", "#D4AC0D"),
        (0.06, 0.46, 0.88, 0.09, "Stacking Meta-Learner & Calibration\n(Logistic Regression, tau* = 0.26)", "#FDEDEC", "#C0392B"),
        (0.06, 0.32, 0.88, 0.09, "Dual-XAI Attribution Engine\n(SHAP TreeExplainer & LIME Surrogates)", "#F4ECF7", "#8E44AD"),
        (0.06, 0.18, 0.88, 0.09, "State-Aware Retention Decision Engine\n(Eligibility Filtering & Action Scoring)", "#EAFAF1", "#27AE60"),
        (0.06, 0.04, 0.88, 0.09, "Operational Retention Delivery\n(Ranked Actions, What-If Simulator)", "#E8F8F5", "#16A085"),
    ]

    for x, y, w, h, text, bg, border in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                              facecolor=bg, edgecolor=border, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=7.2, weight='bold', color='#1A252F')

    # Connectors
    arrow_props = dict(arrowstyle="->", color="#34495E", lw=1.2)
    for y_top, y_bot in [(0.88, 0.83), (0.74, 0.69), (0.60, 0.55), (0.46, 0.41), (0.32, 0.27), (0.18, 0.13)]:
        ax.annotate("", xy=(0.50, y_bot), xytext=(0.50, y_top), arrowprops=arrow_props)

    plt.tight_layout(pad=0.1)
    out_p = os.path.join(FIGURES_DIR, 'fig1_system_architecture.png')
    plt.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 1 generated:", out_p)

# ==============================================================================
# FIG 2. Evaluation Metrics Values for Random Forest Model (4-bar chart)
# ==============================================================================
def generate_fig2():
    fig, ax = plt.subplots(figsize=(3.45, 2.7), dpi=300)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    rf_vals = [95.4, 94.6, 96.3, 95.4]
    colors = ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C']
    
    x = np.arange(len(metrics))
    bars = ax.bar(x, rf_vals, color=colors, width=0.55, edgecolor='#2C3E50', linewidth=0.7)
    
    ax.set_title('Random Forest Evaluation Metrics', fontsize=8.5, weight='bold', pad=8)
    ax.set_ylabel('Percentage (%)', fontsize=7.5, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7.2, weight='bold')
    ax.set_ylim(80, 105)
    ax.tick_params(axis='both', which='major', labelsize=6.8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, rf_vals):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.0, weight='bold')

    plt.tight_layout(pad=0.2)
    out_p = os.path.join(FIGURES_DIR, 'fig2_model_rf_metrics.png')
    plt.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 2 generated:", out_p)

# ==============================================================================
# FIG 3. Evaluation Metrics Values for Gradient Boosted Models (XGBoost & LightGBM)
# ==============================================================================
def generate_fig3():
    fig, ax = plt.subplots(figsize=(3.45, 2.7), dpi=300)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    xgb_vals = [94.9, 92.5, 97.6, 95.0]
    lgbm_vals = [95.0, 93.4, 96.9, 95.1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    b1 = ax.bar(x - width/2, xgb_vals, width, label='XGBoost (tau*=0.29)', color='#E67E22', edgecolor='#2C3E50', linewidth=0.6)
    b2 = ax.bar(x + width/2, lgbm_vals, width, label='LightGBM (tau*=0.32)', color='#2ECC71', edgecolor='#2C3E50', linewidth=0.6)
    
    ax.set_title('Gradient Boosted Models Evaluation Metrics', fontsize=8.2, weight='bold', pad=8)
    ax.set_ylabel('Percentage (%)', fontsize=7.5, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7.2, weight='bold')
    ax.set_ylim(80, 105)
    ax.tick_params(axis='both', which='major', labelsize=6.8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=6.2, framealpha=0.9)

    for bar in b1:
        val = bar.get_height()
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=5.8, weight='bold')
    for bar in b2:
        val = bar.get_height()
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=5.8, weight='bold')

    plt.tight_layout(pad=0.2)
    out_p = os.path.join(FIGURES_DIR, 'fig3_model_gb_metrics.png')
    plt.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 3 generated:", out_p)

# ==============================================================================
# FIG 4. Evaluation Metrics Values for Proposed Stacking Model (4-bar chart)
# ==============================================================================
def generate_fig4():
    fig, ax = plt.subplots(figsize=(3.45, 2.7), dpi=300)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    stack_vals = [95.4, 94.7, 96.3, 95.5]
    colors = ['#3498DB', '#E67E22', '#2ECC71', '#9B59B6']
    
    x = np.arange(len(metrics))
    bars = ax.bar(x, stack_vals, color=colors, width=0.55, edgecolor='#2C3E50', linewidth=0.7)
    
    ax.set_title('Stacking Ensemble Evaluation Metrics (tau*=0.26)', fontsize=8.2, weight='bold', pad=8)
    ax.set_ylabel('Percentage (%)', fontsize=7.5, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7.2, weight='bold')
    ax.set_ylim(80, 105)
    ax.tick_params(axis='both', which='major', labelsize=6.8)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars, stack_vals):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.0, weight='bold')

    plt.tight_layout(pad=0.2)
    out_p = os.path.join(FIGURES_DIR, 'fig4_model_stacking_metrics.png')
    plt.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 4 generated:", out_p)

# ==============================================================================
# FIG 5. Working Flow Diagram of Decision Support System
# ==============================================================================
def generate_fig5():
    fig, ax = plt.subplots(figsize=(3.45, 4.4), dpi=300)
    ax.axis('off')

    flow_boxes = [
        (0.08, 0.88, 0.84, 0.085, "Customer Account Input\n(Billing, Services, Contract)", "#FCF3CF", "#F39C12"),
        (0.08, 0.74, 0.84, 0.085, "Feature Extraction & Scaling\n(27 Engineered Features)", "#EBF5FB", "#2980B9"),
        (0.08, 0.60, 0.84, 0.085, "Meta-Learner Probability Score\n(Stacking Ensemble: 0.0 to 1.0)", "#FDEDEC", "#C0392B"),
        (0.08, 0.46, 0.84, 0.085, "Optimized Threshold Check\n(Prob >= 0.26 => At-Risk Subscriber)", "#EAF2F8", "#2E86C1"),
        (0.08, 0.32, 0.84, 0.085, "Dual-XAI Feature Attribution\n(SHAP & LIME Root-Cause Drivers)", "#F4ECF7", "#8E44AD"),
        (0.08, 0.18, 0.84, 0.085, "State-Aware Rule & Action Scoring\n(Pruning Ineligible Offers, Ranking ROI)", "#EAFAF1", "#27AE60"),
        (0.08, 0.04, 0.84, 0.085, "Operational Decision Output\n(Ranked Retention Package & Simulator)", "#D5F5E3", "#1E8449"),
    ]

    for x, y, w, h, text, bg, border in flow_boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                              facecolor=bg, edgecolor=border, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=7.1, weight='bold', color='#1A252F')

    arrow_props = dict(arrowstyle="->", color="#34495E", lw=1.2)
    for y_top, y_bot in [(0.88, 0.825), (0.74, 0.685), (0.60, 0.545), (0.46, 0.405), (0.32, 0.265), (0.18, 0.125)]:
        ax.annotate("", xy=(0.50, y_bot), xytext=(0.50, y_top), arrowprops=arrow_props)

    plt.tight_layout(pad=0.1)
    out_p = os.path.join(FIGURES_DIR, 'fig5_working_flow_diagram.png')
    plt.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 5 generated:", out_p)

# ==============================================================================
# FIG 6. Model Performance — Heatmap
# ==============================================================================
def generate_fig6():
    fig, ax = plt.subplots(figsize=(3.45, 2.7), dpi=300)

    models_labels = ['RF (Text/Num)', 'XGBoost', 'LightGBM', 'Stacking (Proposed)']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

    # Performance matrix (%)
    matrix = np.array([
        [95.4, 94.6, 96.3, 95.4, 98.7],
        [94.9, 92.5, 97.6, 95.0, 99.0],
        [95.0, 93.4, 96.9, 95.1, 98.9],
        [95.4, 94.7, 96.3, 95.5, 98.9]
    ])

    im = ax.imshow(matrix, cmap='Blues', aspect='auto', vmin=85, vmax=100)

    ax.set_xticks(np.arange(len(metrics_labels)))
    ax.set_yticks(np.arange(len(models_labels)))
    ax.set_xticklabels(metrics_labels, fontsize=6.2, weight='bold', rotation=15, ha='right')
    ax.set_yticklabels(models_labels, fontsize=6.5, weight='bold')
    ax.set_title('Model Performance — Heatmap', fontsize=8.2, weight='bold', pad=8)

    # Values in cells
    for i in range(len(models_labels)):
        for j in range(len(metrics_labels)):
            val = matrix[i, j]
            color = 'white' if val > 95.0 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=6.8, weight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6.2)
    cbar.set_label('%', fontsize=6.8, weight='bold')

    plt.tight_layout(pad=0.2)
    out_p = os.path.join(FIGURES_DIR, 'fig6_heatmap_all_algorithms.png')
    plt.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 6 generated:", out_p)

# ==============================================================================
# FIG 7. Confusion Matrix for Proposed Model
# ==============================================================================
def generate_fig7():
    fig, ax = plt.subplots(figsize=(3.45, 2.7), dpi=300)

    # Test set: 800 non-churn, 800 churn
    # TN=757, FP=43, FN=30, TP=770
    cm = np.array([
        [757, 43],
        [30, 770]
    ])

    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=800)

    classes = ['Loyal (0)', 'Churn (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=7.2, weight='bold')
    ax.set_yticklabels(classes, fontsize=7.2, weight='bold')
    ax.set_xlabel('Predicted Label', fontsize=7.5, weight='bold')
    ax.set_ylabel('Actual Label', fontsize=7.5, weight='bold')
    ax.set_title('Confusion Matrix: Stacking Model\nSamples: 1600 | Accuracy: 95.4%', fontsize=8.0, weight='bold', pad=8)

    # Text in cells
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            color = 'white' if count > 400 else 'black'
            rate = count / 800.0 * 100.0
            ax.text(j, i, f'{count}\n({rate:.1f}%)', ha='center', va='center',
                    color=color, fontsize=7.5, weight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6.2)

    plt.tight_layout(pad=0.2)
    out_p = os.path.join(FIGURES_DIR, 'fig7_confusion_matrix_proposed.png')
    plt.savefig(out_p, dpi=300, bbox_inches='tight')
    plt.close()
    print("Fig 7 generated:", out_p)

if __name__ == '__main__':
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    print("All 7 IEEE single-column figures generated successfully.")

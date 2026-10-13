"""
build_final_paper.py — Compiles the final IEEE conference research paper across
all required file formats: LaTeX (.tex), HTML, PDF, and DOCX.
Based strictly on the CURRENT balanced_telco_churn.csv experiment and PRIMARY SOURCE OF TRUTH metrics.
"""

import os
import sys
import subprocess
import zipfile
import base64
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGURES_DIR = os.path.join(ROOT_DIR, 'paper_figures')

# Primary metrics dictionary
METRICS = {
    'RF': {'recall': '96.3%', 'f1': '95.4%', 'accuracy': '95.4%', 'roc_auc': '98.7%', 'brier': '0.0369', 'precision': '94.6%', 'threshold': '0.38', 'tn': 756, 'fp': 44, 'fn': 30, 'tp': 770},
    'XGB': {'recall': '97.6%', 'f1': '95.0%', 'accuracy': '94.9%', 'roc_auc': '99.0%', 'brier': '0.0347', 'precision': '92.5%', 'threshold': '0.29', 'tn': 737, 'fp': 63, 'fn': 19, 'tp': 781},
    'LGBM': {'recall': '96.9%', 'f1': '95.1%', 'accuracy': '95.0%', 'roc_auc': '98.9%', 'brier': '0.0353', 'precision': '93.4%', 'threshold': '0.32', 'tn': 745, 'fp': 55, 'fn': 25, 'tp': 775},
    'STACK': {'recall': '96.3%', 'f1': '95.5%', 'accuracy': '95.4%', 'roc_auc': '98.9%', 'brier': '0.0362', 'precision': '94.7%', 'threshold': '0.26', 'tn': 757, 'fp': 43, 'fn': 30, 'tp': 770},
}

def get_base64_image(filename):
    p = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""

print("Loaded primary source of truth metrics.")

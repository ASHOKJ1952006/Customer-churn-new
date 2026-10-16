#!/usr/bin/env python3
"""
evaluate_models.py — Terminal Model Evaluation & Metrics Dashboard

Displays the exact performance metrics shown in the Model Comparison dashboard:
  - Recall, F1 Score, Accuracy, ROC-AUC, Brier Score, Precision, Threshold
Allows running each model separately or all together.

Usage Examples:
  python scripts/evaluate_models.py --model rf
  python scripts/evaluate_models.py --model xgb
  python scripts/evaluate_models.py --model lgbm
  python scripts/evaluate_models.py --model ensemble
  python scripts/evaluate_models.py --model all
  python scripts/evaluate_models.py --interactive
  python scripts/evaluate_models.py --model rf --threshold 0.50
  python scripts/evaluate_models.py --mode benchmark
"""

import os
import sys
import json
import time
import argparse

# Ensure UTF-8 output on Windows consoles
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from feature_engineering_v2 import engineer_features_v2, get_churn_v2_feature_names
from stacking_model_v2 import ChurnV2StackingEnsemble

# Default paths
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'test_original.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
BACKEND_MODELS_DIR = os.path.join(ROOT_DIR, 'backend', 'models')
BENCHMARK_JSON_PATH = os.path.join(MODELS_DIR, 'model_comparison_results.json')

# Model Configurations matching UI
MODELS_CATALOG = {
    'rf': {
        'id': 'rf',
        'aliases': ['rf', 'random_forest', 'randomforest', '1'],
        'name': 'RANDOM FOREST',
        'display_name': 'Random Forest',
        'file': 'churn_v2_rf.pkl',
        'default_threshold': 0.38,
        'color_code': '\033[94m',      # Blue
        'bg_code': '\033[44m',
        'is_deployed': False,
        'description': 'Balanced Bagging Trees with max_depth=12'
    },
    'xgb': {
        'id': 'xgb',
        'aliases': ['xgb', 'xgboost', '2'],
        'name': 'XGBOOST',
        'display_name': 'XGBoost',
        'file': 'churn_v2_xgb.pkl',
        'default_threshold': 0.29,
        'color_code': '\033[93m',      # Yellow / Orange
        'bg_code': '\033[43m',
        'is_deployed': False,
        'description': 'Gradient Boosting with L1/L2 regularization & tree subsampling'
    },
    'lgbm': {
        'id': 'lgbm',
        'aliases': ['lgbm', 'lightgbm', '3'],
        'name': 'LIGHTGBM',
        'display_name': 'LightGBM',
        'file': 'churn_v2_lgbm.pkl',
        'default_threshold': 0.32,
        'color_code': '\033[92m',      # Green
        'bg_code': '\033[42m',
        'is_deployed': False,
        'description': 'Leaf-wise Gradient Boosting with depth & child constraints'
    },
    'ensemble': {
        'id': 'ensemble',
        'aliases': ['ensemble', 'stacking', 'stack', 'stacked', '4'],
        'name': 'STACKING ENSEMBLE',
        'display_name': 'Stacking Ensemble',
        'file': 'churn_v2_stacking.pkl',
        'default_threshold': 0.26,
        'color_code': '\033[95m',      # Purple / Magenta
        'bg_code': '\033[45m',
        'is_deployed': True,
        'description': 'Stacking (RF + XGB + LGBM) via Logistic Regression meta-learner'
    }
}

# ANSI Color formatting
RESET = '\033[0m'
BOLD = '\033[1m'
DIM = '\033[2m'
UNDERLINE = '\033[4m'
CYAN = '\033[96m'
WHITE = '\033[97m'
GRAY = '\033[90m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'


def get_model_file(filename: str) -> str:
    """Finds model artifact in models/ or backend/models/."""
    p1 = os.path.join(MODELS_DIR, filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(BACKEND_MODELS_DIR, filename)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Model artifact not found: {filename} in {MODELS_DIR} or {BACKEND_MODELS_DIR}")


def load_test_dataset(data_path: str = DATA_PATH):
    """Loads and preprocesses the test dataset using churn_v2 pipeline."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df_test_raw = pd.read_csv(data_path)
    if 'target' not in df_test_raw.columns:
        if 'Churn' in df_test_raw.columns:
            df_test_raw['target'] = df_test_raw['Churn'].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'true'] else 0)
        else:
            raise ValueError(f"Ground truth label column ('target' or 'Churn') not found in {data_path}")

    y_test = df_test_raw['target'].values
    scale_cols, feature_cols = get_churn_v2_feature_names()
    df_test_fe = engineer_features_v2(df_test_raw)
    X_test_raw = df_test_fe[feature_cols].copy().astype(float)

    scaler_path = get_model_file('churn_v2_scaler.pkl')
    scaler = joblib.load(scaler_path)

    X_test = X_test_raw.copy()
    X_test[scale_cols] = scaler.transform(X_test_raw[scale_cols])

    return X_test, y_test, len(df_test_raw)


def load_model(model_key: str):
    """Loads specified trained model artifact."""
    cfg = MODELS_CATALOG[model_key]
    model_path = get_model_file(cfg['file'])
    model = joblib.load(model_path)
    return model


def evaluate_model_live(model, X_test, y_test, threshold: float) -> Dict[str, Any]:
    """Runs live inference and calculates all evaluation metrics."""
    t0 = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    latency = time.time() - t0

    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, zero_division=0)
    prec = precision_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        'accuracy': float(acc),
        'recall': float(rec),
        'precision': float(prec),
        'f1': float(f1),
        'roc_auc': float(roc),
        'pr_auc': float(pr_auc),
        'brier': float(brier),
        'threshold': float(threshold),
        'confusion_matrix': cm.tolist(),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'latency_sec': latency,
        'test_records': len(y_test),
        'y_true': y_test,
        'y_pred': preds
    }


import re
from decimal import Decimal, ROUND_HALF_UP

ANSI_REGEX = re.compile(r'\x1b\[[0-9;]*m')

def visual_len(s: str) -> int:
    """Calculates visible character width by ignoring ANSI escape codes."""
    return len(ANSI_REGEX.sub('', s))

def fmt_pct(val: float, decimals: int = 1) -> str:
    """Formats float to percentage matching JavaScript .toFixed(1) rounding."""
    try:
        d = Decimal(str(val * 100))
        rounded = d.quantize(Decimal('1.' + '0' * decimals), rounding=ROUND_HALF_UP)
        return f"{rounded}%"
    except Exception:
        return f"{val * 100:.{decimals}f}%"

def load_benchmark_metrics() -> Dict[str, Dict[str, Any]]:
    """Loads metrics directly from model_comparison_results.json."""
    if not os.path.exists(BENCHMARK_JSON_PATH):
        raise FileNotFoundError(f"Benchmark results JSON not found: {BENCHMARK_JSON_PATH}")
    with open(BENCHMARK_JSON_PATH, 'r') as f:
        data = json.load(f)

    results = {}
    for item in data.get('models', []):
        name_lower = item['name'].lower()
        if 'forest' in name_lower:
            key = 'rf'
        elif 'xgboost' in name_lower:
            key = 'xgb'
        elif 'lightgbm' in name_lower:
            key = 'lgbm'
        elif 'stack' in name_lower or 'ensemble' in name_lower:
            key = 'ensemble'
        else:
            continue
        results[key] = {
            'accuracy': item['accuracy'],
            'recall': item['recall'],
            'precision': item['precision'],
            'f1': item['f1'],
            'roc_auc': item['roc_auc'],
            'pr_auc': item.get('pr_auc', 0.0),
            'brier': item['brier'],
            'threshold': item['threshold'],
            'confusion_matrix': item.get('confusion_matrix', [[0, 0], [0, 0]]),
            'tn': item.get('tn', 0),
            'fp': item.get('fp', 0),
            'fn': item.get('fn', 0),
            'tp': item.get('tp', 0),
            'test_records': data.get('test_records', 1600),
            'latency_sec': None
        }
    return results


def print_model_card(model_key: str, metrics: Dict[str, Any], use_color: bool = True):
    """Prints a styled terminal card that visually mirrors the UI card."""
    cfg = MODELS_CATALOG[model_key]
    col = cfg['color_code'] if use_color else ''
    rst = RESET if use_color else ''
    bld = BOLD if use_color else ''
    dim = DIM if use_color else ''
    c_cyan = CYAN if use_color else ''

    deployed_badge = f"  {c_cyan}[Deployed]{rst}" if cfg['is_deployed'] else ""
    title = f"{bld}{col}{cfg['name']}{rst}{deployed_badge}"

    # Percentage values formatted matching UI
    rec_val = fmt_pct(metrics['recall'], 1)
    f1_val = fmt_pct(metrics['f1'], 1)
    acc_val = fmt_pct(metrics['accuracy'], 1)
    auc_val = fmt_pct(metrics['roc_auc'], 1)
    brier_val = f"{metrics['brier']:.4f}"
    prec_val = fmt_pct(metrics['precision'], 1)
    thresh_val = f"{metrics['threshold']:.2f}"
    records = metrics.get('test_records', 1600)

    width = 56
    border_col = col if use_color else ''

    def print_line(content: str):
        v_len = visual_len(content)
        pad = max(width - 2 - v_len, 0)
        print(f"{border_col}│{rst}{content}{' ' * pad}{border_col}│{rst}")

    print()
    print(f"{border_col}┌{'─' * (width - 2)}┐{rst}")
    
    # Header Line
    print_line(f"  {title}")
    print_line("")

    # Giant Primary Metric: Recall
    rec_display = f"{bld}{col}{rec_val}{rst}"
    raw_rec_len = visual_len(rec_display)
    l_pad = (width - 2 - raw_rec_len) // 2
    print_line(f"{' ' * l_pad}{rec_display}")

    label = "Recall"
    label_pad = (width - 2 - len(label)) // 2
    print_line(f"{' ' * label_pad}{dim}{label}{rst}")
    print_line("")

    # Row 1: F1 & ACCURACY
    row1_labels = f"        {dim}F1{rst}                    {dim}ACCURACY{rst}"
    row1_vals = f"      {bld}{f1_val:^6}{rst}                 {bld}{acc_val:^8}{rst}"
    print_line(row1_labels)
    print_line(row1_vals)

    # Row 2: AUC & BRIER
    row2_labels = f"       {dim}AUC{rst}                     {dim}BRIER{rst}"
    row2_vals = f"      {bld}{auc_val:^6}{rst}                 {bld}{brier_val:^8}{rst}"
    print_line(row2_labels)
    print_line(row2_vals)
    print_line("")

    # Footer Info
    footer_text = f"Precision: {bld}{prec_val}{rst}  ·  Thresh: {bld}{thresh_val}{rst}  ·  Test: {records}"
    f_pad = (width - 2 - visual_len(footer_text)) // 2
    print_line(f"{' ' * f_pad}{footer_text}")
    print(f"{border_col}└{'─' * (width - 2)}┘{rst}")



def print_detailed_breakdown(model_key: str, metrics: Dict[str, Any], use_color: bool = True):
    """Prints detailed confusion matrix, sample counts, and classification report."""
    cfg = MODELS_CATALOG[model_key]
    col = cfg['color_code'] if use_color else ''
    rst = RESET if use_color else ''
    bld = BOLD if use_color else ''
    dim = DIM if use_color else ''
    c_green = GREEN if use_color else ''
    c_red = RED if use_color else ''

    print(f"\n{bld}{col}--- {cfg['display_name']} Detailed Breakdown ---{rst}")
    print(f"{dim}Description: {cfg['description']}{rst}")
    if metrics.get('latency_sec') is not None:
        print(f"Inference Latency: {metrics['latency_sec']*1000:.1f} ms on {metrics.get('test_records', 1600)} samples")

    tn = metrics.get('tn', 0)
    fp = metrics.get('fp', 0)
    fn = metrics.get('fn', 0)
    tp = metrics.get('tp', 0)
    total = tn + fp + fn + tp if (tn + fp + fn + tp) > 0 else metrics.get('test_records', 1600)

    print("\nConfusion Matrix:")
    print(f"  ┌────────────────────────┬────────────────────────┐")
    print(f"  │  True Negatives (TN)   │  False Positives (FP)  │")
    print(f"  │  {c_green}{tn:>10} ({tn/total*100:5.1f}%){rst}   │  {c_red}{fp:>10} ({fp/total*100:5.1f}%){rst}   │")
    print(f"  ├────────────────────────┼────────────────────────┤")
    print(f"  │  False Negatives (FN)  │  True Positives (TP)   │")
    print(f"  │  {c_red}{fn:>10} ({fn/total*100:5.1f}%){rst}   │  {c_green}{tp:>10} ({tp/total*100:5.1f}%){rst}   │")
    print(f"  └────────────────────────┴────────────────────────┘")

    if 'y_true' in metrics and 'y_pred' in metrics:
        print("\nClassification Report:")
        rep = classification_report(
            metrics['y_true'], metrics['y_pred'],
            target_names=['No Churn (0)', 'Churn (1)'],
            digits=4
        )
        print(rep)


def print_comparison_table(all_metrics: Dict[str, Dict[str, Any]], use_color: bool = True):
    """Prints a comparison table of all models matching dashboard metrics."""
    rst = RESET if use_color else ''
    bld = BOLD if use_color else ''
    dim = DIM if use_color else ''
    c_green = GREEN if use_color else ''

    print(f"\n{bld}{'=' * 84}{rst}")
    print(f"{bld}  MODEL PERFORMANCE COMPARISON (Untouched Held-Out Test Set: 1,600 Records){rst}")
    print(f"{bld}{'=' * 84}{rst}")

    header = (
        f"{'Model Name':<20} "
        f"{'Recall':>9} "
        f"{'F1 Score':>10} "
        f"{'Accuracy':>10} "
        f"{'ROC-AUC':>9} "
        f"{'Brier':>8} "
        f"{'Precision':>10} "
        f"{'Threshold':>10}"
    )
    print(f"{bld}{header}{rst}")
    print(f"{'─' * 84}")

    # Identify best values
    recalls = {k: m['recall'] for k, m in all_metrics.items()}
    f1s = {k: m['f1'] for k, m in all_metrics.items()}
    accs = {k: m['accuracy'] for k, m in all_metrics.items()}
    aucs = {k: m['roc_auc'] for k, m in all_metrics.items()}
    briers = {k: m['brier'] for k, m in all_metrics.items()}

    best_rec = max(recalls.values())
    best_f1 = max(f1s.values())
    best_acc = max(accs.values())
    best_auc = max(aucs.values())
    best_brier = min(briers.values())

    for k in ['rf', 'xgb', 'lgbm', 'ensemble']:
        if k not in all_metrics:
            continue
        m = all_metrics[k]
        cfg = MODELS_CATALOG[k]
        col = cfg['color_code'] if use_color else ''

        name_str = f"{col}{bld}{cfg['display_name']}{rst}"
        if cfg['is_deployed']:
            name_str += f" {DIM}*{rst}"

        # Formatted percentages
        fmt_rec = fmt_pct(m['recall'], 1)
        fmt_f1 = fmt_pct(m['f1'], 1)
        fmt_acc = fmt_pct(m['accuracy'], 1)
        fmt_auc = fmt_pct(m['roc_auc'], 1)
        fmt_brier = f"{m['brier']:.4f}"
        fmt_prec = fmt_pct(m['precision'], 1)
        fmt_thresh = f"{m['threshold']:.2f}"

        # Highlighting best
        disp_rec = f"{c_green}{bld}{fmt_rec}{rst}" if (m['recall'] == best_rec and use_color) else fmt_rec
        disp_f1 = f"{c_green}{bld}{fmt_f1}{rst}" if (m['f1'] == best_f1 and use_color) else fmt_f1
        disp_acc = f"{c_green}{bld}{fmt_acc}{rst}" if (m['accuracy'] == best_acc and use_color) else fmt_acc
        disp_auc = f"{c_green}{bld}{fmt_auc}{rst}" if (m['roc_auc'] == best_auc and use_color) else fmt_auc
        disp_brier = f"{c_green}{bld}{fmt_brier}{rst}" if (m['brier'] == best_brier and use_color) else fmt_brier

        col_name = name_str + ' ' * max(20 - visual_len(name_str), 0)
        col_rec = ' ' * max(9 - visual_len(disp_rec), 0) + disp_rec
        col_f1 = ' ' * max(10 - visual_len(disp_f1), 0) + disp_f1
        col_acc = ' ' * max(10 - visual_len(disp_acc), 0) + disp_acc
        col_auc = ' ' * max(9 - visual_len(disp_auc), 0) + disp_auc
        col_brier = ' ' * max(8 - visual_len(disp_brier), 0) + disp_brier
        col_prec = ' ' * max(10 - visual_len(fmt_prec), 0) + fmt_prec
        col_thresh = ' ' * max(10 - visual_len(fmt_thresh), 0) + fmt_thresh

        print(f"{col_name} {col_rec} {col_f1} {col_acc} {col_auc} {col_brier} {col_prec} {col_thresh}")

    print(f"{'─' * 84}")
    print(f"{dim}* = Production Deployed Model  |  Green/Bold = Best in Metric Category{rst}\n")


def resolve_model_key(input_str: str) -> Optional[str]:
    """Matches user input to a model key ('rf', 'xgb', 'lgbm', 'ensemble', or 'all')."""
    s = input_str.strip().lower()
    if s in ['all', '0', 'a']:
        return 'all'
    for k, cfg in MODELS_CATALOG.items():
        if s == k or s in cfg['aliases']:
            return k
    return None


def interactive_menu() -> str:
    """Displays an interactive selection prompt."""
    print(f"\n{BOLD}Select Model to Evaluate:{RESET}")
    print("  [1] Random Forest      (RF)")
    print("  [2] XGBoost            (XGB)")
    print("  [3] LightGBM           (LGBM)")
    print("  [4] Stacking Ensemble  (Deployed)")
    print("  [5] All Models         (Comparison View)")
    print("  [q] Quit")
    choice = input(f"\n{CYAN}Enter selection [1-5 or q]: {RESET}").strip()
    if choice.lower() in ['q', 'exit', 'quit']:
        sys.exit(0)
    mapping = {
        '1': 'rf',
        '2': 'xgb',
        '3': 'lgbm',
        '4': 'ensemble',
        '5': 'all'
    }
    return mapping.get(choice, resolve_model_key(choice) or 'all')


def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn — Model Terminal Metrics & Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_models.py --model rf
  python scripts/evaluate_models.py --model xgb
  python scripts/evaluate_models.py --model lgbm
  python scripts/evaluate_models.py --model ensemble
  python scripts/evaluate_models.py --model all
  python scripts/evaluate_models.py --threshold 0.35 --details
  python scripts/evaluate_models.py --mode benchmark
        """
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help="Model to evaluate: 'rf', 'xgb', 'lgbm', 'ensemble' (or 'stacking'), 'all'"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'benchmark'],
        default='live',
        help="'live' = run inference on test dataset (default); 'benchmark' = display cached JSON metrics"
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=None,
        help="Override decision threshold (e.g. 0.50). If omitted, uses model's calibrated optimal threshold."
    )
    parser.add_argument(
        '-d', '--data',
        type=str,
        default=DATA_PATH,
        help="Path to evaluation test CSV (default: data/test_original.csv)"
    )
    parser.add_argument(
        '--details',
        action='store_true',
        help="Show confusion matrix and classification report breakdown"
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help="Disable ANSI color output in terminal"
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help="Prompt interactively for model selection"
    )

    args = parser.parse_args()
    use_color = not args.no_color and sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else not args.no_color

    # Determine target model
    if args.interactive:
        selected_model = interactive_menu()
    elif args.model is not None:
        selected_model = resolve_model_key(args.model)
        if selected_model is None:
            print(f"{RED}Error: Unknown model '{args.model}'. Choose from: rf, xgb, lgbm, ensemble, all.{RESET}")
            sys.exit(1)
    else:
        # Default when no args given: show all models
        selected_model = 'all'

    # Execute evaluation
    if args.mode == 'benchmark':
        benchmarks = load_benchmark_metrics()
        if selected_model == 'all':
            for k in ['rf', 'xgb', 'lgbm', 'ensemble']:
                print_model_card(k, benchmarks[k], use_color=use_color)
                if args.details:
                    print_detailed_breakdown(k, benchmarks[k], use_color=use_color)
            print_comparison_table(benchmarks, use_color=use_color)
        else:
            m = benchmarks[selected_model]
            print_model_card(selected_model, m, use_color=use_color)
            print_detailed_breakdown(selected_model, m, use_color=use_color)
    else:
        # Live evaluation mode
        print(f"{DIM}Loading test dataset from: {args.data}...{RESET}")
        try:
            X_test, y_test, num_records = load_test_dataset(args.data)
        except Exception as e:
            print(f"{RED}Error loading dataset: {e}{RESET}")
            sys.exit(1)

        target_keys = ['rf', 'xgb', 'lgbm', 'ensemble'] if selected_model == 'all' else [selected_model]
        live_results = {}

        for k in target_keys:
            cfg = MODELS_CATALOG[k]
            thresh = args.threshold if args.threshold is not None else cfg['default_threshold']
            try:
                model = load_model(k)
                metrics = evaluate_model_live(model, X_test, y_test, threshold=thresh)
                live_results[k] = metrics
            except Exception as e:
                print(f"{RED}Error evaluating model '{cfg['name']}': {e}{RESET}")
                continue

        if selected_model == 'all':
            for k in ['rf', 'xgb', 'lgbm', 'ensemble']:
                if k in live_results:
                    print_model_card(k, live_results[k], use_color=use_color)
                    if args.details:
                        print_detailed_breakdown(k, live_results[k], use_color=use_color)
            print_comparison_table(live_results, use_color=use_color)
        else:
            if selected_model in live_results:
                m = live_results[selected_model]
                print_model_card(selected_model, m, use_color=use_color)
                print_detailed_breakdown(selected_model, m, use_color=use_color)


if __name__ == '__main__':
    main()

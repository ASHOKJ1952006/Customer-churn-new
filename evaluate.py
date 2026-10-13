#!/usr/bin/env python3
"""
Root entrypoint for evaluate_models.py
Allows running:
  python evaluate.py --model rf
  python evaluate.py --model xgb
  python evaluate.py --model lgbm
  python evaluate.py --model ensemble
  python evaluate.py --model all
"""
import os
import sys

# Ensure UTF-8 output on Windows consoles
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Add scripts directory to path
scripts_path = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from evaluate_models import main

if __name__ == '__main__':
    main()

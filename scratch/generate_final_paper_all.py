"""
generate_final_paper_all.py — Generates the final conference research paper across
LaTeX (.tex), HTML, PDF, and DOCX formats.
Follows all IEEE conference conventions, formatting rules, and strict primary truth metrics.
"""

import os
import sys
import subprocess
import zipfile
import base64
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGURES_DIR = os.path.join(ROOT_DIR, 'paper_figures')

def get_base64_image(filename):
    p = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""

print("Figure loader ready.")

# -*- coding: utf-8 -*-
"""
generate_all_paper_artifacts.py — Full production generator for the IEEE conference paper:
  - Explainable_Ensemble_Learning_Telecom_Churn_Final.tex
  - Explainable_Ensemble_Learning_Telecom_Churn_Final.html
  - Explainable_Ensemble_Learning_Telecom_Churn_Final.pdf
  - Explainable_Ensemble_Learning_Telecom_Churn_Final.docx

Strictly aligns all figures and tables into single-column IEEE layout with zero overlapping,
matching the uploaded ICERCS reference paper format.
"""

import os
import sys
import subprocess
import base64
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGURES_DIR = os.path.join(ROOT_DIR, 'paper_figures')

# Import generator functions from build_perfect_ieee_paper
from build_perfect_ieee_paper import generate_html, convert_html_to_pdf, generate_docx, generate_latex

if __name__ == '__main__':
    print("Generating IEEE paper artifacts with zero overlapping and perfected photo alignment...")
    generate_html()
    convert_html_to_pdf()
    generate_docx()
    generate_latex()
    print("All paper files generated successfully.")

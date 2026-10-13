"""
gen_docx.py — Generate DOCX research paper using python-docx.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGS = os.path.join(ROOT, 'paper_figures')
DOCX_PATH = os.path.join(ROOT, 'Explainable_Ensemble_Learning_Telecom_Churn_Final.docx')

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

doc = Document()

# -----------------------------------------------------------------------
# DOCUMENT SETTINGS
# -----------------------------------------------------------------------
section = doc.sections[0]
section.page_width  = Cm(21.59)
section.page_height = Cm(27.94)
section.left_margin  = Cm(2.5)
section.right_margin = Cm(2.5)
section.top_margin   = Cm(2.5)
section.bottom_margin = Cm(2.5)

# Default font
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(10)

# -----------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------
def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
    return p


def add_para(doc, text, bold=False, italic=False, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return p


def add_mixed_para(doc, parts, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    """parts = list of (text, bold, italic, size)"""
    p = doc.add_paragraph()
    p.alignment = align
    for text, bold, italic, size in parts:
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(size if size else 10)
        run.bold = bold
        run.italic = italic
    return p


def add_figure(doc, filename, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fig_path = os.path.join(FIGS, filename)
    if os.path.exists(fig_path):
        run = p.add_run()
        run.add_picture(fig_path, width=Inches(5.5))
    else:
        p.add_run(f'[Figure: {filename} not found]')
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9)
        run.italic = True
    return p


def set_table_style(table):
    table.style = 'Table Grid'
    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(9)


def shade_row(row, hex_color='2C3E50'):
    for cell in row.cells:
        tc_pr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), hex_color)
        tc_pr.append(shd)
        for run in cell.paragraphs[0].runs:
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.bold = True


# -----------------------------------------------------------------------
# TITLE
# -----------------------------------------------------------------------
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run(
    'Explainable Stacking Ensemble Learning and Decision Support Framework\n'
    'for Telecom Customer Churn Prediction'
)
run.font.name = 'Times New Roman'
run.font.size = Pt(16)
run.bold = True

doc.add_paragraph()  # spacer

# -----------------------------------------------------------------------
# AUTHORS
# -----------------------------------------------------------------------
authors_table = doc.add_table(rows=1, cols=3)
authors_table.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (name, dept, inst, city, email) in enumerate([
    ('Abinash J.', 'Dept. of Computer Science & Engineering',
     'Kongu Engineering College', 'Erode, Tamil Nadu, India', 'abinash.cse@kongu.edu'),
    ('Chandru Vignesh C.', 'Dept. of Information Security',
     'Vellore Institute of Technology', 'Vellore, Tamil Nadu, India', 'chandruvignesh@vit.ac.in'),
    ('Deepak Raja V.', 'Dept. of Computer Technology',
     'Kongu Engineering College', 'Erode, Tamil Nadu, India', 'deepakraja.tech@gmail.com'),
]):
    cell = authors_table.rows[0].cells[i]
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(name + '\n')
    r.font.name = 'Times New Roman'
    r.font.size = Pt(10)
    r.bold = True
    for detail in [dept, inst, city, email]:
        r2 = p.add_run(detail + '\n')
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(9)
        r2.italic = True

doc.add_paragraph()

# -----------------------------------------------------------------------
# ABSTRACT
# -----------------------------------------------------------------------
abs_p = doc.add_paragraph()
abs_p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
r1 = abs_p.add_run('Abstract\u2014')
r1.font.name = 'Times New Roman'
r1.font.size = Pt(9)
r1.bold = True
r1.italic = True

abs_text = (
    'Customer defection is one of the most financially damaging operational challenges '
    'in the telecommunications industry, where replacing a departing subscriber costs '
    'five to seven times more than retaining an existing one. Conventional churn '
    'pipelines often apply a fixed \u03c4\u202f=\u202f0.50 classification boundary that '
    'optimises for accuracy but systematically underdetects vulnerable subscribers. '
    'This paper presents an end-to-end explainable ensemble learning and retention '
    'decision-support system evaluated on balanced_telco_churn.csv (N\u202f=\u202f8,000 records; '
    'exactly 50\u202f% churners). Three tree-based base classifiers\u2014Random Forest, '
    'Extreme Gradient Boosting (XGBoost), and Light Gradient Boosting Machine '
    '(LightGBM)\u2014are trained independently on SMOTE-augmented data and fused '
    'through an out-of-fold Stacking Ensemble governed by a Logistic Regression '
    'meta-learner. A domain-informed feature engineering pipeline expands 21 raw '
    'attributes into 83 model-ready features. Per-model decision thresholds are '
    'optimised: XGBoost achieves 97.6\u202f% recall, 94.9\u202f% accuracy, and 99.0\u202f% '
    'ROC-AUC at \u03c4\u202f=\u202f0.29; LightGBM attains 96.9\u202f% recall, 95.0\u202f% accuracy, '
    'and 98.9\u202f% ROC-AUC at \u03c4\u202f=\u202f0.32; Random Forest records 96.3\u202f% recall, '
    '95.4\u202f% accuracy, and 98.7\u202f% ROC-AUC at \u03c4\u202f=\u202f0.38; and the Stacking Ensemble '
    'delivers 96.3\u202f% recall, 95.4\u202f% accuracy, 95.5\u202f% F1-score, 94.7\u202f% precision, '
    '98.9\u202f% ROC-AUC, and a Brier score of 0.0362 at \u03c4\u202f=\u202f0.26. A dual SHAP and LIME '
    'explainability pipeline feeds a state-aware retention recommendation engine. '
    'The complete framework is deployed as a production web application (FastAPI, React/Vite, SQLite).'
)
r2 = abs_p.add_run(abs_text)
r2.font.name = 'Times New Roman'
r2.font.size = Pt(9)

kw = doc.add_paragraph()
kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
kwr = kw.add_run('Index Terms\u2014')
kwr.bold = True
kwr.italic = True
kwr.font.size = Pt(9)
kwr.font.name = 'Times New Roman'
r3 = kw.add_run('Customer Churn Prediction, Telecom Analytics, Stacking Ensemble Learning, '
                 'Decision Threshold Optimization, Explainable AI, SHAP, LIME, Retention Recommendations')
r3.font.size = Pt(9)
r3.font.name = 'Times New Roman'

doc.add_paragraph()

# -----------------------------------------------------------------------
# SECTION I: INTRODUCTION
# -----------------------------------------------------------------------
add_heading(doc, 'I. INTRODUCTION', level=1)
add_para(doc,
    'The rapid expansion of mobile broadband, fibre-optic internet, and bundled digital '
    'services has transformed telecommunications into one of the most intensely competitive '
    'sectors globally. Number-portability legislation, aggressive promotional pricing, and '
    'minimal switching barriers give subscribers unprecedented freedom to migrate between '
    'providers with limited friction. Operators routinely face annual attrition rates '
    'exceeding 20\u201330\u202f%, with monthly churn rates between 1.5\u202f% and 3.0\u202f% in mature markets [1].')
add_para(doc,
    'Acquiring a replacement subscriber costs five to seven times more than retaining an '
    'existing one, accounting for sales commissions, device subsidies, onboarding support, '
    'and margin loss during re-acquisition [2]. High-value subscribers with premium plans '
    'and long tenure represent a disproportionate share of profitability, creating cascading '
    'effects on ARPU and investor valuation when lost.')
add_para(doc,
    'Manual identification through rule-based heuristics is inadequate at scale. Subscriber '
    'behaviour emerges from interactions of contract type, billing level, payment modality, '
    'service portfolio, and tenure that static rules cannot reliably parse [3]. Machine '
    'learning addresses this by modelling multi-feature interactions without explicit rule '
    'specification. Tree-based ensembles\u2014Random Forest, XGBoost, and LightGBM\u2014have '
    'delivered consistently strong performance on tabular CRM data [4].')
add_para(doc,
    'Despite these advances, most deployed systems apply the default 0.50 probability '
    'threshold, which is sub-optimal when false negatives carry substantially higher cost [5]. '
    'Furthermore, even accurate models remain actionability-blind: they produce a probability '
    'label but no guidance on why a customer is at risk [6]. This work addresses these gaps '
    'through a layered approach combining feature engineering, stacking ensembles, threshold '
    'optimisation, dual explainability, and state-aware retention recommendations.')

# -----------------------------------------------------------------------
# SECTION II: LITERATURE REVIEW
# -----------------------------------------------------------------------
add_heading(doc, 'II. LITERATURE REVIEW', level=1)
add_para(doc,
    'Churn prediction has attracted sustained research across machine learning and '
    'telecommunications management. Early studies relied on logistic regression and decision '
    'trees [7], establishing the relevance of contract type, tenure, and monthly '
    'charges\u2014features that continue to dominate modern importance rankings. Hadden et al. '
    '[8] showed that bagging multiple decision trees significantly stabilised accuracy across '
    'heterogeneous customer segments. Verbeke et al. [9] confirmed ensemble superiority, '
    'particularly under class imbalance.')
add_para(doc,
    'XGBoost [10] and LightGBM [11] introduced more powerful optimisation frameworks through '
    'additive tree structures with regularisation. Farquad et al. [12] demonstrated gradient-boosted '
    'trees capture non-obvious billing and contract interactions. Chawla et al. [13] introduced SMOTE '
    'for class rebalancing; fold-isolated SMOTE prevents target leakage [14]. SHAP [15] and LIME [16] '
    'provide global and local feature interpretability. Stacking generalisation [18] trains a '
    'meta-learner on out-of-fold predictions, capturing complementary strengths. ROC analysis [20] '
    'provides the principled basis for threshold selection under asymmetric misclassification costs.')
add_para(doc,
    'The research gap is the integration of all these individually validated components into a '
    'coherent end-to-end deployed production system that connects model outputs to business action.')

# -----------------------------------------------------------------------
# SECTION III: PROPOSED FRAMEWORK
# -----------------------------------------------------------------------
add_heading(doc, 'III. PROPOSED FRAMEWORK OVERVIEW', level=1)
add_para(doc,
    'The developed framework processes subscriber records through a seven-stage pipeline from '
    'raw data ingestion to actionable retention guidance (Fig. 1):')
for item in [
    '1. Data Ingestion: Subscriber attributes received via form or CSV batch upload.',
    '2. Feature Engineering: 83-feature vector constructed from six core inputs.',
    '3. Base Model Prediction: RF, XGBoost, and LightGBM estimate churn probability independently.',
    '4. Ensemble Aggregation: Logistic Regression meta-learner combines three base outputs.',
    '5. Threshold Decision: Each model applies its optimised \u03c4; probabilities \u2265 \u03c4 trigger churn flags.',
    '6. Explainability: SHAP TreeExplainer and LIME compute per-feature attributions.',
    '7. Recommendation: Retention engine maps attributions to prioritised, state-filtered interventions.',
]:
    p = doc.add_paragraph(item, style='List Number')
    p.paragraph_format.left_indent = Cm(1)

add_figure(doc, 'fig1_system_architecture.png',
    'Fig. 1. End-to-end system architecture: customer data \u2192 feature engineering \u2192 '
    'ensemble prediction \u2192 SHAP/LIME \u2192 retention recommendations \u2192 web dashboard.')

# -----------------------------------------------------------------------
# SECTION IV: DATASET AND PREPROCESSING
# -----------------------------------------------------------------------
add_heading(doc, 'IV. DATASET AND PREPROCESSING', level=1)
add_heading(doc, 'A. Dataset Description', level=2)
add_para(doc,
    'All experiments use balanced_telco_churn.csv, a domain-consistent benchmark derived '
    'from the IBM Watson Telco Customer Churn corpus. The balanced variant contains '
    'N\u202f=\u202f8,000 subscriber records with exactly 4,000 churned and 4,000 loyal accounts '
    '(50/50 class distribution). Table I summarises the dataset characteristics.')

# Table I
doc.add_paragraph('TABLE I \u2014 Dataset Characteristics').alignment = WD_ALIGN_PARAGRAPH.CENTER
t1 = doc.add_table(rows=11, cols=2)
set_table_style(t1)
shade_row(t1.rows[0])
headers = ['Property', 'Value']
rows_data = [
    ('Total records', '8,000'),
    ('Churners (Churn = Yes)', '4,000 (50.0%)'),
    ('Loyal (Churn = No)', '4,000 (50.0%)'),
    ('Raw columns', '22 (incl. customerID)'),
    ('Engineered features', '83'),
    ('Numerical features', '3 (tenure, MonthlyCharges, TotalCharges)'),
    ('Missing values', 'None'),
    ('Target variable', 'Churn (Yes=1, No=0)'),
    ('Train partition', '80% (N=6,400) stratified'),
    ('Test partition', '20% (N=1,600) held-out'),
]
for i, h in enumerate(headers):
    t1.rows[0].cells[i].paragraphs[0].runs[0].text if t1.rows[0].cells[i].paragraphs[0].runs else None
    c = t1.rows[0].cells[i]
    c.paragraphs[0].clear()
    r = c.paragraphs[0].add_run(h)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(9)
    r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

for i, (k, v) in enumerate(rows_data):
    row = t1.rows[i + 1]
    row.cells[0].paragraphs[0].clear()
    run1 = row.cells[0].paragraphs[0].add_run(k)
    run1.font.name = 'Times New Roman'
    run1.font.size = Pt(9)
    row.cells[1].paragraphs[0].clear()
    run2 = row.cells[1].paragraphs[0].add_run(v)
    run2.font.name = 'Times New Roman'
    run2.font.size = Pt(9)
doc.add_paragraph()

add_heading(doc, 'B. Primary Customer Input Features', level=2)
add_para(doc,
    'The web interface accepts six core attributes sufficient to construct the full '
    '83-feature vector. Table II describes each attribute\'s churn relevance.')

doc.add_paragraph('TABLE II \u2014 Primary Customer Input Features').alignment = WD_ALIGN_PARAGRAPH.CENTER
t2 = doc.add_table(rows=7, cols=3)
set_table_style(t2)
shade_row(t2.rows[0])
for i, h in enumerate(['Feature', 'Type', 'Churn Relevance']):
    c = t2.rows[0].cells[i]
    c.paragraphs[0].clear()
    r = c.paragraphs[0].add_run(h)
    r.font.name = 'Times New Roman'; r.font.size = Pt(9); r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

feat_rows = [
    ('Tenure', 'Integer', 'Shorter tenure = higher early-phase defection risk'),
    ('MonthlyCharges', 'Float', 'Higher billing correlates with dissatisfaction risk'),
    ('TotalCharges', 'Float', 'Cumulative revenue; encodes tenure\xd7charge history'),
    ('Contract', 'Categor.', 'Month-to-month plans carry substantially higher defection risk'),
    ('InternetService', 'Categor.', 'Fiber-optic subscribers show elevated churn in competitive markets'),
    ('PaymentMethod', 'Categor.', 'Electronic check payers are a consistently high-risk segment'),
]
for i, (f, t, r_txt) in enumerate(feat_rows):
    row = t2.rows[i + 1]
    for j, val in enumerate([f, t, r_txt]):
        row.cells[j].paragraphs[0].clear()
        rn = row.cells[j].paragraphs[0].add_run(val)
        rn.font.name = 'Times New Roman'
        rn.font.size = Pt(9)
doc.add_paragraph()

add_heading(doc, 'C. Preprocessing Pipeline', level=2)
add_para(doc,
    'All transformations are fitted exclusively on the training partition and applied '
    'identically to the test set (zero leakage). Steps: customerID removal; TotalCharges '
    'numeric coercion; Yes/No binary encoding; 83-feature engineering; one-hot encoding '
    'with drop-first; StandardScaler on continuous columns; stratified 80/20 split; '
    'fold-isolated SMOTE on training data only.')

# -----------------------------------------------------------------------
# SECTION V: FEATURE ENGINEERING
# -----------------------------------------------------------------------
add_heading(doc, 'V. FEATURE ENGINEERING', level=1)
add_para(doc,
    'The 21 informative raw attributes are expanded to 83 model-ready columns through '
    'domain-driven engineering, encoding non-linear relationships that tree-based models '
    'can exploit directly. Table III summarises the principal engineered feature groups.')

doc.add_paragraph('TABLE III \u2014 Engineered Feature Groups').alignment = WD_ALIGN_PARAGRAPH.CENTER
t3 = doc.add_table(rows=9, cols=3)
set_table_style(t3)
shade_row(t3.rows[0])
for i, h in enumerate(['Category', 'N', 'Representative Features']):
    c = t3.rows[0].cells[i]
    c.paragraphs[0].clear()
    r = c.paragraphs[0].add_run(h)
    r.font.name = 'Times New Roman'; r.font.size = Pt(9); r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

fe_rows = [
    ('Revenue / Billing', '7', 'AvgMonthlySpend, ChargeIncrease, ChargeIncreaseRatio'),
    ('Tenure Signals', '6', 'TenureSquared, TenureLog, IsNewCustomer, IsLongTermCustomer'),
    ('Contract Flags', '4', 'IsMonthToMonth, ContractLength, IsLongTermContract'),
    ('Payment Flags', '4', 'IsElectronicCheck, IsAutomaticPayment'),
    ('Service Counts', '2', 'NumServices, ServiceRiskCount'),
    ('Compound Risk', '5', 'HighRiskCombo, VeryHighRisk, MultipleRiskFactors'),
    ('One-Hot Encoded', '55', 'Contract, PaymentMethod, InternetService, service cols'),
    ('Total', '83', ''),
]
for i, (cat, n, feats) in enumerate(fe_rows):
    row = t3.rows[i + 1]
    for j, val in enumerate([cat, n, feats]):
        row.cells[j].paragraphs[0].clear()
        rn = row.cells[j].paragraphs[0].add_run(val)
        rn.font.name = 'Times New Roman'
        rn.font.size = Pt(9)
        if i == len(fe_rows) - 1:
            rn.bold = True
doc.add_paragraph()

add_para(doc,
    'AvgMonthlySpend = TotalCharges / (tenure + 1) estimates the historical average billing '
    'rate. ChargeIncrease flags recent billing escalation; ChargeIncreaseRatio expresses it '
    'relative to historical spend. TenureSquared and TenureLog model the non-linear lifecycle. '
    'HighRiskCombo flags month-to-month plus electronic check customers. VeryHighRisk extends '
    'this to require tenure \u2264 12 months, capturing the most vulnerable early-phase subscribers.')

# -----------------------------------------------------------------------
# SECTION VI: MODEL DEVELOPMENT
# -----------------------------------------------------------------------
add_heading(doc, 'VI. MODEL DEVELOPMENT', level=1)
add_heading(doc, 'A. Random Forest', level=2)
add_para(doc,
    'A Random Forest [22] builds an ensemble of T independent decision trees, each trained '
    'on a bootstrap sample with a random feature subset at each split node. Bootstrap aggregation '
    'suppresses individual-tree variance. Eq. (1): P_RF(x) = (1/T) \u03a3 h_t(x). Configuration: '
    '200 trees, max depth 8, min samples per leaf 3, Gini impurity, balanced class weights. '
    'At \u03c4 = 0.38: Recall 96.3%, Accuracy 95.4%, ROC-AUC 98.7%.')

add_heading(doc, 'B. XGBoost', level=2)
add_para(doc,
    'XGBoost [10] trains trees sequentially to correct residuals. Regularised objective: '
    'L^(k) = \u03a3 l(y_i, \u0177^(k-1) + h_k(x_i)) + \u03a9(h_k), where \u03a9(h_k) = \u03b3T + \u03bb||w||^2/2. '
    'Configuration: 150 trees, lr 0.04, max depth 3, \u03bb = 2.0, \u03b1 = 0.2. '
    'At \u03c4 = 0.29: Recall 97.6%, Accuracy 94.9%, ROC-AUC 99.0%, Brier 0.0347\u2014highest '
    'recall and ROC-AUC of all evaluated models.')

add_heading(doc, 'C. LightGBM', level=2)
add_para(doc,
    'LightGBM [11] employs leaf-wise growth, splitting the leaf with the highest loss gain. '
    'GOSS retains all high-gradient instances while sub-sampling low-gradient ones. Configuration: '
    '120 trees, lr 0.04, 24 leaves, max depth 3, L2 reg 2.5. At \u03c4 = 0.32: '
    'Recall 96.9%, Accuracy 95.0%, ROC-AUC 98.9%, Brier 0.0353.')

add_heading(doc, 'D. Stacking Ensemble', level=2)
add_para(doc,
    'The stacking architecture fuses three base classifiers through a Logistic Regression '
    'meta-learner trained on out-of-fold predictions from 5-fold stratified cross-validation. '
    'For each fold: (i) SMOTE applied to training fold only; (ii) each base model trains on '
    'the resampled fold; (iii) probability predictions collected on held-out validation fold. '
    'Meta-feature matrix M = [p_RF | p_XGB | p_LGBM] \u2208 R^(n\xd73). Meta-learner: '
    'P_Ens(x) = \u03c3(w0 + w1*p_RF + w2*p_XGB + w3*p_LGBM). At \u03c4 = 0.26: '
    'Recall 96.3%, F1 95.5%, Accuracy 95.4%, Precision 94.7%, ROC-AUC 98.9%, Brier 0.0362.')

add_figure(doc, 'fig2_ml_workflow.png',
    'Fig. 2. Stacking ensemble: base classifiers generate OOF probability vectors assembled '
    'into the meta-feature matrix; Logistic Regression meta-learner produces the final churn probability.')

# -----------------------------------------------------------------------
# SECTION VII: METRICS AND THRESHOLD
# -----------------------------------------------------------------------
add_heading(doc, 'VII. PERFORMANCE METRICS & THRESHOLD OPTIMIZATION', level=1)
add_heading(doc, 'A. Evaluation Metrics', level=2)
add_para(doc,
    'Let TP, TN, FP, FN denote true positives, true negatives, false positives, and false '
    'negatives. Key metrics: Accuracy = (TP+TN)/(TP+TN+FP+FN); Precision = TP/(TP+FP); '
    'Recall = TP/(TP+FN); F1 = 2\xb7Precision\xb7Recall/(Precision+Recall); '
    'Brier Score = (1/n)\u03a3(p_i - y_i)^2. ROC-AUC quantifies rank separation across '
    'all thresholds; Brier measures calibration quality.')

add_heading(doc, 'B. Decision Threshold Optimization', level=2)
add_para(doc,
    'The default \u03c4 = 0.50 is sub-optimal when false negatives carry substantially '
    'higher cost than false positives. Binary prediction: \u0177_i = 1 if p_i \u2265 \u03c4, else 0. '
    'Reducing \u03c4 below 0.50 increases recall at the cost of some precision. Table IV '
    'shows the optimised thresholds; all four models operate below the 0.50 default.')

doc.add_paragraph('TABLE IV \u2014 Optimised Decision Thresholds').alignment = WD_ALIGN_PARAGRAPH.CENTER
t4 = doc.add_table(rows=5, cols=3)
set_table_style(t4)
shade_row(t4.rows[0])
for i, h in enumerate(['Model', 'Threshold (\u03c4)', 'vs. Default 0.50']):
    c = t4.rows[0].cells[i]
    c.paragraphs[0].clear()
    r = c.paragraphs[0].add_run(h)
    r.font.name = 'Times New Roman'; r.font.size = Pt(9); r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

tau_rows = [
    ('Random Forest', '0.38', 'Lower'),
    ('XGBoost', '0.29', 'Lower'),
    ('LightGBM', '0.32', 'Lower'),
    ('Stacking Ensemble', '0.26', 'Lower'),
]
for i, (m, tau, rel) in enumerate(tau_rows):
    row = t4.rows[i + 1]
    for j, val in enumerate([m, tau, rel]):
        row.cells[j].paragraphs[0].clear()
        rn = row.cells[j].paragraphs[0].add_run(val)
        rn.font.name = 'Times New Roman'
        rn.font.size = Pt(9)
doc.add_paragraph()

# -----------------------------------------------------------------------
# SECTION VIII: EXPERIMENTAL RESULTS
# -----------------------------------------------------------------------
add_heading(doc, 'VIII. EXPERIMENTAL RESULTS', level=1)
add_para(doc,
    'All models are evaluated on the withheld 20\u202f% test partition (N\u202f=\u202f1,600) '
    'untouched during training, feature engineering fitting, SMOTE, and cross-validation. '
    'Table V presents the complete performance profile.')

doc.add_paragraph('TABLE V \u2014 Model Performance on Held-Out Test Set (N = 1,600)').alignment = WD_ALIGN_PARAGRAPH.CENTER
t5 = doc.add_table(rows=5, cols=7)
set_table_style(t5)
shade_row(t5.rows[0])
headers5 = ['Model', 'Recall (%)', 'F1 (%)', 'Acc. (%)', 'AUC (%)', 'Brier', 'Prec. (%)']
for i, h in enumerate(headers5):
    c = t5.rows[0].cells[i]
    c.paragraphs[0].clear()
    r = c.paragraphs[0].add_run(h)
    r.font.name = 'Times New Roman'; r.font.size = Pt(9); r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

results_rows = [
    ('Random Forest', '96.3', '95.4', '95.4', '98.7', '0.0369', '94.6'),
    ('XGBoost', '97.6', '95.0', '94.9', '99.0', '0.0347', '92.5'),
    ('LightGBM', '96.9', '95.1', '95.0', '98.9', '0.0353', '93.4'),
    ('Stacking Ens.', '96.3', '95.5', '95.4', '98.9', '0.0362', '94.7'),
]
best_col = {0: 'XGBoost', 1: 'Stacking Ens.', 2: 'LightGBM', 3: 'XGBoost', 4: 'XGBoost', 5: 'Stacking Ens.'}
for i, row_data in enumerate(results_rows):
    row = t5.rows[i + 1]
    for j, val in enumerate(row_data):
        row.cells[j].paragraphs[0].clear()
        rn = row.cells[j].paragraphs[0].add_run(val)
        rn.font.name = 'Times New Roman'
        rn.font.size = Pt(9)
        if j > 0 and best_col.get(j - 1) == row_data[0]:
            rn.bold = True
doc.add_paragraph()

add_figure(doc, 'fig5_model_performance_comparison.png',
    'Fig. 3. Model performance comparison across Recall, F1-Score, Accuracy, and ROC-AUC. '
    'All models exceed 94.9% accuracy and 96.3% recall after threshold optimisation.')

# -----------------------------------------------------------------------
# SECTION IX: RESULTS ANALYSIS
# -----------------------------------------------------------------------
add_heading(doc, 'IX. RESULTS ANALYSIS', level=1)
add_para(doc,
    'All four classifiers achieve ROC-AUC between 98.7% and 99.0%, confirming near-perfect '
    'rank separation between churners and non-churners.')
add_para(doc,
    'XGBoost records the highest recall (97.6%) and ROC-AUC (99.0%), correctly identifying '
    '97.6% of actual churners. The Brier score of 0.0347 is also the lowest. The trade-off '
    'is precision of 92.5%, reflecting the false-positive cost of \u03c4 = 0.29.')
add_para(doc,
    'LightGBM achieves 96.9% recall, the highest accuracy (95.0%), and Brier 0.0353. '
    'Its leaf-wise growth produces a favourable recall\u2013precision balance at \u03c4 = 0.32.')
add_para(doc,
    'Random Forest achieves 96.3% recall and 95.4% accuracy with the most conservative '
    'threshold (0.38). Its precision of 94.6% is the second highest.')
add_para(doc,
    'The Stacking Ensemble achieves the highest F1 (95.5%) and precision (94.7%). Operating '
    'at \u03c4 = 0.26, it matches Random Forest recall while delivering a more balanced '
    'precision\u2013recall profile. The Logistic Regression meta-learner successfully reduces the '
    'false-positive rate relative to individual gradient-boosted models.')

doc.add_paragraph('TABLE VI \u2014 Error Rate Summary (%)').alignment = WD_ALIGN_PARAGRAPH.CENTER
t6 = doc.add_table(rows=5, cols=5)
set_table_style(t6)
shade_row(t6.rows[0])
for i, h in enumerate(['Model', 'Acc. Err', 'Prec. Err', 'Rec. Err', 'F1 Err']):
    c = t6.rows[0].cells[i]
    c.paragraphs[0].clear()
    r = c.paragraphs[0].add_run(h)
    r.font.name = 'Times New Roman'; r.font.size = Pt(9); r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

err_rows = [
    ('Random Forest', '4.6', '5.4', '3.7', '4.6'),
    ('XGBoost', '5.1', '7.5', '2.4', '5.0'),
    ('LightGBM', '5.0', '6.6', '3.1', '4.9'),
    ('Stacking (Prop.)', '4.6', '5.3', '3.7', '4.5'),
]
for i, row_data in enumerate(err_rows):
    row = t6.rows[i + 1]
    for j, val in enumerate(row_data):
        row.cells[j].paragraphs[0].clear()
        rn = row.cells[j].paragraphs[0].add_run(val)
        rn.font.name = 'Times New Roman'
        rn.font.size = Pt(9)
doc.add_paragraph()

# -----------------------------------------------------------------------
# SECTION X: SYSTEM IMPLEMENTATION
# -----------------------------------------------------------------------
add_heading(doc, 'X. SYSTEM IMPLEMENTATION', level=1)
doc.add_paragraph('TABLE VII \u2014 Implementation Technology Stack').alignment = WD_ALIGN_PARAGRAPH.CENTER
t7 = doc.add_table(rows=11, cols=3)
set_table_style(t7)
shade_row(t7.rows[0])
for i, h in enumerate(['Layer', 'Technology', 'Role']):
    c = t7.rows[0].cells[i]
    c.paragraphs[0].clear()
    r = c.paragraphs[0].add_run(h)
    r.font.name = 'Times New Roman'; r.font.size = Pt(9); r.bold = True
    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

tech_rows = [
    ('Frontend', 'React (Vite)', 'UI, routing, visualisation'),
    ('Frontend', 'React Router', 'Multi-page navigation'),
    ('Backend', 'FastAPI', 'REST API, prediction serving'),
    ('ML Runtime', 'Python 3.11', 'Core computation'),
    ('Ensemble ML', 'scikit-learn', 'RF, Logistic Regression, preprocessing'),
    ('Ensemble ML', 'XGBoost 4.x', 'Gradient-boosted classifier'),
    ('Ensemble ML', 'LightGBM 4.7', 'Leaf-wise boosted classifier'),
    ('Explainability', 'SHAP 0.51', 'Global and local attributions'),
    ('Explainability', 'LIME 0.2', 'Local surrogate explanations'),
    ('Database', 'SQLite', 'Prediction history persistence'),
]
for i, row_data in enumerate(tech_rows):
    row = t7.rows[i + 1]
    for j, val in enumerate(row_data):
        row.cells[j].paragraphs[0].clear()
        rn = row.cells[j].paragraphs[0].add_run(val)
        rn.font.name = 'Times New Roman'
        rn.font.size = Pt(9)
doc.add_paragraph()
add_para(doc,
    'The web interface exposes five modules: Single Customer Prediction, Batch Upload, '
    'What-If Simulator, Dashboard, and Model Comparison.')

# -----------------------------------------------------------------------
# SECTION XI: EXPLAINABLE AI
# -----------------------------------------------------------------------
add_heading(doc, 'XI. EXPLAINABLE AI & RETENTION RECOMMENDATIONS', level=1)
add_heading(doc, 'A. SHAP Feature Attribution', level=2)
add_para(doc,
    'SHAP [15] satisfies: f(x^(i)) = \u03c6_0 + \u03a3 \u03c6_j^(i), where \u03c6_0 is the '
    'baseline mean prediction and \u03c6_j^(i) is the contribution of feature j. Positive values '
    'push toward churn; negative values push toward retention. The system uses SHAP '
    'TreeExplainer (exact for tree-based models).')

add_heading(doc, 'B. LIME Explanation Service', level=2)
add_para(doc,
    'LIME [16] approximates the ensemble with a sparse linear surrogate near each prediction: '
    '\u03be(x_0) = argmin_g L(f, g, \u03c0_{x_0}) + \u03a9(g). The surrogate\'s coefficients '
    'serve as local feature importance weights, complementing SHAP from an independent perspective.')

add_heading(doc, 'C. Retention Recommendation Engine', level=2)
add_para(doc,
    'The engine operates in four stages: (1) Risk Classification (Low/Medium/High/Critical); '
    '(2) Driver Extraction (SHAP+LIME mapped to business factor categories: Contract, Payment, '
    'Billing, Lifecycle, Service Experience); (3) State-Aware Filtering (prune inapplicable '
    'interventions); (4) Prioritised Ranking by composite score. Interventions: contract upgrade '
    'incentives, automatic payment migration, billing discount review, onboarding support, '
    'service bundles, technical support, and fibre quality investigation.')

add_figure(doc, 'fig7_xai_recommendation_workflow.png',
    'Fig. 4. Retention recommendation workflow: SHAP/LIME drivers mapped to business factor '
    'categories, state-filtered, scored, and returned as ranked intervention packages.')

add_heading(doc, 'D. What-If Simulator & Batch Prediction', level=2)
add_para(doc,
    'The What-If Simulator allows analysts to modify customer attributes and observe how '
    'the ensemble probability responds (without asserting causal effects). Batch Upload enables '
    'CSV-based multi-customer prediction with downloadable results including probability, '
    'risk level, and primary SHAP-identified driver.')

# -----------------------------------------------------------------------
# SECTION XII: DISCUSSION
# -----------------------------------------------------------------------
add_heading(doc, 'XII. DISCUSSION', level=1)
add_para(doc,
    'The consistently strong performance of all three tree-based classifiers reflects the '
    'suitability of gradient-boosted and bagged ensembles for tabular CRM data. The stacking '
    'ensemble\'s value lies in robustness: achieving the highest F1 and precision while '
    'matching other models on recall indicates the meta-learner identifies regions where '
    'each base classifier is most reliable. Decision threshold optimisation proved essential: '
    'operating all models at 0.50 would produce substantially lower recall. The dual SHAP '
    'and LIME layer enables retention agents to understand why a subscriber was flagged, '
    'enabling more targeted interventions than a probability label alone provides. Results '
    'apply to the balanced benchmark and require recalibration for real-world 15\u201330% prevalence.')

# -----------------------------------------------------------------------
# SECTION XIII: LIMITATIONS
# -----------------------------------------------------------------------
add_heading(doc, 'XIII. LIMITATIONS', level=1)
add_para(doc,
    'The 50/50 class distribution is artificial; production deployments would alter thresholds '
    'and false-positive rates. All evaluation uses a single dataset; external validation is needed. '
    'SHAP/LIME attributions describe model behaviour, not causal subscriber behaviour. '
    'Retention recommendations are heuristic and have not been validated in controlled field trials.')

# -----------------------------------------------------------------------
# SECTION XIV: CONCLUSION
# -----------------------------------------------------------------------
add_heading(doc, 'XIV. CONCLUSION', level=1)
add_para(doc,
    'This work developed an end-to-end explainable ensemble learning framework for telecom '
    'customer churn prediction. Feature engineering expanded 21 raw attributes to 83 model-ready '
    'features. Three tree classifiers were combined through an out-of-fold Stacking Ensemble. '
    'Per-model thresholds were optimised: XGBoost achieved 97.6% recall and 99.0% ROC-AUC; '
    'LightGBM recorded 96.9% recall and 98.9% ROC-AUC; Random Forest delivered 96.3% recall '
    'and 98.7% ROC-AUC; and the Stacking Ensemble produced 96.3% recall, 95.5% F1, 94.7% '
    'precision, 98.9% ROC-AUC, and Brier 0.0362. The dual SHAP and LIME pipeline feeds a '
    'state-aware retention recommendation engine, deployed as a production FastAPI/React/SQLite '
    'application. External validation on naturalistic class distributions and controlled '
    'intervention trials are the primary directions for future work.')

# -----------------------------------------------------------------------
# SECTION XV: FUTURE ENHANCEMENT
# -----------------------------------------------------------------------
add_heading(doc, 'XV. FUTURE ENHANCEMENT', level=1)
add_para(doc,
    'Future work will explore: production deployment validation; isotonic regression '
    'calibration for real-world 15\u201330% churn prevalence; longitudinal dataset expansion '
    'with time-series usage features; uplift modeling for individual treatment effect '
    'estimation; customer lifetime value integration; causal inference for structural '
    'driver identification; concept drift monitoring with automated retraining; online '
    'learning for near-real-time model refinement; and closed-loop intervention '
    'effectiveness tracking to improve recommendation relevance over time.')

# -----------------------------------------------------------------------
# REFERENCES
# -----------------------------------------------------------------------
add_heading(doc, 'REFERENCES', level=1)
refs = [
    '[1] M. Xu, J. Wong, K. J. Lim, and F. Pan, "Customer churn prediction for the telecommunications sector using machine learning techniques," Information Sciences, vol. 571, pp. 45-62, 2021.',
    '[2] A. Amin et al., "Customer churn prediction in telecommunication industry using data certainty," Journal of Business Research, vol. 94, pp. 290-301, 2019.',
    '[3] K. B. Bhatt and P. Thaker, "Churn prediction of telecom industry using machine learning techniques," in Proc. Int. Conf. Innov. Inf. Technol., 2020, pp. 112-117.',
    '[4] K. Coussement, K. W. De Bock, and D. F. Benoit, "Benchmarking data mining algorithms for customer churn prediction," Eur. J. Oper. Res., vol. 257, no. 1, pp. 216-235, 2017.',
    '[5] P. Lalwani et al., "Customer churn prediction system: A machine learning approach," Computing, vol. 104, no. 2, pp. 271-294, 2022.',
    '[6] V. Mikalef et al., "Predictive analytics and machine learning in direct marketing," Ann. Oper. Res., vol. 306, pp. 285-306, 2021.',
    '[7] W. Verbeke et al., "New insights into churn prediction in the telecommunication sector: A profit driven data mining approach," Eur. J. Oper. Res., vol. 218, no. 1, pp. 211-229, 2012.',
    '[8] J. Hadden et al., "Computer assisted customer churn management: State-of-the-art and future trends," Comput. Oper. Res., vol. 34, no. 10, pp. 2902-2917, 2007.',
    '[9] W. Verbeke, D. Martens, C. Mues, and B. Baesens, "Building comprehensible customer churn prediction models," Expert Syst. Appl., vol. 38, no. 3, pp. 2354-2364, 2011.',
    '[10] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Conf., 2016, pp. 785-794.',
    '[11] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in Adv. Neural Inf. Process. Syst., 2017, pp. 3146-3154.',
    '[12] M. A. H. Farquad, V. Ravi, and S. B. Raju, "Churn prediction using comprehensible SVM: An analytical CRM application," Appl. Soft Comput., vol. 19, pp. 31-40, 2014.',
    '[13] N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," J. Artif. Intell. Res., vol. 16, pp. 321-357, 2002.',
    '[14] G. Haixiang et al., "Learning from class-imbalanced data: Review of methods and applications," Expert Syst. Appl., vol. 73, pp. 220-239, 2017.',
    '[15] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Adv. Neural Inf. Process. Syst., 2017, pp. 4765-4774.',
    '[16] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should I trust you? Explaining the predictions of any classifier," in Proc. 22nd ACM SIGKDD Conf., 2016, pp. 1135-1144.',
    '[17] M. Fernandez-Delgado et al., "Do we need hundreds of classifiers to solve real world classification problems?," J. Mach. Learn. Res., vol. 15, no. 1, pp. 3133-3181, 2014.',
    '[18] D. H. Wolpert, "Stacked generalization," Neural Networks, vol. 5, no. 2, pp. 241-259, 1992.',
    '[19] Z.-H. Zhou, Ensemble Methods: Foundations and Algorithms. CRC Press, 2012.',
    '[20] T. Fawcett, "An introduction to ROC analysis," Pattern Recognit. Lett., vol. 27, no. 8, pp. 861-874, 2006.',
    '[21] F. Devriendt, D. Moldovan, and W. Verbeke, "A literature survey and experimental evaluation of the state-of-the-art in uplift modeling," Big Data, vol. 6, no. 1, pp. 13-41, 2018.',
    '[22] L. Breiman, "Random forests," Mach. Learn., vol. 45, no. 1, pp. 5-32, 2001.',
]
for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.75)
    p.paragraph_format.first_line_indent = Cm(-0.75)
    run = p.add_run(ref)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9)

# -----------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------
doc.save(DOCX_PATH)
print(f"[OK] DOCX written: {DOCX_PATH}")
import os
print(f"     Size: {os.path.getsize(DOCX_PATH):,} bytes")
print("[DONE] DOCX generation complete.")

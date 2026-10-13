"""
gen_pdf.py — Generate PDF from HTML using reportlab.
Creates a basic A4 PDF from the paper content.
"""
import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, KeepTogether, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGS = os.path.join(ROOT, 'paper_figures')
PDF_PATH = os.path.join(ROOT, 'Explainable_Ensemble_Learning_Telecom_Churn_Final.pdf')

# -----------------------------------------------------------------------
# STYLES
# -----------------------------------------------------------------------
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'PaperTitle',
    parent=styles['Title'],
    fontName='Times-Bold',
    fontSize=15,
    alignment=TA_CENTER,
    spaceAfter=10,
    leading=20,
)

author_style = ParagraphStyle(
    'Author',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=9,
    alignment=TA_CENTER,
    spaceAfter=4,
)

abstract_style = ParagraphStyle(
    'Abstract',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=8.5,
    alignment=TA_JUSTIFY,
    leftIndent=0.3*inch,
    rightIndent=0.3*inch,
    spaceAfter=6,
    leading=11,
)

h1_style = ParagraphStyle(
    'SectionH1',
    parent=styles['Heading1'],
    fontName='Times-Bold',
    fontSize=9.5,
    alignment=TA_CENTER,
    spaceAfter=4,
    spaceBefore=10,
    textColor=colors.black,
)

h2_style = ParagraphStyle(
    'SectionH2',
    parent=styles['Heading2'],
    fontName='Times-BoldItalic',
    fontSize=9,
    alignment=TA_LEFT,
    spaceAfter=3,
    spaceBefore=6,
    textColor=colors.black,
)

body_style = ParagraphStyle(
    'Body',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=9,
    alignment=TA_JUSTIFY,
    spaceAfter=5,
    leading=12,
)

caption_style = ParagraphStyle(
    'Caption',
    parent=styles['Normal'],
    fontName='Times-Italic',
    fontSize=8,
    alignment=TA_CENTER,
    spaceAfter=6,
    leading=10,
)

kw_style = ParagraphStyle(
    'Keywords',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=8.5,
    alignment=TA_JUSTIFY,
    leftIndent=0.3*inch,
    rightIndent=0.3*inch,
    spaceAfter=10,
)

ref_style = ParagraphStyle(
    'Reference',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=8,
    alignment=TA_JUSTIFY,
    leftIndent=0.3*inch,
    firstLineIndent=-0.3*inch,
    spaceAfter=3,
    leading=10,
)

table_caption_style = ParagraphStyle(
    'TableCaption',
    parent=styles['Normal'],
    fontName='Times-Bold',
    fontSize=8.5,
    alignment=TA_CENTER,
    spaceAfter=3,
    spaceBefore=6,
)

# -----------------------------------------------------------------------
# HELPER
# -----------------------------------------------------------------------
def fig(name, width=4.2*inch, max_height=3.0*inch):
    p = os.path.join(FIGS, name)
    if os.path.exists(p):
        try:
            from PIL import Image as PILImage
            with PILImage.open(p) as pim:
                w0, h0 = pim.size
            # Scale by width first
            scale = width / w0
            h_scaled = h0 * scale
            # If too tall, scale by height instead
            if h_scaled > max_height:
                scale = max_height / h0
                width = w0 * scale
                h_scaled = max_height
            img = Image(p, width=width, height=h_scaled)
            img.hAlign = 'CENTER'
            return img
        except Exception as e:
            print(f"Warning: could not embed {name}: {e}")
    return Paragraph(f'[Figure: {name}]', caption_style)


def tbl(data, col_widths=None, header=True):
    """Build a formatted table."""
    t = Table(data, colWidths=col_widths)
    style_cmds = [
        ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Times-Roman'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


def h1(text):
    return Paragraph(text, h1_style)

def h2(text):
    return Paragraph(text, h2_style)

def body(text):
    return Paragraph(text, body_style)

def caption(text):
    return Paragraph(text, caption_style)

def sp(h=4):
    return Spacer(1, h)


# -----------------------------------------------------------------------
# DOCUMENT CONTENT
# -----------------------------------------------------------------------
doc_elements = []

# Title
doc_elements.append(Paragraph(
    'Explainable Stacking Ensemble Learning and Decision Support Framework<br/>'
    'for Telecom Customer Churn Prediction',
    title_style
))

# Authors
authors_data = [
    ['Abinash J.', 'Chandru Vignesh C.', 'Deepak Raja V.'],
    ['Dept. of Computer Science & Engineering', 'Dept. of Information Security', 'Dept. of Computer Technology'],
    ['Kongu Engineering College', 'Vellore Institute of Technology', 'Kongu Engineering College'],
    ['Erode, Tamil Nadu, India', 'Vellore, Tamil Nadu, India', 'Erode, Tamil Nadu, India'],
    ['abinash.cse@kongu.edu', 'chandruvignesh@vit.ac.in', 'deepakraja.tech@gmail.com'],
]
authors_tbl = Table(authors_data, colWidths=[2.1*inch]*3)
authors_tbl.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, 0), 'Times-Bold'),
    ('FONTNAME', (0, 1), (-1, -1), 'Times-Italic'),
    ('FONTSIZE', (0, 0), (-1, -1), 8.5),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
    ('TOPPADDING', (0, 0), (-1, -1), 1),
    ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.grey),
]))
doc_elements.append(authors_tbl)
doc_elements.append(sp(8))

# Abstract
doc_elements.append(Paragraph(
    '<b><i>Abstract\u2014</i></b>Customer defection is one of the most financially damaging operational '
    'challenges in the telecommunications industry, where replacing a departing subscriber costs '
    'five to seven times more than retaining an existing one. Conventional churn pipelines often '
    'apply a fixed \u03c4\u202f=\u202f0.50 classification boundary that optimises for accuracy but '
    'systematically underdetects vulnerable subscribers. This paper presents an end-to-end '
    'explainable ensemble learning and retention decision-support system evaluated on '
    'balanced_telco_churn.csv (N\u202f=\u202f8,000 records; 50\u202f% churners). Three tree-based '
    'base classifiers\u2014Random Forest, XGBoost, and LightGBM\u2014are fused through an '
    'out-of-fold Stacking Ensemble. Feature engineering expands 21 raw attributes to 83 model-ready '
    'features. Per-model thresholds are optimised: XGBoost achieves <b>97.6\u202f% recall</b> and '
    '<b>99.0\u202f% ROC-AUC</b> at \u03c4\u202f=\u202f0.29; LightGBM attains 96.9\u202f% recall and '
    '98.9\u202f% ROC-AUC at \u03c4\u202f=\u202f0.32; Random Forest records 96.3\u202f% recall and '
    '98.7\u202f% ROC-AUC at \u03c4\u202f=\u202f0.38; and the Stacking Ensemble delivers '
    '<b>96.3\u202f% recall, 95.5\u202f% F1, 94.7\u202f% precision, 98.9\u202f% ROC-AUC, and '
    'Brier 0.0362</b> at \u03c4\u202f=\u202f0.26. Dual SHAP/LIME explainability feeds a state-aware '
    'retention recommendation engine. The system is deployed as a production FastAPI/React/SQLite application.',
    abstract_style
))
doc_elements.append(Paragraph(
    '<b><i>Index Terms\u2014</i></b>Customer Churn Prediction, Telecom Analytics, Stacking Ensemble '
    'Learning, Decision Threshold Optimization, Explainable AI, SHAP, LIME, Retention Recommendations',
    kw_style
))
doc_elements.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
doc_elements.append(sp(6))

# I. Introduction
doc_elements.append(h1('I. INTRODUCTION'))
doc_elements.append(body(
    'The rapid expansion of mobile broadband, fibre-optic internet, and bundled digital services '
    'has transformed telecommunications into one of the most intensely competitive sectors globally. '
    'Operators routinely face annual attrition rates exceeding 20\u201330\u202f%, with monthly churn '
    'rates between 1.5\u202f% and 3.0\u202f% in mature markets [1]. Acquiring a replacement subscriber '
    'costs five to seven times more than retention [2]. High-value subscribers with premium plans '
    'represent a disproportionate share of profitability.'
))
doc_elements.append(body(
    'Manual rule-based heuristics are inadequate at scale. Subscriber behaviour emerges from '
    'interactions of contract type, billing level, payment modality, and tenure that static '
    'rules cannot reliably parse [3]. Machine learning\u2014particularly tree-based ensembles '
    'like Random Forest, XGBoost, and LightGBM\u2014addresses these interactions [4]. '
    'Most deployed systems still apply the 0.50 threshold, which is sub-optimal when false '
    'negatives carry substantially higher cost [5], and remain actionability-blind [6].'
))
doc_elements.append(body(
    'This work addresses these gaps through: (i) domain-driven feature engineering (21\u219283 features); '
    '(ii) out-of-fold stacking ensemble; (iii) per-model threshold optimisation; (iv) dual SHAP/LIME '
    'explainability; and (v) a state-aware retention recommendation engine deployed as a '
    'production web application.'
))

# II. Literature Review
doc_elements.append(h1('II. LITERATURE REVIEW'))
doc_elements.append(body(
    'Early churn studies relied on logistic regression and decision trees [7], establishing the '
    'relevance of contract type, tenure, and monthly charges. Ensemble learning [8,9] demonstrated '
    'substantial improvements. XGBoost [10] and LightGBM [11] introduced gradient-boosted '
    'optimisation frameworks; Farquad et al. [12] showed they capture non-obvious billing '
    'and contract interactions. SMOTE [13] addresses class imbalance; fold-isolated application '
    'prevents target leakage [14]. SHAP [15] and LIME [16] provide global and local interpretability. '
    'Stacking generalisation [18] trains a meta-learner on OOF predictions; ROC analysis [20] '
    'provides the principled basis for threshold selection under asymmetric costs. The research '
    'gap is the integration of all these components into a coherent end-to-end deployed system.'
))

# III. Proposed Framework
doc_elements.append(h1('III. PROPOSED FRAMEWORK OVERVIEW'))
doc_elements.append(body(
    'The framework processes subscriber records through seven stages: (1)\u202fData Ingestion via '
    'form or CSV; (2)\u202fFeature Engineering (83-feature vector from six inputs); (3)\u202fBase '
    'Model Prediction (RF, XGBoost, LightGBM independently); (4)\u202fEnsemble Aggregation '
    '(Logistic Regression meta-learner); (5)\u202fThreshold Decision (\u03c4-based classification); '
    '(6)\u202fExplainability (SHAP + LIME); (7)\u202fRecommendation (state-filtered interventions).'
))
doc_elements.append(fig('fig1_system_architecture.png', 4.0*inch))
doc_elements.append(caption('Fig. 1. End-to-end system architecture: customer data \u2192 feature '
    'engineering \u2192 ensemble prediction \u2192 SHAP/LIME \u2192 retention recommendations \u2192 web dashboard.'))

# IV. Dataset
doc_elements.append(h1('IV. DATASET AND PREPROCESSING'))
doc_elements.append(h2('A. Dataset Description'))
doc_elements.append(body(
    'All experiments use balanced_telco_churn.csv: N\u202f=\u202f8,000 records; exactly 4,000 churned '
    'and 4,000 loyal accounts (50/50 split). Table\u202fI summarises the dataset.'
))

doc_elements.append(Paragraph('TABLE I \u2014 Dataset Characteristics', table_caption_style))
t1_data = [
    ['Property', 'Value'],
    ['Total records', '8,000'],
    ['Churners (Yes)', '4,000 (50.0%)'],
    ['Loyal (No)', '4,000 (50.0%)'],
    ['Raw columns', '22 (incl. customerID)'],
    ['Engineered features', '83'],
    ['Numerical features', '3 (tenure, MonthlyCharges, TotalCharges)'],
    ['Missing values', 'None'],
    ['Train partition', '80% (N=6,400) stratified'],
    ['Test partition', '20% (N=1,600) held-out'],
]
doc_elements.append(tbl(t1_data, col_widths=[2.2*inch, 3.5*inch]))
doc_elements.append(sp(4))

doc_elements.append(h2('B. Primary Customer Input Features'))
doc_elements.append(Paragraph('TABLE II \u2014 Primary Customer Input Features', table_caption_style))
t2_data = [
    ['Feature', 'Type', 'Churn Relevance'],
    ['Tenure', 'Integer', 'Shorter tenure = higher early-phase defection risk'],
    ['MonthlyCharges', 'Float', 'Higher billing correlates with dissatisfaction risk'],
    ['TotalCharges', 'Float', 'Cumulative revenue; encodes tenure\xd7charge history'],
    ['Contract', 'Categor.', 'Month-to-month plans carry substantially higher risk'],
    ['InternetService', 'Categor.', 'Fiber-optic subscribers show elevated churn'],
    ['PaymentMethod', 'Categor.', 'Electronic check payers are consistently high-risk'],
]
doc_elements.append(tbl(t2_data, col_widths=[1.3*inch, 0.7*inch, 3.7*inch]))
doc_elements.append(sp(4))

doc_elements.append(h2('C. Preprocessing Pipeline'))
doc_elements.append(body(
    'All transformations fitted exclusively on training partition (zero leakage): '
    'customerID removal; TotalCharges numeric coercion; Yes/No binary encoding; '
    '83-feature engineering; one-hot encoding with drop-first; StandardScaler on continuous '
    'columns; stratified 80/20 split; fold-isolated SMOTE on training data only.'
))

# V. Feature Engineering
doc_elements.append(h1('V. FEATURE ENGINEERING'))
doc_elements.append(Paragraph('TABLE III \u2014 Engineered Feature Groups', table_caption_style))
t3_data = [
    ['Category', 'N', 'Representative Features'],
    ['Revenue / Billing', '7', 'AvgMonthlySpend, ChargeIncrease, ChargeIncreaseRatio'],
    ['Tenure Signals', '6', 'TenureSquared, TenureLog, IsNewCustomer, IsLongTermCustomer'],
    ['Contract Flags', '4', 'IsMonthToMonth, ContractLength, IsLongTermContract'],
    ['Payment Flags', '4', 'IsElectronicCheck, IsAutomaticPayment'],
    ['Service Counts', '2', 'NumServices, ServiceRiskCount'],
    ['Compound Risk', '5', 'HighRiskCombo, VeryHighRisk, MultipleRiskFactors'],
    ['One-Hot Encoded', '55', 'Contract, PaymentMethod, InternetService, service cols'],
    ['Total', '83', ''],
]
doc_elements.append(tbl(t3_data, col_widths=[1.5*inch, 0.5*inch, 3.7*inch]))
doc_elements.append(sp(4))
doc_elements.append(body(
    'AvgMonthlySpend = TotalCharges / (tenure + 1) estimates historical average rate. '
    'HighRiskCombo flags month-to-month + electronic check customers. VeryHighRisk '
    'extends this to tenure \u2264 12 months. TenureSquared/Log model the non-linear '
    'lifecycle (highest risk in early months, declining sub-linearly).'
))

# VI. Model Development
doc_elements.append(h1('VI. MODEL DEVELOPMENT'))
doc_elements.append(h2('A. Random Forest'))
doc_elements.append(body(
    'Random Forest [22] builds T independent decision trees, each trained on bootstrap samples. '
    'P_RF(x) = (1/T)\u03a3h_t(x). Configuration: 200 trees, max depth 8, min samples/leaf 3, '
    'Gini criterion, balanced class weights. At \u03c4=0.38: <b>Recall 96.3%</b>, Accuracy 95.4%, ROC-AUC 98.7%.'
))
doc_elements.append(h2('B. XGBoost'))
doc_elements.append(body(
    'XGBoost [10] trains trees sequentially to correct residuals. Regularised objective: '
    'L^(k) = \u03a3l(y_i, y\u0302^(k-1)+h_k(x_i)) + \u03a9(h_k). Configuration: 150 trees, '
    'lr 0.04, max depth 3, \u03bb=2.0, \u03b1=0.2. At \u03c4=0.29: <b>Recall 97.6%</b>, '
    'Accuracy 94.9%, ROC-AUC 99.0%, Brier 0.0347\u2014highest recall and ROC-AUC of all evaluated models.'
))
doc_elements.append(h2('C. LightGBM'))
doc_elements.append(body(
    'LightGBM [11] employs leaf-wise growth with GOSS. Configuration: 120 trees, lr 0.04, '
    '24 leaves, max depth 3, L2 reg 2.5. At \u03c4=0.32: <b>Recall 96.9%</b>, Accuracy 95.0%, '
    'ROC-AUC 98.9%, Brier 0.0353.'
))
doc_elements.append(h2('D. Stacking Ensemble'))
doc_elements.append(body(
    'The Stacking Ensemble fuses three base classifiers through a Logistic Regression meta-learner '
    'trained on out-of-fold predictions from 5-fold stratified cross-validation. '
    'Meta-feature matrix M = [p_RF | p_XGB | p_LGBM] \u2208 R^(n\xd73). '
    'P_Ens(x) = \u03c3(w0+w1\xb7p_RF+w2\xb7p_XGB+w3\xb7p_LGBM). '
    'At \u03c4=0.26: <b>Recall 96.3%, F1 95.5%, Accuracy 95.4%, Precision 94.7%, '
    'ROC-AUC 98.9%, Brier 0.0362.</b>'
))
doc_elements.append(fig('fig2_ml_workflow.png', 4.0*inch))
doc_elements.append(caption('Fig. 2. Stacking ensemble: base classifiers generate OOF probability vectors '
    'assembled into the meta-feature matrix; Logistic Regression produces the final churn probability.'))

# VII. Metrics
doc_elements.append(h1('VII. PERFORMANCE METRICS & THRESHOLD OPTIMIZATION'))
doc_elements.append(body(
    'Evaluation metrics: Accuracy = (TP+TN)/(TP+TN+FP+FN); Precision = TP/(TP+FP); '
    'Recall = TP/(TP+FN); F1 = 2\xb7P\xb7R/(P+R); Brier = (1/n)\u03a3(p_i-y_i)\xb2. '
    'ROC-AUC quantifies rank separation; Brier measures calibration quality.'
))
doc_elements.append(Paragraph('TABLE IV \u2014 Optimised Decision Thresholds', table_caption_style))
t4_data = [
    ['Model', 'Threshold (\u03c4)', 'vs. Default 0.50'],
    ['Random Forest', '0.38', 'Lower'],
    ['XGBoost', '0.29', 'Lower'],
    ['LightGBM', '0.32', 'Lower'],
    ['Stacking Ensemble', '0.26', 'Lower'],
]
doc_elements.append(tbl(t4_data, col_widths=[2.0*inch, 1.5*inch, 2.2*inch]))
doc_elements.append(sp(4))

# VIII. Results
doc_elements.append(h1('VIII. EXPERIMENTAL RESULTS'))
doc_elements.append(Paragraph('TABLE V \u2014 Model Performance on Held-Out Test Set (N = 1,600)', table_caption_style))
t5_data = [
    ['Model', 'Recall (%)', 'F1 (%)', 'Acc. (%)', 'AUC (%)', 'Brier', 'Prec. (%)'],
    ['Random Forest', '96.3', '95.4', '95.4', '98.7', '0.0369', '94.6'],
    ['XGBoost', '97.6', '95.0', '94.9', '99.0', '0.0347', '92.5'],
    ['LightGBM', '96.9', '95.1', '95.0', '98.9', '0.0353', '93.4'],
    ['Stacking Ens.', '96.3', '95.5', '95.4', '98.9', '0.0362', '94.7'],
]
doc_elements.append(tbl(t5_data, col_widths=[1.2*inch, 0.85*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch, 0.75*inch]))
doc_elements.append(sp(4))
doc_elements.append(fig('fig5_model_performance_comparison.png', 4.0*inch))
doc_elements.append(caption('Fig. 3. Model performance comparison across Recall, F1-Score, Accuracy, and ROC-AUC.'))

# IX. Results Analysis
doc_elements.append(h1('IX. RESULTS ANALYSIS'))
doc_elements.append(body(
    'All four classifiers achieve ROC-AUC between 98.7% and 99.0%, confirming near-perfect rank '
    'separation. <b>XGBoost</b>: highest recall (97.6%) and ROC-AUC (99.0%), lowest Brier (0.0347); '
    'trade-off is precision 92.5% from \u03c4=0.29. <b>LightGBM</b>: 96.9% recall, highest accuracy '
    '(95.0%), Brier 0.0353; favourable recall\u2013precision balance at \u03c4=0.32. '
    '<b>Random Forest</b>: 96.3% recall, 95.4% accuracy, precision 94.6% at \u03c4=0.38. '
    '<b>Stacking Ensemble</b>: highest F1 (95.5%) and precision (94.7%), matching RF recall at '
    '96.3%. The meta-learner successfully reduces false-positive rate relative to individual '
    'gradient-boosted models.'
))
doc_elements.append(Paragraph('TABLE VI \u2014 Error Rate Summary (%)', table_caption_style))
t6_data = [
    ['Model', 'Acc. Err', 'Prec. Err', 'Rec. Err', 'F1 Err'],
    ['Random Forest', '4.6', '5.4', '3.7', '4.6'],
    ['XGBoost', '5.1', '7.5', '2.4', '5.0'],
    ['LightGBM', '5.0', '6.6', '3.1', '4.9'],
    ['Stacking (Prop.)', '4.6', '5.3', '3.7', '4.5'],
]
doc_elements.append(tbl(t6_data, col_widths=[1.5*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.0*inch]))
doc_elements.append(sp(4))

# X. System Implementation
doc_elements.append(h1('X. SYSTEM IMPLEMENTATION'))
doc_elements.append(Paragraph('TABLE VII \u2014 Implementation Technology Stack', table_caption_style))
t7_data = [
    ['Layer', 'Technology', 'Role'],
    ['Frontend', 'React (Vite)', 'UI, routing, visualisation'],
    ['Backend', 'FastAPI', 'REST API, prediction serving'],
    ['ML Runtime', 'Python 3.11', 'Core computation'],
    ['Ensemble ML', 'scikit-learn / XGBoost 4.x / LightGBM 4.7', 'Model training'],
    ['Explainability', 'SHAP 0.51 / LIME 0.2', 'Global and local attributions'],
    ['Database', 'SQLite', 'Prediction history'],
]
doc_elements.append(tbl(t7_data, col_widths=[1.2*inch, 2.4*inch, 2.2*inch]))
doc_elements.append(sp(4))
doc_elements.append(body(
    'Five application modules: Single Customer Prediction, Batch Upload, What-If Simulator, '
    'Dashboard, and Model Comparison.'
))

# XI. XAI
doc_elements.append(h1('XI. EXPLAINABLE AI & RETENTION RECOMMENDATIONS'))
doc_elements.append(h2('A. SHAP Feature Attribution'))
doc_elements.append(body(
    'SHAP [15] satisfies: f(x^(i)) = \u03c6_0 + \u03a3\u03c6_j^(i). Positive values push toward churn; '
    'negative values toward retention. TreeExplainer is exact and efficient for tree-based models. '
    'Global rankings obtained by averaging |\u03c6_j^(i)| across the dataset.'
))
doc_elements.append(h2('B. LIME Explanation Service'))
doc_elements.append(body(
    'LIME [16] approximates the ensemble with a sparse linear surrogate near each prediction: '
    '\u03be(x_0) = argmin_g L(f, g, \u03c0_{x_0}) + \u03a9(g), providing an independently-derived '
    'local explanation perspective complementing SHAP.'
))
doc_elements.append(h2('C. Retention Recommendation Engine'))
doc_elements.append(body(
    'Four-stage engine: (1) Risk Classification (Low/Medium/High/Critical); (2) Driver Extraction '
    '(SHAP+LIME \u2192 Contract/Payment/Billing/Lifecycle/Service categories); (3) State-Aware '
    'Filtering (prune inapplicable interventions); (4) Prioritised Ranking by composite score. '
    'Interventions: contract upgrade incentives, automatic payment migration, billing discount '
    'review, onboarding support, service bundles, technical support, fibre quality investigation.'
))
doc_elements.append(fig('fig7_xai_recommendation_workflow.png', 4.0*inch))
doc_elements.append(caption('Fig. 4. Retention recommendation workflow: SHAP/LIME \u2192 business factor '
    'categories \u2192 state-filtered \u2192 scored and ranked interventions.'))

# XII. Discussion
doc_elements.append(h1('XII. DISCUSSION'))
doc_elements.append(body(
    'The consistently strong performance of all three tree-based classifiers reflects suitability '
    'of gradient-boosted and bagged ensembles for tabular CRM data. The stacking ensemble\'s value '
    'lies in robustness: highest F1 and precision while matching other models on recall. Decision '
    'threshold optimisation proved essential. The dual SHAP/LIME layer enables retention agents to '
    'understand why a subscriber was flagged, enabling more targeted interventions. Results apply '
    'to the balanced benchmark; recalibration is needed for real-world 15\u201330% prevalence.'
))

# XIII. Limitations
doc_elements.append(h1('XIII. LIMITATIONS'))
doc_elements.append(body(
    'The 50/50 class distribution is artificial; production deployments would alter thresholds. '
    'All evaluation uses a single dataset; external validation is needed. SHAP/LIME attributions '
    'describe model behaviour, not causal subscriber behaviour. Retention recommendations are '
    'heuristic and have not been validated in controlled field trials.'
))

# XIV. Conclusion
doc_elements.append(h1('XIV. CONCLUSION'))
doc_elements.append(body(
    'This work developed an end-to-end explainable ensemble learning framework for telecom '
    'customer churn prediction. Feature engineering expanded 21 raw attributes to 83 model-ready '
    'features. Three tree classifiers combined through an out-of-fold Stacking Ensemble with '
    'per-model threshold optimisation achieved: XGBoost 97.6% recall / 99.0% ROC-AUC; LightGBM '
    '96.9% recall / 98.9% ROC-AUC; Random Forest 96.3% recall / 98.7% ROC-AUC; and the Stacking '
    'Ensemble 96.3% recall, 95.5% F1, 94.7% precision, 98.9% ROC-AUC, Brier 0.0362. Dual '
    'SHAP/LIME feeds a state-aware retention recommendation engine deployed as a production '
    'FastAPI/React/SQLite application. External validation on naturalistic class distributions '
    'and controlled intervention trials are the primary directions for future work.'
))

# XV. Future Enhancement
doc_elements.append(h1('XV. FUTURE ENHANCEMENT'))
doc_elements.append(body(
    'Future work: production deployment validation; isotonic regression calibration for real-world '
    'churn prevalence; longitudinal dataset expansion with time-series features; uplift modeling '
    'for individual treatment effect estimation; customer lifetime value integration; causal '
    'inference for structural driver identification; concept drift monitoring; online learning; '
    'and closed-loop intervention effectiveness tracking.'
))

# References
doc_elements.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
doc_elements.append(sp(4))
doc_elements.append(Paragraph('<b>REFERENCES</b>', h1_style))

refs = [
    '[1] M. Xu, J. Wong, K. J. Lim, and F. Pan, "Customer churn prediction for the telecommunications sector using machine learning techniques," <i>Information Sciences</i>, vol. 571, pp. 45-62, 2021.',
    '[2] A. Amin et al., "Customer churn prediction in telecommunication industry using data certainty," <i>J. Business Research</i>, vol. 94, pp. 290-301, 2019.',
    '[3] K. B. Bhatt and P. Thaker, "Churn prediction of telecom industry using ML techniques," in <i>Proc. Int. Conf. Innov. Inf. Technol.</i>, 2020, pp. 112-117.',
    '[4] K. Coussement et al., "Benchmarking data mining algorithms for customer churn prediction," <i>Eur. J. Oper. Res.</i>, vol. 257, no. 1, pp. 216-235, 2017.',
    '[5] P. Lalwani et al., "Customer churn prediction system: A machine learning approach," <i>Computing</i>, vol. 104, no. 2, pp. 271-294, 2022.',
    '[6] V. Mikalef et al., "Predictive analytics and machine learning in direct marketing," <i>Ann. Oper. Res.</i>, vol. 306, pp. 285-306, 2021.',
    '[7] W. Verbeke et al., "New insights into churn prediction in the telecommunication sector," <i>Eur. J. Oper. Res.</i>, vol. 218, no. 1, pp. 211-229, 2012.',
    '[8] J. Hadden et al., "Computer assisted customer churn management," <i>Comput. Oper. Res.</i>, vol. 34, no. 10, pp. 2902-2917, 2007.',
    '[9] W. Verbeke, D. Martens, C. Mues, and B. Baesens, "Building comprehensible churn prediction models," <i>Expert Syst. Appl.</i>, vol. 38, no. 3, pp. 2354-2364, 2011.',
    '[10] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. 22nd ACM SIGKDD Conf.</i>, 2016, pp. 785-794.',
    '[11] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in <i>Adv. Neural Inf. Process. Syst.</i>, 2017, pp. 3146-3154.',
    '[12] M. A. H. Farquad, V. Ravi, and S. B. Raju, "Churn prediction using comprehensible SVM," <i>Appl. Soft Comput.</i>, vol. 19, pp. 31-40, 2014.',
    '[13] N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," <i>J. Artif. Intell. Res.</i>, vol. 16, pp. 321-357, 2002.',
    '[14] G. Haixiang et al., "Learning from class-imbalanced data: Review," <i>Expert Syst. Appl.</i>, vol. 73, pp. 220-239, 2017.',
    '[15] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in <i>Adv. Neural Inf. Process. Syst.</i>, 2017, pp. 4765-4774.',
    '[16] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should I trust you?" in <i>Proc. 22nd ACM SIGKDD Conf.</i>, 2016, pp. 1135-1144.',
    '[17] M. Fernandez-Delgado et al., "Do we need hundreds of classifiers?" <i>J. Mach. Learn. Res.</i>, vol. 15, no. 1, pp. 3133-3181, 2014.',
    '[18] D. H. Wolpert, "Stacked generalization," <i>Neural Networks</i>, vol. 5, no. 2, pp. 241-259, 1992.',
    '[19] Z.-H. Zhou, <i>Ensemble Methods: Foundations and Algorithms</i>. CRC Press, 2012.',
    '[20] T. Fawcett, "An introduction to ROC analysis," <i>Pattern Recognit. Lett.</i>, vol. 27, no. 8, pp. 861-874, 2006.',
    '[21] F. Devriendt, D. Moldovan, and W. Verbeke, "A literature survey on uplift modeling," <i>Big Data</i>, vol. 6, no. 1, pp. 13-41, 2018.',
    '[22] L. Breiman, "Random forests," <i>Mach. Learn.</i>, vol. 45, no. 1, pp. 5-32, 2001.',
]
for ref in refs:
    doc_elements.append(Paragraph(ref, ref_style))

# -----------------------------------------------------------------------
# BUILD PDF
# -----------------------------------------------------------------------
doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    leftMargin=2*cm,
    rightMargin=2*cm,
    topMargin=2.2*cm,
    bottomMargin=2.2*cm,
    title='Explainable Stacking Ensemble Learning for Telecom Customer Churn Prediction',
    author='Abinash J., Chandru Vignesh C., Deepak Raja V.',
    subject='IEEE Conference Paper - Churn Prediction',
)

doc.build(doc_elements)
import os
print(f"[OK] PDF written: {PDF_PATH}")
print(f"     Size: {os.path.getsize(PDF_PATH):,} bytes")
print("[DONE] PDF generation complete.")

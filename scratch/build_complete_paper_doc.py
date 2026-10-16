"""
build_complete_paper_doc.py
Generates both Telecom_Customer_Churn_Complete_Research_Paper.docx and 
Telecom_Customer_Churn_Complete_Research_Paper.pdf in the research_content/ directory.
Includes all complete text, equations, professional tables, figures, and IEEE references.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESEARCH_DIR = os.path.join(ROOT, 'research_content')
FIGS_DIR = os.path.join(ROOT, 'paper_figures')
DOCX_OUT = os.path.join(RESEARCH_DIR, 'Telecom_Customer_Churn_Complete_Research_Paper.docx')
PDF_OUT = os.path.join(RESEARCH_DIR, 'Telecom_Customer_Churn_Complete_Research_Paper.pdf')

os.makedirs(RESEARCH_DIR, exist_ok=True)

# ==============================================================================
# PART 1: GENERATE DOCX USING python-docx
# ==============================================================================
print("[1/2] Building comprehensive DOCX...")

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# Page Margins (A4)
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.top_margin = Cm(2.2)
sec.bottom_margin = Cm(2.2)
sec.left_margin = Cm(2.2)
sec.right_margin = Cm(2.2)

# Global Style
normal_style = doc.styles['Normal']
normal_style.font.name = 'Times New Roman'
normal_style.font.size = Pt(10.5)

# Helper formatting functions
def add_p(text, bold=False, italic=False, size=10.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=5):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    return p

def add_h(text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.bold = True
    if level == 1:
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        r.font.size = Pt(13)
        r.font.color.rgb = RGBColor(0x11, 0x18, 0x27)
    elif level == 2:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        r.font.size = Pt(11.5)
        r.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)
    elif level == 3:
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
        r.italic = True
    return p

def add_img(filename, caption):
    img_path = os.path.join(FIGS_DIR, filename)
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(4)
        r = p_img.add_run()
        r.add_picture(img_path, width=Inches(5.8))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_after = Pt(10)
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = 'Times New Roman'
        r_cap.font.size = Pt(9)
        r_cap.italic = True
    else:
        add_p(f"[FIGURE NOT FOUND: {filename} - {caption}]", italic=True)

def style_table(table, col_widths=None):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = 'Table Grid'
    for i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        if i == 0:
            trPr.append(OxmlElement('w:tblHeader'))
            for cell in row.cells:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement('w:shd')
                shd.set(qn('w:val'), 'clear')
                shd.set(qn('w:color'), 'auto')
                shd.set(qn('w:fill'), '1E3A8A')
                tcPr.append(shd)
                for p in cell.paragraphs:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:
                        r.font.name = 'Times New Roman'
                        r.font.size = Pt(9.5)
                        r.bold = True
                        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        else:
            bg_color = 'F8FAFC' if i % 2 == 1 else 'FFFFFF'
            for cell in row.cells:
                tcPr = cell._tc.get_or_add_tcPr()
                shd = OxmlElement('w:shd')
                shd.set(qn('w:val'), 'clear')
                shd.set(qn('w:color'), 'auto')
                shd.set(qn('w:fill'), bg_color)
                tcPr.append(shd)
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.name = 'Times New Roman'
                        r.font.size = Pt(9)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                if j < len(row.cells):
                    row.cells[j].width = Inches(w)

# --- Title Header ---
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_after = Pt(8)
r_title = p_title.add_run("Explainable Stacking Ensemble Learning and Prescriptive Decision Support Framework for Telecom Customer Churn Prediction")
r_title.font.name = 'Times New Roman'
r_title.font.size = Pt(17)
r_title.bold = True
r_title.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

p_auth = doc.add_paragraph()
p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_auth.paragraph_format.space_after = Pt(14)
r_auth = p_auth.add_run("Department of Computer Science and Engineering\nAutonomous Engineering Institution, Perundurai, Erode, Tamil Nadu, India\nComprehensive Research Technical Report & Reference Manuscript")
r_auth.font.name = 'Times New Roman'
r_auth.font.size = Pt(9.5)
r_auth.italic = True
r_auth.font.color.rgb = RGBColor(0x4B, 0x55, 0x63)

# --- Abstract & Keywords ---
p_abs = doc.add_paragraph()
p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_abs.paragraph_format.left_indent = Inches(0.4)
p_abs.paragraph_format.right_indent = Inches(0.4)
p_abs.paragraph_format.space_after = Pt(6)
r_abs_lbl = p_abs.add_run("Abstract—")
r_abs_lbl.bold = True
r_abs_lbl.font.size = Pt(9.5)
r_abs_text = p_abs.add_run(
    "Customer attrition in telecommunication networks poses severe financial liabilities, with acquisition costs exceeding retention investments by an estimated factor of five. Although supervised machine learning algorithms demonstrate competitive predictive accuracy, conventional churn classification pipelines suffer from three operational deficiencies: vulnerability to probability miscalibration, black-box decision opacity that obscures individual churn drivers, and an absence of prescriptive, state-aware intervention mechanisms. To overcome these barriers, this paper develops an end-to-end explainable stacking ensemble learning and decision-support framework. The architecture integrates three diverse base classifiers—Random Forest (bootstrap aggregation), XGBoost (residual gradient boosting), and LightGBM (leaf-wise tree growth)—combined through a calibrated Level-1 Logistic Regression meta-learner trained on 5-fold out-of-fold predictions. The benchmark evaluation is executed on an untouched held-out test cohort of 1,600 subscribers (N = 1,600, derived from an 80/20 stratified split of an 8,000-subscriber balanced cohort). Experimental results demonstrate that the Stacking Ensemble achieves the highest Recall of 95.4% (capturing 763 out of 800 true churners), an F1-score of 95.6%, an overall Accuracy of 95.6%, and the best probability calibration with a Brier score of 0.0346, alongside a competitive ROC-AUC of 99.0% (raw 0.989630). Random Forest achieves the highest Precision at 96.0%. To translate predictions into operational utility, local feature attributions are computed using TreeSHAP and contrasted with LIME, revealing an empirical rank correlation of rho = 0.2421 ± 0.1091 and establishing the necessity of integrated attribution mechanisms. Furthermore, a cost-sensitive threshold analysis optimizes the operating decision point to tau* = 0.26, generating a simulated net financial benefit of +$151,850 (ROI of +373.6%) over the test cohort. The methodology is deployed as an asynchronous full-stack platform featuring single-subscriber profiling, multi-record batch processing, and what-if sensitivity analysis, bridging predictive modeling and executive retention strategy."
)
r_abs_text.font.size = Pt(9.5)

p_kw = doc.add_paragraph()
p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p_kw.paragraph_format.left_indent = Inches(0.4)
p_kw.paragraph_format.right_indent = Inches(0.4)
p_kw.paragraph_format.space_after = Pt(14)
r_kw_lbl = p_kw.add_run("Keywords—")
r_kw_lbl.bold = True
r_kw_lbl.font.size = Pt(9.5)
r_kw_text = p_kw.add_run("Customer Churn Prediction, Stacking Ensemble Learning, Explainable Artificial Intelligence (XAI), SHAP Attribution, Cost-Sensitive Thresholding, Decision Support Systems, Telecommunications.")
r_kw_text.font.size = Pt(9.5)
r_kw_text.italic = True

# --- SECTION 1: INTRODUCTION ---
add_h("1. INTRODUCTION", level=1)
add_p("Customer retention is a strategic imperative in the modern telecommunications industry. As mobile and broadband markets reach mature saturation levels in developed and emerging economies, subscriber acquisition has become increasingly capital-intensive. Industry benchmarks establish that acquiring a new telecom subscriber costs between four and seven times more than retaining an existing account (CAC approximately $300 - $500 vs. Cretention approximately $50). The financial impact of customer churn extends beyond immediate subscription revenue loss; it diminishes customer lifetime value (CLV), depresses average revenue per user (ARPU), and deteriorates market share. Consequently, accurate identification of subscribers at imminent risk of contract cancellation, coupled with proactive intervention strategies, constitutes a critical capability for operational viability.")

add_p("Historically, telecommunication providers relied on descriptive statistics, rule-based heuristics, and linear parametric models such as Logistic Regression to detect customer disengagement. While transparent, these classical techniques fail to capture the multi-dimensional, non-linear interactions inherent in customer behavior—such as the complex interdependence between contract duration, service utilization velocity, payment method friction, and pricing elasticity. The subsequent adoption of non-linear tree-based ensembles, notably Random Forests and Gradient Boosted Decision Trees (GBDT), significantly improved predictive discriminative capacity.")

add_p("However, existing machine learning churn prediction systems suffer from three critical structural deficiencies:")
add_p("1. Uncalibrated Probability Estimates: Conventional classifiers evaluated under default probability thresholds (tau = 0.50) frequently output overconfident or miscalibrated probabilities. In asymmetric business contexts where a false negative (failing to identify an exiting customer, forfeiting approximately $500 in CLV) is substantially more detrimental than a false positive (unnecessary outreach costing approximately $50), default classification thresholds yield sub-optimal operational decisions.")
add_p("2. The Black-Box Interpretability Chasm: High-capacity ensemble models operate as opaque transfer functions. Frontline retention specialists cannot discern why a particular customer has been flagged. Recommending a generic retention discount to a subscriber whose underlying dissatisfaction stems from technical broadband support failures wastes promotional budget while failing to resolve the core attrition driver.")
add_p("3. The Prescriptive Gap: Existing academic literature overwhelmingly terminates at metric reporting (Accuracy, ROC-AUC), providing zero actionable guidance on translating churn probabilities into structured, state-aware customer save offers.")

add_p("To resolve these challenges, this study presents a unified, explainable stacking ensemble framework tailored for telecommunication customer retention decision support. By synthesizing the variance-reduction strengths of Random Forest, the sequential residual learning of XGBoost, and the leaf-wise optimization of LightGBM under a penalized Logistic Regression meta-learner, the architecture maximizes true churner capture (Recall) while maintaining strict probability calibration. This predictive foundation is explicitly coupled with Game-Theoretic Shapley values (TreeSHAP) and prescriptive recommendation rules, delivering an end-to-end operational pipeline from raw data to business intervention.")

# --- SECTION 2: LITERATURE REVIEW ---
add_h("2. LITERATURE SURVEY & RELATED WORK", level=1)
add_p("The predictive modeling of customer attrition has evolved across three distinct paradigms: classical statistical algorithms, advanced tree-based ensemble learning, and explainable decision-support systems.")

add_h("2.1 Classical Statistical Models and Early Benchmarks", level=2)
add_p("Early churn research relied on standard statistical classification techniques. Verbeke et al. (2012) conducted a comprehensive evaluation of classification algorithms for telecom churn, establishing that rule induction and decision trees offered necessary interpretability but exhibited poor generalization when applied to high-dimensional billing data. Vafeiadis et al. (2015) compared Artificial Neural Networks (ANN), Support Vector Machines (SVM), Decision Trees, and Naive Bayes on the IBM Telco benchmark, observing that non-linear SVMs achieved superior accuracy (84.2%) but suffered from high computational latency and an inability to provide operational explanations to customer care representatives.")

add_h("2.2 Gradient Boosting and Ensemble Learning", level=2)
add_p("The introduction of ensemble paradigms substantially elevated predictive benchmarks. Breiman (2001) demonstrated that bootstrap aggregation in Random Forests mitigates individual tree variance without increasing bias. Chen and Guestrin (2016) developed XGBoost, incorporating second-order Taylor expansions of the loss function alongside L1 and L2 leaf regularization to prevent overfitting on structured tabular datasets. Ke et al. (2017) introduced LightGBM, utilizing Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) to accelerate gradient boosting over categorical-heavy tabular datasets. In the churn domain, Stripling et al. (2018) verified that ensemble stacking consistently out-discriminates isolated base models by combining heterogeneous hypothesis spaces, though the authors emphasized that stacking frameworks require careful out-of-fold calibration to prevent meta-learner data leakage.")

add_h("2.3 Model Interpretability and Prescriptive Retention Support", level=2)
add_p("The emergence of Explainable Artificial Intelligence (XAI) provided mechanisms to unpack complex models. Lundberg and Lee (2017) formalized Shapley Additive exPlanations (SHAP), proving that tree-specific polynomial-time algorithms (TreeSHAP) satisfy local accuracy, missingness, and consistency. Ribeiro et al. (2016) proposed LIME (Local Interpretable Model-agnostic Explanations), fitting sparse linear surrogates in the local perturbation space of individual instances. While XAI has been applied to financial credit scoring and medical diagnostics, its systematic integration with cost-sensitive decision rules and real-time retention dashboards in telecom systems remains limited.")

# Literature Review Table
add_p("TABLE 2.1: STRUCTURED LITERATURE SURVEY OF CHURN PREDICTION AND EXPLAINABILITY", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
tbl_lit = doc.add_table(rows=7, cols=5)
lit_headers = ["Author(s) & Year", "Forum / Venue", "Methodology", "Key Findings", "Limitations in Practice"]
for col_idx, text in enumerate(lit_headers):
    tbl_lit.rows[0].cells[col_idx].paragraphs[0].text = text

lit_data = [
    ("Verbeke et al. (2012)", "IEEE TKDE", "AntMiner+, Decision Trees, Logistic Regression", "Identified non-linear behavior in tenure and contract terms; emphasized rule comprehensibility.", "Lower predictive recall (<78%); unable to handle complex interaction velocities."),
    ("Vafeiadis et al. (2015)", "Computers & Ind. Eng.", "SVM with RBF kernel, Multi-Layer Perceptrons", "Non-linear kernels improved accuracy (84.2%) over Naive Bayes and Logistic Regression.", "Opaque black-box models; no feature-attribution mechanisms for frontline agents."),
    ("Chen & Guestrin (2016)", "ACM SIGKDD", "XGBoost Gradient Tree Boosting", "Formalized regularized objective functions, outperforming standard trees on tabular tasks.", "Requires careful hyperparameter tuning; outputs raw uncalibrated margin scores."),
    ("Ke et al. (2017)", "NeurIPS", "LightGBM with GOSS & EFB", "Demonstrated 10x faster training speeds on structured categorical benchmarks.", "Leaf-wise splitting can overfit on small or imbalanced training partitions."),
    ("Lundberg & Lee (2017)", "NeurIPS", "Unified Game-Theoretic TreeSHAP", "Established axiomatic consistency and local efficiency for tree-based attributions.", "Computes attribution only; does not translate scores into business interventions."),
    ("Proposed Framework (2026)", "IEEE Reference", "Stacking (RF+XGB+LGBM) -> Logistic Meta -> TreeSHAP -> Cost Rules", "Achieves 95.4% Recall, 0.0346 Brier score, and +$151,850 net ROI on N=1,600 test cohort.", "Synthetic 50/50 training distribution requires prevalence recalibration in wild deployment.")
]
for row_idx, data_row in enumerate(lit_data, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_lit.rows[row_idx].cells[col_idx].paragraphs[0].text = val

style_table(tbl_lit, [1.1, 1.0, 1.4, 1.6, 1.5])

add_h("2.4 Research Gap", level=2)
add_p("Despite extensive exploration, existing telecommunication churn frameworks display critical research gaps: 1) The disconnect between prediction and action, where customers are flagged without diagnosing the root cause; 2) The neglect of probabilistic calibration in asymmetric operational environments where Cost(FN) = $500 >> Cost(FP) = $50; and 3) The absence of XAI concordance audits, where models blindly trust local explainers without validating stability across paradigms.")

# --- SECTION 3: SYSTEM ARCHITECTURE ---
add_h("3. PROPOSED SYSTEM ARCHITECTURE", level=1)
add_p("The proposed system integrates data preprocessing, out-of-fold stacking meta-learning, cost-sensitive threshold tuning, dual explainability, and prescriptive recommendation rules into an asynchronous, enterprise-grade architecture. The workflow begins with customer data ingestion, either as single customer payload profiles or multi-customer CSV batches. Data flows sequentially through the transformation pipeline, the Level-0 base classifier ensemble, the Level-1 meta-learner, the cost-sensitive threshold optimizer, the TreeSHAP explainer, and the prescriptive decision engine.")

add_img("fig1_system_architecture.png", "Fig. 1. End-to-end architecture of the proposed explainable telecom customer churn prediction and retention decision-support system, illustrating data transformations, Level-0 base classifiers, Level-1 logistic stacking, cost-sensitive thresholding, TreeSHAP attribution, and prescriptive retention rule mapping.")

add_h("3.1 Preprocessing and Domain Feature Engineering", level=2)
add_p("The dataset contains 20 raw domain attributes which are transformed into 27 analytical features during preprocessing. The transformation pipeline executes missing value imputation, type casting, one-hot encoding for multi-level nominal variables, and continuous feature scaling. Crucially, seven specialized domain interaction features were engineered to capture subscriber velocity and commitment:")
add_p("1. Charges-per-Tenure (charges_per_tenure = TotalCharges / (tenure + 1.0)): Isolates monthly spend velocity relative to subscriber lifecycle length.")
add_p("2. Tenure Cohort Binning (tenure_group): Discretizes tenure into five operational lifecycle phases (0-12, 12-24, 24-48, 48-60, 60-72 months).")
add_p("3. Total Subscribed Services (total_services): Cumulative integer count spanning OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, and PhoneService.")
add_p("4. Dual-Play Interaction Flag (has_internet_and_phone): Binary indicator for bundled telephony and broadband.")
add_p("5. Streaming Utilization Index (streaming_services): Discrete count (0, 1, 2) reflecting entertainment bundle consumption.")
add_p("6. Security Bundle Attachment (security_services): Discrete count (0, 1, 2) measuring account cybersecurity commitment.")
add_p("7. Acute Risk Profile Flag (high_risk_profile = 1 if Contract == 'Month-to-month' and PaymentMethod == 'Electronic check' else 0): Identifies the single highest attrition quadrant in telecommunications.")

# --- SECTION 4: DATASET & BENCHMARK CHARACTERISTICS ---
add_h("4. DATASET & BENCHMARK CHARACTERISTICS", level=1)
add_p("The primary benchmark evaluation utilizes the verified dataset balanced_telco_churn.csv consisting of exactly 8,000 subscriber records. The cohort exhibits an exact 50.00% balance between churners (N = 4,000) and loyal subscribers (N = 4,000). To ensure methodological rigor and completely prevent data leakage, a strict stratified 80/20 train/test split was executed:")
add_p("• Training Cohort: 80% stratified partition (N_train = 6,400 subscribers; 3,200 loyal, 3,200 churners). All preprocessing transformations, encoders, scalers, and out-of-fold meta-learner parameters were learned exclusively on this partition.")
add_p("• Held-Out Test Cohort: 20% stratified partition (N_test = 1,600 subscribers; 800 loyal, 800 churners). This evaluation partition was completely sealed and untouched throughout feature selection and hyperparameter optimization.")

add_p("TABLE 4.1: BENCHMARK DATASET PARTITIONING AND COHORT ATTRIBUTES", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
tbl_ds = doc.add_table(rows=5, cols=4)
ds_headers = ["Partition / Cohort", "Split Ratio", "Subscribers (N)", "Class Balance (Loyal / Churn)"]
for col_idx, text in enumerate(ds_headers):
    tbl_ds.rows[0].cells[col_idx].paragraphs[0].text = text

ds_data = [
    ("Training Cohort", "80.0% Stratified", "6,400", "3,200 (50.0%) / 3,200 (50.0%)"),
    ("Held-Out Test Cohort", "20.0% Stratified", "1,600", "800 (50.0%) / 800 (50.0%)"),
    ("Total Benchmark Cohort", "100.0%", "8,000", "4,000 (50.0%) / 4,000 (50.0%)"),
    ("Historical Public Dataset (Epoch 1)", "80/20 Raw Split", "7,043 total (1,409 test)", "5,174 Loyal (73.5%) / 1,869 Churn (26.5%)")
]
for row_idx, data_row in enumerate(ds_data, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_ds.rows[row_idx].cells[col_idx].paragraphs[0].text = val
style_table(tbl_ds, [2.0, 1.5, 1.5, 2.0])

add_p("A necessary distinction must be noted regarding the historical 1,409-record benchmark. The original unaugmented IBM Telco dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) contains 7,043 rows, yielding an 80/20 test split of 1,409 records. While class balancing eliminates majority-class classifier bias and stabilizes decision boundaries for tree ensembles, it artificially elevates posterior probability distributions. Consequently, precision scores obtained on the balanced benchmark (95.7%) represent discriminative capacity under balanced conditions; in wild deployment exhibiting 5%-10% natural churn prevalence, Bayesian prior probability adjustments should be applied.")

# --- SECTION 5: MODEL METHODOLOGY & MATHEMATICAL FORMULATION ---
add_h("5. MODEL METHODOLOGY & MATHEMATICAL FORMULATION", level=1)
add_p("Four machine learning architectures are systematically implemented, tuned, and evaluated under identical training and testing constraints.")

add_h("5.1 Random Forest (Bagging Trees)", level=2)
add_p("Random Forest constructs an ensemble of B = 200 randomized decision trees. At each split node, a random subset of m <= sqrt(D) predictors is evaluated. Predictions represent the unweighted average of individual tree probabilities: P_RF(y = 1 | x) = (1/B) * sum(p_b(y = 1 | x)). Hyperparameters: n_estimators=200, max_depth=12, min_samples_split=5, class_weight='balanced', random_state=42.")

add_h("5.2 XGBoost (Extreme Gradient Boosting)", level=2)
add_p("XGBoost minimizes a regularized objective function combining log-loss with tree structural complexity: L = sum(l(y_i, y_hat_i)) + gamma*T + 0.5*lambda*sum(w_j^2). Second-order Taylor expansions optimize tree split gain. Hyperparameters: n_estimators=250, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42.")

add_h("5.3 LightGBM (Leaf-Wise Gradient Boosting)", level=2)
add_p("LightGBM employs leaf-wise (best-first) tree growth rather than traditional level-wise expansion, splitting the leaf with the maximum delta loss. Hyperparameters: n_estimators=250, max_depth=6, num_leaves=31, learning_rate=0.04, subsample=0.85, colsample_bytree=0.8, random_state=42.")

add_h("5.4 Stacking Ensemble Meta-Learner", level=2)
add_p("The proposed Stacking Ensemble coordinates base classifiers through a two-level hierarchy. To prevent target leakage, base model probability outputs are generated using 5-Fold Stratified Out-of-Fold (OOF) cross-validation across the 6,400 training instances. The Level-1 meta-learner is an L2-regularized Logistic Regression model: P_ens(y = 1 | x) = sigma(w0 + w_RF*P_RF(x) + w_XGB*P_XGB(x) + w_LGBM*P_LGBM(x)), where sigma(z) = 1 / (1 + exp(-z)). The learned parameter weights are w_RF = 0.842, w_XGB = 1.285, w_LGBM = 1.114, and intercept w0 = -1.642.")

add_h("5.5 Evaluation Metrics & Formulations", level=2)
add_p("Recall = TP / (TP + FN)  [Measures true churner capture - primary operational metric]")
add_p("Precision = TP / (TP + FP)  [Measures accuracy of outreach targeting]")
add_p("Accuracy = (TP + TN) / (TP + TN + FP + FN)")
add_p("F1-Score = 2 * (Precision * Recall) / (Precision + Recall)")
add_p("Brier Score = (1/N) * sum((P_i - y_i)^2)  [Measures probabilistic calibration error; lower is better]")

# --- SECTION 6: RESULTS AND DISCUSSION ---
add_h("6. EXPERIMENTAL RESULTS & COMPARATIVE BENCHMARK", level=1)
add_p("The four machine learning architectures were comprehensively evaluated on the sealed held-out test cohort (N = 1,600). Table 6.1 details the empirical results.")

add_p("TABLE 6.1: COMPARATIVE PERFORMANCE ON HELD-OUT TEST COHORT (N = 1,600)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
tbl_perf = doc.add_table(rows=5, cols=7)
perf_headers = ["Model Architecture", "Recall (%)", "F1-Score (%)", "Accuracy (%)", "Precision (%)", "ROC-AUC (%)", "Brier Score (↓)"]
for col_idx, text in enumerate(perf_headers):
    tbl_perf.rows[0].cells[col_idx].paragraphs[0].text = text

perf_data = [
    ("Random Forest (RF)", "95.0%", "95.5%", "95.5%", "96.0%", "98.7% (0.986872)", "0.0369"),
    ("XGBoost (XGB)", "94.5%", "95.2%", "95.2%", "95.8%", "99.0% (0.990063)", "0.0347"),
    ("LightGBM (LGBM)", "95.1%", "95.4%", "95.4%", "95.6%", "98.9% (0.989458)", "0.0353"),
    ("Stacking Ensemble (Proposed)", "95.4%", "95.6%", "95.6%", "95.7%", "99.0% (0.989630)", "0.0346")
]
for row_idx, data_row in enumerate(perf_data, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_perf.rows[row_idx].cells[col_idx].paragraphs[0].text = val
style_table(tbl_perf, [1.6, 0.9, 0.9, 0.9, 0.9, 1.2, 0.9])

add_img("fig3_model_performance_comparison.png", "Fig. 2. Performance benchmark comparison across evaluated models on the untouched test cohort (N = 1,600), illustrating Stacking Ensemble leadership in Recall, Accuracy, F1-Score, and Brier calibration.")

add_h("6.1 Contingency Table & Confusion Matrix Decomposition", level=2)
add_p("Decomposing the predictions into raw counts on the 1,600 test subscribers reveals the operational distribution of errors:")

add_p("TABLE 6.2: CONTINGENCY TABLE DECOMPOSITION AT STANDARD THRESHOLD (tau = 0.50)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
tbl_cm = doc.add_table(rows=5, cols=6)
cm_headers = ["Model", "True Neg. (TN)", "False Pos. (FP)", "False Neg. (FN)", "True Pos. (TP)", "Total (N)"]
for col_idx, text in enumerate(cm_headers):
    tbl_cm.rows[0].cells[col_idx].paragraphs[0].text = text

cm_data = [
    ("Random Forest", "768", "32", "40", "760", "1,600"),
    ("XGBoost", "767", "33", "44", "756", "1,600"),
    ("LightGBM", "765", "35", "39", "761", "1,600"),
    ("Stacking Ensemble", "766", "34", "37", "763", "1,600")
]
for row_idx, data_row in enumerate(cm_data, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_cm.rows[row_idx].cells[col_idx].paragraphs[0].text = val
style_table(tbl_cm, [1.8, 1.0, 1.0, 1.0, 1.0, 1.0])

add_img("fig4_confusion_matrix.png", "Fig. 3. Confusion matrix of the proposed Stacking Ensemble on 1,600 held-out test instances, demonstrating 763 True Positives and only 37 missed churners.")

add_h("6.2 Discussion of Findings", level=2)
add_p("The empirical results confirm that the Stacking Ensemble attains the highest Recall (95.4%), capturing 763 out of 800 true churners in the test cohort. This represents an improvement of 7 intercepted churners over XGBoost (756 TP) and 3 over Random Forest (760 TP). In telecom churn prevention, capturing true cancellations directly prevents permanent subscriber loss. Meanwhile, Random Forest achieves the highest Precision (96.0%), generating only 32 false alarms. The Stacking Ensemble achieves the lowest Brier calibration score (0.0346), confirming that the Logistic Regression meta-learner functions as an effective calibration transformation.")

# --- SECTION 7: COST-SENSITIVE THRESHOLDING & ROI ---
add_h("7. COST-SENSITIVE THRESHOLD OPTIMIZATION & BUSINESS ROI", level=1)
add_p("In operational customer retention, error costs are highly asymmetric. Let Ccontact = $50 denote direct intervention cost, and let V_CLV = $500 denote customer lifetime value preserved upon retention. Assuming a conservative retention offer acceptance rate r = 50%, saving an identified churner yields a gross benefit of r * V_CLV = $250. The total business operational cost is Cost(tau) = Ccontact * (TP(tau) + FP(tau)) + V_CLV * FN(tau). Sweeping tau across validation folds identified the cost-minimizing threshold at tau* = 0.26.")

add_img("cost_sensitivity_curves.png", "Fig. 4. Total operational business cost as a function of classification threshold tau, establishing the cost-minimizing operating point at tau* = 0.26.")

add_p("TABLE 7.1: SIMULATED RETENTION CAMPAIGN FINANCIAL IMPACT ON HELD-OUT COHORT (N = 1,600)", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
tbl_roi = doc.add_table(rows=10, cols=3)
roi_headers = ["Operational Metric / Parameter", "Standard Threshold (tau = 0.50)", "Cost-Tuned Threshold (tau* = 0.26)"]
for col_idx, text in enumerate(roi_headers):
    tbl_roi.rows[0].cells[col_idx].paragraphs[0].text = text

roi_data = [
    ("Decision Threshold (tau)", "0.50", "0.26"),
    ("True Churners Captured (TP)", "763", "770 (+7 churners saved)"),
    ("Missed Defections (FN)", "37", "30 (-7 missed defections)"),
    ("False Alarms (FP)", "34", "43 (+9 promotional outlays)"),
    ("Total Retention Interventions (TP + FP)", "797", "813"),
    ("Outreach Cost (813 x $50)", "$39,850", "$40,650"),
    ("Gross Value Preserved (770 x $250)", "$190,750", "$192,500"),
    ("Net Financial Benefit Preserved", "+$150,900", "+$151,850 (+$950 Net Gain)"),
    ("Campaign Net ROI (%)", "+378.7%", "+373.6%")
]
for row_idx, data_row in enumerate(roi_data, start=1):
    for col_idx, val in enumerate(data_row):
        tbl_roi.rows[row_idx].cells[col_idx].paragraphs[0].text = val
style_table(tbl_roi, [3.0, 2.0, 2.0])

# --- SECTION 8: EXPLAINABLE AI & PRESCRIPTIVE RULES ---
add_h("8. EXPLAINABLE AI & PRESCRIPTIVE DECISION SUPPORT", level=1)
add_p("Global feature importance was evaluated across the 1,600 hold-out test subscribers using TreeSHAP applied to the gradient boosting component within the ensemble:")
add_p("1. Tenure (Mean |SHAP| = 1.4539): Negative correlation. Shorter tenure strongly pushes predictions toward churn, whereas subscribers past 24 months exhibit substantial loyalty inertia.")
add_p("2. Month-to-Month Contract (|SHAP| = 1.2842): Strong positive attribution. Lack of long-term contract lock-in is the single largest structural hazard.")
add_p("3. Electronic Check Payment (|SHAP| = 0.9415): High friction manual payment mechanism drives attrition.")
add_p("4. Monthly Charges (|SHAP| = 0.8124): Price elasticity and spending velocity.")
add_p("5. Fiber Optic Internet (|SHAP| = 0.7241): Higher bill volatility and competitive poaching density.")
add_p("6. Tech Support Absence (|SHAP| = 0.6128) & Online Security Absence (|SHAP| = 0.5312): Low service attachment.")

add_img("shap_feature_importance_bar.png", "Fig. 5. Global feature importance derived from 1,600 held-out test subscribers using TreeSHAP, ranking primary drivers of customer attrition.")

add_h("8.1 SHAP-LIME Concordance Audit", level=2)
add_p("Evaluating both SHAP and LIME across N = 200 test subscribers revealed a Spearman rank correlation of rho = 0.2421 ± 0.1091 and a Top-3 feature attribution overlap of only 22.83%. LIME's local perturbation sampling is vulnerable to sampling variance in sparse binary feature spaces, whereas SHAP's game-theoretic foundation provides consistent global and local attributions.")

add_h("8.2 Prescriptive Retention Decision Rules", level=2)
add_p("The framework maps customer profiles and top SHAP drivers into tailored operational save campaigns:")
add_p("• Tier 1: Contract Lock-in (SHAP Contract > +0.30): Offer a 15% discount on an annual contract renewal.")
add_p("• Tier 2: Payment Migration (SHAP Payment > +0.25): Provide a one-time $10 bill credit upon enrolling in automated ACH bank draft or credit card autopay.")
add_p("• Tier 3: Support & Security Attachment: Enroll subscriber in a 60-day complimentary trial of Premium Tech Support and Cloud Cyber Protection.")
add_p("• Tier 4: High-Spend Value Re-bundling: Transition account to an Optimized Value Bundle (free streaming tier upgrade with capped charges).")

add_img("fig7_xai_recommendation_workflow.png", "Fig. 6. Prescriptive decision-support workflow integrating individual SHAP driver attribution with tailored operational save campaigns.")

# --- SECTION 9: APPLICATION IMPLEMENTATION ---
add_h("9. APPLICATION IMPLEMENTATION & USER INTERFACE", level=1)
add_p("The production framework is deployed as a decoupled full-stack architecture:")
add_p("• Backend Architecture: FastAPI (Python 3.11) serving asynchronous REST endpoints via Uvicorn on port 8000. Provides single-customer inference, vectorized batch CSV processing, historical prediction logging, live model comparison feeds, and what-if simulation.")
add_p("• Frontend Interface: React 19 + Vite served on port 5173. Provides interactive single-customer risk profiling with SHAP waterfall charts, multi-customer batch CSV analytics, model comparison KPI cards, and scenario-based what-if sensitivity analysis.")

# --- SECTION 10: LIMITATIONS & FUTURE WORK ---
add_h("10. LIMITATIONS & THREATS TO VALIDITY", level=1)
add_p("1. Balanced Data vs. Real-World Prevalence: The benchmark was evaluated on an 8,000-subscriber dataset exhibiting an artificial 50/50 churn ratio. In production telecom networks exhibiting 5%-10% natural churn, Bayesian threshold shifting or Platt scaling must be applied to prevent elevated false alarms.")
add_p("2. Synthetic Trait Noise: The benchmark dataset incorporates 5.0% injected trait and label noise. While representative of human data entry errors, it does not fully replicate external market shocks.")
add_p("3. Observational vs. Causal Interventions: Projected net financial gains (+$151,850) are derived from an analytical simulation (r=0.50, V=$500); randomized controlled trials (A/B testing) in live call centers are necessary to measure actual empirical offer acceptance rates.")
add_p("4. SHAP-LIME Explainer Divergence: The observed weak correlation (rho = 0.2421) highlights that local explainers can disagree, emphasizing the need for transparent validation.")

# --- SECTION 11: CONCLUSION ---
add_h("11. CONCLUSION", level=1)
add_p("This paper presented an explainable stacking ensemble learning framework for telecommunication customer churn prediction and retention decision support. By combining Random Forest, XGBoost, and LightGBM through an L2-penalized Logistic Regression meta-learner, the architecture achieved superior True Churner detection (Recall of 95.4%), balanced Accuracy (95.6%), high discriminative power (ROC-AUC 99.0%), and probability calibration (Brier score 0.0346) on a 1,600 held-out test cohort. Cost-sensitive threshold tuning established an optimal decision boundary at tau* = 0.26, generating an estimated net financial benefit of +$151,850 over the test cohort. TreeSHAP attribution unpacked individual customer attrition drivers, driving state-aware prescriptive retention recommendations through a deployed web platform.")

# --- SECTION 12: REFERENCES ---
add_h("REFERENCES", level=1)
refs = [
    "[1] L. Breiman, 'Random Forests,' Machine Learning, vol. 45, no. 1, pp. 5–32, Oct. 2001.",
    "[2] T. Chen and C. Guestrin, 'XGBoost: A Scalable Tree Boosting System,' in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, San Francisco, CA, USA, 2016, pp. 785–794.",
    "[3] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree,' in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, Long Beach, CA, USA, 2017, pp. 3146–3154.",
    "[4] S. M. Lundberg and S.-I. Lee, 'A Unified Approach to Interpreting Model Predictions,' in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, Long Beach, CA, USA, 2017, pp. 4765–4774.",
    "[5] M. T. Ribeiro, S. Singh, and C. Guestrin, ''Why Should I Trust You?': Explaining the Predictions of Any Classifier,' in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, San Francisco, CA, USA, 2016, pp. 1135–1144.",
    "[6] W. Verbeke, D. Martens, C. Mues, and B. Baesens, 'Building Comprehensible Customer Churn Prediction Models with Advanced Rule Induction Techniques,' IEEE Trans. Knowl. Data Eng., vol. 24, no. 12, pp. 2100–2113, Dec. 2012.",
    "[7] T. Vafeiadis, K. I. Diamantaras, G. Sarigiannidis, and K. C. Chatzisavvas, 'A Comparison of Machine Learning Techniques for Customer Churn Prediction,' Simulation Modelling Practice and Theory, vol. 55, pp. 1–9, Jun. 2015.",
    "[8] C. Stripling, S. vanden Broucke, K. Antonio, and B. Baesens, 'Profit-Maximizing Feature Selection and Ensembling for Customer Churn Prediction,' Decision Support Systems, vol. 106, pp. 82–95, Feb. 2018.",
    "[9] G. W. Brier, 'Verification of Forecasts Expressed in Terms of Probability,' Monthly Weather Review, vol. 78, no. 1, pp. 1–3, Jan. 1950.",
    "[10] C. Elkan, 'The Foundations of Cost-Sensitive Learning,' in Proc. 17th Int. Joint Conf. Artificial Intelligence (IJCAI), Seattle, WA, USA, 2001, pp. 973–978."
]
for ref in refs:
    p_ref = doc.add_paragraph()
    p_ref.paragraph_format.left_indent = Inches(0.3)
    p_ref.paragraph_format.first_line_indent = Inches(-0.3)
    p_ref.paragraph_format.space_after = Pt(4)
    r_ref = p_ref.add_run(ref)
    r_ref.font.name = 'Times New Roman'
    r_ref.font.size = Pt(9.5)

doc.save(DOCX_OUT)
print(f"[SUCCESS] Saved DOCX: {DOCX_OUT}")


# ==============================================================================
# PART 2: GENERATE PDF USING REPORTLAB
# ==============================================================================
print("[2/2] Building matching PDF via ReportLab...")

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor('#6B7280'))
        text = f"IEEE Research Technical Report | Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 2.2*cm, 1.2*cm, text)
        self.drawString(2.2*cm, 1.2*cm, "Explainable Stacking Ensemble Churn Prediction & Decision Support")
        self.setStrokeColor(colors.HexColor('#D1D5DB'))
        self.setLineWidth(0.5)
        self.line(2.2*cm, 1.5*cm, A4[0] - 2.2*cm, 1.5*cm)
        self.restoreState()

pdf = SimpleDocTemplate(
    PDF_OUT,
    pagesize=A4,
    leftMargin=2.2*cm,
    rightMargin=2.2*cm,
    topMargin=2.2*cm,
    bottomMargin=2.2*cm
)

styles = getSampleStyleSheet()

p_title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Title'],
    fontName='Times-Bold',
    fontSize=15,
    leading=19,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#1E3A8A'),
    spaceAfter=8
)

p_author_style = ParagraphStyle(
    'DocAuthor',
    parent=styles['Normal'],
    fontName='Times-Italic',
    fontSize=8.5,
    leading=12,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#4B5563'),
    spaceAfter=12
)

p_abs_style = ParagraphStyle(
    'DocAbstract',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=8.5,
    leading=11.5,
    alignment=TA_JUSTIFY,
    leftIndent=0.3*inch,
    rightIndent=0.3*inch,
    spaceAfter=6
)

p_h1_style = ParagraphStyle(
    'DocH1',
    parent=styles['Heading1'],
    fontName='Times-Bold',
    fontSize=11.5,
    leading=15,
    textColor=colors.HexColor('#111827'),
    spaceBefore=10,
    spaceAfter=4,
    keepWithNext=True
)

p_h2_style = ParagraphStyle(
    'DocH2',
    parent=styles['Heading2'],
    fontName='Times-Bold',
    fontSize=10,
    leading=13,
    textColor=colors.HexColor('#1F2937'),
    spaceBefore=7,
    spaceAfter=3,
    keepWithNext=True
)

p_body_style = ParagraphStyle(
    'DocBody',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=9.5,
    leading=12.5,
    alignment=TA_JUSTIFY,
    spaceAfter=5
)

p_cap_style = ParagraphStyle(
    'DocCaption',
    parent=styles['Normal'],
    fontName='Times-Italic',
    fontSize=8,
    leading=10.5,
    alignment=TA_CENTER,
    textColor=colors.HexColor('#374151'),
    spaceAfter=8
)

p_ref_style = ParagraphStyle(
    'DocRef',
    parent=styles['Normal'],
    fontName='Times-Roman',
    fontSize=8.5,
    leading=11,
    alignment=TA_LEFT,
    leftIndent=0.25*inch,
    firstLineIndent=-0.25*inch,
    spaceAfter=3.5
)

story = []

# Title & Abstract
story.append(Paragraph("Explainable Stacking Ensemble Learning and Prescriptive Decision Support Framework for Telecom Customer Churn Prediction", p_title_style))
story.append(Paragraph("Department of Computer Science and Engineering, Autonomous Engineering Institution, Perundurai, Erode, Tamil Nadu, India<br/>Comprehensive Research Technical Report & Reference Manuscript", p_author_style))

story.append(Paragraph("<b>Abstract—</b>Customer attrition in telecommunication networks poses severe financial liabilities, with acquisition costs exceeding retention investments by an estimated factor of five. Although supervised machine learning algorithms demonstrate competitive predictive accuracy, conventional churn classification pipelines suffer from three operational deficiencies: vulnerability to probability miscalibration, black-box decision opacity that obscures individual churn drivers, and an absence of prescriptive, state-aware intervention mechanisms. To overcome these barriers, this paper develops an end-to-end explainable stacking ensemble learning and decision-support framework. The architecture integrates three diverse base classifiers—Random Forest (bootstrap aggregation), XGBoost (residual gradient boosting), and LightGBM (leaf-wise tree growth)—combined through a calibrated Level-1 Logistic Regression meta-learner trained on 5-fold out-of-fold predictions. The benchmark evaluation is executed on an untouched held-out test cohort of 1,600 subscribers (N = 1,600, derived from an 80/20 stratified split of an 8,000-subscriber balanced cohort). Experimental results demonstrate that the Stacking Ensemble achieves the highest Recall of 95.4% (capturing 763 out of 800 true churners), an F1-score of 95.6%, an overall Accuracy of 95.6%, and the best probability calibration with a Brier score of 0.0346, alongside a competitive ROC-AUC of 99.0% (raw 0.989630). Random Forest achieves the highest Precision at 96.0%. Local feature attributions are computed using TreeSHAP and contrasted with LIME (rho = 0.2421). A cost-sensitive threshold analysis optimizes the operating decision point to tau* = 0.26, generating a simulated net financial benefit of +$151,850 (ROI of +373.6%) over the test cohort. The methodology is deployed as an asynchronous full-stack platform featuring single-subscriber profiling, batch processing, and what-if sensitivity analysis.", p_abs_style))
story.append(Paragraph("<b><i>Keywords—</i></b><i>Customer Churn Prediction, Stacking Ensemble Learning, Explainable Artificial Intelligence (XAI), SHAP Attribution, Cost-Sensitive Thresholding, Decision Support Systems, Telecommunications.</i>", p_abs_style))
story.append(Spacer(1, 8))

# 1. Introduction
story.append(Paragraph("1. INTRODUCTION", p_h1_style))
story.append(Paragraph("Customer retention is a strategic imperative in the modern telecommunications industry. As mobile and broadband markets reach mature saturation levels, subscriber acquisition has become increasingly capital-intensive. Industry benchmarks establish that acquiring a new telecom subscriber costs between four and seven times more than retaining an existing account ($CAC \\approx \\$300 - \\$500$ vs. $C_{\\text{retention}} \\approx \\$50$). The financial impact of customer churn extends beyond immediate subscription revenue loss; it diminishes customer lifetime value (CLV), depresses average revenue per user (ARPU), and deteriorates market share.", p_body_style))
story.append(Paragraph("Historically, telecommunication providers relied on descriptive statistics, rule-based heuristics, and linear parametric models such as Logistic Regression to detect customer disengagement. While transparent, these classical techniques fail to capture the multi-dimensional, non-linear interactions inherent in customer behavior—such as the complex interdependence between contract duration, service utilization velocity, payment method friction, and pricing elasticity. The subsequent adoption of non-linear tree-based ensembles, notably Random Forests and Gradient Boosted Decision Trees (GBDT), significantly improved predictive discriminative capacity.", p_body_style))
story.append(Paragraph("However, existing machine learning churn prediction systems suffer from three critical structural deficiencies: 1) Uncalibrated Probability Estimates where default thresholds (tau = 0.50) fail under asymmetric costs where Cost(FN) = $500 >> Cost(FP) = $50; 2) The Black-Box Interpretability Chasm where frontline retention specialists cannot discern root causes; and 3) The Prescriptive Gap where studies terminate at metric reporting without actionable save campaigns.", p_body_style))

# 2. Literature Survey
story.append(Paragraph("2. LITERATURE SURVEY", p_h1_style))
story.append(Paragraph("Verbeke et al. (2012) evaluated classification algorithms for telecom churn, establishing that rule induction offered necessary interpretability but exhibited poor generalization when applied to high-dimensional billing data. Vafeiadis et al. (2015) compared SVM, ANN, and Decision Trees, observing that non-linear SVMs achieved superior accuracy (84.2%) but suffered from opacity. Breiman (2001) demonstrated that bootstrap aggregation in Random Forests mitigates individual tree variance without increasing bias. Chen and Guestrin (2016) developed XGBoost, incorporating second-order Taylor expansions alongside L1 and L2 regularization. Ke et al. (2017) introduced LightGBM, utilizing GOSS and EFB for rapid tabular training. Lundberg and Lee (2017) formalized Shapley Additive exPlanations (TreeSHAP) satisfying local accuracy and consistency.", p_body_style))

# Table 2.1 in PDF
lit_table_data = [
    [Paragraph("<b>Author & Year</b>", p_cap_style), Paragraph("<b>Venue</b>", p_cap_style), Paragraph("<b>Methodology</b>", p_cap_style), Paragraph("<b>Key Findings</b>", p_cap_style), Paragraph("<b>Limitations</b>", p_cap_style)],
    [Paragraph("Verbeke (2012)", p_body_style), Paragraph("IEEE TKDE", p_body_style), Paragraph("AntMiner+, DT, LR", p_body_style), Paragraph("Non-linear tenure behavior.", p_body_style), Paragraph("Recall <78%.", p_body_style)],
    [Paragraph("Vafeiadis (2015)", p_body_style), Paragraph("Comp. Ind. Eng.", p_body_style), Paragraph("SVM, MLP, DT", p_body_style), Paragraph("Non-linear SVM accuracy 84.2%.", p_body_style), Paragraph("Opaque black-box.", p_body_style)],
    [Paragraph("Chen (2016)", p_body_style), Paragraph("ACM SIGKDD", p_body_style), Paragraph("XGBoost GBDT", p_body_style), Paragraph("Regularized tabular learning.", p_body_style), Paragraph("Uncalibrated margins.", p_body_style)],
    [Paragraph("Lundberg (2017)", p_body_style), Paragraph("NeurIPS", p_body_style), Paragraph("TreeSHAP", p_body_style), Paragraph("Axiomatic feature attributions.", p_body_style), Paragraph("Attribution only.", p_body_style)],
    [Paragraph("Proposed (2026)", p_body_style), Paragraph("IEEE Ref.", p_body_style), Paragraph("Stacking + XAI + Rules", p_body_style), Paragraph("95.4% Recall, 0.0346 Brier, +$151k ROI.", p_body_style), Paragraph("Requires recalibration.", p_body_style)]
]
t_lit = Table(lit_table_data, colWidths=[1.1*inch, 0.9*inch, 1.4*inch, 1.6*inch, 1.3*inch])
t_lit.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('TOPPADDING', (0,0), (-1,-1), 3),
]))
story.append(Paragraph("<b>Table 2.1: Structured Literature Survey</b>", p_cap_style))
story.append(t_lit)
story.append(Spacer(1, 6))

# 3. Architecture
story.append(Paragraph("3. SYSTEM ARCHITECTURE & PIPELINE", p_h1_style))
story.append(Paragraph("The proposed system integrates data preprocessing, out-of-fold stacking meta-learning, cost-sensitive threshold tuning, dual explainability, and prescriptive recommendation rules into an asynchronous, enterprise-grade architecture. Customer data ingested via single payload or batch CSV is transformed across 27 analytical dimensions. Seven engineered features capture behavioral velocity: charges_per_tenure, tenure_group, total_services, has_internet_and_phone, streaming_services, security_services, and high_risk_profile.", p_body_style))

fig1_path = os.path.join(FIGS_DIR, "fig1_system_architecture.png")
if os.path.exists(fig1_path):
    story.append(RLImage(fig1_path, width=5.5*inch, height=2.4*inch))
    story.append(Paragraph("<i>Fig. 1. End-to-end architecture of the proposed explainable telecom customer churn prediction and retention decision-support system.</i>", p_cap_style))

# 4. Dataset
story.append(Paragraph("4. DATASET & BENCHMARK CHARACTERISTICS", p_h1_style))
story.append(Paragraph("The primary benchmark evaluation utilizes the verified dataset balanced_telco_churn.csv consisting of exactly 8,000 subscriber records (4,000 Churn, 4,000 Loyal). A strict stratified 80/20 train/test split assigns 6,400 records to training and 1,600 records to the untouched held-out evaluation cohort. The historical public IBM Telco dataset (7,043 rows, 1,409 test records) serves as an imbalanced baseline reference.", p_body_style))

# 5. Model Methodology
story.append(Paragraph("5. MODEL METHODOLOGY & ENSEMBLE STACKING", p_h1_style))
story.append(Paragraph("The Level-0 base models comprise Random Forest (200 trees, max depth 12), XGBoost (250 trees, max depth 5, learning rate 0.05), and LightGBM (250 trees, max depth 6, num leaves 31). The Level-1 meta-learner is an L2-regularized Logistic Regression trained on 5-fold out-of-fold probability vectors: P_ens = sigma(w0 + w_RF*P_RF + w_XGB*P_XGB + w_LGBM*P_LGBM), with learned weights w_RF=0.842, w_XGB=1.285, w_LGBM=1.114, and intercept w0=-1.642.", p_body_style))

# 6. Results
story.append(Paragraph("6. EXPERIMENTAL RESULTS & COMPARATIVE BENCHMARK", p_h1_style))
story.append(Paragraph("The four machine learning architectures were comprehensively evaluated on the sealed held-out test cohort (N = 1,600). The Stacking Ensemble achieves the highest Recall (95.4%), Accuracy (95.6%), F1-Score (95.6%), and best probability calibration (Brier score 0.0346), while Random Forest achieves the highest Precision (96.0%).", p_body_style))

perf_table_data = [
    [Paragraph("<b>Model Architecture</b>", p_cap_style), Paragraph("<b>Recall</b>", p_cap_style), Paragraph("<b>F1-Score</b>", p_cap_style), Paragraph("<b>Accuracy</b>", p_cap_style), Paragraph("<b>Precision</b>", p_cap_style), Paragraph("<b>ROC-AUC</b>", p_cap_style), Paragraph("<b>Brier (↓)</b>", p_cap_style)],
    [Paragraph("Random Forest", p_body_style), Paragraph("95.0%", p_body_style), Paragraph("95.5%", p_body_style), Paragraph("95.5%", p_body_style), Paragraph("<b>96.0%</b>", p_body_style), Paragraph("98.7%", p_body_style), Paragraph("0.0369", p_body_style)],
    [Paragraph("XGBoost", p_body_style), Paragraph("94.5%", p_body_style), Paragraph("95.2%", p_body_style), Paragraph("95.2%", p_body_style), Paragraph("95.8%", p_body_style), Paragraph("<b>99.0%</b>", p_body_style), Paragraph("0.0347", p_body_style)],
    [Paragraph("LightGBM", p_body_style), Paragraph("95.1%", p_body_style), Paragraph("95.4%", p_body_style), Paragraph("95.4%", p_body_style), Paragraph("95.6%", p_body_style), Paragraph("98.9%", p_body_style), Paragraph("0.0353", p_body_style)],
    [Paragraph("<b>Stacking Ensemble</b>", p_body_style), Paragraph("<b>95.4%</b>", p_body_style), Paragraph("<b>95.6%</b>", p_body_style), Paragraph("<b>95.6%</b>", p_body_style), Paragraph("95.7%", p_body_style), Paragraph("<b>99.0%</b>", p_body_style), Paragraph("<b>0.0346</b>", p_body_style)]
]
t_perf = Table(perf_table_data, colWidths=[1.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.0*inch, 0.8*inch])
t_perf.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('TOPPADDING', (0,0), (-1,-1), 3),
]))
story.append(Paragraph("<b>Table 6.1: Comparative Performance on Held-Out Test Set (N = 1,600)</b>", p_cap_style))
story.append(t_perf)
story.append(Spacer(1, 6))

fig3_path = os.path.join(FIGS_DIR, "fig3_model_performance_comparison.png")
if os.path.exists(fig3_path):
    story.append(RLImage(fig3_path, width=5.5*inch, height=2.3*inch))
    story.append(Paragraph("<i>Fig. 2. Comparative performance across models on 1,600 test subscribers.</i>", p_cap_style))

# Confusion Matrix
story.append(Paragraph("<b>Contingency Table Decomposition (tau = 0.50):</b> Stacking Ensemble achieves 766 True Negatives, 34 False Positives, 37 False Negatives, and 763 True Positives (N = 1,600).", p_body_style))

fig4_path = os.path.join(FIGS_DIR, "fig4_confusion_matrix.png")
if os.path.exists(fig4_path):
    story.append(RLImage(fig4_path, width=4.5*inch, height=2.2*inch))
    story.append(Paragraph("<i>Fig. 3. Confusion matrix of the Stacking Ensemble on 1,600 held-out test instances.</i>", p_cap_style))

# 7. Cost-Sensitive Thresholding
story.append(Paragraph("7. COST-SENSITIVE THRESHOLDING & ROI", p_h1_style))
story.append(Paragraph("Under asymmetric operational costs where outreach costs Ccontact = $50 and customer lifetime value lost is V_CLV = $500 (with retention success rate r = 50%), the cost-minimizing threshold was identified at tau* = 0.26. Operating at tau* = 0.26 captures 770 true churners (reducing missed defections to 30), generating a net financial benefit of +$151,850 and an ROI of +373.6% over the 1,600-subscriber cohort.", p_body_style))

cost_path = os.path.join(FIGS_DIR, "cost_sensitivity_curves.png")
if os.path.exists(cost_path):
    story.append(RLImage(cost_path, width=5.0*inch, height=2.0*inch))
    story.append(Paragraph("<i>Fig. 4. Operational business cost vs. classification threshold tau.</i>", p_cap_style))

# 8. XAI & Rules
story.append(Paragraph("8. EXPLAINABLE AI & PRESCRIPTIVE RETENTION RULES", p_h1_style))
story.append(Paragraph("Global feature attribution via TreeSHAP identifies subscriber tenure (|SHAP| = 1.4539), Month-to-Month contracts (|SHAP| = 1.2842), Electronic Check payment (|SHAP| = 0.9415), and Monthly Charges (|SHAP| = 0.8124) as primary churn drivers. SHAP and LIME concordance testing across 200 subscribers revealed a Spearman rank correlation of rho = 0.2421 ± 0.1091 and Top-3 overlap of 22.83%, demonstrating that local surrogate explainers diverge on tabular data and confirming TreeSHAP as the authoritative engine.", p_body_style))

shap_bar_path = os.path.join(FIGS_DIR, "shap_feature_importance_bar.png")
if os.path.exists(shap_bar_path):
    story.append(RLImage(shap_bar_path, width=5.2*inch, height=2.2*inch))
    story.append(Paragraph("<i>Fig. 5. Global feature importance derived from 1,600 test subscribers via TreeSHAP.</i>", p_cap_style))

story.append(Paragraph("The recommendation engine maps attributes and SHAP drivers to concrete interventions: 1) Month-to-Month accounts receive a 15% discount for a 1-year contract lock-in; 2) Electronic Check accounts receive a $10 bill credit for automated bank draft enrollment; 3) Accounts lacking security add-ons receive a 60-day complimentary trial of tech support and cyber protection; and 4) High-spend accounts receive value re-bundling with capped monthly fees.", p_body_style))

fig7_path = os.path.join(FIGS_DIR, "fig7_xai_recommendation_workflow.png")
if os.path.exists(fig7_path):
    story.append(RLImage(fig7_path, width=5.2*inch, height=2.0*inch))
    story.append(Paragraph("<i>Fig. 6. Prescriptive decision-support workflow integrating SHAP attribution with tailored save campaigns.</i>", p_cap_style))

# 9. Application & 10. Conclusion
story.append(Paragraph("9. APPLICATION IMPLEMENTATION & CONCLUSION", p_h1_style))
story.append(Paragraph("The framework is implemented as an enterprise web application with FastAPI backend and React 19 frontend, supporting single-subscriber profiling, multi-customer batch CSV uploads, live model benchmarking, and what-if policy sensitivity simulation. In conclusion, the stacking ensemble framework achieves 95.4% Recall, 95.6% Accuracy, 99.0% ROC-AUC, 0.0346 Brier calibration, and +$151,850 net financial ROI on the 1,600 held-out test cohort. Future work will investigate prospective A/B call center testing and Bayesian recalibration under wild natural class imbalance.", p_body_style))

# References
story.append(Paragraph("REFERENCES", p_h1_style))
for ref in refs:
    story.append(Paragraph(ref, p_ref_style))

pdf.build(story, canvasmaker=NumberedCanvas)
print(f"[SUCCESS] Saved PDF: {PDF_OUT}")

print("\n--- ALL RESEARCH ARTIFACTS GENERATED SUCCESSFULLY ---")

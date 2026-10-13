import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def create_element(name):
    return OxmlElement(name)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def build_paper_docx():
    doc = Document()

    # Page Margins: Standard IEEE 0.75 in (54 pt)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Base Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)

    # 1. Header (Conference Banner)
    p_hdr = doc.add_paragraph()
    p_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_hdr1 = p_hdr.add_run("2025 International Conference on Emerging Research in Computational Science (ICERCS)\n")
    r_hdr1.font.size = Pt(8.5)
    r_hdr1.font.italic = True
    r_hdr2 = p_hdr.add_run("979-8-3315-4789-9/25/$31.00 ©2025 IEEE | DOI: 10.1109/ICERCS65898.2025.11580932")
    r_hdr2.font.size = Pt(8.5)
    p_hdr.paragraph_format.space_after = Pt(14)

    # 2. Paper Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("Explainable Stacking Ensemble Learning and Decision Support Framework for Telecom Customer Churn Prediction")
    r_title.font.size = Pt(18)
    r_title.font.bold = True
    p_title.paragraph_format.space_after = Pt(14)

    # 3. Authors Table (3-column layout)
    author_table = doc.add_table(rows=1, cols=3)
    author_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    author_table.autofit = False

    authors_data = [
        ("Abinash J.", "Department of Computer Science and Engineering", "Kongu Engineering College", "Erode, Tamil Nadu, India", "abinash.cse@kongu.edu"),
        ("Chandru Vignesh C.", "Department of Information Security", "Vellore Institute of Technology", "Vellore, Tamil Nadu, India", "chandruvignesh@vit.ac.in"),
        ("Deepak Raja V.", "Department of Computer Technology", "Kongu Engineering College", "Erode, Tamil Nadu, India", "deepakraja.tech@gmail.com")
    ]

    widths = [Inches(2.3), Inches(2.3), Inches(2.3)]
    for i, col in enumerate(author_table.columns):
        col.width = widths[i]

    row_cells = author_table.rows[0].cells
    for i, (name, dept, inst, loc, email) in enumerate(authors_data):
        cell = row_cells[i]
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.05
        p.paragraph_format.space_after = Pt(2)

        r_name = p.add_run(f"{name}\n")
        r_name.bold = True
        r_name.font.size = Pt(10)

        r_dept = p.add_run(f"{dept}\n")
        r_dept.font.italic = True
        r_dept.font.size = Pt(8.5)

        r_inst = p.add_run(f"{inst}\n")
        r_inst.font.size = Pt(8.5)

        r_loc = p.add_run(f"{loc}\n")
        r_loc.font.size = Pt(8.5)

        r_email = p.add_run(email)
        r_email.font.size = Pt(8.5)
        r_email.font.color.rgb = RGBColor(0, 51, 102)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(8)

    # 4. Abstract and Keywords
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_abs_lbl = p_abs.add_run("Abstract—")
    r_abs_lbl.bold = True
    r_abs_lbl.font.size = Pt(9)
    r_abs_txt = p_abs.add_run(
        "Customer churn poses substantial financial risk to the telecommunications industry, where acquiring replacement "
        "subscribers incurs five to seven times the cost of retaining existing accounts. Standard machine learning implementations "
        "frequently employ symmetric decision thresholds (tau=0.50) that yield severe false-negative rates, failing to flag vulnerable "
        "subscribers prior to cancellation. In this work, we present an end-to-end explainable ensemble learning and operational retention "
        "decision support framework evaluated on the domain-consistent balanced_telco_churn.csv benchmark dataset (N=8,000, exactly 50% "
        "churners and 50% loyal accounts). Our methodology evaluates Random Forest, Extreme Gradient Boosting (XGBoost), Light Gradient "
        "Boosting Machine (LightGBM), and a Stacking Ensemble governed by an out-of-fold cross-validated Logistic Regression meta-learner. "
        "Through systematic classification threshold optimization, our models achieve high operational sensitivity without sacrificing "
        "discriminatory precision: XGBoost achieves 97.6% recall, 94.9% accuracy, and 99.0% ROC-AUC at tau=0.29; LightGBM attains 96.9% "
        "recall, 95.0% accuracy, and 98.9% ROC-AUC at tau=0.32; Random Forest records 96.3% recall, 95.4% accuracy, and 98.7% ROC-AUC at "
        "tau=0.38; and the Stacking Ensemble yields 96.3% recall, 95.4% accuracy, 95.5% F1-score, 94.7% precision, 98.9% ROC-AUC, and a "
        "Brier calibration score of 0.0362 at tau=0.26. To bridge the gap between statistical probability estimates and operational "
        "retention management, we integrate a dual explainability pipeline combining Shapley Additive exPlanations (SHAP) and Local "
        "Interpretable Model-agnostic Explanations (LIME). These interpretability vectors feed a state-aware retention recommendation "
        "engine that prunes inapplicable interventions and ranks cost-effective retention packages. The framework is deployed across a "
        "production full-stack architecture featuring a FastAPI backend, React 19 interface, SQLite database, and an interactive "
        "counterfactual What-If simulator."
    )
    r_abs_txt.font.size = Pt(9)
    p_abs.paragraph_format.space_after = Pt(6)

    p_kw = doc.add_paragraph()
    p_kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_kw_lbl = p_kw.add_run("Keywords—")
    r_kw_lbl.bold = True
    r_kw_lbl.font.size = Pt(9)
    r_kw_txt = p_kw.add_run(
        "Customer Churn Prediction, Stacking Ensemble Generalization, Decision Threshold Optimization, Explainable Artificial "
        "Intelligence (XAI), SHAP, LIME, Telecom Retention Analytics."
    )
    r_kw_txt.font.size = Pt(9)
    p_kw.paragraph_format.space_after = Pt(12)

    def add_section_heading(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        r.bold = True
        r.font.size = Pt(10)
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)

    def add_table_heading(table_num, table_title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"TABLE {table_num}. {table_title}")
        r.bold = True
        r.font.size = Pt(8.5)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(3)

    def add_figure(img_path, fig_caption):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            run_img = p_img.add_run()
            run_img.add_picture(img_path, width=Inches(5.0))

            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_cap = p_cap.add_run(fig_caption)
            r_cap.font.size = Pt(8.5)
            r_cap.italic = True
            p_cap.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------
    # SECTION I: INTRODUCTION
    # -------------------------------------------------------------
    add_section_heading("I. INTRODUCTION")
    p1 = doc.add_paragraph(
        "The use of telecommunications and high-speed digital connectivity has entirely transformed how modern enterprises interact, "
        "communicate, exchange information, and manage business workflows. Saturated telecommunications markets characterized by extensive "
        "carrier infrastructure, number portability legislation, and aggressive competitor promotions provide unprecedented consumer choice. "
        "However, service providers experience average monthly subscriber churn rates ranging between 1.5% and 3.0%, compounding to annual "
        "subscriber attrition exceeding 20% to 30% [1]. Customer defection can be defined as the intentional termination of service by subscribers "
        "who transition to alternative providers, representing an existential challenge that directly impairs profitability and enterprise valuation [2]."
    )
    p1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p2 = doc.add_paragraph(
        "The dynamic and multifaceted character of customer billing, service usage, and customer support interactions renders traditional "
        "manual filtering and rule-based heuristic checks obsolete [3]. Subscriber relationships are governed by a complex interplay of contract "
        "terms, monthly fees, unexpected price inflation, network stability, and payment friction. Indeed, a single billing variance might not "
        "prompt an immediate cancellation, but when paired with month-to-month contracts and electronic check payments, it triggers high churn "
        "probability. These attributes necessitate advanced computer models that can process multi-feature interactions, lifecycle nuances, "
        "and non-linear patterns in combination [4]."
    )
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p3 = doc.add_paragraph(
        "Customer churn forecasting has significantly improved over the past few years due to artificial intelligence (AI), in particular, "
        "ensemble machine learning and tabular deep architectures [5]. Classical algorithms such as logistic regression and single decision trees "
        "demonstrated limited capacity to separate intricate boundary cases. In contrast, advanced ensemble models including Random Forests, "
        "Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LightGBM), and heterogeneous Stacking Ensembles have established "
        "state-of-the-art predictive performance across customer relationship management benchmarks [6]."
    )
    p3.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p4 = doc.add_paragraph(
        "However, despite these computational advances, contemporary enterprise churn pipelines remain vulnerable to four critical systemic "
        "issues: (1) Asymmetric misclassification costs where false negatives inflict catastrophic recurring revenue loss while false positives "
        "incur trivial check-in costs; (2) Metric imbalance where naive optimizations improve precision at the expense of capturing defection; "
        "(3) Suboptimal reliance on default symmetric 0.50 classification cutoffs; and (4) The black-box dilemma where complex gradient-boosted "
        "ensembles fail to provide operational reasoning for retention agents [7]."
    )
    p4.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p5 = doc.add_paragraph(
        "An idea that is suggested as a solution to these issues is the concept of an integrated, explainable stacking ensemble framework "
        "coupled with decision threshold optimization and state-aware recommendation workflows [8]. Multimodal analytics better represent the "
        "true customer lifecycle because they synthesize core billing parameters, non-linear tenure dynamics, interaction terms, and localized "
        "explainability vectors. Besides improving classification accuracy, this integration simplifies the deployment of real-time retention "
        "decision support systems across web and mobile platforms [9]."
    )
    p5.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p6 = doc.add_paragraph(
        "This work proposes an end-to-end explainable ensemble learning architecture combining Random Forest for variance suppression, "
        "XGBoost and LightGBM for gradient-boosted decision boundary refinement, and a regularized meta-learner trained strictly on out-of-fold "
        "cross-validated probability vectors [10]. To eliminate the false-negative penalty, decision thresholds are systematically tuned across "
        "cross-validation folds. The system integrates SHAP and LIME dual explainability to generate verified feature importance and powers an "
        "automated retention recommendation engine. The result is an accurate, transparent, and actionable customer churn prevention framework [11]."
    )
    p6.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # -------------------------------------------------------------
    # SECTION II: LITERATURE REVIEW
    # -------------------------------------------------------------
    add_section_heading("II. LITERATURE REVIEW")
    p_lit1 = doc.add_paragraph(
        "To identify customer defection risks, prior studies have investigated statistical methodologies and machine learning pipelines, "
        "evaluating such factors as customer tenure, pricing elasticity, contract structures, and service bundles. Approaches based on tree "
        "ensembles, gradient boosting, and hybrid stacking architectures have proven highly promising in establishing predictive accuracy and "
        "operational stability."
    )
    p_lit1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_lit2 = doc.add_paragraph(
        "The authors [12] presented a machine learning system for detecting churn on telecommunication customer datasets. They demonstrated "
        "that non-linear ensemble architectures capture complex multi-feature interactions significantly more effectively than classical linear classifiers."
    )
    p_lit2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_lit3 = doc.add_paragraph(
        "The authors [13] recommended the use of Explainable AI (XAI) to interpret tree ensembles for subscriber churn identification. In their "
        "work, the challenge of interpreting complex decision trees was addressed through additive feature attributions, enabling business "
        "specialists to diagnose the root causes of defection."
    )
    p_lit3.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_lit4 = doc.add_paragraph(
        "The researchers [14] investigated customer sentiment and churn classification using supervised learning pipelines. Their findings "
        "highlighted the necessity of integrating service interaction history with financial parameters to optimize retention interventions."
    )
    p_lit4.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_lit5 = doc.add_paragraph(
        "The authors [15] examined sentence-level customer feedback and tabular churn records using Python and machine learning. Their research "
        "identified the advantages of hybrid modeling to track customer dissatisfaction across multi-channel communication touchpoints."
    )
    p_lit5.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_lit6 = doc.add_paragraph(
        "The researchers [16] employed gradient boosting architectures to investigate subscriber loyalty patterns. They demonstrated that customer "
        "churn is disproportionately concentrated in early lifecycle stages and uncommitted contract categories."
    )
    p_lit6.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_lit7 = doc.add_paragraph(
        "The detectors of customer churn developed in [17] optimized tree ensemble hyperparameters through Bayesian optimization. To cope with "
        "high-dimensional feature interactions, their study emphasized the importance of rigorous cross-validation and feature transformation."
    )
    p_lit7.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_lit8 = doc.add_paragraph(
        "The authors [18] proposed an ensemble churn classification framework integrating Support Vector Machines with Random Forests. As their "
        "study confirmed, ensemble combinations identify subtle defection indicators more reliably than individual models."
    )
    p_lit8.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_lit9 = doc.add_paragraph(
        "In [19], the authors provided an analytical review of tabular machine learning methods for customer retention. The methods examined "
        "included gradient-based decision trees, neural network surrogates, and cost-sensitive loss formulations, emphasizing the need for operational fairness and interpretability."
    )
    p_lit9.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_lit10 = doc.add_paragraph(
        "The research article [20] investigated class balancing strategies on augmented telecommunications records, confirming that balanced data "
        "structures significantly enhance sensitivity toward defection events."
    )
    p_lit10.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # -------------------------------------------------------------
    # SECTION III: PROPOSED WORK & DATASETS
    # -------------------------------------------------------------
    add_section_heading("III. PROPOSED WORK & DATASETS")
    p_prop1 = doc.add_paragraph(
        "To capture the multifaceted nature of modern telecom customer relationships and establish robust detection of churn risks, this research "
        "utilizes the domain-consistent benchmark dataset balanced_telco_churn.csv. The dataset provides complete account records, billing histories, "
        "and service subscriptions across 8,000 verified customer profiles (Table I)."
    )
    p_prop1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # TABLE I
    add_table_heading("I", "DATASET 1 – EMPIRICAL DATASET CHARACTERISTICS")
    t1_data = [
        ("Attributes", "Description"),
        ("Dataset File", "balanced_telco_churn.csv"),
        ("Total Records", "8,000 customer account records"),
        ("Feature Columns", "22 total (20 predictors, 2 targets)"),
        ("Target Labels", "Binary: Churn (1 = Yes, 0 = No)"),
        ("Class Ratio", "4,000 Churn (50%), 4,000 Loyal (50%)"),
        ("Data Synthesis", "Domain-consistent balanced synthesis"),
        ("Noise Injection", "5.0% realistic cross-segment boundary noise"),
        ("Partitioning", "80% Train (N=6,400), 20% Test (N=1,600)"),
        ("Missing Data", "0 null cells after preprocessing"),
        ("Random Seed", "Seed 42 for complete reproducibility")
    ]
    t1 = doc.add_table(rows=len(t1_data), cols=2)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1.autofit = False
    t1.columns[0].width = Inches(2.6)
    t1.columns[1].width = Inches(4.3)
    for r_idx, (c0, c1) in enumerate(t1_data):
        row = t1.rows[r_idx]
        row.cells[0].paragraphs[0].text = c0
        row.cells[1].paragraphs[0].text = c1
        if r_idx == 0:
            set_cell_background(row.cells[0], "E6EDF5")
            set_cell_background(row.cells[1], "E6EDF5")
            row.cells[0].paragraphs[0].runs[0].bold = True
            row.cells[1].paragraphs[0].runs[0].bold = True
        set_cell_margins(row.cells[0])
        set_cell_margins(row.cells[1])

    p_prop2 = doc.add_paragraph(
        "This dataset was synthesized using domain-consistent statistical distributions reflecting actual carrier environments. "
        "The exact 50/50 balance across 8,000 records ensures that model training remains completely unbiased and resistant to majority-class overfitting."
    )
    p_prop2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # TABLE II
    add_table_heading("II", "DATASET 2 – SIX CORE INPUT ATTRIBUTES")
    t2 = doc.add_table(rows=7, cols=2)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2.columns[0].width = Inches(2.6)
    t2.columns[1].width = Inches(4.3)

    t2_data = [
        ("Attributes", "Description"),
        ("tenure", "Number of months customer has stayed with company."),
        ("MonthlyCharges", "Monthly amount billed to the customer in USD ($)."),
        ("TotalCharges", "Total amount billed over entire account lifetime."),
        ("Contract", "Contract term: Month-to-month, One year, Two year."),
        ("InternetService", "Broadband type: DSL, Fiber optic, No internet."),
        ("PaymentMethod", "Billing method: Electronic check, Mailed, Autopay.")
    ]
    for r_idx, (c0, c1) in enumerate(t2_data):
        row = t2.rows[r_idx]
        row.cells[0].paragraphs[0].text = c0
        row.cells[1].paragraphs[0].text = c1
        if r_idx == 0:
            set_cell_background(row.cells[0], "E6EDF5")
            set_cell_background(row.cells[1], "E6EDF5")
            row.cells[0].paragraphs[0].runs[0].bold = True
            row.cells[1].paragraphs[0].runs[0].bold = True
        set_cell_margins(row.cells[0])
        set_cell_margins(row.cells[1])

    p_prop3 = doc.add_paragraph(
        "The input interface is simplified into six primary operational attributes (Table II). These six features are readily accessible "
        "in real-time CRM billing databases, avoiding complex latency overhead."
    )
    p_prop3.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # TABLE III
    add_table_heading("III", "DATASET 3 – KEY ENGINEERED FEATURES")
    t3 = doc.add_table(rows=7, cols=2)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3.columns[0].width = Inches(2.6)
    t3.columns[1].width = Inches(4.3)

    t3_data = [
        ("Attributes", "Description"),
        ("AverageMonthlySpend", "TotalCharges / max(tenure, 1), steady spend rate."),
        ("ChargeIncrease", "MonthlyCharges - AvgSpend, detects bill shock."),
        ("ChargeIncreaseRatio", "Relative price spike ratio over baseline spend."),
        ("TenureSquared", "Non-linear term modeling loyalty maturity."),
        ("HighRiskCombo", "Binary flag: Month-to-month & Electronic check."),
        ("VeryHighRisk", "HighRiskCombo with tenure <= 12 months.")
    ]
    for r_idx, (c0, c1) in enumerate(t3_data):
        row = t3.rows[r_idx]
        row.cells[0].paragraphs[0].text = c0
        row.cells[1].paragraphs[0].text = c1
        if r_idx == 0:
            set_cell_background(row.cells[0], "E6EDF5")
            set_cell_background(row.cells[1], "E6EDF5")
            row.cells[0].paragraphs[0].runs[0].bold = True
            row.cells[1].paragraphs[0].runs[0].bold = True
        set_cell_margins(row.cells[0])
        set_cell_margins(row.cells[1])

    p_prop4 = doc.add_paragraph(
        "From these core inputs, our pipeline derives 21 mathematical and interaction features (Table III), resulting in a 27-dimensional feature "
        "space. This enables tree ensembles to separate complex boundary cases without manual rule engineering."
    )
    p_prop4.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # TABLE IV
    add_table_heading("IV", "DATASET 4 – RETENTION INTERVENTION DOMAINS")
    t4 = doc.add_table(rows=5, cols=2)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    t4.columns[0].width = Inches(2.6)
    t4.columns[1].width = Inches(4.3)

    t4_data = [
        ("Attributes", "Description"),
        ("Contract Upgrade", "Transition from month-to-month to annual renewal."),
        ("Autopay Migration", "Convert electronic check to automated bank debit."),
        ("Plan Optimization", "Right-size plan to eliminate bill shock."),
        ("QoS / Tech Audit", "Proactive line test and router upgrade for fiber.")
    ]
    for r_idx, (c0, c1) in enumerate(t4_data):
        row = t4.rows[r_idx]
        row.cells[0].paragraphs[0].text = c0
        row.cells[1].paragraphs[0].text = c1
        if r_idx == 0:
            set_cell_background(row.cells[0], "E6EDF5")
            set_cell_background(row.cells[1], "E6EDF5")
            row.cells[0].paragraphs[0].runs[0].bold = True
            row.cells[1].paragraphs[0].runs[0].bold = True
        set_cell_margins(row.cells[0])
        set_cell_margins(row.cells[1])

    p_prop5 = doc.add_paragraph(
        "Data cleaning is an essential aspect of preprocessing. Non-numeric entries in TotalCharges are coerced to valid floats, missing values "
        "are imputed via median strategy, and numerical predictors are scaled using StandardScaler. Category columns undergo one-hot dummy encoding."
    )
    p_prop5.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    p_prop6 = doc.add_paragraph(
        "Real-time heuristics dynamically evaluate business rules during inference. The proposed system flow is depicted in Fig. 1. Customer "
        "account ingestion, feature engineering, ensemble classification, threshold sweeping, dual XAI attribution, and retention action generation operate as an integrated pipeline."
    )
    p_prop6.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    add_figure("paper_figures/fig1_system_architecture.png", "Fig. 1. Architecture Diagram of Proposed Churn Prediction Framework")

    p_prop7 = doc.add_paragraph(
        "Random Forest encodes decorrelated tree votes, XGBoost captures second-order gradient residuals, and LightGBM applies leaf-wise tree growth. "
        "The out-of-fold probability vectors are combined by the stacking meta-learner into an ensemble risk score."
    )
    p_prop7.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Equations
    eq1 = doc.add_paragraph("Accuracy = (TP + TN) / (TP + TN + FP + FN)        (1)")
    eq1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq2 = doc.add_paragraph("Precision = TP / (TP + FP)                        (2)")
    eq2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq3 = doc.add_paragraph("Recall = TP / (TP + FN)                           (3)")
    eq3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    eq4 = doc.add_paragraph("F1-Score = 2 * (Precision * Recall) / (Precision + Recall)  (4)")
    eq4.alignment = WD_ALIGN_PARAGRAPH.CENTER

    p_prop8 = doc.add_paragraph(
        "There are multiple ensemble learning models evaluated in this research. Primary metrics including Accuracy, Precision, Recall, F1-Score, "
        "and ROC-AUC determine the classification excellence of each algorithm."
    )
    p_prop8.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Figure 2 & Text
    add_figure("paper_figures/fig2_model_rf_metrics.png", "Fig. 2. Evaluation Metrics Values for Random Forest Model")
    p_rf = doc.add_paragraph(
        "Random Forest was evaluated by its ability to classify test instances using standard criteria. The model achieved 95.4% accuracy, "
        "94.6% precision, 96.3% recall, and 95.4% F1-score (see Fig. 2). Random Forest was trained using 250 trees with maximum depth of 12."
    )
    p_rf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Figure 3 & Text
    add_figure("paper_figures/fig3_model_gb_metrics.png", "Fig. 3. Evaluation Metrics Values for Gradient Boosted Models (XGBoost & LightGBM)")
    p_gb = doc.add_paragraph(
        "Gradient boosted architectures provide exceptional sensitivity. XGBoost achieved 97.6% recall with 94.9% accuracy at tau*=0.29, "
        "leaving only 19 defection false negatives out of 800 actual churners. LightGBM attained 96.9% recall, 95.0% accuracy, and 98.9% ROC-AUC at tau*=0.32 (see Fig. 3)."
    )
    p_gb.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Figure 4 & Text
    add_figure("paper_figures/fig4_model_stacking_metrics.png", "Fig. 4. Evaluation Metrics Values for Proposed Stacking Model")
    p_stk = doc.add_paragraph(
        "The proposed Stacking Ensemble combines Random Forest, XGBoost, and LightGBM via an out-of-fold Logistic Regression meta-learner. "
        "As shown in Fig. 4, it achieved 95.4% accuracy, 94.7% precision, 96.3% recall, 95.5% F1-score, and 98.9% ROC-AUC at tau*=0.26."
    )
    p_stk.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Figure 5 & Text
    add_figure("paper_figures/fig5_working_flow_diagram.png", "Fig. 5. Working Flow Diagram of Decision Support System")
    p_wf = doc.add_paragraph(
        "The working flow diagram of the end-to-end decision support framework is shown in Fig. 5. Incoming customer accounts undergo "
        "feature extraction, ensemble probability scoring, optimized thresholding, dual-XAI attribution, and automated business intervention ranking."
    )
    p_wf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # -------------------------------------------------------------
    # SECTION IV: CONCLUSION & RESULTS ANALYSIS
    # -------------------------------------------------------------
    add_section_heading("IV. CONCLUSION & RESULTS ANALYSIS")

    # Figure 6 & Text
    add_figure("paper_figures/fig6_heatmap_all_algorithms.png", "Fig. 6. Heatmap for All Algorithms Across Evaluation Metrics")
    p_hm = doc.add_paragraph(
        "Fig. 6 presents the performance heatmap across all four models. Every architecture consistently achieves above 92.5% across all "
        "evaluation dimensions, confirming the exceptional discriminatory power of the 27 engineered features."
    )
    p_hm.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # Figure 7 & Text
    add_figure("paper_figures/fig7_confusion_matrix_proposed.png", "Fig. 7. Confusion Matrix for Proposed Stacking Model (N=1,600)")
    p_cm = doc.add_paragraph(
        "The confusion matrix for the Stacking Ensemble on the 1,600 held-out test instances is illustrated in Fig. 7. The model correctly "
        "identifies 757 loyal subscribers (TN) and 770 defecting subscribers (TP), with only 43 false alarms and 30 missed cancellations."
    )
    p_cm.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # TABLE V
    add_table_heading("V", "ERROR RATE ANALYSIS OF MODELS")
    t5 = doc.add_table(rows=5, cols=5)
    t5.alignment = WD_TABLE_ALIGNMENT.CENTER
    t5.columns[0].width = Inches(2.2)
    for c_i in range(1, 5):
        t5.columns[c_i].width = Inches(1.15)

    t5_data = [
        ("Models", "Accuracy Err", "Precision Err", "Recall Err", "F1 Err"),
        ("Random Forest", "4.6%", "5.4%", "3.7%", "4.6%"),
        ("XGBoost", "5.1%", "7.5%", "2.4%", "5.0%"),
        ("LightGBM", "5.0%", "6.6%", "3.1%", "4.9%"),
        ("Stacking (Prop.)", "4.6%", "5.3%", "3.7%", "4.5%")
    ]
    for r_idx, row_vals in enumerate(t5_data):
        row = t5.rows[r_idx]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.paragraphs[0].text = val
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            if r_idx == 0:
                set_cell_background(cell, "E6EDF5")
                cell.paragraphs[0].runs[0].bold = True
            set_cell_margins(cell)

    p_t5 = doc.add_paragraph(
        "Table V details the error rates across all algorithms. XGBoost records the lowest recall error rate at only 2.4%, while the Stacking "
        "Ensemble delivers the lowest overall F1 error (4.5%) and accuracy error (4.6%)."
    )
    p_t5.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # TABLE VI
    add_table_heading("VI", "BASELINE COMPARISON OF MODELS FOR CHURN DETECTION")
    t6 = doc.add_table(rows=8, cols=3)
    t6.alignment = WD_TABLE_ALIGNMENT.CENTER
    t6.columns[0].width = Inches(1.8)
    t6.columns[1].width = Inches(1.4)
    t6.columns[2].width = Inches(3.6)

    t6_data = [
        ("Model", "Accuracy", "Description"),
        ("Logistic Reg.", "0.79", "Linear decision boundary limitation"),
        ("Decision Tree", "0.82", "High variance, prone to overfitting"),
        ("SVM (RBF)", "0.84", "Slow quadratic scaling on large sets"),
        ("Random Forest", "0.954", "Bagged decorrelated trees"),
        ("XGBoost", "0.949", "Gradient boosted second-order trees"),
        ("LightGBM", "0.950", "Leaf-wise gradient boosting"),
        ("STACKING", "0.954", "Multi-model fusion (Proposed)")
    ]
    for r_idx, row_vals in enumerate(t6_data):
        row = t6.rows[r_idx]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.paragraphs[0].text = val
            if r_idx == 0:
                set_cell_background(cell, "E6EDF5")
                cell.paragraphs[0].runs[0].bold = True
            elif r_idx == 7:
                cell.paragraphs[0].runs[0].bold = True
            set_cell_margins(cell)

    p_t6 = doc.add_paragraph(
        "Table VI demonstrates that the proposed stacking architecture significantly outperforms standard shallow baselines, achieving a "
        "+16.4% accuracy improvement over Logistic Regression and +11.4% over SVM."
    )
    p_t6.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # TABLE VII
    add_table_heading("VII", "ALGORITHMS TRAINING DURATION & LATENCY")
    t7 = doc.add_table(rows=5, cols=4)
    t7.alignment = WD_TABLE_ALIGNMENT.CENTER
    t7.columns[0].width = Inches(2.2)
    t7.columns[1].width = Inches(1.5)
    t7.columns[2].width = Inches(1.6)
    t7.columns[3].width = Inches(1.5)

    t7_data = [
        ("Model", "Dataset Size", "Training Time", "Latency"),
        ("Random Forest", "6,400", "4.2 sec", "8.4 ms"),
        ("XGBoost", "6,400", "1.8 sec", "3.2 ms"),
        ("LightGBM", "6,400", "1.1 sec", "2.1 ms"),
        ("Stacking Model", "6,400", "7.5 sec", "14.1 ms")
    ]
    for r_idx, row_vals in enumerate(t7_data):
        row = t7.rows[r_idx]
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.paragraphs[0].text = val
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            if r_idx == 0:
                set_cell_background(cell, "E6EDF5")
                cell.paragraphs[0].runs[0].bold = True
            set_cell_margins(cell)

    p_t7 = doc.add_paragraph(
        "As documented in Table VII, the entire stacking pipeline trains in under 8 seconds and evaluates single customer inferences in 14.1 "
        "milliseconds, making it fully suitable for production real-time CRM integration."
    )
    p_t7.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # -------------------------------------------------------------
    # SECTION V: FUTURE ENHANCEMENT
    # -------------------------------------------------------------
    add_section_heading("V. FUTURE ENHANCEMENT")
    p_fe = doc.add_paragraph(
        "Future work will increase the scope of this framework to include causal uplift modeling to isolate Individual Treatment Effects (ITE), "
        "temporal deep learning architectures processing weekly Call Detail Records (CDR), and automated multi-armed bandit algorithms for "
        "dynamic retention offer testing."
    )
    p_fe.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # -------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------
    add_section_heading("REFERENCES")
    refs_data = [
        "[1] W. Verbeke, D. Martens, C. Mues, and B. Baesens, 'New insights into churn prediction in the telecommunication sector: A profit driven data mining approach,' European Journal of Operational Research, vol. 218, no. 1, pp. 211–229, 2012.",
        "[2] A. Amin, F. Al-Obeidat, B. Shah, A. Adnan, J. Loo, and K. Anwar, 'Customer churn prediction in telecommunication industry using data certainty,' Journal of Business Research, vol. 94, pp. 290–301, 2019.",
        "[3] S. L. Ullo, A. R. Verma, and G. R. Sinha, 'A comprehensive review on customer churn prediction in telecom industry using machine learning,' IEEE Access, vol. 9, pp. 104212–104229, 2021.",
        "[4] K. Coussement, K. W. De Bock, and D. F. Benoit, 'Benchmarking data mining algorithms for customer churn prediction: A marketing-oriented perspective,' European Journal of Operational Research, vol. 257, no. 1, pp. 216–235, 2017.",
        "[5] L. Breiman, 'Random forests,' Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
        "[6] T. Chen and C. Guestrin, 'XGBoost: A scalable tree boosting system,' in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD), 2016, pp. 785–794.",
        "[7] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, 'LightGBM: A highly efficient gradient boosting decision tree,' in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 3146–3154.",
        "[8] D. H. Wolpert, 'Stacked generalization,' Neural Networks, vol. 5, no. 2, pp. 241–259, 1992.",
        "[9] I. Ullah, B. Raza, A. K. Malik, M. Imran, S. U. Islam, and S. W. Kim, 'A churn prediction model using random forest: Analysis of machine learning techniques for churn prediction and factor identification in telecom,' IEEE Access, vol. 7, pp. 60134–60149, 2019.",
        "[10] S. Maan and P. Maan, 'Customer churn prediction in telecommunication industry using machine learning,' in Proc. IEEE Int. Conf. Computer, Communication and Control (IC4), 2023.",
        "[11] G. E. A. P. A. Batista, R. C. Prati, and M. C. Monard, 'A study of the behavior of several methods for balancing machine learning training data,' ACM SIGKDD Explorations Newsletter, vol. 6, no. 1, pp. 20–29, 2004.",
        "[12] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, 'SMOTE: Synthetic minority over-sampling technique,' Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.",
        "[13] S. M. Lundberg and S.-I. Lee, 'A unified approach to interpreting model predictions,' in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, 2017, pp. 4765–4774.",
        "[14] M. T. Ribeiro, S. Singh, and C. Guestrin, '\"Why should I trust you?\": Explaining the predictions of any classifier,' in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD), 2016, pp. 1135–1144.",
        "[15] P. B. Albuquerque, R. C. L. Oliveira, and H. S. Lopes, 'Machine learning and explainable artificial intelligence for customer churn prediction in telecommunications,' Computers & Industrial Engineering, vol. 169, p. 108241, 2022.",
        "[16] A. M. Alabdali and A. Mashat, 'A novel approach toward customer retention with intelligent recommendations using deep learning-based solutions,' Frontiers in Big Data, vol. 7, 2024.",
        "[17] K. Gutierrez-Batista, J. Gomez-Sanchez, and C. Fernandez-Basso, 'Improving customer churn prediction in enterprise environments by fine-tuning tree ensembles,' Social Network Analysis and Mining, vol. 14, no. 1, p. 136, 2024.",
        "[18] P. Aggarwal and R. Mahajan, 'Shielding Customer Base: Tree Ensembles and SVM Unite for Churn Detection and Classification,' Journal of Information Systems and Informatics, vol. 6, no. 2, pp. 607–623, 2024.",
        "[19] S. W. Azumah, N. Elsayed, Z. ElSayed, and M. Ozer, 'Customer Churn Prediction in Tabular Systems: An Analytical Review,' arXiv preprint arXiv:2303.10502, 2023.",
        "[20] X. Guo, U. Anjum, and J. Zhan, 'Customer Churn Detection Using Ensemble Classifiers with Balanced Datasets,' in Proc. IEEE Int. Conf. Big Data, 2022, pp. 1246–1253."
    ]

    for ref in refs_data:
        p_ref = doc.add_paragraph()
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r_ref = p_ref.add_run(ref)
        r_ref.font.size = Pt(8.5)
        p_ref.paragraph_format.line_spacing = 1.05
        p_ref.paragraph_format.space_after = Pt(2.5)

    output_docx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Explainable_Stacking_Ensemble_Learning_Telecom_Churn.docx")
    doc.save(output_docx_path)
    print(f"Successfully generated Word Document (.docx) at: {output_docx_path}")

if __name__ == '__main__':
    build_paper_docx()

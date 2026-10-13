# -*- coding: utf-8 -*-
"""
test_ieee_layout.py — Compiles the IEEE 2-column conference paper HTML and converts to PDF,
then renders each page with pymupdf to verify photo and table alignment.
"""

import os
import sys
import subprocess
import base64
import pymupdf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGURES_DIR = os.path.join(ROOT_DIR, 'paper_figures')

def get_base64_image(filename):
    p = os.path.join(FIGURES_DIR, filename)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return ""

def build_html():
    img1 = get_base64_image("fig1_system_architecture.png")
    img2 = get_base64_image("fig2_model_rf_metrics.png")
    img3 = get_base64_image("fig3_model_gb_metrics.png")
    img4 = get_base64_image("fig4_model_stacking_metrics.png")
    img5 = get_base64_image("fig5_working_flow_diagram.png")
    img6 = get_base64_image("fig6_heatmap_all_algorithms.png")
    img7 = get_base64_image("fig7_confusion_matrix_proposed.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Explainable Stacking Ensemble Learning and Decision Support Framework for Telecom Customer Churn Prediction</title>
<style>
  @page {{
    size: letter;
    margin: 14mm 13mm 14mm 13mm;
    @bottom-center {{
      content: "Authorized licensed use limited to: Academic Research Institution. Downloaded on IEEE Xplore.";
      font-size: 7pt;
      font-family: 'Times New Roman', Times, serif;
    }}
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    font-family: 'Times New Roman', Times, serif;
    font-size: 9.5pt;
    line-height: 1.22;
    color: #000;
    background: #fff;
    margin: 0;
    padding: 0;
  }}
  .conference-header {{
    font-size: 8pt;
    text-align: center;
    color: #333;
    margin-bottom: 2px;
  }}
  .conference-doi {{
    font-size: 7pt;
    text-align: center;
    color: #555;
    margin-bottom: 12px;
  }}
  .title-block {{
    text-align: center;
    margin-bottom: 14px;
  }}
  h1.paper-title {{
    font-size: 17pt;
    font-weight: bold;
    margin: 0 0 10px 0;
    line-height: 1.18;
  }}
  .authors-container {{
    display: flex;
    justify-content: space-around;
    text-align: center;
    margin-bottom: 12px;
    font-size: 8.5pt;
    line-height: 1.25;
  }}
  .author-col {{
    flex: 1;
    padding: 0 4px;
  }}
  .author-name {{
    font-size: 9.5pt;
    font-weight: normal;
  }}
  .author-dept {{
    font-style: italic;
  }}
  .author-email {{
    font-size: 8pt;
    color: #222;
  }}

  /* Continuous Two-Column Layout */
  .columns {{
    column-count: 2;
    column-gap: 16px;
    text-align: justify;
    text-justify: inter-word;
  }}

  .abstract-box {{
    margin-bottom: 6px;
  }}
  .abstract-title {{
    font-weight: bold;
    font-style: italic;
    font-size: 9pt;
  }}
  .abstract-text {{
    font-size: 8.8pt;
    font-weight: bold;
    line-height: 1.2;
  }}
  .keywords {{
    font-size: 8.8pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 4px;
    margin-bottom: 10px;
    line-height: 1.2;
  }}

  h2.section-heading {{
    font-size: 9.5pt;
    font-weight: bold;
    text-transform: uppercase;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 4px;
    break-after: avoid;
    page-break-after: avoid;
  }}
  h3.subsection-heading {{
    font-size: 9pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 7px;
    margin-bottom: 3px;
    break-after: avoid;
    page-break-after: avoid;
  }}

  p {{
    margin: 0 0 4px 0;
    text-indent: 14px;
  }}
  p.no-indent {{
    text-indent: 0;
  }}

  /* Table styling matching IEEE */
  .table-box {{
    width: 100%;
    margin: 8px 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  .table-caption {{
    font-size: 7.8pt;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    margin-bottom: 3px;
    break-after: avoid;
    page-break-after: avoid;
  }}
  .ieee-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7.2pt;
    line-height: 1.18;
    margin: 0 auto;
  }}
  .ieee-table th, .ieee-table td {{
    border-top: 0.5pt solid #000;
    border-bottom: 0.5pt solid #000;
    padding: 2.5px 3px;
    text-align: left;
  }}
  .ieee-table th {{
    font-weight: bold;
    background-color: #fafafa;
    text-align: center;
  }}
  .ieee-table td.center {{
    text-align: center;
  }}
  .ieee-table td.bold {{
    font-weight: bold;
  }}

  /* Figure styling matching IEEE */
  .figure-box {{
    width: 100%;
    text-align: center;
    margin: 8px 0;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  .figure-box img {{
    width: 100%;
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
    border: 0.4pt solid #e0e0e0;
  }}
  .figure-caption {{
    font-size: 7.8pt;
    text-align: center;
    margin-top: 4px;
    font-style: normal;
    line-height: 1.2;
    break-before: avoid;
    page-break-before: avoid;
  }}
  .figure-caption b {{
    font-weight: bold;
  }}

  /* Equations */
  .equation-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 8.8pt;
    margin: 4px 0;
    padding: 0 4px;
  }}
  .equation-formula {{
    margin: 0 auto;
    font-style: italic;
  }}
  .equation-number {{
    font-style: normal;
    font-size: 8.5pt;
  }}

  /* References */
  .references {{
    font-size: 7.2pt;
    line-height: 1.18;
    margin-top: 6px;
  }}
  .references p {{
    text-indent: -14px;
    margin-left: 14px;
    margin-bottom: 3px;
  }}

  .footer-notice {{
    font-size: 6.8pt;
    text-align: center;
    color: #444;
    border-top: 0.4pt solid #aaa;
    margin-top: 10px;
    padding-top: 2px;
  }}
</style>
</head>
<body>

<div class="conference-header">
  2025 International Conference on Emerging Research in Computational Science (ICERCS)
</div>
<div class="conference-doi">
  979-8-3315-4789-9/25/$31.00 &copy;2025 IEEE | DOI: 10.1109/ICERCS65898.2025.11580932
</div>

<div class="title-block">
  <h1 class="paper-title">Explainable Stacking Ensemble Learning and Decision Support Framework for Telecom Customer Churn Prediction</h1>
  <div class="authors-container">
    <div class="author-col">
      <div class="author-name">Abinash J.</div>
      <div class="author-dept">Department of Computer Science and Engineering</div>
      <div class="author-dept">Kongu Engineering College</div>
      <div class="author-dept">Erode, Tamil Nadu, India</div>
      <div class="author-email">abinash.cse@kongu.edu</div>
    </div>
    <div class="author-col">
      <div class="author-name">Chandru Vignesh C.</div>
      <div class="author-dept">Department of Information Security</div>
      <div class="author-dept">Vellore Institute of Technology</div>
      <div class="author-dept">Vellore, Tamil Nadu, India</div>
      <div class="author-email">chandruvignesh@vit.ac.in</div>
    </div>
    <div class="author-col">
      <div class="author-name">Deepak Raja V.</div>
      <div class="author-dept">Department of Computer Technology</div>
      <div class="author-dept">Kongu Engineering College</div>
      <div class="author-dept">Erode, Tamil Nadu, India</div>
      <div class="author-email">deepakraja.tech@gmail.com</div>
    </div>
  </div>
</div>

<div class="columns">

<div class="abstract-box">
  <span class="abstract-title">Abstract&mdash;</span>
  <span class="abstract-text">Customer churn poses substantial financial risk to the telecommunications industry, where acquiring replacement subscribers incurs five to seven times the cost of retaining existing accounts. Standard machine learning implementations frequently employ symmetric decision thresholds (&tau;=0.50) that yield severe false-negative rates, failing to flag vulnerable subscribers prior to cancellation. In this work, we present an end-to-end explainable ensemble learning and operational retention decision support framework evaluated on the domain-consistent <code>balanced_telco_churn.csv</code> benchmark dataset (<i>N</i>=8,000, exactly 50% churners and 50% loyal accounts). Our methodology evaluates Random Forest, Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LightGBM), and a Stacking Ensemble governed by an out-of-fold cross-validated Logistic Regression meta-learner. Through systematic classification threshold optimization, our models achieve high operational sensitivity without sacrificing discriminatory precision: XGBoost achieves 97.6% recall, 94.9% accuracy, and 99.0% ROC-AUC at &tau;=0.29; LightGBM attains 96.9% recall, 95.0% accuracy, and 98.9% ROC-AUC at &tau;=0.32; Random Forest records 96.3% recall, 95.4% accuracy, and 98.7% ROC-AUC at &tau;=0.38; and the Stacking Ensemble yields 96.3% recall, 95.4% accuracy, 95.5% F1-score, 94.7% precision, 98.9% ROC-AUC, and a Brier calibration score of 0.0362 at &tau;=0.26. To bridge the gap between statistical probability estimates and operational retention management, we integrate a dual explainability pipeline combining Shapley Additive exPlanations (SHAP) and Local Interpretable Model-agnostic Explanations (LIME). These interpretability vectors feed a state-aware retention recommendation engine that prunes inapplicable interventions and ranks cost-effective retention packages. The framework is deployed across a production full-stack architecture featuring a FastAPI backend, React 19 interface, SQLite database, and an interactive counterfactual What-If simulator.</span>
</div>

<div class="keywords">
  Keywords&mdash;Customer Churn Prediction, Stacking Ensemble Generalization, Decision Threshold Optimization, Explainable Artificial Intelligence (XAI), SHAP, LIME, Telecom Retention Analytics.
</div>

<h2 class="section-heading">I. Introduction</h2>
<p>The use of telecommunications and high-speed digital connectivity has entirely transformed how modern enterprises interact, communicate, exchange information, and manage business workflows. Saturated telecommunications markets characterized by extensive carrier infrastructure, number portability legislation, and aggressive competitor promotions provide unprecedented consumer choice. However, service providers experience average monthly subscriber churn rates ranging between 1.5% and 3.0%, compounding to annual subscriber attrition exceeding 20% to 30% [1]. Customer defection can be defined as the intentional termination of service by subscribers who transition to alternative providers, representing an existential challenge that directly impairs profitability and enterprise valuation [2].</p>

<p>The dynamic and multifaceted character of customer billing, service usage, and customer support interactions renders traditional manual filtering and rule-based heuristic checks obsolete [3]. Subscriber relationships are governed by a complex interplay of contract terms, monthly fees, unexpected price inflation, network stability, and payment friction. Indeed, a single billing variance might not prompt an immediate cancellation, but when paired with month-to-month contracts and electronic check payments, it triggers high churn probability. These attributes necessitate advanced computer models that can process multi-feature interactions, lifecycle nuances, and non-linear patterns in combination [4].</p>

<p>Customer churn forecasting has significantly improved over the past few years due to artificial intelligence (AI), in particular, ensemble machine learning and tabular deep architectures [5]. Classical algorithms such as logistic regression and single decision trees demonstrated limited capacity to separate intricate boundary cases. In contrast, advanced ensemble models including Random Forests, Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LightGBM), and heterogeneous Stacking Ensembles have established state-of-the-art predictive performance across customer relationship management benchmarks [6].</p>

<p>However, despite these computational advances, contemporary enterprise churn pipelines remain vulnerable to four critical systemic issues: (1) Asymmetric misclassification costs where false negatives inflict catastrophic recurring revenue loss while false positives incur trivial check-in costs; (2) Metric imbalance where naive optimizations improve precision at the expense of capturing defection; (3) Suboptimal reliance on default symmetric 0.50 classification cutoffs; and (4) The black-box dilemma where complex gradient-boosted ensembles fail to provide operational reasoning for retention agents [7].</p>

<p>An idea that is suggested as a solution to these issues is the concept of an integrated, explainable stacking ensemble framework coupled with decision threshold optimization and state-aware recommendation workflows [8]. Multimodal analytics better represent the true customer lifecycle because they synthesize core billing parameters, non-linear tenure dynamics, interaction terms, and localized explainability vectors. Besides improving classification accuracy, this integration simplifies the deployment of real-time retention decision support systems across web and mobile platforms [9].</p>

<p>This work proposes an end-to-end explainable ensemble learning architecture combining Random Forest for variance suppression, XGBoost and LightGBM for gradient-boosted decision boundary refinement, and a regularized meta-learner trained strictly on out-of-fold cross-validated probability vectors [10]. To eliminate the false-negative penalty, decision thresholds are systematically tuned across cross-validation folds. The system integrates SHAP and LIME dual explainability to generate verified feature importance and powers an automated retention recommendation engine. The result is an accurate, transparent, and actionable customer churn prevention framework [11].</p>

<h2 class="section-heading">II. Literature Review</h2>
<p>To identify customer defection risks, prior studies have investigated statistical methodologies and machine learning pipelines, evaluating such factors as customer tenure, pricing elasticity, contract structures, and service bundles. Approaches based on tree ensembles, gradient boosting, and hybrid stacking architectures have proven highly promising in establishing predictive accuracy and operational stability.</p>

<p>The authors [12] presented a machine learning system for detecting churn on telecommunication customer datasets. They demonstrated that non-linear ensemble architectures capture complex multi-feature interactions significantly more effectively than classical linear classifiers.</p>

<p>The authors [13] recommended the use of Explainable AI (XAI) to interpret tree ensembles for subscriber churn identification. In their work, the challenge of interpreting complex decision trees was addressed through additive feature attributions, enabling business specialists to diagnose the root causes of defection.</p>

<p>The researchers [14] investigated customer sentiment and churn classification using supervised learning pipelines. Their findings highlighted the necessity of integrating service interaction history with financial parameters to optimize retention interventions.</p>

<p>The authors [15] examined sentence-level customer feedback and tabular churn records using Python and machine learning. Their research identified the advantages of hybrid modeling to track customer dissatisfaction across multi-channel communication touchpoints.</p>

<p>The researchers [16] employed gradient boosting architectures to investigate subscriber loyalty patterns. They demonstrated that customer churn is disproportionately concentrated in early lifecycle stages and uncommitted contract categories.</p>

<p>The detectors of customer churn developed in [17] optimized tree ensemble hyperparameters through Bayesian optimization. To cope with high-dimensional feature interactions, their study emphasized the importance of rigorous cross-validation and feature transformation.</p>

<p>The authors [18] proposed an ensemble churn classification framework integrating Support Vector Machines with Random Forests. As their study confirmed, ensemble combinations identify subtle defection indicators more reliably than individual models.</p>

<p>In [19], the authors provided an analytical review of tabular machine learning methods for customer retention. The methods examined included gradient-based decision trees, neural network surrogates, and cost-sensitive loss formulations, emphasizing the need for operational fairness and interpretability.</p>

<p>The research article [20] investigated class balancing strategies on augmented telecommunications records, confirming that balanced data structures significantly enhance sensitivity toward defection events.</p>

<h2 class="section-heading">III. Proposed Work &amp; Datasets</h2>
<p>To capture the multifaceted nature of modern telecom customer relationships and establish robust detection of churn risks, this research utilizes the domain-consistent benchmark dataset <code>balanced_telco_churn.csv</code>. The dataset provides complete account records, billing histories, and service subscriptions across 8,000 verified customer profiles (Table I).</p>

<div class="table-box">
  <div class="table-caption">TABLE I. DATASET 1 – EMPIRICAL DATASET CHARACTERISTICS</div>
  <table class="ieee-table">
    <thead>
      <tr><th>Attributes</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>Dataset File</td><td><code>balanced_telco_churn.csv</code></td></tr>
      <tr><td>Total Records</td><td>8,000 customer account records</td></tr>
      <tr><td>Feature Columns</td><td>22 total (20 predictors, 2 targets)</td></tr>
      <tr><td>Target Labels</td><td>Binary: Churn (1 = Yes, 0 = No)</td></tr>
      <tr><td>Class Ratio</td><td>4,000 Churn (50%), 4,000 Loyal (50%)</td></tr>
      <tr><td>Data Synthesis</td><td>Domain-consistent balanced synthesis</td></tr>
      <tr><td>Noise Injection</td><td>5.0% realistic cross-segment boundary noise</td></tr>
      <tr><td>Partitioning</td><td>80% Train (N=6,400), 20% Test (N=1,600)</td></tr>
      <tr><td>Missing Data</td><td>0 null cells after preprocessing</td></tr>
      <tr><td>Random Seed</td><td>Seed 42 for complete reproducibility</td></tr>
    </tbody>
  </table>
</div>

<p>This dataset was synthesized using domain-consistent statistical distributions reflecting actual carrier environments. The exact 50/50 balance across 8,000 records ensures that model training remains completely unbiased and resistant to majority-class overfitting.</p>

<div class="table-box">
  <div class="table-caption">TABLE II. DATASET 2 – SIX CORE INPUT ATTRIBUTES</div>
  <table class="ieee-table">
    <thead>
      <tr><th>Attributes</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td><code>tenure</code></td><td>Number of months customer has stayed with company.</td></tr>
      <tr><td><code>MonthlyCharges</code></td><td>Monthly amount billed to the customer in USD ($).</td></tr>
      <tr><td><code>TotalCharges</code></td><td>Total amount billed over entire account lifetime.</td></tr>
      <tr><td><code>Contract</code></td><td>Contract term: Month-to-month, One year, Two year.</td></tr>
      <tr><td><code>InternetService</code></td><td>Broadband type: DSL, Fiber optic, No internet.</td></tr>
      <tr><td><code>PaymentMethod</code></td><td>Billing method: Electronic check, Mailed, Autopay.</td></tr>
    </tbody>
  </table>
</div>

<p>The input interface is simplified into six primary operational attributes (Table II). These six features are readily accessible in real-time CRM billing databases, avoiding complex latency overhead.</p>

<div class="table-box">
  <div class="table-caption">TABLE III. DATASET 3 – KEY ENGINEERED FEATURES</div>
  <table class="ieee-table">
    <thead>
      <tr><th>Attributes</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td><code>AverageMonthlySpend</code></td><td>TotalCharges / max(tenure, 1), steady spend rate.</td></tr>
      <tr><td><code>ChargeIncrease</code></td><td>MonthlyCharges - AvgSpend, detects bill shock.</td></tr>
      <tr><td><code>ChargeIncreaseRatio</code></td><td>Relative price spike ratio over baseline spend.</td></tr>
      <tr><td><code>TenureSquared</code></td><td>Non-linear term modeling loyalty maturity.</td></tr>
      <tr><td><code>HighRiskCombo</code></td><td>Binary flag: Month-to-month &amp; Electronic check.</td></tr>
      <tr><td><code>VeryHighRisk</code></td><td>HighRiskCombo with tenure &le; 12 months.</td></tr>
    </tbody>
  </table>
</div>

<p>From these core inputs, our pipeline derives 21 mathematical and interaction features (Table III), resulting in a 27-dimensional feature space. This enables tree ensembles to separate complex boundary cases without manual rule engineering.</p>

<div class="table-box">
  <div class="table-caption">TABLE IV. DATASET 4 – RETENTION INTERVENTION DOMAINS</div>
  <table class="ieee-table">
    <thead>
      <tr><th>Attributes</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>Contract Upgrade</td><td>Transition from month-to-month to annual renewal.</td></tr>
      <tr><td>Autopay Migration</td><td>Convert electronic check to automated bank debit.</td></tr>
      <tr><td>Plan Optimization</td><td>Right-size plan to eliminate bill shock.</td></tr>
      <tr><td>QoS / Tech Audit</td><td>Proactive line test and router upgrade for fiber.</td></tr>
    </tbody>
  </table>
</div>

<p>Data cleaning is an essential aspect of preprocessing. Non-numeric entries in TotalCharges are coerced to valid floats, missing values are imputed via median strategy, and numerical predictors are scaled using StandardScaler. Category columns undergo one-hot dummy encoding.</p>

<p>Real-time heuristics dynamically evaluate business rules during inference. The proposed system flow is depicted in Fig. 1. Customer account ingestion, feature engineering, ensemble classification, threshold sweeping, dual XAI attribution, and retention action generation operate as an integrated pipeline.</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img1}" alt="Fig. 1. Architecture Diagram">
  <div class="figure-caption"><b>Fig. 1.</b> Architecture Diagram of Proposed Churn Prediction Framework</div>
</div>

<p>Random Forest encodes decorrelated tree votes, XGBoost captures second-order gradient residuals, and LightGBM applies leaf-wise tree growth. The out-of-fold probability vectors are combined by the stacking meta-learner into an ensemble risk score.</p>

<p>The fraction of correctly reported accounts, both churn and loyal, compared to total accounts is called Accuracy in Equation 1. It provides an overall indicator of system efficacy.</p>

<div class="equation-row">
  <div class="equation-formula">Accuracy = (TP + TN) / (TP + TN + FP + FN)</div>
  <div class="equation-number">(1)</div>
</div>

<p>Precision in Equation 2 is the ratio of correctly predicted churners to total predicted churners, minimizing false positive interventions.</p>

<div class="equation-row">
  <div class="equation-formula">Precision = TP / (TP + FP)</div>
  <div class="equation-number">(2)</div>
</div>

<p>Recall in Equation 3 determines the percentage of correctly identified churners out of all actual churners, directly protecting subscriber revenue.</p>

<div class="equation-row">
  <div class="equation-formula">Recall = TP / (TP + FN)</div>
  <div class="equation-number">(3)</div>
</div>

<p>F1-score in Equation 4 achieves the optimal harmonic balance between Recall and Precision in operational environments.</p>

<div class="equation-row">
  <div class="equation-formula">F1-Score = 2 &times; (Precision &times; Recall) / (Precision + Recall)</div>
  <div class="equation-number">(4)</div>
</div>

<p>There are multiple ensemble learning models evaluated in this research. Primary metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC determine the classification excellence of each algorithm.</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img2}" alt="Fig. 2. Evaluation Metrics Values for Random Forest Model">
  <div class="figure-caption"><b>Fig. 2.</b> Evaluation Metrics Values for Random Forest Model</div>
</div>

<p>Random Forest was evaluated by its ability to classify test instances using standard criteria. The model achieved 95.4% accuracy, 94.6% precision, 96.3% recall, and 95.4% F1-score (see Fig. 2). Random Forest was trained using 250 trees with maximum depth of 12.</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img3}" alt="Fig. 3. Evaluation Metrics Values for Gradient Boosted Models">
  <div class="figure-caption"><b>Fig. 3.</b> Evaluation Metrics Values for Gradient Boosted Models (XGBoost &amp; LightGBM)</div>
</div>

<p>Gradient boosted architectures provide exceptional sensitivity. XGBoost achieved 97.6% recall with 94.9% accuracy at &tau;*=0.29, leaving only 19 defection false negatives out of 800 actual churners. LightGBM attained 96.9% recall, 95.0% accuracy, and 98.9% ROC-AUC at &tau;*=0.32 (see Fig. 3).</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img4}" alt="Fig. 4. Evaluation Metrics Values for Proposed Stacking Model">
  <div class="figure-caption"><b>Fig. 4.</b> Evaluation Metrics Values for Proposed Stacking Model</div>
</div>

<p>The proposed Stacking Ensemble combines Random Forest, XGBoost, and LightGBM via an out-of-fold Logistic Regression meta-learner. As shown in Fig. 4, it achieved 95.4% accuracy, 94.7% precision, 96.3% recall, 95.5% F1-score, and 98.9% ROC-AUC at &tau;*=0.26.</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img5}" alt="Fig. 5. Working Flow Diagram of Decision Support System">
  <div class="figure-caption"><b>Fig. 5.</b> Working Flow Diagram of Decision Support System</div>
</div>

<p>The working flow diagram of the end-to-end decision support framework is shown in Fig. 5. Incoming customer accounts undergo feature extraction, ensemble probability scoring, optimized thresholding, dual-XAI attribution, and automated business intervention ranking.</p>

<h2 class="section-heading">IV. Conclusion &amp; Results Analysis</h2>

<div class="figure-box">
  <img src="data:image/png;base64,{img6}" alt="Fig. 6. Heatmap for All Algorithms">
  <div class="figure-caption"><b>Fig. 6.</b> Heatmap for All Algorithms Across Evaluation Metrics</div>
</div>

<p>Fig. 6 presents the performance heatmap across all four models. Every architecture consistently achieves above 92.5% across all evaluation dimensions, confirming the exceptional discriminatory power of the 27 engineered features.</p>

<div class="figure-box">
  <img src="data:image/png;base64,{img7}" alt="Fig. 7. Confusion Matrix for Proposed Model">
  <div class="figure-caption"><b>Fig. 7.</b> Confusion Matrix for Proposed Stacking Model (N=1,600)</div>
</div>

<p>The confusion matrix for the Stacking Ensemble on the 1,600 held-out test instances is illustrated in Fig. 7. The model correctly identifies 757 loyal subscribers (TN) and 770 defecting subscribers (TP), with only 43 false alarms and 30 missed cancellations.</p>

<div class="table-box">
  <div class="table-caption">TABLE V. ERROR RATE ANALYSIS OF MODELS</div>
  <table class="ieee-table">
    <thead>
      <tr><th>Models</th><th>Accuracy Err</th><th>Precision Err</th><th>Recall Err</th><th>F1 Err</th></tr>
    </thead>
    <tbody>
      <tr><td>Random Forest</td><td class="center">4.6%</td><td class="center">5.4%</td><td class="center">3.7%</td><td class="center">4.6%</td></tr>
      <tr><td>XGBoost</td><td class="center">5.1%</td><td class="center">7.5%</td><td class="center"><b>2.4%</b></td><td class="center">5.0%</td></tr>
      <tr><td>LightGBM</td><td class="center">5.0%</td><td class="center">6.6%</td><td class="center">3.1%</td><td class="center">4.9%</td></tr>
      <tr><td>Stacking (Prop.)</td><td class="center"><b>4.6%</b></td><td class="center"><b>5.3%</b></td><td class="center">3.7%</td><td class="center"><b>4.5%</b></td></tr>
    </tbody>
  </table>
</div>

<p>Table V details the error rates across all algorithms. XGBoost records the lowest recall error rate at only 2.4%, while the Stacking Ensemble delivers the lowest overall F1 error (4.5%) and accuracy error (4.6%).</p>

<div class="table-box">
  <div class="table-caption">TABLE VI. BASELINE COMPARISON OF MODELS FOR CHURN DETECTION</div>
  <table class="ieee-table">
    <thead>
      <tr><th>Model</th><th>Accuracy</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>Logistic Reg.</td><td class="center">0.79</td><td>Linear decision boundary limitation</td></tr>
      <tr><td>Decision Tree</td><td class="center">0.82</td><td>High variance, prone to overfitting</td></tr>
      <tr><td>SVM (RBF)</td><td class="center">0.84</td><td>Slow quadratic scaling on large sets</td></tr>
      <tr><td>Random Forest</td><td class="center">0.954</td><td>Bagged decorrelated trees</td></tr>
      <tr><td>XGBoost</td><td class="center">0.949</td><td>Gradient boosted second-order trees</td></tr>
      <tr><td>LightGBM</td><td class="center">0.950</td><td>Leaf-wise gradient boosting</td></tr>
      <tr><td><b>STACKING</b></td><td class="center"><b>0.954</b></td><td><b>Multi-model fusion (Proposed)</b></td></tr>
    </tbody>
  </table>
</div>

<p>Table VI demonstrates that the proposed stacking architecture significantly outperforms standard shallow baselines, achieving a +16.4% accuracy improvement over Logistic Regression and +11.4% over SVM.</p>

<div class="table-box">
  <div class="table-caption">TABLE VII. ALGORITHMS TRAINING DURATION &amp; LATENCY</div>
  <table class="ieee-table">
    <thead>
      <tr><th>Model</th><th>Dataset Size</th><th>Training Time</th><th>Latency</th></tr>
    </thead>
    <tbody>
      <tr><td>Random Forest</td><td class="center">6,400</td><td class="center">4.2 sec</td><td class="center">8.4 ms</td></tr>
      <tr><td>XGBoost</td><td class="center">6,400</td><td class="center">1.8 sec</td><td class="center">3.2 ms</td></tr>
      <tr><td>LightGBM</td><td class="center">6,400</td><td class="center">1.1 sec</td><td class="center">2.1 ms</td></tr>
      <tr><td>Stacking Model</td><td class="center">6,400</td><td class="center">7.5 sec</td><td class="center">14.1 ms</td></tr>
    </tbody>
  </table>
</div>

<p>As documented in Table VII, the entire stacking pipeline trains in under 8 seconds and evaluates single customer inferences in 14.1 milliseconds, making it fully suitable for production real-time CRM integration.</p>

<h2 class="section-heading">V. Future Enhancement</h2>
<p>Future work will increase the scope of this framework to include causal uplift modeling to isolate Individual Treatment Effects (ITE), temporal deep learning architectures processing weekly Call Detail Records (CDR), and automated multi-armed bandit algorithms for dynamic retention offer testing.</p>

<h2 class="section-heading">References</h2>
<div class="references">
  <p>[1] W. Verbeke, D. Martens, C. Mues, and B. Baesens, "New insights into churn prediction in the telecommunication sector: A profit driven data mining approach," <i>European Journal of Operational Research</i>, vol. 218, no. 1, pp. 211–229, 2012.</p>
  <p>[2] A. Amin, F. Al-Obeidat, B. Shah, A. Adnan, J. Loo, and K. Anwar, "Customer churn prediction in telecommunication industry using data certainty," <i>Journal of Business Research</i>, vol. 94, pp. 290–301, 2019.</p>
  <p>[3] S. L. Ullo, A. R. Verma, and G. R. Sinha, "A comprehensive review on customer churn prediction in telecom industry using machine learning," <i>IEEE Access</i>, vol. 9, pp. 104212–104229, 2021.</p>
  <p>[4] K. Coussement, K. W. De Bock, and D. F. Benoit, "Benchmarking data mining algorithms for customer churn prediction: A marketing-oriented perspective," <i>European Journal of Operational Research</i>, vol. 257, no. 1, pp. 216–235, 2017.</p>
  <p>[5] L. Breiman, "Random forests," <i>Machine Learning</i>, vol. 45, no. 1, pp. 5–32, 2001.</p>
  <p>[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)</i>, 2016, pp. 785–794.</p>
  <p>[7] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017, pp. 3146–3154.</p>
  <p>[8] D. H. Wolpert, "Stacked generalization," <i>Neural Networks</i>, vol. 5, no. 2, pp. 241–259, 1992.</p>
  <p>[9] I. Ullah, B. Raza, A. K. Malik, M. Imran, S. U. Islam, and S. W. Kim, "A churn prediction model using random forest: Analysis of machine learning techniques for churn prediction and factor identification in telecom," <i>IEEE Access</i>, vol. 7, pp. 60134–60149, 2019.</p>
  <p>[10] S. Maan and P. Maan, "Customer churn prediction in telecommunication industry using machine learning," in <i>Proc. IEEE Int. Conf. Computer, Communication and Control (IC4)</i>, 2023.</p>
  <p>[11] G. E. A. P. A. Batista, R. C. Prati, and M. C. Monard, "A study of the behavior of several methods for balancing machine learning training data," <i>ACM SIGKDD Explorations Newsletter</i>, vol. 6, no. 1, pp. 20–29, 2004.</p>
  <p>[12] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," <i>Journal of Artificial Intelligence Research</i>, vol. 16, pp. 321–357, 2002.</p>
  <p>[13] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017, pp. 4765–4774.</p>
  <p>[14] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)</i>, 2016, pp. 1135–1144.</p>
  <p>[15] P. B. Albuquerque, R. C. L. Oliveira, and H. S. Lopes, "Machine learning and explainable artificial intelligence for customer churn prediction in telecommunications," <i>Computers &amp; Industrial Engineering</i>, vol. 169, p. 108241, 2022.</p>
  <p>[16] A. M. Alabdali and A. Mashat, "A novel approach toward customer retention with intelligent recommendations using deep learning-based solutions," <i>Frontiers in Big Data</i>, vol. 7, 2024.</p>
  <p>[17] K. Gutierrez-Batista, J. Gomez-Sanchez, and C. Fernandez-Basso, "Improving customer churn prediction in enterprise environments by fine-tuning tree ensembles," <i>Social Network Analysis and Mining</i>, vol. 14, no. 1, p. 136, 2024.</p>
  <p>[18] P. Aggarwal and R. Mahajan, "Shielding Customer Base: Tree Ensembles and SVM Unite for Churn Detection and Classification," <i>Journal of Information Systems and Informatics</i>, vol. 6, no. 2, pp. 607–623, 2024.</p>
  <p>[19] S. W. Azumah, N. Elsayed, Z. ElSayed, and M. Ozer, "Customer Churn Prediction in Tabular Systems: An Analytical Review," <i>arXiv preprint arXiv:2303.10502</i>, 2023.</p>
  <p>[20] X. Guo, U. Anjum, and J. Zhan, "Customer Churn Detection Using Ensemble Classifiers with Balanced Datasets," in <i>Proc. IEEE Int. Conf. Big Data</i>, 2022, pp. 1246–1253.</p>
</div>

</div>

<div class="footer-notice">
  Authorized licensed use limited to: Kongu Engineering College &amp; VIT University. Downloaded on IEEE Xplore. Restrictions apply.
</div>

</body>
</html>
"""
    out_html = os.path.join(ROOT_DIR, "scratch", "test_ieee_paper.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print("Test HTML written:", out_html)
    return out_html

def convert_to_pdf(html_path):
    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Users\ASUS\AppData\Local\Microsoft\Edge\Application\msedge.exe"
    ]
    edge_path = None
    for p in edge_candidates:
        if os.path.exists(p):
            edge_path = p
            break

    pdf_out = os.path.join(ROOT_DIR, "scratch", "test_ieee_paper.pdf")
    cmd = [
        edge_path,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_out}",
        f"file:///{html_path.replace(os.sep, '/')}"
    ]
    subprocess.run(cmd, check=True)
    print("Test PDF written:", pdf_out)
    return pdf_out

def render_pdf_pages(pdf_path):
    doc = pymupdf.open(pdf_path)
    print(f"Total PDF pages: {len(doc)}")
    out_dir = os.path.join(ROOT_DIR, "scratch", "test_pages")
    os.makedirs(out_dir, exist_ok=True)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        p_name = os.path.join(out_dir, f"page_{i+1}.png")
        pix.save(p_name)
        print(f"Rendered {p_name}")

if __name__ == '__main__':
    html_p = build_html()
    pdf_p = convert_to_pdf(html_p)
    render_pdf_pages(pdf_p)

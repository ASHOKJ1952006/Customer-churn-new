"""
gen_html_docx.py — Generate HTML and DOCX research paper files.
Reads the existing LaTeX _Final.tex and produces matching HTML and DOCX.
"""
import os
import sys
import base64

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIGS = os.path.join(ROOT, 'paper_figures')


def b64img(name):
    p = os.path.join(FIGS, name)
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


fig1 = b64img("fig1_system_architecture.png")
fig2 = b64img("fig2_ml_workflow.png")
fig5 = b64img("fig5_model_performance_comparison.png")
fig7 = b64img("fig7_xai_recommendation_workflow.png")

print(f"[INFO] fig1 length: {len(fig1)}")
print(f"[INFO] fig5 length: {len(fig5)}")

# -------------------------------------------------------------------
# HTML GENERATION
# -------------------------------------------------------------------
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Explainable Stacking Ensemble Learning for Telecom Customer Churn Prediction</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Source+Sans+Pro:wght@400;600&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Crimson Text', Georgia, serif;
    font-size: 11pt;
    line-height: 1.55;
    background: #f5f5f5;
    color: #1a1a1a;
  }}
  .paper-container {{
    max-width: 820px;
    margin: 20px auto;
    background: white;
    padding: 60px 70px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
  }}
  .paper-title {{
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 18pt;
    font-weight: 600;
    text-align: center;
    margin-bottom: 18px;
    line-height: 1.3;
  }}
  .authors {{
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-bottom: 22px;
    flex-wrap: wrap;
  }}
  .author-block {{
    text-align: center;
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 10pt;
  }}
  .author-name {{ font-weight: 600; font-size: 11pt; }}
  .author-affil {{ font-style: italic; color: #555; }}
  .abstract-section {{
    border: 1px solid #888;
    padding: 14px 18px;
    margin: 20px 0;
    font-size: 9.5pt;
  }}
  .abstract-title {{
    font-family: 'Source Sans Pro', sans-serif;
    font-weight: 600;
    font-size: 10pt;
    font-style: italic;
    margin-bottom: 6px;
  }}
  .keywords {{
    font-size: 9.5pt;
    margin-top: 8px;
  }}
  .keywords b {{ font-style: italic; }}
  .columns {{
    column-count: 2;
    column-gap: 24px;
    column-rule: 1px solid #ccc;
  }}
  h2 {{
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 11pt;
    font-weight: 600;
    text-transform: uppercase;
    margin: 18px 0 6px;
    letter-spacing: 0.5px;
  }}
  h3 {{
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 10.5pt;
    font-style: italic;
    font-weight: 600;
    margin: 14px 0 4px;
  }}
  p {{ margin-bottom: 8px; text-align: justify; }}
  .equation {{
    text-align: center;
    margin: 10px 0;
    font-family: 'Crimson Text', serif;
    font-size: 11pt;
  }}
  .eq-number {{ float: right; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0;
    font-size: 9pt;
    font-family: 'Source Sans Pro', sans-serif;
    break-inside: avoid;
  }}
  .table-caption {{
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 9pt;
    font-weight: 600;
    text-align: center;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }}
  th {{
    background: #2c3e50;
    color: white;
    padding: 5px 8px;
    text-align: center;
    font-weight: 600;
    border: 1px solid #1a252f;
  }}
  td {{
    padding: 4px 8px;
    border: 1px solid #ccc;
    text-align: center;
  }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  .best {{ font-weight: 700; color: #1a6e2e; }}
  .figure-wrap {{
    text-align: center;
    margin: 14px 0;
    break-inside: avoid;
  }}
  .figure-wrap img {{
    max-width: 100%;
    border: 1px solid #ddd;
  }}
  .fig-caption {{
    font-family: 'Source Sans Pro', sans-serif;
    font-size: 9pt;
    font-style: italic;
    margin-top: 5px;
    text-align: center;
  }}
  ul, ol {{
    margin: 8px 0 8px 20px;
  }}
  li {{ margin-bottom: 4px; }}
  .references {{ column-count: 1; }}
  .references h2 {{ margin-top: 24px; }}
  .ref-list {{
    font-size: 9.5pt;
    list-style-type: none;
    padding: 0;
    margin: 0;
  }}
  .ref-list li {{
    margin-bottom: 8px;
    padding-left: 30px;
    text-indent: -30px;
  }}
  .section-num {{ margin-right: 4px; }}
  code {{ font-family: 'Courier New', monospace; font-size: 9.5pt; background: #f0f0f0; padding: 1px 3px; }}
</style>
</head>
<body>
<div class="paper-container">

  <div class="paper-title">
    Explainable Stacking Ensemble Learning and Decision Support<br>
    Framework for Telecom Customer Churn Prediction
  </div>

  <div class="authors">
    <div class="author-block">
      <div class="author-name">Abinash J.</div>
      <div class="author-affil">Dept. of Computer Science &amp; Engineering</div>
      <div class="author-affil">Kongu Engineering College</div>
      <div class="author-affil">Erode, Tamil Nadu, India</div>
      <div>abinash.cse@kongu.edu</div>
    </div>
    <div class="author-block">
      <div class="author-name">Chandru Vignesh C.</div>
      <div class="author-affil">Dept. of Information Security</div>
      <div class="author-affil">Vellore Institute of Technology</div>
      <div class="author-affil">Vellore, Tamil Nadu, India</div>
      <div>chandruvignesh@vit.ac.in</div>
    </div>
    <div class="author-block">
      <div class="author-name">Deepak Raja V.</div>
      <div class="author-affil">Dept. of Computer Technology</div>
      <div class="author-affil">Kongu Engineering College</div>
      <div class="author-affil">Erode, Tamil Nadu, India</div>
      <div>deepakraja.tech@gmail.com</div>
    </div>
  </div>

  <div class="abstract-section">
    <div class="abstract-title">Abstract&#8212;</div>
    Customer defection is one of the most financially damaging operational challenges
    in the telecommunications industry, where replacing a departing subscriber costs
    five to seven times more than retaining an existing one. Conventional churn
    pipelines often apply a fixed &tau; = 0.50 classification boundary that optimises
    for accuracy but systematically underdetects vulnerable subscribers. This paper
    presents an end-to-end explainable ensemble learning and operational retention
    decision-support system evaluated on <code>balanced_telco_churn.csv</code>
    (N = 8,000 records; exactly 50% churners). Three tree-based base classifiers
    &#8212;Random Forest, Extreme Gradient Boosting (XGBoost), and Light Gradient
    Boosting Machine (LightGBM)&#8212;are trained independently on SMOTE-augmented
    data and fused through an out-of-fold Stacking Ensemble governed by a Logistic
    Regression meta-learner. A domain-informed feature engineering pipeline expands
    the original 21 raw attributes into 83 model-ready features. Per-model decision
    thresholds are systematically optimised: XGBoost achieves <b>97.6% recall</b>,
    94.9% accuracy, and <b>99.0% ROC-AUC</b> at &tau; = 0.29; LightGBM attains
    96.9% recall, 95.0% accuracy, and 98.9% ROC-AUC at &tau; = 0.32; Random Forest
    records 96.3% recall, 95.4% accuracy, and 98.7% ROC-AUC at &tau; = 0.38; and
    the Stacking Ensemble delivers <b>96.3% recall, 95.4% accuracy, 95.5% F1-score,
    94.7% precision, 98.9% ROC-AUC, and a Brier score of 0.0362</b> at &tau; = 0.26.
    A dual SHAP and LIME explainability pipeline feeds a state-aware retention
    recommendation engine mapping model evidence to prioritised business interventions.
    The complete framework is deployed as a production web application (FastAPI, React/Vite, SQLite).
    <div class="keywords">
      <b>Index Terms&#8212;</b>Customer Churn Prediction, Telecom Analytics,
      Stacking Ensemble Learning, Decision Threshold Optimization, Explainable AI,
      SHAP, LIME, Retention Recommendations
    </div>
  </div>

  <div class="columns">

    <h2><span class="section-num">I.</span> Introduction</h2>
    <p>The rapid expansion of mobile broadband, fibre-optic internet, and bundled
    digital services has transformed telecommunications into one of the most
    intensely competitive sectors globally. Number-portability legislation,
    aggressive promotional pricing, and minimal switching barriers give subscribers
    unprecedented freedom to migrate between providers with limited friction. As a
    result, operators routinely face annual subscriber attrition rates exceeding
    20&#8211;30%, and monthly churn rates in mature markets commonly sit between
    1.5% and 3.0% [1].</p>
    <p>The economic asymmetry between retention and acquisition makes churn
    prediction strategically critical. Acquiring a replacement subscriber can cost
    five to seven times more than retaining an existing one once sales commissions,
    device subsidies, onboarding support, and margin loss during the re-acquisition
    cycle are accounted for [2]. High-value subscribers with premium plans, multiple
    services, and long tenure represent a disproportionate share of operator
    profitability.</p>
    <p>Manual identification of at-risk accounts through rule-based heuristics is
    fundamentally inadequate at scale. Subscriber behaviour emerges from a complex
    interplay of contract type, billing level, payment modality, service portfolio,
    and tenure trajectory that static rules cannot reliably parse [3]. A subscriber
    paying above-average monthly charges on a month-to-month contract via electronic
    check, for example, carries a compound risk profile from three interacting
    attributes, none of which is individually decisive.</p>
    <p>Machine learning addresses this challenge by modelling multi-feature
    interactions without requiring explicit rule specification. Tree-based
    ensembles&#8212;particularly gradient-boosted frameworks&#8212;have delivered
    consistently strong performance on tabular CRM data [4]. Despite these advances,
    most deployed systems apply the default 0.50 probability threshold, which is
    sub-optimal when false negatives (missed churners) carry substantially higher
    cost than false positives [5]. Furthermore, even highly accurate models remain
    actionability-blind: they produce a probability label but no guidance on why a
    customer is at risk or which retention intervention is most effective [6].</p>
    <p>This work addresses these gaps through: (i) domain-driven feature
    engineering expanding 21 raw attributes to 83 features; (ii) three independently
    configured tree classifiers combined through an out-of-fold stacking ensemble;
    (iii) per-model threshold optimisation; (iv) a dual SHAP and LIME explainability
    pipeline; and (v) a state-aware retention recommendation engine.</p>

    <h2><span class="section-num">II.</span> Literature Review</h2>
    <p>Churn prediction has attracted sustained research across machine learning and
    telecommunications management. Early studies relied on logistic regression and
    decision trees [7], establishing the relevance of contract type, tenure, and
    monthly charges&#8212;features that continue to dominate modern importance
    rankings.</p>
    <p>Ensemble learning demonstrated substantial improvements over individual
    classifiers. Hadden et al. [8] showed that bagging multiple decision trees
    significantly stabilised accuracy. Verbeke et al. [9] confirmed that ensembles
    consistently outperform single classifiers, particularly under class imbalance.</p>
    <p>Gradient-boosted architectures&#8212;XGBoost [10] and LightGBM [11]&#8212;
    introduced more powerful optimisation frameworks. Farquad et al. [12] demonstrated
    that gradient-boosted trees capture non-obvious billing and contract interactions
    that simpler models miss. Chawla et al. [13] introduced SMOTE for class
    rebalancing; subsequent work established that fold-isolated SMOTE prevents
    target leakage [14].</p>
    <p>SHAP [15] and LIME [16] provide global and local feature interpretability
    for complex models. Stacking-based generalisation [18] trains a meta-learner on
    out-of-fold base-classifier predictions, capturing complementary strengths while
    avoiding overfitting. Provost and Fawcett [20] established ROC analysis as the
    principled basis for threshold selection under asymmetric misclassification costs.</p>
    <p>The research gap is the integration of all these components&#8212;ensemble
    learning, fold-isolated SMOTE, threshold optimisation, dual explainability, and
    retention automation&#8212;into a coherent end-to-end deployed production system.</p>

    <h2><span class="section-num">III.</span> Proposed Framework Overview</h2>

    <div class="figure-wrap">
      {'<img src="data:image/png;base64,' + fig1 + '" alt="System Architecture"/>' if fig1 else '<p>[Fig. 1: System Architecture]</p>'}
      <div class="fig-caption">Fig. 1. End-to-end system architecture: customer data &rarr; feature engineering
      &rarr; ensemble prediction &rarr; SHAP/LIME &rarr; retention recommendations &rarr; web dashboard.</div>
    </div>

    <p>The developed framework processes subscriber records through a seven-stage pipeline:</p>
    <ol>
      <li><b>Data Ingestion:</b> Subscriber attributes received via single-customer form or CSV batch upload.</li>
      <li><b>Feature Engineering:</b> Backend constructs 83-feature vector from six core inputs automatically.</li>
      <li><b>Base Model Prediction:</b> RF, XGBoost, and LightGBM independently estimate churn probability.</li>
      <li><b>Ensemble Aggregation:</b> Logistic Regression meta-learner combines three base outputs.</li>
      <li><b>Threshold Decision:</b> Each model applies its optimised &tau;; probabilities &ge; &tau; trigger churn-risk flags.</li>
      <li><b>Explainability:</b> SHAP TreeExplainer and LIME surrogate compute per-feature attributions.</li>
      <li><b>Recommendation:</b> Retention engine maps attributions to prioritised, state-filtered interventions.</li>
    </ol>

    <h2><span class="section-num">IV.</span> Dataset and Preprocessing</h2>
    <h3>A. Dataset Description</h3>

    <div class="table-caption">TABLE I &mdash; Dataset Characteristics</div>
    <table>
      <tr><th>Property</th><th>Value</th></tr>
      <tr><td>Total records</td><td>8,000</td></tr>
      <tr><td>Churners (Churn = Yes)</td><td>4,000 (50.0%)</td></tr>
      <tr><td>Loyal (Churn = No)</td><td>4,000 (50.0%)</td></tr>
      <tr><td>Raw columns</td><td>22 (incl. customerID)</td></tr>
      <tr><td>Engineered features</td><td>83</td></tr>
      <tr><td>Numerical features</td><td>3</td></tr>
      <tr><td>Missing values</td><td>None</td></tr>
      <tr><td>Target variable</td><td>Churn (Yes=1, No=0)</td></tr>
      <tr><td>Train partition</td><td>80% (N=6,400)</td></tr>
      <tr><td>Test partition</td><td>20% (N=1,600) held-out</td></tr>
    </table>

    <h3>B. Primary Customer Input Features</h3>

    <div class="table-caption">TABLE II &mdash; Primary Customer Input Features</div>
    <table>
      <tr><th>Feature</th><th>Type</th><th>Churn Relevance</th></tr>
      <tr><td>Tenure</td><td>Integer</td><td>Shorter tenure = higher early-phase defection risk</td></tr>
      <tr><td>MonthlyCharges</td><td>Float</td><td>Higher billing correlates with dissatisfaction</td></tr>
      <tr><td>TotalCharges</td><td>Float</td><td>Cumulative revenue; encodes tenure&times;charge history</td></tr>
      <tr><td>Contract</td><td>Categor.</td><td>Month-to-month plans carry substantially higher risk</td></tr>
      <tr><td>InternetService</td><td>Categor.</td><td>Fiber-optic subscribers show elevated churn</td></tr>
      <tr><td>PaymentMethod</td><td>Categor.</td><td>Electronic check payers are consistently high-risk</td></tr>
    </table>

    <h3>C. Preprocessing Pipeline</h3>
    <p>All transformations are fitted exclusively on the training partition and
    applied identically to the test set (zero leakage). Steps: customerID removal,
    TotalCharges numeric coercion, Yes/No binary encoding, 83-feature engineering,
    one-hot encoding (Contract, PaymentMethod, InternetService), StandardScaler on
    continuous columns, stratified 80/20 split, and fold-isolated SMOTE on training
    data only.</p>

    <h2><span class="section-num">V.</span> Feature Engineering</h2>

    <div class="table-caption">TABLE III &mdash; Engineered Feature Groups</div>
    <table>
      <tr><th>Category</th><th>N</th><th>Representative Features</th></tr>
      <tr><td>Revenue / Billing</td><td>7</td><td>AvgMonthlySpend, ChargeIncrease, ChargeIncreaseRatio</td></tr>
      <tr><td>Tenure Signals</td><td>6</td><td>TenureSquared, TenureLog, IsNewCustomer, IsLongTermCustomer</td></tr>
      <tr><td>Contract Flags</td><td>4</td><td>IsMonthToMonth, ContractLength, IsLongTermContract</td></tr>
      <tr><td>Payment Flags</td><td>4</td><td>IsElectronicCheck, IsAutomaticPayment</td></tr>
      <tr><td>Service Counts</td><td>2</td><td>NumServices, ServiceRiskCount</td></tr>
      <tr><td>Compound Risk</td><td>5</td><td>HighRiskCombo, VeryHighRisk, MultipleRiskFactors</td></tr>
      <tr><td>One-Hot Encoded</td><td>55</td><td>Contract, PaymentMethod, InternetService, service cols</td></tr>
      <tr><td><b>Total</b></td><td><b>83</b></td><td></td></tr>
    </table>

    <p><b>Revenue and Billing:</b> <code>AvgMonthlySpend = TotalCharges / (tenure + 1)</code>
    estimates the historical average rate. <code>ChargeIncrease</code> flags recent
    billing escalation; <code>ChargeIncreaseRatio</code> expresses it relative to
    historical spend.</p>
    <p><b>Tenure Signals:</b> <code>TenureSquared</code> and <code>TenureLog</code>
    model the non-linear lifecycle: churn risk is highest in early months and declines
    sub-linearly. Binary flags <code>IsNewCustomer</code> (&le;3 months) and
    <code>IsLongTermCustomer</code> (&ge;48 months) provide discrete lifecycle anchors.</p>
    <p><b>Compound Risk:</b> <code>HighRiskCombo</code> flags month-to-month plus
    electronic check customers. <code>VeryHighRisk</code> extends this by requiring
    tenure &le; 12 months. <code>MultipleRiskFactors</code> aggregates four binary
    risk signals into a single integer score.</p>

    <h2><span class="section-num">VI.</span> Model Development</h2>
    <h3>A. Random Forest</h3>
    <p>A Random Forest [22] builds an ensemble of T independent decision trees, each
    trained on a bootstrap sample with a random feature subset at each split node.
    Churn probability for subscriber <b>x</b>:</p>
    <div class="equation">P<sub>RF</sub>(<b>x</b>) = (1/T) &Sigma;<sub>t=1</sub><sup>T</sup> h<sub>t</sub>(<b>x</b>)<span class="eq-number">(1)</span></div>
    <p>Configuration: 200 trees, max depth 8, min samples per leaf 3, Gini impurity,
    balanced class weights. At &tau; = 0.38: <b>Recall 96.3%</b>, Accuracy 95.4%, ROC-AUC 98.7%.</p>

    <h3>B. XGBoost</h3>
    <p>XGBoost [10] trains trees sequentially to correct residuals, minimising the
    regularised objective:</p>
    <div class="equation">L<sup>(k)</sup> = &Sigma;<sub>i</sub> l(y<sub>i</sub>, &#375;<sub>i</sub><sup>(k-1)</sup> + h<sub>k</sub>(<b>x</b><sub>i</sub>)) + &Omega;(h<sub>k</sub>)<span class="eq-number">(2)</span></div>
    <p>where &Omega;(h<sub>k</sub>) = &gamma;T + &lambda;||<b>w</b>||&sup2;/2. Configuration:
    150 trees, lr 0.04, max depth 3, &lambda; = 2.0, &alpha; = 0.2. At &tau; = 0.29:
    <b>Recall 97.6%</b>, Accuracy 94.9%, ROC-AUC 99.0%, Brier 0.0347&#8212;highest recall
    and ROC-AUC of all evaluated models.</p>

    <h3>C. LightGBM</h3>
    <p>LightGBM [11] employs leaf-wise growth, splitting the leaf with the highest
    loss gain. GOSS retains all high-gradient instances while sub-sampling
    low-gradient ones. Configuration: 120 trees, lr 0.04, 24 leaves, max depth 3,
    L2 reg 2.5. At &tau; = 0.32: <b>Recall 96.9%</b>, Accuracy 95.0%, ROC-AUC 98.9%,
    Brier 0.0353.</p>

    <h3>D. Stacking Ensemble</h3>
    <p>The stacking architecture fuses three base classifiers through a Logistic
    Regression meta-learner trained on out-of-fold (OOF) predictions from 5-fold
    stratified cross-validation. The meta-feature matrix:</p>
    <div class="equation"><b>M</b> = [<b>p</b><sub>RF</sub> | <b>p</b><sub>XGB</sub> | <b>p</b><sub>LGBM</sub>] &isin; &Ropf;<sup>n&times;3</sup><span class="eq-number">(3)</span></div>
    <div class="equation">P<sub>Ens</sub>(<b>x</b>) = &sigma;(w<sub>0</sub> + w<sub>1</sub>p<sub>RF</sub> + w<sub>2</sub>p<sub>XGB</sub> + w<sub>3</sub>p<sub>LGBM</sub>)<span class="eq-number">(4)</span></div>

    <div class="figure-wrap">
      {'<img src="data:image/png;base64,' + fig2 + '" alt="Stacking Ensemble Architecture"/>' if fig2 else '<p>[Fig. 2: Stacking Architecture]</p>'}
      <div class="fig-caption">Fig. 2. Stacking ensemble: base classifiers generate OOF probability vectors assembled
      into the meta-feature matrix; Logistic Regression meta-learner produces final churn probability.</div>
    </div>

    <p>At &tau; = 0.26: <b>Recall 96.3%, F1 95.5%, Accuracy 95.4%, Precision 94.7%, ROC-AUC 98.9%, Brier 0.0362.</b></p>

    <h2><span class="section-num">VII.</span> Performance Metrics &amp; Threshold Optimization</h2>
    <h3>A. Evaluation Metrics</h3>
    <div class="equation">Accuracy = (TP+TN)/(TP+TN+FP+FN)<span class="eq-number">(5)</span></div>
    <div class="equation">Precision = TP/(TP+FP)<span class="eq-number">(6)</span></div>
    <div class="equation">Recall = TP/(TP+FN)<span class="eq-number">(7)</span></div>
    <div class="equation">F<sub>1</sub> = 2&middot;Precision&middot;Recall / (Precision+Recall)<span class="eq-number">(8)</span></div>
    <div class="equation">Brier = (1/n) &Sigma;<sub>i</sub> (p<sub>i</sub> &minus; y<sub>i</sub>)&sup2;<span class="eq-number">(9)</span></div>

    <h3>B. Decision Threshold Optimization</h3>
    <div class="equation">&#375;<sub>i</sub> = 1 if p<sub>i</sub> &ge; &tau;, else 0<span class="eq-number">(10)</span></div>

    <div class="table-caption">TABLE IV &mdash; Optimised Decision Thresholds</div>
    <table>
      <tr><th>Model</th><th>Threshold (&tau;)</th><th>vs. Default 0.50</th></tr>
      <tr><td>Random Forest</td><td>0.38</td><td>Lower</td></tr>
      <tr><td>XGBoost</td><td>0.29</td><td>Lower</td></tr>
      <tr><td>LightGBM</td><td>0.32</td><td>Lower</td></tr>
      <tr><td>Stacking Ensemble</td><td>0.26</td><td>Lower</td></tr>
    </table>

    <h2><span class="section-num">VIII.</span> Experimental Results</h2>

    <div class="table-caption">TABLE V &mdash; Model Performance on Held-Out Test Set (N = 1,600)</div>
    <table>
      <tr><th>Model</th><th>Recall (%)</th><th>F1 (%)</th><th>Acc. (%)</th><th>AUC (%)</th><th>Brier</th><th>Prec. (%)</th></tr>
      <tr>
        <td>Random Forest</td><td>96.3</td><td>95.4</td><td>95.4</td><td>98.7</td><td>0.0369</td><td>94.6</td>
      </tr>
      <tr>
        <td>XGBoost</td><td class="best">97.6</td><td>95.0</td><td>94.9</td><td class="best">99.0</td><td class="best">0.0347</td><td>92.5</td>
      </tr>
      <tr>
        <td>LightGBM</td><td>96.9</td><td>95.1</td><td class="best">95.0</td><td>98.9</td><td>0.0353</td><td>93.4</td>
      </tr>
      <tr>
        <td><b>Stacking Ens.</b></td><td>96.3</td><td class="best">95.5</td><td>95.4</td><td>98.9</td><td>0.0362</td><td class="best">94.7</td>
      </tr>
    </table>

    <div class="figure-wrap">
      {'<img src="data:image/png;base64,' + fig5 + '" alt="Model Performance Comparison"/>' if fig5 else '<p>[Fig. 5: Model Comparison Chart]</p>'}
      <div class="fig-caption">Fig. 3. Model performance comparison across Recall, F1-Score, Accuracy,
      and ROC-AUC. All models exceed 94.9% accuracy and 96.3% recall after threshold optimisation.</div>
    </div>

    <h2><span class="section-num">IX.</span> Results Analysis</h2>
    <p>All four classifiers achieve ROC-AUC between 98.7% and 99.0%, indicating
    near-perfect rank separation between churners and non-churners.</p>
    <p><b>XGBoost</b> records the highest recall (97.6%) and ROC-AUC (99.0%).
    It correctly identifies 97.6% of actual churners, leaving only 2.4% undetected.
    The Brier score of 0.0347 is also the lowest, suggesting the best-calibrated
    probability outputs. The trade-off is precision of 92.5%, reflecting the false-positive
    cost of the aggressive &tau; = 0.29 threshold.</p>
    <p><b>LightGBM</b> achieves 96.9% recall, the highest accuracy (95.0%), and a
    Brier score of 0.0353. Its leaf-wise growth strategy produces a favourable balance
    between false-negative reduction and false-positive control.</p>
    <p><b>Random Forest</b> achieves 96.3% recall and 95.4% accuracy with the most
    conservative threshold (0.38). Its precision of 94.6% is the second highest.</p>
    <p><b>Stacking Ensemble</b> achieves the highest F1-score (95.5%) and precision
    (94.7%), with ROC-AUC 98.9% and Brier 0.0362. The meta-learner successfully
    reduces the false-positive rate relative to individual gradient-boosted models,
    producing a more stable profile for operational deployment.</p>

    <div class="table-caption">TABLE VI &mdash; Error Rate Summary (%)</div>
    <table>
      <tr><th>Model</th><th>Acc. Err</th><th>Prec. Err</th><th>Rec. Err</th><th>F1 Err</th></tr>
      <tr><td>Random Forest</td><td>4.6</td><td>5.4</td><td>3.7</td><td>4.6</td></tr>
      <tr><td>XGBoost</td><td>5.1</td><td>7.5</td><td class="best">2.4</td><td>5.0</td></tr>
      <tr><td>LightGBM</td><td>5.0</td><td>6.6</td><td>3.1</td><td>4.9</td></tr>
      <tr><td><b>Stacking (Prop.)</b></td><td class="best">4.6</td><td class="best">5.3</td><td>3.7</td><td class="best">4.5</td></tr>
    </table>

    <h2><span class="section-num">X.</span> System Implementation</h2>

    <div class="table-caption">TABLE VII &mdash; Implementation Technology Stack</div>
    <table>
      <tr><th>Layer</th><th>Technology</th><th>Role</th></tr>
      <tr><td>Frontend</td><td>React (Vite)</td><td>UI, routing, visualisation</td></tr>
      <tr><td>Frontend</td><td>React Router</td><td>Multi-page navigation</td></tr>
      <tr><td>Backend</td><td>FastAPI</td><td>REST API, prediction serving</td></tr>
      <tr><td>ML Runtime</td><td>Python 3.11</td><td>Core computation</td></tr>
      <tr><td>Ensemble ML</td><td>scikit-learn</td><td>RF, Logistic Regression</td></tr>
      <tr><td>Ensemble ML</td><td>XGBoost 4.x</td><td>Gradient-boosted classifier</td></tr>
      <tr><td>Ensemble ML</td><td>LightGBM 4.7</td><td>Leaf-wise boosted classifier</td></tr>
      <tr><td>Explainability</td><td>SHAP 0.51</td><td>Global and local attributions</td></tr>
      <tr><td>Explainability</td><td>LIME 0.2</td><td>Local surrogate explanations</td></tr>
      <tr><td>Database</td><td>SQLite</td><td>Prediction history</td></tr>
    </table>

    <p>The web interface exposes five modules: Single Customer Prediction, Batch
    Upload, What-If Simulator, Dashboard, and Model Comparison.</p>

    <h2><span class="section-num">XI.</span> Explainable AI &amp; Retention Recommendations</h2>
    <h3>A. SHAP Feature Attribution</h3>
    <p>For subscriber <i>i</i> with <i>p</i> features, SHAP [15] satisfies:</p>
    <div class="equation">f(<b>x</b><sup>(i)</sup>) = &phi;<sub>0</sub> + &Sigma;<sub>j=1</sub><sup>p</sup> &phi;<sub>j</sub><sup>(i)</sup><span class="eq-number">(11)</span></div>
    <p>Features with positive &phi;<sub>j</sub> push toward churn; negative values push toward
    retention. The system uses SHAP TreeExplainer (exact for tree-based models). Global
    rankings are obtained by averaging |&phi;<sub>j</sub><sup>(i)</sup>| across the dataset.</p>

    <h3>B. LIME Explanation Service</h3>
    <p>LIME [16] approximates the ensemble with a sparse linear surrogate near each prediction:</p>
    <div class="equation">&xi;(<b>x</b><sub>0</sub>) = argmin<sub>g</sub> L(f, g, &pi;<sub><b>x</b><sub>0</sub></sub>) + &Omega;(g)<span class="eq-number">(12)</span></div>

    <h3>C. Retention Recommendation Engine</h3>
    <div class="figure-wrap">
      {'<img src="data:image/png;base64,' + fig7 + '" alt="Retention Recommendation Workflow"/>' if fig7 else '<p>[Fig. 7: Retention Workflow]</p>'}
      <div class="fig-caption">Fig. 4. Retention recommendation workflow: SHAP/LIME drivers mapped to business
      factor categories, state-filtered, scored, and returned as ranked intervention packages.</div>
    </div>

    <p>The engine operates in four stages: (1) Risk Classification (Low/Medium/High/Critical);
    (2) Driver Extraction (SHAP+LIME &rarr; business factor categories); (3) State-Aware
    Filtering (prune inapplicable interventions); (4) Prioritised Ranking by composite score.
    Interventions include: contract upgrade incentives, automatic payment migration, billing
    discount review, onboarding support, service bundles, technical support, and fibre
    quality investigation.</p>

    <h3>D. What-If Simulator &amp; Batch Prediction</h3>
    <p>The What-If Simulator allows analysts to modify customer attributes and observe
    how the ensemble probability responds, without asserting causal effects. Batch
    Upload enables CSV-based multi-customer prediction with downloadable results
    including probability, risk level, and primary SHAP driver.</p>

    <h2><span class="section-num">XII.</span> Discussion</h2>
    <p>The consistently strong performance of all three tree-based classifiers reflects
    the suitability of gradient-boosted and bagged ensembles for tabular CRM data.
    The stacking ensemble's value lies in robustness: achieving the highest F1 and
    precision while matching other models on recall indicates the meta-learner
    successfully identifies regions where each base classifier is most reliable.</p>
    <p>Decision threshold optimisation proved essential. Operating all models at 0.50
    would produce substantially lower recall&#8212;a trade-off that is operationally
    sub-optimal when the cost of missing a churner exceeds unnecessary outreach cost.</p>
    <p>The dual SHAP and LIME layer enables retention agents to understand <i>why</i> a
    subscriber was flagged, enabling more targeted interventions than a probability
    label alone provides. Results apply to the balanced benchmark and require
    recalibration for real-world 15&#8211;30% churn prevalence.</p>

    <h2><span class="section-num">XIII.</span> Limitations</h2>
    <p>The 50/50 class distribution is artificial; production deployments would alter
    thresholds and false-positive rates. All evaluation uses a single dataset; external
    validation is needed. SHAP/LIME attributions describe model behaviour, not causal
    subscriber behaviour. Retention recommendations are heuristic and untested in
    controlled field trials.</p>

    <h2><span class="section-num">XIV.</span> Conclusion</h2>
    <p>This work developed an end-to-end explainable ensemble learning framework for
    telecom customer churn prediction. Starting from 8,000 balanced subscriber records,
    feature engineering expanded 21 raw attributes to 83 model-ready features. Three
    tree classifiers were combined through an out-of-fold Stacking Ensemble. Per-model
    thresholds were optimised: XGBoost achieved 97.6% recall and 99.0% ROC-AUC;
    LightGBM recorded 96.9% recall and 98.9% ROC-AUC; Random Forest delivered 96.3%
    recall and 98.7% ROC-AUC; and the Stacking Ensemble produced 96.3% recall, 95.5%
    F1, 94.7% precision, 98.9% ROC-AUC, and Brier 0.0362.</p>
    <p>The dual SHAP and LIME pipeline feeds a state-aware retention recommendation
    engine, deployed as a production FastAPI/React/SQLite application. External
    validation on naturalistic class distributions and controlled intervention trials
    are the primary directions for future work.</p>

    <h2><span class="section-num">XV.</span> Future Enhancement</h2>
    <p>Future work will explore: production deployment validation; isotonic regression
    calibration for natural 15&#8211;30% churn prevalence; longitudinal dataset
    expansion with time-series usage features; uplift modeling for individual treatment
    effect estimation; customer lifetime value integration; causal inference for
    structural driver identification; concept drift monitoring with automated
    retraining; online learning for near-real-time model refinement; and closed-loop
    intervention effectiveness tracking.</p>

  </div><!-- end columns -->

  <div class="references">
    <h2>References</h2>
    <ol class="ref-list">
      <li>[1] M. Xu, J. Wong, K. J. Lim, and F. Pan, "Customer churn prediction for the telecommunications sector using machine learning techniques," <i>Information Sciences</i>, vol. 571, pp. 45&#8211;62, 2021.</li>
      <li>[2] A. Amin et al., "Customer churn prediction in telecommunication industry using data certainty," <i>Journal of Business Research</i>, vol. 94, pp. 290&#8211;301, 2019.</li>
      <li>[3] K. B. Bhatt and P. Thaker, "Churn prediction of telecom industry using machine learning techniques," in <i>Proc. Int. Conf. Innov. Inf. Technol.</i>, 2020, pp. 112&#8211;117.</li>
      <li>[4] K. Coussement, K. W. De Bock, and D. F. Benoit, "Benchmarking data mining algorithms for customer churn prediction," <i>Eur. J. Oper. Res.</i>, vol. 257, no. 1, pp. 216&#8211;235, 2017.</li>
      <li>[5] P. Lalwani et al., "Customer churn prediction system: A machine learning approach," <i>Computing</i>, vol. 104, no. 2, pp. 271&#8211;294, 2022.</li>
      <li>[6] V. Mikalef et al., "Predictive analytics and machine learning in direct marketing," <i>Ann. Oper. Res.</i>, vol. 306, pp. 285&#8211;306, 2021.</li>
      <li>[7] W. Verbeke et al., "New insights into churn prediction in the telecommunication sector: A profit driven data mining approach," <i>Eur. J. Oper. Res.</i>, vol. 218, no. 1, pp. 211&#8211;229, 2012.</li>
      <li>[8] J. Hadden et al., "Computer assisted customer churn management: State-of-the-art and future trends," <i>Comput. Oper. Res.</i>, vol. 34, no. 10, pp. 2902&#8211;2917, 2007.</li>
      <li>[9] W. Verbeke, D. Martens, C. Mues, and B. Baesens, "Building comprehensible customer churn prediction models," <i>Expert Syst. Appl.</i>, vol. 38, no. 3, pp. 2354&#8211;2364, 2011.</li>
      <li>[10] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. 22nd ACM SIGKDD Conf.</i>, 2016, pp. 785&#8211;794.</li>
      <li>[11] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in <i>Adv. Neural Inf. Process. Syst.</i>, 2017, pp. 3146&#8211;3154.</li>
      <li>[12] M. A. H. Farquad, V. Ravi, and S. B. Raju, "Churn prediction using comprehensible SVM: An analytical CRM application," <i>Appl. Soft Comput.</i>, vol. 19, pp. 31&#8211;40, 2014.</li>
      <li>[13] N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," <i>J. Artif. Intell. Res.</i>, vol. 16, pp. 321&#8211;357, 2002.</li>
      <li>[14] G. Haixiang et al., "Learning from class-imbalanced data: Review of methods and applications," <i>Expert Syst. Appl.</i>, vol. 73, pp. 220&#8211;239, 2017.</li>
      <li>[15] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in <i>Adv. Neural Inf. Process. Syst.</i>, 2017, pp. 4765&#8211;4774.</li>
      <li>[16] M. T. Ribeiro, S. Singh, and C. Guestrin, "Why should I trust you? Explaining the predictions of any classifier," in <i>Proc. 22nd ACM SIGKDD Conf.</i>, 2016, pp. 1135&#8211;1144.</li>
      <li>[17] M. Fernandez-Delgado et al., "Do we need hundreds of classifiers to solve real world classification problems?," <i>J. Mach. Learn. Res.</i>, vol. 15, no. 1, pp. 3133&#8211;3181, 2014.</li>
      <li>[18] D. H. Wolpert, "Stacked generalization," <i>Neural Networks</i>, vol. 5, no. 2, pp. 241&#8211;259, 1992.</li>
      <li>[19] Z.-H. Zhou, <i>Ensemble Methods: Foundations and Algorithms</i>. CRC Press, 2012.</li>
      <li>[20] T. Fawcett, "An introduction to ROC analysis," <i>Pattern Recognit. Lett.</i>, vol. 27, no. 8, pp. 861&#8211;874, 2006.</li>
      <li>[21] F. Devriendt, D. Moldovan, and W. Verbeke, "A literature survey and experimental evaluation of the state-of-the-art in uplift modeling," <i>Big Data</i>, vol. 6, no. 1, pp. 13&#8211;41, 2018.</li>
      <li>[22] L. Breiman, "Random forests," <i>Mach. Learn.</i>, vol. 45, no. 1, pp. 5&#8211;32, 2001.</li>
    </ol>
  </div>

</div><!-- end paper-container -->
</body>
</html>
"""

html_path = os.path.join(ROOT, 'Explainable_Ensemble_Learning_Telecom_Churn_Final.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f"[OK] HTML written: {html_path}")
print(f"     Size: {os.path.getsize(html_path):,} bytes")
print("[DONE] HTML generation complete.")

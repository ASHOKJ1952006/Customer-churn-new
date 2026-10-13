# Section III Draft: Datasets and Feature Engineering

*(Neutral IEEE Conference Style — Ready for Inclusion in Main Manuscript)*

---

## III. DATASETS AND FEATURE SPECIFICATION

To ensure empirical validity, reproducibility, and practical operational viability, this research conducts a dual-benchmark investigation combining: (1) a controlled, domain-consistent synthetic telecommunications benchmark designed to evaluate feature interaction dynamics and multi-model boundary behaviors under known generative parameters; and (2) the canonical public IBM Telco Customer Churn benchmark dataset ($N=7,043$) reflecting real-world class imbalance, heterogeneous payment behaviors, and noise distributions.

### A. Synthetic Benchmark Generator (`balanced_telco_churn.csv`)

The controlled synthetic benchmark comprises $N=8,000$ subscriber records generated under calibrated domain rules reflecting telecommunications CRM distributions. Unlike heuristic data generators, the process models churn defection as an outcome of non-linear service friction, tenure maturity, and pricing shocks.

The generation protocol is formalized as follows:
1. **Target Stratification:** The dataset is partitioned into exactly $N_0 = 4,000$ loyal subscribers ($Y=0$) and $N_1 = 4,000$ defecting subscribers ($Y=1$).
2. **Feature Distributions by Class:**
   * *Tenure ($T$):* Loyal subscribers follow a mature tenure distribution $T \sim \mathcal{N}(\mu=46, \sigma=18)$ clipped to $[6, 72]$ months, whereas churners exhibit an exponential early-stage attrition profile $T \sim \text{Exp}(\lambda=1/9) + 1$ clipped to $[1, 32]$ months.
   * *Contract Type ($C$):* Loyal subscribers predominantly hold multi-year commitments (Two-year: 52%, One-year: 35%, Month-to-month: 13%), whereas churners are heavily concentrated in Month-to-month arrangements (90%) and One-year contracts (10%).
   * *Payment Friction ($P$):* Loyal accounts utilize automated clearinghouse transactions (Credit card: 42%, Bank transfer: 40%, Mailed check: 12%, Electronic check: 6%), whereas churners exhibit high electronic check exposure (70%).
   * *Monthly Charges ($MC$):* Base spend is conditioned on broadband tier: DSL ($\mu=\$55–\$65$), Fiber optic ($\mu=\$82–\$90$), or No internet ($\mu=\$25–\$30$), with additive Gaussian variance. Total lifetime charges ($TC$) satisfy $TC = MC \cdot T \cdot \epsilon$, where $\epsilon \sim \mathcal{U}(0.92, 1.05)$.
3. **Calibrated Noise Floor (The 5% Noise Ceiling):** To model imperfect enterprise observability and human behavioral variance, a cross-segment noise parameter $\eta = 0.05$ (5.0%) is injected into each class. With probability $\eta$, loyal accounts inherit high-risk traits ($T \sim \mathcal{N}(18, 10)$, 70% month-to-month), and churn accounts inherit mature traits ($T \sim \mathcal{N}(32, 14)$, multi-year contracts). This injection imposes a theoretical Bayes error rate of $\approx 5.0\%$, establishing an artificial accuracy ceiling of $\approx 95.0\%$.

### B. Real Public Benchmark: IBM Telco Customer Churn

To validate framework generalizability beyond synthetic distributions, all pipelines are evaluated against the public IBM Telco benchmark ($N=7,043$). The dataset exhibits natural enterprise skew: 5,174 non-churners (73.46%) and 1,869 churners (26.54%). TotalCharges contains 11 whitespace entries corresponding to new subscribers ($T=0$), which are deterministically imputed to zero.

### C. Six Core Production Inputs vs. 27-Dimensional Model Space

Prior research frequently assumes customer service agents have real-time access to 20+ static CRM fields. In production CRM environments, querying dozens of multi-service tables incurs severe latency penalties. To bridge this operational gap, our pipeline requires strictly **six operational input attributes** readily accessible from real-time billing APIs:

$$\mathcal{X}_{\text{raw}} = \{\text{tenure}, \text{MonthlyCharges}, \text{TotalCharges}, \text{Contract}, \text{InternetService}, \text{PaymentMethod}\}$$

From these six raw inputs, the deterministic feature engineering engine derives 21 domain-engineered mathematical features, resulting in an exact **27-dimensional feature space**:

$$\mathcal{X}_{\text{model}} \in \mathbb{R}^{27} = \mathcal{X}_{\text{scaled}}^{(11)} \cup \mathcal{X}_{\text{binary}}^{(9)} \cup \mathcal{X}_{\text{dummy}}^{(7)}$$

1. **Continuous Scaled Predictors ($\mathcal{X}_{\text{scaled}} \in \mathbb{R}^{11}$):** Normalized via $\mu=0, \sigma=1$ `StandardScaler`:
   * Baseline accounting: `tenure`, `MonthlyCharges`, `TotalCharges`
   * Spend dynamics: `AverageMonthlySpend` ($TC / \max(T, 1)$), `ChargeIncrease` ($MC - \text{AvgSpend}$), and relative `ChargeIncreaseRatio`
   * Non-linear lifecycle: `TenureSquared` ($T^2$) and `TenureLog` ($\ln(1+T)$)
   * Financial-tenure cross-terms: $T \times MC$, $T \times \mathbb{I}(\text{M2M})$, and $MC \times \mathbb{I}(\text{M2M})$
2. **Binary Risk Interaction Flags ($\mathcal{X}_{\text{binary}} \in \{0, 1\}^9$):**
   * Account lifecycle flags: `IsNewCustomer` ($T \le 12$), `IsLongTermCustomer` ($T \ge 48$)
   * Friction indicators: `IsMonthToMonth`, `IsLongTermContract`, `IsElectronicCheck`
   * Compounded risk: `HighRiskCombo` ($\text{M2M} \land \text{E-Check}$), `VeryHighRisk` ($\text{HighRiskCombo} \land T \le 12$)
   * Service-billing cross-terms: `FiberOptic_x_MonthToMonth`, `FiberOptic_x_ElectronicCheck`
3. **One-Hot Encoded Categorical Dummies ($\mathcal{X}_{\text{dummy}} \in \{0, 1\}^7$):**
   * Contract: `One year`, `Two year` (Month-to-month reference baseline)
   * Internet Service: `Fiber optic`, `No` (DSL reference baseline)
   * Payment Method: `Credit card (auto)`, `Electronic check`, `Mailed check` (Bank transfer reference baseline)

This design ensures complete mathematical consistency: every model input dimension is unambiguously accounted for, and single-customer inference requires only six inputs.

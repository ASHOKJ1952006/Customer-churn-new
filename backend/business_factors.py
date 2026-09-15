"""
business_factors.py — Centralized Business Factor Dictionary & Taxonomy
Maps technical ML features to Telecom Business Domains, defines actionability,
business rationale, priority weights, owner teams, cost/impact categories, and simulation rules.
"""

from typing import Dict, Any, List, Optional

# ==============================================================================
# 1. TELECOM BUSINESS DOMAIN CATEGORIES
# ==============================================================================
CATEGORY_CONTRACT = "Contract & Commitment Risk"
CATEGORY_PAYMENT = "Payment & Billing Friction"
CATEGORY_PRICING = "Pricing & Billing Cost"
CATEGORY_LIFECYCLE = "Customer Lifecycle & Tenure"
CATEGORY_SERVICE_TYPE = "Service Type & Technology"
CATEGORY_SERVICE_EXP = "Service Experience & Support"
CATEGORY_ENGAGEMENT = "Service Engagement"
CATEGORY_COMPOUND = "Compounded Multi-Factor Risk"
CATEGORY_DEMOGRAPHIC = "Customer Demographics"

# ==============================================================================
# 2. FEATURE TO BUSINESS FACTOR MAPPING
# ==============================================================================
FEATURE_BUSINESS_MAPPING: Dict[str, Dict[str, Any]] = {
    # Contract Commitment
    'Contract_Month-to-month': {
        'category': CATEGORY_CONTRACT,
        'business_name': 'Month-to-Month Contract',
        'is_actionable': True,
        'meaning': 'Customer has no long-term commitment and can terminate service at any billing cycle.',
        'risk_direction': 'increases_churn',
    },
    'Contract_One year': {
        'category': CATEGORY_CONTRACT,
        'business_name': 'One-Year Contract Commitment',
        'is_actionable': True,
        'meaning': 'Moderate contract lock-in defending against immediate switching.',
        'risk_direction': 'decreases_churn',
    },
    'Contract_Two year': {
        'category': CATEGORY_CONTRACT,
        'business_name': 'Two-Year Contract Commitment',
        'is_actionable': True,
        'meaning': 'Strong contractual lock-in providing high account stability.',
        'risk_direction': 'decreases_churn',
    },
    'ContractLength': {
        'category': CATEGORY_CONTRACT,
        'business_name': 'Contract Duration',
        'is_actionable': True,
        'meaning': 'Contractual duration commitments directly govern churn exposure.',
        'risk_direction': 'decreases_churn',
    },
    'IsMonthToMonth': {
        'category': CATEGORY_CONTRACT,
        'business_name': 'Flexible Month-to-Month Term',
        'is_actionable': True,
        'meaning': 'Absence of long-term contract lock-in.',
        'risk_direction': 'increases_churn',
    },
    'IsLongTermContract': {
        'category': CATEGORY_CONTRACT,
        'business_name': 'Long-Term Contract Status',
        'is_actionable': True,
        'meaning': 'Established annual or multi-year commitment.',
        'risk_direction': 'decreases_churn',
    },

    # Payment & Billing Friction
    'PaymentMethod_Electronic check': {
        'category': CATEGORY_PAYMENT,
        'business_name': 'Electronic Check Payment Method',
        'is_actionable': True,
        'meaning': 'Manual, non-automated payment with high friction, missed payments, and switching ease.',
        'risk_direction': 'increases_churn',
    },
    'PaymentMethod_Mailed check': {
        'category': CATEGORY_PAYMENT,
        'business_name': 'Mailed Check Payment Method',
        'is_actionable': True,
        'meaning': 'Traditional manual paper billing with payment delay risk.',
        'risk_direction': 'increases_churn',
    },
    'PaymentMethod_Bank transfer (automatic)': {
        'category': CATEGORY_PAYMENT,
        'business_name': 'Bank Transfer Autopay',
        'is_actionable': True,
        'meaning': 'Automated recurring bank debit fostering friction-free billing stability.',
        'risk_direction': 'decreases_churn',
    },
    'PaymentMethod_Credit card (automatic)': {
        'category': CATEGORY_PAYMENT,
        'business_name': 'Credit Card Autopay',
        'is_actionable': True,
        'meaning': 'Automated recurring card payment reducing missed billing cycles.',
        'risk_direction': 'decreases_churn',
    },
    'IsElectronicCheck': {
        'category': CATEGORY_PAYMENT,
        'business_name': 'Electronic Check Usage',
        'is_actionable': True,
        'meaning': 'Electronic check usage flags manual billing friction.',
        'risk_direction': 'increases_churn',
    },
    'IsAutomaticPayment': {
        'category': CATEGORY_PAYMENT,
        'business_name': 'Autopay Enrollment',
        'is_actionable': True,
        'meaning': 'Autopay enrollment provides seamless renewal continuity.',
        'risk_direction': 'decreases_churn',
    },
    'PaperlessBilling': {
        'category': CATEGORY_PAYMENT,
        'business_name': 'Paperless Billing Preference',
        'is_actionable': True,
        'meaning': 'Digital billing channel interaction.',
        'risk_direction': 'increases_churn',
    },

    # Pricing & Spend
    'MonthlyCharges': {
        'category': CATEGORY_PRICING,
        'business_name': 'Monthly Service Charges',
        'is_actionable': True,
        'meaning': 'High recurring monthly fee creating price sensitivity and budget strain.',
        'risk_direction': 'increases_churn',
    },
    'TotalCharges': {
        'category': CATEGORY_PRICING,
        'business_name': 'Cumulative Lifetime Spend',
        'is_actionable': False,
        'meaning': 'Historical revenue contribution and sunk investment in telecom account.',
        'risk_direction': 'decreases_churn',
    },
    'ChargeIncrease': {
        'category': CATEGORY_PRICING,
        'business_name': 'Recent Monthly Charge Increase',
        'is_actionable': True,
        'meaning': 'Current monthly bill exceeds historical average, causing bill shock or fee resentment.',
        'risk_direction': 'increases_churn',
    },
    'ChargeIncreaseRatio': {
        'category': CATEGORY_PRICING,
        'business_name': 'Charge Spike Ratio',
        'is_actionable': True,
        'meaning': 'Proportional bill inflation causing perceived overcharging.',
        'risk_direction': 'increases_churn',
    },
    'AvgMonthlySpend': {
        'category': CATEGORY_PRICING,
        'business_name': 'Average Historical Spend',
        'is_actionable': True,
        'meaning': 'Baseline spending trajectory across customer lifetime.',
        'risk_direction': 'neutral',
    },
    'MonthlyChargesPerTenure': {
        'category': CATEGORY_PRICING,
        'business_name': 'Monthly Rate Relative to Tenure',
        'is_actionable': True,
        'meaning': 'High early-tenure rate increases dissatisfaction before loyalty builds.',
        'risk_direction': 'increases_churn',
    },
    'TotalChargesPerTenure': {
        'category': CATEGORY_PRICING,
        'business_name': 'Average Monthly Realized Spend',
        'is_actionable': False,
        'meaning': 'Realized effective bill amount per active month.',
        'risk_direction': 'neutral',
    },
    'ChargeToTenureRatio': {
        'category': CATEGORY_PRICING,
        'business_name': 'Charge-to-Tenure Cost Ratio',
        'is_actionable': True,
        'meaning': 'Spending pressure relative to account maturity.',
        'risk_direction': 'increases_churn',
    },
    'TotalChargesDifference': {
        'category': CATEGORY_PRICING,
        'business_name': 'Bill Trajectory Variance',
        'is_actionable': True,
        'meaning': 'Discrepancy between actual billing and steady-state expectations.',
        'risk_direction': 'increases_churn',
    },

    # Customer Lifecycle & Tenure
    'tenure': {
        'category': CATEGORY_LIFECYCLE,
        'business_name': 'Customer Tenure (Months Active)',
        'is_actionable': False,
        'meaning': 'Overall relationship duration. Early tenure (<12 mo) lacks switching friction; mature tenure builds loyalty.',
        'risk_direction': 'decreases_churn',
    },
    'TenureSquared': {
        'category': CATEGORY_LIFECYCLE,
        'business_name': 'Non-linear Lifecycle Maturity',
        'is_actionable': False,
        'meaning': 'Long-term compounding retention effect of established tenure.',
        'risk_direction': 'decreases_churn',
    },
    'TenureLog': {
        'category': CATEGORY_LIFECYCLE,
        'business_name': 'Logarithmic Tenure Maturity',
        'is_actionable': False,
        'meaning': 'Diminishing marginal churn sensitivity as relationship deepens.',
        'risk_direction': 'decreases_churn',
    },
    'IsNewCustomer': {
        'category': CATEGORY_LIFECYCLE,
        'business_name': 'New Customer Onboarding Window (<=12 Months)',
        'is_actionable': False,
        'meaning': 'High vulnerability period where brand loyalty is not yet established.',
        'risk_direction': 'increases_churn',
    },
    'IsLoyalCustomer': {
        'category': CATEGORY_LIFECYCLE,
        'business_name': 'Established Loyal Customer (>=36 Months)',
        'is_actionable': False,
        'meaning': 'Deeply embedded account with significant habitual stickiness.',
        'risk_direction': 'decreases_churn',
    },
    'TenureGroup_7-12mo': {
        'category': CATEGORY_LIFECYCLE,
        'business_name': 'Tenure 7-12 Months Window',
        'is_actionable': False,
        'meaning': 'Critical decision window nearing first annual milestone.',
        'risk_direction': 'increases_churn',
    },
    'TenureGroup_13-24mo': {
        'category': CATEGORY_LIFECYCLE,
        'business_name': 'Tenure 1-2 Years',
        'is_actionable': False,
        'meaning': 'Transition phase from onboarding to established customer.',
        'risk_direction': 'decreases_churn',
    },
    'TenureGroup_25-48mo': {
        'category': CATEGORY_LIFECYCLE,
        'business_name': 'Tenure 2-4 Years',
        'is_actionable': False,
        'meaning': 'Stable mid-lifecycle customer segment.',
        'risk_direction': 'decreases_churn',
    },
    'TenureGroup_49-72mo': {
        'category': CATEGORY_LIFECYCLE,
        'business_name': 'Tenure 4+ Years',
        'is_actionable': False,
        'meaning': 'Highest loyalty tier with strong brand retention.',
        'risk_direction': 'decreases_churn',
    },

    # Service Type
    'InternetService_Fiber optic': {
        'category': CATEGORY_SERVICE_TYPE,
        'business_name': 'Fiber Optic Internet Tier',
        'is_actionable': True,
        'meaning': 'Premium high-speed internet tier; often subject to intense competitor poaching or service/price sensitivity.',
        'risk_direction': 'increases_churn',
    },
    'InternetService_DSL': {
        'category': CATEGORY_SERVICE_TYPE,
        'business_name': 'DSL Internet Tier',
        'is_actionable': True,
        'meaning': 'Standard copper broadband; lower speed but typically lower price point and moderate churn.',
        'risk_direction': 'decreases_churn',
    },
    'InternetService_No': {
        'category': CATEGORY_SERVICE_TYPE,
        'business_name': 'No Internet Service (Voice Only)',
        'is_actionable': False,
        'meaning': 'Voice-only customer with low monthly billing and low churn volatility.',
        'risk_direction': 'decreases_churn',
    },
    'HasInternet': {
        'category': CATEGORY_SERVICE_TYPE,
        'business_name': 'Active Broadband Subscription',
        'is_actionable': True,
        'meaning': 'Broadband subscriber subject to internet service quality perception.',
        'risk_direction': 'neutral',
    },
    'PhoneService': {
        'category': CATEGORY_SERVICE_TYPE,
        'business_name': 'Fixed Telephone Service',
        'is_actionable': True,
        'meaning': 'Standard voice line connection.',
        'risk_direction': 'neutral',
    },
    'MultipleLines_Yes': {
        'category': CATEGORY_SERVICE_TYPE,
        'business_name': 'Multiple Telephone Lines',
        'is_actionable': True,
        'meaning': 'Household or business multi-line usage enhancing account value.',
        'risk_direction': 'decreases_churn',
    },

    # Service Experience & Support
    'TechSupport_Yes': {
        'category': CATEGORY_SERVICE_EXP,
        'business_name': 'Dedicated Tech Support Add-on',
        'is_actionable': True,
        'meaning': 'Active technical support coverage resolving connectivity issues promptly and building stickiness.',
        'risk_direction': 'decreases_churn',
    },
    'TechSupport_No': {
        'category': CATEGORY_SERVICE_EXP,
        'business_name': 'Lack of Dedicated Tech Support',
        'is_actionable': True,
        'meaning': 'Customer has no technical assistance, risking frustration when technical outages or router issues occur.',
        'risk_direction': 'increases_churn',
    },
    'OnlineSecurity_Yes': {
        'category': CATEGORY_SERVICE_EXP,
        'business_name': 'Online Security Protection Add-on',
        'is_actionable': True,
        'meaning': 'Network security protection strengthening product value and trust.',
        'risk_direction': 'decreases_churn',
    },
    'OnlineSecurity_No': {
        'category': CATEGORY_SERVICE_EXP,
        'business_name': 'Lack of Online Security Protection',
        'is_actionable': True,
        'meaning': 'Missing security add-on represents an unaddressed account risk and lower product stickiness.',
        'risk_direction': 'increases_churn',
    },
    'OnlineBackup_Yes': {
        'category': CATEGORY_SERVICE_EXP,
        'business_name': 'Cloud Backup Subscription',
        'is_actionable': True,
        'meaning': 'Customer cloud data stored on telecom infrastructure enhances switching costs.',
        'risk_direction': 'decreases_churn',
    },
    'DeviceProtection_Yes': {
        'category': CATEGORY_SERVICE_EXP,
        'business_name': 'Device Protection Warranty',
        'is_actionable': True,
        'meaning': 'Hardware warranty coverage increases account stickiness.',
        'risk_direction': 'decreases_churn',
    },

    # Service Engagement
    'StreamingTV_Yes': {
        'category': CATEGORY_ENGAGEMENT,
        'business_name': 'Streaming TV Service',
        'is_actionable': True,
        'meaning': 'High-bandwidth digital entertainment service increasing household utility.',
        'risk_direction': 'neutral',
    },
    'StreamingMovies_Yes': {
        'category': CATEGORY_ENGAGEMENT,
        'business_name': 'Streaming Movies Service',
        'is_actionable': True,
        'meaning': 'On-demand film streaming engagement fostering multi-screen usage.',
        'risk_direction': 'neutral',
    },
    'NumServices': {
        'category': CATEGORY_ENGAGEMENT,
        'business_name': 'Total Number of Subscribed Services',
        'is_actionable': True,
        'meaning': 'Multi-product penetration; accounts with multiple integrated services exhibit much higher retention.',
        'risk_direction': 'decreases_churn',
    },

    # Compounded Multi-Factor Risk
    'HighRiskCombo': {
        'category': CATEGORY_COMPOUND,
        'business_name': 'Acute High-Risk Profile (M2M + Electronic Check)',
        'is_actionable': True,
        'meaning': 'Dangerous convergence of zero contract lock-in and high-friction electronic check billing.',
        'risk_direction': 'increases_churn',
    },
    'VeryHighRisk': {
        'category': CATEGORY_COMPOUND,
        'business_name': 'Compounded Fragility (New Customer + High-Risk Combo)',
        'is_actionable': True,
        'meaning': 'Extreme churn probability: new tenure combined with month-to-month and electronic check.',
        'risk_direction': 'increases_churn',
    },
    'FiberAndMonthToMonth': {
        'category': CATEGORY_COMPOUND,
        'business_name': 'Fiber Internet on Month-to-Month Term',
        'is_actionable': True,
        'meaning': 'High recurring expense without contract lock-in makes customer vulnerable to competitor fiber deals.',
        'risk_direction': 'increases_churn',
    },
    'FiberAndElectronicCheck': {
        'category': CATEGORY_COMPOUND,
        'business_name': 'Fiber Internet with Electronic Check',
        'is_actionable': True,
        'meaning': 'Premium high-speed fee combined with manual payment friction.',
        'risk_direction': 'increases_churn',
    },
    'HighChargesAndMonthToMonth': {
        'category': CATEGORY_COMPOUND,
        'business_name': 'Premium Charges on Month-to-Month Contract',
        'is_actionable': True,
        'meaning': 'Customer pays top-tier rates without any contractual loyalty discount or protection.',
        'risk_direction': 'increases_churn',
    },

    # Demographic Features (Responsible AI Audit - Non Actionable)
    'gender': {
        'category': CATEGORY_DEMOGRAPHIC,
        'business_name': 'Customer Gender',
        'is_actionable': False,
        'meaning': 'Demographic profile for model auditing only; strictly prohibited as a basis for retention actions.',
        'risk_direction': 'neutral',
    },
    'SeniorCitizen': {
        'category': CATEGORY_DEMOGRAPHIC,
        'business_name': 'Senior Citizen Demographic',
        'is_actionable': False,
        'meaning': 'Protected demographic indicator; actions must focus on service suitability and usability rather than age.',
        'risk_direction': 'neutral',
    },
    'Partner': {
        'category': CATEGORY_DEMOGRAPHIC,
        'business_name': 'Household Partner Status',
        'is_actionable': False,
        'meaning': 'Household demographic indicator; multi-member households often benefit from family plans.',
        'risk_direction': 'neutral',
    },
    'Dependents': {
        'category': CATEGORY_DEMOGRAPHIC,
        'business_name': 'Family Dependents Status',
        'is_actionable': False,
        'meaning': 'Household composition indicator for family plan eligibility.',
        'risk_direction': 'neutral',
    },
}

# ==============================================================================
# 3. RETENTION ACTION CATALOG (SPECIFIC TELECOM INTERVENTIONS)
# ==============================================================================
RETENTION_ACTIONS_CATALOG: Dict[str, Dict[str, Any]] = {
    # Action 1: Month-to-Month -> Contract Lock-in
    'ACT_CONTRACT_UPGRADE': {
        'action_id': 'ACT_CONTRACT_UPGRADE',
        'title': 'Contract Commitment Upgrade Offer',
        'category': CATEGORY_CONTRACT,
        'action': 'Offer a 12-month or 24-month contract renewal package with a targeted loyalty rate or streaming add-on credit.',
        'why': 'Customer is on a flexible month-to-month contract, which is the single largest structural driver of churn exposure.',
        'business_objective': 'Establish contract lock-in, eliminate immediate cancellation vulnerability, and secure predictable annual revenue.',
        'owner': 'Retention Team',
        'priority': 'HIGH',
        'cost': 'MEDIUM',
        'impact': 'HIGH',
        'timeframe': 'Immediate (24–48 hours)',
        'simulation_key': 'contract',
        'simulation_value': 'Two year',
    },

    # Action 2: Electronic Check -> Autopay Migration
    'ACT_PAYMENT_MIGRATION': {
        'action_id': 'ACT_PAYMENT_MIGRATION',
        'title': 'Autopay Migration with One-Time Credit',
        'category': CATEGORY_PAYMENT,
        'action': 'Migrate customer from Electronic Check to Automatic Bank Transfer or Credit Card Autopay with an immediate $10 one-time bill credit.',
        'why': 'Electronic check payments experience recurring payment friction, manual transaction fatigue, and high voluntary churn correlation.',
        'business_objective': 'Automate monthly revenue collection, eradicate missed billing cycles, and reduce billing contact friction.',
        'owner': 'Billing Operations',
        'priority': 'HIGH',
        'cost': 'LOW',
        'impact': 'MEDIUM',
        'timeframe': 'Immediate (24–48 hours)',
        'simulation_key': 'payment_method',
        'simulation_value': 'Bank transfer (automatic)',
    },

    # Action 3: High Monthly Charges -> Plan Optimization
    'ACT_PRICING_REVIEW': {
        'action_id': 'ACT_PRICING_REVIEW',
        'title': 'Account Plan & Rate Optimization Review',
        'category': CATEGORY_PRICING,
        'action': 'Conduct a comprehensive tariff review to right-size unused features or migrate the customer to a high-value bundled package saving $15–$20/mo.',
        'why': 'High monthly charges create acute price sensitivity and motivate active competitor rate shopping.',
        'business_objective': 'Protect account lifetime value by proactively resolving perceived price uncompetitiveness.',
        'owner': 'Account Management',
        'priority': 'HIGH',
        'cost': 'LOW',
        'impact': 'HIGH',
        'timeframe': 'Short-Term (7 days)',
        'simulation_key': 'monthly_charges',
        'simulation_delta': -15.0,
    },

    # Action 4: Recent Charge Increase -> Bill Transparency & Relief
    'ACT_CHARGE_SPIKE_RELIEF': {
        'action_id': 'ACT_CHARGE_SPIKE_RELIEF',
        'title': 'Bill Shock Investigation & Temporary Loyalty Credit',
        'category': CATEGORY_PRICING,
        'action': 'Review recent invoice changes, transparently explain rate adjustments, and apply a 3-month courtesy loyalty credit to cushion the transition.',
        'why': 'A detected increase in monthly charges over historical baseline causes customer dissatisfaction and sharp churn risk.',
        'business_objective': 'Alleviate customer grievance stemming from unexpected fee adjustments and restore brand confidence.',
        'owner': 'Customer Support / Billing',
        'priority': 'HIGH',
        'cost': 'LOW',
        'impact': 'MEDIUM',
        'timeframe': 'Immediate (24–48 hours)',
        'simulation_key': 'charge_increase',
        'simulation_value': 0.0,
    },

    # Action 5: Short Tenure -> Onboarding Success Outreach
    'ACT_NEW_CUSTOMER_ONBOARDING': {
        'action_id': 'ACT_NEW_CUSTOMER_ONBOARDING',
        'title': 'Early Lifecycle Onboarding & Care Outreach',
        'category': CATEGORY_LIFECYCLE,
        'action': 'Enroll customer into the VIP Early-Tenure Onboarding program with a dedicated care check-in call and welcome service verification.',
        'why': 'Customers in their first 12 months have not established habitual stickiness and are most susceptible to early drop-off.',
        'business_objective': 'Build early product satisfaction, verify service health, and accelerate transition into long-term loyal segment.',
        'owner': 'Customer Success Team',
        'priority': 'HIGH',
        'cost': 'LOW',
        'impact': 'MEDIUM',
        'timeframe': 'Immediate (24–48 hours)',
        'simulation_key': 'tenure',
        'simulation_delta': 6,
    },

    # Action 6: Fiber Optic -> Quality of Service Health Audit
    'ACT_FIBER_SERVICE_AUDIT': {
        'action_id': 'ACT_FIBER_SERVICE_AUDIT',
        'title': 'Fiber Quality-of-Service Audit & Speed Guarantee',
        'category': CATEGORY_SERVICE_TYPE,
        'action': 'Conduct a line stability test, check customer terminal optical power, and provide a complimentary Wi-Fi 6 router upgrade or speed reassurance.',
        'why': 'Fiber optic subscription contributes positively to churn, indicating customer expectations of premium performance are not being met.',
        'business_objective': 'Ensure premium fiber experience matches premium price point and pre-empt competitor gigabit promotions.',
        'owner': 'Technical Operations',
        'priority': 'MEDIUM',
        'cost': 'LOW',
        'impact': 'MEDIUM',
        'timeframe': 'Short-Term (7 days)',
        'simulation_key': 'service_audit',
        'simulation_value': 'verified',
    },

    # Action 7: Missing Tech Support -> Free Support Add-on
    'ACT_ADD_TECH_SUPPORT': {
        'action_id': 'ACT_ADD_TECH_SUPPORT',
        'title': 'Complimentary 24/7 Tech Support Trial',
        'category': CATEGORY_SERVICE_EXP,
        'action': 'Attach 6 months of complimentary 24/7 dedicated Technical Support to provide rapid troubleshooting and live remote assistance.',
        'why': 'Lack of dedicated technical support leaves customer vulnerable to unresolved connectivity friction.',
        'business_objective': 'Remove technical frustration barrier and increase service perception and engagement.',
        'owner': 'Customer Retention',
        'priority': 'MEDIUM',
        'cost': 'LOW',
        'impact': 'MEDIUM',
        'timeframe': 'Short-Term (7 days)',
        'simulation_key': 'tech_support',
        'simulation_value': 'Yes',
    },

    # Action 8: Missing Online Security -> Security Suite Bundle
    'ACT_ADD_ONLINE_SECURITY': {
        'action_id': 'ACT_ADD_ONLINE_SECURITY',
        'title': 'Complimentary Online Security Suite Activation',
        'category': CATEGORY_SERVICE_EXP,
        'action': 'Activate complimentary cyber-security DNS filtering, antivirus license, and family malware protection for 6 months.',
        'why': 'Absence of security add-ons correlates with lower product investment and lower switching costs.',
        'business_objective': 'Increase account stickiness, provide peace-of-mind security value, and increase product depth.',
        'owner': 'Product Marketing / Retention',
        'priority': 'MEDIUM',
        'cost': 'LOW',
        'impact': 'MEDIUM',
        'timeframe': 'Short-Term (7 days)',
        'simulation_key': 'online_security',
        'simulation_value': 'Yes',
    },

    # Action 9: Multi-Service Optimization & Family Bundle
    'ACT_SERVICE_BUNDLE_OFFER': {
        'action_id': 'ACT_SERVICE_BUNDLE_OFFER',
        'title': 'Integrated Multi-Service Loyalty Bundle',
        'category': CATEGORY_ENGAGEMENT,
        'action': 'Package broadband, streaming entertainment, and backup into a unified family plan with an consolidated bill discount.',
        'why': 'Multi-product accounts with high platform integration exhibit up to 4x lower churn rates.',
        'business_objective': 'Deepen multi-service penetration and maximize switching friction across the household.',
        'owner': 'Marketing / Commercial',
        'priority': 'MEDIUM',
        'cost': 'MEDIUM',
        'impact': 'HIGH',
        'timeframe': 'Long-Term (14–30 days)',
        'simulation_key': 'multiple_services',
        'simulation_value': 'bundled',
    },

    # Action 10: Multi-Factor Compounded Risk -> Executive Outreach
    'ACT_MULTI_FACTOR_INTERVENTION': {
        'action_id': 'ACT_MULTI_FACTOR_INTERVENTION',
        'title': 'VIP Proactive Retention Intervention',
        'category': CATEGORY_COMPOUND,
        'action': 'Dispatch an expedited senior retention specialist with authority to bundle a multi-year loyalty rate, autopay transition credit, and free device upgrade.',
        'why': 'Customer exhibits a critical confluence of high-risk drivers (month-to-month, electronic check, and high monthly billing).',
        'business_objective': 'Defend strategic account from imminent termination through a coordinated multi-touch retention package.',
        'owner': 'Senior Retention Desk',
        'priority': 'CRITICAL',
        'cost': 'HIGH',
        'impact': 'HIGH',
        'timeframe': 'Immediate (24–48 hours)',
        'simulation_key': 'compound_retention',
        'simulation_value': 'vip_package',
    },

    # Action 11: Loyal Customer Appreciation (For Low-Risk / Protective Profiles)
    'ACT_LOYALTY_APPRECIATION': {
        'action_id': 'ACT_LOYALTY_APPRECIATION',
        'title': 'Customer Loyalty Appreciation & Upgrade Privilege',
        'category': CATEGORY_LIFECYCLE,
        'action': 'Send personalized customer appreciation communication with an exclusive no-commitment speed boost or loyalty reward voucher.',
        'why': 'Customer displays healthy account stability and strong protective factors; aggressive retention discounts are inappropriate.',
        'business_objective': 'Deepen positive brand affinity, stimulate word-of-mouth advocacy, and reward continued loyalty.',
        'owner': 'Customer Relationship Management (CRM)',
        'priority': 'LOW',
        'cost': 'LOW',
        'impact': 'MEDIUM',
        'timeframe': 'Long-Term (30 days)',
        'simulation_key': 'loyalty_perk',
        'simulation_value': 'appreciation',
    },
}

# ==============================================================================
# 4. HELPER ACCESSORS
# ==============================================================================
def get_business_factor(feature_name: str) -> Dict[str, Any]:
    """Retrieve business factor mapping for a feature name, with fallback."""
    if feature_name in FEATURE_BUSINESS_MAPPING:
        return FEATURE_BUSINESS_MAPPING[feature_name]
    
    # Heuristic category guessing if unknown
    low = feature_name.lower()
    if 'contract' in low:
        cat = CATEGORY_CONTRACT
    elif 'payment' in low or 'check' in low or 'paperless' in low:
        cat = CATEGORY_PAYMENT
    elif 'charge' in low or 'spend' in low:
        cat = CATEGORY_PRICING
    elif 'tenure' in low:
        cat = CATEGORY_LIFECYCLE
    elif 'internet' in low or 'fiber' in low or 'dsl' in low or 'phone' in low:
        cat = CATEGORY_SERVICE_TYPE
    elif 'security' in low or 'support' in low or 'backup' in low or 'protection' in low:
        cat = CATEGORY_SERVICE_EXP
    elif 'stream' in low:
        cat = CATEGORY_ENGAGEMENT
    else:
        cat = "General Account Attributes"
        
    return {
        'category': cat,
        'business_name': feature_name.replace('_', ' ').title(),
        'is_actionable': True,
        'meaning': f"Metric reflecting customer profile characteristic: {feature_name}.",
        'risk_direction': 'neutral',
    }

def get_action_definition(action_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full action definition from catalog."""
    return RETENTION_ACTIONS_CATALOG.get(action_id)

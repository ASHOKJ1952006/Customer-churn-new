# Open Issues & Verification Audit Log

This document tracks all audited claims, verification statuses, and resolution actions completed for IEEE submission compliance.

| ID | Item / Claim | Final Resolution Status | Action Taken / Verification Result |
|:---|:---|:---|:---|
| ISS-01 | Prior Publication Status / ICERCS Header | **RESOLVED** | Removed all references to ICERCS 2025, ISBN, copyright lines, and DOIs. Formatted manuscript with double-blind neutral placeholders (`Anonymous Authors`, `[Target IEEE Conference — Under Review]`). |
| ISS-02 | Reference Verification: Ullo et al. (2021) | **RESOLVED** | Verified IEEE Access (vol. 9, pp. 104212–104229, 2021) publication metadata in `paper/refs.bib`. |
| ISS-03 | Reference Verification: Albuquerque et al. (2022) | **RESOLVED** | Verified Computers & Industrial Engineering (vol. 169, p. 108241, 2022) in `paper/refs.bib`. |
| ISS-04 | Reference Verification: Aggarwal & Mahajan (2024) | **RESOLVED** | Verified Journal of Information Systems and Informatics (vol. 6, no. 2, pp. 607–623, 2024) in `paper/refs.bib`. |
| ISS-05 | Canonical ML Citations | **RESOLVED** | Populated `paper/refs.bib` with canonical, verified papers: Breiman (2001), Chen & Guestrin (2016), Ke et al. (2017), Wolpert (1992), Lundberg & Lee (2017), Ribeiro et al. (2016), Elkan (2001), Verbeke et al. (2012), Coussement et al. (2017), Amin et al. (2019), Chawla et al. (2002), Brier (1950), Davis & Goadrich (2006), Ullah et al. (2019). |
| ISS-06 | Acquisition Cost Multiplier ("5 to 7 times") | **RESOLVED** | Explicitly anchored to peer-reviewed retention economics citations (Amin et al., 2019; Verbeke et al., 2012). |
| ISS-07 | Stacking Superiority Claim | **RESOLVED** | Eliminated false claims of stacking superiority. Reported empirical repeated cross-validation metrics honestly (XGBoost 94.98%, Stacking 94.96%). Repositioned paper as an explainable, cost-sensitive decision-support framework. |
| ISS-08 | Dataset Scope & Feature Dimensions | **RESOLVED** | Documented dataset as balanced benchmark ($N=8,000$) with exact 27-dimensional feature transformation derived from 6 raw inputs. Zero external data mixed into final evaluation. |
| ISS-09 | SHAP vs. LIME Concordance Claim | **RESOLVED** | Replaced visual assumptions of agreement with empirical quantitative concordance across $N=200$ test customer instances (Spearman $\rho = 0.2421 \pm 0.1091$, Top-3 overlap $22.83\%$). Documented mathematical causes of divergence. |
| ISS-10 | Master Reproducibility Package | **RESOLVED** | Built `reproducibility/reproduce.py` with pinned `reproducibility/requirements.txt`, fully tested and verified. Created `README_REPRODUCTION.md`. |

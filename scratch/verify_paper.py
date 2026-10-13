import os

ROOT = r'd:\ABI\YEAR 4\SEM 7\churn\Customer-churn-new'

files = [
    'Explainable_Ensemble_Learning_Telecom_Churn_Final.tex',
    'Explainable_Ensemble_Learning_Telecom_Churn_Final.html',
    'Explainable_Ensemble_Learning_Telecom_Churn_Final.docx',
    'Explainable_Ensemble_Learning_Telecom_Churn_Final.pdf',
]

print('=== PAPER OUTPUT FILES ===')
for f in files:
    fp = os.path.join(ROOT, f)
    size = os.path.getsize(fp) if os.path.exists(fp) else -1
    status = 'OK' if size > 0 else 'MISSING'
    ext = os.path.splitext(f)[1].upper()
    print('  [%s] %s: %10d bytes  %s' % (status, ext, size, f))

print()
print('=== METRIC VERIFICATION IN LaTeX ===')
with open(os.path.join(ROOT, files[0]), 'r', encoding='utf-8') as fh:
    tex = fh.read()

metrics = {
    'XGB Recall 97.6%': '97.6',
    'XGB AUC 99.0%': '99.0',
    'LGBM Recall 96.9%': '96.9',
    'LGBM AUC 98.9%': '98.9',
    'RF Recall 96.3%': '96.3',
    'RF AUC 98.7%': '98.7',
    'Stack F1 95.5%': '95.5',
    'Stack Prec 94.7%': '94.7',
    'Stack Brier 0.0362': '0.0362',
    'XGB thresh 0.29': '0.29',
    'LGBM thresh 0.32': '0.32',
    'RF thresh 0.38': '0.38',
    'Stack thresh 0.26': '0.26',
}

all_ok = True
for name, val in metrics.items():
    found = val in tex
    if not found:
        all_ok = False
    print('  %s: %s (%s)' % ('OK' if found else 'MISSING', name, val))

print()
if all_ok:
    print('All metrics verified!')
else:
    print('WARNING: some metrics are missing!')

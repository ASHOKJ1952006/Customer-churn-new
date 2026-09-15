import sys, os, asyncio, json
sys.path.append(os.path.abspath('backend'))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from main import predict_batch
from fastapi import UploadFile

csv_path = 'data/test_batch_upload.csv'
with open(csv_path, 'rb') as f:
    upload_file = UploadFile(file=f, filename='test_batch_upload.csv')
    res = asyncio.run(predict_batch(file=upload_file, dataset='ds1'))

print("=== BATCH PREDICTION TEST ===")
print("Batch ID:", res['batch_id'])
print("Total customers:", res['summary']['total'])
print("High risk:", res['summary']['high'])
print("Medium risk:", res['summary']['medium'])
print("Low risk:", res['summary']['low'])

print("\nTop Organizational Churn Drivers:")
for d in res['summary']['top_organizational_drivers']:
    print(f"  - {d['driver']}: {d['percentage']}% ({d['count']} accounts)")

print("\nTop Retention Campaigns:")
for s in res['summary']['top_retention_strategies']:
    print(f"  * {s['strategy']}: {s['percentage']}% ({s['count']} accounts)")

assert 'top_organizational_drivers' in res['summary']
assert 'top_retention_strategies' in res['summary']
print("\nBATCH TEST PASSED SUCCESSFULLY!")

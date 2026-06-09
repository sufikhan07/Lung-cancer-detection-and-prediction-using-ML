import os
import traceback
from collections import Counter

try:
    from app import preprocess_ct_image, ct_model
except Exception as e:
    print('Failed to import from app.py:', e)
    traceback.print_exc()
    raise

BASE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE, '..', 'lung-cancer-ct-scan-detection-using-cnn', 'Data', 'test')
# Fallback if relative path different
if not os.path.exists(TEST_DIR):
    TEST_DIR = os.path.join(BASE, 'Data', 'test')

print('Looking for test data in:', TEST_DIR)
if not os.path.exists(TEST_DIR):
    print('Test directory not found. Check path. Exiting.')
    raise SystemExit(1)

classes = ['Cancerous', 'Non-Cancerous']
results = []

for cls in classes:
    cls_dir = os.path.join(TEST_DIR, cls)
    if not os.path.exists(cls_dir):
        print('Warning: class dir not found:', cls_dir)
        continue
    files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir,f))]
    print(f'Found {len(files)} files for class {cls}')
    for f in files[:100]:  # limit to first 100 per class to keep quick
        try:
            img = preprocess_ct_image(f)
            if img is None:
                print('Preprocess failed for', f)
                continue
            pred = ct_model.predict(img)
            val = float(pred[0][0])
            # Model's sigmoid outputs probability for class index 1 (second folder).
            # During training folders were ["Cancerous","Non-Cancerous"], so:
            #   val > 0.5 means 'Non-Cancerous'
            predicted_label = 'Non-Cancerous' if val > 0.5 else 'Cancerous'
            results.append((cls, predicted_label, val))
        except Exception as e:
            print('Error on file', f, e)

# Summarize
counter = Counter()
for true, pred_label, val in results:
    key = (true, pred_label)
    counter[key] += 1

print('\nConfusion counts:')
for t in classes:
    for p in classes:
        print(f'True={t} Pred={p}:', counter.get((t,p), 0))

# Accuracy
correct = sum(1 for t,p,v in results if t==p)
total = len(results)
acc = (correct/total*100) if total>0 else 0
print(f'Overall evaluated: {total} images, Accuracy: {acc:.2f}%')

# Show a few misclassified examples
mis = [(t,p,v) for t,p,v in results if t!=p]
print('\nSample misclassifications (up to 20):')
for item in mis[:20]:
    print(item)

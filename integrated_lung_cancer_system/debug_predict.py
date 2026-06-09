import os
import traceback
from pprint import pprint

try:
    from app import preprocess_ct_image, ct_model
except Exception as e:
    print('Failed to import from app.py:', e)
    traceback.print_exc()
    raise

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, 'static', 'uploads')

print('Uploads folder:', UPLOADS)
if not os.path.exists(UPLOADS):
    print('Uploads folder does not exist')
    raise SystemExit(1)

files = [os.path.join(UPLOADS, f) for f in os.listdir(UPLOADS) if os.path.isfile(os.path.join(UPLOADS,f))]
if not files:
    print('No files found in uploads')
    raise SystemExit(1)

files = sorted(files, key=os.path.getmtime, reverse=True)
latest = files[0]
print('Using latest file:', latest)
try:
    img = preprocess_ct_image(latest)
    print('preprocess result is None?', img is None)
    if img is not None:
        print('img shape:', getattr(img, 'shape', None))
        import numpy as np
        print('img dtype:', img.dtype, 'min/max:', np.min(img), np.max(img))

    if ct_model is None:
        print('CT model is not loaded (ct_model is None)')
        raise SystemExit(1)

    print('Running model.predict...')
    pred = ct_model.predict(img)
    print('Raw model output:', pred)
    try:
        val = float(pred[0][0])
    except Exception as e:
        print('Could not extract float from prediction array:', e)
        raise
    print('pred_value (0-1):', val)
    # Interpret according to training folder ordering: sigmoid => prob of second class
    predicted_label = 'Non-Cancerous' if val > 0.5 else 'Cancerous'
    print('Interpreted result:', predicted_label)
    # Confidence for predicted class
    confidence = val*100 if val > 0.5 else (1-val)*100
    print('Confidence (percent for predicted class):', confidence)

except Exception as e:
    print('Error during debug prediction:', e)
    traceback.print_exc()
    raise

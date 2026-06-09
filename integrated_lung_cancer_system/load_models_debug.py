import os
import zipfile
import tensorflow as tf

base = r'c:\Users\Anand Singh\OneDrive\Desktop\Major\integrated_lung_cancer_system'
ct_path = os.path.join(base, 'lung_cancer_cnn_model.keras')
risk_path = os.path.join(base, 'risk_assessment_model.h5')

print('Files present:')
for p in [ct_path, risk_path]:
    print(p, '->', os.path.exists(p), 'size=', os.path.getsize(p) if os.path.exists(p) else 'N/A')

print('\nInspecting .keras file structure:')
try:
    with zipfile.ZipFile(ct_path, 'r') as z:
        print('Entries in .keras archive (first 20):')
        for i, name in enumerate(z.namelist()):
            print(' ', name)
            if i >= 19:
                break
except Exception as e:
    print('Error opening .keras as zip:', e)

print('\nAttempting to load CT model (keras file) with compile=False:')
try:
    m = tf.keras.models.load_model(ct_path, compile=False)
    print('CT model loaded, type:', type(m))
except Exception as e:
    print('CT model load error:', repr(e))

print('\nAttempting to load risk model (.h5) with compile=False:')
try:
    m2 = tf.keras.models.load_model(risk_path, compile=False)
    print('Risk model loaded, type:', type(m2))
except Exception as e:
    print('Risk model load error:', repr(e))

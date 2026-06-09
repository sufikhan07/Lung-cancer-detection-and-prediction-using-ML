import importlib.util
p = 'c:/Users/Anand Singh/OneDrive/Desktop/Major/integrated_lung_cancer_system/app.py'
spec = importlib.util.spec_from_file_location('integ_app', p)
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    print('Imported OK')
    print('ct_model loaded:', mod.ct_model is not None)
    print('risk_model loaded:', mod.risk_model is not None)
except Exception as e:
    print('Import error:', e)

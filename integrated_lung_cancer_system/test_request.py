import requests
try:
    r = requests.get('http://127.0.0.1:5000/')
    print(f'Status: {r.status_code}')
    print('Headers:', r.headers)
    print('Content:', r.text[:500] if r.text else 'No content')
except Exception as e:
    print(f'Error: {e}')
from io import BytesIO

import numpy as np
from PIL import Image

from integrated_lung_cancer_system.app import app


def test_health_reports_both_models_loaded():
    response = app.test_client().get('/health')
    assert response.status_code == 200
    assert response.get_json() == {
        'status': 'healthy',
        'ct_model_loaded': True,
        'risk_model_loaded': True,
    }


def test_main_pages_render():
    client = app.test_client()
    for route in ('/', '/ct-scan', '/risk-assessment', '/blood-report'):
        assert client.get(route).status_code == 200


def test_invalid_ct_file_is_rejected_safely():
    response = app.test_client().post(
        '/ct-scan',
        data={'file': (BytesIO(b'not an image'), 'scan.png')},
        content_type='multipart/form-data',
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b'Error processing CT scan' in response.data


def test_ct_prediction_returns_result(monkeypatch):
    client = app.test_client()
    image = Image.new('RGB', (224, 224), color=(80, 80, 80))
    upload = BytesIO()
    image.save(upload, format='PNG')
    upload.seek(0)
    monkeypatch.setattr(
        'integrated_lung_cancer_system.app.ct_model.predict',
        lambda _image: np.array([[0.8]], dtype=np.float32),
    )
    response = client.post(
        '/ct-scan',
        data={'file': (upload, 'scan.png')},
        content_type='multipart/form-data',
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b'Non-Cancerous' in response.data


def test_risk_assessment_returns_result():
    risk_fields = (
        'air_pollution', 'alcohol_use', 'dust_allergy',
        'occupational_hazards', 'genetic_risk', 'chronic_lung_disease',
        'balanced_diet', 'obesity', 'smoking', 'passive_smoker',
        'chest_pain', 'coughing_blood', 'fatigue', 'weight_loss',
        'shortness_of_breath', 'wheezing', 'swallowing_difficulty',
        'clubbing', 'frequent_cold', 'dry_cough', 'snoring',
    )
    form = {'age': '42', 'gender': 'male'}
    form.update({field: '5' for field in risk_fields})
    response = app.test_client().post(
        '/risk-assessment', data=form, follow_redirects=True
    )
    assert response.status_code == 200
    assert b'High' in response.data
    assert b'Error processing risk assessment' not in response.data

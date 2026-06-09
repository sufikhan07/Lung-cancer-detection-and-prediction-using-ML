"""
Unit tests for Lung Cancer Diagnostic System
"""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

# Import application modules
from config import config
from src.models.model_manager import ModelManager
from src.preprocessing.preprocessors import ImagePreprocessor, RiskDataPreprocessor
from src.services.diagnostic_services import CTScanService, RiskAssessmentService

class TestModelManager:
    """Test cases for ModelManager"""

    def test_model_manager_initialization(self):
        """Test ModelManager initialization"""
        manager = ModelManager()
        assert manager.ct_model is None
        assert manager.risk_model is None
        assert manager.scaler is None
        assert not manager.models_loaded

    @patch('os.path.exists')
    @patch('tensorflow.keras.models.load_model')
    def test_load_models_success(self, mock_load_model, mock_exists):
        """Test successful model loading"""
        mock_exists.return_value = True
        mock_load_model.return_value = Mock()

        manager = ModelManager()
        result = manager.load_models()

        assert result is True
        assert manager.models_loaded is True
        assert manager.ct_model is not None
        assert manager.risk_model is not None

    @patch('os.path.exists')
    def test_load_models_partial_failure(self, mock_exists):
        """Test partial model loading failure"""
        mock_exists.return_value = False

        manager = ModelManager()
        result = manager.load_models()

        assert result is False
        assert manager.models_loaded is True  # Still True due to graceful handling

    def test_predict_ct_scan_without_model(self):
        """Test CT prediction without loaded model"""
        manager = ModelManager()

        with pytest.raises(ValueError, match="CT model not loaded"):
            manager.predict_ct_scan(np.random.rand(1, 224, 224, 3))

    @patch('tensorflow.keras.models.load_model')
    def test_predict_ct_scan_success(self, mock_load_model):
        """Test successful CT scan prediction"""
        # Mock the model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.7]])  # Non-cancerous prediction
        mock_load_model.return_value = mock_model

        manager = ModelManager()
        manager.ct_model = mock_model

        test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
        result = manager.predict_ct_scan(test_image)

        assert 'prediction' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert result['prediction'] == 'Non-Cancerous'
        assert result['confidence'] > 50

class TestImagePreprocessor:
    """Test cases for ImagePreprocessor"""

    def test_preprocessor_initialization(self):
        """Test ImagePreprocessor initialization"""
        processor = ImagePreprocessor()
        assert processor.target_size == (224, 224)

    def test_preprocess_ct_image_nonexistent_file(self):
        """Test preprocessing with nonexistent file"""
        processor = ImagePreprocessor()
        result = processor.preprocess_ct_image('/nonexistent/file.jpg')
        assert result is None

    def test_preprocess_ct_image_valid_image(self):
        """Test preprocessing with valid image"""
        processor = ImagePreprocessor()

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Create a simple RGB image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp_file.name)

            try:
                result = processor.preprocess_ct_image(tmp_file.name)

                assert result is not None
                assert result.shape == (1, 224, 224, 3)
                assert result.dtype == np.float32
                assert np.all(result >= 0) and np.all(result <= 1)  # Normalized

            finally:
                os.unlink(tmp_file.name)

    def test_validate_medical_image_square_image(self):
        """Test medical image validation with square image"""
        processor = ImagePreprocessor()

        # Create square grayscale-like image
        img = np.full((100, 100, 3), [128, 130, 132], dtype=np.uint8)
        result = processor._validate_medical_image(img)

        assert result['is_valid'] is True

    def test_validate_medical_image_extreme_aspect_ratio(self):
        """Test medical image validation with extreme aspect ratio"""
        processor = ImagePreprocessor()

        # Create very wide image
        img = np.zeros((50, 200, 3), dtype=np.uint8)
        result = processor._validate_medical_image(img)

        assert result['is_valid'] is False
        assert 'aspect ratio' in result['message']

class TestRiskDataPreprocessor:
    """Test cases for RiskDataPreprocessor"""

    def test_preprocessor_initialization(self):
        """Test RiskDataPreprocessor initialization"""
        processor = RiskDataPreprocessor()
        assert processor.expected_features == 23

    def test_preprocess_risk_data_missing_fields(self):
        """Test preprocessing with missing required fields"""
        processor = RiskDataPreprocessor()

        incomplete_data = {'age': '30', 'gender': 'male'}  # Missing other fields
        result = processor.preprocess_risk_data(incomplete_data)

        assert result is None

    def test_preprocess_risk_data_valid_data(self):
        """Test preprocessing with valid data"""
        processor = RiskDataPreprocessor()

        # Create complete test data
        test_data = {
            'age': '45',
            'gender': 'female',
            'air_pollution': '3',
            'alcohol_use': '2',
            'dust_allergy': '4',
            'occupational_hazards': '3',
            'genetic_risk': '2',
            'chronic_lung_disease': '1',
            'balanced_diet': '7',
            'obesity': '5',
            'smoking': '1',
            'passive_smoker': '2',
            'chest_pain': '3',
            'coughing_blood': '1',
            'fatigue': '4',
            'weight_loss': '2',
            'shortness_of_breath': '3',
            'wheezing': '2',
            'swallowing_difficulty': '1',
            'clubbing': '2',
            'frequent_cold': '4',
            'dry_cough': '3',
            'snoring': '5'
        }

        result = processor.preprocess_risk_data(test_data)

        assert result is not None
        assert result.shape == (23,)
        assert result[0] == 45.0  # Age
        assert result[1] == 2.0   # Female

    def test_preprocess_risk_data_invalid_age(self):
        """Test preprocessing with invalid age"""
        processor = RiskDataPreprocessor()

        test_data = {
            'age': '150',  # Invalid age
            'gender': 'male',
            # ... other fields would be needed
        }

        result = processor.preprocess_risk_data(test_data)
        assert result is None

class TestCTScanService:
    """Test cases for CTScanService"""

    def test_service_initialization(self):
        """Test CTScanService initialization"""
        service = CTScanService()
        assert service.upload_folder == config.upload.upload_folder

    def test_allowed_file_valid_extensions(self):
        """Test file extension validation"""
        service = CTScanService()

        valid_files = ['test.jpg', 'test.png', 'test.jpeg']
        for filename in valid_files:
            assert service._allowed_file(filename) is True

    def test_allowed_file_invalid_extensions(self):
        """Test invalid file extension rejection"""
        service = CTScanService()

        invalid_files = ['test.txt', 'test.exe', 'test']
        for filename in invalid_files:
            assert service._allowed_file(filename) is False

    @patch('src.services.diagnostic_services.model_manager')
    @patch('src.preprocessing.preprocessors.image_preprocessor')
    def test_analyze_ct_scan_success(self, mock_preprocessor, mock_model_manager):
        """Test successful CT scan analysis"""
        # Setup mocks
        mock_preprocessor.preprocess_ct_image.return_value = np.random.rand(1, 224, 224, 3)
        mock_model_manager.models_loaded = True
        mock_model_manager.predict_ct_scan.return_value = {
            'prediction': 'Non-Cancerous',
            'confidence': 75.5,
            'probabilities': {'non_cancerous': 75.5, 'cancerous': 24.5}
        }

        service = CTScanService()

        # Create mock file
        mock_file = Mock()
        mock_file.filename = 'test.jpg'

        result = service.analyze_ct_scan(mock_file)

        assert result['success'] is True
        assert result['prediction'] == 'Non-Cancerous'
        assert result['confidence'] == 75.5
        assert 'processing_time' in result

class TestRiskAssessmentService:
    """Test cases for RiskAssessmentService"""

    @patch('src.services.diagnostic_services.model_manager')
    @patch('src.preprocessing.preprocessors.risk_preprocessor')
    def test_assess_risk_success(self, mock_preprocessor, mock_model_manager):
        """Test successful risk assessment"""
        # Setup mocks
        mock_preprocessor.preprocess_risk_data.return_value = np.random.rand(23)
        mock_model_manager.models_loaded = True
        mock_model_manager.predict_risk_assessment.return_value = {
            'risk_level': 'Medium',
            'risk_percentage': 45.0,
            'confidence': 65.0,
            'probabilities': {'Low': 35.0, 'Medium': 65.0, 'High': 0.0}
        }

        service = RiskAssessmentService()

        test_data = {'age': '50', 'gender': 'male'}  # Simplified for test

        result = service.assess_risk(test_data)

        assert result['success'] is True
        assert result['risk_level'] == 'Medium'
        assert result['risk_percentage'] == 45.0
        assert 'processing_time' in result

# Integration test fixtures
@pytest.fixture
def app():
    """Create test Flask app"""
    from src.web.app import create_app
    app = create_app('testing')
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()

class TestAPIIntegration:
    """Integration tests for API endpoints"""

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/api/v1/health')
        assert response.status_code == 200

        data = response.get_json()
        assert 'overall_status' in data
        assert 'models' in data
        assert 'system' in data

    def test_ct_scan_analyze_no_file(self, client):
        """Test CT scan analysis with no file"""
        response = client.post('/api/v1/ct-scan/analyze')
        assert response.status_code == 400

        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data

    def test_risk_assess_no_data(self, client):
        """Test risk assessment with no data"""
        response = client.post('/api/v1/risk/assess',
                             content_type='application/json')
        assert response.status_code == 400

        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data

    def test_models_status_endpoint(self, client):
        """Test models status endpoint"""
        response = client.get('/api/v1/models/status')
        assert response.status_code == 200

        data = response.get_json()
        assert 'models_loaded' in data
        assert 'timestamp' in data

if __name__ == '__main__':
    pytest.main([__file__])</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\tests\unit\test_diagnostic_system.py
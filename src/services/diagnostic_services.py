"""
Business logic services for Lung Cancer Diagnostic System
"""
import time
import os
from typing import Dict, Any, Optional
from datetime import datetime
from werkzeug.utils import secure_filename
from config import config
from src.models.model_manager import model_manager
from src.preprocessing.preprocessors import image_preprocessor, risk_preprocessor
from src.utils.logging_config import logger

class CTScanService:
    """Service for CT scan analysis"""

    def __init__(self):
        self.upload_folder = config.upload.upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)

    def analyze_ct_scan(self, file) -> Dict[str, Any]:
        """Analyze uploaded CT scan image"""
        start_time = time.time()

        try:
            # Validate file
            if not file or file.filename == '':
                raise ValueError("No file provided")

            if not self._allowed_file(file.filename):
                raise ValueError(f"File type not allowed. Allowed: {config.upload.allowed_extensions}")

            # Save file securely
            filename = self._generate_secure_filename(file.filename)
            filepath = os.path.join(self.upload_folder, filename)
            file.save(filepath)

            logger.info(f"Saved uploaded file: {filepath}")

            # Preprocess image
            processed_image = image_preprocessor.preprocess_ct_image(filepath)
            if processed_image is None:
                raise ValueError("Image preprocessing failed")

            # Make prediction
            if not model_manager.models_loaded:
                model_manager.load_models()

            prediction_result = model_manager.predict_ct_scan(processed_image)

            # Calculate processing time
            processing_time = time.time() - start_time

            result = {
                'success': True,
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities'],
                'image_path': f'/static/uploads/{filename}',
                'processing_time': processing_time,
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': '1.0.0'
            }

            logger.info(f"CT scan analysis completed in {processing_time:.2f}s: {prediction_result['prediction']}")
            return result

        except Exception as e:
            logger.error(f"CT scan analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.utcnow().isoformat()
            }
        finally:
            # Clean up old files (keep last 10)
            self._cleanup_old_files()

    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in config.upload.allowed_extensions

    def _generate_secure_filename(self, original_filename: str) -> str:
        """Generate secure filename with timestamp"""
        extension = original_filename.rsplit('.', 1)[1].lower()
        timestamp = int(time.time())
        return f"ct_scan_{timestamp}.{extension}"

    def _cleanup_old_files(self, keep_count: int = 10):
        """Clean up old uploaded files"""
        try:
            files = []
            for filename in os.listdir(self.upload_folder):
                filepath = os.path.join(self.upload_folder, filename)
                if os.path.isfile(filepath):
                    files.append((filepath, os.path.getctime(filepath)))

            # Sort by creation time (oldest first)
            files.sort(key=lambda x: x[1])

            # Remove old files beyond keep_count
            for filepath, _ in files[:-keep_count]:
                try:
                    os.remove(filepath)
                    logger.debug(f"Cleaned up old file: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to remove old file {filepath}: {e}")

        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")

class RiskAssessmentService:
    """Service for risk assessment analysis"""

    def assess_risk(self, form_data: Dict[str, str]) -> Dict[str, Any]:
        """Assess lung cancer risk from patient data"""
        start_time = time.time()

        try:
            # Preprocess form data
            processed_data = risk_preprocessor.preprocess_risk_data(form_data)
            if processed_data is None:
                raise ValueError("Data preprocessing failed")

            # Load models if needed
            if not model_manager.models_loaded:
                model_manager.load_models()

            # Make prediction
            prediction_result = model_manager.predict_risk_assessment(processed_data)

            # Calculate processing time
            processing_time = time.time() - start_time

            result = {
                'success': True,
                'risk_level': prediction_result['risk_level'],
                'risk_percentage': prediction_result['risk_percentage'],
                'confidence': prediction_result['confidence'],
                'probabilities': prediction_result['probabilities'],
                'patient_info': {
                    'age': form_data.get('age'),
                    'gender': form_data.get('gender', '').capitalize()
                },
                'processing_time': processing_time,
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': '1.0.0'
            }

            logger.info(f"Risk assessment completed in {processing_time:.2f}s: {prediction_result['risk_level']}")
            return result

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.utcnow().isoformat()
            }

class HealthCheckService:
    """Service for system health monitoring"""

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'services': {},
            'models': {},
            'system': {}
        }

        # Check model status
        try:
            model_validation = model_manager.validate_models()
            health_status['models'] = {
                'status': 'healthy' if all(model_validation.values()) else 'degraded',
                'details': model_validation
            }
        except Exception as e:
            health_status['models'] = {
                'status': 'unhealthy',
                'error': str(e)
            }

        # Check file system
        try:
            upload_dir_exists = os.path.exists(config.upload.upload_folder)
            upload_dir_writable = os.access(config.upload.upload_folder, os.W_OK) if upload_dir_exists else False

            health_status['system']['file_system'] = {
                'upload_directory_exists': upload_dir_exists,
                'upload_directory_writable': upload_dir_writable,
                'status': 'healthy' if upload_dir_exists and upload_dir_writable else 'unhealthy'
            }
        except Exception as e:
            health_status['system']['file_system'] = {
                'status': 'unhealthy',
                'error': str(e)
            }

        # Overall status
        services_healthy = all(
            service.get('status') == 'healthy'
            for service in health_status['services'].values()
        )
        models_healthy = health_status['models'].get('status') == 'healthy'
        system_healthy = all(
            system.get('status') == 'healthy'
            for system in health_status['system'].values()
        )

        health_status['overall_status'] = 'healthy' if all([
            services_healthy, models_healthy, system_healthy
        ]) else 'degraded'

        return health_status

# Global service instances
ct_scan_service = CTScanService()
risk_assessment_service = RiskAssessmentService()
health_check_service = HealthCheckService()</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\src\services\diagnostic_services.py
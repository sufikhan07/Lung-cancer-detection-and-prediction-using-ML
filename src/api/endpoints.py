"""
REST API endpoints for Lung Cancer Diagnostic System
"""
from flask import Blueprint, request, jsonify, current_app
from werkzeug.exceptions import BadRequest
from src.services.diagnostic_services import (
    ct_scan_service,
    risk_assessment_service,
    health_check_service
)
from src.utils.logging_config import logger
import time

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

@api_bp.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    try:
        health_status = health_check_service.get_system_health()
        status_code = 200 if health_status['overall_status'] == 'healthy' else 503
        return jsonify(health_status), status_code
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Health check failed',
            'timestamp': time.time()
        }), 500

@api_bp.route('/ct-scan/analyze', methods=['POST'])
def analyze_ct_scan():
    """Analyze CT scan image"""
    try:
        start_time = time.time()

        # Validate request
        if 'file' not in request.files:
            raise BadRequest("No file provided in request")

        file = request.files['file']
        if file.filename == '':
            raise BadRequest("Empty filename")

        # Process the analysis
        result = ct_scan_service.analyze_ct_scan(file)

        # Log API call
        processing_time = time.time() - start_time
        logger.info(f"CT scan API call completed in {processing_time:.2f}s")

        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code

    except BadRequest as e:
        logger.warning(f"Bad request for CT scan analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error in CT scan analysis: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': time.time()
        }), 500

@api_bp.route('/risk/assess', methods=['POST'])
def assess_risk():
    """Assess lung cancer risk"""
    try:
        start_time = time.time()

        # Get JSON data
        data = request.get_json()
        if not data:
            raise BadRequest("No JSON data provided")

        # Process the assessment
        result = risk_assessment_service.assess_risk(data)

        # Log API call
        processing_time = time.time() - start_time
        logger.info(f"Risk assessment API call completed in {processing_time:.2f}s")

        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code

    except BadRequest as e:
        logger.warning(f"Bad request for risk assessment: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 400
    except Exception as e:
        logger.error(f"Unexpected error in risk assessment: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'timestamp': time.time()
        }), 500

@api_bp.route('/models/status', methods=['GET'])
def get_model_status():
    """Get status of loaded models"""
    try:
        from src.models.model_manager import model_manager

        status = {
            'models_loaded': model_manager.models_loaded,
            'ct_model_available': model_manager.ct_model is not None,
            'risk_model_available': model_manager.risk_model is not None,
            'scaler_available': model_manager.scaler is not None,
            'timestamp': time.time()
        }

        return jsonify(status), 200

    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({
            'error': 'Failed to get model status',
            'timestamp': time.time()
        }), 500

@api_bp.route('/diagnostics/history', methods=['GET'])
def get_diagnostics_history():
    """Get diagnostics history (placeholder for future implementation)"""
    # This would typically query a database
    # For now, return a placeholder response
    return jsonify({
        'message': 'Diagnostics history feature coming soon',
        'total_records': 0,
        'records': [],
        'timestamp': time.time()
    }), 200

# Error handlers
@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'timestamp': time.time()
    }), 404

@api_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'timestamp': time.time()
    }), 405

@api_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'timestamp': time.time()
    }), 500</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\src\api\endpoints.py
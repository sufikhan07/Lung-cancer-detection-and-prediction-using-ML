"""
Main Flask application for Lung Cancer Diagnostic System
"""
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from config import config
from src.api.endpoints import api_bp
from src.models.model_manager import model_manager
from src.services.diagnostic_services import ct_scan_service, risk_assessment_service
from src.utils.logging_config import logger

def create_app(config_name: str = 'development') -> Flask:
    """Application factory pattern"""
    app = Flask(__name__)

    # Configure app
    app.config['SECRET_KEY'] = config.api.secret_key
    app.config['MAX_CONTENT_LENGTH'] = config.api.max_content_length
    app.config['UPLOAD_FOLDER'] = config.upload.upload_folder

    # Ensure upload directory exists
    os.makedirs(config.upload.upload_folder, exist_ok=True)

    # Register blueprints
    app.register_blueprint(api_bp)

    # Load models on startup
    with app.app_context():
        logger.info("Loading ML models...")
        if not model_manager.load_models():
            logger.warning("Some models failed to load. Application will continue with limited functionality.")

    # Web routes
    @app.route('/')
    def home():
        """Home page"""
        return render_template('home.html')

    @app.route('/ct-scan', methods=['GET', 'POST'])
    def ct_scan():
        """CT scan analysis page"""
        if request.method == 'POST':
            try:
                result = ct_scan_service.analyze_ct_scan(request.files.get('file'))

                if result['success']:
                    return render_template('result.html',
                                         result=result['prediction'],
                                         probability=result['confidence'],
                                         image_path=result['image_path'],
                                         is_ct_scan=True,
                                         processing_time=result['processing_time'])
                else:
                    flash(f'Error: {result.get("error", "Unknown error")}', 'error')
                    return redirect(request.url)

            except RequestEntityTooLarge:
                flash('File too large. Maximum size is 16MB.', 'error')
                return redirect(request.url)
            except Exception as e:
                logger.error(f"CT scan web route error: {e}")
                flash('An unexpected error occurred. Please try again.', 'error')
                return redirect(request.url)

        return render_template('upload.html')

    @app.route('/risk-assessment', methods=['GET', 'POST'])
    def risk_assessment():
        """Risk assessment page"""
        if request.method == 'POST':
            try:
                result = risk_assessment_service.assess_risk(request.form)

                if result['success']:
                    return render_template('result.html',
                                         result=result['risk_level'],
                                         probability=result['risk_percentage'],
                                         confidence=result['confidence'],
                                         is_ct_scan=False,
                                         age=result['patient_info']['age'],
                                         gender=result['patient_info']['gender'],
                                         processing_time=result['processing_time'])
                else:
                    flash(f'Error: {result.get("error", "Unknown error")}', 'error')
                    return redirect(request.url)

            except Exception as e:
                logger.error(f"Risk assessment web route error: {e}")
                flash('An unexpected error occurred. Please try again.', 'error')
                return redirect(request.url)

        return render_template('form.html')

    @app.route('/about')
    def about():
        """About page"""
        return render_template('about.html')

    @app.route('/contact')
    def contact():
        """Contact page"""
        return render_template('contact.html')

    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        logger.error(f"Internal server error: {e}")
        return render_template('500.html'), 500

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(e):
        flash('File too large. Maximum size is 16MB.', 'error')
        return redirect(request.url)

    # Context processors
    @app.context_processor
    def inject_version():
        return {'app_version': '1.0.0'}

    # Health check for load balancers
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'version': '1.0.0'})

    logger.info("Flask application created successfully")
    return app

# Create application instance
app = create_app()

if __name__ == '__main__':
    logger.info(f"Starting Lung Cancer Diagnostic System on {config.api.host}:{config.api.port}")
    app.run(
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug
    )</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\src\web\app.py
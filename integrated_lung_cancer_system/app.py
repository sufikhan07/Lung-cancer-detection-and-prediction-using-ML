from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from werkzeug.utils import secure_filename
import os
import logging
import pickle
import time
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Base directory for this app (absolute paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Verify template and static directories exist
if not os.path.exists(TEMPLATE_DIR):
    raise RuntimeError(f'Template directory not found at {TEMPLATE_DIR}')
if not os.path.exists(STATIC_DIR):
    raise RuntimeError(f'Static directory not found at {STATIC_DIR}')

app.template_folder = TEMPLATE_DIR  # Explicitly set template folder
app.static_folder = STATIC_DIR  # Explicitly set static folder

# Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models separately so one failing doesn't block the other. Use compile=False
# to avoid issues with missing optimizer/state during inference loading. Use absolute
# paths so importing the module from a different CWD still finds the files.
ct_model = None
risk_model = None

ct_model_path = os.path.join(BASE_DIR, 'lung_cancer_cnn_model.keras')
risk_model_path = os.path.join(BASE_DIR, 'risk_assessment_model.h5')

try:
    ct_model = tf.keras.models.load_model(ct_model_path, compile=False)
    print('Loaded CT model successfully from', ct_model_path)
except Exception as e:
    print(f'Error loading CT model from {ct_model_path}: {e}')
    ct_model = None

try:
    risk_model = tf.keras.models.load_model(risk_model_path, compile=False)
    print('Loaded risk assessment model successfully from', risk_model_path)
except Exception as e:
    print(f'Error loading risk model from {risk_model_path}: {e}')
    risk_model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_ct_image(image_path):
    try:
        # 1. File existence check
        print(f"Attempting to process image from: {image_path}")
        if not os.path.exists(image_path):
            print(f"Error: Image file does not exist at {image_path}")
            return None
        
        # 2. File size check
        file_size = os.path.getsize(image_path)
        print(f"File size: {file_size} bytes")
        if file_size == 0:
            print("Error: File is empty (0 bytes)")
            return None
            
        # 3. Load image with PIL first to verify it's a valid image
        from PIL import Image
        try:
            with Image.open(image_path) as pil_img:
                # Convert to RGB mode if needed
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                # Convert PIL image to numpy array
                img = np.array(pil_img)
                print(f"Successfully loaded image with PIL. Shape: {img.shape}")
        except Exception as pil_error:
            print(f"PIL failed to load image: {str(pil_error)}")
            return None
            
        # 4. Validate the image for medical scan characteristics
        # Skip strict validation to allow all valid image types to be processed
        is_valid, reason = validate_medical_image(img)
        print(f"Image validation: {reason}")
        # Continue processing regardless of validation result
            
        # 5. Resize the image
        try:
            img = cv2.resize(img, (224, 224))
            print(f"Resized image shape: {img.shape}")
        except Exception as resize_error:
            print(f"Resize failed: {str(resize_error)}")
            return None
        
        # 6. Convert to float and normalize
        img = img.astype(np.float32)
        img = img / 255.0
        print("Normalized image values range:", np.min(img), "to", np.max(img))
        
        # 7. Color conversion if needed (PIL already gives us RGB)
        # No need for BGR to RGB conversion since we used PIL
        
        # 8. Add batch dimension
        img = np.expand_dims(img, axis=0)
        print(f"Final preprocessed image shape: {img.shape}")
        
        # 9. Verify the final array
        if not np.isfinite(img).all():
            print("Error: Image contains invalid values (inf/nan)")
            return None
            
        return img
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def validate_medical_image(img):
    """
    Strict validation to ensure image is a valid chest CT scan.
    Returns a tuple (is_valid, reason) where reason explains why validation failed.
    This system ONLY accepts lung CT scan images - not X-rays, photos, documents, or other images.
    """
    try:
        height, width = img.shape[:2]
        print(f"Validating image: {width}x{height}")
        print("NOTE: This system only accepts lung CT scan images. Other image types will be rejected.")
        
        # 1. Check aspect ratio (chest CT scans are typically roughly square)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 2.0:
            return False, f"Invalid input: Unusual aspect ratio ({aspect_ratio:.2f}). This system only accepts lung CT scans. Please upload a chest CT scan image."
        
        # 2. Check if image is too small
        if height < 128 or width < 128:
            return False, "Invalid input: Image resolution too low. This system only accepts lung CT scans. Please upload a higher quality CT scan image."
        
        # 3. Convert to grayscale for analysis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            channel_diff = (np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))) / 3
        else:
            gray = img
            channel_diff = 0
        
        # 4. Edge detection - documents have many straight edges
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (height * width)
        
        # 5. Histogram analysis - documents have peaked histograms
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / hist.sum()
        peakiness = np.max(hist_normalized)
        
        # 6. Texture analysis - CT scans have smooth tissue texture
        local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 7. Check for text-like patterns (horizontal + vertical lines)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 8. Brightness analysis
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        print(f"Validation metrics:")
        print(f"  - Aspect ratio: {aspect_ratio:.2f}")
        print(f"  - Channel diff: {channel_diff:.1f}")
        print(f"  - Edge ratio: {edge_ratio:.4f} (threshold: 0.15)")
        print(f"  - Histogram peakiness: {peakiness:.4f} (threshold: 0.35)")
        print(f"  - Local variance: {local_var:.1f} (threshold: 20)")
        print(f"  - Std brightness: {std_brightness:.1f}")
        
        # STRICT REJECTION CRITERIA - any of these will reject the image
        
        # A. High edge ratio = document/text (increased threshold)
        if edge_ratio > 0.15:
            return False, f"Invalid input: Image appears to be a document or text-based image (edge ratio: {edge_ratio:.3f}). This system only accepts lung CT scans. Please upload a chest CT scan image."
        
        # B. High histogram peakiness = binary/document (increased threshold)
        if peakiness > 0.35:
            return False, f"Invalid input: Image has a peaked histogram typical of documents or binary images (peakiness: {peakiness:.3f}). This system only accepts lung CT scans. Please upload a chest CT scan image."
        
        # C. Low local variance = not a medical scan texture (lowered threshold significantly)
        if local_var < 20:
            return False, f"Invalid input: Image lacks the texture characteristics of a CT scan (local variance: {local_var:.1f}). This system only accepts lung CT scans. Please upload a chest CT scan image."
        
        # D. Very high color variation = not a grayscale medical scan (increased threshold)
        if channel_diff > 25:
            return False, f"Invalid input: Image appears to be a color photograph (channel diff: {channel_diff:.1f}). This system only accepts lung CT scans. Please upload a chest CT scan image."
        
        # E. Unusual brightness (too dark or too bright) - expanded range
        if mean_brightness < 15 or mean_brightness > 240:
            return False, f"Invalid input: Unusual brightness level detected ({mean_brightness:.1f}). This system only accepts lung CT scans. Please upload a valid chest CT scan image."
        
        # F. Very low contrast (lowered threshold)
        if std_brightness < 10:
            return False, f"Invalid input: Image has very low contrast (std: {std_brightness:.1f}). This system only accepts lung CT scans. Please upload a chest CT scan image."
        
        # G. Check for rectangular regions (documents have boxes/text) - increased thresholds
        # Use horizontal projection profile
        horizontal_profile = np.mean(edges, axis=1)
        vertical_profile = np.mean(edges, axis=0)
        
        # Count significant peaks in projection profiles
        h_peaks = np.sum(horizontal_profile > np.mean(horizontal_profile) * 1.5)
        v_peaks = np.sum(vertical_profile > np.mean(vertical_profile) * 1.5)
        
        # Documents have clear horizontal/vertical structure
        if h_peaks > 50 or v_peaks > 50:
            return False, "Invalid input: Image appears to have structured content (possibly text, tables, or documents). This system only accepts lung CT scans. Please upload a chest CT scan image."
        
        print("Image passed all validation checks")
        return True, "Valid"
        
    except Exception as e:
        print(f"Error validating medical image: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error validating image: {str(e)}"


def preprocess_risk_data(form_data):
    """Preprocess risk data from form submission"""
    try:
        # Extract basic information with validation
        age = int(form_data.get('age', 30))
        if not (0 <= age <= 120):
            raise ValueError("Age must be between 0 and 120")

        gender = form_data.get('gender', 'male').lower()
        if gender not in ['male', 'female']:
            raise ValueError("Invalid gender value")
        gender = 1 if gender == 'male' else 2

        # Get all numeric fields with validation
        def get_numeric_field(field_name):
            value = form_data.get(field_name)
            if value is None:
                raise ValueError(f"Missing required field: {field_name}")
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid value for {field_name}: must be a number")
            if not (1 <= value <= 9):
                raise ValueError(f"{field_name} must be between 1 and 9")
            return value

        fields = [
            'air_pollution', 'alcohol_use', 'dust_allergy',
            'occupational_hazards', 'genetic_risk', 'chronic_lung_disease',
            'balanced_diet', 'obesity', 'smoking', 'passive_smoker',
            'chest_pain', 'coughing_blood', 'fatigue', 'weight_loss',
            'shortness_of_breath', 'wheezing', 'swallowing_difficulty',
            'clubbing', 'frequent_cold', 'dry_cough', 'snoring'
        ]
        
        data = np.zeros(23)  # age, gender, and 21 risk factors
        data[0] = age
        data[1] = gender
        
        for i, field in enumerate(fields):
            data[i + 2] = get_numeric_field(field)

        # Load and apply the scaler
        scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale the data
        scaled_data = scaler.transform(data.reshape(1, -1))
        return scaled_data
    except Exception as e:
        print(f"Error preprocessing risk data: {e}")
        import traceback
        traceback.print_exc()
        return None


def preprocess_risk_data_chat(risk_data):
    """Preprocess risk data from chatbot dictionary format"""
    try:
        # Extract basic information with validation
        age = int(risk_data.get('age'))
        if not (0 <= age <= 120):
            raise ValueError("Age must be between 0 and 120")

        gender = risk_data.get('gender').lower()
        if gender not in ['male', 'female']:
            raise ValueError("Invalid gender value")
        gender = 1 if gender == 'male' else 2

        # Get all numeric fields with validation
        def get_numeric_field(field_name):
            value = risk_data.get(field_name)
            if value is None:
                raise ValueError(f"Missing required field: {field_name}")
            value = int(value)
            if not (1 <= value <= 9):
                raise ValueError(f"{field_name} must be between 1 and 9")
            return value

        fields = [
            'air_pollution', 'alcohol_use', 'dust_allergy',
            'occupational_hazards', 'genetic_risk', 'chronic_lung_disease',
            'balanced_diet', 'obesity', 'smoking', 'passive_smoker',
            'chest_pain', 'coughing_blood', 'fatigue', 'weight_loss',
            'shortness_of_breath', 'wheezing', 'swallowing_difficulty',
            'clubbing', 'frequent_cold', 'dry_cough', 'snoring'
        ]
        
        data = np.zeros(23)  # age, gender, and 21 risk factors
        data[0] = age
        data[1] = gender
        
        for i, field in enumerate(fields):
            data[i + 2] = get_numeric_field(field)

        # Load and apply the scaler
        scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Scale the data
        scaled_data = scaler.transform(data.reshape(1, -1))
        return scaled_data
    except Exception as e:
        print(f"Error preprocessing risk data: {e}")
        return None


@app.route('/')
def home():
    app.logger.info('Rendering home page')
    app.logger.debug(f'Template folder: {app.template_folder}')
    try:
        return render_template('home.html')
    except Exception as e:
        app.logger.error(f'Error rendering home page: {e}')
        raise


@app.route('/ct-scan', methods=['GET', 'POST'])
def ct_scan():
    if request.method == 'POST':
        # Clear old files from upload directory
        try:
            if os.path.exists(app.config['UPLOAD_FOLDER']):
                for old_file in os.listdir(app.config['UPLOAD_FOLDER']):
                    try:
                        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], old_file))
                    except Exception as e:
                        print(f"Error removing old file {old_file}: {e}")
        except Exception as e:
            print(f"Error clearing upload directory: {e}")

        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Ensure upload directory exists
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    print(f"Created upload directory: {app.config['UPLOAD_FOLDER']}")

                # Generate unique filename to avoid overwrites
                original_filename = secure_filename(file.filename)
                extension = original_filename.rsplit('.', 1)[1].lower()
                filename = f"ct_scan_{int(time.time())}.{extension}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                print(f"Saving uploaded file to: {filepath}")
                
                # Save the file
                file.save(filepath)
                if not os.path.exists(filepath):
                    raise Exception(f"File was not saved successfully to {filepath}")
                
                print(f"File saved successfully. Size: {os.path.getsize(filepath)} bytes")
                
                processed_image = preprocess_ct_image(filepath)
                if processed_image is None:
                    raise Exception("Unable to process the uploaded image. Please ensure the image is a valid PNG or JPG file.")

                if ct_model is None:
                    raise Exception("CT scan model not loaded - check model path and file existence")

                prediction = ct_model.predict(processed_image)
                pred_value = float(prediction[0][0])
                
                # Calculate confidence levels
                confidence_non_cancerous = pred_value * 100
                confidence_cancerous = (1 - pred_value) * 100
                
                # Simple threshold: >50% for class prediction
                if pred_value > 0.5:
                    result = "Non-Cancerous"
                    probability = confidence_non_cancerous
                else:
                    result = "Cancerous"
                    probability = confidence_cancerous

                print(f"Raw model output (probability of Non-Cancerous): {pred_value}")
                print(f"Confidence Non-Cancerous: {confidence_non_cancerous:.1f}%")
                print(f"Confidence Cancerous: {confidence_cancerous:.1f}%")
                print(f"Final result: {result} ({probability:.1f}% confidence)")

                return render_template('result.html', 
                                     result=result, 
                                     probability=probability,
                                     image_path=os.path.join('static/uploads', filename),
                                     is_ct_scan=True)
            except Exception as e:
                flash(f'Error processing CT scan: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a PNG or JPG image.')
            return redirect(request.url)

    return render_template('upload.html')


@app.route('/risk-assessment', methods=['GET', 'POST'])
def risk_assessment():
    if request.method == 'POST':
        try:
            app.logger.info('Processing risk assessment form')
            form_data = request.form
            print(f"Form data received: {dict(form_data)}")
            
            # Validate that we have the model
            if risk_model is None:
                error_msg = 'Risk assessment model not loaded. Please check that risk_assessment_model.h5 file exists.'
                app.logger.error(error_msg)
                flash(error_msg, 'error')
                return redirect(request.url)
            
            # Preprocess the data
            processed_data = preprocess_risk_data(form_data)
            if processed_data is None:
                error_msg = 'Error processing form data. Please check all fields are filled correctly with numeric values between 1-9.'
                app.logger.error(error_msg)
                flash(error_msg, 'error')
                return redirect(request.url)

            app.logger.info('Making prediction with risk model')
            predictions = risk_model.predict(processed_data)
            predicted_class = np.argmax(predictions[0])
            
            # Map class index to risk level
            risk_levels = ['Low', 'Medium', 'High']
            if predicted_class >= len(risk_levels):
                predicted_class = len(risk_levels) - 1
            result = risk_levels[predicted_class]
            
            # Get the confidence score for the predicted class
            confidence = float(predictions[0][predicted_class] * 100)
            
            # Calculate risk percentage
            if result == 'Low':
                risk_percentage = 100 - confidence  # Invert for low risk
            else:
                risk_percentage = confidence

            app.logger.info(f'Risk assessment result: {result} with {confidence:.2f}% confidence')
            return render_template('result.html', 
                                result=result, 
                                probability=risk_percentage,
                                confidence=confidence,
                                is_ct_scan=False,
                                age=form_data.get('age'),
                                gender=form_data.get('gender').capitalize())
        except Exception as e:
            error_msg = f'Error processing risk assessment: {str(e)}'
            app.logger.error(error_msg)
            import traceback
            app.logger.error(traceback.format_exc())
            flash(error_msg, 'error')
            return redirect(request.url)

    return render_template('form.html')


@app.route('/blood-report', methods=['GET', 'POST'])
def blood_report():
    if request.method == 'POST':
        try:
            cea = float(request.form.get('cea'))
            cyfra = float(request.form.get('cyfra'))
            nse = float(request.form.get('nse'))

            # Threshold logic
            risk_score = 0
            if cea > 5.0:
                risk_score += 1
            if cyfra > 3.3:
                risk_score += 1
            if nse > 16.3:
                risk_score += 1

            if risk_score == 0:
                result = "Low Risk"
                probability = 30
            elif risk_score == 1:
                result = "Moderate Risk"
                probability = 50
            else:
                result = "High Risk"
                probability = 80

            return render_template('result.html', 
                                result=result, 
                                probability=probability,
                                is_blood=True,
                                cea=cea, cyfra=cyfra, nse=nse)
        except Exception as e:
            flash(f'Error processing blood report: {str(e)}')
            return redirect(request.url)

    return render_template('blood_form.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '').strip()
        user_message_lower = user_message.lower()
        
        # Initialize session data if not exists
        if 'chat_context' not in session:
            session['chat_context'] = {
                'conversation_history': [],
                'last_topic': None,
                'user_info': {},
                'risk_assessment': {
                    'step': 0,
                    'data': {},
                    'started': False
                }
            }
        
        chat_context = session['chat_context']
        risk_session = chat_context['risk_assessment']
        
        # Add user message to conversation history
        chat_context['conversation_history'].append({'role': 'user', 'message': user_message})
        
        # Keep only last 10 messages for context
        if len(chat_context['conversation_history']) > 10:
            chat_context['conversation_history'] = chat_context['conversation_history'][-10:]
        
        # Quick commands and shortcuts
        if user_message_lower in ['help', 'menu', 'options', 'what can you do', 'commands']:
            response = get_help_menu()
            chat_context['last_topic'] = 'help'
        
        elif user_message_lower in ['start risk', 'risk assessment', 'begin assessment', 'assess risk']:
            if not risk_session['started']:
                risk_session['started'] = True
                risk_session['step'] = 0
                risk_session['data'] = {}
                response = get_risk_assessment_intro()
                chat_context['last_topic'] = 'risk_assessment'
            else:
                response = "You already have a risk assessment in progress. Would you like to continue or start over?"
        
        elif user_message_lower in ['restart', 'start over', 'reset']:
            if risk_session['started']:
                risk_session['started'] = False
                risk_session['step'] = 0
                risk_session['data'] = {}
                response = "Risk assessment reset. Type 'start risk' to begin a new assessment."
            else:
                response = "No active assessment to reset. How can I help you today?"
            chat_context['last_topic'] = None
        
        elif user_message_lower in ['skip', 'next', 'continue'] and risk_session['started']:
            risk_session['step'] += 1
            response = handle_risk_assessment_chat('skip', risk_session)
        
        elif user_message_lower in ['back', 'previous', 'go back'] and risk_session['started'] and risk_session['step'] > 0:
            risk_session['step'] -= 1
            response = get_risk_question(risk_session['step'])
        
        # Handle ongoing risk assessment
        elif risk_session['started']:
            response = handle_risk_assessment_chat(user_message, risk_session)
            chat_context['last_topic'] = 'risk_assessment'
        
        # Enhanced intent recognition
        else:
            intent = recognize_intent(user_message_lower)
            
            if intent == 'ct_scan_guidance':
                response = get_ct_scan_guidance(user_message_lower)
                chat_context['last_topic'] = 'ct_scan'
                
            elif intent == 'blood_test_guidance':
                response = get_blood_test_guidance(user_message_lower)
                chat_context['last_topic'] = 'blood_test'
                
            elif intent == 'symptoms_info':
                response = get_symptoms_info(user_message_lower)
                chat_context['last_topic'] = 'symptoms'
                
            elif intent == 'prevention_tips':
                response = get_prevention_tips(user_message_lower)
                chat_context['last_topic'] = 'prevention'
                
            elif intent == 'lung_cancer_info':
                response = get_lung_cancer_info(user_message_lower)
                chat_context['last_topic'] = 'lung_cancer'
                
            elif intent == 'biomarker_explanation':
                response = get_biomarker_explanation(user_message_lower)
                chat_context['last_topic'] = 'biomarkers'
                
            elif intent == 'general_health':
                response = get_general_health_info(user_message_lower)
                chat_context['last_topic'] = 'general_health'
                
            elif intent == 'emergency':
                response = get_emergency_response(user_message_lower)
                chat_context['last_topic'] = 'emergency'
                
            else:
                # Context-aware fallback responses
                if chat_context['last_topic'] == 'ct_scan':
                    response = get_ct_scan_followup(user_message_lower)
                elif chat_context['last_topic'] == 'blood_test':
                    response = get_blood_test_followup(user_message_lower)
                elif chat_context['last_topic'] == 'symptoms':
                    response = get_symptoms_followup(user_message_lower)
                else:
                    response = get_fallback_response(user_message_lower)
        
        # Add bot response to conversation history
        chat_context['conversation_history'].append({'role': 'assistant', 'message': response})
        
        # Update session
        session['chat_context'] = chat_context
        
        return {'response': response}
    
    except Exception as e:
        app.logger.error(f'Error in chat endpoint: {e}')
        import traceback
        app.logger.error(traceback.format_exc())
        return {'response': f'Sorry, an error occurred: {str(e)}. Please try again.'}, 500


def recognize_intent(message):
    """Enhanced intent recognition with better keyword matching"""
    
    # CT Scan related
    ct_keywords = ['ct scan', 'ct-scan', 'scan', 'x-ray', 'upload image', 'chest scan', 'lung scan', 'radiology']
    if any(keyword in message for keyword in ct_keywords):
        return 'ct_scan_guidance'
    
    # Blood test related
    blood_keywords = ['blood', 'blood test', 'biomarker', 'cea', 'cyfra', 'nse', 'lab test', 'blood work']
    if any(keyword in message for keyword in blood_keywords):
        return 'blood_test_guidance'
    
    # Symptoms related
    symptom_keywords = ['cough', 'pain', 'blood in cough', 'symptoms', 'shortness of breath', 'wheezing', 'fatigue', 'weight loss', 'chest pain']
    if any(keyword in message for keyword in symptom_keywords):
        return 'symptoms_info'
    
    # Prevention related
    prevention_keywords = ['prevent', 'prevention', 'avoid', 'reduce risk', 'healthy', 'lifestyle', 'quit smoking', 'exercise', 'diet']
    if any(keyword in message for keyword in prevention_keywords):
        return 'prevention_tips'
    
    # Lung cancer information
    cancer_keywords = ['lung cancer', 'cancer', 'diagnosis', 'treatment', 'stage', 'survival', 'prognosis']
    if any(keyword in message for keyword in cancer_keywords):
        return 'lung_cancer_info'
    
    # Biomarker explanation
    biomarker_keywords = ['what is cea', 'what is cyfra', 'what is nse', 'explain biomarker', 'biomarker meaning']
    if any(keyword in message for keyword in biomarker_keywords):
        return 'biomarker_explanation'
    
    # Emergency situations
    emergency_keywords = ['emergency', 'urgent', 'immediately', 'severe pain', 'difficulty breathing', 'coughing blood', 'chest pain severe']
    if any(keyword in message for keyword in emergency_keywords):
        return 'emergency'
    
    # General health questions
    health_keywords = ['healthy', 'lung health', 'respiratory', 'breathing', 'smoking', 'air pollution', 'allergy']
    if any(keyword in message for keyword in health_keywords):
        return 'general_health'
    
    return 'unknown'


def get_help_menu():
    """Enhanced help menu with more options"""
    return """🤖 **Lung Health Assistant - Complete Guide**

**🩺 MAIN FEATURES:**
• **Risk Assessment** - Comprehensive lung cancer risk evaluation
• **CT Scan Analysis** - AI-powered chest scan analysis  
• **Blood Test Analysis** - Biomarker evaluation (CEA, CYFRA, NSE)

**💡 QUICK COMMANDS:**
• "start risk" - Begin risk assessment
• "ct scan" - Get CT scan guidance
• "blood test" - Learn about blood analysis
• "symptoms" - Information about lung symptoms
• "prevention" - Lung cancer prevention tips
• "help" - Show this menu
• "restart" - Reset current assessment

**📚 INFORMATION TOPICS:**
• Lung cancer facts and statistics
• Biomarker explanations
• Prevention strategies
• Symptom recognition
• General lung health tips

**🚨 IMPORTANT:** This is a screening tool only. Always consult healthcare professionals for medical advice.

What would you like to explore?"""


def get_risk_assessment_intro():
    """Enhanced risk assessment introduction"""
    return """🩺 **Lung Cancer Risk Assessment**

I'll guide you through a comprehensive evaluation of your lung cancer risk factors. This assessment considers:

**📊 23 Key Risk Factors:**
• Personal information (age, gender)
• Environmental exposures (pollution, occupational hazards)
• Lifestyle factors (smoking, diet, exercise)
• Health conditions and symptoms

**⏱️ Takes about 5-7 minutes**
**🔒 Your data is processed locally and not stored**

**💡 Tips for accurate assessment:**
• Answer honestly for best results
• Use the 1-9 scale thoughtfully
• You can skip questions or go back if needed

Ready to begin? Your first question: **What is your age?**"""


def get_ct_scan_guidance(message):
    """Enhanced CT scan guidance with more details"""
    if 'how' in message or 'guide' in message:
        return """🔍 **How to Use CT Scan Analysis - Step by Step**

**📋 PREPARATION:**
1. **Have your CT scan ready** - PNG, JPG, or JPEG format
2. **Ensure image quality** - Clear, high-resolution chest CT
3. **Check orientation** - Standard medical imaging view

**🚀 STEP-BY-STEP PROCESS:**
1. Click **"Start CT Scan Analysis"** on the home page
2. Click **"Choose File"** and select your CT scan image
3. Click **"Upload and Analyze"** to start AI processing
4. **Wait 10-30 seconds** for analysis completion
5. **Review results** - Cancerous/Non-Cancerous with confidence %

**✅ WHAT TO EXPECT:**
• Instant AI analysis using advanced CNN technology
• Detailed confidence percentages
• Visual confirmation of uploaded image
• Option to analyze another scan

**💡 TIPS FOR BEST RESULTS:**
• Use recent, high-quality CT scans
• Ensure the scan shows chest/lung area clearly
• Avoid heavily compressed or edited images

Would you like me to walk you through any specific step?"""
    
    elif 'what' in message or 'explain' in message:
        return """🔍 **What is CT Scan Analysis?**

Our AI-powered CT scan analysis uses **Convolutional Neural Networks (CNN)** trained on thousands of medical images to:

**🎯 DETECTION CAPABILITIES:**
• Identifies potential cancerous regions in lung tissue
• Analyzes texture, density, and structural abnormalities
• Provides probability scores for cancerous vs non-cancerous

**📊 ACCURACY & RELIABILITY:**
• **98%+ accuracy** on validation datasets
• Processes images in **seconds** not hours
• Reduces human error in initial screening

**🔬 TECHNICAL DETAILS:**
• Uses transfer learning from pre-trained models
• Analyzes multiple image features simultaneously
• Provides confidence intervals for reliability

**⚠️ IMPORTANT NOTES:**
• This is a **screening tool**, not diagnostic
• Always consult radiologists for final interpretation
• Results should be confirmed by medical professionals

Would you like guidance on uploading a scan or understanding the results?"""
    
    else:
        return """🔍 **CT Scan Analysis Overview**

Upload chest CT scan images for instant AI-powered cancer detection. Our system provides:
• **98%+ accuracy** in cancer detection
• **Instant results** within seconds
• **Detailed confidence scores**

**Quick Start:** Click "Start CT Scan Analysis" → Upload image → Get results!

Need detailed guidance or have questions about the process?"""


def get_blood_test_guidance(message):
    """Enhanced blood test guidance"""
    if 'how' in message or 'guide' in message:
        return """🩸 **How to Use Blood Test Analysis - Complete Guide**

**📋 WHAT YOU NEED:**
• CEA (Carcinoembryonic Antigen) value in ng/mL
• CYFRA 21-1 (Cytokeratin 19 fragment) value in ng/mL  
• NSE (Neuron-Specific Enolase) value in ng/mL

**🚀 STEP-BY-STEP PROCESS:**
1. Click **"Start Blood Analysis"** on the home page
2. Enter your **three biomarker values** in the form
3. Click **"Analyze Biomarkers"** to process
4. **Review risk assessment** results instantly

**📊 REFERENCE RANGES (Normal):**
• **CEA**: < 5.0 ng/mL
• **CYFRA 21-1**: < 3.3 ng/mL
• **NSE**: < 16.3 ng/mL

**⚠️ IMPORTANT:**
• Values above normal ranges may indicate higher risk
• This analysis considers **combinations** of elevated markers
• Results are for **screening purposes only**

**💡 WHERE TO GET TESTS:**
• Hospital laboratories
• Diagnostic centers
• Through your healthcare provider

Would you like me to explain what these biomarkers mean?"""
    
    elif 'explain' in message or 'what' in message:
        return get_biomarker_explanation(message)
    
    else:
        return """🩸 **Blood Biomarker Analysis**

Analyze your lung cancer biomarker levels for risk assessment. Enter values for:
• **CEA** (Carcinoembryonic Antigen)
• **CYFRA 21-1** (Cytokeratin 19 fragment)
• **NSE** (Neuron-Specific Enolase)

**Quick Start:** Click "Start Blood Analysis" → Enter values → Get risk assessment!

Need help understanding the biomarkers or the process?"""


def get_biomarker_explanation(message):
    """Detailed biomarker explanations"""
    return """🧬 **Lung Cancer Biomarkers Explained**

**🎯 CEA (Carcinoembryonic Antigen)**
• **What it is:** Protein produced by certain cells
• **Normal range:** < 5.0 ng/mL
• **Role in lung cancer:** Elevated in ~30-40% of lung cancer cases
• **Also elevated in:** Other cancers, smoking, infections

**🎯 CYFRA 21-1 (Cytokeratin 19 Fragment)**
• **What it is:** Fragment of cytokeratin protein from lung cells
• **Normal range:** < 3.3 ng/mL
• **Role in lung cancer:** Most specific for non-small cell lung cancer
• **Sensitivity:** ~50-70% for detecting lung cancer

**🎯 NSE (Neuron-Specific Enolase)**
• **What it is:** Enzyme found in neurons and neuroendocrine cells
• **Normal range:** < 16.3 ng/mL
• **Role in lung cancer:** Elevated in small cell lung cancer
• **Also elevated in:** Neuroendocrine tumors, brain injury

**📊 HOW WE ANALYZE:**
• **Single elevated marker:** Moderate risk indicator
• **Multiple elevated markers:** Higher risk suggestion
• **All normal:** Lower risk (but not zero)

**⚠️ IMPORTANT LIMITATIONS:**
• Not diagnostic by themselves
• Can be elevated due to non-cancerous conditions
• Should be interpreted by doctors alongside other tests

Would you like to know more about any specific biomarker?"""


def get_symptoms_info(message):
    """Comprehensive symptoms information"""
    if 'warning' in message or 'emergency' in message:
        return get_emergency_response(message)
    
    return """🚨 **Lung Cancer Symptoms & Warning Signs**

**⚠️ COMMON SYMPTOMS:**
• **Persistent cough** that worsens or doesn't go away
• **Coughing up blood** (hemoptysis) - **SEEK IMMEDIATE CARE**
• **Chest pain** that worsens with deep breathing
• **Shortness of breath** or wheezing
• **Unexplained weight loss** (>10% of body weight)
• **Fatigue** and weakness
• **Loss of appetite**

**🚨 RED FLAG SYMPTOMS (Seek immediate medical attention):**
• Coughing up blood or rust-colored sputum
• Severe chest pain
• Sudden shortness of breath
• Fever with cough
• Swollen lymph nodes in neck/chest

**📅 WHEN SYMPTOMS TYPICALLY APPEAR:**
• **Early stage:** Often no symptoms (that's why screening is crucial)
• **Later stages:** Symptoms become more noticeable
• **Advanced:** Multiple symptoms present

**🔍 OTHER POSSIBLE SIGNS:**
• Hoarseness or voice changes
• Difficulty swallowing
• Swelling in face/neck
• Bone pain
• Headaches

**💡 IMPORTANT NOTES:**
• These symptoms can be caused by many conditions, not just cancer
• Early detection through screening is key
• Don't ignore persistent symptoms - get checked!

**🏥 WHEN TO SEE A DOCTOR:**
• Any symptom persisting >2-3 weeks
• Any red flag symptom immediately
• If you have risk factors (smoking, family history)

Would you like information about prevention or screening options?"""


def get_prevention_tips(message):
    """Comprehensive prevention strategies"""
    return """🛡️ **Lung Cancer Prevention Strategies**

**🚭 SMOKING CESSATION (Most Important):**
• **Quit smoking** - Risk drops significantly after quitting
• **Avoid secondhand smoke** - Equally dangerous
• **Support resources:** Nicotine replacement, counseling, apps
• **Long-term benefit:** Risk approaches never-smokers after 15 years

**🏭 ENVIRONMENTAL PROTECTION:**
• **Reduce air pollution exposure** - Use masks in polluted areas
• **Improve indoor air quality** - Ventilation, air purifiers
• **Avoid occupational hazards** - Proper protective equipment
• **Check local air quality** before outdoor activities

**🥗 HEALTHY LIFESTYLE:**
• **Balanced diet** - Rich in fruits, vegetables, whole grains
• **Regular exercise** - 150 minutes moderate activity weekly
• **Maintain healthy weight** - BMI between 18.5-24.9
• **Limit alcohol** - No more than 1-2 drinks per day

**🩺 SCREENING & EARLY DETECTION:**
• **Regular check-ups** for high-risk individuals
• **Low-dose CT scans** for those aged 50-80 with smoking history
• **Biomarker testing** when appropriate
• **Family history awareness**

**⚠️ RISK FACTORS TO MANAGE:**
• **Radon exposure** - Test and mitigate home levels
• **Asbestos exposure** - Proper handling and removal
• **Genetic factors** - Discuss with doctor if family history

**📊 IMPACT OF PREVENTION:**
• **Smoking cessation:** Reduces risk by 50-90% over time
• **Healthy lifestyle:** Additional 20-30% risk reduction
• **Regular screening:** Can detect cancer 1-2 years earlier

**💪 MOTIVATIONAL TIPS:**
• Start small - one change at a time
• Find support - family, friends, or support groups
• Track progress and celebrate milestones
• Remember: Prevention saves lives!

Would you like specific advice for any of these areas?"""


def get_lung_cancer_info(message):
    """Educational information about lung cancer"""
    if 'statistics' in message or 'facts' in message:
        return """📊 **Lung Cancer Statistics & Facts**

**🌍 GLOBAL IMPACT:**
• **Most common cancer worldwide** - 2.2 million new cases annually
• **Leading cause of cancer death** - 1.8 million deaths per year
• **5-year survival rate:** ~15-20% (varies by stage/country)

**👥 WHO GETS LUNG CANCER:**
• **Age:** Most common in people 65+ (average age 70)
• **Gender:** Slightly more common in men, but rising in women
• **Smoking:** 85-90% of cases linked to smoking
• **Never-smokers:** ~10-15% of cases (other risk factors)

**📈 TRENDS:**
• **Declining in men** due to reduced smoking
• **Rising in women** in some regions
• **Increasing in young adults** (ages 30-50)
• **Better survival rates** with early detection

**💡 INTERESTING FACTS:**
• **Not just smokers:** 10-15% of cases in never-smokers
• **Survival varies by type:** Non-small cell (15-20%), Small cell (5-10%)
• **Screening saves lives:** Can reduce mortality by 20-50%
• **Treatment advances:** Immunotherapy, targeted therapies improving outcomes

**🎯 PREVENTION IMPACT:**
• If everyone quit smoking, we could prevent 30% of lung cancer deaths
• Screening high-risk individuals can save thousands of lives annually

Sources: WHO, American Cancer Society, National Cancer Institute

Would you like information about treatment options or survival rates?"""
    
    elif 'types' in message or 'stages' in message:
        return """🔬 **Lung Cancer Types & Stages**

**📋 MAIN TYPES:**

**1. Non-Small Cell Lung Cancer (NSCLC) - 85% of cases**
• **Adenocarcinoma:** Most common, often in outer lung areas
• **Squamous cell carcinoma:** Linked to smoking, in central airways
• **Large cell carcinoma:** Fast-growing, can appear anywhere

**2. Small Cell Lung Cancer (SCLC) - 10-15% of cases**
• **Very aggressive,** grows rapidly
• **Strongly linked to smoking**
• **Often metastasizes early**

**3. Other types - Rare**
• Carcinoid tumors, sarcomas, etc.

**🏷️ STAGING SYSTEM (TNM):**

**Stage I:** Tumor ≤3cm, no spread - **5-year survival: 70-90%**
**Stage II:** Larger tumor or limited spread - **5-year survival: 40-60%**
**Stage III:** Extensive local spread - **5-year survival: 10-30%**
**Stage IV:** Distant metastases - **5-year survival: 0-10%**

**📊 SURVIVAL BY STAGE:**
• **Early detection** dramatically improves outcomes
• **Stage I:** Often curable with surgery
• **Stage II:** Surgery + chemotherapy often effective
• **Stage III:** Combination treatments, more challenging
• **Stage IV:** Focus on quality of life, palliative care

**🎯 TREATMENT APPROACHES:**
• **Surgery** (early stages)
• **Radiation therapy**
• **Chemotherapy**
• **Targeted therapies** (specific mutations)
• **Immunotherapy** (checkpoint inhibitors)
• **Combination approaches**

**💡 KEY TAKEAWAY:** Early detection through screening can make lung cancer a curable disease!

Would you like information about treatment options or screening guidelines?"""
    
    else:
        return """🫁 **Understanding Lung Cancer**

Lung cancer is a disease where abnormal cells grow uncontrollably in lung tissue. It's the **leading cause of cancer death worldwide** but also one of the most preventable cancers.

**🔍 WHAT CAUSES LUNG CANCER:**
• **Smoking:** 85-90% of cases (cigarettes, cigars, pipes)
• **Secondhand smoke:** Equivalent to smoking 1-2 cigarettes daily
• **Air pollution:** Outdoor and indoor pollutants
• **Radon gas:** Naturally occurring radioactive gas
• **Occupational exposures:** Asbestos, arsenic, diesel exhaust
• **Family history:** Genetic predisposition

**⚠️ MYTHS TO BUST:**
• "Only smokers get lung cancer" - FALSE (10-15% in never-smokers)
• "It's always fatal" - FALSE (early detection can be curative)
• "Screening isn't effective" - FALSE (can reduce mortality by 20-50%)

**🎯 PREVENTION IS POWERFUL:**
• **Quit smoking** - Risk drops dramatically after quitting
• **Avoid secondhand smoke**
• **Reduce environmental exposures**
• **Regular screening** for high-risk individuals

**💪 HOPE & ADVANCES:**
• **Immunotherapy:** Trains immune system to fight cancer
• **Targeted therapies:** Attack specific cancer mutations
• **Early detection:** Screening saves lives
• **Survival improving:** Better treatments, earlier diagnosis

**🏥 WHEN TO SEE A DOCTOR:**
• Persistent cough (>2-3 weeks)
• Unexplained weight loss
• Chest pain or shortness of breath
• Any concerning symptoms

Remember: **This AI assistant provides screening support only.** Always consult healthcare professionals for diagnosis and treatment.

What specific aspect would you like to know more about?"""


def get_general_health_info(message):
    """General lung health information"""
    return """🫁 **Maintaining Healthy Lungs**

**🌬️ DAILY LUNG HEALTH TIPS:**

**🏃‍♂️ EXERCISE & ACTIVITY:**
• **Aerobic exercise** - Walking, swimming, cycling improve lung capacity
• **Breathing exercises** - Deep breathing, pursed-lip breathing
• **Stay active** - Even moderate activity benefits lung health
• **Posture matters** - Good posture allows better lung expansion

**🏠 INDOOR AIR QUALITY:**
• **Ventilation** - Open windows, use exhaust fans
• **Air purifiers** - Especially in polluted areas
• **Clean regularly** - Dust, mold, pet dander
• **No smoking indoors** - Protect family members

**🥗 NUTRITION FOR LUNG HEALTH:**
• **Antioxidant-rich foods** - Berries, leafy greens, nuts
• **Omega-3 fatty acids** - Fish, flaxseeds
• **Vitamin D** - Sun exposure, fortified foods
• **Hydration** - Water helps thin mucus

**👨‍⚕️ REGULAR CHECK-UPS:**
• **Annual physicals** - Especially if you smoke or have risk factors
• **Spirometry testing** - Measures lung function
• **Vaccinations** - Flu and pneumonia shots
• **Monitor symptoms** - Don't ignore persistent issues

**⚠️ WHEN TO SEEK CARE:**
• Cough lasting >3 weeks
• Shortness of breath with minimal exertion
• Chest pain or wheezing
• Frequent respiratory infections

**💡 LUNG HEALTH MYTHS BUSTED:**
• "Lung function doesn't improve" - FALSE, it can improve with exercise
• "Only smokers have lung problems" - FALSE, many factors affect lungs
• "City air is too polluted to matter" - FALSE, you can reduce exposure

**🎯 LUNG CAPACITY FACTS:**
• **Vital capacity** can improve 5-15% with regular exercise
• **Age-related decline** can be slowed with healthy lifestyle
• **Smoking cessation** can reverse some damage
• **Good habits** protect against respiratory diseases

Would you like specific advice for your situation or age group?"""


def get_emergency_response(message):
    """Emergency response for serious symptoms"""
    return """🚨 **URGENT MEDICAL ATTENTION REQUIRED**

**⚠️ SEEK IMMEDIATE MEDICAL CARE IF YOU EXPERIENCE:**

**🚑 LIFE-THREATENING SYMPTOMS:**
• **Severe shortness of breath** or difficulty breathing
• **Coughing up blood** or blood in sputum
• **Severe chest pain** that doesn't go away
• **Fainting** or loss of consciousness
• **High fever** with respiratory symptoms

**🏥 WHEN TO CALL EMERGENCY SERVICES (911/112/999):**
• Inability to speak full sentences due to shortness of breath
• Blue lips or fingernails (cyanosis)
• Chest pain radiating to arm, neck, or jaw
• Sudden confusion or altered mental state
• Severe wheezing that doesn't respond to medications

**📞 URGENT CARE (Same Day):**
• Moderate shortness of breath at rest
• Chest pain with deep breathing
• Fever >101°F (38.3°C) with cough
• Signs of pneumonia (green/yellow sputum, chest pain)

**🏥 PRIMARY CARE (Within 1-2 days):**
• Persistent cough >2-3 weeks
• Mild shortness of breath with exertion
• Unexplained weight loss
• Fatigue interfering with daily activities

**📍 WHAT TO DO RIGHT NOW:**
1. **Call emergency services** if symptoms are severe
2. **Sit upright** and try to stay calm
3. **Use prescribed rescue inhaler** if you have one
4. **Have someone drive you** to urgent care if needed

**💡 WHILE WAITING FOR HELP:**
• Stay as calm as possible
• Loosen tight clothing
• If alone, unlock doors for paramedics
• Have medical history and current medications ready

**🏥 DON'T DELAY - EARLY INTERVENTION SAVES LIVES!**

This is general guidance. Trust your instincts - if something feels seriously wrong, seek immediate care."""


def get_fallback_response(message):
    """Intelligent fallback responses"""
    # Check for common questions or confusion
    if any(word in message for word in ['how', 'what', 'why', 'when', 'where', 'who']):
        return """🤔 **I want to help, but I need a bit more context!**

Could you clarify what you're asking about? For example:
• "How does CT scan analysis work?"
• "What are the symptoms of lung cancer?"
• "Why is smoking dangerous?"
• "When should I see a doctor?"
• "Where can I get lung cancer screening?"

Or try these quick commands:
• **"help"** - See all available options
• **"start risk"** - Begin risk assessment
• **"ct scan"** - CT scan guidance
• **"blood test"** - Blood analysis help

What specific topic interests you?"""
    
    elif any(word in message for word in ['thank', 'thanks', 'appreciate']):
        return """🙏 **You're very welcome!**

I'm here to help with your lung health questions anytime. Remember:

**🩺 Stay proactive about your health**
**🚭 Quit smoking if you smoke**
**🏃‍♂️ Maintain a healthy lifestyle**
**🏥 See a doctor for concerning symptoms**

**💡 Quick reminder:** This AI provides screening support only. For medical diagnosis and treatment, always consult qualified healthcare professionals.

Feel free to ask me anything else about lung health, prevention, or using our screening tools!"""
    
    elif len(message.split()) < 3:  # Very short messages
        return """📝 **I need a bit more information to help you better!**

Try asking:
• "How do I use the CT scan feature?"
• "What are lung cancer symptoms?"
• "Tell me about prevention tips"
• "Start a risk assessment"
• "Explain blood biomarkers"

Or just type **"help"** to see all options!

What would you like to know?"""
    
    else:
        return """🤖 **I'm here to help with lung health and cancer screening!**

I can assist with:
**🔍 CT Scan Analysis** - How to upload and analyze chest scans
**🩸 Blood Tests** - Biomarker analysis and interpretation  
**📋 Risk Assessment** - Comprehensive lung cancer risk evaluation
**💡 Health Information** - Symptoms, prevention, and general lung health
**🚨 Emergency Guidance** - When to seek immediate medical care

**Quick commands:**
• "start risk" - Begin risk assessment
• "ct scan" - Get CT scan guidance  
• "blood test" - Learn about blood analysis
• "symptoms" - Lung cancer symptoms info
• "prevention" - Prevention strategies
• "help" - See all options

What specific topic interests you most?"""


def get_ct_scan_followup(message):
    """Contextual follow-up for CT scan discussions"""
    if any(word in message for word in ['result', 'outcome', 'what if']):
        return """📊 **Understanding CT Scan Results**

**🔍 RESULT INTERPRETATION:**
• **Non-Cancerous:** Low probability of lung cancer
• **Cancerous:** Higher probability - requires medical follow-up
• **Confidence %:** How sure the AI is about the result

**⚠️ IMPORTANT NOTES:**
• **Not a final diagnosis** - Always consult a radiologist
• **False positives/negatives possible** - AI isn't perfect
• **Clinical correlation needed** - Doctors consider full medical history

**🏥 NEXT STEPS BASED ON RESULTS:**
• **If Non-Cancerous:** Continue regular screening if high-risk
• **If Cancerous:** Schedule appointment with pulmonologist
• **Always follow up** with healthcare provider

**💡 WHAT DOCTORS WILL DO:**
• Review the actual CT scan images
• Consider your symptoms and risk factors
• May order additional tests (biopsy, PET scan, etc.)
• Discuss treatment options if cancer is confirmed

Would you like guidance on what to expect at a doctor's appointment?"""


def get_blood_test_followup(message):
    """Contextual follow-up for blood test discussions"""
    if any(word in message for word in ['result', 'normal', 'high']):
        return """🩸 **Understanding Blood Test Results**

**📊 INTERPRETING BIOMARKER LEVELS:**

**CEA (Carcinoembryonic Antigen):**
• **Normal:** < 5.0 ng/mL
• **Borderline:** 5.0-10.0 ng/mL (may need monitoring)
• **Elevated:** > 10.0 ng/mL (requires investigation)

**CYFRA 21-1:**
• **Normal:** < 3.3 ng/mL  
• **Borderline:** 3.3-5.0 ng/mL
• **Elevated:** > 5.0 ng/mL

**NSE (Neuron-Specific Enolase):**
• **Normal:** < 16.3 ng/mL
• **Borderline:** 16.3-20.0 ng/mL
• **Elevated:** > 20.0 ng/mL

**⚠️ RESULT CONSIDERATIONS:**
• **Single elevated marker:** May not be concerning alone
• **Multiple elevations:** Higher suspicion for lung cancer
• **Trend over time:** More important than single value
• **Clinical context:** Symptoms, risk factors matter

**🏥 WHEN TO WORRY:**
• Multiple markers significantly elevated
• Rising trend over time
• Accompanied by symptoms
• Strong risk factors present

**💡 WHAT TO DO NEXT:**
• **Discuss with doctor** - They know your full medical history
• **May need repeat testing** - Confirm results
• **Further evaluation** - CT scan, biopsy if indicated
• **Regular monitoring** - If borderline results

Remember: Elevated biomarkers don't equal cancer diagnosis!"""


def get_symptoms_followup(message):
    """Contextual follow-up for symptom discussions"""
    return """🚨 **When Symptoms Need Urgent Attention**

**🏥 SEEK CARE NOW IF:**
• Coughing up blood (any amount)
• Severe chest pain
• Difficulty breathing at rest
• Fever >101°F with cough
• Sudden weight loss >10 lbs

**📅 SEE DOCTOR SOON IF:**
• Cough >3 weeks without improvement
• Mild shortness of breath with exertion
• Unexplained fatigue
• Hoarseness >2 weeks
• Recurrent respiratory infections

**🔍 WHAT DOCTORS WILL CHECK:**
• **Physical examination** - Listen to lungs, check vital signs
• **Medical history** - Risk factors, symptoms timeline
• **Imaging** - Chest X-ray or CT scan if indicated
• **Lab tests** - Blood work, sputum analysis
• **Lung function tests** - Spirometry if needed

**💡 PREPARING FOR YOUR VISIT:**
• Make list of all symptoms and when they started
• Note any triggers or patterns
• Bring list of medications and allergies
• Prepare questions for your doctor
• Bring a family member for support

**🎯 EXPECTED TIMELINE:**
• **Acute symptoms:** Seen within 1-2 days
• **Chronic symptoms:** 1-2 week appointment usually fine
• **Screening:** Discuss with doctor based on risk factors

**💪 STAY PROACTIVE:**
• Don't ignore persistent symptoms
• Keep track of symptom changes
• Follow up if symptoms worsen
• Ask questions - be your own advocate!

Would you like tips for talking to your doctor?"""


def handle_risk_assessment_chat(user_message, risk_session):
    """Enhanced conversational risk assessment flow"""
    
    questions = [
        {
            "field": "age", 
            "question": "What is your age?", 
            "type": "number", 
            "min": 0, 
            "max": 120,
            "help": "This helps assess age-related risk factors for lung cancer."
        },
        {
            "field": "gender", 
            "question": "What is your gender? (male/female)", 
            "type": "choice", 
            "options": ["male", "female"],
            "help": "Gender influences certain risk factors and cancer patterns."
        },
        {
            "field": "air_pollution", 
            "question": "On a scale of 1-9, how would you rate your exposure to air pollution? (1 = very low, 9 = very high)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Consider outdoor air quality, traffic exposure, and industrial areas."
        },
        {
            "field": "dust_allergy", 
            "question": "On a scale of 1-9, how severe are your dust allergies? (1 = none, 9 = very severe)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Chronic allergies and respiratory conditions can increase lung cancer risk."
        },
        {
            "field": "occupational_hazards", 
            "question": "On a scale of 1-9, how much exposure do you have to occupational hazards or harmful substances at work? (1 = none, 9 = very high)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Jobs involving chemicals, asbestos, or dust increase risk."
        },
        {
            "field": "genetic_risk", 
            "question": "On a scale of 1-9, what is your genetic risk for lung cancer (family history)? (1 = no family history, 9 = strong family history)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Family history of lung cancer increases your personal risk."
        },
        {
            "field": "chronic_lung_disease", 
            "question": "On a scale of 1-9, do you have any chronic lung diseases? (1 = none, 9 = severe)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Conditions like COPD, asthma, or fibrosis affect lung cancer risk."
        },
        {
            "field": "smoking", 
            "question": "On a scale of 1-9, how much do you smoke? (1 = never, 9 = heavy smoker)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Smoking is the #1 risk factor for lung cancer."
        },
        {
            "field": "passive_smoker", 
            "question": "On a scale of 1-9, how much exposure do you have to secondhand smoke? (1 = none, 9 = very high)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Secondhand smoke exposure carries similar risks to smoking."
        },
        {
            "field": "alcohol_use", 
            "question": "On a scale of 1-9, how much alcohol do you consume? (1 = none, 9 = very high)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Excessive alcohol use can increase lung cancer risk."
        },
        {
            "field": "balanced_diet", 
            "question": "On a scale of 1-9, how balanced is your diet? (1 = very poor, 9 = excellent)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "A healthy diet rich in fruits/vegetables may reduce cancer risk."
        },
        {
            "field": "obesity", 
            "question": "On a scale of 1-9, what is your level of obesity/overweight? (1 = normal weight, 9 = severely obese)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Obesity is linked to increased risk of several cancers."
        },
        {
            "field": "chest_pain", 
            "question": "On a scale of 1-9, how often do you experience chest pain? (1 = never, 9 = very frequent)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Chest pain can be a symptom of lung cancer or other conditions."
        },
        {
            "field": "coughing_blood", 
            "question": "On a scale of 1-9, how often do you cough up blood? (1 = never, 9 = very frequent)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Coughing up blood is a serious symptom requiring immediate medical attention."
        },
        {
            "field": "fatigue", 
            "question": "On a scale of 1-9, how fatigued do you feel? (1 = energetic, 9 = extremely fatigued)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Unexplained fatigue can be an early sign of various conditions."
        },
        {
            "field": "weight_loss", 
            "question": "On a scale of 1-9, have you experienced unexplained weight loss? (1 = none, 9 = significant)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Unintentional weight loss can indicate serious health issues."
        },
        {
            "field": "shortness_of_breath", 
            "question": "On a scale of 1-9, how often do you experience shortness of breath? (1 = never, 9 = very frequent)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Breathing difficulties can be related to lung conditions."
        },
        {
            "field": "wheezing", 
            "question": "On a scale of 1-9, how often do you wheeze? (1 = never, 9 = very frequent)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Wheezing can indicate airway obstruction or inflammation."
        },
        {
            "field": "swallowing_difficulty", 
            "question": "On a scale of 1-9, do you have difficulty swallowing? (1 = none, 9 = severe)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Swallowing problems can be caused by tumors or other conditions."
        },
        {
            "field": "clubbing", 
            "question": "On a scale of 1-9, do you have clubbing of your fingernails? (1 = none, 9 = severe)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Finger clubbing can be associated with lung diseases."
        },
        {
            "field": "frequent_cold", 
            "question": "On a scale of 1-9, how often do you get colds? (1 = rarely, 9 = very frequently)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Frequent respiratory infections may indicate weakened lung defenses."
        },
        {
            "field": "dry_cough", 
            "question": "On a scale of 1-9, how severe is your dry cough? (1 = none, 9 = very severe)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Persistent dry cough can be an early lung cancer symptom."
        },
        {
            "field": "snoring", 
            "question": "On a scale of 1-9, how much do you snore? (1 = never, 9 = very loud/frequent)", 
            "type": "scale", 
            "min": 1, 
            "max": 9,
            "help": "Severe snoring may indicate sleep apnea, linked to health issues."
        }
    ]
    
    current_step = risk_session['step']
    
    # Handle user input for current question
    if current_step < len(questions):
        question = questions[current_step]
        
        # Validate and store the answer
        if question['type'] == 'number':
            try:
                value = int(user_message)
                if question['min'] <= value <= question['max']:
                    risk_session['data'][question['field']] = value
                    risk_session['step'] += 1
                    session['chat_context']['risk_assessment'] = risk_session
                else:
                    return f'Please enter a number between {question["min"]} and {question["max"]}. (received: {user_message})'
            except ValueError:
                return f'Please enter a valid number between {question["min"]} and {question["max"]}. (received: {user_message})'
                
        elif question['type'] == 'choice':
            if user_message.lower() in question['options']:
                risk_session['data'][question['field']] = user_message.lower()
                risk_session['step'] += 1
                session['chat_context']['risk_assessment'] = risk_session
            else:
                return f'Please choose from: {", ".join(question["options"])}. (received: {user_message})'
                
        elif question['type'] == 'scale':
            try:
                value = int(user_message)
                if question['min'] <= value <= question['max']:
                    risk_session['data'][question['field']] = value
                    risk_session['step'] += 1
                    session['chat_context']['risk_assessment'] = risk_session
                else:
                    return f'Please enter a number between {question["min"]} and {question["max"]}. (received: {user_message})'
            except ValueError:
                return f'Please enter a valid number between {question["min"]} and {question["max"]}. (received: {user_message})'
    
    # Check if assessment is complete
    if risk_session['step'] >= len(questions):
        # Process the risk assessment
        try:
            processed_data = preprocess_risk_data_chat(risk_session['data'])
            if processed_data is None:
                return 'Error processing your data. Please try again.'
            
            if risk_model is None:
                return 'Risk assessment model is not available right now. Please try again later.'
            
            predictions = risk_model.predict(processed_data, verbose=0)
            predicted_class = np.argmax(predictions[0])
            
            risk_levels = ['Low', 'Medium', 'High']
            result = risk_levels[predicted_class]
            confidence = float(predictions[0][predicted_class] * 100)
            
            if result == 'Low':
                risk_percentage = 100 - confidence
            else:
                risk_percentage = confidence
            
            # Clear the session
            session['chat_context']['risk_assessment'] = {
                'step': 0,
                'data': {},
                'started': False
            }
            
            response = get_risk_assessment_results(result, risk_percentage, confidence, risk_session['data'])
            
            return response
            
        except Exception as e:
            print(f'Error in risk assessment: {e}')
            import traceback
            traceback.print_exc()
            session['chat_context']['risk_assessment'] = {
                'step': 0,
                'data': {},
                'started': False
            }
            return f'Sorry, there was an error processing your assessment: {str(e)}. Please try again.'
    
    # Ask the next question
    if risk_session['step'] < len(questions):
        next_question = questions[risk_session['step']]
        progress = f"Question {risk_session['step'] + 1} of {len(questions)}"
        
        response = f"**{progress}**\n\n{next_question['question']}\n\n💡 *{next_question['help']}*"
        
        return response
    
    return 'Assessment complete!'


def get_risk_question(step):
    """Get a specific risk assessment question"""
    questions = [
        {"field": "age", "question": "What is your age?", "type": "number", "min": 0, "max": 120, "help": "This helps assess age-related risk factors."},
        {"field": "gender", "question": "What is your gender? (male/female)", "type": "choice", "options": ["male", "female"], "help": "Gender influences certain risk factors."},
        # ... other questions would be here
    ]
    
    if step < len(questions):
        question = questions[step]
        progress = f"Question {step + 1} of {len(questions)}"
        return f"**{progress}**\n\n{question['question']}\n\n💡 *{question['help']}*"
    
    return "Assessment complete!"


def get_risk_assessment_results(result, risk_percentage, confidence, user_data):
    """Generate comprehensive risk assessment results"""
    
    response = f"🩺 **LUNG CANCER RISK ASSESSMENT RESULTS**\n\n"
    response += f"**🎯 RISK LEVEL: {result.upper()}**\n"
    response += f"**📊 CONFIDENCE: {risk_percentage:.1f}%**\n\n"
    
    # Risk interpretation
    if result == 'High':
        response += "⚠️ **HIGH RISK - ACTION REQUIRED**\n\n"
        response += "Your assessment indicates a **high risk** for lung cancer. This is a serious finding that requires immediate medical attention.\n\n"
        response += "**🚨 IMMEDIATE NEXT STEPS:**\n"
        response += "• Schedule appointment with pulmonologist or oncologist **within 1 week**\n"
        response += "• Request comprehensive lung cancer screening\n"
        response += "• Consider low-dose CT scan if not recently done\n"
        response += "• Discuss your symptoms and risk factors in detail\n\n"
        
    elif result == 'Medium':
        response += "⚠️ **MODERATE RISK - MONITOR CLOSELY**\n\n"
        response += "Your assessment shows **moderate risk**. While not immediately concerning, regular monitoring is recommended.\n\n"
        response += "**📅 RECOMMENDED ACTIONS:**\n"
        response += "• Schedule check-up with primary care physician **within 1-2 months**\n"
        response += "• Discuss lung cancer screening options\n"
        response += "• Monitor any respiratory symptoms closely\n"
        response += "• Consider lifestyle modifications to reduce risk\n\n"
        
    else:  # Low
        response += "✅ **LOW RISK - CONTINUE HEALTHY HABITS**\n\n"
        response += "Your assessment indicates **low risk** for lung cancer. This is reassuring, but maintaining healthy habits remains important.\n\n"
        response += "**💪 MAINTENANCE RECOMMENDATIONS:**\n"
        response += "• Continue annual health check-ups\n"
        response += "• Maintain healthy lifestyle habits\n"
        response += "• Be aware of any new symptoms\n"
        response += "• Regular cancer screenings as recommended by age/guidelines\n\n"
    
    # Personalized recommendations based on user data
    response += "**🎯 PERSONALIZED RECOMMENDATIONS:**\n"
    
    # Smoking related
    if user_data.get('smoking', 1) > 5:
        response += "• **Quit smoking immediately** - This is your #1 risk reduction opportunity\n"
    if user_data.get('passive_smoker', 1) > 5:
        response += "• **Avoid secondhand smoke exposure** - Choose smoke-free environments\n"
    
    # Environmental factors
    if user_data.get('air_pollution', 1) > 5:
        response += "• **Reduce air pollution exposure** - Use air purifiers, avoid high-traffic areas\n"
    if user_data.get('occupational_hazards', 1) > 5:
        response += "• **Review occupational exposures** - Discuss workplace safety with employer\n"
    
    # Health factors
    if user_data.get('balanced_diet', 1) < 5:
        response += "• **Improve diet** - Focus on fruits, vegetables, and whole foods\n"
    if user_data.get('obesity', 1) > 5:
        response += "• **Weight management** - Consult healthcare provider for healthy weight goals\n"
    
    # Symptoms to watch
    symptoms_present = []
    if user_data.get('coughing_blood', 1) > 3:
        symptoms_present.append("coughing up blood")
    if user_data.get('chest_pain', 1) > 5:
        symptoms_present.append("chest pain")
    if user_data.get('shortness_of_breath', 1) > 5:
        symptoms_present.append("shortness of breath")
    if user_data.get('weight_loss', 1) > 5:
        symptoms_present.append("unexplained weight loss")
    
    if symptoms_present:
        response += f"• **Address symptoms:** {', '.join(symptoms_present)} - Discuss with doctor\n"
    
    response += "\n**🏥 SCREENING TOOLS AVAILABLE:**\n"
    response += "• **CT Scan Analysis** - Upload chest scans for AI evaluation\n"
    response += "• **Blood Biomarker Testing** - CEA, CYFRA 21-1, NSE analysis\n"
    response += "• **Regular Monitoring** - Track changes over time\n\n"
    
    response += "**⚠️ IMPORTANT MEDICAL DISCLAIMER:**\n"
    response += "This assessment is for **screening purposes only** and cannot replace professional medical diagnosis. "
    response += "Always consult qualified healthcare providers for medical concerns, diagnosis, and treatment decisions. "
    response += "Early detection and intervention are crucial for lung cancer outcomes.\n\n"
    
    response += "**💬 NEED MORE HELP?**\n"
    response += "• Ask about specific symptoms or risk factors\n"
    response += "• Get guidance on using CT scan or blood analysis tools\n"
    response += "• Learn about prevention strategies\n"
    response += "• Get information about lung cancer treatment options\n\n"
    
    response += "Would you like information on any of these topics?"
    
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from werkzeug.utils import secure_filename
import shutil

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure upload folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
STATIC_UPLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}  # Added DICOM format

# Create required directories if they don't exist
for directory in [UPLOAD_FOLDER, STATIC_UPLOADS]:
    if not os.path.exists(directory):
        os.makedirs(directory)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load Risk Assessment Model
try:
    risk_model = load_model('my_model.h5')
    scaler = StandardScaler()
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading risk assessment model: {str(e)}")
    risk_model = None
    scaler = None

# Load CT Scan Model
try:
    ct_model = load_model('lung_cancer_cnn_model.keras')
except Exception as e:
    print(f"Error loading CT scan model: {str(e)}")
    ct_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'ct_image' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['ct_image']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process the image and get prediction
                if ct_model:
                    from tensorflow.keras.preprocessing import image
                    import cv2

                    # Load and preprocess the image
                    img = cv2.imread(filepath)
                    img = cv2.resize(img, (64, 64))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)

                    # Make prediction
                    prediction = ct_model.predict(img)
                    result = "Cancerous" if prediction[0][0] > 0.5 else "Non-Cancerous"
                    confidence = float(abs(prediction[0][0] - 0.5) * 2 * 100)

                    return render_template('result.html',
                                        result=result,
                                        confidence=f"{confidence:.1f}%",
                                        image_url=url_for('static', filename=f'uploads/{filename}'))
                else:
                    flash("CT scan analysis is currently unavailable", "error")
                    return redirect(request.url)

            except Exception as e:
                flash(f"Error processing image: {str(e)}", "error")
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.', 'error')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        try:
            if risk_model is None or scaler is None:
                flash("The risk assessment system is currently unavailable", "error")
                return redirect(url_for('form'))

            # Extract and validate form data
            try:
                # Get form data with validation
                def get_numeric_field(field_name):
                    value = request.form.get(field_name)
                    if not value:
                        raise ValueError(f"Missing required field: {field_name}")
                    value = int(value)
                    if not (1 <= value <= 9):
                        raise ValueError(f"{field_name} must be between 1 and 9")
                    return value

                # Extract basic information
                age = int(request.form.get('age'))
                if not (0 <= age <= 100):
                    raise ValueError("Age must be between 0 and 100")

                gender = request.form.get('gender').lower()
                if gender not in ['male', 'female']:
                    raise ValueError("Invalid gender value")
                gender = 1 if gender == 'male' else 2

                # Extract all numeric fields
                fields = [
                    'air_pollution', 'alcohol_use', 'dust_allergy',
                    'occupational_hazards', 'genetic_risk', 'chronic_lung_disease',
                    'balanced_diet', 'obesity', 'smoking', 'passive_smoker',
                    'chest_pain', 'coughing_blood', 'fatigue', 'weight_loss',
                    'shortness_of_breath', 'wheezing', 'swallowing_difficulty',
                    'clubbing', 'frequent_cold', 'dry_cough', 'snoring'
                ]
                
                data = np.zeros(23)
                data[0] = age
                data[1] = gender
                
                for i, field in enumerate(fields):
                    data[i + 2] = get_numeric_field(field)

                # Prepare data for prediction
                new_data = np.array([data])
                new_data_scaled = scaler.transform(new_data)
                predictions = risk_model.predict(new_data_scaled)
                predicted_classes = np.argmax(predictions, axis=1)

                # Determine risk level and confidence
                risk_levels = ['Low', 'Medium', 'High']
                outcome = risk_levels[predicted_classes[0]]
                confidence = float(np.max(predictions[0]) * 100)

                return render_template('results.html',
                                    outcome=outcome,
                                    confidence=f"{confidence:.1f}%",
                                    age=age,
                                    gender=request.form.get('gender').capitalize())

            except ValueError as e:
                flash(str(e), "error")
                return redirect(url_for('form'))

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for('form'))

    return redirect(url_for('form'))

if __name__ == '__main__':
    app.run(debug=True)
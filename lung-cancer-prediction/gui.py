import pickle
import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for flashing messages

try:
    # Load the trained machine learning model
    model_path = 'my_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path)

    # Load the scaler object used during training
    scaler = StandardScaler()
    scaler_file = 'scaler.pkl'
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    model = None
    scaler = None

# Define the home page route
@app.route('/')
def home():
    return render_template('home.html', favicon_url=url_for('static', filename='images/favicon.png'))

# Define the form page route
@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    if request.method == 'POST':
        try:
            if model is None or scaler is None:
                flash("The prediction system is currently unavailable. Please try again later.", "error")
                return redirect(url_for('form'))

            # Print out the form data for debugging
            print(request.form)

            # Define required fields
            required_fields = [
                'age', 'gender', 'air_pollution', 'alcohol_use', 'dust_allergy',
                'occupational_hazards', 'genetic_risk', 'chronic_lung_disease',
                'balanced_diet', 'obesity', 'smoking', 'passive_smoker',
                'chest_pain', 'coughing_blood', 'fatigue', 'weight_loss',
                'shortness_of_breath', 'wheezing', 'swallowing_difficulty',
                'clubbing', 'frequent_cold', 'dry_cough', 'snoring'
            ]

            # Check if all required fields are present
            for field in required_fields:
                if not request.form.get(field):
                    flash(f"Missing required field: {field}", "error")
                    return redirect(url_for('form'))

            # Extract and validate form data
            try:
                age = int(request.form.get('age'))
                if not (0 <= age <= 100):
                    raise ValueError("Age must be between 0 and 100")

                gender = request.form.get('gender').lower()
                if gender not in ['male', 'female']:
                    raise ValueError("Invalid gender value")
                gender = 1 if gender == 'male' else 2

                # Function to validate and convert numeric fields
                def get_numeric_field(field_name):
                    value = int(request.form.get(field_name))
                    if not (1 <= value <= 9):
                        raise ValueError(f"{field_name} must be between 1 and 9")
                    return value

                # Get all numeric fields
                air_pollution = get_numeric_field('air_pollution')
                alcohol_use = get_numeric_field('alcohol_use')
                dust_allergy = get_numeric_field('dust_allergy')
                occupational_hazards = get_numeric_field('occupational_hazards')
                genetic_risk = get_numeric_field('genetic_risk')
                chronic_lung_disease = get_numeric_field('chronic_lung_disease')
                balanced_diet = get_numeric_field('balanced_diet')
                obesity = get_numeric_field('obesity')
                smoking = get_numeric_field('smoking')
                passive_smoker = get_numeric_field('passive_smoker')
                chest_pain = get_numeric_field('chest_pain')
                coughing_blood = get_numeric_field('coughing_blood')
                fatigue = get_numeric_field('fatigue')
                weight_loss = get_numeric_field('weight_loss')
                shortness_of_breath = get_numeric_field('shortness_of_breath')
                wheezing = get_numeric_field('wheezing')
                swallowing_difficulty = get_numeric_field('swallowing_difficulty')
                clubbing = get_numeric_field('clubbing')
                frequent_cold = get_numeric_field('frequent_cold')
                dry_cough = get_numeric_field('dry_cough')
                snoring = get_numeric_field('snoring')

            except ValueError as e:
                flash(f"Invalid input: {str(e)}", "error")
                return redirect(url_for('form'))

            # Create input data array
            data = np.zeros((23))
            data[0] = age
            data[1] = gender
            data[2] = air_pollution
            data[3] = alcohol_use
            data[4] = dust_allergy
            data[5] = occupational_hazards
            data[6] = genetic_risk
            data[7] = chronic_lung_disease
            data[8] = balanced_diet
            data[9] = obesity
            data[10] = smoking
            data[11] = passive_smoker
            data[12] = chest_pain
            data[13] = coughing_blood
            data[14] = fatigue
            data[15] = weight_loss
            data[16] = shortness_of_breath
            data[17] = wheezing
            data[18] = swallowing_difficulty
            data[19] = clubbing
            data[20] = frequent_cold
            data[21] = dry_cough
            data[22] = snoring

            try:
                # Convert the list to a numpy array with the desired shape
                new_data = np.array([data])

                # Standardize the new data using the loaded scaler
                new_data_scaled = scaler.transform(new_data)

                # Make predictions
                predictions = model.predict(new_data_scaled)

                # Convert the predictions to class labels
                predicted_classes = np.argmax(predictions, axis=1)

                # Determine the predicted outcome based on the prediction
                if predicted_classes[0] == 0:
                    outcome = 'Low'
                elif predicted_classes[0] == 1:
                    outcome = 'Medium'
                else:
                    outcome = 'High'

                # Store form data in session for reference
                confidence = float(np.max(predictions[0]) * 100)
                
                # Render the results template with the predicted outcome and confidence
                return render_template('results.html', 
                                    outcome=outcome,
                                    confidence=f"{confidence:.1f}%",
                                    age=age,
                                    gender=request.form.get('gender').capitalize())

            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                flash("An error occurred during prediction. Please try again.", "error")
                return redirect(url_for('form'))

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for('form'))

    # If the request method is not POST, redirect to the form page
    return redirect(url_for('form'))

if __name__ == '__main__':
    app.run(debug=True)

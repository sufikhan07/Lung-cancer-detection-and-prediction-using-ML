#!/usr/bin/env python
"""Complete test of the risk assessment pipeline"""

import sys
import os

# Suppress TensorFlow logging initially
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
                raise ValueError(f"Invalid value for {field_name}: must be a number, got {value}")
            if not (1 <= value <= 9):
                raise ValueError(f"{field_name} must be between 1 and 9, got {value}")
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
        print(f"ERROR in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

# Test the full pipeline
if __name__ == '__main__':
    print("=" * 70)
    print("COMPLETE RISK ASSESSMENT PIPELINE TEST")
    print("=" * 70)
    
    # Simulate form submission
    form_data = {
        'age': '22',
        'gender': 'male',
        'air_pollution': '1',
        'alcohol_use': '1',
        'dust_allergy': '1',
        'occupational_hazards': '1',
        'genetic_risk': '1',
        'chronic_lung_disease': '1',
        'balanced_diet': '1',
        'obesity': '1',
        'smoking': '1',
        'passive_smoker': '1',
        'chest_pain': '1',
        'coughing_blood': '1',
        'fatigue': '1',
        'weight_loss': '1',
        'shortness_of_breath': '1',
        'wheezing': '1',
        'swallowing_difficulty': '1',
        'clubbing': '1',
        'frequent_cold': '1',
        'dry_cough': '1',
        'snoring': '1'
    }
    
    print("\n[1/3] Testing form preprocessing...")
    processed_data = preprocess_risk_data(form_data)
    
    if processed_data is None:
        print("✗ FAILED: Could not preprocess form")
        sys.exit(1)
    
    print(f"✓ SUCCESS: Form preprocessed")
    print(f"  Shape: {processed_data.shape}")
    print(f"  Sample: {processed_data[0][:3]}")
    
    # Now test model loading and prediction
    print("\n[2/3] Loading risk assessment model...")
    try:
        import tensorflow as tf
        
        risk_model_path = os.path.join(BASE_DIR, 'risk_assessment_model.h5')
        
        if not os.path.exists(risk_model_path):
            print(f"✗ FAILED: Model file not found at {risk_model_path}")
            sys.exit(1)
        
        print(f"  Loading from: {risk_model_path}")
        print(f"  File size: {os.path.getsize(risk_model_path)} bytes")
        
        risk_model = tf.keras.models.load_model(risk_model_path, compile=False)
        print(f"✓ SUCCESS: Model loaded")
        print(f"  Model type: {type(risk_model)}")
        print(f"  Model input shape: {risk_model.input_shape}")
        print(f"  Model output shape: {risk_model.output_shape}")
        
    except Exception as e:
        print(f"✗ FAILED to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test prediction
    print("\n[3/3] Making prediction...")
    try:
        predictions = risk_model.predict(processed_data, verbose=0)
        print(f"✓ SUCCESS: Prediction made")
        print(f"  Raw output: {predictions}")
        print(f"  Output shape: {predictions.shape}")
        
        predicted_class = np.argmax(predictions[0])
        risk_levels = ['Low', 'Medium', 'High']
        result = risk_levels[predicted_class]
        confidence = float(predictions[0][predicted_class] * 100)
        
        print(f"\n  RESULT:")
        print(f"    Risk Level: {result}")
        print(f"    Confidence: {confidence:.2f}%")
        print(f"    Predicted Class Index: {predicted_class}")
        
    except Exception as e:
        print(f"✗ FAILED to make prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)

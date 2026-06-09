#!/usr/bin/env python
"""Test the risk assessment preprocessing function"""

import sys
import os
import numpy as np
import pickle

# Add the current directory to the path
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

# Test with sample data
if __name__ == '__main__':
    print("Testing risk assessment preprocessing...")
    
    test_form_data = {
        'age': '45',
        'gender': 'male',
        'air_pollution': '5',
        'alcohol_use': '3',
        'dust_allergy': '4',
        'occupational_hazards': '2',
        'genetic_risk': '6',
        'chronic_lung_disease': '3',
        'balanced_diet': '7',
        'obesity': '4',
        'smoking': '8',
        'passive_smoker': '3',
        'chest_pain': '2',
        'coughing_blood': '1',
        'fatigue': '5',
        'weight_loss': '3',
        'shortness_of_breath': '4',
        'wheezing': '2',
        'swallowing_difficulty': '1',
        'clubbing': '2',
        'frequent_cold': '5',
        'dry_cough': '6',
        'snoring': '3'
    }
    
    print("\nScaler file status:")
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        print(f"✓ Scaler file found at {scaler_path}")
        print(f"  File size: {os.path.getsize(scaler_path)} bytes")
    else:
        print(f"✗ Scaler file NOT found at {scaler_path}")
    
    print("\nProcessing test form data...")
    result = preprocess_risk_data(test_form_data)
    
    if result is not None:
        print("✓ Preprocessing successful!")
        print(f"  Output shape: {result.shape}")
        print(f"  Output sample (first 5 values): {result[0][:5]}")
    else:
        print("✗ Preprocessing failed!")
        sys.exit(1)

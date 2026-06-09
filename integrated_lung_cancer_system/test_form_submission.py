#!/usr/bin/env python
"""Test the complete risk assessment flow"""

import sys
import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def preprocess_risk_data(form_data):
    """Preprocess risk data from form submission"""
    try:
        print(f"DEBUG: Received form_data keys: {list(form_data.keys())}")
        print(f"DEBUG: Full form_data: {form_data}")
        
        # Extract basic information with validation
        age = int(form_data.get('age', 30))
        print(f"DEBUG: Age = {age}")
        
        if not (0 <= age <= 120):
            raise ValueError("Age must be between 0 and 120")

        gender = form_data.get('gender', 'male').lower()
        print(f"DEBUG: Gender raw = {form_data.get('gender')}, normalized = {gender}")
        
        if gender not in ['male', 'female']:
            raise ValueError("Invalid gender value")
        gender = 1 if gender == 'male' else 2

        # Get all numeric fields with validation
        def get_numeric_field(field_name):
            value = form_data.get(field_name)
            print(f"DEBUG: Field '{field_name}' = {value}")
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
        print(f"DEBUG: Scaling successful, output shape = {scaled_data.shape}")
        return scaled_data
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Simulate form data - THIS IS WHAT THE FORM SENDS
if __name__ == '__main__':
    print("=" * 60)
    print("Testing form submission simulation")
    print("=" * 60)
    
    # This simulates what request.form looks like
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
    
    print("\nProcessing form data...")
    result = preprocess_risk_data(form_data)
    
    if result is not None:
        print("\n✓ SUCCESS: Form processed correctly!")
        print(f"  Output shape: {result.shape}")
    else:
        print("\n✗ FAILED: Error processing form data!")
        sys.exit(1)

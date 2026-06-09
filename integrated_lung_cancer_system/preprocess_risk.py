import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def load_and_preprocess_data(data_path='cancer patient data sets.csv'):
    # Read the dataset
    df = pd.read_csv(data_path)
    
    # Extract features (excluding 'index', 'Patient Id', and 'Level')
    features = df.drop(['index', 'Patient Id', 'Level'], axis=1)
    
    # Convert gender to numeric if not already (1 for Male, 2 for Female)
    if features['Gender'].dtype == object:
        features['Gender'] = features['Gender'].map({'Male': 1, 'Female': 2})
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler on all numeric features
    scaler.fit(features)
    
    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return scaler

def preprocess_input(input_data, scaler=None):
    """
    Preprocess input data using the saved scaler
    
    Parameters:
    input_data: dict or array-like
        Input data containing features in the same order as training data
    scaler: StandardScaler, optional
        Pre-fitted scaler. If None, will load from file
    
    Returns:
    array-like: Preprocessed input data
    """
    if scaler is None:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    if isinstance(input_data, dict):
        # Convert dictionary to array maintaining correct order
        ordered_features = ['Age', 'Gender', 'Air Pollution', 'Alcohol use', 
                          'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk',
                          'chronic Lung Disease', 'Balanced Diet', 'Obesity', 
                          'Smoking', 'Passive Smoker', 'Chest Pain', 
                          'Coughing of Blood', 'Fatigue', 'Weight Loss',
                          'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
                          'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough',
                          'Snoring']
        input_array = np.array([[input_data[feature] for feature in ordered_features]])
    else:
        input_array = np.array(input_data).reshape(1, -1)
    
    # Apply scaling
    scaled_input = scaler.transform(input_array)
    return scaled_input

if __name__ == '__main__':
    # Train and save the scaler using the dataset
    scaler = load_and_preprocess_data()
    print("Scaler has been trained and saved successfully.")
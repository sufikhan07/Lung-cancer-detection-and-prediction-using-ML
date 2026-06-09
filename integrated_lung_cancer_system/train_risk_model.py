import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from preprocess_risk import load_and_preprocess_data
import pickle

def train_risk_assessment_model(data_path='cancer patient data sets.csv', epochs=50, batch_size=32):
    # Load and preprocess data
    df = pd.read_csv(data_path)
    
    # Get preprocessed features using our preprocessing module
    scaler = load_and_preprocess_data(data_path)
    
    # Extract features (excluding index, Patient Id, and Level)
    features = df.drop(['index', 'Patient Id', 'Level'], axis=1)
    
    # Scale features
    X = scaler.transform(features)
    
    # Get target variable
    y = df['Level'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_dim=X.shape[1]),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save the model
    model.save('risk_assessment_model.h5')
    print("Model saved successfully!")
    
    return history

if __name__ == '__main__':
    history = train_risk_assessment_model()
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
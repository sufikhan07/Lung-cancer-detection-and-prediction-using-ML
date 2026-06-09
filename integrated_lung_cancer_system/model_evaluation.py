import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def evaluate_model_on_test_data(model_path, test_data_dir):
    """Evaluate model performance on test dataset"""
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Compile the model for evaluation
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Get test data
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    test_data = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )

    print(f"Test dataset: {test_data.samples} images")
    print(f"Classes: {test_data.class_indices}")

    # Evaluate
    print("Evaluating model...")
    loss, accuracy = model.evaluate(test_data, verbose=1)
    print(".4f")
    print(".4f")

    # Get predictions
    predictions = model.predict(test_data, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = test_data.classes

    # Classification report
    class_names = list(test_data.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("\nConfusion Matrix:")
    print(cm)

    return accuracy, cm

def analyze_prediction_confidence(model_path, test_image_path):
    """Analyze prediction confidence for a specific image"""
    print(f"\nAnalyzing prediction for: {test_image_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    img = load_and_preprocess_image(test_image_path)

    if img is None:
        print("Could not load image")
        return

    prediction = model.predict(img)[0][0]
    confidence_non_cancerous = prediction * 100
    confidence_cancerous = (1 - prediction) * 100

    print(".2f")
    print(".2f")

    if confidence_non_cancerous >= 70:
        result = "Non-Cancerous (High Confidence)"
    elif confidence_cancerous >= 70:
        result = "Cancerous (High Confidence)"
    else:
        result = "Uncertain (Low Confidence)"

    print(f"Result: {result}")
    return prediction

if __name__ == "__main__":
    # Model and data paths
    model_path = "lung_cancer_cnn_model.keras"
    test_data_dir = "../lung-cancer-ct-scan-detection-using-cnn/Data/test"

    if os.path.exists(model_path):
        print("=== Model Evaluation ===")
        accuracy, cm = evaluate_model_on_test_data(model_path, test_data_dir)

        print("\n=== Suggestions for Improvement ===")
        if accuracy < 0.85:
            print("1. Model accuracy is below 85%. Consider:")
            print("   - Training with more data")
            print("   - Using data augmentation")
            print("   - Trying a deeper architecture (ResNet, EfficientNet)")
            print("   - Using transfer learning with pre-trained models")

        # Check for class imbalance
        if cm.shape[0] > 1:
            cancerous_correct = cm[0][0]
            cancerous_total = cm[0].sum()
            non_cancerous_correct = cm[1][1]
            non_cancerous_total = cm[1].sum()

            print("2. Class-wise performance:")
            print(".2f")
            print(".2f")

            if min(cancerous_correct/cancerous_total, non_cancerous_correct/non_cancerous_total) < 0.8:
                print("   - Consider addressing class imbalance")
                print("   - Use weighted loss function")
                print("   - Oversample minority class")

        print("3. General improvements:")
        print("   - Implement cross-validation")
        print("   - Add regularization (dropout, batch normalization)")
        print("   - Use learning rate scheduling")
        print("   - Implement early stopping")
        print("   - Add model ensemble methods")

    else:
        print(f"Model file {model_path} not found")
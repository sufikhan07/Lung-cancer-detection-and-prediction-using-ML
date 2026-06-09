import tensorflow as tf
import numpy as np
import os
import cv2

def quick_confidence_check(model_path, test_data_dir):
    """Quick check of model confidence on a few samples"""
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Get test data
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    test_data = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=10,  # Get 10 samples at once
        class_mode="binary",
        shuffle=False
    )

    print("Analyzing 10 samples...")

    # Get one batch
    batch = next(test_data)
    images, true_labels = batch

    # Make predictions
    predictions = model.predict(images, verbose=0)

    print("\nSample Predictions:")
    print("True Label -> Prediction -> Cancerous Conf -> Non-Cancerous Conf")
    print("-" * 60)

    for i in range(len(predictions)):
        pred_value = predictions[i][0]
        true_label = "Cancerous" if true_labels[i] == 0 else "Non-Cancerous"
        conf_cancerous = (1 - pred_value) * 100
        conf_non_cancerous = pred_value * 100

        print(".1f")

        # Check thresholds
        if conf_cancerous >= 60:
            result = "Cancerous"
        elif conf_non_cancerous >= 75:
            result = "Non-Cancerous"
        else:
            result = "Uncertain"

        print(f"    -> Classified as: {result}")

if __name__ == "__main__":
    model_path = "lung_cancer_cnn_model.keras"
    test_data_dir = "../lung-cancer-ct-scan-detection-using-cnn/Data/test"

    if os.path.exists(model_path):
        quick_confidence_check(model_path, test_data_dir)
    else:
        print(f"Model file {model_path} not found")
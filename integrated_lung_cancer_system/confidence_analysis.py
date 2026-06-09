import tensorflow as tf
import numpy as np
import os
import cv2
from collections import defaultdict

def analyze_model_confidence(model_path, test_data_dir, num_samples=50):
    """Analyze typical confidence levels from the model on test data"""
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Get test data
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
    test_data = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=1,  # Process one image at a time
        class_mode="binary",
        shuffle=False
    )

    confidence_stats = defaultdict(list)
    predictions = []

    print(f"Analyzing {min(num_samples, test_data.samples)} samples...")

    for i in range(min(num_samples, test_data.samples)):
        # Get one batch (one image)
        batch = next(test_data)
        image, true_label = batch

        # Make prediction
        pred_value = model.predict(image, verbose=0)[0][0]
        confidence_non_cancerous = pred_value * 100
        confidence_cancerous = (1 - pred_value) * 100

        # Store stats
        label_name = "Cancerous" if true_label[0] == 0 else "Non-Cancerous"
        confidence_stats[label_name].append({
            'non_cancerous_conf': confidence_non_cancerous,
            'cancerous_conf': confidence_cancerous,
            'prediction': pred_value,
            'true_label': int(true_label[0])
        })

        predictions.append(pred_value)

    # Analyze results
    print("\n=== Confidence Analysis ===")

    for label in ['Cancerous', 'Non-Cancerous']:
        if label in confidence_stats:
            confs = confidence_stats[label]
            print(f"\n{label} images ({len(confs)} samples):")

            if label == 'Cancerous':
                # For cancerous images, we care about cancerous confidence
                cancerous_confs = [c['cancerous_conf'] for c in confs]
                print(".1f")
                print(".1f")
                print(".1f")
                print(f"  Samples >= 60% confidence: {sum(1 for c in cancerous_confs if c >= 60)}/{len(cancerous_confs)} ({100*sum(1 for c in cancerous_confs if c >= 60)/len(cancerous_confs):.1f}%)")
                print(f"  Samples >= 70% confidence: {sum(1 for c in cancerous_confs if c >= 70)}/{len(cancerous_confs)} ({100*sum(1 for c in cancerous_confs if c >= 70)/len(cancerous_confs):.1f}%)")

            else:  # Non-Cancerous
                # For non-cancerous images, we care about non-cancerous confidence
                non_cancerous_confs = [c['non_cancerous_conf'] for c in confs]
                print(".1f")
                print(".1f")
                print(".1f")
                print(f"  Samples >= 75% confidence: {sum(1 for c in non_cancerous_confs if c >= 75)}/{len(non_cancerous_confs)} ({100*sum(1 for c in non_cancerous_confs if c >= 75)/len(non_cancerous_confs):.1f}%)")
                print(f"  Samples >= 80% confidence: {sum(1 for c in non_cancerous_confs if c >= 80)}/{len(non_cancerous_confs)} ({100*sum(1 for c in non_cancerous_confs if c >= 80)/len(non_cancerous_confs):.1f}%)")

    # Overall distribution
    all_predictions = np.array(predictions)
    print("\n=== Overall Prediction Distribution ===")
    print(".3f")
    print(".3f")
    print(".3f")

    # Check how many would be uncertain with current thresholds
    uncertain_count = 0
    for pred in predictions:
        conf_non_canc = pred * 100
        conf_canc = (1 - pred) * 100
        if not (conf_canc >= 60 or conf_non_canc >= 75):
            uncertain_count += 1

    print(f"\nWith current thresholds (60% for cancerous, 75% for non-cancerous):")
    print(f"  Uncertain predictions: {uncertain_count}/{len(predictions)} ({100*uncertain_count/len(predictions):.1f}%)")

if __name__ == "__main__":
    model_path = "lung_cancer_cnn_model.keras"
    test_data_dir = "../lung-cancer-ct-scan-detection-using-cnn/Data/test"

    if os.path.exists(model_path):
        analyze_model_confidence(model_path, test_data_dir)
    else:
        print(f"Model file {model_path} not found")
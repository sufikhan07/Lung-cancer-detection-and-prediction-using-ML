def test_prediction_logic():
    """Test the prediction logic with different confidence levels"""

    test_cases = [
        # (prediction_value, expected_result, expected_confidence_level)
        (0.95, "Non-Cancerous", "High"),  # Very confident non-cancerous
        (0.80, "Non-Cancerous", "High"),  # Confident non-cancerous
        (0.70, "Non-Cancerous", "High"),  # Above 50% threshold
        (0.30, "Cancerous", "High"),     # Below 50% threshold
        (0.20, "Cancerous", "High"),     # Very confident cancerous
        (0.50, "Non-Cancerous", "High"), # Exactly 50% (goes to non-cancerous)
        (0.49, "Cancerous", "High"),     # Just below 50%
    ]

    print("Testing Prediction Logic:")
    print("=" * 50)

    for pred_value, expected_result, expected_level in test_cases:
        # Simulate the logic from app.py
        confidence_non_cancerous = pred_value * 100
        confidence_cancerous = (1 - pred_value) * 100

        # Simple threshold: >50% for class prediction
        if pred_value > 0.5:
            result = "Non-Cancerous"
            probability = confidence_non_cancerous
        else:
            result = "Cancerous"
            probability = confidence_cancerous

if __name__ == "__main__":
    test_prediction_logic()
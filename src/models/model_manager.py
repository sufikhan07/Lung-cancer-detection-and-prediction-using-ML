"""
Model management for Lung Cancer Diagnostic System
Handles loading, validation, and inference for ML models
"""
import os
import pickle
from typing import Optional, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from config import config
from src.utils.logging_config import logger

class ModelManager:
    """Manages ML model loading and inference"""

    def __init__(self):
        self.ct_model: Optional[tf.keras.Model] = None
        self.risk_model: Optional[tf.keras.Model] = None
        self.scaler = None
        self.models_loaded = False

    def load_models(self) -> bool:
        """Load all ML models"""
        try:
            # Load CT scan model
            if os.path.exists(config.model.ct_model_path):
                self.ct_model = tf.keras.models.load_model(
                    config.model.ct_model_path,
                    compile=False
                )
                logger.info(f"Loaded CT model from {config.model.ct_model_path}")
            else:
                logger.warning(f"CT model not found at {config.model.ct_model_path}")

            # Load risk assessment model
            if os.path.exists(config.model.risk_model_path):
                self.risk_model = tf.keras.models.load_model(
                    config.model.risk_model_path,
                    compile=False
                )
                logger.info(f"Loaded risk model from {config.model.risk_model_path}")
            else:
                logger.warning(f"Risk model not found at {config.model.risk_model_path}")

            # Load scaler
            if os.path.exists(config.model.scaler_path):
                with open(config.model.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {config.model.scaler_path}")
            else:
                logger.warning(f"Scaler not found at {config.model.scaler_path}")

            self.models_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def predict_ct_scan(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Predict lung cancer from CT scan image"""
        if not self.ct_model:
            raise ValueError("CT model not loaded")

        try:
            # Ensure correct input shape
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)

            prediction = self.ct_model.predict(image_array, verbose=0)
            pred_value = float(prediction[0][0])

            # Calculate confidence levels
            confidence_non_cancerous = pred_value * 100
            confidence_cancerous = (1 - pred_value) * 100

            # Determine result
            if pred_value > 0.5:
                result = "Non-Cancerous"
                confidence = confidence_non_cancerous
            else:
                result = "Cancerous"
                confidence = confidence_cancerous

            return {
                'prediction': result,
                'confidence': confidence,
                'probabilities': {
                    'non_cancerous': confidence_non_cancerous,
                    'cancerous': confidence_cancerous
                },
                'raw_prediction': pred_value
            }

        except Exception as e:
            logger.error(f"Error in CT scan prediction: {e}")
            raise

    def predict_risk_assessment(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict lung cancer risk from patient features"""
        if not self.risk_model:
            raise ValueError("Risk model not loaded")

        if not self.scaler:
            raise ValueError("Scaler not loaded")

        try:
            # Scale features
            scaled_features = self.scaler.transform(features.reshape(1, -1))

            # Make prediction
            predictions = self.risk_model.predict(scaled_features, verbose=0)
            predicted_class = np.argmax(predictions[0])

            # Map to risk levels
            risk_levels = ['Low', 'Medium', 'High']
            result = risk_levels[predicted_class]
            confidence = float(predictions[0][predicted_class] * 100)

            # Calculate risk percentage
            if result == 'Low':
                risk_percentage = 100 - confidence
            else:
                risk_percentage = confidence

            return {
                'risk_level': result,
                'risk_percentage': risk_percentage,
                'confidence': confidence,
                'probabilities': {
                    level: float(predictions[0][i] * 100)
                    for i, level in enumerate(risk_levels)
                }
            }

        except Exception as e:
            logger.error(f"Error in risk assessment prediction: {e}")
            raise

    def validate_models(self) -> Dict[str, bool]:
        """Validate that all models are loaded and functional"""
        validation_results = {
            'ct_model_loaded': self.ct_model is not None,
            'risk_model_loaded': self.risk_model is not None,
            'scaler_loaded': self.scaler is not None,
            'models_functional': False
        }

        if all(validation_results.values()):
            try:
                # Test CT model with dummy data
                dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
                self.predict_ct_scan(dummy_image)

                # Test risk model with dummy data
                dummy_features = np.random.rand(23).astype(np.float32)
                self.predict_risk_assessment(dummy_features)

                validation_results['models_functional'] = True
            except Exception as e:
                logger.error(f"Model validation failed: {e}")

        return validation_results

# Global model manager instance
model_manager = ModelManager()</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\src\models\model_manager.py
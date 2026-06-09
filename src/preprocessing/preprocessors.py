"""
Image and data preprocessing utilities
"""
import os
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from config import config
from src.utils.logging_config import logger

class ImagePreprocessor:
    """Handles CT scan image preprocessing"""

    def __init__(self):
        self.target_size = config.model.img_size

    def preprocess_ct_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess CT scan image for model inference"""
        try:
            logger.debug(f"Processing image: {image_path}")

            # Validate file existence and size
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError("Image file is empty")

            if file_size > config.upload.max_file_size:
                raise ValueError(f"Image file too large: {file_size} bytes")

            # Load image with PIL for better format support
            with Image.open(image_path) as pil_img:
                # Convert to RGB if necessary
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                # Convert to numpy array
                img = np.array(pil_img)
                logger.debug(f"Loaded image shape: {img.shape}")

            # Validate image dimensions
            if len(img.shape) < 2:
                raise ValueError("Invalid image dimensions")

            # Basic medical image validation
            validation_result = self._validate_medical_image(img)
            if not validation_result['is_valid']:
                logger.warning(f"Image validation warning: {validation_result['message']}")

            # Resize image
            img = cv2.resize(img, self.target_size)
            logger.debug(f"Resized image shape: {img.shape}")

            # Normalize pixel values
            img = img.astype(np.float32) / 255.0

            # Ensure values are finite
            if not np.isfinite(img).all():
                raise ValueError("Image contains invalid pixel values")

            # Add batch dimension
            img = np.expand_dims(img, axis=0)

            logger.debug(f"Final preprocessed shape: {img.shape}")
            return img

        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None

    def _validate_medical_image(self, img: np.ndarray) -> Dict[str, Any]:
        """Validate if image appears to be a medical scan"""
        try:
            height, width = img.shape[:2]

            # Check aspect ratio
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 2.5:
                return {
                    'is_valid': False,
                    'message': f'Unusual aspect ratio: {aspect_ratio:.2f}'
                }

            # Check if image is mostly grayscale-like
            if len(img.shape) == 3:
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                rg_diff = np.mean(np.abs(r - g))
                rb_diff = np.mean(np.abs(r - b))
                gb_diff = np.mean(np.abs(g - b))
                channel_diff = (rg_diff + rb_diff + gb_diff) / 3

                if channel_diff > 15:
                    return {
                        'is_valid': False,
                        'message': f'Image appears colorful (channel diff: {channel_diff:.1f})'
                    }

            return {'is_valid': True, 'message': 'Image passed validation'}

        except Exception as e:
            logger.error(f"Error validating medical image: {e}")
            return {'is_valid': False, 'message': f'Validation error: {str(e)}'}

class RiskDataPreprocessor:
    """Handles risk assessment data preprocessing"""

    def __init__(self):
        self.expected_features = 23  # age + gender + 21 risk factors

    def preprocess_risk_data(self, form_data: Dict[str, str]) -> Optional[np.ndarray]:
        """Preprocess risk assessment form data"""
        try:
            logger.debug("Preprocessing risk assessment data")

            # Validate required fields
            required_fields = [
                'age', 'gender', 'air_pollution', 'alcohol_use', 'dust_allergy',
                'occupational_hazards', 'genetic_risk', 'chronic_lung_disease',
                'balanced_diet', 'obesity', 'smoking', 'passive_smoker',
                'chest_pain', 'coughing_blood', 'fatigue', 'weight_loss',
                'shortness_of_breath', 'wheezing', 'swallowing_difficulty',
                'clubbing', 'frequent_cold', 'dry_cough', 'snoring'
            ]

            missing_fields = [field for field in required_fields if field not in form_data]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

            # Initialize data array
            data = np.zeros(self.expected_features)

            # Process age
            age = self._validate_numeric_field(form_data['age'], 0, 100)
            data[0] = age

            # Process gender
            gender = form_data['gender'].lower()
            if gender not in ['male', 'female']:
                raise ValueError("Gender must be 'male' or 'female'")
            data[1] = 1 if gender == 'male' else 2

            # Process risk factors (1-9 scale)
            risk_fields = required_fields[2:]  # Skip age and gender
            for i, field in enumerate(risk_fields):
                value = self._validate_numeric_field(form_data[field], 1, 9)
                data[i + 2] = value

            logger.debug(f"Preprocessed data shape: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"Error preprocessing risk data: {e}")
            return None

    def _validate_numeric_field(self, value: str, min_val: int, max_val: int) -> float:
        """Validate and convert numeric field"""
        try:
            numeric_value = float(value)
            if not (min_val <= numeric_value <= max_val):
                raise ValueError(f"Value {numeric_value} not in range [{min_val}, {max_val}]")
            return numeric_value
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric value '{value}': {e}")

# Global preprocessor instances
image_preprocessor = ImagePreprocessor()
risk_preprocessor = RiskDataPreprocessor()</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\src\preprocessing\preprocessors.py
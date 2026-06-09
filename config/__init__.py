"""
Configuration management for Lung Cancer Diagnostic System
"""
import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    name: str = os.getenv('DB_NAME', 'lung_cancer_db')
    user: str = os.getenv('DB_USER', 'postgres')
    password: str = os.getenv('DB_PASSWORD', '')
    url: str = f"postgresql://{user}:{password}@{host}:{port}/{name}"

@dataclass
class ModelConfig:
    """ML model configuration"""
    ct_model_path: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'lung_cancer_cnn_model.keras')
    risk_model_path: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'risk_assessment_model.h5')
    scaler_path: str = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
    img_size: tuple = (224, 224)
    batch_size: int = 32

@dataclass
class APIConfig:
    """API configuration"""
    host: str = os.getenv('API_HOST', '0.0.0.0')
    port: int = int(os.getenv('API_PORT', '5000'))
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    secret_key: str = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    max_content_length: int = 16 * 1024 * 1024  # 16MB

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret_key: str = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
    jwt_algorithm: str = 'HS256'
    jwt_expiration_hours: int = 24
    bcrypt_rounds: int = 12
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path: str = os.path.join(os.path.dirname(__file__), '..', 'logs', 'app.log')
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class UploadConfig:
    """File upload configuration"""
    upload_folder: str = os.path.join(os.path.dirname(__file__), '..', 'static', 'uploads')
    allowed_extensions: set = frozenset(['png', 'jpg', 'jpeg', 'dcm'])
    max_file_size: int = 10 * 1024 * 1024  # 10MB

class Config:
    """Main configuration class"""
    def __init__(self):
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        self.upload = UploadConfig()

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'database': self.database.__dict__,
            'model': self.model.__dict__,
            'api': self.api.__dict__,
            'security': self.security.__dict__,
            'logging': self.logging.__dict__,
            'upload': self.upload.__dict__
        }

# Global configuration instance
config = Config()</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\config\__init__.py
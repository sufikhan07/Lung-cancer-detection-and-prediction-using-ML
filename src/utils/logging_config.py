"""
Logging configuration for the Lung Cancer Diagnostic System
"""
import logging
import logging.handlers
from pathlib import Path
from config import config

def setup_logging():
    """Setup application logging"""
    # Create logs directory if it doesn't exist
    log_dir = Path(config.logging.file_path).parent
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger('lung_cancer_system')
    logger.setLevel(getattr(logging, config.logging.level.upper()))

    # Create formatters
    formatter = logging.Formatter(config.logging.format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.logging.level.upper()))
    console_handler.setFormatter(formatter)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        config.logging.file_path,
        maxBytes=config.logging.max_file_size,
        backupCount=config.logging.backup_count
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Global logger instance
logger = setup_logging()</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\src\utils\logging_config.py
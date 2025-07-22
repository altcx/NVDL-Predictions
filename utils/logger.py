"""
Logging configuration for NVDL Stock Predictor
"""
import logging
import logging.handlers
import os
from typing import Optional
from config import config


class StructuredLogger:
    """Structured logging utility for different components"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """Get or create a logger for a specific component"""
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatter
        formatter = logging.Formatter(
            fmt=config.LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_filename = log_file or config.LOG_FILE
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Component-specific file handler
        if name != 'main':
            component_log = f"{name}_{config.LOG_FILE}"
            component_handler = logging.handlers.RotatingFileHandler(
                component_log,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            component_handler.setLevel(logging.DEBUG)
            component_handler.setFormatter(formatter)
            logger.addHandler(component_handler)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def setup_logging(cls):
        """Initialize logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(config.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))


# Component-specific logger functions
def get_data_logger() -> logging.Logger:
    """Get logger for data collection and preprocessing"""
    return StructuredLogger.get_logger('data')


def get_model_logger() -> logging.Logger:
    """Get logger for model training and prediction"""
    return StructuredLogger.get_logger('models')


def get_evaluation_logger() -> logging.Logger:
    """Get logger for model evaluation and metrics"""
    return StructuredLogger.get_logger('evaluation')


def get_trading_logger() -> logging.Logger:
    """Get logger for trading simulation"""
    return StructuredLogger.get_logger('trading')


def get_visualization_logger() -> logging.Logger:
    """Get logger for visualization generation"""
    return StructuredLogger.get_logger('visualization')


def get_main_logger() -> logging.Logger:
    """Get main application logger"""
    return StructuredLogger.get_logger('main')


# Initialize logging on import
StructuredLogger.setup_logging()
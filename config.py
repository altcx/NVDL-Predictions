"""
Configuration management for NVDL Stock Predictor
"""
import os
import sys
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path


# Import ConfigurationError from utils.error_handler if available
try:
    from utils.error_handler import ConfigurationError
except ImportError:
    # Define locally if not available to avoid circular imports
    class ConfigurationError(Exception):
        """Error raised when configuration is invalid"""
        pass


@dataclass
class Config:
    """Configuration class with environment variables and model parameters"""
    
    # API Configuration
    ALPACA_API_KEY: str = field(default="")
    ALPACA_SECRET_KEY: str = field(default="")
    ALPACA_BASE_URL: str = field(default="https://paper-api.alpaca.markets")
    
    # Model Parameters
    LSTM_SEQUENCE_LENGTH: int = field(default=60)
    LSTM_UNITS: int = field(default=50)
    LSTM_DROPOUT: float = field(default=0.2)
    LSTM_EPOCHS: int = field(default=100)
    LSTM_BATCH_SIZE: int = field(default=32)
    
    # ARIMA Parameters
    ARIMA_MAX_P: int = field(default=5)
    ARIMA_MAX_D: int = field(default=2)
    ARIMA_MAX_Q: int = field(default=5)
    
    # Data Parameters
    SYMBOL: str = field(default="NVDL")
    LOOKBACK_YEARS: int = field(default=2)
    TEST_SIZE: float = field(default=0.2)
    
    # Trading Parameters
    INITIAL_CAPITAL: float = field(default=10000.0)
    RISK_FREE_RATE: float = field(default=0.02)
    
    # Logging Configuration
    LOG_LEVEL: str = field(default="INFO")
    LOG_FORMAT: str = field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE: str = field(default="nvdl_predictor.log")
    
    # Error Handling Configuration
    MAX_RETRIES: int = field(default=3)
    RETRY_DELAY: float = field(default=1.0)
    RETRY_BACKOFF: float = field(default=2.0)
    
    # Error Thresholds
    MAX_API_ERRORS: int = field(default=5)
    MAX_DATA_ERRORS: int = field(default=3)
    MAX_MODEL_ERRORS: int = field(default=3)
    MAX_VIZ_ERRORS: int = field(default=5)
    MAX_CONFIG_ERRORS: int = field(default=1)
    MAX_SYSTEM_ERRORS: int = field(default=2)
    
    def __post_init__(self):
        """Load configuration from environment variables"""
        # Load from environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # API Configuration
        self.ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', self.ALPACA_API_KEY)
        self.ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', self.ALPACA_SECRET_KEY)
        self.ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', self.ALPACA_BASE_URL)
        
        # Model Parameters
        self.LSTM_SEQUENCE_LENGTH = int(os.getenv('LSTM_SEQUENCE_LENGTH', str(self.LSTM_SEQUENCE_LENGTH)))
        self.LSTM_UNITS = int(os.getenv('LSTM_UNITS', str(self.LSTM_UNITS)))
        self.LSTM_DROPOUT = float(os.getenv('LSTM_DROPOUT', str(self.LSTM_DROPOUT)))
        self.LSTM_EPOCHS = int(os.getenv('LSTM_EPOCHS', str(self.LSTM_EPOCHS)))
        self.LSTM_BATCH_SIZE = int(os.getenv('LSTM_BATCH_SIZE', str(self.LSTM_BATCH_SIZE)))
        
        # ARIMA Parameters
        self.ARIMA_MAX_P = int(os.getenv('ARIMA_MAX_P', str(self.ARIMA_MAX_P)))
        self.ARIMA_MAX_D = int(os.getenv('ARIMA_MAX_D', str(self.ARIMA_MAX_D)))
        self.ARIMA_MAX_Q = int(os.getenv('ARIMA_MAX_Q', str(self.ARIMA_MAX_Q)))
        
        # Data Parameters
        self.SYMBOL = os.getenv('SYMBOL', self.SYMBOL)
        self.LOOKBACK_YEARS = int(os.getenv('LOOKBACK_YEARS', str(self.LOOKBACK_YEARS)))
        self.TEST_SIZE = float(os.getenv('TEST_SIZE', str(self.TEST_SIZE)))
        
        # Trading Parameters
        self.INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', str(self.INITIAL_CAPITAL)))
        self.RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', str(self.RISK_FREE_RATE)))
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', self.LOG_LEVEL)
        self.LOG_FORMAT = os.getenv('LOG_FORMAT', self.LOG_FORMAT)
        self.LOG_FILE = os.getenv('LOG_FILE', self.LOG_FILE)
        
        # Error Handling Configuration
        self.MAX_RETRIES = int(os.getenv('MAX_RETRIES', str(self.MAX_RETRIES)))
        self.RETRY_DELAY = float(os.getenv('RETRY_DELAY', str(self.RETRY_DELAY)))
        self.RETRY_BACKOFF = float(os.getenv('RETRY_BACKOFF', str(self.RETRY_BACKOFF)))
        
        # Error Thresholds
        self.MAX_API_ERRORS = int(os.getenv('MAX_API_ERRORS', str(self.MAX_API_ERRORS)))
        self.MAX_DATA_ERRORS = int(os.getenv('MAX_DATA_ERRORS', str(self.MAX_DATA_ERRORS)))
        self.MAX_MODEL_ERRORS = int(os.getenv('MAX_MODEL_ERRORS', str(self.MAX_MODEL_ERRORS)))
        self.MAX_VIZ_ERRORS = int(os.getenv('MAX_VIZ_ERRORS', str(self.MAX_VIZ_ERRORS)))
        self.MAX_CONFIG_ERRORS = int(os.getenv('MAX_CONFIG_ERRORS', str(self.MAX_CONFIG_ERRORS)))
        self.MAX_SYSTEM_ERRORS = int(os.getenv('MAX_SYSTEM_ERRORS', str(self.MAX_SYSTEM_ERRORS)))
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load configuration from JSON file
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            True if loaded successfully, False otherwise
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        try:
            with open(filepath, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration with file values
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            return True
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {filepath}")
        except json.JSONDecodeError:
            raise ConfigurationError(f"Invalid JSON in configuration file: {filepath}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save configuration to JSON file
        
        Args:
            filepath: Path to save configuration
            
        Returns:
            True if saved successfully, False otherwise
            
        Raises:
            ConfigurationError: If file cannot be saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Convert to dictionary and save
            config_dict = asdict(self)
            
            # Remove sensitive information
            if 'ALPACA_API_KEY' in config_dict:
                config_dict['ALPACA_API_KEY'] = '[REDACTED]'
            if 'ALPACA_SECRET_KEY' in config_dict:
                config_dict['ALPACA_SECRET_KEY'] = '[REDACTED]'
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            return True
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {str(e)}")
    
    def validate(self) -> bool:
        """
        Validate required configuration parameters
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        errors = []
        
        # API Configuration
        if not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY:
            errors.append("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        
        # Model Parameters
        if self.LSTM_SEQUENCE_LENGTH <= 0:
            errors.append("LSTM_SEQUENCE_LENGTH must be positive")
        
        if self.LSTM_DROPOUT < 0 or self.LSTM_DROPOUT >= 1:
            errors.append("LSTM_DROPOUT must be between 0 and 1")
        
        if self.LSTM_EPOCHS <= 0:
            errors.append("LSTM_EPOCHS must be positive")
        
        if self.LSTM_BATCH_SIZE <= 0:
            errors.append("LSTM_BATCH_SIZE must be positive")
        
        # ARIMA Parameters
        if self.ARIMA_MAX_P < 0:
            errors.append("ARIMA_MAX_P must be non-negative")
        
        if self.ARIMA_MAX_D < 0:
            errors.append("ARIMA_MAX_D must be non-negative")
        
        if self.ARIMA_MAX_Q < 0:
            errors.append("ARIMA_MAX_Q must be non-negative")
        
        # Data Parameters
        if not self.SYMBOL:
            errors.append("SYMBOL must be set")
        
        if self.LOOKBACK_YEARS <= 0:
            errors.append("LOOKBACK_YEARS must be positive")
        
        if not (0 < self.TEST_SIZE < 1):
            errors.append("TEST_SIZE must be between 0 and 1")
        
        # Trading Parameters
        if self.INITIAL_CAPITAL <= 0:
            errors.append("INITIAL_CAPITAL must be positive")
        
        # Error Handling Configuration
        if self.MAX_RETRIES <= 0:
            errors.append("MAX_RETRIES must be positive")
        
        if self.RETRY_DELAY <= 0:
            errors.append("RETRY_DELAY must be positive")
        
        if self.RETRY_BACKOFF <= 1:
            errors.append("RETRY_BACKOFF must be greater than 1")
            
        # Error Thresholds
        if self.MAX_API_ERRORS <= 0:
            errors.append("MAX_API_ERRORS must be positive")
            
        if self.MAX_DATA_ERRORS <= 0:
            errors.append("MAX_DATA_ERRORS must be positive")
            
        if self.MAX_MODEL_ERRORS <= 0:
            errors.append("MAX_MODEL_ERRORS must be positive")
            
        if self.MAX_VIZ_ERRORS <= 0:
            errors.append("MAX_VIZ_ERRORS must be positive")
            
        if self.MAX_CONFIG_ERRORS <= 0:
            errors.append("MAX_CONFIG_ERRORS must be positive")
            
        if self.MAX_SYSTEM_ERRORS <= 0:
            errors.append("MAX_SYSTEM_ERRORS must be positive")
        
        if errors:
            raise ConfigurationError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True
    
    def get_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Get configuration as dictionary
        
        Args:
            include_sensitive: Whether to include sensitive information
            
        Returns:
            Dictionary with configuration values
        """
        config_dict = asdict(self)
        
        if not include_sensitive:
            # Remove sensitive information
            if 'ALPACA_API_KEY' in config_dict:
                config_dict['ALPACA_API_KEY'] = '[REDACTED]'
            if 'ALPACA_SECRET_KEY' in config_dict:
                config_dict['ALPACA_SECRET_KEY'] = '[REDACTED]'
        
        return config_dict


# Function to load configuration from .env file
def load_dotenv(env_file: str = '.env') -> bool:
    """
    Load environment variables from .env file
    
    Args:
        env_file: Path to .env file
        
    Returns:
        True if loaded successfully, False otherwise
    """
    try:
        if not os.path.exists(env_file):
            return False
        
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
        
        return True
    except Exception:
        return False


# Try to load from .env file
load_dotenv()

# Global configuration instance
config = Config()

# Try to validate configuration
try:
    config.validate()
except ConfigurationError as e:
    # Log error if logging is available
    try:
        from utils.logger import get_config_logger
        logger = get_config_logger()
        logger.error(f"Configuration error: {str(e)}")
    except ImportError:
        # Fall back to basic logging if logger is not available
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Configuration error: {str(e)}")
    
    # Don't raise here to allow the application to start with default values
    # The application code should check configuration validity before using it
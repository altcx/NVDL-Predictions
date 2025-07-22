"""
Configuration management for NVDL Stock Predictor
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class with environment variables and model parameters"""
    
    # API Configuration
    ALPACA_API_KEY: str = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY: str = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL: str = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Model Parameters
    LSTM_SEQUENCE_LENGTH: int = int(os.getenv('LSTM_SEQUENCE_LENGTH', '60'))
    LSTM_UNITS: int = int(os.getenv('LSTM_UNITS', '50'))
    LSTM_DROPOUT: float = float(os.getenv('LSTM_DROPOUT', '0.2'))
    LSTM_EPOCHS: int = int(os.getenv('LSTM_EPOCHS', '100'))
    LSTM_BATCH_SIZE: int = int(os.getenv('LSTM_BATCH_SIZE', '32'))
    
    # ARIMA Parameters
    ARIMA_MAX_P: int = int(os.getenv('ARIMA_MAX_P', '5'))
    ARIMA_MAX_D: int = int(os.getenv('ARIMA_MAX_D', '2'))
    ARIMA_MAX_Q: int = int(os.getenv('ARIMA_MAX_Q', '5'))
    
    # Data Parameters
    SYMBOL: str = os.getenv('SYMBOL', 'NVDL')
    LOOKBACK_YEARS: int = int(os.getenv('LOOKBACK_YEARS', '2'))
    TEST_SIZE: float = float(os.getenv('TEST_SIZE', '0.2'))
    
    # Trading Parameters
    INITIAL_CAPITAL: float = float(os.getenv('INITIAL_CAPITAL', '10000'))
    RISK_FREE_RATE: float = float(os.getenv('RISK_FREE_RATE', '0.02'))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE: str = os.getenv('LOG_FILE', 'nvdl_predictor.log')
    
    def validate(self) -> bool:
        """Validate required configuration parameters"""
        if not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        
        if self.LSTM_SEQUENCE_LENGTH <= 0:
            raise ValueError("LSTM_SEQUENCE_LENGTH must be positive")
        
        if self.LOOKBACK_YEARS <= 0:
            raise ValueError("LOOKBACK_YEARS must be positive")
        
        if not (0 < self.TEST_SIZE < 1):
            raise ValueError("TEST_SIZE must be between 0 and 1")
        
        return True


# Global configuration instance
config = Config()
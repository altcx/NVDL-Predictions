"""
Models module for NVDL Stock Predictor
"""
import warnings

# Import model evaluator and trading simulator (no external dependencies)
from models.model_evaluator import ModelEvaluator
from models.trading_simulator import TradingSimulator

# Conditionally import model classes based on available dependencies
__all__ = ['ModelEvaluator', 'TradingSimulator']

# Try to import LSTM predictor
try:
    from models.lstm_predictor import LSTMPredictor
    __all__.append('LSTMPredictor')
except ImportError:
    warnings.warn("TensorFlow not available, LSTMPredictor will not be imported")

# Try to import ARIMA predictor
try:
    from models.arima_predictor import ARIMAPredictor
    __all__.append('ARIMAPredictor')
except ImportError:
    warnings.warn("Statsmodels not available, ARIMAPredictor will not be imported")
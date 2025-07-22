"""
Integration tests for NVDL Stock Predictor pipeline
Tests end-to-end execution of the complete pipeline
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import warnings

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check for required packages
TENSORFLOW_AVAILABLE = False
STATSMODELS_AVAILABLE = False
PLOTLY_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    warnings.warn("TensorFlow not available, some tests will be skipped")

try:
    import statsmodels
    STATSMODELS_AVAILABLE = True
except ImportError:
    warnings.warn("Statsmodels not available, some tests will be skipped")

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    warnings.warn("Plotly not available, some tests will be skipped")

# Import our modules
from data.collector import DataCollector
from data.preprocessor import DataPreprocessor
from models.model_evaluator import ModelEvaluator
from models.trading_simulator import TradingSimulator
from visualization.visualization_engine import VisualizationEngine

# Conditionally import model classes
if TENSORFLOW_AVAILABLE:
    from models.lstm_predictor import LSTMPredictor
if STATSMODELS_AVAILABLE:
    from models.arima_predictor import ARIMAPredictor

# Import main pipeline
from main import NVDLPredictorPipeline


class TestIntegrationPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests"""
        # Suppress warnings during tests
        warnings.filterwarnings('ignore')
        
        # Create sample data for testing
        cls.sample_data = cls._create_sample_data()
    
    @staticmethod
    def _create_sample_data():
        """Create synthetic data for testing"""
        # Create date range
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create price data with some trend and noise
        np.random.seed(42)
        n = len(dates)
        
        # Base trend
        trend = np.linspace(100, 150, n)
        
        # Add seasonality
        seasonality = 10 * np.sin(np.linspace(0, 12 * np.pi, n))
        
        # Add noise
        noise = np.random.normal(0, 5, n)
        
        # Combine components
        close = trend + seasonality + noise
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': close - np.random.uniform(0, 2, n),
            'high': close + np.random.uniform(1, 3, n),
            'low': close - np.random.uniform(1, 3, n),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        
        return data
    
    def test_data_collection_and_preprocessing(self):
        """Test data collection and preprocessing steps"""
        # Create pipeline instance
        pipeline = NVDLPredictorPipeline()
        
        # Skip data collection and directly test preprocessing
        pipeline.data_preprocessor = DataPreprocessor()
        
        # Test preprocessing
        processed_data = pipeline.preprocess_data(self.sample_data)
        
        # Verify processed data structure
        self.assertIn('X_train', processed_data)
        self.assertIn('y_train', processed_data)
        self.assertIn('X_test', processed_data)
        self.assertIn('y_test', processed_data)
        self.assertIn('metadata', processed_data)
        self.assertIn('train_df', processed_data)
        self.assertIn('test_df', processed_data)
        
        # Verify data shapes are reasonable
        # The difference is due to NaN values being dropped during feature creation
        self.assertGreater(len(processed_data['X_train']) + len(processed_data['X_test']), 
                          len(self.sample_data) * 0.8)  # At least 80% of data should be preserved
    
    @unittest.skipIf(not TENSORFLOW_AVAILABLE or not STATSMODELS_AVAILABLE, 
                 "TensorFlow or Statsmodels not available")
    def test_model_training_and_evaluation(self):
        """Test model training and evaluation steps"""
        # Create pipeline instance
        pipeline = NVDLPredictorPipeline()
        
        # Preprocess data
        processed_data = pipeline.preprocess_data(self.sample_data)
        
        # Test LSTM model training with reduced parameters for testing
        # Monkey patch config values for faster testing
        import config
        original_sequence_length = config.config.LSTM_SEQUENCE_LENGTH
        original_lstm_epochs = config.config.LSTM_EPOCHS
        
        config.config.LSTM_SEQUENCE_LENGTH = 10
        config.config.LSTM_EPOCHS = 2
        
        try:
            lstm_results = pipeline.train_lstm_model(processed_data)
            
            # Verify LSTM results structure
            self.assertIn('model', lstm_results)
            self.assertIn('predictions', lstm_results)
            self.assertIn('y_true', lstm_results)
            
            # Test ARIMA model training
            arima_results = pipeline.train_arima_model(processed_data)
            
            # Verify ARIMA results structure
            self.assertIn('model', arima_results)
            self.assertIn('predictions', arima_results)
            self.assertIn('y_true', arima_results)
            
            # Test model evaluation
            evaluation_results = pipeline.evaluate_models(lstm_results, arima_results)
            
            # Verify evaluation results structure
            self.assertIn('LSTM', evaluation_results)
            self.assertIn('ARIMA', evaluation_results)
            self.assertIn('comparison', evaluation_results)
            
            # Check metrics
            self.assertIn('accuracy', evaluation_results['LSTM'])
            self.assertIn('precision', evaluation_results['LSTM'])
            self.assertIn('recall', evaluation_results['LSTM'])
            self.assertIn('f1_score', evaluation_results['LSTM'])
            
        finally:
            # Restore original config values
            config.config.LSTM_SEQUENCE_LENGTH = original_sequence_length
            config.config.LSTM_EPOCHS = original_lstm_epochs
    
    @unittest.skipIf(not PLOTLY_AVAILABLE, "Plotly not available")
    def test_trading_simulation_and_visualization(self):
        """Test trading simulation and visualization steps"""
        # Create pipeline instance
        pipeline = NVDLPredictorPipeline()
        
        # Preprocess data
        processed_data = pipeline.preprocess_data(self.sample_data)
        
        # Extract test prices
        test_df = processed_data['test_df']
        test_prices = test_df['close']
        test_volume = test_df['volume']
        
        # Create mock model results
        lstm_results = {
            'test_dates': test_df.index,
            'predictions': np.random.randint(0, 2, len(test_df)),
            'probabilities': np.random.random(len(test_df)),
            'y_true': test_df['target'].values
        }
        
        arima_results = {
            'test_dates': test_df.index,
            'predictions': np.random.randint(0, 2, len(test_df)),
            'y_true': test_df['target'].values
        }
        
        # Test trading simulation
        trading_results = pipeline.simulate_trading(lstm_results, arima_results, test_prices)
        
        # Verify trading results structure
        self.assertIn('LSTM', trading_results)
        self.assertIn('ARIMA', trading_results)
        self.assertIn('equity_curve', trading_results['LSTM'])
        self.assertIn('total_return', trading_results['LSTM'])
        self.assertIn('sharpe_ratio', trading_results['LSTM'])
        self.assertIn('max_drawdown', trading_results['LSTM'])
        self.assertIn('win_rate', trading_results['LSTM'])
        
        # Create mock evaluation results
        evaluation_results = {
            'LSTM': {
                'accuracy': 0.6,
                'precision': 0.65,
                'recall': 0.7,
                'f1_score': 0.67,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.15,
                'win_rate': 0.55
            },
            'ARIMA': {
                'accuracy': 0.58,
                'precision': 0.62,
                'recall': 0.68,
                'f1_score': 0.65,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.18,
                'win_rate': 0.52
            }
        }
        
        # Test visualization generation
        visualization_results = pipeline.generate_visualizations(
            lstm_results,
            arima_results,
            evaluation_results,
            trading_results,
            test_prices,
            test_volume
        )
        
        # Verify visualization results structure
        self.assertIn('lstm_price_chart', visualization_results)
        self.assertIn('arima_price_chart', visualization_results)
        self.assertIn('equity_comparison', visualization_results)
        self.assertIn('performance_comparison', visualization_results)
        self.assertIn('dashboard', visualization_results)


if __name__ == '__main__':
    unittest.main()
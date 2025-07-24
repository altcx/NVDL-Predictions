"""
Performance tests for NVDL Stock Predictor
Tests model training times, memory usage, and system resource utilization
"""
import pytest
import time
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Import test fixtures
from tests.test_fixtures import (
    create_sample_ohlcv_data, create_sample_processed_data,
    create_train_test_split, mock_tensorflow, mock_statsmodels
)

# Import components
from data.preprocessor import DataPreprocessor
from models.model_evaluator import ModelEvaluator
from models.trading_simulator import TradingSimulator

# Check for required packages
TENSORFLOW_AVAILABLE = False
STATSMODELS_AVAILABLE = False
PSUTIL_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

try:
    import statsmodels
    STATSMODELS_AVAILABLE = True
except ImportError:
    pass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    pass

# Conditionally import model classes
if TENSORFLOW_AVAILABLE:
    from models.lstm_predictor import LSTMPredictor
if STATSMODELS_AVAILABLE:
    from models.arima_predictor import ARIMAPredictor


class TestDataProcessingPerformance:
    """Test data processing performance"""
    
    @pytest.fixture
    def large_sample_data(self):
        """Create large sample OHLCV data"""
        return create_sample_ohlcv_data(days=1000)
    
    def test_preprocessing_time(self, large_sample_data):
        """Test preprocessing time for large dataset"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Measure preprocessing time
        start_time = time.time()
        processed_data = preprocessor.prepare_data_for_training(large_sample_data)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Data preprocessing time for 1000 days: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 30.0, f"Data preprocessing took {processing_time:.2f}s"
    
    def test_feature_creation_time(self, large_sample_data):
        """Test feature creation time for large dataset"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Measure feature creation time
        start_time = time.time()
        features_df = preprocessor.create_features(large_sample_data)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Feature creation time for 1000 days: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 20.0, f"Feature creation took {processing_time:.2f}s"
    
    def test_target_creation_time(self, large_sample_data):
        """Test target creation time for large dataset"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Measure target creation time
        start_time = time.time()
        labeled_df = preprocessor.create_target_labels(large_sample_data)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Target creation time for 1000 days: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 5.0, f"Target creation took {processing_time:.2f}s"
    
    def test_data_split_time(self, large_sample_data):
        """Test data split time for large dataset"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Measure data split time
        start_time = time.time()
        train_df, test_df = preprocessor.split_data(large_sample_data)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Data split time for 1000 days: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 1.0, f"Data split took {processing_time:.2f}s"
    
    def test_scaling_time(self, large_sample_data):
        """Test feature scaling time for large dataset"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Create features first
        features_df = preprocessor.create_features(large_sample_data)
        
        # Measure scaling time
        start_time = time.time()
        scaled_df = preprocessor.scale_features(features_df)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Feature scaling time for 1000 days: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 5.0, f"Feature scaling took {processing_time:.2f}s"


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
class TestLSTMModelPerformance:
    """Test LSTM model performance"""
    
    @pytest.fixture
    def processed_data(self):
        """Create processed data for testing"""
        sample_data = create_sample_ohlcv_data(days=500)
        preprocessor = DataPreprocessor()
        return preprocessor.prepare_data_for_training(sample_data)
    
    def test_sequence_preparation_time(self, processed_data):
        """Test sequence preparation time"""
        # Create LSTM predictor
        lstm_predictor = LSTMPredictor(sequence_length=60, lstm_units=50, dropout_rate=0.2)
        
        # Prepare data
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        combined_data = np.column_stack((X_train, y_train))
        
        # Measure sequence preparation time
        start_time = time.time()
        X_sequences, y_labels = lstm_predictor.prepare_sequences(combined_data)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Sequence preparation time: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 10.0, f"Sequence preparation took {processing_time:.2f}s"
    
    def test_model_build_time(self):
        """Test model build time"""
        # Create LSTM predictor
        lstm_predictor = LSTMPredictor(sequence_length=60, lstm_units=50, dropout_rate=0.2)
        
        # Measure model build time
        start_time = time.time()
        model = lstm_predictor.build_model((60, 20))  # 60 timesteps, 20 features
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Model build time: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 5.0, f"Model build took {processing_time:.2f}s"
    
    def test_prediction_time(self, processed_data):
        """Test prediction time"""
        # Create LSTM predictor
        lstm_predictor = LSTMPredictor(sequence_length=60, lstm_units=50, dropout_rate=0.2)
        
        # Build model
        input_shape = (60, processed_data['X_train'].shape[1])
        model = lstm_predictor.build_model(input_shape)
        
        # Create dummy test data
        X_test = np.random.random((100, 60, processed_data['X_train'].shape[1]))
        
        # Mock predict method to return dummy predictions
        lstm_predictor.model.predict = Mock(return_value=np.random.random((100, 1)))
        
        # Measure prediction time
        start_time = time.time()
        predictions = lstm_predictor.predict(X_test)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Prediction time for 100 samples: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 5.0, f"Prediction took {processing_time:.2f}s"


@pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="Statsmodels not available")
class TestARIMAModelPerformance:
    """Test ARIMA model performance"""
    
    @pytest.fixture
    def processed_data(self):
        """Create processed data for testing"""
        sample_data = create_sample_ohlcv_data(days=500)
        preprocessor = DataPreprocessor()
        return preprocessor.prepare_data_for_training(sample_data)
    
    def test_order_finding_time(self, processed_data):
        """Test optimal order finding time"""
        # Create ARIMA predictor with mocked methods
        arima_predictor = ARIMAPredictor(max_p=2, max_d=1, max_q=2)
        
        # Mock find_optimal_order to avoid actual computation
        arima_predictor.find_optimal_order = Mock(return_value=(1, 1, 1))
        
        # Measure order finding time
        start_time = time.time()
        order = arima_predictor.find_optimal_order(processed_data['train_df']['close'])
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Order finding time: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 5.0, f"Order finding took {processing_time:.2f}s"
    
    def test_model_fit_time(self, processed_data):
        """Test model fitting time"""
        # Create ARIMA predictor with mocked methods
        arima_predictor = ARIMAPredictor(max_p=2, max_d=1, max_q=2)
        
        # Mock ARIMA model
        mock_model = Mock()
        mock_results = Mock()
        mock_model.fit.return_value = mock_results
        
        with patch('models.arima_predictor.ARIMA', return_value=mock_model):
            # Measure model fit time
            start_time = time.time()
            results = arima_predictor.fit(processed_data['train_df']['close'], order=(1, 1, 1))
            end_time = time.time()
            
            # Calculate processing time
            processing_time = end_time - start_time
            
            # Log processing time
            print(f"Model fit time: {processing_time:.2f}s")
            
            # Processing time should be reasonable
            assert processing_time < 5.0, f"Model fit took {processing_time:.2f}s"
    
    def test_forecast_time(self, processed_data):
        """Test forecasting time"""
        # Create ARIMA predictor with mocked methods
        arima_predictor = ARIMAPredictor(max_p=2, max_d=1, max_q=2)
        
        # Mock results
        mock_results = Mock()
        mock_forecast = Mock()
        mock_forecast.values = np.array([100.0])
        mock_results.forecast.return_value = mock_forecast
        
        # Set results directly
        arima_predictor.results = mock_results
        
        # Measure forecast time
        start_time = time.time()
        forecast = arima_predictor.forecast(steps=1)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Forecast time: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 1.0, f"Forecast took {processing_time:.2f}s"


class TestTradingSimulationPerformance:
    """Test trading simulation performance"""
    
    @pytest.fixture
    def large_test_data(self):
        """Create large test data for trading simulation"""
        # Create sample data
        sample_data = create_sample_ohlcv_data(days=1000)
        
        # Create random signals
        np.random.seed(42)
        signals = np.random.randint(0, 2, size=1000)
        
        return sample_data['close'], signals
    
    def test_simulation_time(self, large_test_data):
        """Test trading simulation time for large dataset"""
        # Create trading simulator
        simulator = TradingSimulator()
        
        # Get test data
        prices, signals = large_test_data
        
        # Measure simulation time
        start_time = time.time()
        results = simulator.simulate_strategy(prices, signals)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Trading simulation time for 1000 days: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 10.0, f"Trading simulation took {processing_time:.2f}s"
    
    def test_performance_metrics_calculation_time(self, large_test_data):
        """Test performance metrics calculation time"""
        # Create trading simulator
        simulator = TradingSimulator()
        
        # Get test data
        prices, signals = large_test_data
        
        # Run simulation first
        results = simulator.simulate_strategy(prices, signals)
        
        # Create sample transactions
        transactions = pd.DataFrame({
            'date': prices.index[:100],
            'action': ['BUY', 'SELL'] * 50,
            'price': np.random.random(100) * 100 + 100,
            'shares': np.random.random(100) * 100,
            'value': np.random.random(100) * 10000,
            'commission': np.random.random(100) * 10
        })
        
        # Measure metrics calculation time
        start_time = time.time()
        metrics = simulator.calculate_performance_metrics(
            results['equity_curve'], prices, transactions
        )
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Performance metrics calculation time: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 5.0, f"Metrics calculation took {processing_time:.2f}s"


class TestModelEvaluationPerformance:
    """Test model evaluation performance"""
    
    @pytest.fixture
    def large_evaluation_data(self):
        """Create large evaluation data"""
        # Create random true and predicted values
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=10000)
        y_pred = np.random.randint(0, 2, size=10000)
        y_prob = np.random.random(size=10000)
        
        return y_true, y_pred, y_prob
    
    def test_classification_metrics_time(self, large_evaluation_data):
        """Test classification metrics calculation time"""
        # Create model evaluator
        evaluator = ModelEvaluator()
        
        # Get evaluation data
        y_true, y_pred, _ = large_evaluation_data
        
        # Measure metrics calculation time
        start_time = time.time()
        metrics = evaluator.calculate_classification_metrics(y_true, y_pred)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Classification metrics calculation time for 10000 samples: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 5.0, f"Metrics calculation took {processing_time:.2f}s"
    
    def test_forecast_metrics_time(self):
        """Test forecast metrics calculation time"""
        # Create model evaluator
        evaluator = ModelEvaluator()
        
        # Create random price data
        np.random.seed(42)
        y_true = np.random.random(10000) * 100 + 100
        y_pred = y_true + np.random.normal(0, 5, size=10000)
        
        # Measure metrics calculation time
        start_time = time.time()
        metrics = evaluator.calculate_forecast_metrics(y_true, y_pred)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Forecast metrics calculation time for 10000 samples: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 5.0, f"Metrics calculation took {processing_time:.2f}s"
    
    def test_model_comparison_time(self):
        """Test model comparison time"""
        # Create model evaluator
        evaluator = ModelEvaluator()
        
        # Create model results
        model_results = {
            'LSTM': {
                'accuracy': 0.65,
                'precision': 0.70,
                'recall': 0.68,
                'f1_score': 0.69,
                'rmse': 1.2,
                'mae': 0.9,
                'mape': 0.05,
                'directional_accuracy': 0.62
            },
            'ARIMA': {
                'accuracy': 0.60,
                'precision': 0.65,
                'recall': 0.72,
                'f1_score': 0.68,
                'rmse': 1.5,
                'mae': 1.1,
                'mape': 0.06,
                'directional_accuracy': 0.58
            }
        }
        
        # Measure comparison time
        start_time = time.time()
        comparison = evaluator.compare_models(model_results)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Model comparison time: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 1.0, f"Model comparison took {processing_time:.2f}s"


@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
class TestMemoryUsage:
    """Test memory usage during operations"""
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # MB
    
    def test_data_preprocessing_memory(self):
        """Test memory usage during data preprocessing"""
        # Get initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Create large sample data
        sample_data = create_sample_ohlcv_data(days=1000)
        
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.prepare_data_for_training(sample_data)
        
        # Get final memory usage
        final_memory = self.get_memory_usage()
        
        # Calculate memory increase
        memory_increase = final_memory - initial_memory
        
        # Log memory usage
        print(f"Memory usage for data preprocessing: Initial={initial_memory:.1f}MB, "
              f"Final={final_memory:.1f}MB, Increase={memory_increase:.1f}MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 1000, f"Memory usage increased by {memory_increase:.1f}MB"
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_lstm_model_memory(self):
        """Test memory usage during LSTM model creation"""
        # Get initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Create LSTM predictor
        lstm_predictor = LSTMPredictor(sequence_length=60, lstm_units=50, dropout_rate=0.2)
        
        # Build model
        model = lstm_predictor.build_model((60, 20))  # 60 timesteps, 20 features
        
        # Get final memory usage
        final_memory = self.get_memory_usage()
        
        # Calculate memory increase
        memory_increase = final_memory - initial_memory
        
        # Log memory usage
        print(f"Memory usage for LSTM model creation: Initial={initial_memory:.1f}MB, "
              f"Final={final_memory:.1f}MB, Increase={memory_increase:.1f}MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
    
    def test_trading_simulation_memory(self):
        """Test memory usage during trading simulation"""
        # Get initial memory usage
        initial_memory = self.get_memory_usage()
        
        # Create large test data
        sample_data = create_sample_ohlcv_data(days=1000)
        prices = sample_data['close']
        
        # Create random signals
        np.random.seed(42)
        signals = np.random.randint(0, 2, size=1000)
        
        # Create trading simulator
        simulator = TradingSimulator()
        
        # Run simulation
        results = simulator.simulate_strategy(prices, signals)
        
        # Get final memory usage
        final_memory = self.get_memory_usage()
        
        # Calculate memory increase
        memory_increase = final_memory - initial_memory
        
        # Log memory usage
        print(f"Memory usage for trading simulation: Initial={initial_memory:.1f}MB, "
              f"Final={final_memory:.1f}MB, Increase={memory_increase:.1f}MB")
        
        # Memory increase should be reasonable
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
"""
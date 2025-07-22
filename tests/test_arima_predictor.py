"""
Unit tests for ARIMA predictor model
"""
import os
import sys
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Mock statsmodels before importing ARIMAPredictor
sys.modules['statsmodels'] = MagicMock()
sys.modules['statsmodels.tsa'] = MagicMock()
sys.modules['statsmodels.tsa.arima'] = MagicMock()
sys.modules['statsmodels.tsa.arima.model'] = MagicMock()
sys.modules['statsmodels.tools'] = MagicMock()
sys.modules['statsmodels.tools.sm_exceptions'] = MagicMock()

# Now import the module that uses statsmodels
from models.arima_predictor import ARIMAPredictor


class TestARIMAPredictor:
    """Test suite for ARIMA predictor model"""
    
    @pytest.fixture
    def arima_predictor(self):
        """Create ARIMA predictor instance for testing"""
        return ARIMAPredictor(max_p=2, max_d=1, max_q=2)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing"""
        # Generate synthetic time series: 100 samples with trend and noise
        np.random.seed(42)
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 1, 100)
        data = trend + noise
        return pd.Series(data)
    
    def test_init(self):
        """Test initialization with custom parameters"""
        predictor = ARIMAPredictor(max_p=3, max_d=2, max_q=3)
        assert predictor.max_p == 3
        assert predictor.max_d == 2
        assert predictor.max_q == 3
        assert predictor.model is None
        assert predictor.results is None
        assert predictor.order is None
    
    @patch('models.arima_predictor.ARIMA')
    def test_find_optimal_order(self, mock_arima, arima_predictor, sample_data):
        """Test optimal order finding using AIC/BIC criteria"""
        # Setup mock results
        mock_result1 = MagicMock()
        mock_result1.aic = 200
        
        mock_result2 = MagicMock()
        mock_result2.aic = 150  # Better AIC
        
        # Setup mock model instances
        mock_model1 = MagicMock()
        mock_model1.fit.return_value = mock_result1
        
        mock_model2 = MagicMock()
        mock_model2.fit.return_value = mock_result2
        
        # Configure ARIMA mock to return different models based on parameters
        def mock_arima_init(data, order):
            if order == (1, 0, 0):
                return mock_model1
            else:
                return mock_model2
        
        mock_arima.side_effect = mock_arima_init
        
        # Call the method
        order = arima_predictor.find_optimal_order(sample_data)
        
        # Check that optimal order was found
        assert isinstance(order, tuple)
        assert len(order) == 3
        assert arima_predictor.order == order
    
    def test_find_optimal_order_insufficient_data(self, arima_predictor):
        """Test optimal order finding with insufficient data"""
        # Create small dataset
        small_data = pd.Series(np.random.randn(5))
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            arima_predictor.find_optimal_order(small_data)
    
    @patch('models.arima_predictor.ARIMA')
    def test_fit(self, mock_arima, arima_predictor, sample_data):
        """Test model fitting with specified order"""
        # Setup mock model and results
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_model.fit.return_value = mock_results
        mock_arima.return_value = mock_model
        
        # Call fit with specific order
        results = arima_predictor.fit(sample_data, order=(1, 1, 1))
        
        # Check that ARIMA was called with correct parameters
        mock_arima.assert_called_once()
        args, kwargs = mock_arima.call_args
        assert kwargs['order'] == (1, 1, 1)
        
        # Check that fit was called
        mock_model.fit.assert_called_once()
        
        # Check that results were stored
        assert arima_predictor.model == mock_model
        assert arima_predictor.results == mock_results
        assert arima_predictor.order == (1, 1, 1)
        assert results == mock_results
    
    @patch('models.arima_predictor.ARIMA')
    def test_fit_with_auto_order(self, mock_arima, arima_predictor, sample_data):
        """Test model fitting with automatic order selection"""
        # Setup mock for find_optimal_order
        arima_predictor.find_optimal_order = MagicMock(return_value=(2, 1, 1))
        
        # Setup mock model and results
        mock_model = MagicMock()
        mock_results = MagicMock()
        mock_model.fit.return_value = mock_results
        mock_arima.return_value = mock_model
        
        # Call fit without specifying order
        results = arima_predictor.fit(sample_data)
        
        # Check that find_optimal_order was called
        arima_predictor.find_optimal_order.assert_called_once_with(sample_data)
        
        # Check that ARIMA was called with correct parameters
        mock_arima.assert_called_once()
        args, kwargs = mock_arima.call_args
        assert kwargs['order'] == (2, 1, 1)
        
        # Check that results were stored
        assert arima_predictor.results == mock_results
        assert arima_predictor.order == (2, 1, 1)
    
    @patch('models.arima_predictor.ARIMA')
    def test_fit_error_handling(self, mock_arima, arima_predictor, sample_data):
        """Test error handling during model fitting"""
        # Setup mock to raise exception on first call, succeed on second
        mock_model1 = MagicMock()
        mock_model1.fit.side_effect = Exception("Convergence error")
        
        mock_model2 = MagicMock()
        mock_results = MagicMock()
        mock_model2.fit.return_value = mock_results
        
        # Configure ARIMA mock to return different models based on parameters
        def mock_arima_init(data, order):
            if order == (2, 1, 1):
                return mock_model1
            else:
                return mock_model2
        
        mock_arima.side_effect = mock_arima_init
        
        # Set order to trigger error
        arima_predictor.order = (2, 1, 1)
        
        # Call fit - should fall back to default parameters
        results = arima_predictor.fit(sample_data)
        
        # Check that fallback model was used
        assert arima_predictor.order == (1, 1, 0)
        assert arima_predictor.results == mock_results
    
    def test_forecast_without_fitting(self, arima_predictor):
        """Test forecasting without fitting"""
        # Should raise ValueError
        with pytest.raises(ValueError):
            arima_predictor.forecast()
    
    def test_forecast(self, arima_predictor):
        """Test forecasting with fitted model"""
        # Create mock results
        mock_results = MagicMock()
        mock_forecast = MagicMock()
        mock_forecast.values = np.array([105.5])
        mock_results.forecast.return_value = mock_forecast
        
        # Set results directly
        arima_predictor.results = mock_results
        
        # Call forecast
        forecast = arima_predictor.forecast(steps=1)
        
        # Check forecast
        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 1
        assert forecast[0] == 105.5
        
        # Check that forecast was called with correct parameters
        mock_results.forecast.assert_called_once_with(steps=1)
    
    def test_predict_direction_without_fitting(self, arima_predictor):
        """Test direction prediction without fitting"""
        # Should raise ValueError
        with pytest.raises(ValueError):
            arima_predictor.predict_direction(100.0)
    
    def test_predict_direction(self, arima_predictor):
        """Test direction prediction with fitted model"""
        # Create mock results
        mock_results = MagicMock()
        mock_forecast = MagicMock()
        mock_forecast.values = np.array([105.5])  # Higher than current price
        mock_results.forecast.return_value = mock_forecast
        
        # Set results and last price directly
        arima_predictor.results = mock_results
        arima_predictor.last_price = 100.0
        
        # Call predict_direction with current price
        direction = arima_predictor.predict_direction(100.0)
        
        # Check direction (should be 1 for up)
        assert direction == 1
        
        # Test with higher current price
        direction = arima_predictor.predict_direction(110.0)
        
        # Check direction (should be 0 for down)
        assert direction == 0
        
        # Test without providing current price
        direction = arima_predictor.predict_direction()
        
        # Check direction (should use last_price)
        assert direction == 1
    
    @patch('models.arima_predictor.pickle.dump')
    def test_save_model(self, mock_dump, arima_predictor, tmp_path):
        """Test saving model to file"""
        # Create mock results
        mock_results = MagicMock()
        
        # Set model attributes
        arima_predictor.results = mock_results
        arima_predictor.order = (1, 1, 1)
        arima_predictor.last_price = 100.0
        
        # Save model
        model_path = os.path.join(tmp_path, "arima_model.pkl")
        arima_predictor.save_model(model_path)
        
        # Check that pickle.dump was called
        mock_dump.assert_called_once()
        
        # Check that model data was correctly prepared
        args, kwargs = mock_dump.call_args
        model_data = args[0]
        assert model_data['order'] == (1, 1, 1)
        assert model_data['last_price'] == 100.0
        assert model_data['results'] == mock_results
    
    def test_save_model_without_fitting(self, arima_predictor, tmp_path):
        """Test saving model without fitting"""
        # Should raise ValueError
        with pytest.raises(ValueError):
            arima_predictor.save_model(os.path.join(tmp_path, "model.pkl"))
    
    @patch('models.arima_predictor.os.path.exists')
    @patch('models.arima_predictor.open', create=True)
    @patch('models.arima_predictor.pickle.load')
    def test_load_model(self, mock_load, mock_open, mock_exists, arima_predictor, tmp_path):
        """Test loading model from file"""
        # Setup mock for os.path.exists
        mock_exists.return_value = True
        
        # Setup mock for pickle.load
        mock_results = MagicMock()
        mock_load.return_value = {
            'order': (1, 1, 1),
            'last_price': 100.0,
            'results': mock_results
        }
        
        # Setup mock for open
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Load model
        model_path = os.path.join(tmp_path, "arima_model.pkl")
        arima_predictor.load_model(model_path)
        
        # Check that model attributes were set correctly
        assert arima_predictor.order == (1, 1, 1)
        assert arima_predictor.last_price == 100.0
        assert arima_predictor.results == mock_results
    
    def test_load_model_file_not_found(self, arima_predictor, tmp_path):
        """Test loading model with non-existent file"""
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            arima_predictor.load_model(os.path.join(tmp_path, "nonexistent.pkl"))
    
    def test_evaluate(self, arima_predictor):
        """Test model evaluation on test data"""
        # Create mock results
        mock_results = MagicMock()
        mock_forecast = MagicMock()
        mock_forecast.values = np.array([101.0, 102.0, 103.0])
        mock_results.forecast.return_value = mock_forecast
        
        # Set results directly
        arima_predictor.results = mock_results
        
        # Mock fit method to avoid actual fitting
        arima_predictor.fit = MagicMock()
        
        # Create test data
        test_data = pd.Series([100.0, 101.0, 102.0, 103.0])
        
        # Call evaluate
        metrics = arima_predictor.evaluate(test_data)
        
        # Check metrics
        assert isinstance(metrics, dict)
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'direction_accuracy' in metrics
    
    def test_evaluate_without_fitting(self, arima_predictor):
        """Test evaluation without fitting"""
        # Create test data
        test_data = pd.Series([100.0, 101.0, 102.0, 103.0])
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            arima_predictor.evaluate(test_data)
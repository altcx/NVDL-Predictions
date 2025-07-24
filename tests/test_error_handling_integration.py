"""
Integration tests for error handling and logging functionality
Tests the interaction between components and error handling
"""
import pytest
import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.error_handler import (
    ErrorHandler, retry_on_exception, safe_execute, log_execution_time,
    APIConnectionError, DataValidationError, ModelTrainingError,
    ConfigurationError, VisualizationError, ErrorContext,
    TimeoutError, RateLimitError, DataIntegrityError, ModelConvergenceError
)
from utils.logger import get_main_logger, LogContext, get_progress_logger
from utils.error_handling_integration import (
    handle_api_failure, handle_data_validation, handle_model_training,
    handle_visualization, log_execution_with_progress, monitor_system_resources,
    create_error_report, setup_error_handling
)
from data.collector import DataCollector
from models.lstm_predictor import LSTMPredictor
from models.arima_predictor import ARIMAPredictor
from models.model_evaluator import ModelEvaluator
from visualization.visualization_engine import VisualizationEngine


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components"""
    
    @pytest.fixture
    def error_handler(self):
        """Create ErrorHandler instance with mocked logger"""
        handler = ErrorHandler()
        handler.logger = MagicMock()
        handler.reset_error_count()
        return handler
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(102, 5, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        return data
    
    def test_api_error_handling_with_retry(self, error_handler):
        """Test API error handling with retry decorator"""
        # Create a function that fails with API error then succeeds
        mock_func = Mock(side_effect=[
            APIConnectionError("API connection failed"),
            "success"
        ])
        
        # Apply retry decorator
        decorated_func = retry_on_exception(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=[APIConnectionError]
        )(mock_func)
        
        # Call decorated function
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = decorated_func()
        
        # Verify function was called twice and succeeded
        assert mock_func.call_count == 2
        assert result == "success"
    
    def test_data_validation_error_handling(self, error_handler, sample_data):
        """Test data validation error handling"""
        # Create DataCollector with mocked components
        with patch('data.collector.StockHistoricalDataClient'):
            collector = DataCollector(api_key="test", secret_key="test")
            collector.logger = MagicMock()
        
        # Create invalid data
        invalid_data = sample_data.copy()
        invalid_data.loc[invalid_data.index[0], 'close'] = -100  # Negative price
        
        # Test error handling
        with pytest.raises(DataValidationError):
            with ErrorContext(error_handler, "Data validation"):
                collector.validate_data_completeness(invalid_data)
        
        # Verify error handler was called
        assert error_handler.logger.error.called
    
    def test_model_training_error_handling(self, error_handler):
        """Test model training error handling"""
        # Create a mock model with training error
        mock_model = MagicMock()
        mock_model.train.side_effect = ModelTrainingError("Failed to converge")
        
        # Test error handling
        with pytest.raises(ModelTrainingError):
            with ErrorContext(error_handler, "Model training", model_type="LSTM"):
                mock_model.train(MagicMock(), MagicMock())
        
        # Verify error handler methods were called
        assert error_handler.logger.error.called
    
    def test_visualization_error_handling(self, error_handler):
        """Test visualization error handling"""
        # Create a mock visualization engine with error
        mock_viz = MagicMock()
        mock_viz.plot_price_with_signals.side_effect = VisualizationError("Failed to generate chart")
        
        # Test error handling
        with pytest.raises(VisualizationError):
            with ErrorContext(error_handler, "Visualization", chart_type="price"):
                mock_viz.plot_price_with_signals(MagicMock(), MagicMock())
        
        # Verify error handler methods were called
        assert error_handler.logger.error.called
    
    def test_error_threshold_tracking_across_components(self, error_handler):
        """Test error threshold tracking across multiple components"""
        # Reset error counts
        error_handler.reset_error_count()
        
        # Simulate multiple API errors
        for i in range(4):
            error_handler.handle_api_error(
                APIConnectionError(f"API error {i}"),
                {"attempt": i}
            )
        
        # Next error should exceed threshold
        error_handler.handle_api_error(
            APIConnectionError("Final API error"),
            {"attempt": 5}
        )
        
        # Verify critical log was called
        error_handler.logger.critical.assert_called()
    
    def test_log_context_with_error(self):
        """Test LogContext with error handling"""
        logger = MagicMock()
        
        # Test with error
        try:
            with LogContext(logger, "Test operation"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Verify logger methods were called
        logger.log.assert_called()
        logger.error.assert_called()
    
    def test_progress_logger_tracking(self):
        """Test progress logger tracking"""
        logger = MagicMock()
        progress_logger = get_progress_logger("test", 100)
        progress_logger.logger = logger
        
        # Start tracking
        progress_logger.start("Test operation")
        
        # Update progress
        for i in range(0, 101, 10):
            progress_logger.update(i, f"Step {i}")
        
        # Complete tracking
        progress_logger.complete("Test operation")
        
        # Verify logger methods were called
        assert logger.info.call_count >= 10  # At least 10 progress updates
    
    def test_safe_execute_with_error_suppression(self):
        """Test safe_execute decorator with error suppression"""
        # Create a function that raises an exception
        @safe_execute(error_message="Operation failed", raise_exception=False)
        def failing_func():
            raise ValueError("Test error")
        
        # Call function and verify it doesn't raise
        result = failing_func()
        assert result is None
    
    def test_log_execution_time(self):
        """Test log_execution_time decorator"""
        logger = MagicMock()
        
        # Create a function with the decorator
        @log_execution_time
        def slow_func():
            import time
            time.sleep(0.01)
            return "done"
        
        # Mock the logger
        with patch('utils.error_handler.get_main_logger', return_value=logger):
            # Call function
            result = slow_func()
            
            # Verify logger was called and function returned correctly
            assert logger.log.called
            assert result == "done"
    
    # New tests for enhanced error handling
    
    def test_handle_api_failure_decorator(self):
        """Test handle_api_failure decorator with retry logic"""
        logger = MagicMock()
        
        # Create a function that fails with API error then succeeds
        @handle_api_failure
        def api_func():
            if not hasattr(api_func, 'called'):
                api_func.called = True
                raise APIConnectionError("API connection failed")
            return "success"
        
        # Mock dependencies
        with patch('utils.error_handling_integration.get_data_logger', return_value=logger), \
             patch('utils.error_handling_integration.config') as mock_config, \
             patch('time.sleep'):
            
            # Configure mock config
            mock_config.MAX_RETRIES = 3
            mock_config.RETRY_DELAY = 0.01
            mock_config.RETRY_BACKOFF = 2.0
            
            # Call function
            result = api_func()
            
            # Verify function succeeded after retry
            assert result == "success"
            assert logger.info.called
    
    def test_handle_data_validation_decorator(self):
        """Test handle_data_validation decorator"""
        logger = MagicMock()
        error_handler_mock = MagicMock()
        
        # Create a function that raises data validation error
        @handle_data_validation
        def validation_func():
            raise DataValidationError("Invalid data")
        
        # Mock dependencies
        with patch('utils.error_handling_integration.get_data_logger', return_value=logger), \
             patch('utils.error_handling_integration.error_handler', error_handler_mock):
            
            # Call function and expect exception
            with pytest.raises(DataValidationError):
                validation_func()
            
            # Verify error was handled
            assert logger.error.called
            assert error_handler_mock.handle_data_validation_error.called
    
    def test_handle_model_training_decorator(self):
        """Test handle_model_training decorator"""
        logger = MagicMock()
        progress_logger = MagicMock()
        error_handler_mock = MagicMock()
        
        # Create a function that raises model training error
        @handle_model_training
        def training_func():
            raise ModelTrainingError("Training failed", model_type="LSTM", epoch=10)
        
        # Mock dependencies
        with patch('utils.error_handling_integration.get_model_logger', return_value=logger), \
             patch('utils.error_handling_integration.get_progress_logger', return_value=progress_logger), \
             patch('utils.error_handling_integration.log_memory_usage'), \
             patch('utils.error_handling_integration.error_handler', error_handler_mock):
            
            # Call function and expect exception
            with pytest.raises(ModelTrainingError):
                training_func()
            
            # Verify error was handled
            assert logger.error.called
            assert error_handler_mock.handle_model_error.called
            assert progress_logger.start.called
    
    def test_handle_visualization_decorator(self):
        """Test handle_visualization decorator"""
        logger = MagicMock()
        error_handler_mock = MagicMock()
        
        # Create a function that raises visualization error
        @handle_visualization
        def visualization_func():
            raise VisualizationError("Visualization failed")
        
        # Mock dependencies
        with patch('utils.error_handling_integration.get_visualization_logger', return_value=logger), \
             patch('utils.error_handling_integration.error_handler', error_handler_mock):
            
            # Call function and expect exception
            with pytest.raises(VisualizationError):
                visualization_func()
            
            # Verify error was handled
            assert logger.error.called
            assert error_handler_mock.handle_visualization_error.called
    
    def test_log_execution_with_progress_decorator(self):
        """Test log_execution_with_progress decorator"""
        logger = MagicMock()
        progress_logger = MagicMock()
        
        # Create a function with the decorator
        @log_execution_with_progress(total_steps=50)
        def process_func():
            return "completed"
        
        # Mock dependencies
        with patch('utils.error_handling_integration.get_data_logger', return_value=logger), \
             patch('utils.error_handling_integration.get_progress_logger', return_value=progress_logger):
            
            # Call function
            result = process_func()
            
            # Verify progress tracking was used
            assert progress_logger.start.called
            assert progress_logger.complete.called
            assert result == "completed"
    
    def test_monitor_system_resources(self):
        """Test monitor_system_resources function"""
        logger = MagicMock()
        
        # Mock psutil functions
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Configure mocks
            mock_memory.return_value = MagicMock(
                percent=75.0,
                used=8 * 1024 * 1024 * 1024,  # 8 GB
                total=16 * 1024 * 1024 * 1024  # 16 GB
            )
            mock_cpu.return_value = 50.0
            mock_disk.return_value = MagicMock(
                percent=60.0,
                used=500 * 1024 * 1024 * 1024,  # 500 GB
                total=1000 * 1024 * 1024 * 1024  # 1 TB
            )
            
            # Call function
            monitor_system_resources(logger)
            
            # Verify logger was called
            assert logger.info.called
    
    def test_create_error_report(self):
        """Test create_error_report function"""
        logger = MagicMock()
        error_handler_mock = MagicMock()
        error_handler_mock._error_counts = {'api': 2, 'data': 1}
        error_handler_mock._error_thresholds = {'api': 5, 'data': 3}
        
        # Mock dependencies
        with patch('utils.error_handling_integration.get_main_logger', return_value=logger), \
             patch('utils.error_handling_integration.error_handler', error_handler_mock), \
             patch('utils.error_handling_integration.get_recent_errors', return_value=["Error 1", "Error 2"]):
            
            # Call function
            report = create_error_report()
            
            # Verify report structure
            assert 'timestamp' in report
            assert 'error_counts' in report
            assert 'error_thresholds' in report
            assert 'system_info' in report
            assert 'recent_errors' in report
            assert report['error_counts']['api'] == 2
            assert report['error_counts']['data'] == 1
    
    def test_setup_error_handling(self):
        """Test setup_error_handling function"""
        logger = MagicMock()
        error_handler_mock = MagicMock()
        
        # Mock dependencies
        with patch('utils.error_handling_integration.get_main_logger', return_value=logger), \
             patch('utils.error_handling_integration.error_handler', error_handler_mock), \
             patch('utils.error_handling_integration.register_component_cleanup_handlers'), \
             patch('utils.error_handling_integration.configure_error_thresholds'), \
             patch('utils.error_handling_integration.setup_system_monitoring'):
            
            # Call function
            setup_error_handling()
            
            # Verify logger was called
            assert logger.info.called


if __name__ == '__main__':
    pytest.main([__file__])
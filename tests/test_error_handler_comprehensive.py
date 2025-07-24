"""
Comprehensive tests for error handling system
Tests error handler, retry logic, and error recovery mechanisms
"""
import pytest
import time
import threading
import os
import logging
import tempfile
from unittest.mock import Mock, patch, MagicMock, call

from utils.error_handler import (
    ErrorHandler, retry_on_exception, safe_execute, log_execution_time,
    RetryableError, APIConnectionError, DataValidationError, ModelTrainingError,
    ConfigurationError, VisualizationError, ResourceNotFoundError, DataProcessingError,
    ErrorContext
)
from utils.error_handling_integration import (
    setup_error_handling, handle_api_failure, handle_data_validation,
    handle_model_training, handle_visualization, log_execution_with_progress,
    monitor_system_resources, create_error_report
)
from utils.logger import get_main_logger, get_data_logger, get_progress_logger


class TestErrorHandlerInitialization:
    """Test error handler initialization and singleton pattern"""
    
    def test_singleton_pattern(self):
        """Test that ErrorHandler follows singleton pattern"""
        handler1 = ErrorHandler()
        handler2 = ErrorHandler()
        
        # Both instances should be the same object
        assert handler1 is handler2
        
        # Modifying one should affect the other
        handler1._error_counts['api'] = 10
        assert handler2._error_counts['api'] == 10
        
        # Reset for other tests
        handler1._error_counts['api'] = 0
    
    def test_initialization(self):
        """Test error handler initialization"""
        handler = ErrorHandler()
        
        # Check that error counts are initialized
        assert 'api' in handler._error_counts
        assert 'data' in handler._error_counts
        assert 'model' in handler._error_counts
        assert 'visualization' in handler._error_counts
        assert 'configuration' in handler._error_counts
        assert 'system' in handler._error_counts
        
        # Check that error thresholds are initialized
        assert 'api' in handler._error_thresholds
        assert 'data' in handler._error_thresholds
        assert 'model' in handler._error_thresholds
        assert 'visualization' in handler._error_thresholds
        assert 'configuration' in handler._error_thresholds
        assert 'system' in handler._error_thresholds
        
        # Check that logger is initialized
        assert handler.logger is not None


class TestErrorHandlerRetryLogic:
    """Test error handler retry logic"""
    
    def test_retry_decorator_success(self):
        """Test retry decorator with successful function"""
        handler = ErrorHandler()
        
        mock_func = Mock(return_value="success")
        decorated_func = handler.retry()(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_decorator_with_retryable_error(self):
        """Test retry decorator with retryable error"""
        handler = ErrorHandler()
        
        # Mock function that fails twice then succeeds
        mock_func = Mock(side_effect=[
            RetryableError("First failure"),
            RetryableError("Second failure"),
            "success"
        ])
        
        # Create decorated function with no delay
        decorated_func = handler.retry(
            max_retries=3,
            base_delay=0.01,
            backoff_factor=1.0,
            jitter=False
        )(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_decorator_max_retries_exceeded(self):
        """Test retry decorator with max retries exceeded"""
        handler = ErrorHandler()
        
        # Mock function that always fails
        mock_func = Mock(side_effect=RetryableError("Persistent failure"))
        
        # Create decorated function with no delay
        decorated_func = handler.retry(
            max_retries=3,
            base_delay=0.01,
            backoff_factor=1.0,
            jitter=False
        )(mock_func)
        
        with pytest.raises(RetryableError, match="Persistent failure"):
            decorated_func()
        
        assert mock_func.call_count == 3
    
    def test_retry_decorator_with_non_retryable_error(self):
        """Test retry decorator with non-retryable error"""
        handler = ErrorHandler()
        
        # Mock function that raises non-retryable error
        mock_func = Mock(side_effect=ValueError("Non-retryable error"))
        
        # Create decorated function
        decorated_func = handler.retry()(mock_func)
        
        with pytest.raises(ValueError, match="Non-retryable error"):
            decorated_func()
        
        assert mock_func.call_count == 1
    
    def test_retry_decorator_with_timeout(self):
        """Test retry decorator with timeout"""
        handler = ErrorHandler()
        
        # Mock _run_with_timeout to simulate timeout
        handler._run_with_timeout = Mock(side_effect=TimeoutError("Operation timed out"))
        
        # Mock function (should not be called due to timeout)
        mock_func = Mock(return_value="success")
        
        # Create decorated function
        decorated_func = handler.retry(
            timeout=1.0,
            retry_on_timeout=False
        )(mock_func)
        
        with pytest.raises(TimeoutError, match="Operation timed out"):
            decorated_func()
    
    def test_retry_with_custom_exceptions(self):
        """Test retry with custom retryable exceptions"""
        handler = ErrorHandler()
        
        # Mock function that raises custom exception
        mock_func = Mock(side_effect=[
            ValueError("Custom error"),
            "success"
        ])
        
        # Create decorated function with custom retryable exceptions
        decorated_func = handler.retry(
            max_retries=2,
            base_delay=0.01,
            retryable_exceptions=[ValueError],
            jitter=False
        )(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 2


class TestErrorHandlerErrorHandling:
    """Test error handler error handling functions"""
    
    def test_handle_api_error(self):
        """Test API error handling"""
        handler = ErrorHandler()
        
        # Mock logger
        handler.logger = Mock()
        
        # Create API error
        error = APIConnectionError("API connection failed")
        
        # Handle error
        handler.handle_api_error(error, {"endpoint": "test_endpoint"})
        
        # Check that logger was called
        assert handler.logger.error.call_count >= 1
        assert handler.logger.info.call_count >= 1
    
    def test_handle_data_validation_error(self):
        """Test data validation error handling"""
        handler = ErrorHandler()
        
        # Mock logger
        handler.logger = Mock()
        
        # Create data validation error
        error = DataValidationError("Invalid data format")
        
        # Handle error
        handler.handle_data_validation_error(error, {"data_shape": "(100, 5)"})
        
        # Check that logger was called
        assert handler.logger.error.call_count >= 1
        assert handler.logger.info.call_count >= 1
    
    def test_handle_model_error(self):
        """Test model error handling"""
        handler = ErrorHandler()
        
        # Mock logger
        handler.logger = Mock()
        
        # Create model error
        error = ModelTrainingError("Model failed to converge", model_type="LSTM", epoch=10)
        
        # Handle error
        handler.handle_model_error(error, {"model_type": "LSTM", "batch_size": 32})
        
        # Check that logger was called
        assert handler.logger.error.call_count >= 1
        assert handler.logger.info.call_count >= 1
    
    def test_check_error_threshold(self):
        """Test error threshold checking"""
        handler = ErrorHandler()
        
        # Set error threshold
        handler._error_thresholds['api'] = 3
        handler._error_counts['api'] = 0
        
        # Check threshold (should not be exceeded)
        assert not handler.check_error_threshold('api')
        assert handler._error_counts['api'] == 1
        
        # Check threshold again
        assert not handler.check_error_threshold('api')
        assert handler._error_counts['api'] == 2
        
        # Check threshold one more time (should be exceeded)
        assert handler.check_error_threshold('api')
        assert handler._error_counts['api'] == 3
        
        # Reset error count
        handler.reset_error_count('api')
        assert handler._error_counts['api'] == 0
        
        # Reset all error counts
        handler._error_counts['data'] = 2
        handler.reset_error_count()
        assert handler._error_counts['api'] == 0
        assert handler._error_counts['data'] == 0


class TestErrorHandlerUtilityFunctions:
    """Test error handler utility functions"""
    
    def test_retry_on_exception_decorator(self):
        """Test retry_on_exception decorator"""
        # Mock function that fails twice then succeeds
        mock_func = Mock(side_effect=[
            ValueError("First failure"),
            ValueError("Second failure"),
            "success"
        ])
        
        # Create decorated function with no delay
        @retry_on_exception(max_retries=3, delay=0.01, backoff=1.0, exceptions=[ValueError])
        def test_func():
            return mock_func()
        
        result = test_func()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_safe_execute(self):
        """Test safe_execute function"""
        # Test with successful function
        result, error = safe_execute(lambda: "success")
        assert result == "success"
        assert error is None
        
        # Test with function that raises exception
        result, error = safe_execute(lambda: 1/0)
        assert result is None
        assert isinstance(error, ZeroDivisionError)
    
    def test_log_execution_time(self):
        """Test log_execution_time decorator"""
        logger = Mock()
        
        @log_execution_time(logger)
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        
        assert result == "done"
        assert logger.info.call_count >= 2
        
        # Check that execution time was logged
        log_calls = [call[0][0] for call in logger.info.call_args_list]
        assert any("Executing" in str(call) for call in log_calls)
        assert any("completed in" in str(call) for call in log_calls)


class TestErrorHandlingIntegration:
    """Test error handling integration functions"""
    
    @patch('utils.error_handling_integration.error_handler')
    def test_setup_error_handling(self, mock_error_handler):
        """Test setup_error_handling function"""
        setup_error_handling()
        
        # Check that register_cleanup_handler was called
        assert mock_error_handler.register_cleanup_handler.call_count > 0
    
    def test_handle_api_failure_decorator(self):
        """Test handle_api_failure decorator"""
        # Mock function that succeeds
        mock_func = Mock(return_value="success")
        
        # Create decorated function
        decorated_func = handle_api_failure(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
        
        # Mock function that fails with retryable error
        mock_func = Mock(side_effect=APIConnectionError("API error"))
        
        # Create decorated function
        decorated_func = handle_api_failure(mock_func)
        
        with pytest.raises(APIConnectionError):
            decorated_func()
    
    def test_handle_data_validation_decorator(self):
        """Test handle_data_validation decorator"""
        # Mock function that succeeds
        mock_func = Mock(return_value="success")
        
        # Create decorated function
        decorated_func = handle_data_validation(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
        
        # Mock function that fails with data validation error
        mock_func = Mock(side_effect=DataValidationError("Invalid data"))
        
        # Create decorated function
        decorated_func = handle_data_validation(mock_func)
        
        with pytest.raises(DataValidationError):
            decorated_func()
    
    @patch('utils.error_handling_integration.get_progress_logger')
    def test_handle_model_training_decorator(self, mock_get_progress_logger):
        """Test handle_model_training decorator"""
        # Mock progress logger
        mock_progress = Mock()
        mock_get_progress_logger.return_value = mock_progress
        
        # Mock function that succeeds
        mock_func = Mock(return_value="success")
        
        # Create decorated function
        decorated_func = handle_model_training(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
        assert mock_progress.start.call_count == 1
        assert mock_progress.complete.call_count == 1
        
        # Mock function that fails with model training error
        mock_func = Mock(side_effect=ModelTrainingError("Training failed"))
        
        # Create decorated function
        decorated_func = handle_model_training(mock_func)
        
        with pytest.raises(ModelTrainingError):
            decorated_func()
    
    def test_handle_visualization_decorator(self):
        """Test handle_visualization decorator"""
        # Mock function that succeeds
        mock_func = Mock(return_value="success")
        
        # Create decorated function
        decorated_func = handle_visualization(mock_func)
        
        result = decorated_func()
        
        assert result == "success"
        assert mock_func.call_count == 1
        
        # Mock function that fails with visualization error
        mock_func = Mock(side_effect=VisualizationError("Visualization failed"))
        
        # Create decorated function
        decorated_func = handle_visualization(mock_func)
        
        with pytest.raises(VisualizationError):
            decorated_func()
    
    @patch('utils.error_handling_integration.get_progress_logger')
    def test_log_execution_with_progress(self, mock_get_progress_logger):
        """Test log_execution_with_progress decorator"""
        # Mock progress logger
        mock_progress = Mock()
        mock_get_progress_logger.return_value = mock_progress
        
        # Create decorated function
        @log_execution_with_progress(total_steps=100)
        def test_func():
            return "success"
        
        result = test_func()
        
        assert result == "success"
        assert mock_progress.start.call_count == 1
        assert mock_progress.complete.call_count == 1
    
    @patch('utils.error_handling_integration.psutil')
    def test_monitor_system_resources(self, mock_psutil):
        """Test monitor_system_resources function"""
        # Mock psutil values
        mock_memory = Mock()
        mock_memory.percent = 50
        mock_memory.used = 4 * 1024 * 1024 * 1024  # 4 GB
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16 GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_psutil.cpu_percent.return_value = 30
        
        mock_disk = Mock()
        mock_disk.percent = 40
        mock_disk.used = 100 * 1024 * 1024 * 1024  # 100 GB
        mock_disk.total = 500 * 1024 * 1024 * 1024  # 500 GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        # Mock logger
        mock_logger = Mock()
        
        # Call function
        monitor_system_resources(mock_logger)
        
        # Check that logger was called
        assert mock_logger.info.call_count >= 1
    
    @patch('utils.error_handling_integration.get_recent_errors')
    def test_create_error_report(self, mock_get_recent_errors):
        """Test create_error_report function"""
        # Mock get_recent_errors
        mock_get_recent_errors.return_value = ["Error 1", "Error 2"]
        
        # Create error report
        report = create_error_report()
        
        # Check report structure
        assert 'timestamp' in report
        assert 'error_counts' in report
        assert 'error_thresholds' in report
        assert 'system_info' in report
        assert 'recent_errors' in report
        assert report['recent_errors'] == ["Error 1", "Error 2"]


class TestErrorHandlerIntegrationWithLogger:
    """Test integration between error handler and logger"""
    
    def test_error_handler_with_logger(self):
        """Test error handler with real logger"""
        # Create temporary log file
        with tempfile.NamedTemporaryFile(delete=False) as temp_log:
            log_path = temp_log.name
        
        try:
            # Configure logger
            logger = logging.getLogger('test_logger')
            logger.setLevel(logging.INFO)
            
            # Add file handler
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
            logger.addHandler(handler)
            
            # Create error handler with mock logger
            error_handler = ErrorHandler()
            error_handler.logger = logger
            
            # Handle API error
            error = APIConnectionError("Test API error")
            error_handler.handle_api_error(error)
            
            # Check log file content
            with open(log_path, 'r') as f:
                log_content = f.read()
                
            assert "ERROR:API Error" in log_content
            
        finally:
            # Clean up
            if os.path.exists(log_path):
                os.remove(log_path)


class TestErrorHandlerPerformance:
    """Test error handler performance"""
    
    def test_retry_performance(self):
        """Test retry decorator performance"""
        handler = ErrorHandler()
        
        # Create a function that succeeds immediately
        def fast_success():
            return "success"
        
        # Create a decorated function
        decorated_func = handler.retry()(fast_success)
        
        # Measure execution time
        start_time = time.time()
        for _ in range(1000):
            decorated_func()
        end_time = time.time()
        
        # Execution time should be reasonable
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"Retry decorator too slow: {execution_time:.2f}s for 1000 calls"
    
    def test_error_handling_performance(self):
        """Test error handling performance"""
        handler = ErrorHandler()
        
        # Mock logger to prevent actual logging
        handler.logger = Mock()
        
        # Measure execution time for handling errors
        start_time = time.time()
        for _ in range(100):
            error = APIConnectionError("Test API error")
            handler.handle_api_error(error)
        end_time = time.time()
        
        # Execution time should be reasonable
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"Error handling too slow: {execution_time:.2f}s for 100 errors"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
"""
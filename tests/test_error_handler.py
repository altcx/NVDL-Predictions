"""
Tests for error handling utilities
"""
import unittest
import os
import sys
import time
import logging
import threading
import signal
from unittest.mock import patch, MagicMock, call

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.error_handler import ErrorHandler
from utils.error_handler import RetryableError, APIConnectionError, DataValidationError, ModelTrainingError
from utils.error_handler import ConfigurationError, VisualizationError
from utils.error_handler import error_handler, ErrorContext
from utils.error_handler import retry_on_exception, safe_execute, log_execution_time

# Use built-in TimeoutError for tests
from socket import timeout as TimeoutError


class TestErrorHandler(unittest.TestCase):
    """Test error handling utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock logger
        self.mock_logger = MagicMock()
        
        # Create error handler with mock logger
        self.error_handler = ErrorHandler()
        self.error_handler.logger = self.mock_logger
    
    def test_retry_decorator_success(self):
        """Test retry decorator with successful function"""
        # Create a function that succeeds
        mock_func = MagicMock(return_value="success")
        
        # Apply retry decorator
        decorated_func = self.error_handler.retry()(mock_func)
        
        # Call decorated function
        result = decorated_func()
        
        # Verify function was called once
        mock_func.assert_called_once()
        self.assertEqual(result, "success")
    
    def test_retry_decorator_with_retryable_error(self):
        """Test retry decorator with retryable error"""
        # Create a function that fails with retryable error
        mock_func = MagicMock(side_effect=[
            RetryableError("Retry me"),
            RetryableError("Retry me again"),
            "success"
        ])
        
        # Apply retry decorator with no delay
        decorated_func = self.error_handler.retry(
            max_retries=3,
            base_delay=0.01,
            error_message="Test failed"
        )(mock_func)
        
        # Call decorated function
        result = decorated_func()
        
        # Verify function was called three times
        self.assertEqual(mock_func.call_count, 3)
        self.assertEqual(result, "success")
    
    def test_retry_decorator_with_max_retries_exceeded(self):
        """Test retry decorator with max retries exceeded"""
        # Create a function that always fails with retryable error
        mock_func = MagicMock(side_effect=RetryableError("Always fail"))
        
        # Apply retry decorator with no delay
        decorated_func = self.error_handler.retry(
            max_retries=3,
            base_delay=0.01,
            error_message="Test failed"
        )(mock_func)
        
        # Call decorated function and expect exception
        with self.assertRaises(RetryableError):
            decorated_func()
        
        # Verify function was called three times
        self.assertEqual(mock_func.call_count, 3)
    
    def test_retry_decorator_with_non_retryable_error(self):
        """Test retry decorator with non-retryable error"""
        # Create a function that fails with non-retryable error
        mock_func = MagicMock(side_effect=ValueError("Non-retryable"))
        
        # Apply retry decorator
        decorated_func = self.error_handler.retry()(mock_func)
        
        # Call decorated function and expect exception
        with self.assertRaises(ValueError):
            decorated_func()
        
        # Verify function was called once
        mock_func.assert_called_once()
    
    def test_handle_api_error(self):
        """Test API error handling"""
        # Create API error
        api_error = APIConnectionError("API connection failed")
        
        # Handle error
        self.error_handler.handle_api_error(api_error, {"endpoint": "/data"})
        
        # Verify logger was called
        self.mock_logger.error.assert_called()
        self.mock_logger.info.assert_called()
    
    def test_error_context_manager_with_api_error(self):
        """Test ErrorContext with API error"""
        # Create error handler with mock logger
        error_handler = ErrorHandler()
        error_handler.logger = self.mock_logger
        error_handler.handle_api_error = MagicMock()
        
        # Test with API error
        with self.assertRaises(APIConnectionError):
            with ErrorContext(error_handler, "API operation", endpoint="/data"):
                raise APIConnectionError("API failed")
        
        # Verify error handler was called
        error_handler.handle_api_error.assert_called_once()
    
    def test_error_context_manager_with_data_error(self):
        """Test ErrorContext with data validation error"""
        # Create error handler with mock logger
        error_handler = ErrorHandler()
        error_handler.logger = self.mock_logger
        error_handler.handle_data_validation_error = MagicMock()
        
        # Test with data validation error
        with self.assertRaises(DataValidationError):
            with ErrorContext(error_handler, "Data operation", rows=100):
                raise DataValidationError("Data invalid")
        
        # Verify error handler was called
        error_handler.handle_data_validation_error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
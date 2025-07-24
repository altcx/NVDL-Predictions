"""
Extended tests for error handling utilities
Tests the comprehensive error handling and logging functionality
"""
import unittest
import os
import sys
import time
import logging
import threading
import signal
import tempfile
from unittest.mock import patch, MagicMock, call

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.error_handler import (
    ErrorHandler, retry_on_exception, safe_execute, log_execution_time,
    APIConnectionError, DataValidationError, ModelTrainingError,
    ConfigurationError, VisualizationError, ErrorContext
)
# Import these separately to avoid import errors if they don't exist yet
try:
    from utils.error_handler import monitor_system_resources, handle_uncaught_exception
except ImportError:
    # Define mock functions for testing
    def monitor_system_resources(logger=None):
        return {"cpu_percent": 0, "memory_mb": 0, "virtual_memory_mb": 0, "system_memory_percent": 0}
    
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        pass
from utils.logger import get_main_logger, LogContext, get_progress_logger


class TestExtendedErrorHandler(unittest.TestCase):
    """Test extended error handling utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock logger
        self.mock_logger = MagicMock()
        
        # Create error handler with mock logger
        self.error_handler = ErrorHandler()
        self.error_handler.logger = self.mock_logger
        
        # Reset error counts if method exists
        if hasattr(self.error_handler, 'reset_error_count'):
            self.error_handler.reset_error_count()
        else:
            # Manually reset error counts
            self.error_handler._error_counts = {
                'api': 0,
                'data': 0,
                'model': 0,
                'visualization': 0,
                'configuration': 0,
                'system': 0
            }
    
    def test_handle_configuration_error(self):
        """Test configuration error handling"""
        # Create configuration error
        config_error = ConfigurationError("Invalid configuration")
        
        # Handle error
        self.error_handler.handle_configuration_error(config_error, {"file": "config.json"})
        
        # Verify logger was called
        self.mock_logger.error.assert_called()
        self.mock_logger.info.assert_called()
        self.mock_logger.critical.assert_called()
    
    def test_graceful_exit(self):
        """Test graceful exit functionality"""
        # Mock sys.exit to prevent actual exit
        with patch('sys.exit') as mock_exit:
            # Mock cleanup handlers
            self.error_handler._run_cleanup_handlers = MagicMock()
            
            # Call graceful exit
            self.error_handler.graceful_exit(1, "Test exit")
            
            # Verify methods were called
            self.mock_logger.critical.assert_called()
            self.error_handler._run_cleanup_handlers.assert_called_once()
            mock_exit.assert_called_once_with(1)
    
    def test_monitor_system_resources(self):
        """Test system resource monitoring"""
        # Mock psutil
        with patch('psutil.Process') as mock_process:
            # Setup mock process
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value = MagicMock(rss=1024*1024*100, vms=1024*1024*200)
            mock_proc.cpu_percent.return_value = 5.0
            mock_process.return_value = mock_proc
            
            # Mock virtual memory
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value = MagicMock(percent=50.0)
                
                # Call monitor function
                result = monitor_system_resources(self.mock_logger)
                
                # Verify logger was called
                self.mock_logger.info.assert_called()
                
                # Verify result
                self.assertEqual(result["cpu_percent"], 5.0)
                self.assertEqual(result["memory_mb"], 100.0)
                self.assertEqual(result["virtual_memory_mb"], 200.0)
                self.assertEqual(result["system_memory_percent"], 50.0)
    
    def test_error_context_with_configuration_error(self):
        """Test ErrorContext with configuration error"""
        # Create error handler with mock logger
        error_handler = ErrorHandler()
        error_handler.logger = self.mock_logger
        error_handler.handle_configuration_error = MagicMock()
        
        # Test with configuration error
        with self.assertRaises(ConfigurationError):
            with ErrorContext(error_handler, "Config operation", file="config.json"):
                raise ConfigurationError("Config invalid")
        
        # Verify error handler was called
        error_handler.handle_configuration_error.assert_called_once()
    
    def test_safe_execute_decorator(self):
        """Test safe_execute decorator"""
        # Create a function that raises an exception
        @safe_execute(error_message="Test failed", raise_exception=False, default_return="default")
        def failing_func():
            raise ValueError("Test error")
        
        # Call function
        result = failing_func()
        
        # Verify result
        self.assertEqual(result, "default")
    
    def test_log_execution_time_decorator(self):
        """Test log_execution_time decorator"""
        # Create a mock logger
        mock_logger = MagicMock()
        
        # Mock get_main_logger
        with patch('utils.error_handler.get_main_logger', return_value=mock_logger):
            # Create a function with the decorator
            @log_execution_time(level=logging.DEBUG)
            def test_func():
                time.sleep(0.01)
                return "success"
            
            # Call function
            result = test_func()
            
            # Verify logger was called and function returned correctly
            mock_logger.log.assert_called()
            self.assertEqual(result, "success")
    
    def test_log_execution_time_with_threshold(self):
        """Test log_execution_time decorator with threshold"""
        # Create a mock logger
        mock_logger = MagicMock()
        
        # Mock get_main_logger
        with patch('utils.error_handler.get_main_logger', return_value=mock_logger):
            # Create a function with the decorator
            @log_execution_time(threshold=1.0)  # High threshold
            def fast_func():
                return "fast"
            
            # Call function
            result = fast_func()
            
            # Verify logger was not called (execution time below threshold)
            mock_logger.log.assert_not_called()
            self.assertEqual(result, "fast")
    
    def test_handle_uncaught_exception(self):
        """Test uncaught exception handler"""
        # Mock sys.__excepthook__
        with patch('sys.__excepthook__') as mock_excepthook:
            # Mock get_main_logger
            mock_logger = MagicMock()
            with patch('utils.error_handler.get_main_logger', return_value=mock_logger):
                # Mock get_recent_errors
                with patch('utils.error_handler.get_recent_errors', return_value=["Error 1", "Error 2"]):
                    # Mock monitor_system_resources
                    with patch('utils.error_handler.monitor_system_resources'):
                        # Create exception
                        try:
                            raise ValueError("Test error")
                        except ValueError:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                        
                        # Call handler
                        handle_uncaught_exception(exc_type, exc_value, exc_traceback)
                        
                        # Verify logger was called
                        mock_logger.critical.assert_called()
                        
                        # Verify excepthook was called
                        mock_excepthook.assert_called_once()


if __name__ == '__main__':
    unittest.main()
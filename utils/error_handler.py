"""
Error handling utilities for NVDL Stock Predictor
Provides standardized error handling, retry logic, and graceful recovery
"""
import time
import sys
import functools
import traceback
import inspect
import os
from typing import Callable, Any, Dict, Optional, Type, Union, List, Tuple
from utils.logger import get_main_logger


class RetryableError(Exception):
    """Base class for errors that can be retried"""
    pass


class DataValidationError(Exception):
    """Error raised when data validation fails"""
    pass


class ModelTrainingError(Exception):
    """Error raised when model training fails"""
    pass


class APIConnectionError(RetryableError):
    """Error raised when API connection fails"""
    pass


class DataProcessingError(Exception):
    """Error raised when data processing fails"""
    pass


class ConfigurationError(Exception):
    """Error raised when configuration is invalid"""
    pass


class VisualizationError(Exception):
    """Error raised when visualization generation fails"""
    pass


class ResourceNotFoundError(Exception):
    """Error raised when a required resource is not found"""
    pass


class ErrorHandler:
    """
    Centralized error handling utility for NVDL Stock Predictor
    Provides retry logic, error categorization, and graceful recovery
    """
    
    def __init__(self):
        """Initialize error handler with logger"""
        self.logger = get_main_logger()
    
    def retry(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        error_message: str = "Operation failed",
        jitter: bool = True
    ) -> Callable:
        """
        Decorator for retrying operations with exponential backoff
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries in seconds
            backoff_factor: Multiplier for delay after each retry
            retryable_exceptions: List of exception types that should trigger retry
            error_message: Message to log on failure
            jitter: Whether to add random jitter to delay times
            
        Returns:
            Decorated function with retry logic
        """
        if retryable_exceptions is None:
            retryable_exceptions = [RetryableError]
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                # Get function context for better error reporting
                caller_frame = inspect.currentframe().f_back
                caller_info = ""
                if caller_frame:
                    caller_info = f" called from {caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
                
                for attempt in range(max_retries):
                    try:
                        if attempt > 0:
                            self.logger.info(f"Retry attempt {attempt+1}/{max_retries} for {func.__name__}{caller_info}")
                        
                        return func(*args, **kwargs)
                        
                    except tuple(retryable_exceptions) as e:
                        last_exception = e
                        
                        if attempt < max_retries - 1:
                            delay = base_delay * (backoff_factor ** attempt)
                            
                            # Add jitter to prevent thundering herd problem
                            if jitter:
                                import random
                                delay = delay * (0.5 + random.random())
                                
                            self.logger.warning(
                                f"Attempt {attempt+1}/{max_retries} failed: {str(e)}. "
                                f"Retrying in {delay:.2f} seconds..."
                            )
                            time.sleep(delay)
                        else:
                            self.logger.error(
                                f"All {max_retries} attempts failed for {func.__name__}: {str(e)}"
                            )
                    except Exception as e:
                        # Non-retryable exception
                        self.logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                        self.logger.debug(f"Stack trace: {traceback.format_exc()}")
                        raise
                
                # If we get here, all retries failed
                self.logger.error(f"{error_message}: {str(last_exception)}")
                raise last_exception
                
            return wrapper
        return decorator
    
    def handle_api_error(self, error: Exception, api_info: Dict[str, Any] = None) -> None:
        """
        Handle API-related errors with appropriate logging
        
        Args:
            error: The exception that was raised
            api_info: Optional dictionary with information about the API call
            
        Returns:
            None
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        self.logger.error(f"API Error ({error_type}): {error_msg}")
        
        if api_info:
            self.logger.info("API call information:")
            for key, value in api_info.items():
                # Don't log sensitive information like API keys
                if 'key' in key.lower() or 'secret' in key.lower() or 'token' in key.lower():
                    self.logger.info(f"  {key}: [REDACTED]")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        # Categorize common API errors
        if "rate limit" in error_msg.lower():
            self.logger.warning("Rate limit exceeded. Consider reducing request frequency or implementing backoff.")
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            self.logger.error("Authentication failed. Check API credentials and permissions.")
        elif "not found" in error_msg.lower():
            self.logger.error("Requested resource not found. Check symbol or endpoint URL.")
        elif "timeout" in error_msg.lower():
            self.logger.warning("Request timed out. Check network connectivity and API server status.")
        elif "connection" in error_msg.lower():
            self.logger.warning("Connection issue. Check network connectivity and firewall settings.")
        elif "forbidden" in error_msg.lower():
            self.logger.error("Access forbidden. Check API permissions and subscription level.")
        
        # Log the stack trace for debugging
        self.logger.debug(f"API Error stack trace: {traceback.format_exc()}")
        
        # Suggest recovery actions
        self.logger.info("Suggested recovery actions:")
        if "rate limit" in error_msg.lower():
            self.logger.info("  - Wait before making additional requests")
            self.logger.info("  - Reduce request frequency")
            self.logger.info("  - Implement exponential backoff")
        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            self.logger.info("  - Verify API credentials")
            self.logger.info("  - Check if API keys are expired")
            self.logger.info("  - Ensure environment variables are set correctly")
        elif "not found" in error_msg.lower():
            self.logger.info("  - Verify the resource identifier (e.g., stock symbol)")
            self.logger.info("  - Check API endpoint URL")
        elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            self.logger.info("  - Check network connectivity")
            self.logger.info("  - Verify API service status")
            self.logger.info("  - Try again later")
    
    def handle_data_validation_error(self, error: Exception, data_info: Dict[str, Any] = None) -> None:
        """
        Handle data validation errors with detailed diagnostics
        
        Args:
            error: The exception that was raised
            data_info: Optional dictionary with information about the data
            
        Returns:
            None
        """
        error_msg = str(error)
        self.logger.error(f"Data Validation Error: {error_msg}")
        
        if data_info:
            self.logger.info("Data diagnostics:")
            for key, value in data_info.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    self.logger.info(f"  {key}: Shape={value.shape}, Null values={value.isnull().sum().sum()}")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        # Categorize common data validation errors
        if "missing" in error_msg.lower():
            self.logger.error("Missing required data. Check data completeness.")
        elif "invalid" in error_msg.lower() or "format" in error_msg.lower():
            self.logger.error("Invalid data format. Check data types and structure.")
        elif "range" in error_msg.lower() or "bound" in error_msg.lower():
            self.logger.error("Data out of expected range. Check for outliers.")
        elif "duplicate" in error_msg.lower():
            self.logger.error("Duplicate data detected. Check for redundant entries.")
        
        self.logger.debug(f"Data Validation Error stack trace: {traceback.format_exc()}")
        
        # Suggest recovery actions
        self.logger.info("Suggested recovery actions:")
        if "missing" in error_msg.lower():
            self.logger.info("  - Fill missing values using forward/backward fill")
            self.logger.info("  - Remove rows with missing values")
            self.logger.info("  - Use interpolation for missing values")
        elif "invalid" in error_msg.lower() or "format" in error_msg.lower():
            self.logger.info("  - Check data types and convert if necessary")
            self.logger.info("  - Verify date formats")
            self.logger.info("  - Ensure numeric columns contain valid numbers")
        elif "range" in error_msg.lower() or "bound" in error_msg.lower():
            self.logger.info("  - Remove or cap outliers")
            self.logger.info("  - Verify data source accuracy")
        elif "duplicate" in error_msg.lower():
            self.logger.info("  - Remove duplicate entries")
            self.logger.info("  - Check data collection process")
    
    def handle_model_error(self, error: Exception, model_info: Dict[str, Any] = None) -> None:
        """
        Handle model training and prediction errors
        
        Args:
            error: The exception that was raised
            model_info: Optional dictionary with information about the model
            
        Returns:
            None
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        self.logger.error(f"Model Error ({error_type}): {error_msg}")
        
        if model_info:
            self.logger.info("Model diagnostics:")
            for key, value in model_info.items():
                if isinstance(value, dict) and len(value) > 10:
                    self.logger.info(f"  {key}: {type(value)} with {len(value)} items")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        # Categorize common model errors
        if "shape" in error_msg.lower() or "dimension" in error_msg.lower():
            self.logger.error("Input shape mismatch. Check data preprocessing and model input requirements.")
        elif "memory" in error_msg.lower():
            self.logger.error("Memory error. Consider reducing batch size, model complexity, or sequence length.")
        elif "convergence" in error_msg.lower():
            self.logger.warning("Model failed to converge. Consider adjusting learning rate, epochs, or model architecture.")
        elif "nan" in error_msg.lower() or "infinity" in error_msg.lower():
            self.logger.error("NaN or infinity values detected. Check for data normalization issues or gradient explosion.")
        elif "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
            self.logger.error("GPU-related error. Check CUDA installation and GPU memory usage.")
        elif "checkpoint" in error_msg.lower() or "save" in error_msg.lower():
            self.logger.error("Model checkpoint error. Check disk space and permissions.")
        
        self.logger.debug(f"Model Error stack trace: {traceback.format_exc()}")
        
        # Suggest recovery actions
        self.logger.info("Suggested recovery actions:")
        if "shape" in error_msg.lower() or "dimension" in error_msg.lower():
            self.logger.info("  - Verify input data shape matches model expectations")
            self.logger.info("  - Check sequence preparation for time series data")
            self.logger.info("  - Ensure consistent feature dimensions")
        elif "memory" in error_msg.lower():
            self.logger.info("  - Reduce batch size")
            self.logger.info("  - Simplify model architecture")
            self.logger.info("  - Use data generators for large datasets")
        elif "convergence" in error_msg.lower():
            self.logger.info("  - Adjust learning rate")
            self.logger.info("  - Increase maximum iterations")
            self.logger.info("  - Try different optimization algorithm")
        elif "nan" in error_msg.lower() or "infinity" in error_msg.lower():
            self.logger.info("  - Check for NaN values in input data")
            self.logger.info("  - Normalize input features")
            self.logger.info("  - Reduce learning rate")
        elif "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
            self.logger.info("  - Check GPU memory usage")
            self.logger.info("  - Update GPU drivers")
            self.logger.info("  - Fall back to CPU training")
    
    def handle_visualization_error(self, error: Exception, viz_info: Dict[str, Any] = None) -> None:
        """
        Handle visualization generation errors
        
        Args:
            error: The exception that was raised
            viz_info: Optional dictionary with information about the visualization
            
        Returns:
            None
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        self.logger.error(f"Visualization Error ({error_type}): {error_msg}")
        
        if viz_info:
            self.logger.info("Visualization information:")
            for key, value in viz_info.items():
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    self.logger.info(f"  {key}: Shape={value.shape}")
                else:
                    self.logger.info(f"  {key}: {type(value)}")
        
        # Categorize common visualization errors
        if "data" in error_msg.lower() and ("empty" in error_msg.lower() or "none" in error_msg.lower()):
            self.logger.error("Empty data for visualization. Check data availability.")
        elif "layout" in error_msg.lower():
            self.logger.error("Layout configuration error. Check subplot arrangement.")
        elif "color" in error_msg.lower() or "palette" in error_msg.lower():
            self.logger.error("Color specification error. Check color mappings.")
        elif "save" in error_msg.lower() or "write" in error_msg.lower():
            self.logger.error("Error saving visualization. Check file path and permissions.")
        
        self.logger.debug(f"Visualization Error stack trace: {traceback.format_exc()}")
        
        # Suggest recovery actions
        self.logger.info("Suggested recovery actions:")
        if "data" in error_msg.lower() and ("empty" in error_msg.lower() or "none" in error_msg.lower()):
            self.logger.info("  - Verify data is available before visualization")
            self.logger.info("  - Add error handling for empty data cases")
        elif "layout" in error_msg.lower():
            self.logger.info("  - Simplify visualization layout")
            self.logger.info("  - Check subplot grid dimensions")
        elif "save" in error_msg.lower() or "write" in error_msg.lower():
            self.logger.info("  - Verify directory exists")
            self.logger.info("  - Check write permissions")
            self.logger.info("  - Ensure valid file format")
    
    def handle_configuration_error(self, error: Exception, config_info: Dict[str, Any] = None) -> None:
        """
        Handle configuration and environment errors
        
        Args:
            error: The exception that was raised
            config_info: Optional dictionary with configuration information
            
        Returns:
            None
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        self.logger.error(f"Configuration Error ({error_type}): {error_msg}")
        
        if config_info:
            self.logger.info("Configuration information:")
            for key, value in config_info.items():
                # Don't log sensitive information
                if 'key' in key.lower() or 'secret' in key.lower() or 'password' in key.lower():
                    self.logger.info(f"  {key}: [REDACTED]")
                else:
                    self.logger.info(f"  {key}: {value}")
        
        # Categorize common configuration errors
        if "missing" in error_msg.lower() and "environment" in error_msg.lower():
            self.logger.error("Missing environment variable. Check .env file or system environment.")
        elif "invalid" in error_msg.lower() and "value" in error_msg.lower():
            self.logger.error("Invalid configuration value. Check parameter constraints.")
        elif "file" in error_msg.lower() and "not found" in error_msg.lower():
            self.logger.error("Configuration file not found. Check file path.")
        elif "permission" in error_msg.lower():
            self.logger.error("Permission error. Check file access rights.")
        
        self.logger.debug(f"Configuration Error stack trace: {traceback.format_exc()}")
        
        # Suggest recovery actions
        self.logger.info("Suggested recovery actions:")
        if "missing" in error_msg.lower() and "environment" in error_msg.lower():
            self.logger.info("  - Create or update .env file")
            self.logger.info("  - Set required environment variables")
            self.logger.info("  - Check for typos in variable names")
        elif "invalid" in error_msg.lower() and "value" in error_msg.lower():
            self.logger.info("  - Check parameter type and range")
            self.logger.info("  - Refer to documentation for valid values")
        elif "file" in error_msg.lower() and "not found" in error_msg.lower():
            self.logger.info("  - Verify file path")
            self.logger.info("  - Create default configuration file")
        elif "permission" in error_msg.lower():
            self.logger.info("  - Check file permissions")
            self.logger.info("  - Run with appropriate privileges")
    
    def graceful_exit(self, error: Exception, exit_code: int = 1, cleanup_funcs: List[Callable] = None) -> None:
        """
        Perform graceful system exit with proper cleanup
        
        Args:
            error: The exception that caused the exit
            exit_code: Exit code to return to the system
            cleanup_funcs: List of cleanup functions to call before exit
            
        Returns:
            None (exits the program)
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        self.logger.critical(f"Critical error ({error_type}): {error_msg}")
        self.logger.critical(f"Stack trace: {traceback.format_exc()}")
        
        # Get information about where the error occurred
        frame = inspect.trace()[-1]
        filename = frame[1]
        line_number = frame[2]
        function_name = frame[3]
        self.logger.critical(f"Error occurred in {function_name} at {filename}:{line_number}")
        
        # Perform cleanup operations
        if cleanup_funcs:
            self.logger.info("Performing cleanup operations before exit...")
            for cleanup_func in cleanup_funcs:
                try:
                    cleanup_func()
                except Exception as e:
                    self.logger.error(f"Error during cleanup: {str(e)}")
        
        # Save any unsaved work or state if possible
        try:
            self.logger.info("Attempting to save current state...")
            # Example: Save any in-memory data to temporary files
            # This would be implemented based on application needs
        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
        
        self.logger.critical(f"Exiting with code {exit_code}")
        sys.exit(exit_code)


# Import pandas conditionally to avoid circular imports
try:
    import pandas as pd
except ImportError:
    pd = None

class ErrorContext:
    """
    Context manager for error handling
    
    Example:
        with ErrorContext(error_handler, "API operation", api_info={"endpoint": "/data"}):
            response = api.get_data()
    """
    
    def __init__(self, error_handler, operation_name, **context_info):
        """
        Initialize error context
        
        Args:
            error_handler: ErrorHandler instance
            operation_name: Name of the operation being performed
            **context_info: Additional context information for error handling
        """
        self.error_handler = error_handler
        self.operation_name = operation_name
        self.context_info = context_info
        self.logger = get_main_logger()
    
    def __enter__(self):
        """Enter context manager"""
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and handle any exceptions"""
        if exc_type is None:
            self.logger.debug(f"Operation completed successfully: {self.operation_name}")
            return False
        
        # Handle different types of errors
        if issubclass(exc_type, APIConnectionError) or "api" in self.operation_name.lower():
            self.error_handler.handle_api_error(exc_val, self.context_info)
        elif issubclass(exc_type, DataValidationError) or "data" in self.operation_name.lower():
            self.error_handler.handle_data_validation_error(exc_val, self.context_info)
        elif issubclass(exc_type, ModelTrainingError) or "model" in self.operation_name.lower():
            self.error_handler.handle_model_error(exc_val, self.context_info)
        elif issubclass(exc_type, VisualizationError) or "visual" in self.operation_name.lower():
            self.error_handler.handle_visualization_error(exc_val, self.context_info)
        elif issubclass(exc_type, ConfigurationError) or "config" in self.operation_name.lower():
            self.error_handler.handle_configuration_error(exc_val, self.context_info)
        else:
            # Generic error handling
            self.logger.error(f"Error in {self.operation_name}: {str(exc_val)}")
            self.logger.debug(f"Stack trace: {traceback.format_exc()}")
        
        # Don't suppress the exception
        return False


# Global error handler instance
error_handler = ErrorHandler()


# Convenience decorators
def retry_on_exception(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    jitter: bool = True
) -> Callable:
    """Convenience decorator for retry logic"""
    return error_handler.retry(
        max_retries=max_retries,
        base_delay=base_delay,
        backoff_factor=backoff_factor,
        retryable_exceptions=retryable_exceptions,
        jitter=jitter
    )


def safe_execute(
    error_message: str = "Operation failed",
    raise_exception: bool = True,
    error_type: str = "general"
) -> Callable:
    """
    Decorator for safely executing functions with standardized error handling
    
    Args:
        error_message: Message to log on failure
        raise_exception: Whether to re-raise the exception after handling
        error_type: Type of error for specialized handling ('api', 'data', 'model', 'viz', 'config')
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function context for better error reporting
                caller_frame = inspect.currentframe().f_back
                caller_info = ""
                if caller_frame:
                    caller_info = f" called from {caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
                
                # Extract context information from args if first arg is self
                context_info = {}
                if args and hasattr(args[0], '__dict__'):
                    # Extract non-private attributes from self
                    for key, value in args[0].__dict__.items():
                        if not key.startswith('_') and not callable(value):
                            context_info[key] = value
                
                # Handle based on error type
                if error_type == 'api':
                    error_handler.handle_api_error(e, context_info)
                elif error_type == 'data':
                    error_handler.handle_data_validation_error(e, context_info)
                elif error_type == 'model':
                    error_handler.handle_model_error(e, context_info)
                elif error_type == 'viz':
                    error_handler.handle_visualization_error(e, context_info)
                elif error_type == 'config':
                    error_handler.handle_configuration_error(e, context_info)
                else:
                    # Generic error handling
                    logger = get_main_logger()
                    logger.error(f"{error_message}: {str(e)}{caller_info}")
                    logger.debug(f"Stack trace: {traceback.format_exc()}")
                
                if raise_exception:
                    raise
                return None
        return wrapper
    return decorator


def log_execution_time(func):
    """
    Decorator to log function execution time
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_main_logger()
        start_time = time.time()
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper


def validate_input(validation_func):
    """
    Decorator to validate function inputs
    
    Args:
        validation_func: Function that validates inputs and returns (is_valid, error_message)
        
    Returns:
        Decorated function with input validation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            is_valid, error_message = validation_func(*args, **kwargs)
            if not is_valid:
                logger = get_main_logger()
                logger.error(f"Input validation failed for {func.__name__}: {error_message}")
                raise ValueError(f"Invalid input: {error_message}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
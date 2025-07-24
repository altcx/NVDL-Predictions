"""
Error handling integration module for NVDL Stock Predictor
Provides centralized error handling and recovery mechanisms for all components
"""
import os
import sys
import time
import traceback
import threading
import signal
import psutil
import logging
from typing import Dict, Any, Optional, Callable, List, Type, Union
from datetime import datetime

from utils.error_handler import (
    ErrorHandler, RetryableError, APIConnectionError, DataValidationError, 
    ModelTrainingError, ConfigurationError, VisualizationError, ResourceNotFoundError,
    DataProcessingError
)

# Define these locally if not available in error_handler
class TimeoutError(RetryableError):
    """Error raised when an operation times out"""
    pass

class RateLimitError(RetryableError):
    """Error raised when API rate limit is exceeded"""
    pass

class AuthenticationError(Exception):
    """Error raised when authentication fails"""
    pass

class DataIntegrityError(DataValidationError):
    """Error raised when data integrity checks fail"""
    pass

class ModelConvergenceError(ModelTrainingError):
    """Error raised when model fails to converge"""
    pass

class InsufficientDataError(DataValidationError):
    """Error raised when there is not enough data for an operation"""
    pass
from utils.logger import (
    get_main_logger, get_data_logger, get_model_logger, get_evaluation_logger,
    get_trading_logger, get_visualization_logger, get_progress_logger,
    LogContext, log_memory_usage, get_recent_errors
)
from config import config

# Get error handler instance
error_handler = ErrorHandler()


def setup_error_handling():
    """
    Set up comprehensive error handling for the application
    Registers signal handlers, cleanup handlers, and configures error thresholds
    """
    logger = get_main_logger()
    logger.info("Setting up comprehensive error handling")
    
    # Register cleanup handlers for different components
    register_component_cleanup_handlers()
    
    # Configure error thresholds based on config
    configure_error_thresholds()
    
    # Set up system monitoring
    setup_system_monitoring()
    
    logger.info("Error handling setup complete")


def register_component_cleanup_handlers():
    """Register cleanup handlers for different components"""
    logger = get_main_logger()
    
    # Data component cleanup
    def data_cleanup():
        data_logger = get_data_logger()
        data_logger.info("Cleaning up data resources")
        # Close any open file handles, database connections, etc.
    
    error_handler.register_cleanup_handler(data_cleanup, "Data component cleanup")
    
    # Model component cleanup
    def model_cleanup():
        model_logger = get_model_logger()
        model_logger.info("Cleaning up model resources")
        # Release GPU memory, close TensorFlow sessions, etc.
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            model_logger.info("TensorFlow session cleared")
        except (ImportError, AttributeError):
            pass
    
    error_handler.register_cleanup_handler(model_cleanup, "Model component cleanup")
    
    # Visualization component cleanup
    def visualization_cleanup():
        viz_logger = get_visualization_logger()
        viz_logger.info("Cleaning up visualization resources")
        # Close plot figures, release memory, etc.
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
            viz_logger.info("Matplotlib figures closed")
        except (ImportError, AttributeError):
            pass
    
    error_handler.register_cleanup_handler(visualization_cleanup, "Visualization component cleanup")
    
    # Temporary files cleanup
    def temp_files_cleanup():
        logger.info("Cleaning up temporary files")
        temp_dirs = ['./tmp', './cache']
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    for filename in os.listdir(temp_dir):
                        file_path = os.path.join(temp_dir, filename)
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    logger.info(f"Cleaned up {temp_dir} directory")
                except Exception as e:
                    logger.warning(f"Error cleaning up {temp_dir}: {str(e)}")
    
    error_handler.register_cleanup_handler(temp_files_cleanup, "Temporary files cleanup")
    
    logger.info("Component cleanup handlers registered")


def configure_error_thresholds():
    """Configure error thresholds based on config"""
    # These thresholds determine how many errors of each type are allowed
    # before the system should gracefully exit
    error_handler._error_thresholds = {
        'api': config.MAX_API_ERRORS if hasattr(config, 'MAX_API_ERRORS') else 5,
        'data': config.MAX_DATA_ERRORS if hasattr(config, 'MAX_DATA_ERRORS') else 3,
        'model': config.MAX_MODEL_ERRORS if hasattr(config, 'MAX_MODEL_ERRORS') else 3,
        'visualization': config.MAX_VIZ_ERRORS if hasattr(config, 'MAX_VIZ_ERRORS') else 5,
        'configuration': config.MAX_CONFIG_ERRORS if hasattr(config, 'MAX_CONFIG_ERRORS') else 1,
        'system': config.MAX_SYSTEM_ERRORS if hasattr(config, 'MAX_SYSTEM_ERRORS') else 2
    }
    
    get_main_logger().info(f"Error thresholds configured: {error_handler._error_thresholds}")


def setup_system_monitoring():
    """Set up system resource monitoring"""
    logger = get_main_logger()
    
    # Only set up if psutil is available
    if not 'psutil' in sys.modules:
        logger.warning("psutil not available, system monitoring disabled")
        return
    
    # Monitor system resources periodically
    def monitor_system_resources():
        """Monitor system resources and log if thresholds are exceeded"""
        try:
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Log if thresholds are exceeded
            if memory_percent > 90:
                logger.warning(f"High memory usage: {memory_percent}%")
            
            if cpu_percent > 90:
                logger.warning(f"High CPU usage: {cpu_percent}%")
            
            if disk_percent > 90:
                logger.warning(f"High disk usage: {disk_percent}%")
            
            # Log detailed info at debug level
            logger.debug(f"System resources: Memory={memory_percent}%, CPU={cpu_percent}%, Disk={disk_percent}%")
            
        except Exception as e:
            logger.error(f"Error monitoring system resources: {str(e)}")
    
    # Start monitoring in a separate thread
    def start_monitoring():
        while True:
            try:
                monitor_system_resources()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"System monitoring error: {str(e)}")
                time.sleep(300)  # Retry after 5 minutes on error
    
    # Start monitoring thread
    monitoring_thread = threading.Thread(
        target=start_monitoring,
        daemon=True,
        name="SystemMonitor"
    )
    monitoring_thread.start()
    
    logger.info("System resource monitoring started")


def handle_api_failure(func):
    """
    Decorator for handling API failures with retry logic
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with API failure handling
    """
    def wrapper(*args, **kwargs):
        logger = get_data_logger()
        max_retries = config.MAX_RETRIES
        base_delay = config.RETRY_DELAY
        backoff_factor = config.RETRY_BACKOFF
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt}/{max_retries} for {func.__name__}")
                
                return func(*args, **kwargs)
                
            except (APIConnectionError, TimeoutError, RateLimitError) as e:
                if attempt < max_retries:
                    delay = base_delay * (backoff_factor ** attempt)
                    logger.warning(f"API error: {str(e)}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"API error: {str(e)}. Max retries ({max_retries}) exceeded.")
                    error_handler.handle_api_error(e, {
                        "function": func.__name__,
                        "args": str(args),
                        "kwargs": str(kwargs)
                    })
                    raise
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                raise
    
    return wrapper


def handle_data_validation(func):
    """
    Decorator for handling data validation errors
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with data validation error handling
    """
    def wrapper(*args, **kwargs):
        logger = get_data_logger()
        
        try:
            return func(*args, **kwargs)
        except DataValidationError as e:
            logger.error(f"Data validation error: {str(e)}")
            error_handler.handle_data_validation_error(e, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper


def handle_model_training(func):
    """
    Decorator for handling model training errors
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with model training error handling
    """
    def wrapper(*args, **kwargs):
        logger = get_model_logger()
        
        try:
            # Log memory usage before training
            log_memory_usage(logger, "Memory usage before model training")
            
            # Start progress tracking
            progress = get_progress_logger("models", 100)
            progress.start(f"Training {func.__name__}")
            
            # Execute function with progress tracking
            result = func(*args, **kwargs)
            
            # Complete progress tracking
            progress.complete(f"Training {func.__name__}")
            
            # Log memory usage after training
            log_memory_usage(logger, "Memory usage after model training")
            
            return result
            
        except ModelTrainingError as e:
            logger.error(f"Model training error: {str(e)}")
            error_handler.handle_model_error(e, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs),
                "model_type": getattr(e, "model_type", "unknown"),
                "epoch": getattr(e, "epoch", "unknown")
            })
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            # Convert to ModelTrainingError for consistent handling
            model_error = ModelTrainingError(
                f"Training failed: {str(e)}",
                model_type=func.__name__
            )
            error_handler.handle_model_error(model_error, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            raise model_error from e
    
    return wrapper


def handle_visualization(func):
    """
    Decorator for handling visualization errors
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with visualization error handling
    """
    def wrapper(*args, **kwargs):
        logger = get_visualization_logger()
        
        try:
            return func(*args, **kwargs)
        except VisualizationError as e:
            logger.error(f"Visualization error: {str(e)}")
            error_handler.handle_visualization_error(e, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            # Convert to VisualizationError for consistent handling
            viz_error = VisualizationError(f"Visualization failed: {str(e)}")
            error_handler.handle_visualization_error(viz_error, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            raise viz_error from e
    
    return wrapper


def log_execution_with_progress(total_steps=100):
    """
    Decorator for logging function execution with progress tracking
    
    Args:
        total_steps: Total number of steps for progress tracking
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get appropriate logger based on function module
            if 'data' in func.__module__:
                logger = get_data_logger()
                component = "data"
            elif 'model' in func.__module__:
                logger = get_model_logger()
                component = "models"
            elif 'visualization' in func.__module__:
                logger = get_visualization_logger()
                component = "visualization"
            else:
                logger = get_main_logger()
                component = "main"
            
            # Create progress logger
            progress = get_progress_logger(component, total_steps)
            
            # Start tracking
            progress.start(f"Executing {func.__name__}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Complete tracking
                progress.complete(f"Completed {func.__name__}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


def monitor_system_resources(logger=None):
    """
    Monitor system resources and log usage
    
    Args:
        logger: Logger instance (uses main logger if None)
    """
    if logger is None:
        logger = get_main_logger()
    
    try:
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024 ** 3)
        memory_total_gb = memory.total / (1024 ** 3)
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)
        
        # Log resource usage
        logger.info(f"System resources: "
                   f"Memory: {memory_percent}% ({memory_used_gb:.1f}/{memory_total_gb:.1f} GB), "
                   f"CPU: {cpu_percent}%, "
                   f"Disk: {disk_percent}% ({disk_used_gb:.1f}/{disk_total_gb:.1f} GB)")
        
        # Check for critical resource usage
        if memory_percent > 95 or cpu_percent > 95 or disk_percent > 95:
            logger.critical("Critical resource usage detected!")
            
            # Get process information
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / (1024 ** 3)
            process_cpu = process.cpu_percent(interval=1)
            
            logger.critical(f"Current process using {process_memory:.2f} GB memory and {process_cpu}% CPU")
            
            # List top processes by memory usage
            top_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    top_processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # Sort by memory usage and get top 5
            top_processes = sorted(top_processes, key=lambda x: x['memory_percent'], reverse=True)[:5]
            
            logger.critical("Top memory-consuming processes:")
            for proc in top_processes:
                logger.critical(f"PID: {proc['pid']}, Name: {proc['name']}, "
                              f"Memory: {proc['memory_percent']:.1f}%, CPU: {proc['cpu_percent']:.1f}%")
    
    except Exception as e:
        logger.error(f"Error monitoring system resources: {str(e)}")


def create_error_report(include_recent_logs=True):
    """
    Create comprehensive error report for diagnostics
    
    Args:
        include_recent_logs: Whether to include recent logs in the report
        
    Returns:
        Dictionary with error report information
    """
    logger = get_main_logger()
    
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "error_counts": error_handler._error_counts.copy(),
            "error_thresholds": error_handler._error_thresholds.copy(),
            "system_info": {}
        }
        
        # Add system information
        try:
            import platform
            report["system_info"]["python_version"] = platform.python_version()
            report["system_info"]["platform"] = platform.platform()
            report["system_info"]["processor"] = platform.processor()
            
            # Add memory information if psutil is available
            if 'psutil' in sys.modules:
                memory = psutil.virtual_memory()
                report["system_info"]["memory_total_gb"] = memory.total / (1024 ** 3)
                report["system_info"]["memory_available_gb"] = memory.available / (1024 ** 3)
                report["system_info"]["memory_percent"] = memory.percent
                
                # Add CPU information
                report["system_info"]["cpu_percent"] = psutil.cpu_percent(interval=1)
                report["system_info"]["cpu_count"] = psutil.cpu_count()
                
                # Add disk information
                disk = psutil.disk_usage('/')
                report["system_info"]["disk_total_gb"] = disk.total / (1024 ** 3)
                report["system_info"]["disk_free_gb"] = disk.free / (1024 ** 3)
                report["system_info"]["disk_percent"] = disk.percent
        except Exception as e:
            logger.warning(f"Error getting system information: {str(e)}")
        
        # Add recent errors
        if include_recent_logs:
            report["recent_errors"] = get_recent_errors(count=50)
        
        return report
        
    except Exception as e:
        logger.error(f"Error creating error report: {str(e)}")
        return {"error": str(e)}


# Don't initialize error handling when module is imported
# This will be called explicitly from main.py
# setup_error_handling()
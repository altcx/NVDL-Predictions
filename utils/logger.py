"""
Logging configuration for NVDL Stock Predictor
Provides structured logging with component-specific loggers and progress tracking
"""
import logging
import logging.handlers
import os
import sys
import time
import threading
from typing import Optional, Dict, Any, List, Callable
from config import config


class ProgressLogger:
    """
    Logger extension for tracking progress of long-running operations
    Provides methods for logging progress updates with percentage completion
    """
    
    def __init__(self, logger: logging.Logger, total_steps: int = 100):
        """
        Initialize progress logger
        
        Args:
            logger: Base logger to use for logging
            total_steps: Total number of steps in the operation
        """
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.last_update_time = 0
        self.min_update_interval = 1.0  # Minimum seconds between updates
        self.checkpoints = {}
        self.milestone_percentages = [25, 50, 75, 90, 100]
        self.last_milestone = 0
    
    def start(self, operation_name: str) -> None:
        """
        Start progress tracking for an operation
        
        Args:
            operation_name: Name of the operation being tracked
        """
        self.start_time = time.time()
        self.current_step = 0
        self.last_milestone = 0
        self.checkpoints = {'start': self.start_time}
        self.logger.info(f"Starting {operation_name} (0% complete)")
    
    def update(self, step: int, message: str = "") -> None:
        """
        Update progress with current step
        
        Args:
            step: Current step number
            message: Optional message to include with the update
        """
        self.current_step = step
        
        # Throttle updates to avoid log flooding
        current_time = time.time()
        if current_time - self.last_update_time < self.min_update_interval:
            return
            
        self.last_update_time = current_time
        
        if self.total_steps > 0:
            percentage = min(100, int((step / self.total_steps) * 100))
            elapsed = current_time - self.start_time
            
            # Check if we've hit a milestone
            for milestone in self.milestone_percentages:
                if percentage >= milestone and self.last_milestone < milestone:
                    self.last_milestone = milestone
                    self.checkpoints[f'{milestone}%'] = current_time
                    self.logger.info(f"Milestone: {milestone}% complete after {elapsed:.1f}s")
            
            if percentage > 0:
                estimated_total = elapsed / (percentage / 100)
                remaining = estimated_total - elapsed
                
                # Format the remaining time in a more human-readable format
                if remaining < 60:
                    remaining_str = f"{remaining:.1f}s"
                elif remaining < 3600:
                    remaining_str = f"{remaining/60:.1f}m"
                else:
                    remaining_str = f"{remaining/3600:.1f}h"
                
                if message:
                    self.logger.info(f"Progress: {percentage}% complete - {message} "
                                    f"(Est. remaining: {remaining_str})")
                else:
                    self.logger.info(f"Progress: {percentage}% complete "
                                    f"(Est. remaining: {remaining_str})")
    
    def increment(self, increment: int = 1, message: str = "") -> None:
        """
        Increment progress by specified amount
        
        Args:
            increment: Number of steps to increment
            message: Optional message to include with the update
        """
        self.update(self.current_step + increment, message)
    
    def checkpoint(self, checkpoint_name: str) -> None:
        """
        Record a named checkpoint in the progress
        
        Args:
            checkpoint_name: Name of the checkpoint
        """
        current_time = time.time()
        self.checkpoints[checkpoint_name] = current_time
        
        if self.start_time is not None:
            elapsed = current_time - self.start_time
            if self.total_steps > 0:
                percentage = min(100, int((self.current_step / self.total_steps) * 100))
                self.logger.info(f"Checkpoint '{checkpoint_name}' at {percentage}% ({elapsed:.2f}s elapsed)")
            else:
                self.logger.info(f"Checkpoint '{checkpoint_name}' ({elapsed:.2f}s elapsed)")
    
    def complete(self, operation_name: str) -> None:
        """
        Mark operation as complete
        
        Args:
            operation_name: Name of the completed operation
        """
        if self.start_time is not None:
            current_time = time.time()
            elapsed = current_time - self.start_time
            self.checkpoints['complete'] = current_time
            
            # Log completion with detailed timing information
            self.logger.info(f"Completed {operation_name} in {elapsed:.2f} seconds")
            
            # Log checkpoint intervals if there are multiple checkpoints
            if len(self.checkpoints) > 2:  # More than just start and complete
                self.logger.debug("Checkpoint intervals:")
                last_time = self.start_time
                last_name = 'start'
                
                for name, timestamp in sorted(self.checkpoints.items(), key=lambda x: x[1]):
                    if name != 'start':
                        interval = timestamp - last_time
                        self.logger.debug(f"  {last_name} → {name}: {interval:.2f}s")
                        last_time = timestamp
                        last_name = name
            
            self.start_time = None
    
    def get_performance_summary(self) -> dict:
        """
        Get performance summary with timing information
        
        Returns:
            Dictionary with timing information
        """
        if not self.checkpoints:
            return {'error': 'No checkpoints recorded'}
        
        summary = {
            'total_steps': self.total_steps,
            'completed_steps': self.current_step,
            'checkpoints': {}
        }
        
        start_time = self.checkpoints.get('start', 0)
        
        for name, timestamp in self.checkpoints.items():
            if name != 'start':
                summary['checkpoints'][name] = {
                    'elapsed_from_start': timestamp - start_time,
                }
        
        if 'complete' in self.checkpoints:
            summary['total_time'] = self.checkpoints['complete'] - start_time
        
        return summary


class StructuredLogger:
    """Structured logging utility for different components"""
    
    _loggers = {}
    _progress_loggers = {}
    _memory_handlers = {}
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """Get or create a logger for a specific component"""
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatters
        standard_formatter = logging.Formatter(
            fmt=config.LOG_FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # More detailed formatter for debug logs
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(standard_formatter)
        logger.addHandler(console_handler)
        
        # File handler for main log
        log_filename = log_file or config.LOG_FILE
        file_handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        file_handler.setFormatter(standard_formatter)
        logger.addHandler(file_handler)
        
        # Component-specific file handler with more detailed formatting
        if name != 'main':
            component_log = f"{name}_{config.LOG_FILE}"
            component_handler = logging.handlers.RotatingFileHandler(
                component_log,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            component_handler.setLevel(logging.DEBUG)
            component_handler.setFormatter(detailed_formatter)
            logger.addHandler(component_handler)
        
        # In-memory handler for recent logs (useful for error reporting)
        memory_handler = cls._create_memory_handler()
        logger.addHandler(memory_handler)
        cls._memory_handlers[name] = memory_handler
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def _create_memory_handler(cls, capacity: int = 1000):
        """Create an in-memory handler for recent logs"""
        class MemoryHandler(logging.Handler):
            def __init__(self, capacity):
                super().__init__()
                self.capacity = capacity
                self.buffer = []
                self.setLevel(logging.DEBUG)
                self.setFormatter(logging.Formatter(
                    fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
            
            def emit(self, record):
                self.buffer.append(self.format(record))
                if len(self.buffer) > self.capacity:
                    self.buffer.pop(0)
            
            def get_logs(self, level=None, count=None):
                if level:
                    filtered = [log for log in self.buffer if level.upper() in log]
                    return filtered[-count:] if count else filtered
                return self.buffer[-count:] if count else self.buffer.copy()
        
        return MemoryHandler(capacity)
    
    @classmethod
    def get_recent_logs(cls, component: str = None, level: str = None, count: int = 100) -> List[str]:
        """
        Get recent logs from memory buffer
        
        Args:
            component: Component name (or None for all components)
            level: Log level to filter (e.g., 'ERROR', 'WARNING')
            count: Maximum number of logs to return
            
        Returns:
            List of recent log entries
        """
        if component and component in cls._memory_handlers:
            return cls._memory_handlers[component].get_logs(level, count)
        
        # Combine logs from all components
        all_logs = []
        for handler in cls._memory_handlers.values():
            all_logs.extend(handler.get_logs(level))
        
        # Sort by timestamp (assuming standard format)
        all_logs.sort()
        return all_logs[-count:] if count else all_logs
    
    @classmethod
    def get_progress_logger(cls, name: str, total_steps: int = 100) -> ProgressLogger:
        """
        Get or create a progress logger for tracking long operations
        
        Args:
            name: Component name
            total_steps: Total number of steps in the operation
            
        Returns:
            ProgressLogger instance
        """
        logger = cls.get_logger(name)
        
        if name not in cls._progress_loggers:
            cls._progress_loggers[name] = ProgressLogger(logger, total_steps)
        else:
            # Update total steps if different
            cls._progress_loggers[name].total_steps = total_steps
            
        return cls._progress_loggers[name]
    
    @classmethod
    def setup_logging(cls):
        """Initialize logging configuration"""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(config.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        
        # Add a handler for uncaught exceptions
        sys.excepthook = cls._handle_uncaught_exception
        
        # Log system information
        main_logger = cls.get_logger('main')
        cls._log_system_info(main_logger)
    
    @staticmethod
    def _log_system_info(logger):
        """Log system information for diagnostics"""
        import platform
        
        logger.info("System information:")
        logger.info(f"  Python version: {platform.python_version()}")
        logger.info(f"  Platform: {platform.platform()}")
        logger.info(f"  Processor: {platform.processor()}")
        
        # Log memory information if psutil is available
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"  Memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
        except ImportError:
            pass
        
        # Log environment variables (excluding sensitive information)
        logger.debug("Environment variables:")
        for key, value in os.environ.items():
            if not any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                logger.debug(f"  {key}: {value}")
            else:
                logger.debug(f"  {key}: [REDACTED]")
    
    @classmethod
    def _handle_uncaught_exception(cls, exc_type, exc_value, exc_traceback):
        """
        Handler for uncaught exceptions
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # Call the default excepthook for KeyboardInterrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = logging.getLogger('main')
        
        # Get information about where the error occurred
        try:
            tb_info = traceback.extract_tb(exc_traceback)
            filename, line, func, text = tb_info[-1]
            error_location = f"in {func} at {filename}:{line}"
        except:
            error_location = "unknown location"
        
        logger.critical(f"Uncaught exception ({exc_type.__name__}): {str(exc_value)} {error_location}", 
                       exc_info=(exc_type, exc_value, exc_traceback))
        
        # Log recent errors that might be related
        recent_errors = cls.get_recent_logs(level='ERROR', count=5)
        if recent_errors:
            logger.critical("Recent errors that might be related:")
            for error in recent_errors:
                logger.critical(f"  {error}")
        
        # Call the default excepthook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


# Context manager for logging execution blocks
class LogContext:
    """
    Context manager for logging execution blocks with timing
    
    Example:
        with LogContext(logger, "Data processing"):
            process_data()
    """
    
    def __init__(self, logger, operation_name, log_level=logging.INFO):
        """
        Initialize log context
        
        Args:
            logger: Logger instance
            operation_name: Name of the operation being performed
            log_level: Logging level for messages
        """
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        """Enter context manager and log start"""
        self.start_time = time.time()
        self.logger.log(self.log_level, f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and log completion or error"""
        elapsed = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(self.log_level, f"Completed: {self.operation_name} in {elapsed:.2f}s")
        else:
            self.logger.error(f"Failed: {self.operation_name} after {elapsed:.2f}s - {exc_type.__name__}: {exc_val}")
        
        # Don't suppress the exception
        return False


# Function to get recent errors for diagnostics
def get_recent_errors(count: int = 10) -> List[str]:
    """
    Get recent error logs for diagnostics
    
    Args:
        count: Maximum number of error logs to return
        
    Returns:
        List of recent error log entries
    """
    return StructuredLogger.get_recent_logs(level='ERROR', count=count)


# Function to log memory usage
def log_memory_usage(logger: logging.Logger, message: str = "Current memory usage"):
    """
    Log current memory usage
    
    Args:
        logger: Logger instance
        message: Message prefix
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        # Convert to MB for readability
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024)
        
        logger.info(f"{message}: RSS={rss_mb:.1f}MB, VMS={vms_mb:.1f}MB")
    except ImportError:
        logger.debug(f"{message}: psutil not available")


# Component-specific logger functions
def get_data_logger() -> logging.Logger:
    """Get logger for data collection and preprocessing"""
    return StructuredLogger.get_logger('data')


def get_model_logger() -> logging.Logger:
    """Get logger for model training and prediction"""
    return StructuredLogger.get_logger('models')


def get_evaluation_logger() -> logging.Logger:
    """Get logger for model evaluation and metrics"""
    return StructuredLogger.get_logger('evaluation')


def get_trading_logger() -> logging.Logger:
    """Get logger for trading simulation"""
    return StructuredLogger.get_logger('trading')


def get_visualization_logger() -> logging.Logger:
    """Get logger for visualization generation"""
    return StructuredLogger.get_logger('visualization')


def get_main_logger() -> logging.Logger:
    """Get main application logger"""
    return StructuredLogger.get_logger('main')


def get_config_logger() -> logging.Logger:
    """Get logger for configuration management"""
    return StructuredLogger.get_logger('config')


def get_progress_logger(component: str, total_steps: int = 100) -> ProgressLogger:
    """
    Get progress logger for tracking long operations
    
    Args:
        component: Component name ('data', 'models', etc.)
        total_steps: Total number of steps in the operation
        
    Returns:
        ProgressLogger instance
    """
    return StructuredLogger.get_progress_logger(component, total_steps)


# Initialize logging on import
StructuredLogger.setup_logging()
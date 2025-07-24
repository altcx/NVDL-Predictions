"""
Configuration file for pytest
Sets up fixtures and plugins for continuous testing
"""
import pytest
import os
import sys
import logging
import tempfile
import shutil
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test fixtures
from tests.test_fixtures import (
    create_sample_ohlcv_data, create_sample_processed_data,
    create_train_test_split, create_sample_model_results,
    create_sample_trading_results, create_sample_evaluation_results,
    mock_tensorflow, mock_statsmodels, mock_plotly
)


def pytest_configure(config):
    """Configure pytest"""
    # Set up logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_run.log', mode='w')
        ]
    )
    
    # Log test session start
    logging.info(f"Test session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"pytest version: {pytest.__version__}")
    
    # Check for required packages
    required_packages = ['numpy', 'pandas', 'pytest']
    optional_packages = ['tensorflow', 'statsmodels', 'plotly', 'psutil']
    
    logging.info("Checking required packages:")
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            logging.info(f"  {package}: {version}")
        except ImportError:
            logging.error(f"  {package}: Not found (REQUIRED)")
            pytest.exit(f"Required package {package} not found")
    
    logging.info("Checking optional packages:")
    for package in optional_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            logging.info(f"  {package}: {version}")
        except ImportError:
            logging.warning(f"  {package}: Not found (optional)")


def pytest_report_header(config):
    """Add information to test report header"""
    return "NVDL Stock Predictor Test Suite"


def pytest_runtest_setup(item):
    """Set up test environment before each test"""
    # Log test start
    logging.info(f"Running test: {item.name}")


def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test"""
    # Log test end
    logging.info(f"Completed test: {item.name}")


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp(prefix="nvdl_test_")
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration"""
    return {
        'test_size': 0.2,
        'random_seed': 42,
        'lstm_sequence_length': 10,
        'lstm_units': 20,
        'lstm_dropout': 0.2,
        'lstm_epochs': 2,
        'arima_max_p': 2,
        'arima_max_d': 1,
        'arima_max_q': 2,
        'initial_capital': 10000.0
    }


@pytest.fixture(autouse=True)
def run_around_tests():
    """Setup and teardown for each test"""
    # Setup
    start_time = datetime.now()
    
    yield  # This is where the test runs
    
    # Teardown
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    if duration > 5.0:
        logging.warning(f"Test took {duration:.2f} seconds to run")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add summary information at the end of test session"""
    # Log test session summary
    passed = terminalreporter.stats.get('passed', [])
    failed = terminalreporter.stats.get('failed', [])
    skipped = terminalreporter.stats.get('skipped', [])
    errors = terminalreporter.stats.get('error', [])
    
    logging.info(f"Test session completed with exit status: {exitstatus}")
    logging.info(f"Total tests: {len(passed) + len(failed) + len(skipped) + len(errors)}")
    logging.info(f"  Passed: {len(passed)}")
    logging.info(f"  Failed: {len(failed)}")
    logging.info(f"  Skipped: {len(skipped)}")
    logging.info(f"  Errors: {len(errors)}")
    
    # Log failed tests
    if failed:
        logging.error("Failed tests:")
        for test in failed:
            logging.error(f"  {test.nodeid}")
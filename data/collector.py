"""
Data collection module for NVDL Stock Predictor
Handles API integration with Alpaca Markets for historical stock data
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError
from utils.logger import get_data_logger, LogContext
from utils.error_handler import (
    ErrorHandler, retry_on_exception, safe_execute, 
    APIConnectionError, DataValidationError, ErrorContext
)
# Define these locally if not available
class TimeoutError(Exception):
    """Error raised when an operation times out"""
    pass

class InsufficientDataError(DataValidationError):
    """Error raised when there is not enough data for an operation"""
    pass

class DataIntegrityError(DataValidationError):
    """Error raised when data integrity checks fail"""
    pass
from config import config

# Get error handler instance
error_handler = ErrorHandler()


class DataCollector:
    """
    Data collection class with Alpaca Markets API authentication
    Handles fetching historical data with proper error handling and rate limiting
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize DataCollector with API credentials
        
        Args:
            api_key: Alpaca API key (defaults to config)
            secret_key: Alpaca secret key (defaults to config)
        """
        self.logger = get_data_logger()
        self.api_key = api_key or config.ALPACA_API_KEY
        self.secret_key = secret_key or config.ALPACA_SECRET_KEY
        
        if not self.api_key or not self.secret_key:
            raise ValueError("API key and secret key must be provided")
        
        # Initialize Alpaca client
        try:
            self.client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            self.logger.info("Successfully initialized Alpaca Markets API client")
        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {str(e)}")
            raise
        
        # Rate limiting parameters
        self.max_retries = 3
        self.base_delay = 1.0  # Base delay in seconds
        self.backoff_factor = 2.0
    
    @retry_on_exception(
        max_retries=3,
        base_delay=1.0,
        backoff_factor=2.0,
        retryable_exceptions=[APIError, TimeoutError],
        error_message="Failed to fetch historical data"
    )
    def fetch_historical_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        timeframe: TimeFrame = TimeFrame.Day
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'NVDL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Data timeframe (default: daily)
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
            
        Raises:
            ValueError: If date format is invalid
            APIConnectionError: If API request fails after retries
            TimeoutError: If request times out
            RateLimitError: If API rate limit is exceeded
        """
        with LogContext(self.logger, f"Fetching historical data for {symbol}"):
            self.logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            # Validate date format
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError as e:
                self.logger.error(f"Invalid date format: {str(e)}")
                raise ValueError(f"Date must be in YYYY-MM-DD format: {str(e)}")
            
            if start_dt >= end_dt:
                raise ValueError("Start date must be before end date")
            
            # Create request
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=start_dt,
                end=end_dt
            )
            
            # API call context
            api_info = {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "timeframe": str(timeframe)
            }
            
            with ErrorContext(error_handler, "Alpaca API request", **api_info):
                try:
                    # Make API call with timeout
                    bars = error_handler._run_with_timeout(
                        lambda: self.client.get_stock_bars(request_params),
                        timeout=30.0  # 30 second timeout
                    )
                    
                    if not bars.data:
                        self.logger.warning(f"No data returned for {symbol}")
                        return pd.DataFrame()
                    
                    # Convert to DataFrame
                    data_list = []
                    for bar in bars.data[symbol]:
                        data_list.append({
                            'timestamp': bar.timestamp,
                            'open': float(bar.open),
                            'high': float(bar.high),
                            'low': float(bar.low),
                            'close': float(bar.close),
                            'volume': int(bar.volume)
                        })
                    
                    df = pd.DataFrame(data_list)
                    if not df.empty:
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                    
                    self.logger.info(f"Successfully fetched {len(df)} data points for {symbol}")
                    return df
                    
                except APIError as e:
                    # Convert to our custom exception types for better handling
                    if "timeout" in str(e).lower():
                        raise TimeoutError(f"API request timed out: {str(e)}")
                    else:
                        raise APIConnectionError(f"API error: {str(e)}")
                        
                except Exception as e:
                    self.logger.error(f"Unexpected error during data fetch: {str(e)}")
                    raise APIConnectionError(f"Failed to fetch data: {str(e)}") from e
    
    def validate_data_completeness(self, data: pd.DataFrame, min_data_points: int = 100) -> bool:
        """
        Validate data completeness and quality
        
        Args:
            data: DataFrame with OHLCV data
            min_data_points: Minimum required data points
            
        Returns:
            True if data passes validation, False otherwise
            
        Raises:
            DataValidationError: If data fails validation checks
            InsufficientDataError: If there are not enough data points
            DataIntegrityError: If data integrity checks fail
        """
        with LogContext(self.logger, "Validating data completeness"):
            data_info = {
                "rows": len(data) if not data.empty else 0,
                "min_required": min_data_points,
                "date_range": f"{data.index.min()} to {data.index.max()}" if not data.empty and len(data) > 0 else "N/A"
            }
            
            with ErrorContext(error_handler, "Data validation", **data_info):
                if data.empty:
                    error_msg = "Data validation failed: DataFrame is empty"
                    self.logger.error(error_msg)
                    raise DataValidationError(error_msg)
                
                if len(data) < min_data_points:
                    error_msg = f"Data validation failed: Only {len(data)} data points, minimum {min_data_points} required"
                    self.logger.error(error_msg)
                    raise InsufficientDataError(error_msg)
                
                # Check for required columns
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    error_msg = f"Data validation failed: Missing columns {missing_columns}"
                    self.logger.error(error_msg)
                    raise DataValidationError(error_msg)
                
                # Check for null values
                null_counts = data.isnull().sum()
                if null_counts.any():
                    self.logger.warning(f"Data contains null values: {null_counts.to_dict()}")
                
                # Check for negative prices or volumes
                if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
                    error_msg = "Data validation failed: Found non-positive prices"
                    self.logger.error(error_msg)
                    raise DataIntegrityError(error_msg)
                
                if (data['volume'] < 0).any():
                    error_msg = "Data validation failed: Found negative volume"
                    self.logger.error(error_msg)
                    raise DataIntegrityError(error_msg)
                
                # Check for logical price relationships
                invalid_ohlc = (
                    (data['high'] < data['low']) |
                    (data['high'] < data['open']) |
                    (data['high'] < data['close']) |
                    (data['low'] > data['open']) |
                    (data['low'] > data['close'])
                )
                
                if invalid_ohlc.any():
                    error_msg = "Data validation failed: Invalid OHLC relationships found"
                    self.logger.error(error_msg)
                    raise DataIntegrityError(error_msg)
                
                self.logger.info(f"Data validation passed: {len(data)} valid data points")
                return True
    
    @safe_execute(error_message="Failed to handle missing data", raise_exception=True)
    def handle_missing_data(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing data using specified method
        
        Args:
            data: DataFrame with potential missing values
            method: Method to handle missing data ('forward_fill', 'drop', 'interpolate')
            
        Returns:
            DataFrame with missing data handled
            
        Raises:
            DataValidationError: If method is invalid or data handling fails
        """
        with LogContext(self.logger, f"Handling missing data using {method}"):
            if data.empty:
                self.logger.warning("Cannot handle missing data: DataFrame is empty")
                return data
            
            # Validate method first
            valid_methods = ['forward_fill', 'drop', 'interpolate']
            if method not in valid_methods:
                error_msg = f"Unknown missing data method: {method}"
                self.logger.error(error_msg)
                raise DataValidationError(f"Invalid method: {method}. Use 'forward_fill', 'drop', or 'interpolate'")
            
            original_length = len(data)
            null_counts = data.isnull().sum()
            
            if not null_counts.any():
                self.logger.info("No missing data found")
                return data
            
            self.logger.info(f"Handling missing data using method: {method}")
            self.logger.info(f"Missing values per column: {null_counts.to_dict()}")
            
            data_info = {
                "rows": len(data),
                "missing_values": null_counts.sum(),
                "method": method
            }
            
            with ErrorContext(error_handler, "Missing data handling", **data_info):
                try:
                    if method == 'forward_fill':
                        data_cleaned = data.ffill()
                        # If still NaN values at the beginning, use backward fill
                        data_cleaned = data_cleaned.bfill()
                        
                    elif method == 'drop':
                        data_cleaned = data.dropna()
                        
                    elif method == 'interpolate':
                        # Use linear interpolation for price data
                        price_columns = ['open', 'high', 'low', 'close']
                        data_cleaned = data.copy()
                        data_cleaned[price_columns] = data_cleaned[price_columns].interpolate(method='linear')
                        # Forward fill volume data
                        data_cleaned['volume'] = data_cleaned['volume'].ffill()
                    
                    # Check if we still have missing values
                    remaining_nulls = data_cleaned.isnull().sum().sum()
                    if remaining_nulls > 0:
                        self.logger.warning(f"Still have {remaining_nulls} missing values after {method}")
                        # Try additional method if primary method didn't work completely
                        if method != 'forward_fill':
                            self.logger.info("Applying forward fill as secondary method")
                            data_cleaned = data_cleaned.ffill().bfill()
                    
                    final_length = len(data_cleaned)
                    self.logger.info(f"Missing data handled: {original_length} -> {final_length} data points")
                    
                    return data_cleaned
                    
                except Exception as e:
                    error_msg = f"Failed to handle missing data: {str(e)}"
                    self.logger.error(error_msg)
                    raise DataValidationError(error_msg) from e
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for the data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with summary statistics
        """
        if data.empty:
            return {"error": "No data available"}
        
        summary = {
            "total_records": len(data),
            "date_range": {
                "start": data.index.min().strftime('%Y-%m-%d'),
                "end": data.index.max().strftime('%Y-%m-%d')
            },
            "price_statistics": {
                "min_close": float(data['close'].min()),
                "max_close": float(data['close'].max()),
                "mean_close": float(data['close'].mean()),
                "std_close": float(data['close'].std())
            },
            "volume_statistics": {
                "min_volume": int(data['volume'].min()),
                "max_volume": int(data['volume'].max()),
                "mean_volume": float(data['volume'].mean())
            },
            "missing_data": data.isnull().sum().to_dict()
        }
        
        return summary
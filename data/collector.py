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
from utils.logger import get_data_logger
from config import config


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
            APIError: If API request fails after retries
        """
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
        
        # Fetch data with retry logic
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"API request attempt {attempt + 1}/{self.max_retries}")
                
                # Make API call
                bars = self.client.get_stock_bars(request_params)
                
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
                self.logger.warning(f"API error on attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (self.backoff_factor ** attempt)
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to fetch data after {self.max_retries} attempts")
                    raise
                    
            except Exception as e:
                self.logger.error(f"Unexpected error during data fetch: {str(e)}")
                raise
    
    def validate_data_completeness(self, data: pd.DataFrame, min_data_points: int = 100) -> bool:
        """
        Validate data completeness and quality
        
        Args:
            data: DataFrame with OHLCV data
            min_data_points: Minimum required data points
            
        Returns:
            True if data passes validation, False otherwise
        """
        if data.empty:
            self.logger.error("Data validation failed: DataFrame is empty")
            return False
        
        if len(data) < min_data_points:
            self.logger.error(f"Data validation failed: Only {len(data)} data points, minimum {min_data_points} required")
            return False
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Data validation failed: Missing columns {missing_columns}")
            return False
        
        # Check for null values
        null_counts = data.isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Data contains null values: {null_counts.to_dict()}")
        
        # Check for negative prices or volumes
        if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
            self.logger.error("Data validation failed: Found non-positive prices")
            return False
        
        if (data['volume'] < 0).any():
            self.logger.error("Data validation failed: Found negative volume")
            return False
        
        # Check for logical price relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            self.logger.error("Data validation failed: Invalid OHLC relationships found")
            return False
        
        self.logger.info(f"Data validation passed: {len(data)} valid data points")
        return True
    
    def handle_missing_data(self, data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing data using specified method
        
        Args:
            data: DataFrame with potential missing values
            method: Method to handle missing data ('forward_fill', 'drop', 'interpolate')
            
        Returns:
            DataFrame with missing data handled
        """
        if data.empty:
            self.logger.warning("Cannot handle missing data: DataFrame is empty")
            return data
        
        # Validate method first
        valid_methods = ['forward_fill', 'drop', 'interpolate']
        if method not in valid_methods:
            self.logger.error(f"Unknown missing data method: {method}")
            raise ValueError(f"Unknown method: {method}. Use 'forward_fill', 'drop', or 'interpolate'")
        
        original_length = len(data)
        null_counts = data.isnull().sum()
        
        if not null_counts.any():
            self.logger.info("No missing data found")
            return data
        
        self.logger.info(f"Handling missing data using method: {method}")
        self.logger.info(f"Missing values per column: {null_counts.to_dict()}")
        
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

        
        final_length = len(data_cleaned)
        self.logger.info(f"Missing data handled: {original_length} -> {final_length} data points")
        
        return data_cleaned
    
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
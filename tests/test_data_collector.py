"""
Unit tests for DataCollector class
Tests API integration and data validation functions
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from alpaca.common.exceptions import APIError
from alpaca.data.timeframe import TimeFrame

from data.collector import DataCollector


class TestDataCollector:
    """Test suite for DataCollector class"""
    
    @pytest.fixture
    def mock_api_credentials(self):
        """Mock API credentials for testing"""
        return {
            'api_key': 'test_api_key',
            'secret_key': 'test_secret_key'
        }
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """Sample OHLCV data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic stock price data
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            open_price = close * (1 + np.random.normal(0, 0.005))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.uniform(10000, 100000))
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    @pytest.fixture
    def data_collector(self, mock_api_credentials):
        """Create DataCollector instance with mocked API client"""
        with patch('data.collector.StockHistoricalDataClient') as mock_client:
            collector = DataCollector(**mock_api_credentials)
            collector.client = Mock()
            return collector
    
    def test_init_with_credentials(self, mock_api_credentials):
        """Test DataCollector initialization with valid credentials"""
        with patch('data.collector.StockHistoricalDataClient'):
            collector = DataCollector(**mock_api_credentials)
            assert collector.api_key == mock_api_credentials['api_key']
            assert collector.secret_key == mock_api_credentials['secret_key']
    
    def test_init_without_credentials(self):
        """Test DataCollector initialization without credentials raises error"""
        with pytest.raises(ValueError, match="API key and secret key must be provided"):
            DataCollector(api_key="", secret_key="")
    
    def test_init_with_config_credentials(self):
        """Test DataCollector initialization using config credentials"""
        with patch('data.collector.config') as mock_config:
            mock_config.ALPACA_API_KEY = 'config_api_key'
            mock_config.ALPACA_SECRET_KEY = 'config_secret_key'
            
            with patch('data.collector.StockHistoricalDataClient'):
                collector = DataCollector()
                assert collector.api_key == 'config_api_key'
                assert collector.secret_key == 'config_secret_key'
    
    def test_fetch_historical_data_success(self, data_collector, sample_ohlcv_data):
        """Test successful historical data fetch"""
        # Mock API response
        mock_bars = Mock()
        mock_bars.data = {'NVDL': []}
        
        # Create mock bar objects
        for _, row in sample_ohlcv_data.iterrows():
            mock_bar = Mock()
            mock_bar.timestamp = row.name
            mock_bar.open = row['open']
            mock_bar.high = row['high']
            mock_bar.low = row['low']
            mock_bar.close = row['close']
            mock_bar.volume = row['volume']
            mock_bars.data['NVDL'].append(mock_bar)
        
        data_collector.client.get_stock_bars.return_value = mock_bars
        
        # Test data fetch
        result = data_collector.fetch_historical_data('NVDL', '2023-01-01', '2023-04-10')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        assert list(result.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert result.index.name == 'timestamp'
    
    def test_fetch_historical_data_invalid_dates(self, data_collector):
        """Test fetch with invalid date formats"""
        with pytest.raises(ValueError, match="Date must be in YYYY-MM-DD format"):
            data_collector.fetch_historical_data('NVDL', 'invalid-date', '2023-12-31')
        
        with pytest.raises(ValueError, match="Start date must be before end date"):
            data_collector.fetch_historical_data('NVDL', '2023-12-31', '2023-01-01')
    
    def test_fetch_historical_data_no_data(self, data_collector):
        """Test fetch when API returns no data"""
        mock_bars = Mock()
        mock_bars.data = {}
        data_collector.client.get_stock_bars.return_value = mock_bars
        
        result = data_collector.fetch_historical_data('INVALID', '2023-01-01', '2023-12-31')
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_fetch_historical_data_api_error_retry(self, data_collector):
        """Test API error handling with retry logic"""
        # Mock API to fail twice then succeed
        mock_bars = Mock()
        mock_bars.data = {'NVDL': []}
        
        data_collector.client.get_stock_bars.side_effect = [
            APIError("Rate limit exceeded"),
            APIError("Temporary error"),
            mock_bars
        ]
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            result = data_collector.fetch_historical_data('NVDL', '2023-01-01', '2023-01-31')
        
        assert isinstance(result, pd.DataFrame)
        assert data_collector.client.get_stock_bars.call_count == 3
    
    def test_fetch_historical_data_max_retries_exceeded(self, data_collector):
        """Test API error when max retries exceeded"""
        data_collector.client.get_stock_bars.side_effect = APIError("Persistent error")
        
        with patch('time.sleep'):  # Mock sleep to speed up test
            with pytest.raises(APIError):
                data_collector.fetch_historical_data('NVDL', '2023-01-01', '2023-01-31')
        
        assert data_collector.client.get_stock_bars.call_count == 3
    
    def test_validate_data_completeness_valid_data(self, data_collector, sample_ohlcv_data):
        """Test data validation with valid data"""
        result = data_collector.validate_data_completeness(sample_ohlcv_data)
        assert result is True
    
    def test_validate_data_completeness_empty_data(self, data_collector):
        """Test data validation with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = data_collector.validate_data_completeness(empty_df)
        assert result is False
    
    def test_validate_data_completeness_insufficient_data(self, data_collector):
        """Test data validation with insufficient data points"""
        small_df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [10000]
        })
        result = data_collector.validate_data_completeness(small_df, min_data_points=100)
        assert result is False
    
    def test_validate_data_completeness_missing_columns(self, data_collector):
        """Test data validation with missing columns"""
        incomplete_df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [101.0, 102.0],
            'close': [100.5, 101.5]
            # Missing 'low' and 'volume' columns
        })
        result = data_collector.validate_data_completeness(incomplete_df)
        assert result is False
    
    def test_validate_data_completeness_negative_prices(self, data_collector):
        """Test data validation with negative prices"""
        invalid_df = pd.DataFrame({
            'open': [100.0, -50.0],  # Negative price
            'high': [101.0, 102.0],
            'low': [99.0, 98.0],
            'close': [100.5, 101.5],
            'volume': [10000, 20000]
        })
        result = data_collector.validate_data_completeness(invalid_df)
        assert result is False
    
    def test_validate_data_completeness_invalid_ohlc(self, data_collector):
        """Test data validation with invalid OHLC relationships"""
        invalid_df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [99.0, 102.0],  # High < Open for first row
            'low': [98.0, 100.0],
            'close': [100.5, 101.5],
            'volume': [10000, 20000]
        })
        result = data_collector.validate_data_completeness(invalid_df)
        assert result is False
    
    def test_handle_missing_data_forward_fill(self, data_collector, sample_ohlcv_data):
        """Test missing data handling with forward fill"""
        # Introduce missing values
        data_with_missing = sample_ohlcv_data.copy()
        data_with_missing.loc[data_with_missing.index[10:15], 'close'] = np.nan
        data_with_missing.loc[data_with_missing.index[20], 'volume'] = np.nan
        
        result = data_collector.handle_missing_data(data_with_missing, method='forward_fill')
        
        assert not result.isnull().any().any()
        assert len(result) == len(data_with_missing)
    
    def test_handle_missing_data_drop(self, data_collector, sample_ohlcv_data):
        """Test missing data handling by dropping rows"""
        # Introduce missing values
        data_with_missing = sample_ohlcv_data.copy()
        data_with_missing.loc[data_with_missing.index[10:15], 'close'] = np.nan
        
        result = data_collector.handle_missing_data(data_with_missing, method='drop')
        
        assert not result.isnull().any().any()
        assert len(result) < len(data_with_missing)
    
    def test_handle_missing_data_interpolate(self, data_collector, sample_ohlcv_data):
        """Test missing data handling with interpolation"""
        # Introduce missing values
        data_with_missing = sample_ohlcv_data.copy()
        data_with_missing.loc[data_with_missing.index[10:12], 'close'] = np.nan
        
        result = data_collector.handle_missing_data(data_with_missing, method='interpolate')
        
        assert not result.isnull().any().any()
        assert len(result) == len(data_with_missing)
    
    def test_handle_missing_data_invalid_method(self, data_collector, sample_ohlcv_data):
        """Test missing data handling with invalid method"""
        with pytest.raises(ValueError, match="Unknown method"):
            data_collector.handle_missing_data(sample_ohlcv_data, method='invalid_method')
    
    def test_handle_missing_data_empty_dataframe(self, data_collector):
        """Test missing data handling with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = data_collector.handle_missing_data(empty_df)
        assert result.empty
    
    def test_handle_missing_data_no_missing_values(self, data_collector, sample_ohlcv_data):
        """Test missing data handling when no missing values exist"""
        result = data_collector.handle_missing_data(sample_ohlcv_data)
        pd.testing.assert_frame_equal(result, sample_ohlcv_data)
    
    def test_get_data_summary_valid_data(self, data_collector, sample_ohlcv_data):
        """Test data summary generation with valid data"""
        summary = data_collector.get_data_summary(sample_ohlcv_data)
        
        assert 'total_records' in summary
        assert 'date_range' in summary
        assert 'price_statistics' in summary
        assert 'volume_statistics' in summary
        assert 'missing_data' in summary
        
        assert summary['total_records'] == len(sample_ohlcv_data)
        assert 'start' in summary['date_range']
        assert 'end' in summary['date_range']
    
    def test_get_data_summary_empty_data(self, data_collector):
        """Test data summary generation with empty data"""
        empty_df = pd.DataFrame()
        summary = data_collector.get_data_summary(empty_df)
        
        assert 'error' in summary
        assert summary['error'] == "No data available"


class TestDataCollectorIntegration:
    """Integration tests for DataCollector with real-like scenarios"""
    
    def test_full_data_pipeline(self):
        """Test complete data collection and processing pipeline"""
        # This would be a more comprehensive test with mocked API responses
        # that simulate real-world scenarios
        pass
    
    def test_rate_limiting_behavior(self):
        """Test rate limiting and retry behavior under load"""
        # This would test the exponential backoff implementation
        pass
    
    def test_data_quality_edge_cases(self):
        """Test data quality validation with edge cases"""
        # This would test various edge cases in market data
        pass


if __name__ == '__main__':
    pytest.main([__file__])
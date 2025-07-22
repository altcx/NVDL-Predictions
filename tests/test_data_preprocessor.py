"""
Unit tests for DataPreprocessor class
Tests feature engineering and data preparation functions
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

from data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""
    
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
    def preprocessor(self):
        """Create DataPreprocessor instance for testing"""
        return DataPreprocessor()
    
    def test_init(self):
        """Test DataPreprocessor initialization"""
        preprocessor = DataPreprocessor()
        assert isinstance(preprocessor.scaler, StandardScaler)
        assert preprocessor.is_fitted is False
        assert preprocessor.feature_columns == []
        
        # Test with pre-fitted scaler
        scaler = StandardScaler()
        preprocessor = DataPreprocessor(scaler=scaler)
        assert preprocessor.scaler is scaler
        assert preprocessor.is_fitted is True
    
    def test_create_features(self, preprocessor, sample_ohlcv_data):
        """Test feature creation from OHLCV data"""
        features_df = preprocessor.create_features(sample_ohlcv_data)
        
        # Check that features were created
        expected_features = [
            'daily_return', 'prev_close', 'price_change', 'pct_change', 
            'volatility', 'volume_change', 'ma5', 'ma10', 'ma20',
            'ema12', 'ema26', 'macd', 'macd_signal', 'rsi',
            'ma20_std', 'upper_band', 'lower_band', 'bb_position'
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns
        
        # Check that feature columns were stored
        assert len(preprocessor.feature_columns) > 0
        
        # Check that NaN values were dropped
        assert not features_df.isnull().any().any()
    
    def test_create_features_empty_data(self, preprocessor):
        """Test feature creation with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = preprocessor.create_features(empty_df)
        assert result.empty
    
    def test_create_features_missing_columns(self, preprocessor):
        """Test feature creation with missing required columns"""
        incomplete_df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [101.0, 102.0],
            # Missing 'low', 'close', and 'volume' columns
        })
        
        with pytest.raises(ValueError, match="Input data missing required columns"):
            preprocessor.create_features(incomplete_df)
    
    def test_create_target_labels(self, preprocessor, sample_ohlcv_data):
        """Test target label generation"""
        labeled_df = preprocessor.create_target_labels(sample_ohlcv_data)
        
        # Check that target column was created
        assert 'target' in labeled_df.columns
        
        # Check that target is binary (0 or 1)
        assert set(labeled_df['target'].unique()).issubset({0, 1})
        
        # Check that next_close was dropped
        assert 'next_close' not in labeled_df.columns
        
        # Check that the last row was dropped (which would have NaN for next_close)
        assert len(labeled_df) == len(sample_ohlcv_data) - 1
    
    def test_create_target_labels_missing_close(self, preprocessor):
        """Test target label generation with missing close column"""
        incomplete_df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [101.0, 102.0],
            'low': [99.0, 100.0],
            'volume': [10000, 20000]
            # Missing 'close' column
        })
        
        with pytest.raises(ValueError, match="Input data missing 'close' column"):
            preprocessor.create_target_labels(incomplete_df)
    
    def test_scale_features(self, preprocessor, sample_ohlcv_data):
        """Test feature scaling"""
        # First create features
        features_df = preprocessor.create_features(sample_ohlcv_data)
        
        # Then scale them
        scaled_df = preprocessor.scale_features(features_df)
        
        # Check that scaler was fitted
        assert preprocessor.is_fitted is True
        
        # Check that scaled features have mean close to 0 and std close to 1
        for feature in preprocessor.feature_columns:
            assert -0.1 < scaled_df[feature].mean() < 0.1  # Mean close to 0
            assert 0.9 < scaled_df[feature].std() < 1.1  # Std close to 1
    
    def test_scale_features_no_feature_columns(self):
        """Test scaling with no feature columns defined"""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        with pytest.raises(ValueError, match="Feature columns must be defined"):
            preprocessor.scale_features(df)
    
    def test_scale_features_missing_columns(self, preprocessor):
        """Test scaling with missing feature columns"""
        # Set feature columns manually
        preprocessor.feature_columns = ['feature1', 'feature2']
        
        # Create DataFrame without those columns
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        with pytest.raises(ValueError, match="No feature columns found in data"):
            preprocessor.scale_features(df)
    
    def test_split_data(self, preprocessor, sample_ohlcv_data):
        """Test temporal train/test split"""
        train_df, test_df = preprocessor.split_data(sample_ohlcv_data, test_size=0.2)
        
        # Check split sizes
        assert len(train_df) == int(len(sample_ohlcv_data) * 0.8)
        assert len(test_df) == len(sample_ohlcv_data) - len(train_df)
        
        # Check temporal order
        assert train_df.index.max() < test_df.index.min()
    
    def test_split_data_empty_data(self, preprocessor):
        """Test splitting with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input data is empty"):
            preprocessor.split_data(empty_df)
    
    def test_split_data_invalid_test_size(self, preprocessor, sample_ohlcv_data):
        """Test splitting with invalid test_size"""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            preprocessor.split_data(sample_ohlcv_data, test_size=0)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            preprocessor.split_data(sample_ohlcv_data, test_size=1)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            preprocessor.split_data(sample_ohlcv_data, test_size=-0.5)
        
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            preprocessor.split_data(sample_ohlcv_data, test_size=1.5)
    
    def test_prepare_data_for_training(self, preprocessor, sample_ohlcv_data):
        """Test complete data preparation pipeline"""
        result = preprocessor.prepare_data_for_training(sample_ohlcv_data, test_size=0.2)
        
        # Check that all expected keys are present
        expected_keys = ['X_train', 'y_train', 'X_test', 'y_test', 'metadata', 'train_df', 'test_df']
        for key in expected_keys:
            assert key in result
        
        # Check shapes
        assert result['X_train'].shape[0] == result['y_train'].shape[0]
        assert result['X_test'].shape[0] == result['y_test'].shape[0]
        assert result['X_train'].shape[1] == len(preprocessor.feature_columns)
        assert result['X_test'].shape[1] == len(preprocessor.feature_columns)
        
        # Check metadata
        assert 'feature_columns' in result['metadata']
        assert 'train_dates' in result['metadata']
        assert 'test_dates' in result['metadata']
        assert 'train_size' in result['metadata']
        assert 'test_size' in result['metadata']
        assert 'class_distribution' in result['metadata']
        assert 'scaler' in result['metadata']
        
        # Check that train and test DataFrames are provided
        assert isinstance(result['train_df'], pd.DataFrame)
        assert isinstance(result['test_df'], pd.DataFrame)


class TestDataPreprocessorIntegration:
    """Integration tests for DataPreprocessor with real-like scenarios"""
    
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
    
    def test_feature_engineering_pipeline(self, sample_ohlcv_data):
        """Test complete feature engineering pipeline with realistic data"""
        preprocessor = DataPreprocessor()
        
        # Create features
        features_df = preprocessor.create_features(sample_ohlcv_data)
        
        # Create target labels
        labeled_df = preprocessor.create_target_labels(features_df)
        
        # Split data
        train_df, test_df = preprocessor.split_data(labeled_df, test_size=0.2)
        
        # Scale features
        train_scaled = preprocessor.scale_features(train_df, fit=True)
        test_scaled = preprocessor.scale_features(test_df, fit=False)
        
        # Verify results
        assert not train_scaled.isnull().any().any()
        assert not test_scaled.isnull().any().any()
        assert 'target' in train_scaled.columns
        assert 'target' in test_scaled.columns
        
        # Check that scaling was applied correctly
        for feature in preprocessor.feature_columns:
            # Train data should be centered around 0
            assert -0.1 < train_scaled[feature].mean() < 0.1
            
            # Test data might not be perfectly centered since it uses the training scaler
            assert train_scaled[feature].std() > 0.9


if __name__ == '__main__':
    pytest.main([__file__])
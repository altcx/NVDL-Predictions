"""
Data preprocessing module for NVDL Stock Predictor
Handles feature engineering and data preparation for machine learning models
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from sklearn.preprocessing import StandardScaler
from utils.logger import get_data_logger
from config import config


class DataPreprocessor:
    """
    Data preprocessing class with feature engineering and normalization
    Handles feature creation, target label generation, and train/test splitting
    """
    
    def __init__(self, scaler: Optional[StandardScaler] = None):
        """
        Initialize DataPreprocessor with optional scaler
        
        Args:
            scaler: Optional pre-fitted StandardScaler for feature normalization
        """
        self.logger = get_data_logger()
        self.scaler = scaler or StandardScaler()
        self.is_fitted = False if scaler is None else True
        self.feature_columns = []
        
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw OHLCV data
        
        Args:
            data: DataFrame with OHLCV data indexed by timestamp
            
        Returns:
            DataFrame with additional engineered features
            
        Raises:
            ValueError: If input data is missing required columns
        """
        if data.empty:
            self.logger.warning("Cannot create features: DataFrame is empty")
            return data
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Input data missing required columns: {missing_columns}")
        
        self.logger.info(f"Creating features from {len(data)} data points")
        
        # Create a copy to avoid modifying the original data
        features = data.copy()
        
        # Daily return (close - open) / open
        features['daily_return'] = (features['close'] - features['open']) / features['open']
        
        # Previous day's close
        features['prev_close'] = features['close'].shift(1)
        
        # Price change from previous day
        features['price_change'] = features['close'] - features['prev_close']
        
        # Percentage price change from previous day
        features['pct_change'] = features['close'].pct_change()
        
        # Daily volatility (high - low) / open
        features['volatility'] = (features['high'] - features['low']) / features['open']
        
        # Volume change
        features['volume_change'] = features['volume'].pct_change()
        
        # Moving averages
        features['ma5'] = features['close'].rolling(window=5).mean()
        features['ma10'] = features['close'].rolling(window=10).mean()
        features['ma20'] = features['close'].rolling(window=20).mean()
        
        # MACD components
        features['ema12'] = features['close'].ewm(span=12, adjust=False).mean()
        features['ema26'] = features['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = features['ema12'] - features['ema26']
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = features['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
        
        # Replace NaN values with 50 (neutral) for RSI
        features['rsi'] = features['rsi'].fillna(50)
        
        # Bollinger Bands
        features['ma20_std'] = features['close'].rolling(window=20).std()
        features['upper_band'] = features['ma20'] + (features['ma20_std'] * 2)
        features['lower_band'] = features['ma20'] - (features['ma20_std'] * 2)
        
        # Distance from Bollinger Bands
        features['bb_position'] = (features['close'] - features['lower_band']) / (features['upper_band'] - features['lower_band'])
        
        # Store feature columns for later use
        self.feature_columns = [
            'daily_return', 'price_change', 'pct_change', 'volatility',
            'volume_change', 'ma5', 'ma10', 'ma20', 'macd', 'macd_signal',
            'rsi', 'bb_position'
        ]
        
        # Drop rows with NaN values that result from calculations
        features.dropna(inplace=True)
        
        self.logger.info(f"Created {len(self.feature_columns)} features, {len(features)} valid data points")
        return features
    
    def create_target_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target labels for price direction prediction
        
        Args:
            data: DataFrame with OHLCV data indexed by timestamp
            
        Returns:
            DataFrame with added target label column
            
        Raises:
            ValueError: If input data is missing close price column
        """
        if 'close' not in data.columns:
            self.logger.error("Missing 'close' column required for target generation")
            raise ValueError("Input data missing 'close' column")
        
        self.logger.info("Creating target labels for price direction prediction")
        
        # Create a copy to avoid modifying the original data
        labeled_data = data.copy()
        
        # Next day's closing price
        labeled_data['next_close'] = labeled_data['close'].shift(-1)
        
        # Binary label: 1 if tomorrow's close > today's close, 0 otherwise
        labeled_data['target'] = (labeled_data['next_close'] > labeled_data['close']).astype(int)
        
        # Drop the last row which has NaN for next_close
        labeled_data.dropna(subset=['next_close'], inplace=True)
        
        # Drop the next_close column as it contains future information
        labeled_data.drop('next_close', axis=1, inplace=True)
        
        self.logger.info(f"Created target labels with distribution: {labeled_data['target'].value_counts().to_dict()}")
        return labeled_data
    
    def scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using StandardScaler
        
        Args:
            data: DataFrame with feature columns
            fit: Whether to fit the scaler on this data
            
        Returns:
            DataFrame with scaled features
            
        Raises:
            ValueError: If feature columns are not available
        """
        if not self.feature_columns:
            self.logger.error("No feature columns defined for scaling")
            raise ValueError("Feature columns must be defined before scaling")
        
        # Filter to only include available feature columns
        available_features = [col for col in self.feature_columns if col in data.columns]
        
        if not available_features:
            self.logger.error("None of the defined feature columns are in the data")
            raise ValueError("No feature columns found in data")
        
        self.logger.info(f"Scaling {len(available_features)} features")
        
        # Create a copy to avoid modifying the original data
        scaled_data = data.copy()
        
        # Extract features for scaling
        features_to_scale = scaled_data[available_features].values
        
        # Fit or transform based on the fit parameter
        if fit or not self.is_fitted:
            self.logger.info("Fitting scaler on training data")
            scaled_features = self.scaler.fit_transform(features_to_scale)
            self.is_fitted = True
        else:
            self.logger.info("Transforming features using pre-fitted scaler")
            scaled_features = self.scaler.transform(features_to_scale)
        
        # Replace original features with scaled versions
        for i, col in enumerate(available_features):
            scaled_data[col] = scaled_features[:, i]
        
        return scaled_data
    
    def split_data(self, data: pd.DataFrame, test_size: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets maintaining temporal order
        
        Args:
            data: DataFrame with features and target
            test_size: Proportion of data to use for testing (default from config)
            
        Returns:
            Tuple of (train_data, test_data)
            
        Raises:
            ValueError: If data is empty or test_size is invalid
        """
        if data.empty:
            self.logger.error("Cannot split empty DataFrame")
            raise ValueError("Input data is empty")
        
        # Use config test_size if not provided
        if test_size is None:
            test_size = config.TEST_SIZE
        
        if not (0 < test_size < 1):
            self.logger.error(f"Invalid test_size: {test_size}")
            raise ValueError("test_size must be between 0 and 1")
        
        # Calculate split point
        split_idx = int(len(data) * (1 - test_size))
        
        # Split data chronologically
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        self.logger.info(f"Split data into train ({len(train_data)} samples) and test ({len(test_data)} samples)")
        
        return train_data, test_data
    
    def prepare_data_for_training(
        self, 
        data: pd.DataFrame, 
        test_size: float = None
    ) -> Dict[str, Any]:
        """
        Complete data preparation pipeline for model training
        
        Args:
            data: Raw OHLCV data
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with X_train, X_test, y_train, y_test, and metadata
        """
        self.logger.info("Starting complete data preparation pipeline")
        
        # Create features
        features_df = self.create_features(data)
        
        # Create target labels
        labeled_df = self.create_target_labels(features_df)
        
        # Split data
        train_df, test_df = self.split_data(labeled_df, test_size)
        
        # Scale features (fit on training data only)
        train_df_scaled = self.scale_features(train_df, fit=True)
        test_df_scaled = self.scale_features(test_df, fit=False)
        
        # Extract features and target
        X_train = train_df_scaled[self.feature_columns].values
        y_train = train_df_scaled['target'].values
        
        X_test = test_df_scaled[self.feature_columns].values
        y_test = test_df_scaled['target'].values
        
        # Prepare metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'train_dates': train_df.index,
            'test_dates': test_df.index,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'class_distribution': {
                'train': np.bincount(y_train).tolist(),
                'test': np.bincount(y_test).tolist()
            },
            'scaler': self.scaler
        }
        
        self.logger.info("Data preparation pipeline completed successfully")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'metadata': metadata,
            'train_df': train_df,
            'test_df': test_df
        }
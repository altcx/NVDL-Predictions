"""
Test fixtures for NVDL Stock Predictor
Provides common test data and utilities for all test modules
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from unittest.mock import Mock, patch, MagicMock

# Create sample OHLCV data for testing
def create_sample_ohlcv_data(days=100, start_date='2023-01-01'):
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    np.random.seed(42)  # For reproducible tests
    
    # Generate realistic stock price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, days)  # 2% daily volatility
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

# Create sample processed data for testing
def create_sample_processed_data(days=100, start_date='2023-01-01'):
    """Create sample processed data with features and target labels"""
    # Get base OHLCV data
    df = create_sample_ohlcv_data(days, start_date)
    
    # Add features
    df['daily_return'] = df['close'].pct_change()
    df['prev_close'] = df['close'].shift(1)
    df['price_change'] = df['close'] - df['prev_close']
    df['pct_change'] = df['price_change'] / df['prev_close'] * 100
    df['volatility'] = df['daily_return'].rolling(window=5).std()
    df['volume_change'] = df['volume'].pct_change()
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ema12'] = df['close'].ewm(span=12).mean()
    df['ema26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Add RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Add Bollinger Bands
    df['ma20_std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['ma20'] + (df['ma20_std'] * 2)
    df['lower_band'] = df['ma20'] - (df['ma20_std'] * 2)
    df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
    
    # Add target label (1 if tomorrow's close > today's close, 0 otherwise)
    df['next_close'] = df['close'].shift(-1)
    df['target'] = (df['next_close'] > df['close']).astype(int)
    df = df.drop('next_close', axis=1)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Create sample train/test split data
def create_train_test_split(processed_data, test_size=0.2):
    """Split processed data into train and test sets"""
    split_idx = int(len(processed_data) * (1 - test_size))
    train_df = processed_data.iloc[:split_idx].copy()
    test_df = processed_data.iloc[split_idx:].copy()
    
    # Extract features and target
    feature_cols = [col for col in processed_data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_df': train_df,
        'test_df': test_df,
        'metadata': {
            'feature_columns': feature_cols,
            'train_dates': train_df.index,
            'test_dates': test_df.index,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'class_distribution': {
                'train': {0: (y_train == 0).sum(), 1: (y_train == 1).sum()},
                'test': {0: (y_test == 0).sum(), 1: (y_test == 1).sum()}
            }
        }
    }

# Create sample model results
def create_sample_model_results(test_df, accuracy=0.65):
    """Create sample model prediction results"""
    np.random.seed(42)
    
    # Create predictions with specified accuracy
    y_true = test_df['target'].values
    y_pred = np.zeros_like(y_true)
    
    # Set some predictions to match true values based on accuracy
    match_indices = np.random.choice(
        np.arange(len(y_true)), 
        size=int(len(y_true) * accuracy), 
        replace=False
    )
    y_pred[match_indices] = y_true[match_indices]
    
    # For remaining indices, set to opposite of true value
    non_match_indices = np.setdiff1d(np.arange(len(y_true)), match_indices)
    y_pred[non_match_indices] = 1 - y_true[non_match_indices]
    
    # Create probabilities
    probabilities = np.random.random(len(y_true))
    # Adjust probabilities to align with predictions
    probabilities[y_pred == 1] = 0.5 + (probabilities[y_pred == 1] * 0.5)  # Between 0.5 and 1.0
    probabilities[y_pred == 0] = probabilities[y_pred == 0] * 0.5  # Between 0.0 and 0.5
    
    return {
        'predictions': y_pred,
        'probabilities': probabilities,
        'y_true': y_true,
        'test_dates': test_df.index
    }

# Create sample trading results
def create_sample_trading_results(test_df, predictions, initial_capital=10000.0):
    """Create sample trading simulation results"""
    prices = test_df['close']
    signals = predictions
    
    # Create equity curve
    equity = pd.Series(index=prices.index)
    equity.iloc[0] = initial_capital
    
    position = 0  # 0: no position, 1: long position
    shares = 0
    cash = initial_capital
    transactions = []
    
    for i in range(len(signals)):
        current_date = prices.index[i]
        current_price = prices.iloc[i]
        
        # Execute buy signal
        if signals[i] == 1 and position == 0:
            shares = cash / current_price
            commission = min(max(0.01 * shares, 1.0), 20.0)  # Min $1, max $20
            shares -= commission / current_price
            cash = 0
            position = 1
            
            transactions.append({
                'date': current_date,
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'value': shares * current_price,
                'commission': commission
            })
        
        # Execute sell signal
        elif signals[i] == 0 and position == 1:
            cash = shares * current_price
            commission = min(max(0.01 * shares, 1.0), 20.0)  # Min $1, max $20
            cash -= commission
            shares = 0
            position = 0
            
            transactions.append({
                'date': current_date,
                'action': 'SELL',
                'price': current_price,
                'shares': shares,
                'value': cash,
                'commission': commission
            })
        
        # Update equity
        if position == 1:
            equity.iloc[i] = shares * current_price
        else:
            equity.iloc[i] = cash
    
    # Calculate returns
    returns = equity.pct_change().dropna()
    
    # Calculate performance metrics
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    
    # Calculate Sharpe ratio
    risk_free_rate = 0.02 / 252  # Daily risk-free rate (2% annual)
    excess_returns = returns - risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    # Calculate max drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    if len(transactions) > 1:
        buy_indices = [i for i, t in enumerate(transactions) if t['action'] == 'BUY']
        sell_indices = [i for i, t in enumerate(transactions) if t['action'] == 'SELL']
        
        trades = []
        for buy_idx, sell_idx in zip(buy_indices, sell_indices):
            if sell_idx > buy_idx:
                buy_price = transactions[buy_idx]['price']
                sell_price = transactions[sell_idx]['price']
                profit = (sell_price - buy_price) / buy_price
                trades.append(profit)
        
        win_rate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0
        num_trades = len(trades)
    else:
        win_rate = 0
        num_trades = 0
    
    return {
        'equity_curve': equity,
        'transactions': pd.DataFrame(transactions) if transactions else pd.DataFrame(),
        'final_equity': equity.iloc[-1],
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_trades': num_trades
    }

# Create sample evaluation results
def create_sample_evaluation_results(lstm_results, arima_results):
    """Create sample evaluation results for both models"""
    # Calculate metrics for LSTM
    lstm_true = lstm_results['y_true']
    lstm_pred = lstm_results['predictions']
    lstm_prob = lstm_results['probabilities']
    
    lstm_tp = ((lstm_pred == 1) & (lstm_true == 1)).sum()
    lstm_fp = ((lstm_pred == 1) & (lstm_true == 0)).sum()
    lstm_tn = ((lstm_pred == 0) & (lstm_true == 0)).sum()
    lstm_fn = ((lstm_pred == 0) & (lstm_true == 1)).sum()
    
    lstm_accuracy = (lstm_tp + lstm_tn) / len(lstm_true)
    lstm_precision = lstm_tp / (lstm_tp + lstm_fp) if (lstm_tp + lstm_fp) > 0 else 0
    lstm_recall = lstm_tp / (lstm_tp + lstm_fn) if (lstm_tp + lstm_fn) > 0 else 0
    lstm_f1 = 2 * (lstm_precision * lstm_recall) / (lstm_precision + lstm_recall) if (lstm_precision + lstm_recall) > 0 else 0
    
    # Calculate metrics for ARIMA
    arima_true = arima_results['y_true']
    arima_pred = arima_results['predictions']
    
    arima_tp = ((arima_pred == 1) & (arima_true == 1)).sum()
    arima_fp = ((arima_pred == 1) & (arima_true == 0)).sum()
    arima_tn = ((arima_pred == 0) & (arima_true == 0)).sum()
    arima_fn = ((arima_pred == 0) & (arima_true == 1)).sum()
    
    arima_accuracy = (arima_tp + arima_tn) / len(arima_true)
    arima_precision = arima_tp / (arima_tp + arima_fp) if (arima_tp + arima_fp) > 0 else 0
    arima_recall = arima_tp / (arima_tp + arima_fn) if (arima_tp + arima_fn) > 0 else 0
    arima_f1 = 2 * (arima_precision * arima_recall) / (arima_precision + arima_recall) if (arima_precision + arima_recall) > 0 else 0
    
    return {
        'LSTM': {
            'accuracy': lstm_accuracy,
            'precision': lstm_precision,
            'recall': lstm_recall,
            'f1_score': lstm_f1,
            'true_positives': lstm_tp,
            'false_positives': lstm_fp,
            'true_negatives': lstm_tn,
            'false_negatives': lstm_fn
        },
        'ARIMA': {
            'accuracy': arima_accuracy,
            'precision': arima_precision,
            'recall': arima_recall,
            'f1_score': arima_f1,
            'true_positives': arima_tp,
            'false_positives': arima_fp,
            'true_negatives': arima_tn,
            'false_negatives': arima_fn
        },
        'comparison': pd.DataFrame({
            'LSTM': {
                'accuracy': lstm_accuracy,
                'precision': lstm_precision,
                'recall': lstm_recall,
                'f1_score': lstm_f1
            },
            'ARIMA': {
                'accuracy': arima_accuracy,
                'precision': arima_precision,
                'recall': arima_recall,
                'f1_score': arima_f1
            }
        }).T
    }

# Create a temporary directory for test files
@pytest.fixture
def temp_dir(tmpdir):
    """Create a temporary directory for test files"""
    return tmpdir

# Create a fixture for sample data
@pytest.fixture
def sample_data():
    """Create sample OHLCV data"""
    return create_sample_ohlcv_data()

# Create a fixture for processed data
@pytest.fixture
def processed_data():
    """Create sample processed data"""
    return create_sample_processed_data()

# Create a fixture for train/test split data
@pytest.fixture
def train_test_data(processed_data):
    """Create sample train/test split data"""
    return create_train_test_split(processed_data)

# Create fixtures for model results
@pytest.fixture
def lstm_results(train_test_data):
    """Create sample LSTM model results"""
    return create_sample_model_results(train_test_data['test_df'], accuracy=0.65)

@pytest.fixture
def arima_results(train_test_data):
    """Create sample ARIMA model results"""
    return create_sample_model_results(train_test_data['test_df'], accuracy=0.60)

# Create fixtures for trading results
@pytest.fixture
def lstm_trading_results(train_test_data, lstm_results):
    """Create sample LSTM trading results"""
    return create_sample_trading_results(
        train_test_data['test_df'], 
        lstm_results['predictions']
    )

@pytest.fixture
def arima_trading_results(train_test_data, arima_results):
    """Create sample ARIMA trading results"""
    return create_sample_trading_results(
        train_test_data['test_df'], 
        arima_results['predictions']
    )

# Create a fixture for evaluation results
@pytest.fixture
def evaluation_results(lstm_results, arima_results):
    """Create sample evaluation results"""
    return create_sample_evaluation_results(lstm_results, arima_results)

# Create a fixture for mocking TensorFlow
@pytest.fixture
def mock_tensorflow():
    """Mock TensorFlow for testing"""
    with patch.dict('sys.modules', {
        'tensorflow': MagicMock(),
        'tensorflow.keras': MagicMock(),
        'tensorflow.keras.models': MagicMock(),
        'tensorflow.keras.layers': MagicMock(),
        'tensorflow.keras.callbacks': MagicMock(),
        'tensorflow.keras.optimizers': MagicMock(),
        'tensorflow.keras.losses': MagicMock(),
        'tensorflow.keras.metrics': MagicMock()
    }):
        yield

# Create a fixture for mocking statsmodels
@pytest.fixture
def mock_statsmodels():
    """Mock statsmodels for testing"""
    with patch.dict('sys.modules', {
        'statsmodels': MagicMock(),
        'statsmodels.tsa': MagicMock(),
        'statsmodels.tsa.arima': MagicMock(),
        'statsmodels.tsa.arima.model': MagicMock(),
        'statsmodels.tools': MagicMock(),
        'statsmodels.tools.sm_exceptions': MagicMock()
    }):
        yield

# Create a fixture for mocking plotly
@pytest.fixture
def mock_plotly():
    """Mock plotly for testing"""
    with patch.dict('sys.modules', {
        'plotly': MagicMock(),
        'plotly.graph_objects': MagicMock(),
        'plotly.subplots': MagicMock()
    }):
        yield
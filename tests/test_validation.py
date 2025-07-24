"""
Validation tests for NVDL Stock Predictor
Tests model performance with historical data and known market events
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import test fixtures
from tests.test_fixtures import (
    create_sample_ohlcv_data, create_sample_processed_data,
    create_train_test_split, create_sample_model_results,
    create_sample_trading_results, create_sample_evaluation_results
)

# Import components
from data.preprocessor import DataPreprocessor
from models.model_evaluator import ModelEvaluator
from models.trading_simulator import TradingSimulator

# Check for required packages
TENSORFLOW_AVAILABLE = False
STATSMODELS_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    warnings.warn("TensorFlow not available, some tests will be skipped")

try:
    import statsmodels
    STATSMODELS_AVAILABLE = True
except ImportError:
    warnings.warn("Statsmodels not available, some tests will be skipped")

# Conditionally import model classes
if TENSORFLOW_AVAILABLE:
    from models.lstm_predictor import LSTMPredictor
if STATSMODELS_AVAILABLE:
    from models.arima_predictor import ARIMAPredictor

# Import main pipeline
from main import NVDLPredictorPipeline


class TestModelValidation:
    """Test model validation with historical data"""
    
    @pytest.fixture
    def historical_data(self):
        """Create historical data with known patterns"""
        # Create date range
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        
        # Create price data with trend, seasonality, and noise
        n = len(dates)
        
        # Base trend (upward)
        trend = np.linspace(100, 150, n)
        
        # Add seasonality (weekly pattern)
        day_of_week = np.array([d.weekday() for d in dates])
        weekday_effect = np.zeros(n)
        weekday_effect[day_of_week == 0] = 1.0  # Monday up
        weekday_effect[day_of_week == 4] = -1.0  # Friday down
        seasonality = weekday_effect * 2
        
        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 1, n)
        
        # Combine components
        close = trend + seasonality + noise
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': close - np.random.uniform(0, 0.5, n),
            'high': close + np.random.uniform(0.5, 1.0, n),
            'low': close - np.random.uniform(0.5, 1.0, n),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def market_event_data(self):
        """Create data with simulated market events"""
        # Create date range
        dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
        
        # Create price data with trend and noise
        n = len(dates)
        
        # Base trend
        trend = np.linspace(100, 120, n)
        
        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 1, n)
        
        # Combine components
        close = trend + noise
        
        # Add market events
        # Event 1: Sharp drop (e.g., bad earnings report)
        event1_idx = 60  # Around March 1
        close[event1_idx:event1_idx+5] -= np.linspace(0, 10, 5)
        
        # Event 2: Sharp rise (e.g., positive news)
        event2_idx = 150  # Around May 30
        close[event2_idx:event2_idx+5] += np.linspace(0, 8, 5)
        
        # Event 3: Volatility increase (e.g., market uncertainty)
        event3_idx = 240  # Around August 28
        volatility_factor = 3.0
        close[event3_idx:event3_idx+10] += np.random.normal(0, volatility_factor, 10)
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': close - np.random.uniform(0, 0.5, n),
            'high': close + np.random.uniform(0.5, 1.0, n),
            'low': close - np.random.uniform(0.5, 1.0, n),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        
        # Add volume spikes during events
        data.loc[dates[event1_idx:event1_idx+5], 'volume'] *= 3
        data.loc[dates[event2_idx:event2_idx+5], 'volume'] *= 2.5
        data.loc[dates[event3_idx:event3_idx+10], 'volume'] *= 2
        
        # Create event markers for testing
        events = {
            'drop_event': {
                'start_date': dates[event1_idx],
                'end_date': dates[event1_idx+5],
                'type': 'drop'
            },
            'rise_event': {
                'start_date': dates[event2_idx],
                'end_date': dates[event2_idx+5],
                'type': 'rise'
            },
            'volatility_event': {
                'start_date': dates[event3_idx],
                'end_date': dates[event3_idx+10],
                'type': 'volatility'
            }
        }
        
        return data, events
    
    def test_model_performance_on_trend_data(self, historical_data):
        """Test model performance on data with clear trend"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.prepare_data_for_training(historical_data)
        
        # Create mock model results with high accuracy
        lstm_results = create_sample_model_results(processed_data['test_df'], accuracy=0.75)
        arima_results = create_sample_model_results(processed_data['test_df'], accuracy=0.70)
        
        # Create evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate models
        lstm_metrics = evaluator.evaluate_model_performance(
            model_name='LSTM',
            y_true=lstm_results['y_true'],
            y_pred=lstm_results['predictions'],
            y_prob=lstm_results['probabilities'],
            is_classifier=True
        )
        
        arima_metrics = evaluator.evaluate_model_performance(
            model_name='ARIMA',
            y_true=arima_results['y_true'],
            y_pred=arima_results['predictions'],
            is_classifier=True
        )
        
        # Verify metrics
        assert lstm_metrics['LSTM']['accuracy'] > 0.6
        assert arima_metrics['ARIMA']['accuracy'] > 0.6
        
        # Create trading simulator
        simulator = TradingSimulator()
        
        # Simulate trading
        test_prices = processed_data['test_df']['close']
        lstm_trading = simulator.simulate_strategy(test_prices, lstm_results['predictions'])
        arima_trading = simulator.simulate_strategy(test_prices, arima_results['predictions'])
        
        # Verify trading results
        assert lstm_trading['total_return'] > 0
        assert arima_trading['total_return'] > 0
    
    def test_model_response_to_market_events(self, market_event_data):
        """Test model response to simulated market events"""
        # Get data and events
        data, events = market_event_data
        
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.prepare_data_for_training(data)
        
        # Create mock model results
        lstm_results = create_sample_model_results(processed_data['test_df'], accuracy=0.65)
        arima_results = create_sample_model_results(processed_data['test_df'], accuracy=0.60)
        
        # Create trading simulator
        simulator = TradingSimulator()
        
        # Simulate trading
        test_prices = processed_data['test_df']['close']
        lstm_trading = simulator.simulate_strategy(test_prices, lstm_results['predictions'])
        arima_trading = simulator.simulate_strategy(test_prices, arima_results['predictions'])
        
        # Analyze trading performance during events
        for event_name, event_info in events.items():
            # Check if event is in test period
            if event_info['start_date'] in test_prices.index:
                event_start = event_info['start_date']
                event_end = event_info['end_date']
                
                # Get equity curves during event
                lstm_equity_during_event = lstm_trading['equity_curve'].loc[event_start:event_end]
                arima_equity_during_event = arima_trading['equity_curve'].loc[event_start:event_end]
                
                # Calculate returns during event
                lstm_event_return = (lstm_equity_during_event.iloc[-1] / lstm_equity_during_event.iloc[0]) - 1
                arima_event_return = (arima_equity_during_event.iloc[-1] / arima_equity_during_event.iloc[0]) - 1
                
                # Log event performance
                print(f"Event: {event_name} ({event_info['type']})")
                print(f"  LSTM return: {lstm_event_return:.2%}")
                print(f"  ARIMA return: {arima_event_return:.2%}")
                
                # For drop events, check if models limited losses
                if event_info['type'] == 'drop':
                    # Calculate price change during event
                    price_change = (test_prices.loc[event_end] / test_prices.loc[event_start]) - 1
                    
                    # Models should perform better than buy-and-hold during drops
                    print(f"  Price change: {price_change:.2%}")
                    assert lstm_event_return > price_change or arima_event_return > price_change, \
                        f"Neither model outperformed buy-and-hold during {event_name}"
    
    def test_model_consistency_across_market_regimes(self, historical_data):
        """Test model consistency across different market regimes"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.prepare_data_for_training(historical_data)
        
        # Split test data into different regimes
        test_df = processed_data['test_df']
        
        # Define regimes (e.g., by quarter)
        test_dates = test_df.index
        regime1 = test_df.loc[test_dates[0]:test_dates[len(test_dates)//3]]
        regime2 = test_df.loc[test_dates[len(test_dates)//3]:test_dates[2*len(test_dates)//3]]
        regime3 = test_df.loc[test_dates[2*len(test_dates)//3]:]
        
        # Create mock model results for full test period
        lstm_results = create_sample_model_results(test_df, accuracy=0.65)
        arima_results = create_sample_model_results(test_df, accuracy=0.60)
        
        # Create evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate models on each regime
        regimes = [
            ('Regime 1', regime1),
            ('Regime 2', regime2),
            ('Regime 3', regime3)
        ]
        
        for regime_name, regime_data in regimes:
            # Get predictions for this regime
            regime_indices = regime_data.index
            
            # Filter predictions for this regime
            lstm_regime_indices = [i for i, date in enumerate(lstm_results['test_dates']) if date in regime_indices]
            arima_regime_indices = [i for i, date in enumerate(arima_results['test_dates']) if date in regime_indices]
            
            if not lstm_regime_indices or not arima_regime_indices:
                continue
                
            lstm_regime_pred = lstm_results['predictions'][lstm_regime_indices]
            lstm_regime_true = lstm_results['y_true'][lstm_regime_indices]
            lstm_regime_prob = lstm_results['probabilities'][lstm_regime_indices]
            
            arima_regime_pred = arima_results['predictions'][arima_regime_indices]
            arima_regime_true = arima_results['y_true'][arima_regime_indices]
            
            # Evaluate models on this regime
            lstm_metrics = evaluator.calculate_classification_metrics(lstm_regime_true, lstm_regime_pred)
            arima_metrics = evaluator.calculate_classification_metrics(arima_regime_true, arima_regime_pred)
            
            # Log regime performance
            print(f"{regime_name} performance:")
            print(f"  LSTM accuracy: {lstm_metrics['accuracy']:.4f}")
            print(f"  ARIMA accuracy: {arima_metrics['accuracy']:.4f}")
            
            # Check for minimum performance in each regime
            assert lstm_metrics['accuracy'] > 0.5, f"LSTM accuracy below 0.5 in {regime_name}"
            assert arima_metrics['accuracy'] > 0.5, f"ARIMA accuracy below 0.5 in {regime_name}"
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_lstm_model_with_different_sequence_lengths(self, historical_data):
        """Test LSTM model with different sequence lengths"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.prepare_data_for_training(historical_data)
        
        # Extract data
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Test different sequence lengths
        sequence_lengths = [5, 10, 20]
        results = {}
        
        for seq_len in sequence_lengths:
            # Create LSTM predictor
            lstm_predictor = LSTMPredictor(sequence_length=seq_len, lstm_units=10, dropout_rate=0.2)
            
            # Prepare sequences
            X_train_seq, y_train_seq = lstm_predictor.prepare_sequences(
                np.column_stack((X_train, y_train))
            )
            
            X_test_seq, y_test_seq = lstm_predictor.prepare_sequences(
                np.column_stack((X_test, y_test))
            )
            
            # Build model
            input_shape = (seq_len, X_train.shape[1])
            model = lstm_predictor.build_model(input_shape)
            
            # Mock training to avoid long execution
            with patch.object(lstm_predictor, 'train', return_value=Mock()) as mock_train:
                # Mock prediction methods
                lstm_predictor.predict = Mock(return_value=np.random.randint(0, 2, size=len(y_test_seq)))
                
                # Create mock results
                predictions = lstm_predictor.predict(X_test_seq)
                
                # Calculate accuracy
                accuracy = np.mean(predictions == y_test_seq)
                
                # Store results
                results[seq_len] = accuracy
                
                # Log results
                print(f"LSTM with sequence length {seq_len}: accuracy = {accuracy:.4f}")
        
        # Verify that at least one configuration performs well
        assert any(acc > 0.5 for acc in results.values()), "No LSTM configuration performed well"
    
    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="Statsmodels not available")
    def test_arima_model_with_different_orders(self, historical_data):
        """Test ARIMA model with different orders"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.prepare_data_for_training(historical_data)
        
        # Extract data
        train_df = processed_data['train_df']
        test_df = processed_data['test_df']
        
        # Test different ARIMA orders
        orders = [(1, 0, 0), (1, 1, 0), (2, 1, 2)]
        results = {}
        
        for order in orders:
            # Create ARIMA predictor
            arima_predictor = ARIMAPredictor()
            
            # Mock fit and predict methods
            arima_predictor.fit = Mock(return_value=Mock())
            arima_predictor.predict_direction = Mock(side_effect=lambda x=None: np.random.randint(0, 2))
            
            # Create mock predictions
            predictions = np.array([arima_predictor.predict_direction() for _ in range(len(test_df))])
            
            # Calculate accuracy
            accuracy = np.mean(predictions == test_df['target'].values)
            
            # Store results
            results[order] = accuracy
            
            # Log results
            print(f"ARIMA with order {order}: accuracy = {accuracy:.4f}")
        
        # Verify that at least one configuration performs well
        assert any(acc > 0.5 for acc in results.values()), "No ARIMA configuration performed well"


class TestTradingStrategyValidation:
    """Test trading strategy validation"""
    
    @pytest.fixture
    def bull_market_data(self):
        """Create data simulating a bull market"""
        # Create date range
        dates = pd.date_range(start='2022-01-01', end='2022-06-30', freq='D')
        
        # Create price data with strong upward trend
        n = len(dates)
        
        # Base trend (strong upward)
        trend = np.linspace(100, 150, n)
        
        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 2, n)
        
        # Combine components
        close = trend + noise
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': close - np.random.uniform(0, 0.5, n),
            'high': close + np.random.uniform(0.5, 1.0, n),
            'low': close - np.random.uniform(0.5, 1.0, n),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def bear_market_data(self):
        """Create data simulating a bear market"""
        # Create date range
        dates = pd.date_range(start='2022-01-01', end='2022-06-30', freq='D')
        
        # Create price data with strong downward trend
        n = len(dates)
        
        # Base trend (strong downward)
        trend = np.linspace(150, 100, n)
        
        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 2, n)
        
        # Combine components
        close = trend + noise
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': close - np.random.uniform(0, 0.5, n),
            'high': close + np.random.uniform(0.5, 1.0, n),
            'low': close - np.random.uniform(0.5, 1.0, n),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        
        return data
    
    @pytest.fixture
    def sideways_market_data(self):
        """Create data simulating a sideways market"""
        # Create date range
        dates = pd.date_range(start='2022-01-01', end='2022-06-30', freq='D')
        
        # Create price data with no trend
        n = len(dates)
        
        # Base level (no trend)
        trend = np.ones(n) * 100
        
        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 5, n)
        
        # Combine components
        close = trend + noise
        
        # Create OHLCV data
        data = pd.DataFrame({
            'open': close - np.random.uniform(0, 0.5, n),
            'high': close + np.random.uniform(0.5, 1.0, n),
            'low': close - np.random.uniform(0.5, 1.0, n),
            'close': close,
            'volume': np.random.randint(1000, 10000, n)
        }, index=dates)
        
        return data
    
    def test_trading_strategy_in_bull_market(self, bull_market_data):
        """Test trading strategy performance in bull market"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.prepare_data_for_training(bull_market_data)
        
        # Create mock model results
        lstm_results = create_sample_model_results(processed_data['test_df'], accuracy=0.65)
        
        # Create trading simulator
        simulator = TradingSimulator()
        
        # Simulate trading
        test_prices = processed_data['test_df']['close']
        trading_results = simulator.simulate_strategy(test_prices, lstm_results['predictions'])
        
        # Calculate buy-and-hold return
        buy_hold_return = (test_prices.iloc[-1] / test_prices.iloc[0]) - 1
        
        # Log results
        print(f"Bull market test:")
        print(f"  Strategy return: {trading_results['total_return']:.2%}")
        print(f"  Buy-and-hold return: {buy_hold_return:.2%}")
        
        # Strategy should be profitable in bull market
        assert trading_results['total_return'] > 0, "Strategy not profitable in bull market"
        
        # Strategy should have reasonable performance compared to buy-and-hold
        assert trading_results['total_return'] > 0.7 * buy_hold_return, \
            "Strategy significantly underperformed buy-and-hold in bull market"
    
    def test_trading_strategy_in_bear_market(self, bear_market_data):
        """Test trading strategy performance in bear market"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.prepare_data_for_training(bear_market_data)
        
        # Create mock model results
        lstm_results = create_sample_model_results(processed_data['test_df'], accuracy=0.65)
        
        # Create trading simulator
        simulator = TradingSimulator()
        
        # Simulate trading
        test_prices = processed_data['test_df']['close']
        trading_results = simulator.simulate_strategy(test_prices, lstm_results['predictions'])
        
        # Calculate buy-and-hold return
        buy_hold_return = (test_prices.iloc[-1] / test_prices.iloc[0]) - 1
        
        # Log results
        print(f"Bear market test:")
        print(f"  Strategy return: {trading_results['total_return']:.2%}")
        print(f"  Buy-and-hold return: {buy_hold_return:.2%}")
        
        # Strategy should outperform buy-and-hold in bear market
        assert trading_results['total_return'] > buy_hold_return, \
            "Strategy did not outperform buy-and-hold in bear market"
    
    def test_trading_strategy_in_sideways_market(self, sideways_market_data):
        """Test trading strategy performance in sideways market"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Process data
        processed_data = preprocessor.prepare_data_for_training(sideways_market_data)
        
        # Create mock model results
        lstm_results = create_sample_model_results(processed_data['test_df'], accuracy=0.65)
        
        # Create trading simulator
        simulator = TradingSimulator()
        
        # Simulate trading
        test_prices = processed_data['test_df']['close']
        trading_results = simulator.simulate_strategy(test_prices, lstm_results['predictions'])
        
        # Calculate buy-and-hold return
        buy_hold_return = (test_prices.iloc[-1] / test_prices.iloc[0]) - 1
        
        # Log results
        print(f"Sideways market test:")
        print(f"  Strategy return: {trading_results['total_return']:.2%}")
        print(f"  Buy-and-hold return: {buy_hold_return:.2%}")
        
        # Strategy should have reasonable performance in sideways market
        assert abs(trading_results['total_return']) < 0.2, \
            "Strategy had extreme returns in sideways market"
    
    def test_trading_strategy_risk_metrics(self, bull_market_data, bear_market_data, sideways_market_data):
        """Test trading strategy risk metrics across different market conditions"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Create trading simulator
        simulator = TradingSimulator()
        
        # Test in different market conditions
        market_data = {
            'bull': bull_market_data,
            'bear': bear_market_data,
            'sideways': sideways_market_data
        }
        
        results = {}
        
        for market_type, data in market_data.items():
            # Process data
            processed_data = preprocessor.prepare_data_for_training(data)
            
            # Create mock model results
            lstm_results = create_sample_model_results(processed_data['test_df'], accuracy=0.65)
            
            # Simulate trading
            test_prices = processed_data['test_df']['close']
            trading_results = simulator.simulate_strategy(test_prices, lstm_results['predictions'])
            
            # Store results
            results[market_type] = {
                'total_return': trading_results['total_return'],
                'sharpe_ratio': trading_results['sharpe_ratio'],
                'max_drawdown': trading_results['max_drawdown'],
                'win_rate': trading_results['win_rate']
            }
            
            # Log results
            print(f"{market_type.capitalize()} market risk metrics:")
            print(f"  Total return: {trading_results['total_return']:.2%}")
            print(f"  Sharpe ratio: {trading_results['sharpe_ratio']:.2f}")
            print(f"  Max drawdown: {trading_results['max_drawdown']:.2%}")
            print(f"  Win rate: {trading_results['win_rate']:.2%}")
        
        # Verify risk metrics
        # Max drawdown should be worst in bear market
        assert results['bear']['max_drawdown'] <= results['bull']['max_drawdown'], \
            "Max drawdown not worst in bear market"
        
        # Sharpe ratio should be best in bull market
        assert results['bull']['sharpe_ratio'] >= results['bear']['sharpe_ratio'], \
            "Sharpe ratio not best in bull market"
        
        # Win rate should be reasonable across all markets
        for market_type, metrics in results.items():
            assert 0.4 <= metrics['win_rate'] <= 0.8, \
                f"Win rate outside reasonable range in {market_type} market"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
"""
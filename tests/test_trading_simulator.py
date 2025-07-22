"""
Unit tests for TradingSimulator class
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models.trading_simulator import TradingSimulator


class TestTradingSimulator(unittest.TestCase):
    """Test cases for TradingSimulator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = TradingSimulator(initial_capital=10000.0)
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=dates)
        self.prices = prices
        
        # Create sample signals
        self.buy_hold_signals = np.ones(10)  # Buy and hold
        self.sell_signals = np.zeros(10)  # Always sell
        self.alternating_signals = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # Alternating buy/sell
        
        # Create sample equity curve
        self.equity_curve = pd.Series([10000, 10100, 10200, 10300, 10200, 10100, 10200, 10300, 10400, 10500], index=dates)
    
    def test_initialization(self):
        """Test TradingSimulator initialization"""
        self.assertEqual(self.simulator.initial_capital, 10000.0)
        
        # Test default initialization
        default_simulator = TradingSimulator()
        self.assertIsNotNone(default_simulator.initial_capital)
    
    def test_simulate_strategy_buy_hold(self):
        """Test simulation with buy and hold strategy"""
        results = self.simulator.simulate_strategy(self.prices, self.buy_hold_signals)
        
        # Check results structure
        self.assertIn('equity_curve', results)
        self.assertIn('transactions', results)
        self.assertIn('final_equity', results)
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('win_rate', results)
        
        # Check that we have a buy transaction
        self.assertGreater(len(results['transactions']), 0)
        self.assertEqual(results['transactions'].iloc[0]['action'], 'BUY')
    
    def test_simulate_strategy_alternating(self):
        """Test simulation with alternating buy/sell signals"""
        results = self.simulator.simulate_strategy(self.prices, self.alternating_signals)
        
        # Check that we have multiple transactions
        self.assertGreater(len(results['transactions']), 1)
        
        # Check that we have both buy and sell transactions
        actions = results['transactions']['action'].unique()
        self.assertIn('BUY', actions)
        self.assertIn('SELL', actions)
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Create a series with a known drawdown
        equity = pd.Series([100, 110, 105, 95, 90, 100, 110])
        
        # Max drawdown should be (90-110)/110 = 0.1818...
        expected_drawdown = (110 - 90) / 110
        
        drawdown = self.simulator.calculate_max_drawdown(equity)
        self.assertAlmostEqual(drawdown, expected_drawdown, places=4)
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        # Create a series of returns with known mean and std
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        
        # Calculate expected Sharpe ratio
        excess_returns = returns - 0.0  # Assuming 0 risk-free rate for simplicity
        expected_sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        
        sharpe = self.simulator.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        self.assertAlmostEqual(sharpe, expected_sharpe, places=4)
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        # Create sample transactions
        transactions = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=4),
            'action': ['BUY', 'SELL', 'BUY', 'SELL'],
            'price': [100, 110, 105, 115],
            'shares': [100, 100, 95, 95],
            'value': [10000, 11000, 9975, 10925],
            'commission': [10, 11, 9.98, 10.93]
        })
        
        metrics = self.simulator.calculate_performance_metrics(self.equity_curve, self.prices, transactions)
        
        # Check that all expected metrics are present
        expected_metrics = ['total_return', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'num_trades']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check specific values
        self.assertEqual(metrics['num_trades'], 2)
        self.assertEqual(metrics['win_rate'], 1.0)  # Both trades are profitable
    
    def test_length_mismatch_error(self):
        """Test error handling for length mismatch"""
        short_signals = np.ones(5)  # Only 5 signals for 10 prices
        
        with self.assertRaises(ValueError):
            self.simulator.simulate_strategy(self.prices, short_signals)
    
    def test_generate_trade_statistics(self):
        """Test trade statistics generation"""
        # Create sample transactions
        transactions = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=4),
            'action': ['BUY', 'SELL', 'BUY', 'SELL'],
            'price': [100, 110, 105, 95],
            'shares': [100, 100, 95, 95],
            'value': [10000, 11000, 9975, 9025],
            'commission': [10, 11, 9.98, 9.03]
        })
        
        stats = self.simulator.generate_trade_statistics(transactions)
        
        # Check that all expected statistics are present
        expected_stats = ['num_trades', 'win_rate', 'avg_win', 'avg_loss', 'largest_win', 'largest_loss']
        for stat in expected_stats:
            self.assertIn(stat, stats)
        
        # Check specific values
        self.assertEqual(stats['num_trades'], 2)
        self.assertEqual(stats['win_rate'], 0.5)  # One winning trade, one losing trade


if __name__ == '__main__':
    unittest.main()
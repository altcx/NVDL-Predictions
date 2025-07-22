"""
Unit tests for the VisualizationEngine class
"""
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from visualization.visualization_engine import VisualizationEngine


class TestVisualizationEngine:
    """Test suite for VisualizationEngine class"""
    
    @pytest.fixture
    def visualization_engine(self):
        """Create a VisualizationEngine instance for testing"""
        return VisualizationEngine()
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        # Create date range for 30 days
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        
        # Create price series with some volatility
        prices = [100]
        for i in range(1, 30):
            # Random walk with drift
            change = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + change))
        
        return pd.Series(prices, index=dates)
    
    @pytest.fixture
    def sample_volume_data(self, sample_price_data):
        """Create sample volume data for testing"""
        # Create volume data with same index as price data
        volumes = np.random.randint(1000, 10000, size=len(sample_price_data))
        return pd.Series(volumes, index=sample_price_data.index)
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample trading signals for testing"""
        # Create 30 signals with mix of buys (1) and sells (0)
        signals = np.zeros(30)
        signals[5] = 1  # Buy
        signals[10] = 0  # Sell
        signals[15] = 1  # Buy
        signals[25] = 0  # Sell
        return signals
    
    @pytest.fixture
    def sample_equity_curves(self, sample_price_data):
        """Create sample equity curves for testing"""
        # Create LSTM equity curve
        lstm_equity = pd.Series(index=sample_price_data.index)
        lstm_equity.iloc[0] = 10000
        for i in range(1, len(lstm_equity)):
            lstm_equity.iloc[i] = lstm_equity.iloc[i-1] * (1 + np.random.normal(0.002, 0.01))
        
        # Create ARIMA equity curve
        arima_equity = pd.Series(index=sample_price_data.index)
        arima_equity.iloc[0] = 10000
        for i in range(1, len(arima_equity)):
            arima_equity.iloc[i] = arima_equity.iloc[i-1] * (1 + np.random.normal(0.001, 0.015))
        
        return {
            'lstm': lstm_equity,
            'arima': arima_equity
        }
    
    @pytest.fixture
    def sample_metrics_df(self):
        """Create sample metrics DataFrame for testing"""
        data = {
            'LSTM': {
                'accuracy': 0.65,
                'precision': 0.70,
                'recall': 0.68,
                'f1_score': 0.69,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.15,
                'win_rate': 0.62
            },
            'ARIMA': {
                'accuracy': 0.60,
                'precision': 0.65,
                'recall': 0.72,
                'f1_score': 0.68,
                'sharpe_ratio': 0.9,
                'max_drawdown': 0.18,
                'win_rate': 0.58
            }
        }
        return pd.DataFrame(data).T
    
    @pytest.fixture
    def sample_results(self, sample_price_data, sample_signals, sample_equity_curves):
        """Create sample results dictionary for testing"""
        lstm_signals = np.copy(sample_signals)
        arima_signals = np.copy(sample_signals)
        arima_signals[20] = 1  # Add an extra buy signal for ARIMA
        
        return {
            'LSTM': {
                'predictions': lstm_signals,
                'equity_curve': sample_equity_curves['lstm'],
                'accuracy': 0.65,
                'precision': 0.70,
                'recall': 0.68,
                'f1_score': 0.69,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.15,
                'win_rate': 0.62,
                'num_trades': 2
            },
            'ARIMA': {
                'predictions': arima_signals,
                'equity_curve': sample_equity_curves['arima'],
                'accuracy': 0.60,
                'precision': 0.65,
                'recall': 0.72,
                'f1_score': 0.68,
                'sharpe_ratio': 0.9,
                'max_drawdown': 0.18,
                'win_rate': 0.58,
                'num_trades': 3
            }
        }
    
    def test_initialization(self, visualization_engine):
        """Test VisualizationEngine initialization"""
        assert isinstance(visualization_engine, VisualizationEngine)
        assert visualization_engine.colors['lstm'] == '#1f77b4'
        assert visualization_engine.colors['arima'] == '#ff7f0e'
    
    def test_plot_price_with_signals(self, visualization_engine, sample_price_data, sample_signals):
        """Test price chart with signals visualization"""
        fig = visualization_engine.plot_price_with_signals(
            prices=sample_price_data,
            signals=sample_signals,
            model_name="Test Model"
        )
        
        # Check that figure was created
        assert isinstance(fig, go.Figure)
        
        # Check that we have at least 3 traces (price + buy signals + sell signals)
        assert len(fig.data) >= 3
        
        # Check that first trace is the price line
        assert fig.data[0].name == 'Price'
        assert len(fig.data[0].x) == len(sample_price_data)
        assert len(fig.data[0].y) == len(sample_price_data)
    
    def test_plot_price_with_signals_and_volume(self, visualization_engine, sample_price_data, sample_signals, sample_volume_data):
        """Test price chart with signals and volume visualization"""
        fig = visualization_engine.plot_price_with_signals(
            prices=sample_price_data,
            signals=sample_signals,
            model_name="Test Model",
            show_volume=True,
            volume_data=sample_volume_data
        )
        
        # Check that figure was created
        assert isinstance(fig, go.Figure)
        
        # Check that we have at least 4 traces (price + buy signals + sell signals + volume)
        assert len(fig.data) >= 4
        
        # Check that volume data is included
        volume_trace = None
        for trace in fig.data:
            if trace.name == 'Volume':
                volume_trace = trace
                break
        
        assert volume_trace is not None
        assert len(volume_trace.x) == len(sample_volume_data)
        assert len(volume_trace.y) == len(sample_volume_data)
    
    def test_plot_equity_curves(self, visualization_engine, sample_equity_curves):
        """Test equity curves visualization"""
        fig = visualization_engine.plot_equity_curves(
            lstm_equity=sample_equity_curves['lstm'],
            arima_equity=sample_equity_curves['arima'],
            title="Test Equity Curves"
        )
        
        # Check that figure was created
        assert isinstance(fig, go.Figure)
        
        # Check that we have 2 traces (LSTM and ARIMA)
        assert len(fig.data) == 2
        
        # Check trace names
        assert fig.data[0].name == 'LSTM Strategy'
        assert fig.data[1].name == 'ARIMA Strategy'
        
        # Check data lengths
        assert len(fig.data[0].x) == len(sample_equity_curves['lstm'])
        assert len(fig.data[0].y) == len(sample_equity_curves['lstm'])
        assert len(fig.data[1].x) == len(sample_equity_curves['arima'])
        assert len(fig.data[1].y) == len(sample_equity_curves['arima'])
    
    def test_plot_model_comparison(self, visualization_engine, sample_metrics_df):
        """Test model comparison visualization"""
        fig = visualization_engine.plot_model_comparison(
            metrics_df=sample_metrics_df,
            highlight_best=True
        )
        
        # Check that figure was created
        assert isinstance(fig, go.Figure)
        
        # Check that we have traces for each metric
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 
            'sharpe_ratio', 'max_drawdown', 'win_rate'
        ]
        available_metrics = [m for m in expected_metrics if m in sample_metrics_df.columns]
        
        assert len(fig.data) == len(available_metrics)
    
    def test_create_dashboard(self, visualization_engine, sample_results, sample_price_data, sample_volume_data):
        """Test dashboard creation"""
        fig = visualization_engine.create_dashboard(
            results=sample_results,
            prices=sample_price_data,
            volume=sample_volume_data
        )
        
        # Check that figure was created
        assert isinstance(fig, go.Figure)
        
        # Check that we have multiple traces
        assert len(fig.data) > 5
    
    def test_input_validation(self, visualization_engine, sample_price_data):
        """Test input validation for visualizations"""
        # Test with mismatched lengths
        with pytest.raises(ValueError):
            visualization_engine.plot_price_with_signals(
                prices=sample_price_data,
                signals=np.zeros(len(sample_price_data) + 5)  # Mismatched length
            )
        
        # Test with empty metrics DataFrame
        with pytest.raises(ValueError):
            visualization_engine.plot_model_comparison(
                metrics_df=pd.DataFrame()  # Empty DataFrame
            )
        
        # Test with empty results dictionary
        with pytest.raises(ValueError):
            visualization_engine.create_dashboard(
                results={},  # Empty dictionary
                prices=sample_price_data
            )
    
    def test_equity_curves_with_benchmark(self, visualization_engine, sample_equity_curves, sample_price_data):
        """Test equity curves with benchmark visualization"""
        # Create benchmark equity curve (buy and hold)
        benchmark = pd.Series(index=sample_price_data.index)
        benchmark.iloc[0] = 10000
        for i in range(1, len(benchmark)):
            ratio = sample_price_data.iloc[i] / sample_price_data.iloc[0]
            benchmark.iloc[i] = benchmark.iloc[0] * ratio
        
        fig = visualization_engine.plot_equity_curves(
            lstm_equity=sample_equity_curves['lstm'],
            arima_equity=sample_equity_curves['arima'],
            benchmark_equity=benchmark,
            title="Test With Benchmark"
        )
        
        # Check that figure was created
        assert isinstance(fig, go.Figure)
        
        # Check that we have 3 traces (LSTM, ARIMA, and benchmark)
        assert len(fig.data) == 3
        
        # Check trace names
        assert fig.data[0].name == 'LSTM Strategy'
        assert fig.data[1].name == 'ARIMA Strategy'
        assert fig.data[2].name == 'Buy & Hold'
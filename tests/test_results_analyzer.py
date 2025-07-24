"""
Unit tests for the ResultsAnalyzer class
"""
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.results_analyzer import ResultsAnalyzer


class TestResultsAnalyzer:
    """Test suite for ResultsAnalyzer class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        # Create date range
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create test prices
        prices = pd.Series(
            np.cumsum(np.random.normal(0, 1, 100)) + 100,  # Random walk
            index=dates
        )
        
        # Create test volume
        volume = pd.Series(
            np.random.randint(1000, 10000, 100),
            index=dates
        )
        
        # Create model results
        lstm_predictions = np.random.randint(0, 2, 100)  # Binary predictions
        arima_predictions = np.random.randint(0, 2, 100)  # Binary predictions
        y_true = np.random.randint(0, 2, 100)  # True labels
        
        model_results = {
            'LSTM': {
                'predictions': lstm_predictions,
                'probabilities': np.random.random(100),
                'y_true': y_true,
                'test_dates': dates
            },
            'ARIMA': {
                'predictions': arima_predictions,
                'y_true': y_true,
                'test_dates': dates
            }
        }
        
        # Create evaluation results
        evaluation_results = {
            'LSTM': {
                'accuracy': 0.65,
                'precision': 0.70,
                'recall': 0.60,
                'f1_score': 0.65,
                'specificity': 0.68
            },
            'ARIMA': {
                'accuracy': 0.58,
                'precision': 0.62,
                'recall': 0.55,
                'f1_score': 0.58,
                'specificity': 0.60
            }
        }
        
        # Create trading results
        lstm_equity = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.01, 100)) * 10000,
            index=dates
        )
        
        arima_equity = pd.Series(
            np.cumprod(1 + np.random.normal(0.0005, 0.01, 100)) * 10000,
            index=dates
        )
        
        trading_results = {
            'LSTM': {
                'equity_curve': lstm_equity,
                'final_equity': lstm_equity.iloc[-1],
                'total_return': lstm_equity.iloc[-1] / lstm_equity.iloc[0] - 1,
                'annualized_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.12,
                'win_rate': 0.55,
                'profit_factor': 1.3,
                'num_trades': 25
            },
            'ARIMA': {
                'equity_curve': arima_equity,
                'final_equity': arima_equity.iloc[-1],
                'total_return': arima_equity.iloc[-1] / arima_equity.iloc[0] - 1,
                'annualized_return': 0.10,
                'sharpe_ratio': 0.9,
                'max_drawdown': 0.15,
                'win_rate': 0.48,
                'profit_factor': 1.1,
                'num_trades': 30
            }
        }
        
        return {
            'prices': prices,
            'volume': volume,
            'model_results': model_results,
            'evaluation_results': evaluation_results,
            'trading_results': trading_results
        }
    
    def test_init(self):
        """Test initialization of ResultsAnalyzer"""
        analyzer = ResultsAnalyzer()
        assert hasattr(analyzer, 'logger')
        assert hasattr(analyzer, 'model_evaluator')
        assert hasattr(analyzer, 'visualization_engine')
    
    def test_analyze_model_performance(self, sample_data):
        """Test analyze_model_performance method"""
        analyzer = ResultsAnalyzer()
        
        analysis = analyzer.analyze_model_performance(
            sample_data['model_results'],
            sample_data['evaluation_results'],
            sample_data['trading_results']
        )
        
        # Check that analysis contains expected keys
        assert 'summary' in analysis
        assert 'classification_metrics' in analysis
        assert 'trading_metrics' in analysis
        assert 'key_insights' in analysis
        
        # Check that models are included
        assert 'LSTM' in analysis['classification_metrics']
        assert 'ARIMA' in analysis['classification_metrics']
        assert 'LSTM' in analysis['trading_metrics']
        assert 'ARIMA' in analysis['trading_metrics']
        
        # Check that summary contains expected keys
        assert 'best_model_overall' in analysis['summary']
        assert 'best_classification_model' in analysis['summary']
        assert 'best_trading_model' in analysis['summary']
        
        # Check that key insights are generated
        assert len(analysis['key_insights']) > 0
    
    def test_generate_performance_report_html(self, sample_data, tmpdir):
        """Test generate_performance_report method with HTML format"""
        analyzer = ResultsAnalyzer()
        
        # Temporarily change results directory to tmpdir
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs('results/reports', exist_ok=True)
        
        try:
            report_path = analyzer.generate_performance_report(
                sample_data['model_results'],
                sample_data['evaluation_results'],
                sample_data['trading_results'],
                sample_data['prices'],
                sample_data['volume'],
                report_format='html'
            )
            
            # Check that report file exists
            assert os.path.exists(report_path)
            assert report_path.endswith('.html')
            
            # Check file content
            with open(report_path, 'r') as f:
                content = f.read()
                assert 'NVDL Stock Predictor Performance Report' in content
                assert 'LSTM' in content
                assert 'ARIMA' in content
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def test_generate_performance_report_json(self, sample_data, tmpdir):
        """Test generate_performance_report method with JSON format"""
        analyzer = ResultsAnalyzer()
        
        # Temporarily change results directory to tmpdir
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs('results/reports', exist_ok=True)
        
        try:
            report_path = analyzer.generate_performance_report(
                sample_data['model_results'],
                sample_data['evaluation_results'],
                sample_data['trading_results'],
                sample_data['prices'],
                sample_data['volume'],
                report_format='json'
            )
            
            # Check that report file exists
            assert os.path.exists(report_path)
            assert report_path.endswith('.json')
            
            # Check file content
            with open(report_path, 'r') as f:
                content = f.read()
                assert 'NVDL Stock Predictor Performance Report' in content
                assert 'LSTM' in content
                assert 'ARIMA' in content
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def test_create_model_comparison_dashboard(self, sample_data):
        """Test create_model_comparison_dashboard method"""
        analyzer = ResultsAnalyzer()
        
        fig = analyzer.create_model_comparison_dashboard(
            sample_data['model_results'],
            sample_data['evaluation_results'],
            sample_data['trading_results'],
            sample_data['prices'],
            sample_data['volume']
        )
        
        # Check that figure is created
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
    
    def test_export_results_csv(self, sample_data, tmpdir):
        """Test export_results method with CSV format"""
        analyzer = ResultsAnalyzer()
        
        # Temporarily change results directory to tmpdir
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs('results', exist_ok=True)
        
        try:
            export_files = analyzer.export_results(
                sample_data['model_results'],
                sample_data['evaluation_results'],
                sample_data['trading_results'],
                export_format='csv',
                export_path=str(tmpdir.join('exports'))
            )
            
            # Check that export files exist
            assert 'classification_metrics' in export_files
            assert 'trading_metrics' in export_files
            assert 'predictions' in export_files
            
            assert os.path.exists(export_files['classification_metrics'])
            assert os.path.exists(export_files['trading_metrics'])
            assert os.path.exists(export_files['predictions'])
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    def test_export_results_excel(self, sample_data, tmpdir):
        """Test export_results method with Excel format"""
        analyzer = ResultsAnalyzer()
        
        # Temporarily change results directory to tmpdir
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs('results', exist_ok=True)
        
        try:
            export_files = analyzer.export_results(
                sample_data['model_results'],
                sample_data['evaluation_results'],
                sample_data['trading_results'],
                export_format='excel',
                export_path=str(tmpdir.join('exports'))
            )
            
            # Check that export files exist (either Excel or CSV fallback)
            if 'excel' in export_files:
                assert os.path.exists(export_files['excel'])
            else:
                # Check for CSV fallback
                assert 'classification_metrics' in export_files
                assert 'trading_metrics' in export_files
                assert 'predictions' in export_files
                assert os.path.exists(export_files['classification_metrics'])
                assert os.path.exists(export_files['trading_metrics'])
                assert os.path.exists(export_files['predictions'])
        finally:
            # Restore original directory
            os.chdir(original_dir)
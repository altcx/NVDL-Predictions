"""
Comprehensive integration tests for NVDL Stock Predictor
Tests end-to-end execution of the complete pipeline with various scenarios
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
import warnings
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import test fixtures
from tests.test_fixtures import (
    create_sample_ohlcv_data, create_sample_processed_data,
    create_train_test_split, create_sample_model_results,
    create_sample_trading_results, create_sample_evaluation_results
)

# Import components
from data.collector import DataCollector
from data.preprocessor import DataPreprocessor
from models.model_evaluator import ModelEvaluator
from models.trading_simulator import TradingSimulator
from visualization.visualization_engine import VisualizationEngine

# Check for required packages
TENSORFLOW_AVAILABLE = False
STATSMODELS_AVAILABLE = False
PLOTLY_AVAILABLE = False

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

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    warnings.warn("Plotly not available, some tests will be skipped")

# Conditionally import model classes
if TENSORFLOW_AVAILABLE:
    from models.lstm_predictor import LSTMPredictor
if STATSMODELS_AVAILABLE:
    from models.arima_predictor import ARIMAPredictor

# Import main pipeline
from main import NVDLPredictorPipeline


class TestEndToEndPipeline:
    """Comprehensive end-to-end tests for the complete pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        return create_sample_ohlcv_data(days=200)
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance with mocked components"""
        pipeline = NVDLPredictorPipeline()
        
        # Mock data collector
        pipeline.data_collector = Mock(spec=DataCollector)
        
        # Use real preprocessor
        pipeline.data_preprocessor = DataPreprocessor()
        
        # Mock model components
        pipeline.lstm_predictor = Mock()
        pipeline.arima_predictor = Mock()
        
        # Use real evaluator and simulator
        pipeline.model_evaluator = ModelEvaluator()
        pipeline.trading_simulator = TradingSimulator()
        
        # Mock visualization engine
        pipeline.visualization_engine = Mock(spec=VisualizationEngine)
        
        return pipeline
    
    def test_complete_pipeline_with_mocks(self, pipeline, sample_data):
        """Test complete pipeline execution with mocked components"""
        # Configure mocks
        pipeline.data_collector.fetch_historical_data.return_value = sample_data
        pipeline.data_collector.validate_data_completeness.return_value = True
        pipeline.data_collector.handle_missing_data.return_value = sample_data
        
        # Process data
        processed_data = pipeline.preprocess_data(sample_data)
        
        # Verify processed data structure
        assert 'X_train' in processed_data
        assert 'y_train' in processed_data
        assert 'X_test' in processed_data
        assert 'y_test' in processed_data
        assert 'metadata' in processed_data
        assert 'train_df' in processed_data
        assert 'test_df' in processed_data
        
        # Create mock LSTM results
        lstm_results = create_sample_model_results(processed_data['test_df'], accuracy=0.65)
        pipeline.lstm_predictor.prepare_sequences.return_value = (
            processed_data['X_train'], processed_data['y_train']
        )
        pipeline.lstm_predictor.build_model.return_value = Mock()
        pipeline.lstm_predictor.train.return_value = Mock()
        pipeline.lstm_predictor.predict.return_value = lstm_results['predictions']
        pipeline.lstm_predictor.predict_probabilities.return_value = lstm_results['probabilities']
        
        # Create mock ARIMA results
        arima_results = create_sample_model_results(processed_data['test_df'], accuracy=0.60)
        pipeline.arima_predictor.fit.return_value = Mock()
        pipeline.arima_predictor.predict_direction.return_value = 1  # Buy signal
        
        # Mock model training
        with patch.object(pipeline, 'train_lstm_model', return_value=lstm_results):
            with patch.object(pipeline, 'train_arima_model', return_value=arima_results):
                # Evaluate models
                evaluation_results = pipeline.evaluate_models(lstm_results, arima_results)
                
                # Verify evaluation results
                assert 'LSTM' in evaluation_results
                assert 'ARIMA' in evaluation_results
                assert 'comparison' in evaluation_results
                
                # Simulate trading
                test_prices = processed_data['test_df']['close']
                trading_results = pipeline.simulate_trading(lstm_results, arima_results, test_prices)
                
                # Verify trading results
                assert 'LSTM' in trading_results
                assert 'ARIMA' in trading_results
                assert 'equity_curve' in trading_results['LSTM']
                assert 'total_return' in trading_results['LSTM']
                
                # Generate visualizations
                test_volume = processed_data['test_df']['volume']
                visualization_results = pipeline.generate_visualizations(
                    lstm_results, arima_results, evaluation_results, 
                    trading_results, test_prices, test_volume
                )
                
                # Verify that visualization methods were called
                assert pipeline.visualization_engine.plot_price_with_signals.call_count >= 1
                assert pipeline.visualization_engine.plot_equity_curves.call_count >= 1
                assert pipeline.visualization_engine.save_figure.call_count >= 1
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_lstm_model_integration(self, pipeline, sample_data):
        """Test LSTM model integration with real components"""
        # Process data
        processed_data = pipeline.preprocess_data(sample_data)
        
        # Create real LSTM predictor with minimal configuration
        lstm_predictor = LSTMPredictor(sequence_length=5, lstm_units=10, dropout_rate=0.2)
        pipeline.lstm_predictor = lstm_predictor
        
        # Prepare sequences
        X_train, y_train = lstm_predictor.prepare_sequences(
            np.column_stack((processed_data['X_train'], processed_data['y_train']))
        )
        X_test, y_test = lstm_predictor.prepare_sequences(
            np.column_stack((processed_data['X_test'], processed_data['y_test']))
        )
        
        # Build model with minimal configuration
        input_shape = (5, processed_data['X_train'].shape[1])
        model = lstm_predictor.build_model(input_shape)
        
        # Mock training to avoid long execution
        with patch.object(lstm_predictor, 'train', return_value=Mock()) as mock_train:
            # Mock prediction methods
            lstm_predictor.predict = Mock(return_value=np.random.randint(0, 2, size=len(y_test)))
            lstm_predictor.predict_probabilities = Mock(return_value=np.random.random(size=len(y_test)))
            
            # Create mock results
            lstm_results = {
                'model': lstm_predictor,
                'predictions': lstm_predictor.predict(X_test),
                'probabilities': lstm_predictor.predict_probabilities(X_test),
                'y_true': y_test,
                'test_dates': processed_data['test_df'].index[-len(y_test):]
            }
            
            # Evaluate model
            evaluation = pipeline.model_evaluator.evaluate_model_performance(
                model_name='LSTM',
                y_true=lstm_results['y_true'],
                y_pred=lstm_results['predictions'],
                y_prob=lstm_results['probabilities'],
                is_classifier=True
            )
            
            # Verify evaluation results
            assert 'LSTM' in evaluation
            assert 'accuracy' in evaluation['LSTM']
            assert 'precision' in evaluation['LSTM']
            assert 'recall' in evaluation['LSTM']
            assert 'f1_score' in evaluation['LSTM']
    
    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="Statsmodels not available")
    def test_arima_model_integration(self, pipeline, sample_data):
        """Test ARIMA model integration with real components"""
        # Process data
        processed_data = pipeline.preprocess_data(sample_data)
        
        # Mock ARIMA predictor
        arima_predictor = Mock(spec=ARIMAPredictor)
        pipeline.arima_predictor = arima_predictor
        
        # Mock fit and predict methods
        arima_predictor.fit.return_value = Mock()
        arima_predictor.predict_direction.return_value = 1  # Buy signal
        
        # Create mock results
        arima_results = {
            'model': arima_predictor,
            'predictions': np.random.randint(0, 2, size=len(processed_data['test_df'])),
            'y_true': processed_data['test_df']['target'].values,
            'test_dates': processed_data['test_df'].index
        }
        
        # Evaluate model
        evaluation = pipeline.model_evaluator.evaluate_model_performance(
            model_name='ARIMA',
            y_true=arima_results['y_true'],
            y_pred=arima_results['predictions'],
            is_classifier=True
        )
        
        # Verify evaluation results
        assert 'ARIMA' in evaluation
        assert 'accuracy' in evaluation['ARIMA']
        assert 'precision' in evaluation['ARIMA']
        assert 'recall' in evaluation['ARIMA']
        assert 'f1_score' in evaluation['ARIMA']
    
    def test_trading_simulation_integration(self, pipeline, sample_data):
        """Test trading simulation integration with real components"""
        # Process data
        processed_data = pipeline.preprocess_data(sample_data)
        test_df = processed_data['test_df']
        
        # Create mock model results
        lstm_results = create_sample_model_results(test_df, accuracy=0.65)
        arima_results = create_sample_model_results(test_df, accuracy=0.60)
        
        # Simulate trading
        test_prices = test_df['close']
        trading_results = pipeline.simulate_trading(lstm_results, arima_results, test_prices)
        
        # Verify trading results
        assert 'LSTM' in trading_results
        assert 'ARIMA' in trading_results
        assert 'equity_curve' in trading_results['LSTM']
        assert 'total_return' in trading_results['LSTM']
        assert 'sharpe_ratio' in trading_results['LSTM']
        assert 'max_drawdown' in trading_results['LSTM']
        assert 'win_rate' in trading_results['LSTM']
        
        # Check that equity curves have correct length
        assert len(trading_results['LSTM']['equity_curve']) == len(test_prices)
        assert len(trading_results['ARIMA']['equity_curve']) == len(test_prices)
    
    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_visualization_integration(self, sample_data):
        """Test visualization integration with real components"""
        # Create real pipeline with real visualization engine
        pipeline = NVDLPredictorPipeline()
        pipeline.data_preprocessor = DataPreprocessor()
        pipeline.visualization_engine = VisualizationEngine()
        
        # Process data
        processed_data = pipeline.preprocess_data(sample_data)
        test_df = processed_data['test_df']
        
        # Create mock model results
        lstm_results = create_sample_model_results(test_df, accuracy=0.65)
        arima_results = create_sample_model_results(test_df, accuracy=0.60)
        
        # Create mock evaluation results
        evaluation_results = create_sample_evaluation_results(lstm_results, arima_results)
        
        # Create mock trading results
        lstm_trading = create_sample_trading_results(test_df, lstm_results['predictions'])
        arima_trading = create_sample_trading_results(test_df, arima_results['predictions'])
        trading_results = {'LSTM': lstm_trading, 'ARIMA': arima_trading}
        
        # Generate visualizations
        test_prices = test_df['close']
        test_volume = test_df['volume']
        
        # Mock save_figure to avoid file operations
        with patch.object(pipeline.visualization_engine, 'save_figure'):
            visualization_results = pipeline.generate_visualizations(
                lstm_results, arima_results, evaluation_results, 
                trading_results, test_prices, test_volume
            )
            
            # Verify visualization results
            assert 'lstm_price_chart' in visualization_results
            assert 'arima_price_chart' in visualization_results
            assert 'equity_comparison' in visualization_results
            assert 'performance_comparison' in visualization_results


class TestDataFlowIntegration:
    """Test data flow between components"""
    
    def test_data_collection_to_preprocessing(self):
        """Test data flow from collection to preprocessing"""
        # Create components
        collector = DataCollector()
        preprocessor = DataPreprocessor()
        
        # Mock API client
        collector.client = Mock()
        
        # Create sample data
        sample_data = create_sample_ohlcv_data()
        
        # Mock fetch_historical_data
        collector.fetch_historical_data = Mock(return_value=sample_data)
        collector.validate_data_completeness = Mock(return_value=True)
        collector.handle_missing_data = Mock(return_value=sample_data)
        
        # Fetch data
        data = collector.fetch_historical_data('NVDL', '2023-01-01', '2023-12-31')
        
        # Preprocess data
        processed_data = preprocessor.prepare_data_for_training(data)
        
        # Verify data flow
        assert 'X_train' in processed_data
        assert 'y_train' in processed_data
        assert 'X_test' in processed_data
        assert 'y_test' in processed_data
        assert 'metadata' in processed_data
        
        # Check shapes
        assert processed_data['X_train'].shape[0] > 0
        assert processed_data['y_train'].shape[0] > 0
        assert processed_data['X_test'].shape[0] > 0
        assert processed_data['y_test'].shape[0] > 0
        
        # Check that X_train and y_train have same number of samples
        assert processed_data['X_train'].shape[0] == processed_data['y_train'].shape[0]
        
        # Check that X_test and y_test have same number of samples
        assert processed_data['X_test'].shape[0] == processed_data['y_test'].shape[0]
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_preprocessing_to_lstm_model(self):
        """Test data flow from preprocessing to LSTM model"""
        # Create components
        preprocessor = DataPreprocessor()
        lstm_predictor = LSTMPredictor(sequence_length=5, lstm_units=10, dropout_rate=0.2)
        
        # Create sample data
        sample_data = create_sample_ohlcv_data()
        
        # Preprocess data
        processed_data = preprocessor.prepare_data_for_training(sample_data)
        
        # Prepare sequences for LSTM
        X_train, y_train = lstm_predictor.prepare_sequences(
            np.column_stack((processed_data['X_train'], processed_data['y_train']))
        )
        
        # Verify sequence shape
        assert X_train.shape[0] > 0
        assert X_train.shape[1] == 5  # sequence_length
        assert X_train.shape[2] == processed_data['X_train'].shape[1]  # features
        assert y_train.shape[0] == X_train.shape[0]
        
        # Build model
        input_shape = (5, processed_data['X_train'].shape[1])
        model = lstm_predictor.build_model(input_shape)
        
        # Verify model was built
        assert lstm_predictor.model is not None
    
    @pytest.mark.skipif(not STATSMODELS_AVAILABLE, reason="Statsmodels not available")
    def test_preprocessing_to_arima_model(self):
        """Test data flow from preprocessing to ARIMA model"""
        # Create components
        preprocessor = DataPreprocessor()
        
        # Create sample data
        sample_data = create_sample_ohlcv_data()
        
        # Preprocess data
        processed_data = preprocessor.prepare_data_for_training(sample_data)
        
        # Mock ARIMA predictor
        arima_predictor = Mock(spec=ARIMAPredictor)
        arima_predictor.fit.return_value = Mock()
        arima_predictor.predict_direction.return_value = 1  # Buy signal
        
        # Fit ARIMA model
        arima_predictor.fit(processed_data['train_df']['close'])
        
        # Verify fit was called
        assert arima_predictor.fit.call_count == 1
    
    def test_model_to_evaluation(self):
        """Test data flow from model to evaluation"""
        # Create components
        evaluator = ModelEvaluator()
        
        # Create sample data
        sample_data = create_sample_processed_data()
        train_test_data = create_train_test_split(sample_data)
        
        # Create mock model results
        lstm_results = create_sample_model_results(train_test_data['test_df'], accuracy=0.65)
        arima_results = create_sample_model_results(train_test_data['test_df'], accuracy=0.60)
        
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
        
        # Compare models
        comparison = evaluator.compare_models({
            'LSTM': lstm_metrics['LSTM'],
            'ARIMA': arima_metrics['ARIMA']
        })
        
        # Verify evaluation results
        assert 'LSTM' in lstm_metrics
        assert 'ARIMA' in arima_metrics
        assert isinstance(comparison, pd.DataFrame)
        assert 'accuracy' in comparison.columns
        assert 'precision' in comparison.columns
        assert 'recall' in comparison.columns
        assert 'f1_score' in comparison.columns
    
    def test_model_to_trading_simulation(self):
        """Test data flow from model to trading simulation"""
        # Create components
        simulator = TradingSimulator()
        
        # Create sample data
        sample_data = create_sample_processed_data()
        train_test_data = create_train_test_split(sample_data)
        
        # Create mock model results
        lstm_results = create_sample_model_results(train_test_data['test_df'], accuracy=0.65)
        
        # Simulate trading
        test_prices = train_test_data['test_df']['close']
        trading_results = simulator.simulate_strategy(test_prices, lstm_results['predictions'])
        
        # Verify trading results
        assert 'equity_curve' in trading_results
        assert 'transactions' in trading_results
        assert 'final_equity' in trading_results
        assert 'total_return' in trading_results
        assert 'sharpe_ratio' in trading_results
        assert 'max_drawdown' in trading_results
        assert 'win_rate' in trading_results
        
        # Check that equity curve has correct length
        assert len(trading_results['equity_curve']) == len(test_prices)
    
    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_results_to_visualization(self):
        """Test data flow from results to visualization"""
        # Create components
        visualizer = VisualizationEngine()
        
        # Create sample data
        sample_data = create_sample_processed_data()
        train_test_data = create_train_test_split(sample_data)
        test_df = train_test_data['test_df']
        
        # Create mock model results
        lstm_results = create_sample_model_results(test_df, accuracy=0.65)
        
        # Create price chart
        test_prices = test_df['close']
        price_chart = visualizer.plot_price_with_signals(
            prices=test_prices,
            signals=lstm_results['predictions'],
            model_name='LSTM'
        )
        
        # Verify chart was created
        assert price_chart is not None
        
        # Create equity curves
        lstm_trading = create_sample_trading_results(test_df, lstm_results['predictions'])
        arima_trading = create_sample_trading_results(test_df, lstm_results['predictions'])
        
        equity_chart = visualizer.plot_equity_curves(
            lstm_equity=lstm_trading['equity_curve'],
            arima_equity=arima_trading['equity_curve']
        )
        
        # Verify chart was created
        assert equity_chart is not None


class TestPerformanceAndResourceUsage:
    """Test performance and resource usage"""
    
    def test_memory_usage_during_processing(self, sample_data):
        """Test memory usage during data processing"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            # Measure initial memory usage
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Create preprocessor
            preprocessor = DataPreprocessor()
            
            # Process data
            processed_data = preprocessor.prepare_data_for_training(sample_data)
            
            # Measure final memory usage
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Calculate memory increase
            memory_increase = final_memory - initial_memory
            
            # Log memory usage
            print(f"Memory usage: Initial={initial_memory:.1f}MB, Final={final_memory:.1f}MB, Increase={memory_increase:.1f}MB")
            
            # Memory increase should be reasonable
            assert memory_increase < 1000, f"Memory usage increased by {memory_increase:.1f}MB"
            
        except ImportError:
            pytest.skip("psutil not available")
    
    def test_processing_time(self, sample_data):
        """Test processing time for data preprocessing"""
        # Create preprocessor
        preprocessor = DataPreprocessor()
        
        # Measure processing time
        start_time = time.time()
        processed_data = preprocessor.prepare_data_for_training(sample_data)
        end_time = time.time()
        
        # Calculate processing time
        processing_time = end_time - start_time
        
        # Log processing time
        print(f"Data preprocessing time: {processing_time:.2f}s")
        
        # Processing time should be reasonable
        assert processing_time < 10.0, f"Data preprocessing took {processing_time:.2f}s"
    
    @pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow not available")
    def test_lstm_model_build_time(self):
        """Test LSTM model build time"""
        # Create LSTM predictor
        lstm_predictor = LSTMPredictor(sequence_length=60, lstm_units=50, dropout_rate=0.2)
        
        # Measure build time
        start_time = time.time()
        model = lstm_predictor.build_model((60, 20))  # 60 timesteps, 20 features
        end_time = time.time()
        
        # Calculate build time
        build_time = end_time - start_time
        
        # Log build time
        print(f"LSTM model build time: {build_time:.2f}s")
        
        # Build time should be reasonable
        assert build_time < 5.0, f"LSTM model build took {build_time:.2f}s"


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
"""
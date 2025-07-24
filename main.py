"""
Main execution pipeline for NVDL Stock Predictor
Orchestrates all components from data collection to visualization
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import warnings
import sys
import traceback
from typing import Dict, Any, Optional, Tuple, List, Callable

# Import configuration and logger
from config import config
from utils.logger import (
    get_main_logger, get_data_logger, get_model_logger, 
    get_evaluation_logger, get_trading_logger, get_visualization_logger,
    LogContext, log_memory_usage, get_recent_errors
)

# Import error handling utilities
from utils.error_handler import (
    ErrorHandler, retry_on_exception, safe_execute, log_execution_time,
    RetryableError, APIConnectionError, DataValidationError, ModelTrainingError,
    ConfigurationError, VisualizationError, ResourceNotFoundError, DataProcessingError,
    ErrorContext
)
from utils.error_handling_integration import (
    setup_error_handling, handle_api_failure, handle_data_validation,
    handle_model_training, handle_visualization, log_execution_with_progress,
    monitor_system_resources, create_error_report
)

# Import components
from data.collector import DataCollector
from data.preprocessor import DataPreprocessor
from models.model_evaluator import ModelEvaluator
from models.trading_simulator import TradingSimulator
from visualization.visualization_engine import VisualizationEngine
from utils.results_analyzer import ResultsAnalyzer

# Check for required packages
TENSORFLOW_AVAILABLE = False
STATSMODELS_AVAILABLE = False

try:
    from models.lstm_predictor import LSTMPredictor
    TENSORFLOW_AVAILABLE = True
except ImportError:
    warnings.warn("TensorFlow not available, LSTM model will not be available")

try:
    from models.arima_predictor import ARIMAPredictor
    STATSMODELS_AVAILABLE = True
except ImportError:
    warnings.warn("Statsmodels not available, ARIMA model will not be available")


class NVDLPredictorPipeline:
    """
    Main execution pipeline that orchestrates all components
    """
    
    def __init__(self):
        """Initialize pipeline with logger"""
        self.logger = get_main_logger()
        self.logger.info("Initializing NVDL Stock Predictor Pipeline")
        
        # Initialize components
        self.data_collector = None
        self.data_preprocessor = None
        self.lstm_predictor = None
        self.arima_predictor = None
        self.model_evaluator = ModelEvaluator()
        self.trading_simulator = TradingSimulator()
        self.visualization_engine = VisualizationEngine()
        self.results_analyzer = ResultsAnalyzer()
        
        # Create directories if they don't exist
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        self.logger.info("Pipeline initialized successfully")
    
    @handle_api_failure
    def collect_data(self) -> pd.DataFrame:
        """
        Collect historical stock data from Alpaca Markets API
        
        Returns:
            DataFrame with OHLCV data
        """
        self.logger.info("Starting data collection")
        
        # Initialize data collector
        self.data_collector = DataCollector()
        
        # Calculate date range
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * config.LOOKBACK_YEARS)).strftime('%Y-%m-%d')
        
        # Create progress logger for data collection
        progress = get_progress_logger("data", 100)
        progress.start(f"Fetching data for {config.SYMBOL}")
        
        try:
            # Fetch data
            self.logger.info(f"Fetching data for {config.SYMBOL} from {start_date} to {end_date}")
            data = self.data_collector.fetch_historical_data(
                symbol=config.SYMBOL,
                start_date=start_date,
                end_date=end_date
            )
            
            progress.update(50, "Data fetched, validating...")
            
            # Validate data
            if not self.data_collector.validate_data_completeness(data):
                self.logger.error("Data validation failed")
                raise DataValidationError("Collected data failed validation checks")
            
            progress.update(75, "Data validated, handling missing values...")
            
            # Handle missing data
            data = self.data_collector.handle_missing_data(data)
            
            progress.complete(f"Data collection completed: {len(data)} data points")
            self.logger.info(f"Data collection completed: {len(data)} data points")
            return data
            
        except Exception as e:
            progress.update(100, f"Data collection failed: {str(e)}")
            raise
    
    @handle_data_validation
    @log_execution_with_progress(total_steps=100)
    def preprocess_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Preprocess data and prepare for model training
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Dictionary with processed data and metadata
        """
        self.logger.info("Starting data preprocessing")
        
        # Initialize preprocessor
        self.data_preprocessor = DataPreprocessor()
        
        # Create progress logger
        progress = get_progress_logger("data", 100)
        progress.start("Data preprocessing")
        
        try:
            # Log data summary before preprocessing
            self.logger.info(f"Raw data summary: {len(data)} rows, {data.columns.tolist()} columns")
            
            # Check for data anomalies
            null_counts = data.isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Raw data contains null values: {null_counts.to_dict()}")
            
            progress.update(25, "Creating features...")
            
            # Complete preprocessing pipeline
            processed_data = self.data_preprocessor.prepare_data_for_training(
                data=data,
                test_size=config.TEST_SIZE
            )
            
            progress.update(90, "Validating processed data...")
            
            # Validate processed data
            if 'X_train' not in processed_data or 'y_train' not in processed_data:
                raise DataValidationError("Preprocessing failed to produce required training data")
            
            if len(processed_data['X_train']) == 0 or len(processed_data['y_train']) == 0:
                raise DataValidationError("Preprocessing produced empty training data")
            
            # Log preprocessing results
            self.logger.info(f"Preprocessing completed: {len(processed_data['X_train'])} training samples, "
                           f"{len(processed_data['X_test'])} test samples")
            
            progress.complete("Data preprocessing completed")
            return processed_data
            
        except Exception as e:
            progress.update(100, f"Data preprocessing failed: {str(e)}")
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    @handle_model_training
    def train_lstm_model(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train LSTM neural network model
        
        Args:
            processed_data: Dictionary with processed training data
            
        Returns:
            Dictionary with model and predictions
        """
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available, cannot train LSTM model")
            raise ImportError("TensorFlow not available, cannot train LSTM model")
            
        self.logger.info("Starting LSTM model training")
        
        # Create progress logger for model training
        progress = get_progress_logger("models", config.LSTM_EPOCHS)
        progress.start("LSTM model training")
        
        try:
            # Extract data
            X_train = processed_data['X_train']
            y_train = processed_data['y_train']
            X_test = processed_data['X_test']
            y_test = processed_data['y_test']
            metadata = processed_data['metadata']
            
            # Log data shapes
            self.logger.info(f"Training data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
            self.logger.info(f"Testing data shapes: X_test={X_test.shape}, y_test={y_test.shape}")
            
            progress.update(5, "Initializing LSTM model...")
            
            # Initialize LSTM predictor
            self.lstm_predictor = LSTMPredictor(
                sequence_length=config.LSTM_SEQUENCE_LENGTH,
                lstm_units=config.LSTM_UNITS,
                dropout_rate=config.LSTM_DROPOUT
            )
            
            progress.update(10, "Preparing sequences...")
            
            # Prepare sequences for LSTM
            try:
                X_train_seq, y_train_seq = self.lstm_predictor.prepare_sequences(
                    np.column_stack((X_train, y_train))
                )
                
                X_test_seq, y_test_seq = self.lstm_predictor.prepare_sequences(
                    np.column_stack((X_test, y_test))
                )
            except Exception as e:
                raise ModelTrainingError(f"Failed to prepare sequences: {str(e)}", model_type="LSTM")
            
            progress.update(15, "Building model architecture...")
            
            # Build model
            try:
                input_shape = (config.LSTM_SEQUENCE_LENGTH, X_train.shape[1])
                self.lstm_predictor.build_model(input_shape)
            except Exception as e:
                raise ModelTrainingError(f"Failed to build model: {str(e)}", model_type="LSTM")
            
            progress.update(20, "Starting model training...")
            
            # Create a callback to update progress
            class ProgressCallback:
                def __init__(self, progress_logger):
                    self.progress_logger = progress_logger
                    self.base_progress = 20  # Start at 20%
                    
                def on_epoch_end(self, epoch, logs=None):
                    epoch_progress = int(self.base_progress + (epoch + 1) / config.LSTM_EPOCHS * 60)
                    metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in logs.items()])
                    self.progress_logger.update(epoch_progress, f"Epoch {epoch+1}/{config.LSTM_EPOCHS}: {metrics_str}")
            
            progress_callback = ProgressCallback(progress)
            
            # Train model with progress tracking
            try:
                history = self.lstm_predictor.train(
                    X_train=X_train_seq,
                    y_train=y_train_seq,
                    validation_data=(X_test_seq, y_test_seq),
                    epochs=config.LSTM_EPOCHS,
                    batch_size=config.LSTM_BATCH_SIZE,
                    checkpoint_dir='./checkpoints',
                    custom_callback=progress_callback
                )
            except Exception as e:
                raise ModelTrainingError(f"Model training failed: {str(e)}", model_type="LSTM")
            
            progress.update(85, "Generating predictions...")
            
            # Generate predictions
            try:
                predictions = self.lstm_predictor.predict(X_test_seq)
                probabilities = self.lstm_predictor.predict_probabilities(X_test_seq)
            except Exception as e:
                raise ModelTrainingError(f"Failed to generate predictions: {str(e)}", model_type="LSTM")
            
            progress.update(95, "Saving model...")
            
            # Save model
            try:
                self.lstm_predictor.save_model('./checkpoints/lstm_model_final.h5')
            except Exception as e:
                self.logger.warning(f"Failed to save model: {str(e)}")
            
            progress.complete("LSTM model training completed")
            self.logger.info("LSTM model training completed successfully")
            
            # Log training metrics
            if history and hasattr(history, 'history'):
                final_epoch = len(history.history.get('loss', []))
                final_loss = history.history.get('loss', [])[-1] if history.history.get('loss') else None
                final_accuracy = history.history.get('accuracy', [])[-1] if history.history.get('accuracy') else None
                
                self.logger.info(f"Training completed after {final_epoch} epochs")
                self.logger.info(f"Final training metrics - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
            
            return {
                'model': self.lstm_predictor,
                'predictions': predictions,
                'probabilities': probabilities,
                'y_true': y_test_seq,
                'history': history.history if history else {},
                'test_dates': metadata['test_dates'][config.LSTM_SEQUENCE_LENGTH:],
                'train_dates': metadata['train_dates'][config.LSTM_SEQUENCE_LENGTH:]
            }
            
        except Exception as e:
            progress.update(100, f"LSTM model training failed: {str(e)}")
            if isinstance(e, ModelTrainingError):
                raise
            else:
                raise ModelTrainingError(f"Unexpected error during LSTM training: {str(e)}", model_type="LSTM") from e
    
    @handle_model_training
    def train_arima_model(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train ARIMA time series model
        
        Args:
            processed_data: Dictionary with processed training data
            
        Returns:
            Dictionary with model and predictions
        """
        if not STATSMODELS_AVAILABLE:
            self.logger.error("Statsmodels not available, cannot train ARIMA model")
            raise ImportError("Statsmodels not available, cannot train ARIMA model")
            
        self.logger.info("Starting ARIMA model training")
        
        # Create progress logger for ARIMA training
        progress = get_progress_logger("models", 100)
        progress.start("ARIMA model training")
        
        try:
            # Extract data
            train_df = processed_data['train_df']
            test_df = processed_data['test_df']
            
            # Log data shapes
            self.logger.info(f"Training data: {len(train_df)} samples")
            self.logger.info(f"Testing data: {len(test_df)} samples")
            
            progress.update(10, "Initializing ARIMA model...")
            
            # Initialize ARIMA predictor
            self.arima_predictor = ARIMAPredictor(
                max_p=config.ARIMA_MAX_P,
                max_d=config.ARIMA_MAX_D,
                max_q=config.ARIMA_MAX_Q
            )
            
            progress.update(20, "Finding optimal ARIMA parameters...")
            
            # Fit model on training data with error handling
            try:
                self.arima_predictor.fit(train_df['close'])
                
                # Log the selected ARIMA order
                order = self.arima_predictor.get_order()
                self.logger.info(f"Selected ARIMA order: {order}")
                
            except Exception as e:
                raise ModelTrainingError(
                    f"ARIMA model fitting failed: {str(e)}",
                    model_type="ARIMA"
                )
            
            progress.update(60, "Generating predictions...")
            
            # Generate predictions for test data with error handling
            try:
                predictions = []
                total_test_samples = len(test_df)
                
                for i in range(total_test_samples - 1):
                    # Update progress periodically
                    if i % max(1, total_test_samples // 10) == 0:
                        progress_pct = 60 + int((i / total_test_samples) * 30)
                        progress.update(progress_pct, f"Predicting sample {i+1}/{total_test_samples}")
                    
                    # Use current price to predict direction
                    current_price = test_df['close'].iloc[i]
                    direction = self.arima_predictor.predict_direction(current_price)
                    predictions.append(direction)
                
                # Add one more prediction to match length
                predictions.append(predictions[-1] if predictions else 0)
                
            except Exception as e:
                raise ModelTrainingError(
                    f"ARIMA prediction generation failed: {str(e)}",
                    model_type="ARIMA"
                )
            
            progress.update(95, "Saving model...")
            
            # Save model with error handling
            try:
                self.arima_predictor.save_model('./checkpoints/arima_model_final.pkl')
            except Exception as e:
                self.logger.warning(f"Failed to save ARIMA model: {str(e)}")
            
            progress.complete("ARIMA model training completed")
            self.logger.info("ARIMA model training completed successfully")
            
            # Log prediction statistics
            if predictions:
                pred_array = np.array(predictions)
                up_pct = np.mean(pred_array == 1) * 100
                down_pct = np.mean(pred_array == 0) * 100
                self.logger.info(f"Prediction statistics: {up_pct:.1f}% up, {down_pct:.1f}% down")
            
            return {
                'model': self.arima_predictor,
                'predictions': np.array(predictions),
                'y_true': test_df['target'].values,
                'test_dates': test_df.index
            }
            
        except Exception as e:
            progress.update(100, f"ARIMA model training failed: {str(e)}")
            if isinstance(e, ModelTrainingError):
                raise
            else:
                raise ModelTrainingError(f"Unexpected error during ARIMA training: {str(e)}", model_type="ARIMA") from e
    
    def evaluate_models(
        self, 
        lstm_results: Dict[str, Any], 
        arima_results: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate and compare model performance
        
        Args:
            lstm_results: Results from LSTM model
            arima_results: Results from ARIMA model
            
        Returns:
            Dictionary with evaluation metrics for both models
        """
        self.logger.info("Starting model evaluation")
        
        # Evaluate LSTM model
        lstm_metrics = self.model_evaluator.evaluate_model_performance(
            model_name='LSTM',
            y_true=lstm_results['y_true'],
            y_pred=lstm_results['predictions'],
            y_prob=lstm_results['probabilities'],
            is_classifier=True
        )
        
        # Evaluate ARIMA model
        arima_metrics = self.model_evaluator.evaluate_model_performance(
            model_name='ARIMA',
            y_true=arima_results['y_true'],
            y_pred=arima_results['predictions'],
            is_classifier=True
        )
        
        # Compare models
        comparison = self.model_evaluator.compare_models({
            'LSTM': lstm_metrics['LSTM'],
            'ARIMA': arima_metrics['ARIMA']
        })
        
        self.logger.info("Model evaluation completed")
        
        return {
            'LSTM': lstm_metrics['LSTM'],
            'ARIMA': arima_metrics['ARIMA'],
            'comparison': comparison
        }
    
    def simulate_trading(
        self, 
        lstm_results: Dict[str, Any], 
        arima_results: Dict[str, Any],
        test_prices: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Simulate trading strategies based on model predictions
        
        Args:
            lstm_results: Results from LSTM model
            arima_results: Results from ARIMA model
            test_prices: Test set price data
            
        Returns:
            Dictionary with trading simulation results for both models
        """
        self.logger.info("Starting trading simulation")
        
        # Align dates for LSTM predictions
        lstm_dates = lstm_results['test_dates']
        lstm_signals = lstm_results['predictions']
        
        # Get prices for LSTM dates
        lstm_prices = test_prices.loc[lstm_dates]
        
        # Simulate LSTM strategy
        lstm_trading_results = self.trading_simulator.simulate_strategy(
            prices=lstm_prices,
            signals=lstm_signals
        )
        
        # Align dates for ARIMA predictions
        arima_dates = arima_results['test_dates']
        arima_signals = arima_results['predictions']
        
        # Get prices for ARIMA dates
        arima_prices = test_prices.loc[arima_dates]
        
        # Simulate ARIMA strategy
        arima_trading_results = self.trading_simulator.simulate_strategy(
            prices=arima_prices,
            signals=arima_signals
        )
        
        self.logger.info("Trading simulation completed")
        
        return {
            'LSTM': lstm_trading_results,
            'ARIMA': arima_trading_results
        }
    
    def analyze_and_report_results(
        self,
        model_results: Dict[str, Dict[str, Any]],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]],
        test_prices: pd.Series,
        test_volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze model performance and generate comprehensive reports
        
        Args:
            model_results: Dictionary with model predictions and data
            evaluation_results: Dictionary with model evaluation metrics
            trading_results: Dictionary with trading simulation results
            test_prices: Test set price data
            test_volume: Test set volume data
            
        Returns:
            Dictionary with analysis results and report paths
        """
        self.logger.info("Starting comprehensive results analysis and reporting")
        
        # Create progress logger for analysis
        progress = get_progress_logger("analysis", 100)
        progress.start("Analyzing model performance")
        
        try:
            # Perform comprehensive analysis
            analysis = self.results_analyzer.analyze_model_performance(
                model_results, evaluation_results, trading_results
            )
            
            progress.update(30, "Creating model comparison dashboard")
            
            # Create interactive dashboard
            dashboard = self.results_analyzer.create_model_comparison_dashboard(
                model_results, evaluation_results, trading_results,
                test_prices, test_volume, save_path="results/model_comparison_dashboard"
            )
            
            progress.update(60, "Generating performance reports")
            
            # Generate reports in different formats
            html_report = self.results_analyzer.generate_performance_report(
                model_results, evaluation_results, trading_results,
                test_prices, test_volume, report_format='html'
            )
            
            json_report = self.results_analyzer.generate_performance_report(
                model_results, evaluation_results, trading_results,
                test_prices, test_volume, report_format='json'
            )
            
            md_report = self.results_analyzer.generate_performance_report(
                model_results, evaluation_results, trading_results,
                test_prices, test_volume, report_format='md'
            )
            
            progress.update(80, "Exporting results data")
            
            # Export results in different formats
            csv_exports = self.results_analyzer.export_results(
                model_results, evaluation_results, trading_results,
                export_format='csv'
            )
            
            excel_exports = self.results_analyzer.export_results(
                model_results, evaluation_results, trading_results,
                export_format='excel'
            )
            
            progress.complete("Results analysis and reporting completed")
            
            # Compile results
            results = {
                'analysis': analysis,
                'dashboard': dashboard,
                'reports': {
                    'html': html_report,
                    'json': json_report,
                    'markdown': md_report
                },
                'exports': {
                    'csv': csv_exports,
                    'excel': excel_exports
                }
            }
            
            self.logger.info("Results analysis and reporting completed successfully")
            return results
            
        except Exception as e:
            progress.update(100, f"Results analysis failed: {str(e)}")
            self.logger.error(f"Results analysis failed: {str(e)}")
            raise
    
    @handle_visualization
    def generate_visualizations(
        self, 
        lstm_results: Optional[Dict[str, Any]], 
        arima_results: Optional[Dict[str, Any]],
        evaluation_results: Dict[str, Dict[str, Any]],
        trading_results: Dict[str, Dict[str, Any]],
        test_prices: pd.Series,
        test_volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Generate comprehensive visualizations
        
        Args:
            lstm_results: Results from LSTM model (or None if not available)
            arima_results: Results from ARIMA model (or None if not available)
            evaluation_results: Model evaluation results
            trading_results: Trading simulation results
            test_prices: Test set price data
            test_volume: Test set volume data
            
        Returns:
            Dictionary with visualization figures
        """
        self.logger.info("Generating visualizations")
        
        # Create progress logger for visualization generation
        progress = get_progress_logger("visualization", 100)
        progress.start("Generating visualizations")
        
        visualization_results = {}
        
        # Create price charts with signals
        if lstm_results is not None and 'test_dates' in lstm_results:
            lstm_price_chart = self.visualization_engine.plot_price_with_signals(
                prices=test_prices.loc[lstm_results['test_dates']],
                signals=lstm_results['predictions'],
                model_name='LSTM',
                show_volume=True,
                volume_data=test_volume.loc[lstm_results['test_dates']]
            )
            visualization_results['lstm_price_chart'] = lstm_price_chart
            self.visualization_engine.save_figure(lstm_price_chart, 'results/lstm_price_chart')
        
        if arima_results is not None and 'test_dates' in arima_results:
            arima_price_chart = self.visualization_engine.plot_price_with_signals(
                prices=test_prices.loc[arima_results['test_dates']],
                signals=arima_results['predictions'],
                model_name='ARIMA',
                show_volume=True,
                volume_data=test_volume.loc[arima_results['test_dates']]
            )
            visualization_results['arima_price_chart'] = arima_price_chart
            self.visualization_engine.save_figure(arima_price_chart, 'results/arima_price_chart')
        
        # Create equity curve comparison if both models are available
        if 'LSTM' in trading_results and 'ARIMA' in trading_results:
            equity_comparison = self.visualization_engine.plot_equity_curves(
                lstm_equity=trading_results['LSTM']['equity_curve'],
                arima_equity=trading_results['ARIMA']['equity_curve']
            )
            visualization_results['equity_comparison'] = equity_comparison
            self.visualization_engine.save_figure(equity_comparison, 'results/equity_comparison')
        
        # Create model performance comparison if both models are available
        if 'LSTM' in evaluation_results and 'ARIMA' in evaluation_results:
            metrics_df = pd.DataFrame({
                'LSTM': evaluation_results['LSTM'],
                'ARIMA': evaluation_results['ARIMA']
            }).T
            
            performance_comparison = self.visualization_engine.plot_model_comparison(metrics_df)
            visualization_results['performance_comparison'] = performance_comparison
            self.visualization_engine.save_figure(performance_comparison, 'results/performance_comparison')
        
        # Create confusion matrices
        if lstm_results is not None and 'y_true' in lstm_results:
            lstm_cm = self.model_evaluator.plot_confusion_matrix(
                y_true=lstm_results['y_true'],
                y_pred=lstm_results['predictions'],
                model_name='LSTM'
            )
            visualization_results['lstm_confusion_matrix'] = lstm_cm
        
        if arima_results is not None and 'y_true' in arima_results:
            arima_cm = self.model_evaluator.plot_confusion_matrix(
                y_true=arima_results['y_true'],
                y_pred=arima_results['predictions'],
                model_name='ARIMA'
            )
            visualization_results['arima_confusion_matrix'] = arima_cm
        
        # Create comprehensive dashboard
        dashboard_results = {}
        
        if 'LSTM' in evaluation_results and 'LSTM' in trading_results and lstm_results is not None:
            dashboard_results['LSTM'] = {
                **evaluation_results['LSTM'], 
                **trading_results['LSTM'], 
                'predictions': lstm_results['predictions']
            }
            
        if 'ARIMA' in evaluation_results and 'ARIMA' in trading_results and arima_results is not None:
            dashboard_results['ARIMA'] = {
                **evaluation_results['ARIMA'], 
                **trading_results['ARIMA'], 
                'predictions': arima_results['predictions']
            }
        
        if dashboard_results:
            dashboard = self.visualization_engine.create_dashboard(
                results=dashboard_results,
                prices=test_prices,
                volume=test_volume
            )
            visualization_results['dashboard'] = dashboard
            self.visualization_engine.save_figure(dashboard, 'results/dashboard')
        
        progress.complete("Visualizations generated and saved to results directory")
        self.logger.info("Visualizations generated and saved to results directory")
        return visualization_results
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete pipeline from data collection to visualization
        
        Returns:
            Dictionary with all results
        """
        self.logger.info("Starting complete pipeline execution")
        
        try:
            # Step 1: Collect data
            raw_data = self.collect_data()
            
            # Step 2: Preprocess data
            processed_data = self.preprocess_data(raw_data)
            
            # Extract test prices for later use
            test_df = processed_data['test_df']
            test_prices = test_df['close']
            test_volume = test_df['volume']
            
            # Initialize results
            lstm_results = None
            arima_results = None
            
            # Step 3: Train LSTM model if available
            if TENSORFLOW_AVAILABLE:
                try:
                    lstm_results = self.train_lstm_model(processed_data)
                except Exception as e:
                    self.logger.error(f"LSTM model training failed: {str(e)}")
                    lstm_results = None
            else:
                self.logger.warning("TensorFlow not available, skipping LSTM model training")
            
            # Step 4: Train ARIMA model if available
            if STATSMODELS_AVAILABLE:
                try:
                    arima_results = self.train_arima_model(processed_data)
                except Exception as e:
                    self.logger.error(f"ARIMA model training failed: {str(e)}")
                    arima_results = None
            else:
                self.logger.warning("Statsmodels not available, skipping ARIMA model training")
            
            # Check if at least one model was trained
            if lstm_results is None and arima_results is None:
                self.logger.error("No models were successfully trained, cannot continue pipeline")
                raise RuntimeError("No models were successfully trained")
            
            # Step 5: Evaluate models
            evaluation_results = {}
            if lstm_results is not None and arima_results is not None:
                evaluation_results = self.evaluate_models(lstm_results, arima_results)
            elif lstm_results is not None:
                # Evaluate only LSTM
                lstm_metrics = self.model_evaluator.evaluate_model_performance(
                    model_name='LSTM',
                    y_true=lstm_results['y_true'],
                    y_pred=lstm_results['predictions'],
                    y_prob=lstm_results['probabilities'],
                    is_classifier=True
                )
                evaluation_results = {'LSTM': lstm_metrics['LSTM']}
            elif arima_results is not None:
                # Evaluate only ARIMA
                arima_metrics = self.model_evaluator.evaluate_model_performance(
                    model_name='ARIMA',
                    y_true=arima_results['y_true'],
                    y_pred=arima_results['predictions'],
                    is_classifier=True
                )
                evaluation_results = {'ARIMA': arima_metrics['ARIMA']}
            
            # Step 6: Simulate trading
            trading_results = {}
            if lstm_results is not None:
                lstm_trading = self.trading_simulator.simulate_strategy(
                    prices=test_prices.loc[lstm_results['test_dates']],
                    signals=lstm_results['predictions']
                )
                trading_results['LSTM'] = lstm_trading
                
            if arima_results is not None:
                arima_trading = self.trading_simulator.simulate_strategy(
                    prices=test_prices.loc[arima_results['test_dates']],
                    signals=arima_results['predictions']
                )
                trading_results['ARIMA'] = arima_trading
            
            # Step 7: Generate visualizations
            visualization_results = {}
            try:
                visualization_results = self.generate_visualizations(
                    lstm_results, 
                    arima_results, 
                    evaluation_results, 
                    trading_results,
                    test_prices,
                    test_volume
                )
            except Exception as e:
                self.logger.error(f"Visualization generation failed: {str(e)}")
            
            # Step 8: Analyze and report results
            analysis_results = {}
            try:
                # Prepare model results dictionary
                model_results = {}
                if lstm_results is not None:
                    model_results['LSTM'] = lstm_results
                if arima_results is not None:
                    model_results['ARIMA'] = arima_results
                
                # Only run analysis if we have at least one model
                if model_results:
                    analysis_results = self.analyze_and_report_results(
                        model_results, 
                        evaluation_results, 
                        trading_results, 
                        test_prices, 
                        test_volume
                    )
                    self.logger.info("Results analysis and reporting completed")
            except Exception as e:
                self.logger.error(f"Results analysis failed: {str(e)}")
            
            # Combine all results
            all_results = {
                'raw_data': raw_data,
                'processed_data': processed_data,
                'evaluation_results': evaluation_results,
                'trading_results': trading_results,
                'visualization_results': visualization_results,
                'analysis_results': analysis_results
            }
            
            if lstm_results is not None:
                all_results['lstm_results'] = lstm_results
                
            if arima_results is not None:
                all_results['arima_results'] = arima_results
            
            self.logger.info("Pipeline execution completed successfully")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
            raise
    
    def save_results(self, results: Dict[str, Any], filename: str = 'results/pipeline_results.pkl') -> None:
        """
        Save pipeline results to file
        
        Args:
            results: Dictionary with pipeline results
            filename: Path to save results
            
        Returns:
            None
        """
        import pickle
        
        self.logger.info(f"Saving pipeline results to {filename}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save results
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info("Results saved successfully")


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='NVDL Stock Predictor')
    parser.add_argument('--save-results', action='store_true', help='Save pipeline results to file')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose logging')
    parser.add_argument('--monitor-resources', action='store_true', help='Enable system resource monitoring')
    parser.add_argument('--error-report', action='store_true', help='Generate comprehensive error report')
    args = parser.parse_args()
    
    logger = get_main_logger()
    
    # Set up comprehensive error handling
    setup_error_handling()
    
    # Register additional cleanup handler for graceful exit
    def cleanup_handler():
        logger.info("Running cleanup operations before exit")
        # Close any open resources, save state, etc.
        
        # Generate error report on exit if requested
        if args.error_report:
            try:
                report = create_error_report()
                report_path = os.path.join('results', f'error_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                import json
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Error report saved to {report_path}")
            except Exception as e:
                logger.error(f"Failed to generate error report: {str(e)}")
    
    error_handler = ErrorHandler()
    error_handler.register_cleanup_handler(cleanup_handler, "Main cleanup handler")
    
    # Enable debug mode if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # Start resource monitoring if requested
    resource_monitor_thread = None
    if args.monitor_resources:
        logger.info("System resource monitoring enabled")
        
        def resource_monitor_task():
            while True:
                try:
                    monitor_system_resources(logger)
                    time.sleep(30)  # Monitor every 30 seconds
                except Exception as e:
                    logger.warning(f"Resource monitoring error: {str(e)}")
                    time.sleep(60)  # Retry after a minute
        
        resource_monitor_thread = threading.Thread(
            target=resource_monitor_task,
            daemon=True,
            name="ResourceMonitor"
        )
        resource_monitor_thread.start()
        
        # Register cleanup handler for the monitoring thread
        def monitor_cleanup():
            logger.info("Stopping resource monitoring")
            # No need to explicitly stop daemon thread, but log for clarity
        
        error_handler.register_cleanup_handler(monitor_cleanup, "Resource monitor cleanup")
    
    try:
        # Load configuration from file if provided
        if args.config:
            with ErrorContext(error_handler, "Configuration loading", file=args.config):
                try:
                    config.load_from_file(args.config)
                    logger.info(f"Loaded configuration from {args.config}")
                except Exception as e:
                    logger.error(f"Failed to load configuration from {args.config}: {str(e)}")
                    logger.info("Continuing with default configuration")
        
        # Validate configuration
        with ErrorContext(error_handler, "Configuration validation"):
            config.validate()
            logger.info("Configuration validated successfully")
        
        # Initialize and run pipeline
        pipeline = NVDLPredictorPipeline()
        
        # Log memory usage before pipeline execution
        log_memory_usage(logger, "Memory usage before pipeline execution")
        
        # Execute pipeline with error handling
        try:
            with LogContext(logger, "Pipeline execution"):
                # Create progress logger for overall pipeline
                progress = get_progress_logger("main", 100)
                progress.start("Pipeline execution")
                
                # Execute pipeline
                results = pipeline.run_pipeline()
                
                # Mark progress as complete
                progress.complete("Pipeline execution")
            
            # Log memory usage after pipeline execution
            log_memory_usage(logger, "Memory usage after pipeline execution")
            
            # Save results if requested
            if args.save_results:
                with LogContext(logger, "Saving results"):
                    pipeline.save_results(results)
            
            logger.info("Pipeline completed successfully")
            return results
            
        except Exception as e:
            with ErrorContext(error_handler, "Pipeline execution", component="main"):
                # Re-raise to let ErrorContext handle it
                raise
    
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        error_handler.graceful_exit(0, "Interrupted by user")
    except Exception as e:
        # This should only happen if ErrorContext fails
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.critical(f"Stack trace: {traceback.format_exc()}")
        error_handler.graceful_exit(1, f"Unhandled exception: {str(e)}")


if __name__ == '__main__':
    main()
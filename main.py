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
from typing import Dict, Any, Optional, Tuple

# Import configuration and logger
from config import config
from utils.logger import get_main_logger

# Import components
from data.collector import DataCollector
from data.preprocessor import DataPreprocessor
from models.model_evaluator import ModelEvaluator
from models.trading_simulator import TradingSimulator
from visualization.visualization_engine import VisualizationEngine

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
        
        # Create directories if they don't exist
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        self.logger.info("Pipeline initialized successfully")
    
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
        
        # Fetch data
        self.logger.info(f"Fetching data for {config.SYMBOL} from {start_date} to {end_date}")
        data = self.data_collector.fetch_historical_data(
            symbol=config.SYMBOL,
            start_date=start_date,
            end_date=end_date
        )
        
        # Validate data
        if not self.data_collector.validate_data_completeness(data):
            self.logger.error("Data validation failed")
            raise ValueError("Collected data failed validation checks")
        
        # Handle missing data
        data = self.data_collector.handle_missing_data(data)
        
        self.logger.info(f"Data collection completed: {len(data)} data points")
        return data
    
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
        
        # Complete preprocessing pipeline
        processed_data = self.data_preprocessor.prepare_data_for_training(
            data=data,
            test_size=config.TEST_SIZE
        )
        
        self.logger.info("Data preprocessing completed")
        return processed_data
    
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
        
        # Extract data
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        metadata = processed_data['metadata']
        
        # Initialize LSTM predictor
        self.lstm_predictor = LSTMPredictor(
            sequence_length=config.LSTM_SEQUENCE_LENGTH,
            lstm_units=config.LSTM_UNITS,
            dropout_rate=config.LSTM_DROPOUT
        )
        
        # Prepare sequences for LSTM
        X_train_seq, y_train_seq = self.lstm_predictor.prepare_sequences(
            np.column_stack((X_train, y_train))
        )
        
        X_test_seq, y_test_seq = self.lstm_predictor.prepare_sequences(
            np.column_stack((X_test, y_test))
        )
        
        # Build model
        input_shape = (config.LSTM_SEQUENCE_LENGTH, X_train.shape[1])
        self.lstm_predictor.build_model(input_shape)
        
        # Train model
        history = self.lstm_predictor.train(
            X_train=X_train_seq,
            y_train=y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=config.LSTM_EPOCHS,
            batch_size=config.LSTM_BATCH_SIZE,
            checkpoint_dir='./checkpoints'
        )
        
        # Generate predictions
        predictions = self.lstm_predictor.predict(X_test_seq)
        probabilities = self.lstm_predictor.predict_probabilities(X_test_seq)
        
        # Save model
        self.lstm_predictor.save_model('./checkpoints/lstm_model_final.h5')
        
        self.logger.info("LSTM model training completed")
        
        return {
            'model': self.lstm_predictor,
            'predictions': predictions,
            'probabilities': probabilities,
            'y_true': y_test_seq,
            'history': history.history,
            'test_dates': metadata['test_dates'][config.LSTM_SEQUENCE_LENGTH:],
            'train_dates': metadata['train_dates'][config.LSTM_SEQUENCE_LENGTH:]
        }
    
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
        
        # Extract data
        train_df = processed_data['train_df']
        test_df = processed_data['test_df']
        
        # Initialize ARIMA predictor
        self.arima_predictor = ARIMAPredictor(
            max_p=config.ARIMA_MAX_P,
            max_d=config.ARIMA_MAX_D,
            max_q=config.ARIMA_MAX_Q
        )
        
        # Fit model on training data
        self.arima_predictor.fit(train_df['close'])
        
        # Generate predictions for test data
        predictions = []
        for i in range(len(test_df) - 1):
            # Use current price to predict direction
            current_price = test_df['close'].iloc[i]
            direction = self.arima_predictor.predict_direction(current_price)
            predictions.append(direction)
        
        # Add one more prediction to match length
        predictions.append(predictions[-1] if predictions else 0)
        
        # Save model
        self.arima_predictor.save_model('./checkpoints/arima_model_final.pkl')
        
        self.logger.info("ARIMA model training completed")
        
        return {
            'model': self.arima_predictor,
            'predictions': np.array(predictions),
            'y_true': test_df['target'].values,
            'test_dates': test_df.index
        }
    
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
            
            # Combine all results
            all_results = {
                'raw_data': raw_data,
                'processed_data': processed_data,
                'evaluation_results': evaluation_results,
                'trading_results': trading_results,
                'visualization_results': visualization_results
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
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = NVDLPredictorPipeline()
    results = pipeline.run_pipeline()
    
    # Save results if requested
    if args.save_results:
        pipeline.save_results(results)
    
    return results


if __name__ == '__main__':
    main()
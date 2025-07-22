"""
Model evaluation and comparison system for NVDL Stock Predictor
Implements metrics calculation and model comparison functionality
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, mean_absolute_error
)
import plotly.graph_objects as go
import plotly.express as px
from utils.logger import get_evaluation_logger


class ModelEvaluator:
    """
    Model evaluation and comparison system
    Implements metrics calculation and performance analysis for both classification and forecasting models
    """
    
    def __init__(self):
        """Initialize ModelEvaluator with logger"""
        self.logger = get_evaluation_logger()
        self.logger.info("Initialized ModelEvaluator")
    
    def calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics for binary predictions
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            probabilities: Optional prediction probabilities for threshold-based metrics
            
        Returns:
            Dictionary of classification metrics
            
        Raises:
            ValueError: If inputs have different shapes or contain invalid values
        """
        # Validate inputs
        if len(y_true) != len(y_pred):
            self.logger.error(f"Shape mismatch: y_true {len(y_true)}, y_pred {len(y_pred)}")
            raise ValueError("y_true and y_pred must have the same length")
        
        if not np.all(np.isin(y_true, [0, 1])) or not np.all(np.isin(y_pred, [0, 1])):
            self.logger.error("Inputs must contain only binary values (0 or 1)")
            raise ValueError("Inputs must contain only binary values (0 or 1)")
        
        self.logger.info(f"Calculating classification metrics for {len(y_true)} samples")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Calculate confusion matrix values
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
            })
        
        # Add class distribution information
        metrics.update({
            'positive_rate_true': float(np.mean(y_true)),
            'positive_rate_pred': float(np.mean(y_pred))
        })
        
        self.logger.info(f"Classification metrics: {metrics}")
        return metrics
    
    def calculate_forecast_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics for price predictions
        
        Args:
            y_true: Ground truth price values
            y_pred: Forecasted price values
            
        Returns:
            Dictionary of forecast accuracy metrics
            
        Raises:
            ValueError: If inputs have different shapes
        """
        # Validate inputs
        if len(y_true) != len(y_pred):
            self.logger.error(f"Shape mismatch: y_true {len(y_true)}, y_pred {len(y_pred)}")
            raise ValueError("y_true and y_pred must have the same length")
        
        self.logger.info(f"Calculating forecast metrics for {len(y_true)} samples")
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan
        
        # Calculate directional accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred) if len(direction_true) > 0 else np.nan
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
        
        self.logger.info(f"Forecast metrics: {metrics}")
        return metrics
    
    def compare_models(
        self, 
        model_results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Generate side-by-side performance analysis of multiple models
        
        Args:
            model_results: Dictionary mapping model names to their metric dictionaries
            
        Returns:
            DataFrame with comparative metrics
            
        Raises:
            ValueError: If model_results is empty
        """
        if not model_results:
            self.logger.error("No model results provided for comparison")
            raise ValueError("model_results dictionary cannot be empty")
        
        self.logger.info(f"Comparing {len(model_results)} models: {list(model_results.keys())}")
        
        # Create DataFrame for comparison
        comparison_df = pd.DataFrame(model_results).T
        
        # Highlight best model for each metric
        for col in comparison_df.columns:
            if col in ['accuracy', 'precision', 'recall', 'f1_score', 'directional_accuracy']:
                # Higher is better
                best_value = comparison_df[col].max()
                best_model = comparison_df[col].idxmax()
            elif col in ['rmse', 'mae', 'mape']:
                # Lower is better
                best_value = comparison_df[col].min()
                best_model = comparison_df[col].idxmin()
            else:
                # Skip non-comparable metrics
                continue
            
            self.logger.info(f"Best model for {col}: {best_model} ({best_value:.4f})")
        
        return comparison_df
    
    def generate_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> np.ndarray:
        """
        Generate confusion matrix for classification results
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            model_name: Name of the model for logging
            
        Returns:
            Confusion matrix as numpy array
            
        Raises:
            ValueError: If inputs have different shapes
        """
        if len(y_true) != len(y_pred):
            self.logger.error(f"Shape mismatch: y_true {len(y_true)}, y_pred {len(y_pred)}")
            raise ValueError("y_true and y_pred must have the same length")
        
        self.logger.info(f"Generating confusion matrix for {model_name} with {len(y_true)} samples")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Log confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            self.logger.info(f"Confusion matrix for {model_name}:")
            self.logger.info(f"True Negatives: {tn}, False Positives: {fp}")
            self.logger.info(f"False Negatives: {fn}, True Positives: {tp}")
        
        return cm
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Plot confusion matrix as heatmap using Plotly
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            model_name: Name of the model for plot title
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
            
        Raises:
            ValueError: If inputs have different shapes
        """
        cm = self.generate_confusion_matrix(y_true, y_pred, model_name)
        
        # Create labels
        labels = ['Down', 'Up']
        
        # Create heatmap
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            title=f'Confusion Matrix - {model_name}',
            color_continuous_scale='Blues'
        )
        
        # Save if path provided
        if save_path:
            fig.write_image(save_path)
            self.logger.info(f"Confusion matrix plot saved to {save_path}")
        
        return fig
    
    def evaluate_model_performance(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        is_classifier: bool = True,
        true_prices: Optional[np.ndarray] = None,
        pred_prices: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive evaluation of model performance
        
        Args:
            model_name: Name of the model
            y_true: Ground truth labels for classification
            y_pred: Predicted labels for classification
            y_prob: Optional prediction probabilities
            is_classifier: Whether the model is a classifier
            true_prices: Optional ground truth prices for forecasting
            pred_prices: Optional predicted prices for forecasting
            
        Returns:
            Dictionary with all evaluation metrics
            
        Raises:
            ValueError: If required inputs are missing
        """
        self.logger.info(f"Evaluating performance for model: {model_name}")
        
        results = {model_name: {}}
        
        # Classification metrics
        if is_classifier:
            classification_metrics = self.calculate_classification_metrics(y_true, y_pred, y_prob)
            results[model_name].update(classification_metrics)
        
        # Forecasting metrics
        if true_prices is not None and pred_prices is not None:
            forecast_metrics = self.calculate_forecast_metrics(true_prices, pred_prices)
            results[model_name].update(forecast_metrics)
        
        return results
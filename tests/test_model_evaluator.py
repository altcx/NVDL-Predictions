"""
Unit tests for ModelEvaluator class
Tests metrics calculation and model comparison functionality
"""
import unittest
import numpy as np
import pandas as pd
from models.model_evaluator import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = ModelEvaluator()
        
        # Create test data for classification metrics
        self.y_true_binary = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        self.y_pred_binary = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1])
        
        # Create test data for forecast metrics
        self.y_true_prices = np.array([100.0, 101.5, 103.2, 102.7, 104.1, 103.8, 105.2])
        self.y_pred_prices = np.array([100.2, 102.0, 102.8, 103.5, 103.5, 104.5, 104.8])
        
        # Create test data for model comparison
        self.model_results = {
            'LSTM': {
                'accuracy': 0.75,
                'precision': 0.80,
                'recall': 0.70,
                'f1_score': 0.75,
                'rmse': 1.2
            },
            'ARIMA': {
                'accuracy': 0.70,
                'precision': 0.75,
                'recall': 0.65,
                'f1_score': 0.70,
                'rmse': 1.5
            }
        }
    
    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation"""
        metrics = self.evaluator.calculate_classification_metrics(
            self.y_true_binary, self.y_pred_binary
        )
        
        # Check that all required metrics are present
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('true_positives', metrics)
        self.assertIn('false_positives', metrics)
        self.assertIn('true_negatives', metrics)
        self.assertIn('false_negatives', metrics)
        
        # Check accuracy calculation
        expected_accuracy = 0.7  # 7 out of 10 correct
        self.assertAlmostEqual(metrics['accuracy'], expected_accuracy, places=4)
        
        # Check confusion matrix elements
        self.assertEqual(metrics['true_positives'], 4)  # Correctly predicted positives
        self.assertEqual(metrics['true_negatives'], 3)  # Correctly predicted negatives
        self.assertEqual(metrics['false_positives'], 2)  # Incorrectly predicted positives
        self.assertEqual(metrics['false_negatives'], 1)  # Incorrectly predicted negatives
    
    def test_calculate_forecast_metrics(self):
        """Test forecast metrics calculation"""
        metrics = self.evaluator.calculate_forecast_metrics(
            self.y_true_prices, self.y_pred_prices
        )
        
        # Check that all required metrics are present
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('directional_accuracy', metrics)
        
        # Check RMSE calculation
        expected_rmse = np.sqrt(np.mean((self.y_true_prices - self.y_pred_prices) ** 2))
        self.assertAlmostEqual(metrics['rmse'], expected_rmse, places=4)
        
        # Check MAE calculation
        expected_mae = np.mean(np.abs(self.y_true_prices - self.y_pred_prices))
        self.assertAlmostEqual(metrics['mae'], expected_mae, places=4)
    
    def test_compare_models(self):
        """Test model comparison functionality"""
        comparison_df = self.evaluator.compare_models(self.model_results)
        
        # Check that DataFrame has correct shape
        self.assertEqual(comparison_df.shape, (2, 5))
        
        # Check that all models are included
        self.assertIn('LSTM', comparison_df.index)
        self.assertIn('ARIMA', comparison_df.index)
        
        # Check that all metrics are included
        self.assertIn('accuracy', comparison_df.columns)
        self.assertIn('precision', comparison_df.columns)
        self.assertIn('recall', comparison_df.columns)
        self.assertIn('f1_score', comparison_df.columns)
        self.assertIn('rmse', comparison_df.columns)
        
        # Check that values are correctly transferred
        self.assertEqual(comparison_df.loc['LSTM', 'accuracy'], 0.75)
        self.assertEqual(comparison_df.loc['ARIMA', 'precision'], 0.75)
    
    def test_generate_confusion_matrix(self):
        """Test confusion matrix generation"""
        cm = self.evaluator.generate_confusion_matrix(
            self.y_true_binary, self.y_pred_binary
        )
        
        # Check shape of confusion matrix
        self.assertEqual(cm.shape, (2, 2))
        
        # Check values in confusion matrix
        # Format: [[TN, FP], [FN, TP]]
        self.assertEqual(cm[0, 0], 3)  # True Negatives
        self.assertEqual(cm[0, 1], 2)  # False Positives
        self.assertEqual(cm[1, 0], 1)  # False Negatives
        self.assertEqual(cm[1, 1], 4)  # True Positives
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting"""
        fig = self.evaluator.plot_confusion_matrix(
            self.y_true_binary, self.y_pred_binary, "Test Model"
        )
        
        # Check that a figure is returned
        self.assertIsNotNone(fig)
        self.assertEqual(fig.layout.title.text, "Confusion Matrix - Test Model")
    
    def test_input_validation(self):
        """Test input validation for metrics calculation"""
        # Test with mismatched array lengths
        with self.assertRaises(ValueError):
            self.evaluator.calculate_classification_metrics(
                np.array([0, 1, 0]), np.array([1, 0])
            )
        
        # Test with non-binary values
        with self.assertRaises(ValueError):
            self.evaluator.calculate_classification_metrics(
                np.array([0, 1, 2]), np.array([0, 1, 0])
            )
        
        # Test with empty model results
        with self.assertRaises(ValueError):
            self.evaluator.compare_models({})
    
    def test_evaluate_model_performance(self):
        """Test comprehensive model evaluation"""
        results = self.evaluator.evaluate_model_performance(
            model_name="TestModel",
            y_true=self.y_true_binary,
            y_pred=self.y_pred_binary,
            true_prices=self.y_true_prices,
            pred_prices=self.y_pred_prices
        )
        
        # Check that results contain the model name
        self.assertIn("TestModel", results)
        
        # Check that both classification and forecasting metrics are included
        metrics = results["TestModel"]
        self.assertIn('accuracy', metrics)
        self.assertIn('rmse', metrics)


if __name__ == '__main__':
    unittest.main()
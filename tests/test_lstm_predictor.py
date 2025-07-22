"""
Unit tests for LSTM predictor model
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock tensorflow before importing LSTMPredictor
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()
sys.modules['tensorflow.keras.layers'] = MagicMock()
sys.modules['tensorflow.keras.callbacks'] = MagicMock()
sys.modules['tensorflow.keras.optimizers'] = MagicMock()
sys.modules['tensorflow.keras.losses'] = MagicMock()
sys.modules['tensorflow.keras.metrics'] = MagicMock()

# Now import the module that uses tensorflow
from models.lstm_predictor import LSTMPredictor


class TestLSTMPredictor:
    """Test suite for LSTM predictor model"""
    
    @pytest.fixture
    def lstm_predictor(self):
        """Create LSTM predictor instance for testing"""
        return LSTMPredictor(sequence_length=10, lstm_units=32, dropout_rate=0.2)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        # Generate synthetic data: 100 samples, 5 features, 1 target
        np.random.seed(42)
        features = np.random.randn(100, 5)
        target = np.random.randint(0, 2, size=(100, 1))
        data = np.hstack((features, target))
        return data
    
    def test_init(self):
        """Test initialization with custom parameters"""
        predictor = LSTMPredictor(sequence_length=15, lstm_units=64, dropout_rate=0.3)
        assert predictor.sequence_length == 15
        assert predictor.lstm_units == 64
        assert predictor.dropout_rate == 0.3
        assert predictor.model is None
    
    @patch('models.lstm_predictor.Sequential')
    def test_build_model(self, mock_sequential, lstm_predictor):
        """Test model architecture building"""
        # Setup mock model
        mock_model = MagicMock()
        mock_sequential.return_value = mock_model
        
        input_shape = (10, 5)  # (sequence_length, features)
        model = lstm_predictor.build_model(input_shape)
        
        # Check that Sequential was called
        mock_sequential.assert_called_once()
        
        # Check that model was compiled
        mock_model.compile.assert_called_once()
        
        # Check that the model was returned
        assert model == mock_model
    
    def test_prepare_sequences(self, lstm_predictor, sample_data):
        """Test sequence preparation for LSTM input"""
        X_sequences, y_labels = lstm_predictor.prepare_sequences(sample_data)
        
        # Check shapes
        assert X_sequences.shape == (90, 10, 5)  # (samples, sequence_length, features)
        assert y_labels.shape == (90,)  # (samples,)
        
        # Check sequence content for first sample
        expected_first_sequence = sample_data[:10, :-1]
        expected_first_label = sample_data[10, -1]
        
        np.testing.assert_array_equal(X_sequences[0], expected_first_sequence)
        assert y_labels[0] == expected_first_label
    
    def test_prepare_sequences_insufficient_data(self, lstm_predictor):
        """Test sequence preparation with insufficient data"""
        # Create small dataset
        small_data = np.random.randn(5, 6)  # 5 samples, 5 features + 1 target
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            lstm_predictor.prepare_sequences(small_data)
    
    def test_predict_without_training(self, lstm_predictor):
        """Test prediction without training"""
        # Create test data
        X_test = np.random.randn(5, 10, 5)  # 5 samples, 10 timesteps, 5 features
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            lstm_predictor.predict(X_test)
    
    def test_train_and_predict(self, lstm_predictor, sample_data):
        """Test model training and prediction with mocks"""
        # Prepare sequences
        X_sequences, y_labels = lstm_predictor.prepare_sequences(sample_data)
        
        # Split into train and test
        split = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:split], X_sequences[split:]
        y_train, y_test = y_labels[:split], y_labels[split:]
        
        # Mock model
        mock_model = MagicMock()
        mock_history = MagicMock()
        mock_history.history = {
            'loss': [0.5, 0.4], 
            'binary_accuracy': [0.6, 0.7],
            'val_loss': [0.55, 0.45],
            'val_binary_accuracy': [0.58, 0.65]
        }
        mock_model.fit.return_value = mock_history
        mock_model.predict.return_value = np.array([[0.7], [0.3], [0.8]])
        
        # Set mock model
        lstm_predictor.model = mock_model
        
        # Train model
        history = lstm_predictor.train(
            X_train, y_train, 
            validation_data=(X_test, y_test),
            epochs=2,
            batch_size=8
        )
        
        # Check history object
        assert history == mock_history
        assert hasattr(history, 'history')
        assert 'loss' in history.history
        assert 'binary_accuracy' in history.history
        
        # Test prediction
        mock_model.predict.return_value = np.array([[0.7], [0.3], [0.8]])
        predictions = lstm_predictor.predict(X_test)
        assert len(predictions) == 3
        
        # Test probability prediction
        mock_model.predict.return_value = np.array([[0.7], [0.3], [0.8]])
        probabilities = lstm_predictor.predict_probabilities(X_test)
        assert len(probabilities) == 3
    
    @patch('models.lstm_predictor.os.path.exists')
    @patch('models.lstm_predictor.load_model')
    def test_save_load_model(self, mock_load_model, mock_exists, lstm_predictor, tmp_path):
        """Test saving and loading model"""
        # Create test data
        X_test = np.random.randn(5, 10, 5)
        
        # Save model should fail without training
        with pytest.raises(ValueError):
            lstm_predictor.save_model(os.path.join(tmp_path, "model.h5"))
        
        # Create a mock model
        mock_model = MagicMock()
        
        # Assign model directly for testing
        lstm_predictor.model = mock_model
        
        # Mock path exists
        mock_exists.return_value = True
        
        # Save model
        model_path = os.path.join(tmp_path, "model.h5")
        lstm_predictor.save_model(model_path)
        
        # Check save was called
        mock_model.save.assert_called_once_with(model_path)
        
        # Create new predictor
        new_predictor = LSTMPredictor(sequence_length=10, lstm_units=32, dropout_rate=0.2)
        
        # Setup mock for load_model
        mock_loaded_model = MagicMock()
        mock_load_model.return_value = mock_loaded_model
        
        # Load model
        new_predictor.load_model(model_path)
        
        # Check load was called
        mock_load_model.assert_called_once_with(model_path)
        
        # Check model loaded
        assert new_predictor.model == mock_loaded_model
        
        # Test load with non-existent file
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            new_predictor.load_model(os.path.join(tmp_path, "nonexistent.h5"))
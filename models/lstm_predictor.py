"""
LSTM neural network model for NVDL Stock Predictor
Implements a deep learning model for stock price direction prediction
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from typing import Tuple, Dict, Any, Optional, List, Union
from config import config
from utils.logger import get_model_logger


class LSTMPredictor:
    """
    LSTM neural network model for stock price direction prediction
    Implements sequence preparation, model building, training, and prediction
    """
    
    def __init__(
        self,
        sequence_length: int = None,
        lstm_units: int = None,
        dropout_rate: float = None
    ):
        """
        Initialize LSTM predictor with configurable parameters
        
        Args:
            sequence_length: Number of time steps in each input sequence
            lstm_units: Number of LSTM units in each layer
            dropout_rate: Dropout rate for regularization
        """
        self.logger = get_model_logger()
        
        # Use config values if not provided
        self.sequence_length = sequence_length or config.LSTM_SEQUENCE_LENGTH
        self.lstm_units = lstm_units or config.LSTM_UNITS
        self.dropout_rate = dropout_rate or config.LSTM_DROPOUT
        
        self.model = None
        self.history = None
        self.logger.info(f"Initialized LSTM predictor with sequence_length={self.sequence_length}, "
                         f"lstm_units={self.lstm_units}, dropout_rate={self.dropout_rate}")
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build LSTM model architecture with dropout regularization
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras Sequential model
        """
        self.logger.info(f"Building LSTM model with input shape {input_shape}")
        
        model = Sequential([
            # First LSTM layer with return sequences for stacking
            LSTM(
                units=self.lstm_units,
                return_sequences=True,
                input_shape=input_shape
            ),
            Dropout(self.dropout_rate),
            
            # Second LSTM layer
            LSTM(
                units=self.lstm_units // 2,
                return_sequences=False
            ),
            Dropout(self.dropout_rate),
            
            # Output layer with sigmoid activation for binary classification
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model with binary crossentropy loss and Adam optimizer
        model.compile(
            optimizer=Adam(),
            loss=BinaryCrossentropy(),
            metrics=[
                BinaryAccuracy(),
                Precision(),
                Recall(),
                AUC()
            ]
        )
        
        self.model = model
        self.logger.info(f"Model built successfully: {model.summary()}")
        return model
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input sequences for LSTM by reshaping data
        
        Args:
            data: Feature data as 2D array (samples, features)
            
        Returns:
            Tuple of (X_sequences, y_labels) where X_sequences has shape
            (samples, sequence_length, features)
            
        Raises:
            ValueError: If data has insufficient samples for sequence creation
        """
        if len(data) < self.sequence_length + 1:
            self.logger.error(f"Insufficient data: {len(data)} samples, need at least {self.sequence_length + 1}")
            raise ValueError(f"Insufficient data for sequence preparation: need at least {self.sequence_length + 1} samples")
        
        self.logger.info(f"Preparing sequences from {len(data)} samples with sequence_length={self.sequence_length}")
        
        # Number of samples and features
        n_samples = len(data)
        n_features = data.shape[1] - 1  # Exclude target column
        
        # Extract features and target
        features = data[:, :-1]
        target = data[:, -1]
        
        # Create sequences
        X_sequences = []
        y_labels = []
        
        for i in range(n_samples - self.sequence_length):
            # Extract sequence of features
            seq = features[i:i+self.sequence_length]
            # Target is the next value after the sequence
            label = target[i+self.sequence_length]
            
            X_sequences.append(seq)
            y_labels.append(label)
        
        # Convert to numpy arrays
        X_sequences = np.array(X_sequences)
        y_labels = np.array(y_labels)
        
        self.logger.info(f"Created {len(X_sequences)} sequences with shape {X_sequences.shape}")
        return X_sequences, y_labels
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = None,
        batch_size: int = None,
        checkpoint_dir: str = './checkpoints'
    ) -> tf.keras.callbacks.History:
        """
        Train LSTM model with early stopping and checkpoints
        
        Args:
            X_train: Training sequences with shape (samples, sequence_length, features)
            y_train: Training labels
            validation_data: Optional tuple of (X_val, y_val) for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Training history object
            
        Raises:
            ValueError: If model is not built or input data has incorrect shape
        """
        if self.model is None:
            self.logger.error("Model not built. Call build_model() first")
            raise ValueError("Model not built. Call build_model() first")
        
        # Use config values if not provided
        epochs = epochs or config.LSTM_EPOCHS
        batch_size = batch_size or config.LSTM_BATCH_SIZE
        
        # Validate input shape
        expected_shape = (None, self.sequence_length, X_train.shape[2])
        if X_train.shape[1:] != expected_shape[1:]:
            self.logger.error(f"Invalid input shape: {X_train.shape}, expected {expected_shape}")
            raise ValueError(f"Invalid input shape: {X_train.shape}, expected {expected_shape}")
        
        self.logger.info(f"Training LSTM model with {len(X_train)} samples for {epochs} epochs")
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Callbacks for training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            # Model checkpointing
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'lstm_model_{epoch:02d}.h5'),
                save_best_only=True,
                monitor='val_loss' if validation_data else 'loss'
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        self.logger.info("Model training completed")
        
        # Log training metrics
        final_epoch = len(history.history['loss']) - 1
        metrics = {
            'loss': history.history['loss'][final_epoch],
            'binary_accuracy': history.history['binary_accuracy'][final_epoch]
        }
        
        if validation_data:
            metrics.update({
                'val_loss': history.history['val_loss'][final_epoch],
                'val_binary_accuracy': history.history['val_binary_accuracy'][final_epoch]
            })
        
        self.logger.info(f"Final training metrics: {metrics}")
        return history
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate binary predictions for stock price direction
        
        Args:
            X_test: Test sequences with shape (samples, sequence_length, features)
            
        Returns:
            Binary predictions (0 or 1)
            
        Raises:
            ValueError: If model is not trained or input data has incorrect shape
        """
        if self.model is None:
            self.logger.error("Model not trained. Call train() first")
            raise ValueError("Model not trained. Call train() first")
        
        # Validate input shape
        expected_shape = (None, self.sequence_length, X_test.shape[2])
        if X_test.shape[1:] != expected_shape[1:]:
            self.logger.error(f"Invalid input shape: {X_test.shape}, expected {expected_shape}")
            raise ValueError(f"Invalid input shape: {X_test.shape}, expected {expected_shape}")
        
        self.logger.info(f"Generating predictions for {len(X_test)} samples")
        
        # Get raw probabilities
        probabilities = self.model.predict(X_test)
        
        # Convert to binary predictions
        predictions = (probabilities > 0.5).astype(int).flatten()
        
        self.logger.info(f"Generated {len(predictions)} predictions with distribution: {np.bincount(predictions)}")
        return predictions
    
    def predict_probabilities(self, X_test: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions for stock price direction
        
        Args:
            X_test: Test sequences with shape (samples, sequence_length, features)
            
        Returns:
            Probability predictions (0.0 to 1.0)
            
        Raises:
            ValueError: If model is not trained or input data has incorrect shape
        """
        if self.model is None:
            self.logger.error("Model not trained. Call train() first")
            raise ValueError("Model not trained. Call train() first")
        
        # Validate input shape
        expected_shape = (None, self.sequence_length, X_test.shape[2])
        if X_test.shape[1:] != expected_shape[1:]:
            self.logger.error(f"Invalid input shape: {X_test.shape}, expected {expected_shape}")
            raise ValueError(f"Invalid input shape: {X_test.shape}, expected {expected_shape}")
        
        self.logger.info(f"Generating probability predictions for {len(X_test)} samples")
        
        # Get raw probabilities
        probabilities = self.model.predict(X_test).flatten()
        
        return probabilities
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            self.logger.error("No model to save. Train model first")
            raise ValueError("No model to save. Train model first")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.logger.info(f"Saving model to {filepath}")
        self.model.save(filepath)
        self.logger.info("Model saved successfully")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from file
        
        Args:
            filepath: Path to load the model from
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Model file not found: {filepath}")
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.logger.info(f"Loading model from {filepath}")
        self.model = load_model(filepath)
        self.logger.info("Model loaded successfully")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test sequences
            y_test: True labels
            
        Returns:
            Dictionary of evaluation metrics
            
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            self.logger.error("Model not trained. Call train() first")
            raise ValueError("Model not trained. Call train() first")
        
        self.logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Evaluate model
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Create metrics dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
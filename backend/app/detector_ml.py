"""
Machine learning models for seismic event detection.
Includes CNN-based classifiers and autoencoders for anomaly detection.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from typing import Tuple, Optional, Dict, Any
import os


def build_classifier(input_shape: Tuple[int, ...] = (200, 1), 
                    num_classes: int = 2) -> keras.Model:
    """
    Build a CNN classifier for seismic event detection.
    
    Args:
        input_shape: Input shape (window_size, channels)
        num_classes: Number of output classes (2 for binary classification)
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv1D(32, 7, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv1D(64, 5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        # Third convolutional block
        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=0.001)
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def build_compact_classifier(input_shape: Tuple[int, ...] = (200, 1), 
                           num_classes: int = 2) -> keras.Model:
    """
    Build a compact CNN classifier for mobile/edge deployment.
    
    Args:
        input_shape: Input shape (window_size, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled compact Keras model
    """
    model = models.Sequential([
        # Compact convolutional blocks
        layers.Conv1D(16, 7, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        layers.Conv1D(32, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        
        # Compact dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=0.001)
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def build_autoencoder(input_shape: Tuple[int, ...] = (200, 1), 
                     latent_dim: int = 32) -> keras.Model:
    """
    Build an autoencoder for anomaly detection in seismic data.
    
    Args:
        input_shape: Input shape (window_size, channels)
        latent_dim: Dimension of latent space
        
    Returns:
        Compiled autoencoder model
    """
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    
    # Latent representation
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(np.prod(input_shape) // 8, activation='relu')(latent)
    x = layers.Reshape((input_shape[0] // 8, -1))(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 7, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    
    # Output layer
    decoder_output = layers.Conv1D(input_shape[-1], 3, activation='linear', padding='same')(x)
    
    # Create model
    autoencoder = models.Model(encoder_input, decoder_output)
    
    # Compile model
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return autoencoder


def build_lstm_classifier(input_shape: Tuple[int, ...] = (200, 1), 
                         num_classes: int = 2) -> keras.Model:
    """
    Build an LSTM-based classifier for temporal seismic patterns.
    
    Args:
        input_shape: Input shape (window_size, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled LSTM model
    """
    model = models.Sequential([
        # LSTM layers
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=0.001)
    loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model


def create_callbacks(model_path: str, patience: int = 10) -> list:
    """
    Create training callbacks for model training.
    
    Args:
        model_path: Path to save best model
        patience: Early stopping patience
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


def preprocess_for_ml(data: np.ndarray, window_size: int = 200, 
                     overlap: float = 0.5, normalize: bool = True) -> np.ndarray:
    """
    Preprocess seismic data for ML model input.
    
    Args:
        data: 1D seismic data array
        window_size: Size of sliding windows
        overlap: Overlap fraction between windows
        normalize: Whether to normalize windows
        
    Returns:
        Array of windowed data (n_windows, window_size, 1)
    """
    step_size = int(window_size * (1 - overlap))
    n_windows = (len(data) - window_size) // step_size + 1
    
    windows = np.zeros((n_windows, window_size, 1))
    
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window = data[start_idx:end_idx]
        
        if normalize:
            # Normalize each window
            window_mean = np.mean(window)
            window_std = np.std(window)
            if window_std > 0:
                window = (window - window_mean) / window_std
        
        windows[i, :, 0] = window
    
    return windows


def predict_events(model: keras.Model, data: np.ndarray, 
                  threshold: float = 0.5, window_size: int = 200) -> np.ndarray:
    """
    Predict seismic events using a trained model.
    
    Args:
        model: Trained Keras model
        data: 1D seismic data array
        threshold: Classification threshold
        window_size: Window size for prediction
        
    Returns:
        Array of event probabilities for each time step
    """
    # Preprocess data
    windows = preprocess_for_ml(data, window_size=window_size)
    
    # Make predictions
    predictions = model.predict(windows, verbose=0)
    
    # Handle different output formats
    if predictions.shape[-1] == 1:
        # Binary classification with sigmoid
        event_probs = predictions.flatten()
    else:
        # Multi-class or binary with softmax
        event_probs = predictions[:, -1]  # Assume last class is "event"
    
    # Interpolate predictions back to original time series length
    step_size = int(window_size * 0.5)  # Assuming 50% overlap
    full_probs = np.zeros(len(data))
    
    for i, prob in enumerate(event_probs):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, len(data))
        full_probs[start_idx:end_idx] = np.maximum(full_probs[start_idx:end_idx], prob)
    
    return full_probs


def detect_anomalies(autoencoder: keras.Model, data: np.ndarray, 
                    threshold_percentile: float = 95, window_size: int = 200) -> np.ndarray:
    """
    Detect anomalies using an autoencoder.
    
    Args:
        autoencoder: Trained autoencoder model
        data: 1D seismic data array
        threshold_percentile: Percentile for anomaly threshold
        window_size: Window size for analysis
        
    Returns:
        Array of anomaly scores for each time step
    """
    # Preprocess data
    windows = preprocess_for_ml(data, window_size=window_size)
    
    # Get reconstructions
    reconstructions = autoencoder.predict(windows, verbose=0)
    
    # Calculate reconstruction errors
    mse_errors = np.mean((windows - reconstructions) ** 2, axis=(1, 2))
    
    # Set threshold based on percentile
    threshold = np.percentile(mse_errors, threshold_percentile)
    
    # Interpolate errors back to original time series length
    step_size = int(window_size * 0.5)
    full_errors = np.zeros(len(data))
    
    for i, error in enumerate(mse_errors):
        start_idx = i * step_size
        end_idx = min(start_idx + window_size, len(data))
        full_errors[start_idx:end_idx] = np.maximum(full_errors[start_idx:end_idx], error)
    
    # Normalize errors to 0-1 range
    if np.max(full_errors) > 0:
        full_errors = full_errors / np.max(full_errors)
    
    return full_errors


def save_model_info(model: keras.Model, save_path: str, metadata: Dict[str, Any]):
    """
    Save model along with metadata.
    
    Args:
        model: Keras model to save
        save_path: Directory to save model
        metadata: Additional metadata to save
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model.save(os.path.join(save_path, 'model.h5'))
    
    # Save metadata
    import json
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save model summary
    with open(os.path.join(save_path, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def load_model_with_info(model_path: str) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Load model along with metadata.
    
    Args:
        model_path: Directory containing saved model
        
    Returns:
        Tuple of (model, metadata)
    """
    # Load model
    model = keras.models.load_model(os.path.join(model_path, 'model.h5'))
    
    # Load metadata
    import json
    metadata_path = os.path.join(model_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return model, metadata

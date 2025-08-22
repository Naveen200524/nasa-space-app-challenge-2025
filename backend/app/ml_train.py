"""
Training utilities for machine learning models.
Includes synthetic data generation and training pipelines.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
from .detector_ml import (
    build_classifier, build_compact_classifier, build_autoencoder,
    create_callbacks, save_model_info
)


def synthetic_windows(n_samples: int = 1000, ws: int = 200, 
                     quake_frac: float = 0.3, sr: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic seismic windows for training.
    
    Args:
        n_samples: Number of samples to generate
        ws: Window size in samples
        quake_frac: Fraction of samples that contain events
        sr: Sampling rate (Hz)
        
    Returns:
        Tuple of (X, y) where X is data and y is labels
    """
    X = np.zeros((n_samples, ws))
    y = np.zeros(n_samples, dtype=int)
    
    n_quakes = int(n_samples * quake_frac)
    
    for i in range(n_samples):
        if i < n_quakes:
            # Generate earthquake signal
            X[i] = generate_synthetic_earthquake(ws, sr)
            y[i] = 1
        else:
            # Generate noise
            X[i] = generate_synthetic_noise(ws, sr)
            y[i] = 0
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def generate_synthetic_earthquake(ws: int, sr: float = 20.0) -> np.ndarray:
    """
    Generate a synthetic earthquake signal.
    
    Args:
        ws: Window size in samples
        sr: Sampling rate (Hz)
        
    Returns:
        Synthetic earthquake waveform
    """
    t = np.linspace(0, ws / sr, ws)
    
    # Base noise
    signal = np.random.normal(0, 1e-9, ws)
    
    # P-wave arrival (high frequency, low amplitude)
    p_arrival = np.random.uniform(0.1, 0.3) * ws / sr
    p_mask = (t >= p_arrival) & (t <= p_arrival + 2.0)
    if np.any(p_mask):
        p_freq = np.random.uniform(8, 15)
        p_amp = np.random.uniform(2e-8, 5e-8)
        p_decay = np.exp(-(t[p_mask] - p_arrival) / 1.0)
        signal[p_mask] += p_amp * p_decay * np.sin(2 * np.pi * p_freq * (t[p_mask] - p_arrival))
    
    # S-wave arrival (lower frequency, higher amplitude)
    s_arrival = p_arrival + np.random.uniform(1.0, 3.0)
    s_mask = (t >= s_arrival) & (t <= s_arrival + 5.0)
    if np.any(s_mask):
        s_freq = np.random.uniform(2, 8)
        s_amp = np.random.uniform(5e-8, 2e-7)
        s_decay = np.exp(-(t[s_mask] - s_arrival) / 3.0)
        signal[s_mask] += s_amp * s_decay * np.sin(2 * np.pi * s_freq * (t[s_mask] - s_arrival))
    
    # Surface waves (lowest frequency, variable amplitude)
    if s_arrival + 2.0 < ws / sr:
        surf_arrival = s_arrival + np.random.uniform(1.0, 2.0)
        surf_mask = t >= surf_arrival
        if np.any(surf_mask):
            surf_freq = np.random.uniform(0.5, 3.0)
            surf_amp = np.random.uniform(1e-7, 5e-7)
            surf_decay = np.exp(-(t[surf_mask] - surf_arrival) / 8.0)
            signal[surf_mask] += surf_amp * surf_decay * np.sin(2 * np.pi * surf_freq * (t[surf_mask] - surf_arrival))
    
    # Add some random spikes for complexity
    n_spikes = np.random.poisson(2)
    for _ in range(n_spikes):
        spike_time = np.random.uniform(0, ws / sr)
        spike_idx = int(spike_time * sr)
        if 0 <= spike_idx < ws:
            spike_amp = np.random.uniform(1e-8, 3e-8)
            signal[spike_idx] += spike_amp * np.random.choice([-1, 1])
    
    return signal


def generate_synthetic_noise(ws: int, sr: float = 20.0) -> np.ndarray:
    """
    Generate synthetic background noise.
    
    Args:
        ws: Window size in samples
        sr: Sampling rate (Hz)
        
    Returns:
        Synthetic noise waveform
    """
    # Base white noise
    signal = np.random.normal(0, 1e-9, ws)
    
    # Add colored noise components
    t = np.linspace(0, ws / sr, ws)
    
    # Low frequency microseisms
    micro_freq = np.random.uniform(0.1, 0.5)
    micro_amp = np.random.uniform(5e-10, 2e-9)
    signal += micro_amp * np.sin(2 * np.pi * micro_freq * t + np.random.uniform(0, 2*np.pi))
    
    # Cultural noise (if applicable)
    if np.random.random() < 0.3:  # 30% chance of cultural noise
        cultural_freq = np.random.uniform(10, 50)
        cultural_amp = np.random.uniform(1e-9, 5e-9)
        cultural_phase = np.random.uniform(0, 2*np.pi)
        signal += cultural_amp * np.sin(2 * np.pi * cultural_freq * t + cultural_phase)
    
    # Random transients
    n_transients = np.random.poisson(1)
    for _ in range(n_transients):
        trans_start = np.random.uniform(0, ws / sr - 1.0)
        trans_duration = np.random.uniform(0.1, 1.0)
        trans_mask = (t >= trans_start) & (t <= trans_start + trans_duration)
        if np.any(trans_mask):
            trans_freq = np.random.uniform(1, 10)
            trans_amp = np.random.uniform(1e-9, 3e-9)
            trans_decay = np.exp(-(t[trans_mask] - trans_start) / (trans_duration / 3))
            signal[trans_mask] += trans_amp * trans_decay * np.sin(2 * np.pi * trans_freq * (t[trans_mask] - trans_start))
    
    return signal


def train_classifier(epochs: int = 10, ws: int = 200, batch_size: int = 32,
                    model_type: str = 'standard', save_dir: str = 'models/classifier') -> str:
    """
    Train a seismic event classifier.
    
    Args:
        epochs: Number of training epochs
        ws: Window size
        batch_size: Training batch size
        model_type: 'standard' or 'compact'
        save_dir: Directory to save trained model
        
    Returns:
        Path to saved model
    """
    print("Generating synthetic training data...")
    X, y = synthetic_windows(n_samples=5000, ws=ws, quake_frac=0.3)
    X = X.reshape((-1, ws, 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build model
    if model_type == 'compact':
        model = build_compact_classifier(input_shape=(ws, 1))
    else:
        model = build_classifier(input_shape=(ws, 1))
    
    print(f"Model parameters: {model.count_params()}")
    
    # Create callbacks
    os.makedirs(save_dir, exist_ok=True)
    callbacks = create_callbacks(os.path.join(save_dir, 'best_model.h5'))
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Generate predictions for detailed metrics
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Noise', 'Event']))
    
    # Save model and metadata
    metadata = {
        'model_type': 'classifier',
        'architecture': model_type,
        'window_size': ws,
        'training_samples': len(X_train),
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'epochs_trained': epochs,
        'parameters': model.count_params()
    }
    
    save_model_info(model, save_dir, metadata)
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    print(f"Model saved to: {save_dir}")
    return save_dir


def train_autoencoder(epochs: int = 10, ws: int = 200, batch_size: int = 32,
                     save_dir: str = 'models/autoencoder') -> str:
    """
    Train an autoencoder for anomaly detection.
    
    Args:
        epochs: Number of training epochs
        ws: Window size
        batch_size: Training batch size
        save_dir: Directory to save trained model
        
    Returns:
        Path to saved model
    """
    print("Generating synthetic training data for autoencoder...")
    # Use only noise samples for autoencoder training
    X_noise, _ = synthetic_windows(n_samples=4000, ws=ws, quake_frac=0.0)
    X_noise = X_noise.reshape((-1, ws, 1))
    
    # Split data
    X_train, X_test = train_test_split(X_noise, test_size=0.2, random_state=42)
    X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Build autoencoder
    model = build_autoencoder(input_shape=(ws, 1))
    print(f"Model parameters: {model.count_params()}")
    
    # Create callbacks
    os.makedirs(save_dir, exist_ok=True)
    callbacks = create_callbacks(os.path.join(save_dir, 'best_model.h5'))
    
    # Train model
    print("Training autoencoder...")
    history = model.fit(
        X_train, X_train,  # Autoencoder trains to reconstruct input
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("Evaluating autoencoder...")
    test_loss = model.evaluate(X_test, X_test, verbose=0)
    print(f"Test reconstruction loss: {test_loss:.6f}")
    
    # Save model and metadata
    metadata = {
        'model_type': 'autoencoder',
        'window_size': ws,
        'training_samples': len(X_train),
        'test_loss': float(test_loss),
        'epochs_trained': epochs,
        'parameters': model.count_params()
    }
    
    save_model_info(model, save_dir, metadata)
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    print(f"Autoencoder saved to: {save_dir}")
    return save_dir


def plot_training_history(history, save_dir: str):
    """
    Plot and save training history.
    
    Args:
        history: Keras training history
        save_dir: Directory to save plots
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(history.history['loss'], label='Training Loss')
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot accuracy (if available)
        if 'accuracy' in history.history:
            axes[1].plot(history.history['accuracy'], label='Training Accuracy')
            axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        else:
            # For autoencoders, plot MAE if available
            if 'mae' in history.history:
                axes[1].plot(history.history['mae'], label='Training MAE')
                axes[1].plot(history.history['val_mae'], label='Validation MAE')
                axes[1].set_title('Model MAE')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('MAE')
                axes[1].legend()
                axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Could not save training plots: {e}")


def evaluate_model_performance(model_path: str, test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
    """
    Evaluate a trained model's performance.
    
    Args:
        model_path: Path to saved model
        test_data: Optional test data (X, y)
        
    Returns:
        Performance metrics dictionary
    """
    from .detector_ml import load_model_with_info
    
    # Load model
    model, metadata = load_model_with_info(model_path)
    
    # Generate test data if not provided
    if test_data is None:
        ws = metadata.get('window_size', 200)
        X_test, y_test = synthetic_windows(n_samples=1000, ws=ws, quake_frac=0.3)
        X_test = X_test.reshape((-1, ws, 1))
    else:
        X_test, y_test = test_data
    
    # Make predictions
    if metadata.get('model_type') == 'autoencoder':
        # For autoencoders, calculate reconstruction error
        reconstructions = model.predict(X_test)
        mse_errors = np.mean((X_test - reconstructions) ** 2, axis=(1, 2))
        
        # Use threshold to classify anomalies
        threshold = np.percentile(mse_errors, 95)
        y_pred = (mse_errors > threshold).astype(int)
        
        metrics = {
            'model_type': 'autoencoder',
            'threshold': float(threshold),
            'mean_reconstruction_error': float(np.mean(mse_errors)),
            'std_reconstruction_error': float(np.std(mse_errors))
        }
    else:
        # For classifiers
        predictions = model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int).flatten()
        
        metrics = {
            'model_type': 'classifier',
            'accuracy': float(np.mean(y_pred == y_test)),
            'precision': float(np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred == 1), 1)),
            'recall': float(np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1))
        }
        
        # F1 score
        precision = metrics['precision']
        recall = metrics['recall']
        metrics['f1_score'] = float(2 * precision * recall / max(precision + recall, 1e-8))
    
    # Add metadata
    metrics.update(metadata)
    
    return metrics

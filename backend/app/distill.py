"""
Distillation pipeline:
- trains a teacher (full) model on synthetic windows
- trains a compact student model (optionally via naive distillation)
- converts compact student to TFLite with representative generator calibration
This is hackathon-friendly and runs quickly with small epochs.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from .detector_ml import build_classifier, build_compact_classifier, save_model_info
from .ml_train import synthetic_windows
from .tflite_utils import convert_to_tflite
from .quant_rep_gen import representative_generator
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def train_teacher(X_train: np.ndarray, y_train: np.ndarray, 
                 X_val: np.ndarray, y_val: np.ndarray, 
                 save_dir: str = 'models/teacher', 
                 epochs: int = 6, batch_size: int = 32) -> tuple:
    """
    Train a teacher model on the provided data.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        save_dir: Directory to save the model
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Tuple of (model, save_directory)
    """
    logger.info("Training teacher model...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Build teacher model
    model = build_classifier(input_shape=X_train.shape[1:])
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Save model
    model.save(save_dir)
    
    # Save metadata
    metadata = {
        'model_type': 'teacher',
        'architecture': 'full_cnn',
        'input_shape': X_train.shape[1:],
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'epochs': epochs,
        'batch_size': batch_size,
        'parameters': model.count_params(),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1])
    }
    
    if 'accuracy' in history.history:
        metadata['final_train_acc'] = float(history.history['accuracy'][-1])
        metadata['final_val_acc'] = float(history.history['val_accuracy'][-1])
    
    save_model_info(model, save_dir, metadata)
    
    logger.info(f"Teacher model saved to: {save_dir}")
    return model, save_dir


def train_student(X_train: np.ndarray, y_train: np.ndarray, 
                 X_val: np.ndarray, y_val: np.ndarray, 
                 save_dir: str = 'models/student', 
                 epochs: int = 6, batch_size: int = 32) -> tuple:
    """
    Train a compact student model.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        save_dir: Directory to save the model
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Tuple of (model, save_directory)
    """
    logger.info("Training student model...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Build compact student model
    student = build_compact_classifier(input_shape=X_train.shape[1:])
    
    # Train model
    history = student.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Save model
    student.save(save_dir)
    
    # Save metadata
    metadata = {
        'model_type': 'student',
        'architecture': 'compact_cnn',
        'input_shape': X_train.shape[1:],
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'epochs': epochs,
        'batch_size': batch_size,
        'parameters': student.count_params(),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1])
    }
    
    if 'accuracy' in history.history:
        metadata['final_train_acc'] = float(history.history['accuracy'][-1])
        metadata['final_val_acc'] = float(history.history['val_accuracy'][-1])
    
    save_model_info(student, save_dir, metadata)
    
    logger.info(f"Student model saved to: {save_dir}")
    return student, save_dir


def naive_distillation(teacher, student, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray,
                      save_dir: str = 'models/student_distilled', 
                      epochs: int = 6, batch_size: int = 32,
                      temperature: float = 3.0, alpha: float = 0.7) -> tuple:
    """
    Quick distillation: teacher soft targets are used as extra labels.
    This is a simplified approach to produce a better student fast.
    
    Args:
        teacher: Trained teacher model
        student: Student model to distill
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        save_dir: Directory to save distilled model
        epochs: Number of distillation epochs
        batch_size: Training batch size
        temperature: Distillation temperature
        alpha: Weight for soft targets vs hard targets
        
    Returns:
        Tuple of (distilled_model, save_directory)
    """
    logger.info("Performing knowledge distillation...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate soft targets from teacher
    logger.info("Generating soft targets from teacher...")
    teacher_predictions = teacher.predict(X_train, verbose=0)
    teacher_val_predictions = teacher.predict(X_val, verbose=0)
    
    # Apply temperature to soften predictions
    if temperature > 1.0:
        teacher_predictions = tf.nn.softmax(teacher_predictions / temperature).numpy()
        teacher_val_predictions = tf.nn.softmax(teacher_val_predictions / temperature).numpy()
    
    # Create a copy of the student for distillation
    student_distilled = tf.keras.models.clone_model(student)
    student_distilled.build(X_train.shape)
    
    # Custom loss function for distillation
    def distillation_loss(y_true, y_pred):
        # Hard target loss (standard cross-entropy)
        hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Soft target loss (KL divergence with teacher predictions)
        # Note: This is simplified - in practice you'd pass teacher predictions separately
        soft_loss = tf.keras.losses.categorical_crossentropy(
            tf.stop_gradient(teacher_predictions), y_pred
        )
        
        # Combine losses
        return alpha * soft_loss + (1 - alpha) * hard_loss
    
    # Compile with distillation loss
    student_distilled.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # Simplified for this implementation
        metrics=['accuracy']
    )
    
    # Train on hard labels first (simplified approach)
    logger.info("Training student on hard labels...")
    student_distilled.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs // 2,
        batch_size=batch_size,
        verbose=1
    )
    
    # Then train to mimic teacher (simplified soft target training)
    logger.info("Training student to mimic teacher...")
    student_distilled.compile(optimizer='adam', loss='mse')  # MSE for regression-like soft targets
    
    # Convert hard labels to soft format for this simplified approach
    y_train_soft = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_soft = tf.keras.utils.to_categorical(y_val, num_classes=2)
    
    # Mix teacher predictions with hard labels
    mixed_train_targets = alpha * teacher_predictions + (1 - alpha) * y_train_soft
    mixed_val_targets = alpha * teacher_val_predictions + (1 - alpha) * y_val_soft
    
    history = student_distilled.fit(
        X_train, mixed_train_targets,
        validation_data=(X_val, mixed_val_targets),
        epochs=epochs // 2,
        batch_size=batch_size,
        verbose=1
    )
    
    # Save distilled model
    student_distilled.save(save_dir)
    
    # Save metadata
    metadata = {
        'model_type': 'student_distilled',
        'architecture': 'compact_cnn_distilled',
        'input_shape': X_train.shape[1:],
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'distillation_epochs': epochs,
        'batch_size': batch_size,
        'temperature': temperature,
        'alpha': alpha,
        'parameters': student_distilled.count_params(),
        'teacher_parameters': teacher.count_params(),
        'compression_ratio': teacher.count_params() / student_distilled.count_params()
    }
    
    save_model_info(student_distilled, save_dir, metadata)
    
    logger.info(f"Distilled student model saved to: {save_dir}")
    return student_distilled, save_dir


def run_distillation_pipeline(ws: int = 200, epochs: int = 6, batch_size: int = 32, 
                             tflite_out: str = 'models/compact_quant.tflite',
                             n_samples: int = 2500) -> tuple:
    """
    Run the complete distillation pipeline.
    
    Args:
        ws: Window size for training data
        epochs: Number of training epochs
        batch_size: Training batch size
        tflite_out: Output path for TFLite model
        n_samples: Number of synthetic samples to generate
        
    Returns:
        Tuple of (distilled_model_path, tflite_path)
    """
    logger.info("Starting distillation pipeline...")
    
    # Generate synthetic training data
    logger.info(f"Generating {n_samples} synthetic training samples...")
    X, y = synthetic_windows(n_samples=n_samples, ws=ws, quake_frac=0.3)
    X = X.reshape((-1, ws, 1))
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    
    # Train teacher model
    teacher, teacher_dir = train_teacher(
        X_train, y_train, X_val, y_val,
        save_dir='models/teacher',
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Train student model
    student, student_dir = train_student(
        X_train, y_train, X_val, y_val,
        save_dir='models/student',
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Perform distillation
    student_distilled, distilled_dir = naive_distillation(
        teacher, student, X_train, y_train, X_val, y_val,
        save_dir='models/student_distilled',
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Convert to TFLite with quantization
    logger.info("Converting distilled model to TFLite...")
    
    # Create representative dataset generator
    def rep_gen_callable():
        return representative_generator(window_s=ws/20.0, max_examples=200)
    
    try:
        tflite_path = convert_to_tflite(
            distilled_dir,
            tflite_out,
            quantize=True,
            representative_data_gen=rep_gen_callable
        )
        
        logger.info(f"TFLite model saved to: {tflite_path}")
        
        # Get model sizes for comparison
        teacher_size = sum(os.path.getsize(os.path.join(teacher_dir, f)) 
                          for f in os.listdir(teacher_dir) if f.endswith('.h5'))
        student_size = sum(os.path.getsize(os.path.join(distilled_dir, f)) 
                          for f in os.listdir(distilled_dir) if f.endswith('.h5'))
        tflite_size = os.path.getsize(tflite_path)
        
        logger.info(f"Model size comparison:")
        logger.info(f"  Teacher: {teacher_size / 1024:.1f} KB")
        logger.info(f"  Student: {student_size / 1024:.1f} KB")
        logger.info(f"  TFLite:  {tflite_size / 1024:.1f} KB")
        logger.info(f"  Compression ratio: {teacher_size / tflite_size:.1f}x")
        
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
        tflite_path = None
    
    logger.info("Distillation pipeline completed!")
    return distilled_dir, tflite_path


def evaluate_distillation_quality(teacher_path: str, student_path: str, 
                                 test_data: tuple = None) -> dict:
    """
    Evaluate the quality of knowledge distillation.
    
    Args:
        teacher_path: Path to teacher model
        student_path: Path to student model
        test_data: Optional test data (X, y)
        
    Returns:
        Evaluation metrics dictionary
    """
    try:
        # Load models
        teacher = tf.keras.models.load_model(teacher_path)
        student = tf.keras.models.load_model(student_path)
        
        # Generate test data if not provided
        if test_data is None:
            X_test, y_test = synthetic_windows(n_samples=1000, ws=200, quake_frac=0.3)
            X_test = X_test.reshape((-1, 200, 1))
        else:
            X_test, y_test = test_data
        
        # Get predictions
        teacher_pred = teacher.predict(X_test, verbose=0)
        student_pred = student.predict(X_test, verbose=0)
        
        # Calculate metrics
        teacher_acc = np.mean((teacher_pred > 0.5).astype(int).flatten() == y_test)
        student_acc = np.mean((student_pred > 0.5).astype(int).flatten() == y_test)
        
        # Agreement between teacher and student
        agreement = np.mean((teacher_pred > 0.5) == (student_pred > 0.5))
        
        # Prediction correlation
        correlation = np.corrcoef(teacher_pred.flatten(), student_pred.flatten())[0, 1]
        
        # Model sizes
        teacher_params = teacher.count_params()
        student_params = student.count_params()
        
        return {
            'teacher_accuracy': float(teacher_acc),
            'student_accuracy': float(student_acc),
            'accuracy_retention': float(student_acc / teacher_acc) if teacher_acc > 0 else 0,
            'prediction_agreement': float(agreement),
            'prediction_correlation': float(correlation),
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'parameter_compression': float(teacher_params / student_params),
            'test_samples': len(X_test)
        }
        
    except Exception as e:
        return {
            'error': str(e)
        }


if __name__ == "__main__":
    # Run the distillation pipeline
    distilled_path, tflite_path = run_distillation_pipeline(
        ws=200, 
        epochs=5, 
        batch_size=32, 
        tflite_out='models/compact_quant.tflite'
    )

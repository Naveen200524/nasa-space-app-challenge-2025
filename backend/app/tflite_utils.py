"""
TFLite conversion and demo utilities.
- convert_to_tflite(keras_saved_model_dir, tflite_path, quantize=True, representative_data_gen=None)
- demo_tflite_inference(tflite_path, sample_input)
"""

import os
import numpy as np
import tensorflow as tf
from typing import Optional, Callable, Generator, Any
import logging

logger = logging.getLogger(__name__)


def convert_to_tflite(keras_model_path: str, tflite_path: str, 
                     quantize: bool = True, 
                     representative_data_gen: Optional[Callable[[], Generator[np.ndarray, None, None]]] = None) -> str:
    """
    Convert saved Keras SavedModel directory to TFLite file.
    
    Args:
        keras_model_path: Path to Keras SavedModel directory
        tflite_path: Output path for TFLite file
        quantize: Apply quantization optimizations
        representative_data_gen: Generator function for representative dataset
        
    Returns:
        Path to created TFLite file
        
    Raises:
        Exception: If conversion fails
    """
    try:
        logger.info(f"Converting Keras model to TFLite: {keras_model_path} -> {tflite_path}")
        
        # Load the SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(keras_model_path)
        
        if quantize:
            logger.info("Applying quantization optimizations")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_data_gen is not None:
                logger.info("Using representative dataset for quantization")
                converter.representative_dataset = representative_data_gen
                
                # Aim for int8 operations
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                
                # Keep input/output types as uint8 for simpler embedded pipelines
                converter.inference_input_type = tf.uint8
                converter.inference_output_type = tf.uint8
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(tflite_path) or '.', exist_ok=True)
        
        # Save the TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Get model size
        model_size = len(tflite_model)
        logger.info(f"TFLite model saved: {tflite_path} ({model_size / 1024:.1f} KB)")
        
        return tflite_path
        
    except Exception as e:
        logger.error(f"TFLite conversion failed: {e}")
        raise Exception(f"Failed to convert to TFLite: {str(e)}")


def demo_tflite_inference(tflite_path: str, sample_input: np.ndarray) -> np.ndarray:
    """
    Run a single inference using TFLite interpreter.
    
    Args:
        tflite_path: Path to TFLite model file
        sample_input: Input array shaped (window,) or (1, window, 1)
        
    Returns:
        Raw output array from interpreter
        
    Raises:
        Exception: If inference fails
    """
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.debug(f"Input details: {input_details}")
        logger.debug(f"Output details: {output_details}")
        
        # Prepare input data
        x = np.array(sample_input, dtype=np.float32)
        
        # Reshape to expected format (1, window, 1)
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0], 1))
        elif x.ndim == 2 and x.shape[0] != 1:
            x = x.reshape((1, x.shape[0], x.shape[1]))
        
        expected_dtype = input_details[0]['dtype']
        expected_shape = input_details[0]['shape']
        
        logger.debug(f"Expected input shape: {expected_shape}, dtype: {expected_dtype}")
        logger.debug(f"Actual input shape: {x.shape}, dtype: {x.dtype}")
        
        # Handle quantized models (uint8 input)
        if expected_dtype == np.uint8:
            # Simple linear scaling for demonstration
            xmin, xmax = x.min(), x.max()
            if np.isclose(xmax, xmin):
                x_q = np.zeros_like(x, dtype=np.uint8)
            else:
                x_q = ((x - xmin) / (xmax - xmin) * 255.0).clip(0, 255).astype(np.uint8)
            inp = x_q
        else:
            inp = x.astype(expected_dtype)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], inp)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logger.debug(f"Output shape: {output.shape}, dtype: {output.dtype}")
        
        return output
        
    except Exception as e:
        logger.error(f"TFLite inference failed: {e}")
        raise Exception(f"Failed to run TFLite inference: {str(e)}")


def get_tflite_model_info(tflite_path: str) -> dict:
    """
    Get information about a TFLite model.
    
    Args:
        tflite_path: Path to TFLite model file
        
    Returns:
        Dictionary with model information
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Get model size
        model_size = os.path.getsize(tflite_path)
        
        info = {
            'file_path': tflite_path,
            'file_size_bytes': model_size,
            'file_size_kb': model_size / 1024,
            'input_details': input_details,
            'output_details': output_details,
            'num_inputs': len(input_details),
            'num_outputs': len(output_details)
        }
        
        # Add input/output shape and type info
        if input_details:
            info['input_shape'] = input_details[0]['shape'].tolist()
            info['input_dtype'] = str(input_details[0]['dtype'])
        
        if output_details:
            info['output_shape'] = output_details[0]['shape'].tolist()
            info['output_dtype'] = str(output_details[0]['dtype'])
        
        return info
        
    except Exception as e:
        return {
            'file_path': tflite_path,
            'error': str(e)
        }


def benchmark_tflite_model(tflite_path: str, test_data: np.ndarray, 
                          num_runs: int = 100) -> dict:
    """
    Benchmark TFLite model performance.
    
    Args:
        tflite_path: Path to TFLite model file
        test_data: Test input data
        num_runs: Number of inference runs for timing
        
    Returns:
        Benchmark results dictionary
    """
    import time
    
    try:
        # Load model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        x = np.array(test_data, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0], 1))
        
        expected_dtype = input_details[0]['dtype']
        if expected_dtype == np.uint8:
            xmin, xmax = x.min(), x.max()
            if not np.isclose(xmax, xmin):
                x = ((x - xmin) / (xmax - xmin) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = np.zeros_like(x, dtype=np.uint8)
        else:
            x = x.astype(expected_dtype)
        
        # Warm-up run
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        
        return {
            'num_runs': num_runs,
            'mean_time_ms': float(np.mean(times) * 1000),
            'std_time_ms': float(np.std(times) * 1000),
            'min_time_ms': float(np.min(times) * 1000),
            'max_time_ms': float(np.max(times) * 1000),
            'median_time_ms': float(np.median(times) * 1000),
            'throughput_hz': float(1.0 / np.mean(times)),
            'model_size_kb': os.path.getsize(tflite_path) / 1024
        }
        
    except Exception as e:
        return {
            'error': str(e)
        }


def compare_keras_vs_tflite(keras_model_path: str, tflite_path: str, 
                           test_data: np.ndarray, tolerance: float = 1e-5) -> dict:
    """
    Compare outputs between Keras and TFLite models.
    
    Args:
        keras_model_path: Path to Keras SavedModel
        tflite_path: Path to TFLite model
        test_data: Test input data
        tolerance: Tolerance for output comparison
        
    Returns:
        Comparison results dictionary
    """
    try:
        # Load Keras model
        keras_model = tf.keras.models.load_model(keras_model_path)
        
        # Prepare input
        x = np.array(test_data, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0], 1))
        
        # Get Keras prediction
        keras_output = keras_model.predict(x, verbose=0)
        
        # Get TFLite prediction
        tflite_output = demo_tflite_inference(tflite_path, test_data)
        
        # Compare outputs
        if keras_output.shape != tflite_output.shape:
            return {
                'error': f"Output shape mismatch: Keras {keras_output.shape} vs TFLite {tflite_output.shape}"
            }
        
        # Calculate differences
        abs_diff = np.abs(keras_output - tflite_output)
        rel_diff = abs_diff / (np.abs(keras_output) + 1e-8)
        
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)
        mean_abs_diff = np.mean(abs_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        outputs_close = np.allclose(keras_output, tflite_output, atol=tolerance)
        
        return {
            'outputs_match': outputs_close,
            'max_absolute_difference': float(max_abs_diff),
            'max_relative_difference': float(max_rel_diff),
            'mean_absolute_difference': float(mean_abs_diff),
            'mean_relative_difference': float(mean_rel_diff),
            'tolerance_used': tolerance,
            'keras_output_shape': keras_output.shape,
            'tflite_output_shape': tflite_output.shape,
            'keras_output_range': [float(np.min(keras_output)), float(np.max(keras_output))],
            'tflite_output_range': [float(np.min(tflite_output)), float(np.max(tflite_output))]
        }
        
    except Exception as e:
        return {
            'error': str(e)
        }

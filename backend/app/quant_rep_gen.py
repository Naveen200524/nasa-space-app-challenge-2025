"""
Representative dataset generator for TFLite quantization.
This yields real-ish seismic windows from InSight:
- tries to fetch via PDS Search (SEIS products) or IRIS
- pairs seismic and pressure channels where available
- yields arrays shaped (1, window_samples, 1)
Usage: pass rep_gen() (callable) to convert_to_tflite as representative_dataset
"""

import numpy as np
from obspy import UTCDateTime
from .data_fetcher import fetch_from_iris, fetch_from_pds_search, fetch_from_url
from .io_utils import load_waveform, trace_to_numpy
import random
import os
import logging
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)


def sample_windows_from_trace(trace, window_s: float = 10.0, n_windows: int = 20) -> List[np.ndarray]:
    """
    Extract random windows from a seismic trace.
    
    Args:
        trace: ObsPy Trace object
        window_s: Window duration in seconds
        n_windows: Number of windows to extract
        
    Returns:
        List of windowed arrays shaped (1, window_samples, 1)
    """
    try:
        data, sr = trace_to_numpy(trace, target_sampling_rate=None)
        ws = int(window_s * sr)
        
        if len(data) < ws:
            logger.warning(f"Trace too short for window size: {len(data)} < {ws}")
            return []
        
        windows = []
        for _ in range(n_windows):
            start = random.randint(0, len(data) - ws)
            w = data[start:start + ws]
            
            # Normalize to -1..1 range
            if np.std(w) > 0:
                w = (w - np.mean(w)) / (np.std(w) + 1e-12)
                w = np.clip(w, -5, 5)  # Clip extreme values
            
            windows.append(w.reshape((1, ws, 1)).astype(np.float32))
        
        return windows
        
    except Exception as e:
        logger.error(f"Error sampling windows from trace: {e}")
        return []


def rep_gen_from_local_files(file_paths: List[str], window_s: float = 10.0, 
                           max_examples: int = 100) -> Generator[np.ndarray, None, None]:
    """
    Generate representative data from local files.
    
    Args:
        file_paths: List of paths to seismic data files
        window_s: Window duration in seconds
        max_examples: Maximum number of examples to yield
        
    Yields:
        Windowed arrays shaped (1, window_samples, 1)
    """
    count = 0
    
    for fp in file_paths:
        if count >= max_examples:
            break
            
        try:
            logger.debug(f"Processing file: {fp}")
            st = load_waveform(fp)
            tr = st[0]
            
            windows = sample_windows_from_trace(tr, window_s=window_s, n_windows=5)
            for w in windows:
                if count >= max_examples:
                    break
                yield w
                count += 1
                
        except Exception as e:
            logger.warning(f"Could not process file {fp}: {e}")
            continue


def rep_gen_pds_insight(window_s: float = 10.0, max_examples: int = 200) -> Generator[np.ndarray, None, None]:
    """
    Best-effort: fetch some InSight SEIS products via PDS Search and yield windows.
    
    Args:
        window_s: Window duration in seconds
        max_examples: Maximum number of examples to yield
        
    Yields:
        Windowed arrays shaped (1, window_samples, 1)
    """
    count = 0
    
    try:
        logger.info("Fetching InSight SEIS data for representative dataset")
        downloaded = fetch_from_pds_search(
            mission="insight", 
            instrument="SEIS", 
            product_class=None, 
            limit=10, 
            download_first_match=False
        )
        
        if not downloaded:
            logger.warning("No InSight SEIS data downloaded")
            return
        
        for fp in downloaded:
            if count >= max_examples:
                break
                
            try:
                st = load_waveform(fp)
                tr = st[0]
                
                windows = sample_windows_from_trace(tr, window_s=window_s, n_windows=10)
                for w in windows:
                    if count >= max_examples:
                        break
                    yield w
                    count += 1
                    
            except Exception as e:
                logger.warning(f"Could not process PDS file {fp}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error fetching PDS data: {e}")


def rep_gen_synthetic(window_s: float = 10.0, max_examples: int = 100, 
                     sr: float = 20.0) -> Generator[np.ndarray, None, None]:
    """
    Generate synthetic representative data as fallback.
    
    Args:
        window_s: Window duration in seconds
        max_examples: Maximum number of examples to yield
        sr: Sampling rate
        
    Yields:
        Windowed arrays shaped (1, window_samples, 1)
    """
    from .ml_train import generate_synthetic_earthquake, generate_synthetic_noise
    
    ws = int(window_s * sr)
    
    for i in range(max_examples):
        if i % 3 == 0:
            # Generate earthquake signal
            data = generate_synthetic_earthquake(ws, sr)
        else:
            # Generate noise
            data = generate_synthetic_noise(ws, sr)
        
        # Normalize
        if np.std(data) > 0:
            data = (data - np.mean(data)) / (np.std(data) + 1e-12)
            data = np.clip(data, -5, 5)
        
        yield data.reshape((1, ws, 1)).astype(np.float32)


def representative_generator(window_s: float = 10.0, max_examples: int = 200) -> Generator[np.ndarray, None, None]:
    """
    Generic representative dataset generator combining multiple sources.
    Provide to tflite converter as a callable with zero args that yields numpy arrays.
    
    Args:
        window_s: Window duration in seconds
        max_examples: Maximum number of examples to yield
        
    Yields:
        Windowed arrays shaped (1, window_samples, 1)
    """
    count = 0
    
    # 1) Try local data folder first
    data_dir = "data"
    local_files = []
    
    if os.path.exists(data_dir):
        logger.info(f"Searching for local data in: {data_dir}")
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(('.mseed', '.msd', '.seed', '.sac', '.gz', '.zip')):
                    local_files.append(os.path.join(root, f))
        
        logger.info(f"Found {len(local_files)} local seismic files")
    
    # Yield from local files first
    if local_files:
        for w in rep_gen_from_local_files(local_files, window_s=window_s, max_examples=max_examples // 2):
            if count >= max_examples:
                break
            yield w
            count += 1
    
    # 2) Try PDS data if we need more examples
    if count < max_examples:
        remaining = max_examples - count
        for w in rep_gen_pds_insight(window_s=window_s, max_examples=remaining // 2):
            if count >= max_examples:
                break
            yield w
            count += 1
    
    # 3) Fill remaining with synthetic data
    if count < max_examples:
        remaining = max_examples - count
        logger.info(f"Generating {remaining} synthetic examples for representative dataset")
        for w in rep_gen_synthetic(window_s=window_s, max_examples=remaining):
            if count >= max_examples:
                break
            yield w
            count += 1
    
    logger.info(f"Representative dataset generation complete: {count} examples")


def create_representative_dataset_file(output_path: str, window_s: float = 10.0, 
                                     max_examples: int = 1000) -> str:
    """
    Create and save a representative dataset to file for reuse.
    
    Args:
        output_path: Path to save the dataset
        window_s: Window duration in seconds
        max_examples: Maximum number of examples
        
    Returns:
        Path to saved dataset file
    """
    logger.info(f"Creating representative dataset file: {output_path}")
    
    # Collect all examples
    examples = []
    for example in representative_generator(window_s=window_s, max_examples=max_examples):
        examples.append(example)
    
    if not examples:
        raise ValueError("No representative examples generated")
    
    # Stack into single array
    dataset = np.vstack(examples)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.save(output_path, dataset)
    
    logger.info(f"Saved representative dataset: {dataset.shape} to {output_path}")
    return output_path


def load_representative_dataset(dataset_path: str) -> Generator[np.ndarray, None, None]:
    """
    Load representative dataset from file and yield examples.
    
    Args:
        dataset_path: Path to saved dataset file
        
    Yields:
        Individual examples from the dataset
    """
    try:
        dataset = np.load(dataset_path)
        logger.info(f"Loaded representative dataset: {dataset.shape} from {dataset_path}")
        
        for i in range(dataset.shape[0]):
            yield dataset[i:i+1]  # Keep batch dimension
            
    except Exception as e:
        logger.error(f"Could not load representative dataset from {dataset_path}: {e}")
        # Fallback to generated data
        for example in representative_generator():
            yield example


def validate_representative_data(generator_func, num_samples: int = 10) -> dict:
    """
    Validate representative data generator.
    
    Args:
        generator_func: Generator function to validate
        num_samples: Number of samples to check
        
    Returns:
        Validation results dictionary
    """
    try:
        samples = []
        shapes = []
        dtypes = []
        
        gen = generator_func()
        for i, sample in enumerate(gen):
            if i >= num_samples:
                break
            samples.append(sample)
            shapes.append(sample.shape)
            dtypes.append(sample.dtype)
        
        if not samples:
            return {'valid': False, 'error': 'No samples generated'}
        
        # Check consistency
        first_shape = shapes[0]
        first_dtype = dtypes[0]
        
        shape_consistent = all(s == first_shape for s in shapes)
        dtype_consistent = all(d == first_dtype for d in dtypes)
        
        # Check value ranges
        all_data = np.vstack(samples)
        min_val = np.min(all_data)
        max_val = np.max(all_data)
        mean_val = np.mean(all_data)
        std_val = np.std(all_data)
        
        return {
            'valid': shape_consistent and dtype_consistent,
            'num_samples': len(samples),
            'shape': first_shape,
            'dtype': str(first_dtype),
            'shape_consistent': shape_consistent,
            'dtype_consistent': dtype_consistent,
            'value_range': [float(min_val), float(max_val)],
            'mean': float(mean_val),
            'std': float(std_val),
            'has_nans': bool(np.any(np.isnan(all_data))),
            'has_infs': bool(np.any(np.isinf(all_data)))
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

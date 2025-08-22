"""
I/O utilities for seismic data handling.
Functions for loading waveforms, converting traces to numpy arrays, and file management.
"""

import os
import numpy as np
from obspy import read, Stream, Trace, UTCDateTime
import tempfile
import zipfile
import gzip
import shutil
from typing import Tuple, Optional, Union


def load_waveform(file_path: str) -> Stream:
    """
    Load seismic waveform from various formats.
    
    Args:
        file_path: Path to seismic data file
        
    Returns:
        ObsPy Stream object
        
    Raises:
        Exception: If file cannot be read or is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Handle compressed files
    if file_path.endswith('.gz'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp:
            with gzip.open(file_path, 'rb') as gz_file:
                shutil.copyfileobj(gz_file, tmp)
            tmp_path = tmp.name
        try:
            st = read(tmp_path)
        finally:
            os.unlink(tmp_path)
    elif file_path.endswith('.zip'):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                zip_file.extractall(tmp_dir)
                # Find the first seismic data file
                for root, dirs, files in os.walk(tmp_dir):
                    for file in files:
                        if file.lower().endswith(('.mseed', '.msd', '.seed', '.sac')):
                            st = read(os.path.join(root, file))
                            break
                    else:
                        continue
                    break
                else:
                    raise ValueError("No seismic data files found in ZIP archive")
    else:
        st = read(file_path)
    
    if len(st) == 0:
        raise ValueError("No traces found in file")
    
    return st


def trace_to_numpy(trace: Trace, target_sampling_rate: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Convert ObsPy Trace to numpy array with optional resampling.
    
    Args:
        trace: ObsPy Trace object
        target_sampling_rate: Target sampling rate for resampling (optional)
        
    Returns:
        Tuple of (data_array, sampling_rate)
    """
    data = trace.data.astype(np.float64)
    sr = trace.stats.sampling_rate
    
    if target_sampling_rate is not None and abs(sr - target_sampling_rate) > 1e-6:
        # Resample if needed
        trace_copy = trace.copy()
        trace_copy.resample(target_sampling_rate)
        data = trace_copy.data.astype(np.float64)
        sr = target_sampling_rate
    
    return data, sr


def numpy_to_trace(data: np.ndarray, sampling_rate: float, 
                   starttime: Optional[UTCDateTime] = None,
                   station: str = "SYNTH", channel: str = "BHZ") -> Trace:
    """
    Convert numpy array to ObsPy Trace.
    
    Args:
        data: Seismic data array
        sampling_rate: Sampling rate in Hz
        starttime: Start time (defaults to current time)
        station: Station code
        channel: Channel code
        
    Returns:
        ObsPy Trace object
    """
    if starttime is None:
        starttime = UTCDateTime()
    
    trace = Trace(data=data.astype(np.float64))
    trace.stats.sampling_rate = sampling_rate
    trace.stats.starttime = starttime
    trace.stats.station = station
    trace.stats.channel = channel
    trace.stats.network = "XX"
    trace.stats.location = ""
    
    return trace


def save_waveform(stream: Stream, output_path: str, format: str = "MSEED") -> str:
    """
    Save ObsPy Stream to file.
    
    Args:
        stream: ObsPy Stream object
        output_path: Output file path
        format: Output format (MSEED, SAC, etc.)
        
    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    stream.write(output_path, format=format)
    return output_path


def ensure_directory(path: str) -> str:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The directory path
    """
    os.makedirs(path, exist_ok=True)
    return path


def clean_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean old temporary files.
    
    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age in hours before deletion
        
    Returns:
        Number of files deleted
    """
    if not os.path.exists(temp_dir):
        return 0
    
    import time
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    deleted_count = 0
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    deleted_count += 1
            except (OSError, IOError):
                continue
    
    return deleted_count


def validate_seismic_file(file_path: str) -> bool:
    """
    Validate that a file contains readable seismic data.
    
    Args:
        file_path: Path to file to validate
        
    Returns:
        True if file is valid seismic data
    """
    try:
        st = load_waveform(file_path)
        return len(st) > 0 and len(st[0].data) > 0
    except Exception:
        return False


def get_file_info(file_path: str) -> dict:
    """
    Get basic information about a seismic data file.
    
    Args:
        file_path: Path to seismic data file
        
    Returns:
        Dictionary with file information
    """
    try:
        st = load_waveform(file_path)
        tr = st[0]
        
        return {
            'filename': os.path.basename(file_path),
            'filesize': os.path.getsize(file_path),
            'format': tr.stats._format if hasattr(tr.stats, '_format') else 'unknown',
            'station': tr.stats.station,
            'channel': tr.stats.channel,
            'network': tr.stats.network,
            'location': tr.stats.location,
            'starttime': str(tr.stats.starttime),
            'endtime': str(tr.stats.endtime),
            'sampling_rate': tr.stats.sampling_rate,
            'npts': tr.stats.npts,
            'duration': tr.stats.endtime - tr.stats.starttime,
            'traces_count': len(st)
        }
    except Exception as e:
        return {
            'filename': os.path.basename(file_path),
            'filesize': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'error': str(e)
        }

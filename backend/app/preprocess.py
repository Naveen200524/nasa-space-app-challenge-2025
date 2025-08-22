"""
Preprocessing utilities for seismic data.
Includes filtering, detrending, and planet-specific processing configurations.
"""

import numpy as np
from obspy import Trace, Stream
from obspy.signal.filter import bandpass, lowpass, highpass
from typing import Dict, Optional, Tuple, Union


# Planet-specific processing configurations
PLANET_PRESETS = {
    'earth': {
        'fmin': 0.5,
        'fmax': 20.0,
        'corners': 4,
        'zerophase': True,
        'description': 'Earth seismic processing - typical earthquake frequencies'
    },
    'mars': {
        'fmin': 0.1,
        'fmax': 5.0,
        'corners': 4,
        'zerophase': True,
        'description': 'Mars seismic processing - InSight SEIS optimized'
    },
    'moon': {
        'fmin': 0.03,
        'fmax': 2.0,
        'corners': 4,
        'zerophase': True,
        'description': 'Moon seismic processing - Apollo-era frequencies'
    }
}


def get_planet_preset(planet: str) -> Dict:
    """
    Get preprocessing configuration for a specific planet.
    
    Args:
        planet: Planet name ('earth', 'mars', 'moon')
        
    Returns:
        Dictionary with preprocessing parameters
    """
    planet_lower = planet.lower()
    return PLANET_PRESETS.get(planet_lower, PLANET_PRESETS['earth'])


def trace_bandpass(trace: Trace, fmin: float, fmax: float, 
                   corners: int = 4, zerophase: bool = True) -> Trace:
    """
    Apply bandpass filter to trace.
    
    Args:
        trace: ObsPy Trace object
        fmin: Minimum frequency (Hz)
        fmax: Maximum frequency (Hz)
        corners: Filter corners
        zerophase: Use zero-phase filter
        
    Returns:
        Filtered trace
    """
    trace_copy = trace.copy()
    
    # Ensure we don't exceed Nyquist frequency
    nyquist = trace_copy.stats.sampling_rate / 2.0
    fmax = min(fmax, nyquist * 0.9)
    
    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be less than fmax ({fmax})")
    
    trace_copy.filter('bandpass', freqmin=fmin, freqmax=fmax, 
                     corners=corners, zerophase=zerophase)
    
    return trace_copy


def trace_lowpass(trace: Trace, freq: float, corners: int = 4, 
                  zerophase: bool = True) -> Trace:
    """
    Apply lowpass filter to trace.
    
    Args:
        trace: ObsPy Trace object
        freq: Cutoff frequency (Hz)
        corners: Filter corners
        zerophase: Use zero-phase filter
        
    Returns:
        Filtered trace
    """
    trace_copy = trace.copy()
    
    # Ensure we don't exceed Nyquist frequency
    nyquist = trace_copy.stats.sampling_rate / 2.0
    freq = min(freq, nyquist * 0.9)
    
    trace_copy.filter('lowpass', freq=freq, corners=corners, zerophase=zerophase)
    
    return trace_copy


def trace_highpass(trace: Trace, freq: float, corners: int = 4, 
                   zerophase: bool = True) -> Trace:
    """
    Apply highpass filter to trace.
    
    Args:
        trace: ObsPy Trace object
        freq: Cutoff frequency (Hz)
        corners: Filter corners
        zerophase: Use zero-phase filter
        
    Returns:
        Filtered trace
    """
    trace_copy = trace.copy()
    trace_copy.filter('highpass', freq=freq, corners=corners, zerophase=zerophase)
    
    return trace_copy


def preprocess_trace(trace: Trace, planet: str = 'earth', 
                     custom_params: Optional[Dict] = None) -> Trace:
    """
    Apply standard preprocessing pipeline for a specific planet.
    
    Args:
        trace: ObsPy Trace object
        planet: Planet name for preset selection
        custom_params: Override default parameters
        
    Returns:
        Preprocessed trace
    """
    trace_copy = trace.copy()
    
    # Get planet-specific parameters
    params = get_planet_preset(planet)
    if custom_params:
        params.update(custom_params)
    
    # Standard preprocessing pipeline
    # 1. Remove mean and linear trend
    trace_copy.detrend('demean')
    trace_copy.detrend('linear')
    
    # 2. Apply bandpass filter
    trace_copy = trace_bandpass(
        trace_copy,
        fmin=params['fmin'],
        fmax=params['fmax'],
        corners=params['corners'],
        zerophase=params['zerophase']
    )
    
    return trace_copy


def normalize_trace(trace: Trace, method: str = 'max') -> Trace:
    """
    Normalize trace amplitude.
    
    Args:
        trace: ObsPy Trace object
        method: Normalization method ('max', 'std', 'rms')
        
    Returns:
        Normalized trace
    """
    trace_copy = trace.copy()
    data = trace_copy.data.astype(np.float64)
    
    if method == 'max':
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val
    elif method == 'std':
        std_val = np.std(data)
        if std_val > 0:
            data = (data - np.mean(data)) / std_val
    elif method == 'rms':
        rms_val = np.sqrt(np.mean(data**2))
        if rms_val > 0:
            data = data / rms_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    trace_copy.data = data
    return trace_copy


def resample_trace(trace: Trace, target_rate: float) -> Trace:
    """
    Resample trace to target sampling rate.
    
    Args:
        trace: ObsPy Trace object
        target_rate: Target sampling rate (Hz)
        
    Returns:
        Resampled trace
    """
    trace_copy = trace.copy()
    
    if abs(trace_copy.stats.sampling_rate - target_rate) > 1e-6:
        trace_copy.resample(target_rate)
    
    return trace_copy


def trim_trace(trace: Trace, start_offset: float = 0, 
               duration: Optional[float] = None) -> Trace:
    """
    Trim trace to specified time window.
    
    Args:
        trace: ObsPy Trace object
        start_offset: Start offset in seconds from trace start
        duration: Duration in seconds (None for rest of trace)
        
    Returns:
        Trimmed trace
    """
    trace_copy = trace.copy()
    
    start_time = trace_copy.stats.starttime + start_offset
    
    if duration is not None:
        end_time = start_time + duration
        trace_copy.trim(starttime=start_time, endtime=end_time)
    else:
        trace_copy.trim(starttime=start_time)
    
    return trace_copy


def detect_gaps(trace: Trace, threshold: float = 1.5) -> list:
    """
    Detect gaps in trace data based on sampling rate.
    
    Args:
        trace: ObsPy Trace object
        threshold: Gap detection threshold (multiples of sampling interval)
        
    Returns:
        List of gap information dictionaries
    """
    dt = 1.0 / trace.stats.sampling_rate
    times = trace.times()
    
    gaps = []
    for i in range(1, len(times)):
        gap_size = times[i] - times[i-1]
        if gap_size > threshold * dt:
            gaps.append({
                'start_time': trace.stats.starttime + times[i-1],
                'end_time': trace.stats.starttime + times[i],
                'duration': gap_size,
                'samples_missing': int(gap_size / dt) - 1
            })
    
    return gaps


def quality_check(trace: Trace) -> Dict:
    """
    Perform basic quality checks on trace data.
    
    Args:
        trace: ObsPy Trace object
        
    Returns:
        Dictionary with quality metrics
    """
    data = trace.data
    
    # Basic statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Quality flags
    has_nans = np.any(np.isnan(data))
    has_infs = np.any(np.isinf(data))
    is_constant = std_val < 1e-12
    has_spikes = np.any(np.abs(data) > 10 * std_val) if std_val > 0 else False
    
    # Gaps
    gaps = detect_gaps(trace)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'has_nans': has_nans,
        'has_infs': has_infs,
        'is_constant': is_constant,
        'has_spikes': has_spikes,
        'gap_count': len(gaps),
        'total_gap_duration': sum(g['duration'] for g in gaps),
        'sampling_rate': trace.stats.sampling_rate,
        'npts': trace.stats.npts,
        'duration': trace.stats.endtime - trace.stats.starttime
    }

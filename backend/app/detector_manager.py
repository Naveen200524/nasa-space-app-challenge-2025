"""
Seismic event detection manager.
Coordinates multiple detection algorithms and manages model loading.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from obspy import Trace, UTCDateTime
from .io_utils import trace_to_numpy
from .preprocess import preprocess_trace, get_planet_preset
import logging


class DetectorManager:
    """
    Manages seismic event detection using multiple algorithms.
    """
    
    def __init__(self, model_root: str = 'models'):
        self.model_root = model_root
        self.models = {}
        self.logger = logging.getLogger(__name__)
        
        # Ensure model directory exists
        os.makedirs(model_root, exist_ok=True)
        
        # Load available models
        self._load_models()
    
    def _load_models(self):
        """Load available detection models."""
        try:
            # Try to load ML models if available
            self._load_ml_models()
        except Exception as e:
            self.logger.warning(f"Could not load ML models: {e}")
        
        # Always have classical detectors available as fallback
        self._setup_classical_detectors()
    
    def _load_ml_models(self):
        """Load machine learning models."""
        # This will be implemented when ML components are added
        pass
    
    def _setup_classical_detectors(self):
        """Setup classical detection algorithms."""
        self.models['sta_lta'] = {
            'type': 'classical',
            'algorithm': 'sta_lta',
            'params': {
                'sta_len': 1.0,  # Short-term average window (seconds)
                'lta_len': 30.0,  # Long-term average window (seconds)
                'trigger_on': 3.0,  # Trigger threshold
                'trigger_off': 1.5   # De-trigger threshold
            }
        }
        
        self.models['z_detect'] = {
            'type': 'classical',
            'algorithm': 'z_detect',
            'params': {
                'threshold': 4.0,  # Z-score threshold
                'window_len': 10.0  # Analysis window (seconds)
            }
        }
    
    def analyze_trace(self, trace: Trace, planet: str = 'earth', 
                     algorithms: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze trace for seismic events.
        
        Args:
            trace: ObsPy Trace object
            planet: Planet name for processing presets
            algorithms: List of algorithms to use (None for all available)
            
        Returns:
            Tuple of (events_dataframe, diagnostics_dict)
        """
        try:
            # Preprocess trace
            processed_trace = preprocess_trace(trace, planet=planet)
            
            # Convert to numpy for analysis
            data, sr = trace_to_numpy(processed_trace)
            
            # Get planet-specific parameters
            planet_config = get_planet_preset(planet)
            
            # Run detection algorithms
            if algorithms is None:
                algorithms = list(self.models.keys())
            
            all_events = []
            diagnostics = {
                'trace_info': {
                    'station': trace.stats.station,
                    'channel': trace.stats.channel,
                    'sampling_rate': sr,
                    'npts': len(data),
                    'duration': len(data) / sr,
                    'planet': planet
                },
                'processing': planet_config,
                'algorithms_used': algorithms,
                'detection_summary': {}
            }
            
            for algorithm in algorithms:
                if algorithm in self.models:
                    events = self._run_detection(data, sr, algorithm, trace.stats.starttime)
                    all_events.extend(events)
                    diagnostics['detection_summary'][algorithm] = len(events)
            
            # Convert to DataFrame
            if all_events:
                events_df = pd.DataFrame(all_events)
                # Remove duplicates and sort by time
                events_df = events_df.drop_duplicates(subset=['time']).sort_values('time')
                events_df = events_df.reset_index(drop=True)
            else:
                events_df = pd.DataFrame(columns=['time', 'magnitude', 'confidence', 'algorithm'])
            
            diagnostics['total_events'] = len(events_df)
            
            return events_df, diagnostics
            
        except Exception as e:
            self.logger.error(f"Error in trace analysis: {e}")
            # Return empty results on error
            empty_df = pd.DataFrame(columns=['time', 'magnitude', 'confidence', 'algorithm'])
            error_diagnostics = {
                'error': str(e),
                'trace_info': {
                    'station': getattr(trace.stats, 'station', 'unknown'),
                    'channel': getattr(trace.stats, 'channel', 'unknown')
                }
            }
            return empty_df, error_diagnostics
    
    def _run_detection(self, data: np.ndarray, sr: float, algorithm: str, 
                      starttime: UTCDateTime) -> List[Dict]:
        """
        Run a specific detection algorithm.
        
        Args:
            data: Seismic data array
            sr: Sampling rate
            algorithm: Algorithm name
            starttime: Trace start time
            
        Returns:
            List of detected events
        """
        model_config = self.models[algorithm]
        
        if model_config['type'] == 'classical':
            if model_config['algorithm'] == 'sta_lta':
                return self._sta_lta_detection(data, sr, starttime, model_config['params'])
            elif model_config['algorithm'] == 'z_detect':
                return self._z_score_detection(data, sr, starttime, model_config['params'])
        
        return []
    
    def _sta_lta_detection(self, data: np.ndarray, sr: float, 
                          starttime: UTCDateTime, params: Dict) -> List[Dict]:
        """
        STA/LTA detection algorithm.
        """
        from obspy.signal.trigger import classic_sta_lta, trigger_onset
        
        try:
            sta_len_samples = int(params['sta_len'] * sr)
            lta_len_samples = int(params['lta_len'] * sr)
            
            # Calculate STA/LTA
            cft = classic_sta_lta(data, sta_len_samples, lta_len_samples)
            
            # Find triggers
            triggers = trigger_onset(cft, params['trigger_on'], params['trigger_off'])
            
            events = []
            for trigger in triggers:
                trigger_time = starttime + trigger[0] / sr
                
                # Estimate magnitude based on peak amplitude around trigger
                start_idx = max(0, trigger[0] - int(5 * sr))
                end_idx = min(len(data), trigger[0] + int(10 * sr))
                peak_amplitude = np.max(np.abs(data[start_idx:end_idx]))
                
                # Simple magnitude estimation (log scale)
                magnitude = np.log10(peak_amplitude * 1e9) if peak_amplitude > 0 else 0
                magnitude = max(0, min(magnitude, 10))  # Clamp to reasonable range
                
                # Confidence based on STA/LTA ratio
                confidence = min(cft[trigger[0]] / params['trigger_on'], 1.0)
                
                events.append({
                    'time': trigger_time,
                    'magnitude': magnitude,
                    'confidence': confidence,
                    'algorithm': 'sta_lta',
                    'amplitude': peak_amplitude,
                    'duration': (trigger[1] - trigger[0]) / sr if len(trigger) > 1 else 0
                })
            
            return events
            
        except Exception as e:
            self.logger.error(f"STA/LTA detection error: {e}")
            return []
    
    def _z_score_detection(self, data: np.ndarray, sr: float, 
                          starttime: UTCDateTime, params: Dict) -> List[Dict]:
        """
        Z-score based detection algorithm.
        """
        try:
            window_samples = int(params['window_len'] * sr)
            threshold = params['threshold']
            
            # Calculate rolling statistics
            events = []
            
            for i in range(window_samples, len(data) - window_samples, window_samples // 2):
                window_data = data[i-window_samples:i+window_samples]
                
                # Calculate z-score for center point
                mean_val = np.mean(window_data)
                std_val = np.std(window_data)
                
                if std_val > 0:
                    z_score = abs((data[i] - mean_val) / std_val)
                    
                    if z_score > threshold:
                        event_time = starttime + i / sr
                        
                        # Estimate magnitude
                        magnitude = min(z_score / 2, 8)  # Scale z-score to magnitude
                        
                        # Confidence based on z-score
                        confidence = min(z_score / (threshold * 2), 1.0)
                        
                        events.append({
                            'time': event_time,
                            'magnitude': magnitude,
                            'confidence': confidence,
                            'algorithm': 'z_detect',
                            'amplitude': abs(data[i]),
                            'z_score': z_score
                        })
            
            return events
            
        except Exception as e:
            self.logger.error(f"Z-score detection error: {e}")
            return []
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available detection algorithms."""
        return list(self.models.keys())
    
    def get_algorithm_info(self, algorithm: str) -> Optional[Dict]:
        """Get information about a specific algorithm."""
        return self.models.get(algorithm)
    
    def add_custom_detector(self, name: str, detector_config: Dict):
        """Add a custom detection algorithm."""
        self.models[name] = detector_config
    
    def set_algorithm_params(self, algorithm: str, params: Dict):
        """Update parameters for an algorithm."""
        if algorithm in self.models:
            self.models[algorithm]['params'].update(params)
    
    def benchmark_algorithms(self, trace: Trace, planet: str = 'earth') -> Dict:
        """
        Benchmark all available algorithms on a trace.
        
        Args:
            trace: Test trace
            planet: Planet configuration
            
        Returns:
            Performance metrics for each algorithm
        """
        import time
        
        results = {}
        
        for algorithm in self.models.keys():
            start_time = time.time()
            
            try:
                events_df, diagnostics = self.analyze_trace(trace, planet, [algorithm])
                execution_time = time.time() - start_time
                
                results[algorithm] = {
                    'execution_time': execution_time,
                    'events_detected': len(events_df),
                    'success': True,
                    'diagnostics': diagnostics
                }
                
            except Exception as e:
                execution_time = time.time() - start_time
                results[algorithm] = {
                    'execution_time': execution_time,
                    'events_detected': 0,
                    'success': False,
                    'error': str(e)
                }
        
        return results

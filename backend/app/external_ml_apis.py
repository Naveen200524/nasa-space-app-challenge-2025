"""
External ML API integrations for advanced seismic signal processing.
Based on api.md specifications for Hugging Face, AWS, and Google Cloud ML services.
"""

import os
import requests
import numpy as np
import base64
import io
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import asyncio
import aiohttp
from scipy.io import wavfile


logger = logging.getLogger(__name__)


@dataclass
class MLAPIConfig:
    """Configuration for ML API services."""
    name: str
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    rate_limit: Optional[int] = None
    headers: Optional[Dict[str, str]] = None


class ExternalMLAPIs:
    """
    Integration with external ML APIs for seismic signal processing.
    Supports Hugging Face, AWS SageMaker, and Google Cloud AI Platform.
    """
    
    def __init__(self):
        self.apis = {}
        self.rate_limits = {}
        self._initialize_apis()
    
    def _initialize_apis(self):
        """Initialize ML API configurations."""
        
        # Hugging Face Inference API
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_KEY')
        self.apis['huggingface'] = MLAPIConfig(
            name="Hugging Face",
            base_url="https://api-inference.huggingface.co/models/",
            api_key=hf_token,
            headers={'Authorization': f'Bearer {hf_token}'} if hf_token else None
        )
        
        # AWS SageMaker (placeholder - requires specific endpoint configuration)
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.apis['aws_sagemaker'] = MLAPIConfig(
            name="AWS SageMaker",
            base_url="https://runtime.sagemaker.us-east-1.amazonaws.com/",
            api_key=aws_key
        )
        
        # Google Cloud AI Platform
        gcp_key = os.getenv('GOOGLE_API_KEY')
        self.apis['google_ai'] = MLAPIConfig(
            name="Google Cloud AI",
            base_url="https://ml.googleapis.com/v1/",
            api_key=gcp_key
        )
    
    def _prepare_audio_data(self, seismic_data: np.ndarray, sampling_rate: float) -> bytes:
        """
        Convert seismic data to audio format for ML APIs.
        
        Args:
            seismic_data: 1D numpy array of seismic data
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Audio data as bytes
        """
        # Normalize data to audio range
        normalized = seismic_data.copy()
        if np.std(normalized) > 0:
            normalized = (normalized - np.mean(normalized)) / np.std(normalized)
        
        # Scale to 16-bit integer range
        audio_data = (normalized * 32767).astype(np.int16)
        
        # Create WAV file in memory
        buffer = io.BytesIO()
        wavfile.write(buffer, int(sampling_rate), audio_data)
        buffer.seek(0)
        
        return buffer.read()
    
    def _prepare_spectrogram_data(self, seismic_data: np.ndarray, 
                                 sampling_rate: float) -> np.ndarray:
        """
        Convert seismic data to spectrogram for image-based ML models.
        
        Args:
            seismic_data: 1D numpy array of seismic data
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectrogram as 2D array
        """
        from scipy import signal
        
        # Compute spectrogram
        nperseg = min(256, len(seismic_data) // 8)
        f, t, Sxx = signal.spectrogram(seismic_data, sampling_rate, nperseg=nperseg)
        
        # Convert to dB and normalize
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        Sxx_norm = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db))
        
        return Sxx_norm
    
    async def classify_with_huggingface(self, seismic_data: np.ndarray, 
                                       sampling_rate: float,
                                       model_name: str = "facebook/wav2vec2-base") -> Dict[str, Any]:
        """
        Classify seismic signal using Hugging Face models.
        
        Args:
            seismic_data: 1D numpy array of seismic data
            sampling_rate: Sampling rate in Hz
            model_name: Hugging Face model name
            
        Returns:
            Classification results
        """
        try:
            config = self.apis['huggingface']
            if not config.api_key:
                raise ValueError("Hugging Face API key not configured")
            
            # Prepare audio data
            audio_bytes = self._prepare_audio_data(seismic_data, sampling_rate)
            
            # Make API request
            url = config.base_url + model_name
            headers = config.headers.copy()
            headers['Content-Type'] = 'audio/wav'
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
                async with session.post(url, data=audio_bytes, headers=headers) as response:
                    if response.status == 503:
                        # Model is loading, wait and retry
                        await asyncio.sleep(20)
                        async with session.post(url, data=audio_bytes, headers=headers) as retry_response:
                            retry_response.raise_for_status()
                            result = await retry_response.json()
                    else:
                        response.raise_for_status()
                        result = await response.json()
            
            return {
                'model': model_name,
                'provider': 'huggingface',
                'result': result,
                'confidence': result[0]['score'] if isinstance(result, list) and result else None
            }
            
        except Exception as e:
            logger.error(f"Hugging Face classification failed: {e}")
            return {
                'model': model_name,
                'provider': 'huggingface',
                'error': str(e)
            }
    
    def classify_with_huggingface_sync(self, seismic_data: np.ndarray, 
                                      sampling_rate: float,
                                      model_name: str = "facebook/wav2vec2-base") -> Dict[str, Any]:
        """Synchronous wrapper for Hugging Face classification."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.classify_with_huggingface(seismic_data, sampling_rate, model_name)
            )
        finally:
            loop.close()
    
    async def analyze_with_multiple_models(self, seismic_data: np.ndarray, 
                                         sampling_rate: float) -> Dict[str, Any]:
        """
        Analyze seismic data with multiple ML models for ensemble results.
        
        Args:
            seismic_data: 1D numpy array of seismic data
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Combined analysis results
        """
        # Define models to try
        models = [
            "facebook/wav2vec2-base",
            "microsoft/wavlm-base",
            "facebook/hubert-base-ls960"
        ]
        
        results = []
        
        for model in models:
            try:
                result = await self.classify_with_huggingface(seismic_data, sampling_rate, model)
                results.append(result)
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue
        
        # Combine results
        combined = {
            'models_used': len(results),
            'individual_results': results,
            'ensemble_confidence': np.mean([
                r.get('confidence', 0) for r in results if 'confidence' in r
            ]) if results else 0
        }
        
        return combined
    
    def extract_features_with_ml(self, seismic_data: np.ndarray, 
                                sampling_rate: float) -> Dict[str, Any]:
        """
        Extract advanced features using ML models.
        
        Args:
            seismic_data: 1D numpy array of seismic data
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Extracted features
        """
        try:
            # Basic signal features
            features = {
                'statistical': {
                    'mean': float(np.mean(seismic_data)),
                    'std': float(np.std(seismic_data)),
                    'skewness': float(self._calculate_skewness(seismic_data)),
                    'kurtosis': float(self._calculate_kurtosis(seismic_data)),
                    'energy': float(np.sum(seismic_data ** 2)),
                    'zero_crossings': int(np.sum(np.diff(np.sign(seismic_data)) != 0))
                },
                'frequency': {
                    'dominant_frequency': float(self._get_dominant_frequency(seismic_data, sampling_rate)),
                    'spectral_centroid': float(self._spectral_centroid(seismic_data, sampling_rate)),
                    'spectral_rolloff': float(self._spectral_rolloff(seismic_data, sampling_rate))
                }
            }
            
            # Try to get ML-based features if API is available
            if self.apis['huggingface'].api_key:
                try:
                    ml_result = self.classify_with_huggingface_sync(seismic_data, sampling_rate)
                    features['ml_classification'] = ml_result
                except Exception as e:
                    logger.warning(f"ML feature extraction failed: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {'error': str(e)}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _get_dominant_frequency(self, data: np.ndarray, sampling_rate: float) -> float:
        """Get dominant frequency using FFT."""
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
        
        # Get positive frequencies only
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        
        # Find dominant frequency
        dominant_idx = np.argmax(positive_fft)
        return positive_freqs[dominant_idx]
    
    def _spectral_centroid(self, data: np.ndarray, sampling_rate: float) -> float:
        """Calculate spectral centroid."""
        fft = np.abs(np.fft.fft(data))
        freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
        
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft[:len(fft)//2]
        
        if np.sum(positive_fft) == 0:
            return 0
        
        return np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
    
    def _spectral_rolloff(self, data: np.ndarray, sampling_rate: float, 
                         rolloff_percent: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        fft = np.abs(np.fft.fft(data))
        freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
        
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft[:len(fft)//2]
        
        total_energy = np.sum(positive_fft)
        if total_energy == 0:
            return 0
        
        cumulative_energy = np.cumsum(positive_fft)
        rolloff_threshold = rolloff_percent * total_energy
        
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            return positive_freqs[rolloff_idx[0]]
        else:
            return positive_freqs[-1]
    
    def get_api_status(self) -> Dict[str, Dict]:
        """Get status of all configured ML APIs."""
        status = {}
        
        for api_name, config in self.apis.items():
            status[api_name] = {
                'name': config.name,
                'configured': bool(config.api_key),
                'base_url': config.base_url,
                'timeout': config.timeout
            }
        
        return status

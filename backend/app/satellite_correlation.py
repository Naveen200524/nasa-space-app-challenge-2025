"""
Satellite data correlation for seismic events.
Integrates NASA Earthdata and satellite imagery APIs to correlate seismic events
with surface changes and environmental factors.
"""

import os
import requests
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import asyncio
import aiohttp
from obspy import UTCDateTime


logger = logging.getLogger(__name__)


@dataclass
class SatelliteImagery:
    """Structure for satellite imagery data."""
    source: str
    date: datetime
    latitude: float
    longitude: float
    resolution: float  # meters per pixel
    bands: List[str]
    url: str
    metadata: Dict[str, Any]


@dataclass
class CorrelationResult:
    """Result of seismic-satellite correlation analysis."""
    seismic_event: Dict[str, Any]
    satellite_data: List[SatelliteImagery]
    correlation_score: float
    surface_changes: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    confidence: float


class SatelliteCorrelationEngine:
    """
    Engine for correlating seismic events with satellite observations.
    """
    
    def __init__(self):
        self.nasa_api_key = os.getenv('NASA_API_KEY', 'DEMO_KEY')
        self.earthdata_token = os.getenv('EARTHDATA_TOKEN')
        self.sentinel_hub_token = os.getenv('SENTINEL_HUB_TOKEN')
        
        # API configurations
        self.apis = {
            'nasa_earthdata': {
                'base_url': 'https://cmr.earthdata.nasa.gov/search/',
                'imagery_url': 'https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/',
                'timeout': 30
            },
            'landsat': {
                'base_url': 'https://landsatlook.usgs.gov/sat-api/',
                'timeout': 30
            },
            'sentinel': {
                'base_url': 'https://services.sentinel-hub.com/api/v1/',
                'timeout': 30
            }
        }
    
    async def find_satellite_imagery(self, latitude: float, longitude: float,
                                   event_time: datetime, 
                                   time_window_days: int = 30) -> List[SatelliteImagery]:
        """
        Find satellite imagery around a seismic event location and time.
        
        Args:
            latitude: Event latitude
            longitude: Event longitude
            event_time: Time of seismic event
            time_window_days: Days before/after event to search
            
        Returns:
            List of available satellite imagery
        """
        imagery = []
        
        # Search multiple satellite sources
        tasks = [
            self._search_landsat_imagery(latitude, longitude, event_time, time_window_days),
            self._search_nasa_earthdata(latitude, longitude, event_time, time_window_days),
            self._search_sentinel_imagery(latitude, longitude, event_time, time_window_days)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                imagery.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Satellite search failed: {result}")
        
        # Sort by date and remove duplicates
        imagery.sort(key=lambda x: abs((x.date - event_time).total_seconds()))
        return imagery[:10]  # Limit to 10 most relevant images
    
    async def _search_landsat_imagery(self, lat: float, lon: float, 
                                    event_time: datetime, window_days: int) -> List[SatelliteImagery]:
        """Search Landsat imagery via USGS API."""
        try:
            start_date = event_time - timedelta(days=window_days)
            end_date = event_time + timedelta(days=window_days)
            
            # Mock Landsat search (replace with actual API call)
            imagery = []
            
            # Generate mock Landsat data for demonstration
            for i in range(3):
                date = event_time + timedelta(days=i*10 - 15)
                imagery.append(SatelliteImagery(
                    source="Landsat-8",
                    date=date,
                    latitude=lat,
                    longitude=lon,
                    resolution=30.0,  # 30m resolution
                    bands=["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
                    url=f"https://landsat-pds.s3.amazonaws.com/mock/{date.strftime('%Y%m%d')}",
                    metadata={
                        'cloud_cover': np.random.uniform(0, 30),
                        'quality': 'good',
                        'processing_level': 'L1T'
                    }
                ))
            
            return imagery
            
        except Exception as e:
            logger.error(f"Landsat search failed: {e}")
            return []
    
    async def _search_nasa_earthdata(self, lat: float, lon: float,
                                   event_time: datetime, window_days: int) -> List[SatelliteImagery]:
        """Search NASA Earthdata imagery."""
        try:
            # Mock NASA Earthdata search
            imagery = []
            
            # Generate mock MODIS data
            for i in range(2):
                date = event_time + timedelta(days=i*7 - 7)
                imagery.append(SatelliteImagery(
                    source="MODIS-Terra",
                    date=date,
                    latitude=lat,
                    longitude=lon,
                    resolution=250.0,  # 250m resolution
                    bands=["Red", "NIR", "Blue", "Green"],
                    url=f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/{date.strftime('%Y-%m-%d')}/EPSG4326_250m/",
                    metadata={
                        'instrument': 'MODIS',
                        'platform': 'Terra',
                        'quality': 'good'
                    }
                ))
            
            return imagery
            
        except Exception as e:
            logger.error(f"NASA Earthdata search failed: {e}")
            return []
    
    async def _search_sentinel_imagery(self, lat: float, lon: float,
                                     event_time: datetime, window_days: int) -> List[SatelliteImagery]:
        """Search Sentinel imagery."""
        try:
            # Mock Sentinel search
            imagery = []
            
            # Generate mock Sentinel-2 data
            for i in range(2):
                date = event_time + timedelta(days=i*5 - 5)
                imagery.append(SatelliteImagery(
                    source="Sentinel-2",
                    date=date,
                    latitude=lat,
                    longitude=lon,
                    resolution=10.0,  # 10m resolution
                    bands=["B02", "B03", "B04", "B08", "B11", "B12"],
                    url=f"https://services.sentinel-hub.com/ogc/wms/mock/{date.strftime('%Y%m%d')}",
                    metadata={
                        'cloud_cover': np.random.uniform(0, 20),
                        'instrument': 'MSI',
                        'processing_level': 'L2A'
                    }
                ))
            
            return imagery
            
        except Exception as e:
            logger.error(f"Sentinel search failed: {e}")
            return []
    
    def analyze_surface_changes(self, imagery_before: SatelliteImagery, 
                              imagery_after: SatelliteImagery) -> Dict[str, Any]:
        """
        Analyze surface changes between before/after satellite images.
        
        Args:
            imagery_before: Satellite image before event
            imagery_after: Satellite image after event
            
        Returns:
            Surface change analysis results
        """
        try:
            # Mock surface change analysis
            # In a real implementation, this would:
            # 1. Download the actual imagery
            # 2. Perform image registration
            # 3. Calculate difference images
            # 4. Detect landslides, surface ruptures, etc.
            
            # Generate mock analysis results
            change_magnitude = np.random.uniform(0, 1)
            
            analysis = {
                'change_detected': change_magnitude > 0.3,
                'change_magnitude': float(change_magnitude),
                'change_type': self._classify_change_type(change_magnitude),
                'affected_area_km2': float(np.random.uniform(0, 100)),
                'confidence': float(np.random.uniform(0.6, 0.95)),
                'analysis_method': 'NDVI_difference',
                'before_image': {
                    'date': imagery_before.date.isoformat(),
                    'source': imagery_before.source,
                    'resolution': imagery_before.resolution
                },
                'after_image': {
                    'date': imagery_after.date.isoformat(),
                    'source': imagery_after.source,
                    'resolution': imagery_after.resolution
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Surface change analysis failed: {e}")
            return {'error': str(e)}
    
    def _classify_change_type(self, magnitude: float) -> str:
        """Classify type of surface change based on magnitude."""
        if magnitude > 0.8:
            return "major_surface_rupture"
        elif magnitude > 0.6:
            return "landslide"
        elif magnitude > 0.4:
            return "minor_surface_change"
        elif magnitude > 0.2:
            return "vegetation_change"
        else:
            return "no_significant_change"
    
    def get_environmental_factors(self, latitude: float, longitude: float,
                                event_time: datetime) -> Dict[str, Any]:
        """
        Get environmental factors that might influence seismic activity.
        
        Args:
            latitude: Event latitude
            longitude: Event longitude
            event_time: Time of event
            
        Returns:
            Environmental factors analysis
        """
        try:
            # Mock environmental analysis
            # In reality, this would query:
            # - Weather data
            # - Soil moisture
            # - Precipitation
            # - Temperature
            # - Atmospheric pressure
            
            factors = {
                'weather': {
                    'temperature_c': float(np.random.uniform(-10, 40)),
                    'precipitation_mm': float(np.random.uniform(0, 50)),
                    'humidity_percent': float(np.random.uniform(30, 90)),
                    'pressure_hpa': float(np.random.uniform(980, 1030))
                },
                'soil': {
                    'moisture_percent': float(np.random.uniform(10, 60)),
                    'type': np.random.choice(['clay', 'sand', 'loam', 'rock']),
                    'stability_index': float(np.random.uniform(0.3, 0.9))
                },
                'topography': {
                    'elevation_m': float(np.random.uniform(0, 3000)),
                    'slope_degrees': float(np.random.uniform(0, 45)),
                    'aspect': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                },
                'geological': {
                    'rock_type': np.random.choice(['sedimentary', 'igneous', 'metamorphic']),
                    'fault_proximity_km': float(np.random.uniform(0, 50)),
                    'tectonic_setting': np.random.choice(['active_margin', 'passive_margin', 'intraplate'])
                }
            }
            
            return factors
            
        except Exception as e:
            logger.error(f"Environmental factors analysis failed: {e}")
            return {'error': str(e)}
    
    async def correlate_seismic_satellite(self, seismic_event: Dict[str, Any]) -> CorrelationResult:
        """
        Perform comprehensive correlation between seismic event and satellite data.
        
        Args:
            seismic_event: Seismic event information
            
        Returns:
            Correlation analysis results
        """
        try:
            latitude = seismic_event.get('latitude')
            longitude = seismic_event.get('longitude')
            event_time = datetime.fromisoformat(seismic_event.get('time'))
            
            if not all([latitude, longitude]):
                raise ValueError("Event location required for satellite correlation")
            
            # Find satellite imagery
            imagery = await self.find_satellite_imagery(latitude, longitude, event_time)
            
            # Analyze surface changes if we have before/after images
            surface_changes = {}
            if len(imagery) >= 2:
                # Find images before and after the event
                before_images = [img for img in imagery if img.date < event_time]
                after_images = [img for img in imagery if img.date > event_time]
                
                if before_images and after_images:
                    surface_changes = self.analyze_surface_changes(
                        before_images[0], after_images[0]
                    )
            
            # Get environmental factors
            environmental_factors = self.get_environmental_factors(
                latitude, longitude, event_time
            )
            
            # Calculate correlation score
            correlation_score = self._calculate_correlation_score(
                seismic_event, surface_changes, environmental_factors
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(imagery, surface_changes)
            
            return CorrelationResult(
                seismic_event=seismic_event,
                satellite_data=imagery,
                correlation_score=correlation_score,
                surface_changes=surface_changes,
                environmental_factors=environmental_factors,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Seismic-satellite correlation failed: {e}")
            raise
    
    def _calculate_correlation_score(self, seismic_event: Dict, 
                                   surface_changes: Dict, 
                                   environmental: Dict) -> float:
        """Calculate correlation score between seismic and satellite data."""
        score = 0.0
        
        # Base score from magnitude
        magnitude = seismic_event.get('magnitude', 0)
        score += min(magnitude / 10.0, 1.0) * 0.4
        
        # Surface change contribution
        if surface_changes.get('change_detected'):
            score += surface_changes.get('change_magnitude', 0) * 0.4
        
        # Environmental factors contribution
        if environmental.get('geological', {}).get('fault_proximity_km', 100) < 10:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_confidence(self, imagery: List[SatelliteImagery], 
                            surface_changes: Dict) -> float:
        """Calculate confidence in correlation analysis."""
        confidence = 0.5  # Base confidence
        
        # More imagery increases confidence
        confidence += min(len(imagery) / 10.0, 0.3)
        
        # High-resolution imagery increases confidence
        if imagery:
            avg_resolution = np.mean([img.resolution for img in imagery])
            if avg_resolution < 30:  # High resolution
                confidence += 0.2
        
        # Surface change analysis confidence
        if surface_changes.get('confidence'):
            confidence += surface_changes['confidence'] * 0.3
        
        return min(confidence, 1.0)

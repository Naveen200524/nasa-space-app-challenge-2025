"""
Enhanced Data Integration Hub based on api.md specifications.
Provides comprehensive data source integration with caching, rate limiting,
and support for multiple seismic data providers.
"""

import os
import time
import json
import requests
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlencode
import asyncio
import aiohttp
from dataclasses import dataclass
from enum import Enum
from obspy import UTCDateTime


logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Enumeration of supported data sources."""
    USGS_REALTIME = "usgs_realtime"
    USGS_HISTORICAL = "usgs_historical"
    EMSC = "emsc"
    IRIS_EVENTS = "iris_events"
    IRIS_STATIONS = "iris_stations"
    NASA_INSIGHT = "nasa_insight"
    JAPAN_JMA = "japan_jma"
    GEONET_NZ = "geonet_nz"


@dataclass
class APIConfig:
    """Configuration for an API endpoint."""
    url: str
    key: Optional[str] = None
    rate_limit: Optional[int] = None  # requests per hour
    cache_ttl: int = 3600  # seconds
    timeout: int = 30
    headers: Optional[Dict[str, str]] = None


class DataIntegrationHub:
    """
    Enhanced data integration hub with caching, rate limiting, and multiple sources.
    Based on api.md specifications for comprehensive seismic data access.
    """
    
    def __init__(self, cache_dir: str = "cache", config_file: Optional[str] = None):
        self.cache_dir = cache_dir
        self.cache = {}  # In-memory cache
        self.rate_limits = {}  # Track API call timestamps
        self.apis = {}
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize API configurations
        self._initialize_apis(config_file)
        
        logger.info(f"Data Integration Hub initialized with {len(self.apis)} data sources")
    
    def _initialize_apis(self, config_file: Optional[str] = None):
        """Initialize API configurations from api.md specifications."""
        
        # Load custom config if provided
        custom_config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
        
        # USGS Real-time Earthquake Data
        self.apis[DataSource.USGS_REALTIME] = APIConfig(
            url="https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/",
            rate_limit=None,  # No rate limit
            cache_ttl=60,  # 1 minute for real-time data
            **custom_config.get('usgs_realtime', {})
        )
        
        # USGS Historical Data
        self.apis[DataSource.USGS_HISTORICAL] = APIConfig(
            url="https://earthquake.usgs.gov/fdsnws/event/1/",
            rate_limit=None,
            cache_ttl=86400,  # 1 day for historical data
            **custom_config.get('usgs_historical', {})
        )
        
        # European-Mediterranean Seismological Centre
        self.apis[DataSource.EMSC] = APIConfig(
            url="https://www.seismicportal.eu/fdsnws/event/1/",
            rate_limit=1000,  # Conservative estimate
            cache_ttl=300,  # 5 minutes
            **custom_config.get('emsc', {})
        )
        
        # IRIS Event Services
        self.apis[DataSource.IRIS_EVENTS] = APIConfig(
            url="http://service.iris.edu/fdsnws/event/1/",
            rate_limit=None,
            cache_ttl=3600,  # 1 hour
            **custom_config.get('iris_events', {})
        )
        
        # IRIS Station Services
        self.apis[DataSource.IRIS_STATIONS] = APIConfig(
            url="http://service.iris.edu/fdsnws/station/1/",
            rate_limit=None,
            cache_ttl=86400,  # 1 day for station metadata
            **custom_config.get('iris_stations', {})
        )
        
        # NASA InSight Weather API
        self.apis[DataSource.NASA_INSIGHT] = APIConfig(
            url="https://api.nasa.gov/insight_weather/",
            key=os.getenv('NASA_API_KEY', 'DEMO_KEY'),
            rate_limit=1000,  # per hour
            cache_ttl=3600,  # 1 hour
            **custom_config.get('nasa_insight', {})
        )
        
        # GeoNet New Zealand
        self.apis[DataSource.GEONET_NZ] = APIConfig(
            url="https://api.geonet.org.nz/",
            rate_limit=1000,  # Conservative estimate
            cache_ttl=300,  # 5 minutes
            **custom_config.get('geonet_nz', {})
        )
    
    def _get_cache_key(self, source: DataSource, endpoint: str, params: Dict) -> str:
        """Generate cache key for request."""
        param_str = json.dumps(params, sort_keys=True)
        return f"{source.value}_{endpoint}_{hash(param_str)}"
    
    def _check_cache(self, cache_key: str, ttl: int) -> Optional[Any]:
        """Check if cached data is still valid."""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < ttl:
                return cached_data
            else:
                del self.cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Any):
        """Store data in cache."""
        self.cache[cache_key] = (data, time.time())
    
    def _check_rate_limit(self, source: DataSource) -> bool:
        """Check if API rate limit allows request."""
        config = self.apis[source]
        if not config.rate_limit:
            return True
        
        now = time.time()
        calls = self.rate_limits.get(source, [])
        
        # Remove calls older than 1 hour
        recent_calls = [call_time for call_time in calls if now - call_time < 3600]
        
        if len(recent_calls) >= config.rate_limit:
            return False
        
        self.rate_limits[source] = recent_calls
        return True
    
    def _update_rate_limit(self, source: DataSource):
        """Update rate limit tracking."""
        if source not in self.rate_limits:
            self.rate_limits[source] = []
        self.rate_limits[source].append(time.time())
    
    async def fetch_data(self, source: DataSource, endpoint: str = "", 
                        params: Optional[Dict] = None) -> Dict:
        """
        Fetch data from specified source with caching and rate limiting.
        
        Args:
            source: Data source to query
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            API response data
            
        Raises:
            Exception: If request fails or rate limit exceeded
        """
        if params is None:
            params = {}
        
        config = self.apis[source]
        cache_key = self._get_cache_key(source, endpoint, params)
        
        # Check cache first
        cached_data = self._check_cache(cache_key, config.cache_ttl)
        if cached_data:
            logger.debug(f"Cache hit for {source.value}/{endpoint}")
            return cached_data
        
        # Check rate limit
        if not self._check_rate_limit(source):
            raise Exception(f"Rate limit exceeded for {source.value}")
        
        # Prepare request
        url = config.url + endpoint
        if config.key:
            params['api_key'] = config.key
        
        headers = config.headers or {}
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
                async with session.get(url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    
                    # Handle different content types
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        data = await response.json()
                    else:
                        data = await response.text()
            
            # Update rate limit and cache
            self._update_rate_limit(source)
            self._set_cache(cache_key, data)
            
            logger.debug(f"Fetched data from {source.value}/{endpoint}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch from {source.value}: {e}")
            raise
    
    def fetch_data_sync(self, source: DataSource, endpoint: str = "", 
                       params: Optional[Dict] = None) -> Dict:
        """Synchronous wrapper for fetch_data."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.fetch_data(source, endpoint, params))
        finally:
            loop.close()
    
    async def get_recent_earthquakes(self, magnitude: float = 4.0, 
                                   hours: int = 24, sources: Optional[List[DataSource]] = None) -> List[Dict]:
        """
        Get recent earthquakes from multiple sources.
        
        Args:
            magnitude: Minimum magnitude
            hours: Hours back to search
            sources: List of sources to query (default: all available)
            
        Returns:
            List of normalized earthquake events
        """
        if sources is None:
            sources = [DataSource.USGS_REALTIME, DataSource.EMSC]
        
        all_events = []
        
        for source in sources:
            try:
                if source == DataSource.USGS_REALTIME:
                    # USGS real-time feeds
                    feed_map = {
                        1: "all_hour.geojson",
                        24: "4.5_day.geojson",
                        168: "2.5_week.geojson"  # 7 days
                    }
                    
                    # Choose appropriate feed based on time range
                    if hours <= 1:
                        feed = "all_hour.geojson"
                    elif hours <= 24:
                        feed = "4.5_day.geojson"
                    else:
                        feed = "2.5_week.geojson"
                    
                    data = await self.fetch_data(source, feed)
                    events = self._normalize_usgs_data(data, magnitude)
                    
                elif source == DataSource.EMSC:
                    # EMSC FDSN web service
                    start_time = datetime.utcnow() - timedelta(hours=hours)
                    params = {
                        'format': 'json',
                        'minmagnitude': magnitude,
                        'starttime': start_time.isoformat(),
                        'orderby': 'time-asc'
                    }
                    
                    data = await self.fetch_data(source, "query", params)
                    events = self._normalize_emsc_data(data)
                
                all_events.extend(events)
                
            except Exception as e:
                logger.warning(f"Failed to fetch from {source.value}: {e}")
                continue
        
        # Remove duplicates and sort by time
        unique_events = self._deduplicate_events(all_events)
        return sorted(unique_events, key=lambda x: x['time'], reverse=True)
    
    def _normalize_usgs_data(self, data: Dict, min_magnitude: float) -> List[Dict]:
        """Normalize USGS GeoJSON data to common format."""
        events = []
        
        if 'features' not in data:
            return events
        
        for feature in data['features']:
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            
            # Filter by magnitude
            if props.get('mag', 0) < min_magnitude:
                continue
            
            event = {
                'id': props.get('ids', '').split(',')[0] if props.get('ids') else props.get('code'),
                'time': datetime.fromtimestamp(props['time'] / 1000),
                'magnitude': props.get('mag'),
                'magnitude_type': props.get('magType'),
                'location': props.get('place'),
                'latitude': coords[1],
                'longitude': coords[0],
                'depth': coords[2],
                'source': 'USGS',
                'url': props.get('url'),
                'tsunami': props.get('tsunami', 0) == 1
            }
            events.append(event)
        
        return events
    
    def _normalize_emsc_data(self, data: Dict) -> List[Dict]:
        """Normalize EMSC data to common format."""
        events = []
        
        # Handle different EMSC response formats
        if isinstance(data, dict) and 'features' in data:
            # GeoJSON format
            for feature in data['features']:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                event = {
                    'id': props.get('publicID'),
                    'time': datetime.fromisoformat(props['time'].replace('Z', '+00:00')),
                    'magnitude': props.get('mag'),
                    'magnitude_type': props.get('magtype'),
                    'location': props.get('flynn_region'),
                    'latitude': coords[1],
                    'longitude': coords[0],
                    'depth': coords[2],
                    'source': 'EMSC',
                    'url': props.get('url')
                }
                events.append(event)
        
        return events
    
    def _deduplicate_events(self, events: List[Dict]) -> List[Dict]:
        """Remove duplicate events based on time and location proximity."""
        unique_events = []
        
        for event in events:
            is_duplicate = False
            
            for existing in unique_events:
                # Check if events are close in time (within 5 minutes) and space (within 0.1 degrees)
                time_diff = abs((event['time'] - existing['time']).total_seconds())
                lat_diff = abs(event['latitude'] - existing['latitude'])
                lon_diff = abs(event['longitude'] - existing['longitude'])
                
                if time_diff < 300 and lat_diff < 0.1 and lon_diff < 0.1:
                    is_duplicate = True
                    # Keep the one with higher quality/more complete data
                    if event.get('magnitude', 0) > existing.get('magnitude', 0):
                        unique_events.remove(existing)
                        unique_events.append(event)
                    break
            
            if not is_duplicate:
                unique_events.append(event)
        
        return unique_events
    
    def get_api_status(self) -> Dict[str, Dict]:
        """Get status of all configured APIs."""
        status = {}
        
        for source, config in self.apis.items():
            rate_limit_info = {}
            if config.rate_limit:
                recent_calls = len([
                    t for t in self.rate_limits.get(source, [])
                    if time.time() - t < 3600
                ])
                rate_limit_info = {
                    'calls_last_hour': recent_calls,
                    'limit_per_hour': config.rate_limit,
                    'remaining': max(0, config.rate_limit - recent_calls)
                }
            
            status[source.value] = {
                'url': config.url,
                'has_api_key': bool(config.key),
                'cache_ttl': config.cache_ttl,
                'rate_limit': rate_limit_info,
                'cached_entries': len([k for k in self.cache.keys() if k.startswith(source.value)])
            }
        
        return status

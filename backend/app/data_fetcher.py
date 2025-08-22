"""
Data fetching utilities for multiple seismic data sources.
Supports IRIS, PDS, and URL-based data retrieval.
"""

import os
import requests
import tempfile
import time
from typing import List, Optional, Dict, Any
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import json
import urllib.parse
from datetime import datetime, timedelta


class DataFetcher:
    """Base class for data fetching operations."""
    
    def __init__(self, cache_dir: str = "cache", timeout: int = 30):
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, identifier: str) -> str:
        """Generate cache file path for an identifier."""
        safe_id = "".join(c for c in identifier if c.isalnum() or c in "._-")
        return os.path.join(self.cache_dir, f"{safe_id}.mseed")
    
    def _is_cache_valid(self, cache_path: str, max_age_hours: int = 24) -> bool:
        """Check if cached file is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_age = time.time() - os.path.getmtime(cache_path)
        return file_age < (max_age_hours * 3600)


def fetch_from_iris(network: str, station: str, channel: str, 
                   starttime: UTCDateTime, endtime: UTCDateTime,
                   location: str = "*", client_name: str = "IRIS") -> str:
    """
    Fetch seismic data from IRIS FDSN web services.
    
    Args:
        network: Network code (e.g., 'IU')
        station: Station code (e.g., 'ANMO')
        channel: Channel code (e.g., 'BHZ')
        starttime: Start time
        endtime: End time
        location: Location code
        client_name: FDSN client name
        
    Returns:
        Path to downloaded file
        
    Raises:
        Exception: If data cannot be fetched
    """
    try:
        client = Client(client_name)
        
        # Create output filename
        filename = f"{network}_{station}_{location}_{channel}_{starttime.strftime('%Y%m%d_%H%M%S')}.mseed"
        output_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Fetch waveform data
        st = client.get_waveforms(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime
        )
        
        if len(st) == 0:
            raise ValueError("No data returned from IRIS")
        
        # Save to file
        st.write(output_path, format="MSEED")
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Failed to fetch from IRIS: {str(e)}")


def fetch_from_url(url: str, output_dir: Optional[str] = None) -> str:
    """
    Fetch seismic data from a URL.
    
    Args:
        url: URL to seismic data file
        output_dir: Output directory (defaults to temp)
        
    Returns:
        Path to downloaded file
        
    Raises:
        Exception: If download fails
    """
    try:
        if output_dir is None:
            output_dir = tempfile.gettempdir()
        
        # Extract filename from URL
        parsed_url = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = f"downloaded_data_{int(time.time())}.mseed"
        
        output_path = os.path.join(output_dir, filename)
        
        # Download file
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
        
    except Exception as e:
        raise Exception(f"Failed to fetch from URL: {str(e)}")


def fetch_from_pds_search(mission: str = "insight", instrument: str = "SEIS",
                         product_class: Optional[str] = None, limit: int = 10,
                         download_first_match: bool = True) -> List[str]:
    """
    Fetch data from NASA PDS using search API.
    
    Args:
        mission: Mission name (e.g., 'insight')
        instrument: Instrument name (e.g., 'SEIS', 'APSS')
        product_class: Product class filter
        limit: Maximum number of results
        download_first_match: Download first matching product
        
    Returns:
        List of downloaded file paths
        
    Raises:
        Exception: If search or download fails
    """
    try:
        # Mock PDS search implementation for demonstration
        # In a real implementation, this would query the actual PDS API
        
        downloaded_files = []
        
        if mission.lower() == "insight" and instrument.upper() == "SEIS":
            # Generate mock InSight SEIS data
            mock_data = _generate_mock_insight_seis()
            filename = f"insight_seis_mock_{int(time.time())}.mseed"
            output_path = os.path.join(tempfile.gettempdir(), filename)
            
            mock_data.write(output_path, format="MSEED")
            downloaded_files.append(output_path)
            
        elif mission.lower() == "insight" and instrument.upper() == "APSS":
            # Generate mock InSight pressure data
            mock_data = _generate_mock_insight_pressure()
            filename = f"insight_apss_mock_{int(time.time())}.mseed"
            output_path = os.path.join(tempfile.gettempdir(), filename)
            
            mock_data.write(output_path, format="MSEED")
            downloaded_files.append(output_path)
        
        return downloaded_files
        
    except Exception as e:
        raise Exception(f"Failed to fetch from PDS: {str(e)}")


def _generate_mock_insight_seis() -> 'Stream':
    """Generate mock InSight SEIS data for demonstration."""
    from obspy import Stream, Trace
    import numpy as np
    
    # Generate synthetic seismic data with some events
    duration = 3600  # 1 hour
    sampling_rate = 20.0  # 20 Hz
    npts = int(duration * sampling_rate)
    
    # Base noise
    data = np.random.normal(0, 1e-9, npts)
    
    # Add some synthetic events
    event_times = [900, 1800, 2700]  # Events at 15, 30, 45 minutes
    for event_time in event_times:
        event_start = int(event_time * sampling_rate)
        event_duration = int(60 * sampling_rate)  # 60 second event
        
        if event_start + event_duration < npts:
            # Create synthetic seismic event
            t = np.linspace(0, 60, event_duration)
            amplitude = 5e-8 * np.exp(-t/20) * np.sin(2 * np.pi * 2 * t)
            data[event_start:event_start + event_duration] += amplitude
    
    # Create trace
    trace = Trace(data=data)
    trace.stats.sampling_rate = sampling_rate
    trace.stats.starttime = UTCDateTime() - 3600  # 1 hour ago
    trace.stats.station = "ELYSE"
    trace.stats.channel = "BHZ"
    trace.stats.network = "XB"
    trace.stats.location = "02"
    
    return Stream([trace])


def _generate_mock_insight_pressure() -> 'Stream':
    """Generate mock InSight pressure data for demonstration."""
    from obspy import Stream, Trace
    import numpy as np
    
    # Generate synthetic pressure data with wind gusts
    duration = 3600  # 1 hour
    sampling_rate = 20.0  # 20 Hz
    npts = int(duration * sampling_rate)
    
    # Base pressure variations
    t = np.linspace(0, duration, npts)
    data = 610 + 5 * np.sin(2 * np.pi * t / 3600)  # Daily pressure cycle
    
    # Add wind gusts
    gust_times = [600, 1200, 2400, 3000]  # Random gust times
    for gust_time in gust_times:
        gust_start = int(gust_time * sampling_rate)
        gust_duration = int(120 * sampling_rate)  # 2 minute gust
        
        if gust_start + gust_duration < npts:
            # Create pressure gust
            gust_t = np.linspace(0, 120, gust_duration)
            gust_amplitude = 20 * np.exp(-gust_t/60) * (1 + 0.5 * np.random.normal(0, 1, gust_duration))
            data[gust_start:gust_start + gust_duration] += gust_amplitude
    
    # Add noise
    data += np.random.normal(0, 0.5, npts)
    
    # Create trace
    trace = Trace(data=data)
    trace.stats.sampling_rate = sampling_rate
    trace.stats.starttime = UTCDateTime() - 3600  # 1 hour ago
    trace.stats.station = "ELYSE"
    trace.stats.channel = "BDO"  # Pressure channel
    trace.stats.network = "XB"
    trace.stats.location = "02"
    
    return Stream([trace])


def search_available_data(network: str = "*", station: str = "*",
                         channel: str = "*", starttime: Optional[UTCDateTime] = None,
                         endtime: Optional[UTCDateTime] = None,
                         client_name: str = "IRIS") -> List[Dict[str, Any]]:
    """
    Search for available seismic data without downloading.

    Args:
        network: Network code pattern
        station: Station code pattern
        channel: Channel code pattern
        starttime: Start time for search
        endtime: End time for search
        client_name: FDSN client name

    Returns:
        List of available data descriptions
    """
    try:
        client = Client(client_name)

        if starttime is None:
            starttime = UTCDateTime() - 86400  # Last 24 hours
        if endtime is None:
            endtime = UTCDateTime()

        # Get station inventory
        inventory = client.get_stations(
            network=network,
            station=station,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            level="channel"
        )

        available_data = []
        for net in inventory:
            for sta in net:
                for cha in sta:
                    available_data.append({
                        'network': net.code,
                        'station': sta.code,
                        'channel': cha.code,
                        'location': cha.location_code or "",
                        'latitude': cha.latitude,
                        'longitude': cha.longitude,
                        'elevation': cha.elevation,
                        'start_date': str(cha.start_date),
                        'end_date': str(cha.end_date) if cha.end_date else "ongoing",
                        'sampling_rate': cha.sample_rate
                    })

        return available_data

    except Exception as e:
        # Return empty list on error, don't break the application
        print(f"Warning: Could not search available data: {e}")
        return []


def search_iris_events(starttime: UTCDateTime, endtime: UTCDateTime,
                      minmagnitude: float = 4.0, maxmagnitude: float = 10.0,
                      mindepth: float = 0, maxdepth: float = 1000,
                      minlatitude: float = -90, maxlatitude: float = 90,
                      minlongitude: float = -180, maxlongitude: float = 180,
                      client_name: str = "IRIS") -> List[Dict[str, Any]]:
    """
    Search for earthquake events using IRIS event service.

    Args:
        starttime: Start time for search
        endtime: End time for search
        minmagnitude: Minimum magnitude
        maxmagnitude: Maximum magnitude
        mindepth: Minimum depth (km)
        maxdepth: Maximum depth (km)
        minlatitude: Minimum latitude
        maxlatitude: Maximum latitude
        minlongitude: Minimum longitude
        maxlongitude: Maximum longitude
        client_name: FDSN client name

    Returns:
        List of earthquake events
    """
    try:
        client = Client(client_name)

        # Get event catalog
        catalog = client.get_events(
            starttime=starttime,
            endtime=endtime,
            minmagnitude=minmagnitude,
            maxmagnitude=maxmagnitude,
            mindepth=mindepth,
            maxdepth=maxdepth,
            minlatitude=minlatitude,
            maxlatitude=maxlatitude,
            minlongitude=minlongitude,
            maxlongitude=maxlongitude
        )

        events = []
        for event in catalog:
            # Extract event information
            origin = event.preferred_origin() or event.origins[0]
            magnitude = event.preferred_magnitude() or event.magnitudes[0]

            event_info = {
                'id': str(event.resource_id),
                'time': str(origin.time),
                'latitude': float(origin.latitude),
                'longitude': float(origin.longitude),
                'depth': float(origin.depth) / 1000 if origin.depth else None,  # Convert to km
                'magnitude': float(magnitude.mag) if magnitude else None,
                'magnitude_type': str(magnitude.magnitude_type) if magnitude else None,
                'agency': str(origin.creation_info.agency_id) if origin.creation_info else None,
                'location': event.event_descriptions[0].text if event.event_descriptions else None,
                'event_type': str(event.event_type) if event.event_type else None
            }

            # Add uncertainty information if available
            if origin.latitude_errors:
                event_info['latitude_uncertainty'] = float(origin.latitude_errors.uncertainty)
            if origin.longitude_errors:
                event_info['longitude_uncertainty'] = float(origin.longitude_errors.uncertainty)
            if origin.depth_errors:
                event_info['depth_uncertainty'] = float(origin.depth_errors.uncertainty) / 1000

            events.append(event_info)

        return events

    except Exception as e:
        print(f"Warning: Could not search IRIS events: {e}")
        return []


def get_station_availability(network: str, station: str,
                           starttime: UTCDateTime, endtime: UTCDateTime,
                           client_name: str = "IRIS") -> Dict[str, Any]:
    """
    Get data availability for a specific station.

    Args:
        network: Network code
        station: Station code
        starttime: Start time
        endtime: End time
        client_name: FDSN client name

    Returns:
        Station availability information
    """
    try:
        client = Client(client_name)

        # Get availability information
        availability = client.get_availability(
            network=network,
            station=station,
            starttime=starttime,
            endtime=endtime
        )

        # Parse availability response
        availability_info = {
            'network': network,
            'station': station,
            'channels': [],
            'total_channels': 0,
            'time_range': {
                'start': str(starttime),
                'end': str(endtime)
            }
        }

        for net in availability:
            for sta in net:
                for cha in sta:
                    channel_info = {
                        'channel': cha.code,
                        'location': cha.location_code or "",
                        'start_date': str(cha.start_date),
                        'end_date': str(cha.end_date) if cha.end_date else "ongoing",
                        'sample_rate': cha.sample_rate,
                        'restricted': cha.restricted_status == "open"
                    }
                    availability_info['channels'].append(channel_info)

        availability_info['total_channels'] = len(availability_info['channels'])

        return availability_info

    except Exception as e:
        print(f"Warning: Could not get station availability: {e}")
        return {
            'network': network,
            'station': station,
            'error': str(e)
        }


def find_nearby_stations(latitude: float, longitude: float,
                        max_radius: float = 5.0, max_stations: int = 10,
                        client_name: str = "IRIS") -> List[Dict[str, Any]]:
    """
    Find seismic stations near a given location.

    Args:
        latitude: Target latitude
        longitude: Target longitude
        max_radius: Maximum radius in degrees
        max_stations: Maximum number of stations to return
        client_name: FDSN client name

    Returns:
        List of nearby stations
    """
    try:
        client = Client(client_name)

        # Search for stations in the area
        inventory = client.get_stations(
            latitude=latitude,
            longitude=longitude,
            maxradius=max_radius,
            level="station"
        )

        stations = []
        for net in inventory:
            for sta in net:
                # Calculate distance (approximate)
                lat_diff = abs(sta.latitude - latitude)
                lon_diff = abs(sta.longitude - longitude)
                distance = (lat_diff ** 2 + lon_diff ** 2) ** 0.5 * 111  # rough km conversion

                station_info = {
                    'network': net.code,
                    'station': sta.code,
                    'latitude': sta.latitude,
                    'longitude': sta.longitude,
                    'elevation': sta.elevation,
                    'distance_km': distance,
                    'site_name': sta.site.name if sta.site else None,
                    'start_date': str(sta.start_date),
                    'end_date': str(sta.end_date) if sta.end_date else "ongoing",
                    'channel_count': len(sta.channels) if sta.channels else 0
                }
                stations.append(station_info)

        # Sort by distance and limit results
        stations.sort(key=lambda x: x['distance_km'])
        return stations[:max_stations]

    except Exception as e:
        print(f"Warning: Could not find nearby stations: {e}")
        return []

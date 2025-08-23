"""
Flask API with automatic noise masking support.
Endpoints:
 - POST /detect : main endpoint (file upload, iris, url, pds_search)
 Accepts optional form field 'pressure' (file) to mask wind noise.
 Optional 'planet' param tunes detection presets.
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from obspy import UTCDateTime
from datetime import datetime
from .io_utils import load_waveform, trace_to_numpy
from .preprocess import preprocess_trace
from .detector_manager import DetectorManager
from .data_fetcher import (
    fetch_from_iris, fetch_from_url, fetch_from_pds_search,
    search_available_data, search_iris_events, get_station_availability, find_nearby_stations
)
# Set up logging first
logger = logging.getLogger(__name__)

# Enhanced features with graceful fallbacks
try:
    from .data_integration_hub import DataIntegrationHub, DataSource
    DATA_HUB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data integration hub not available: {e}")
    DATA_HUB_AVAILABLE = False

try:
    from .external_ml_apis import ExternalMLAPIs
    ML_APIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"External ML APIs not available: {e}")
    ML_APIS_AVAILABLE = False

try:
    from .websocket_streaming import websocket_streamer, start_websocket_server_thread
    WEBSOCKET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"WebSocket streaming not available: {e}")
    WEBSOCKET_AVAILABLE = False

try:
    from .satellite_correlation import SatelliteCorrelationEngine
    SATELLITE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Satellite correlation not available: {e}")
    SATELLITE_AVAILABLE = False
from .noise_masker import mask_seismic_with_pressure
import numpy as np
import tempfile
import traceback
import asyncio


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for frontend integration
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000", 
                   "http://localhost:5173", "http://127.0.0.1:5173",
                   "http://localhost:8080", "http://127.0.0.1:8080"])

# Initialize detector manager
dm = DetectorManager(model_root='models')

# Initialize enhanced features if available
data_hub = None
ml_apis = None
satellite_engine = None

if DATA_HUB_AVAILABLE:
    try:
        data_hub = DataIntegrationHub(cache_dir='cache')
        logger.info("Data integration hub initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize data hub: {e}")

if ML_APIS_AVAILABLE:
    try:
        ml_apis = ExternalMLAPIs()
        logger.info("External ML APIs initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize ML APIs: {e}")

if SATELLITE_AVAILABLE:
    try:
        satellite_engine = SatelliteCorrelationEngine()
        logger.info("Satellite correlation engine initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize satellite engine: {e}")

# WebSocket server will be started conditionally
websocket_server_started = False


def start_websocket_if_needed():
    """Start WebSocket server if not already started."""
    global websocket_server_started
    if not websocket_server_started and WEBSOCKET_AVAILABLE:
        try:
            start_websocket_server_thread()
            websocket_server_started = True
            logger.info("WebSocket streaming server started")
        except Exception as e:
            logger.warning(f"Could not start WebSocket server: {e}")
    elif not WEBSOCKET_AVAILABLE:
        logger.warning("WebSocket streaming not available - missing dependencies")

# Create necessary directories
UPLOAD_DIR = 'uploads'
OUTPUT_DIR = 'outputs'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'SeismoGuard Backend',
        'version': '1.0.0',
        'available_algorithms': dm.get_available_algorithms()
    })


@app.route('/detect', methods=['POST'])
def detect_endpoint():
    """
    Main detection endpoint supporting multiple data sources.
    
    Form parameters:
    - seismic: File upload (seismic data)
    - pressure: File upload (optional pressure data for noise masking)
    - source: Data source ('iris', 'url', 'pds_search')
    - planet: Planet configuration ('earth', 'mars', 'moon')
    
    IRIS parameters:
    - network, station, channel, starttime, endtime
    
    URL parameters:
    - url: URL to seismic data file
    
    PDS parameters:
    - mission: Mission name (default: 'insight')
    - instrument: Instrument name (default: 'SEIS')
    
    Returns:
    JSON response with detected events and diagnostics
    """
    try:
        # Get planet configuration
        planet = (request.form.get('planet') or request.args.get('planet') or 'earth').lower()
        logger.info(f"Processing request for planet: {planet}")
        
        # Load seismic data
        seismic_trace = None
        data_source_info = {}
        
        # 1) Handle file upload
        if 'seismic' in request.files:
            logger.info("Processing uploaded seismic file")
            f = request.files['seismic']
            if f.filename == '':
                return jsonify({'error': 'No seismic file selected'}), 400
            
            sfile = os.path.join(UPLOAD_DIR, f.filename)
            f.save(sfile)
            
            try:
                st = load_waveform(sfile)
                seismic_trace = st[0]
                data_source_info = {
                    'source': 'file_upload',
                    'filename': f.filename,
                    'file_size': os.path.getsize(sfile)
                }
            except Exception as e:
                return jsonify({'error': f'Failed to load seismic file: {str(e)}'}), 400
        
        # 2) Handle other data sources
        else:
            source = (request.form.get('source') or request.args.get('source') or '').lower()
            
            if source == 'iris':
                logger.info("Fetching data from IRIS")
                network = request.form.get('network') or request.args.get('network')
                station = request.form.get('station') or request.args.get('station')
                channel = request.form.get('channel') or request.args.get('channel')
                starttime = request.form.get('starttime') or request.args.get('starttime')
                endtime = request.form.get('endtime') or request.args.get('endtime')
                
                if not all([network, station, channel, starttime, endtime]):
                    return jsonify({'error': 'Missing IRIS parameters: network, station, channel, starttime, endtime'}), 400
                
                try:
                    t1 = UTCDateTime(starttime)
                    t2 = UTCDateTime(endtime)
                    path = fetch_from_iris(network, station, channel, t1, t2)
                    st = load_waveform(path)
                    seismic_trace = st[0]
                    data_source_info = {
                        'source': 'iris',
                        'network': network,
                        'station': station,
                        'channel': channel,
                        'starttime': str(t1),
                        'endtime': str(t2)
                    }
                except Exception as e:
                    return jsonify({'error': f'Failed to fetch from IRIS: {str(e)}'}), 400
            
            elif source == 'url':
                logger.info("Fetching data from URL")
                url = request.form.get('url') or request.args.get('url')
                if not url:
                    return jsonify({'error': 'Missing url parameter'}), 400
                
                try:
                    path = fetch_from_url(url)
                    st = load_waveform(path)
                    seismic_trace = st[0]
                    data_source_info = {
                        'source': 'url',
                        'url': url
                    }
                except Exception as e:
                    return jsonify({'error': f'Failed to fetch from URL: {str(e)}'}), 400
            
            elif source == 'pds_search':
                logger.info("Fetching data from PDS")
                mission = (request.form.get('mission') or request.args.get('mission') or 'insight').lower()
                instrument = (request.form.get('instrument') or request.args.get('instrument') or 'SEIS')
                
                try:
                    downloaded = fetch_from_pds_search(mission=mission, instrument=instrument, limit=20, download_first_match=True)
                    if not downloaded:
                        return jsonify({'error': 'No PDS products found'}), 404
                    
                    path = downloaded[0]
                    st = load_waveform(path)
                    seismic_trace = st[0]
                    data_source_info = {
                        'source': 'pds_search',
                        'mission': mission,
                        'instrument': instrument,
                        'files_downloaded': len(downloaded)
                    }
                except Exception as e:
                    return jsonify({'error': f'Failed to fetch from PDS: {str(e)}'}), 400
            
            else:
                return jsonify({'error': 'No seismic data source provided. Use file upload or specify source parameter.'}), 400
        
        if seismic_trace is None:
            return jsonify({'error': 'Failed to load seismic data'}), 400
        
        logger.info(f"Loaded seismic trace: {seismic_trace.stats.station}.{seismic_trace.stats.channel}")
        
        # 3) Optionally load pressure channel for noise masking
        pressure_trace = None
        pressure_info = {}
        
        if 'pressure' in request.files:
            logger.info("Processing uploaded pressure file")
            pf = request.files['pressure']
            if pf.filename != '':
                ppath = os.path.join(UPLOAD_DIR, pf.filename)
                pf.save(ppath)
                
                try:
                    pst = load_waveform(ppath)
                    pressure_trace = pst[0]
                    pressure_info = {
                        'source': 'file_upload',
                        'filename': pf.filename
                    }
                except Exception as e:
                    logger.warning(f"Failed to load pressure file: {e}")
        
        # Auto-fetch pressure for InSight PDS data
        elif data_source_info.get('source') == 'pds_search' and data_source_info.get('mission') == 'insight':
            logger.info("Attempting to fetch InSight pressure data")
            try:
                p_downloaded = fetch_from_pds_search(mission='insight', instrument='APSS', limit=5, download_first_match=True)
                if p_downloaded:
                    pst = load_waveform(p_downloaded[0])
                    pressure_trace = pst[0]
                    pressure_info = {
                        'source': 'pds_search_auto',
                        'mission': 'insight',
                        'instrument': 'APSS'
                    }
                    logger.info("Successfully loaded InSight pressure data")
            except Exception as e:
                logger.warning(f"Could not auto-fetch pressure data: {e}")
        
        # 4) Preprocess seismic data
        logger.info("Preprocessing seismic data")
        try:
            processed_trace = preprocess_trace(seismic_trace, planet=planet)
        except Exception as e:
            return jsonify({'error': f'Preprocessing failed: {str(e)}'}), 500
        
        # 5) Apply noise masking if pressure data is available
        masking_applied = False
        if pressure_trace is not None:
            logger.info("Applying pressure-based noise masking")
            try:
                # Align sampling rates
                s_sr = processed_trace.stats.sampling_rate
                if abs(pressure_trace.stats.sampling_rate - s_sr) > 1e-6:
                    pressure_trace.resample(s_sr)
                
                # Convert to numpy arrays
                s_arr, _ = trace_to_numpy(processed_trace, target_sampling_rate=None)
                p_arr, _ = trace_to_numpy(pressure_trace, target_sampling_rate=None)
                
                # Apply masking
                masked, mask = mask_seismic_with_pressure(
                    s_arr, p_arr, sr=s_sr, 
                    win_sec=1.0, std_thresh=3.0, dilate_sec=2.0, mode='zero'
                )
                
                # Handle NaN values
                masked = np.nan_to_num(masked, nan=0.0)
                processed_trace.data = masked.astype(processed_trace.data.dtype)
                
                masking_applied = True
                pressure_info['masking_applied'] = True
                pressure_info['masked_samples'] = int(np.sum(mask))
                pressure_info['mask_percentage'] = float(np.sum(mask) / len(mask) * 100)
                
                logger.info(f"Noise masking applied: {pressure_info['mask_percentage']:.1f}% of samples masked")
                
            except Exception as e:
                logger.warning(f"Noise masking failed: {e}")
                pressure_info['masking_error'] = str(e)
        
    # 6) Run seismic event detection
        logger.info("Running seismic event detection")
        try:
            events_df, diagnostics = dm.analyze_trace(processed_trace, planet=planet)
            events = events_df.to_dict(orient='records') if not events_df.empty else []
            
            # Convert timestamps to strings for JSON serialization
            for event in events:
                if 'time' in event:
                    event['time'] = str(event['time'])
            
            logger.info(f"Detection complete: {len(events)} events found")
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return jsonify({'error': f'Detection failed: {str(e)}'}), 500
        
        # 7) Prepare response (include compact processed timeseries for frontend plotting)
        # Downsample to avoid huge payloads
        try:
            series = None
            try:
                from .io_utils import trace_to_numpy
                arr, sr = trace_to_numpy(processed_trace, target_sampling_rate=None)
            except Exception:
                arr = processed_trace.data
                sr = float(processed_trace.stats.sampling_rate)

            n = len(arr)
            max_points = int(request.args.get('max_points', request.form.get('max_points', 5000)) or 5000)
            step = max(1, n // max_points)
            if step > 1:
                arr_ds = arr[::step]
            else:
                arr_ds = arr
            # Convert to float for JSON and clamp NaNs
            import numpy as _np
            arr_ds = _np.nan_to_num(_np.asarray(arr_ds, dtype=float)).tolist()
            series = {
                'starttime': str(processed_trace.stats.starttime),
                'sampling_rate': float(sr) / step,
                'samples': arr_ds,
                'original_points': n,
                'downsample_step': step
            }
        except Exception as e:
            logger.warning(f"Could not attach timeseries: {e}")
            series = None

        # 8) Prepare response
        response = {
            'events': events,
            'diagnostics': diagnostics,
            'data_source': data_source_info,
            'processing': {
                'planet': planet,
                'masking_applied': masking_applied,
                'pressure_data': pressure_info if pressure_trace else None
            },
            'timeseries': series
        }
        
        logger.info(f"Request completed successfully: {len(events)} events detected")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error in detect endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@app.route('/algorithms', methods=['GET'])
def get_algorithms():
    """Get available detection algorithms."""
    try:
        algorithms = dm.get_available_algorithms()
        algorithm_info = {}
        
        for alg in algorithms:
            algorithm_info[alg] = dm.get_algorithm_info(alg)
        
        return jsonify({
            'algorithms': algorithms,
            'details': algorithm_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/planet-presets', methods=['GET'])
def get_planet_presets():
    """Get available planet processing presets."""
    from .preprocess import PLANET_PRESETS

    return jsonify({
        'presets': PLANET_PRESETS
    })


@app.route('/earthquakes/recent', methods=['GET'])
def get_recent_earthquakes():
    """
    Get recent earthquakes from multiple sources.

    Query parameters:
    - magnitude: Minimum magnitude (default: 4.0)
    - hours: Hours back to search (default: 24)
    - sources: Comma-separated list of sources (usgs,emsc)
    """
    if not DATA_HUB_AVAILABLE or data_hub is None:
        return jsonify({
            'error': 'Data integration hub not available',
            'message': 'Install required dependencies: pip install aiohttp'
        }), 503

    try:
        # Parse parameters
        magnitude = float(request.args.get('magnitude', 4.0))
        hours = int(request.args.get('hours', 24))
        sources_param = request.args.get('sources', 'usgs,emsc')

        # Map source names to DataSource enum
        source_map = {
            'usgs': DataSource.USGS_REALTIME,
            'emsc': DataSource.EMSC,
            'iris': DataSource.IRIS_EVENTS
        }

        sources = []
        for source_name in sources_param.split(','):
            source_name = source_name.strip().lower()
            if source_name in source_map:
                sources.append(source_map[source_name])

        if not sources:
            sources = [DataSource.USGS_REALTIME, DataSource.EMSC]

        # Fetch earthquake data
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                earthquakes = loop.run_until_complete(
                    data_hub.get_recent_earthquakes(magnitude, hours, sources)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to fetch earthquake data: {e}")
            return jsonify({'error': f'Failed to fetch earthquake data: {str(e)}'}), 500

        # Convert datetime objects to strings for JSON serialization
        for eq in earthquakes:
            if 'time' in eq and hasattr(eq['time'], 'isoformat'):
                eq['time'] = eq['time'].isoformat()

        return jsonify({
            'earthquakes': earthquakes,
            'count': len(earthquakes),
            'parameters': {
                'magnitude': magnitude,
                'hours': hours,
                'sources': [s.value for s in sources]
            }
        })

    except Exception as e:
        logger.error(f"Error fetching recent earthquakes: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/data-sources/status', methods=['GET'])
def get_data_sources_status():
    """Get status of all configured data sources."""
    try:
        status = data_hub.get_api_status()
        return jsonify({
            'status': status,
            'timestamp': str(UTCDateTime())
        })

    except Exception as e:
        logger.error(f"Error getting data sources status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/compare/events', methods=['POST'])
def compare_detected_with_catalogs():
    """
    Compare detected events with earthquake catalogs.

    JSON body:
    {
        "events": [{"time": "2023-01-01T12:00:00", "latitude": 34.0, "longitude": -118.0}],
        "time_window": 300,  // seconds
        "distance_threshold": 100  // km
    }
    """
    try:
        data = request.get_json()
        if not data or 'events' not in data:
            return jsonify({'error': 'Missing events data'}), 400

        detected_events = data['events']
        time_window = data.get('time_window', 300)  # 5 minutes
        distance_threshold = data.get('distance_threshold', 100)  # 100 km

        # Get recent earthquakes for comparison
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            catalog_events = loop.run_until_complete(
                data_hub.get_recent_earthquakes(magnitude=2.0, hours=24)
            )
        finally:
            loop.close()

        # Compare events
        matches = []
        for detected in detected_events:
            detected_time = UTCDateTime(detected['time'])
            detected_lat = detected.get('latitude')
            detected_lon = detected.get('longitude')

            for catalog in catalog_events:
                catalog_time = UTCDateTime(catalog['time'])

                # Time comparison
                time_diff = abs((detected_time - catalog_time).total_seconds())
                if time_diff > time_window:
                    continue

                # Location comparison (if available)
                if detected_lat is not None and detected_lon is not None:
                    # Simple distance calculation (approximate)
                    lat_diff = abs(detected_lat - catalog['latitude'])
                    lon_diff = abs(detected_lon - catalog['longitude'])
                    distance_km = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111  # rough km conversion

                    if distance_km > distance_threshold:
                        continue

                # Found a match
                match = {
                    'detected_event': detected,
                    'catalog_event': {
                        'id': catalog['id'],
                        'time': catalog['time'].isoformat() if hasattr(catalog['time'], 'isoformat') else str(catalog['time']),
                        'magnitude': catalog['magnitude'],
                        'location': catalog['location'],
                        'source': catalog['source'],
                        'latitude': catalog['latitude'],
                        'longitude': catalog['longitude'],
                        'depth': catalog['depth']
                    },
                    'time_difference_seconds': time_diff,
                    'distance_km': distance_km if detected_lat is not None else None
                }
                matches.append(match)
                break  # Only match with first found event

        return jsonify({
            'matches': matches,
            'detected_count': len(detected_events),
            'matched_count': len(matches),
            'parameters': {
                'time_window_seconds': time_window,
                'distance_threshold_km': distance_threshold
            }
        })

    except Exception as e:
        logger.error(f"Error comparing events: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/iris/stations/search', methods=['GET'])
def search_iris_stations():
    """
    Search for available IRIS stations.

    Query parameters:
    - network: Network code pattern (default: *)
    - station: Station code pattern (default: *)
    - channel: Channel code pattern (default: *)
    - starttime: Start time (ISO format)
    - endtime: End time (ISO format)
    """
    try:
        network = request.args.get('network', '*')
        station = request.args.get('station', '*')
        channel = request.args.get('channel', '*')

        starttime = None
        endtime = None
        if request.args.get('starttime'):
            starttime = UTCDateTime(request.args.get('starttime'))
        if request.args.get('endtime'):
            endtime = UTCDateTime(request.args.get('endtime'))

        stations = search_available_data(
            network=network,
            station=station,
            channel=channel,
            starttime=starttime,
            endtime=endtime
        )

        return jsonify({
            'stations': stations,
            'count': len(stations),
            'parameters': {
                'network': network,
                'station': station,
                'channel': channel,
                'starttime': str(starttime) if starttime else None,
                'endtime': str(endtime) if endtime else None
            }
        })

    except Exception as e:
        logger.error(f"Error searching IRIS stations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/iris/events/search', methods=['GET'])
def search_iris_earthquake_events():
    """
    Search for earthquake events using IRIS event service.

    Query parameters:
    - starttime: Start time (ISO format, required)
    - endtime: End time (ISO format, required)
    - minmagnitude: Minimum magnitude (default: 4.0)
    - maxmagnitude: Maximum magnitude (default: 10.0)
    - mindepth: Minimum depth in km (default: 0)
    - maxdepth: Maximum depth in km (default: 1000)
    - minlatitude: Minimum latitude (default: -90)
    - maxlatitude: Maximum latitude (default: 90)
    - minlongitude: Minimum longitude (default: -180)
    - maxlongitude: Maximum longitude (default: 180)
    """
    try:
        # Required parameters
        if not request.args.get('starttime') or not request.args.get('endtime'):
            return jsonify({'error': 'starttime and endtime are required'}), 400

        starttime = UTCDateTime(request.args.get('starttime'))
        endtime = UTCDateTime(request.args.get('endtime'))

        # Optional parameters
        minmagnitude = float(request.args.get('minmagnitude', 4.0))
        maxmagnitude = float(request.args.get('maxmagnitude', 10.0))
        mindepth = float(request.args.get('mindepth', 0))
        maxdepth = float(request.args.get('maxdepth', 1000))
        minlatitude = float(request.args.get('minlatitude', -90))
        maxlatitude = float(request.args.get('maxlatitude', 90))
        minlongitude = float(request.args.get('minlongitude', -180))
        maxlongitude = float(request.args.get('maxlongitude', 180))

        events = search_iris_events(
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

        return jsonify({
            'events': events,
            'count': len(events),
            'parameters': {
                'starttime': str(starttime),
                'endtime': str(endtime),
                'magnitude_range': [minmagnitude, maxmagnitude],
                'depth_range': [mindepth, maxdepth],
                'latitude_range': [minlatitude, maxlatitude],
                'longitude_range': [minlongitude, maxlongitude]
            }
        })

    except Exception as e:
        logger.error(f"Error searching IRIS events: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/iris/stations/nearby', methods=['GET'])
def find_nearby_iris_stations():
    """
    Find IRIS stations near a location.

    Query parameters:
    - latitude: Target latitude (required)
    - longitude: Target longitude (required)
    - max_radius: Maximum radius in degrees (default: 5.0)
    - max_stations: Maximum number of stations (default: 10)
    """
    try:
        if not request.args.get('latitude') or not request.args.get('longitude'):
            return jsonify({'error': 'latitude and longitude are required'}), 400

        latitude = float(request.args.get('latitude'))
        longitude = float(request.args.get('longitude'))
        max_radius = float(request.args.get('max_radius', 5.0))
        max_stations = int(request.args.get('max_stations', 10))

        stations = find_nearby_stations(
            latitude=latitude,
            longitude=longitude,
            max_radius=max_radius,
            max_stations=max_stations
        )

        return jsonify({
            'stations': stations,
            'count': len(stations),
            'parameters': {
                'latitude': latitude,
                'longitude': longitude,
                'max_radius_degrees': max_radius,
                'max_stations': max_stations
            }
        })

    except Exception as e:
        logger.error(f"Error finding nearby stations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/iris/stations/<network>/<station>/availability', methods=['GET'])
def get_iris_station_availability(network: str, station: str):
    """
    Get data availability for a specific IRIS station.

    Query parameters:
    - starttime: Start time (ISO format, required)
    - endtime: End time (ISO format, required)
    """
    try:
        if not request.args.get('starttime') or not request.args.get('endtime'):
            return jsonify({'error': 'starttime and endtime are required'}), 400

        starttime = UTCDateTime(request.args.get('starttime'))
        endtime = UTCDateTime(request.args.get('endtime'))

        availability = get_station_availability(
            network=network,
            station=station,
            starttime=starttime,
            endtime=endtime
        )

        return jsonify(availability)

    except Exception as e:
        logger.error(f"Error getting station availability: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/ml/classify', methods=['POST'])
def classify_seismic_with_ml():
    """
    Classify seismic data using external ML APIs.

    JSON body:
    {
        "data": [array of seismic values],
        "sampling_rate": 20.0,
        "model": "facebook/wav2vec2-base",  // optional
        "provider": "huggingface"  // optional
    }
    """
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'Missing seismic data'}), 400

        seismic_data = np.array(data['data'])
        sampling_rate = data.get('sampling_rate', 20.0)
        model = data.get('model', 'facebook/wav2vec2-base')
        provider = data.get('provider', 'huggingface')

        if len(seismic_data) == 0:
            return jsonify({'error': 'Empty seismic data'}), 400

        # Classify using specified provider
        if provider == 'huggingface':
            result = ml_apis.classify_with_huggingface_sync(seismic_data, sampling_rate, model)
        else:
            return jsonify({'error': f'Unsupported provider: {provider}'}), 400

        return jsonify({
            'classification': result,
            'input_info': {
                'data_length': len(seismic_data),
                'sampling_rate': sampling_rate,
                'duration_seconds': len(seismic_data) / sampling_rate
            }
        })

    except Exception as e:
        logger.error(f"Error in ML classification: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/ml/features', methods=['POST'])
def extract_ml_features():
    """
    Extract advanced features from seismic data using ML.

    JSON body:
    {
        "data": [array of seismic values],
        "sampling_rate": 20.0
    }
    """
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'Missing seismic data'}), 400

        seismic_data = np.array(data['data'])
        sampling_rate = data.get('sampling_rate', 20.0)

        if len(seismic_data) == 0:
            return jsonify({'error': 'Empty seismic data'}), 400

        # Extract features
        features = ml_apis.extract_features_with_ml(seismic_data, sampling_rate)

        return jsonify({
            'features': features,
            'input_info': {
                'data_length': len(seismic_data),
                'sampling_rate': sampling_rate,
                'duration_seconds': len(seismic_data) / sampling_rate
            }
        })

    except Exception as e:
        logger.error(f"Error extracting ML features: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/ml/ensemble', methods=['POST'])
def ensemble_ml_analysis():
    """
    Analyze seismic data with multiple ML models for ensemble results.

    JSON body:
    {
        "data": [array of seismic values],
        "sampling_rate": 20.0
    }
    """
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'Missing seismic data'}), 400

        seismic_data = np.array(data['data'])
        sampling_rate = data.get('sampling_rate', 20.0)

        if len(seismic_data) == 0:
            return jsonify({'error': 'Empty seismic data'}), 400

        # Run ensemble analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                ml_apis.analyze_with_multiple_models(seismic_data, sampling_rate)
            )
        finally:
            loop.close()

        return jsonify({
            'ensemble_analysis': result,
            'input_info': {
                'data_length': len(seismic_data),
                'sampling_rate': sampling_rate,
                'duration_seconds': len(seismic_data) / sampling_rate
            }
        })

    except Exception as e:
        logger.error(f"Error in ensemble ML analysis: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/ml/status', methods=['GET'])
def get_ml_apis_status():
    """Get status of all configured ML APIs."""
    try:
        status = ml_apis.get_api_status()
        return jsonify({
            'ml_apis': status,
            'timestamp': str(UTCDateTime())
        })

    except Exception as e:
        logger.error(f"Error getting ML APIs status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stream/status', methods=['GET'])
def get_websocket_status():
    """Get WebSocket streaming server status."""
    try:
        # Start WebSocket server if needed
        start_websocket_if_needed()

        status = websocket_streamer.get_status()
        return jsonify({
            'websocket_server': status,
            'endpoint': f"ws://127.0.0.1:{status['port']}",
            'available_streams': [
                'seismic_data',
                'earthquake_alerts',
                'detection_events',
                'system_status'
            ]
        })

    except Exception as e:
        logger.error(f"Error getting WebSocket status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stream/broadcast', methods=['POST'])
def broadcast_detection_event():
    """
    Broadcast a detection event to WebSocket subscribers.

    JSON body:
    {
        "event_type": "seismic_detection",
        "data": {...}
    }
    """
    try:
        # Start WebSocket server if needed
        start_websocket_if_needed()

        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing event data'}), 400

        # Broadcast to WebSocket subscribers
        websocket_streamer.broadcast_detection_event(data)

        return jsonify({
            'message': 'Event broadcasted',
            'subscribers': len(websocket_streamer.clients),
            'timestamp': str(UTCDateTime())
        })

    except Exception as e:
        logger.error(f"Error broadcasting event: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/satellite/correlate', methods=['POST'])
def correlate_seismic_with_satellite():
    """
    Correlate seismic event with satellite imagery and environmental data.

    JSON body:
    {
        "seismic_event": {
            "time": "2023-01-01T12:00:00",
            "latitude": 34.0,
            "longitude": -118.0,
            "magnitude": 5.5
        }
    }
    """
    try:
        data = request.get_json()
        if not data or 'seismic_event' not in data:
            return jsonify({'error': 'Missing seismic event data'}), 400

        seismic_event = data['seismic_event']

        # Validate required fields
        required_fields = ['time', 'latitude', 'longitude']
        for field in required_fields:
            if field not in seismic_event:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Run correlation analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            correlation_result = loop.run_until_complete(
                satellite_engine.correlate_seismic_satellite(seismic_event)
            )
        finally:
            loop.close()

        # Convert result to JSON-serializable format
        result_dict = {
            'seismic_event': correlation_result.seismic_event,
            'satellite_data': [
                {
                    'source': img.source,
                    'date': img.date.isoformat(),
                    'latitude': img.latitude,
                    'longitude': img.longitude,
                    'resolution': img.resolution,
                    'bands': img.bands,
                    'url': img.url,
                    'metadata': img.metadata
                }
                for img in correlation_result.satellite_data
            ],
            'correlation_score': correlation_result.correlation_score,
            'surface_changes': correlation_result.surface_changes,
            'environmental_factors': correlation_result.environmental_factors,
            'confidence': correlation_result.confidence,
            'analysis_timestamp': str(UTCDateTime())
        }

        return jsonify(result_dict)

    except Exception as e:
        logger.error(f"Error in satellite correlation: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/satellite/imagery/search', methods=['GET'])
def search_satellite_imagery():
    """
    Search for satellite imagery around a location and time.

    Query parameters:
    - latitude: Target latitude (required)
    - longitude: Target longitude (required)
    - date: Target date in ISO format (required)
    - window_days: Days before/after to search (default: 30)
    """
    try:
        # Validate parameters
        if not all([request.args.get('latitude'), request.args.get('longitude'), request.args.get('date')]):
            return jsonify({'error': 'latitude, longitude, and date are required'}), 400

        latitude = float(request.args.get('latitude'))
        longitude = float(request.args.get('longitude'))
        target_date = datetime.fromisoformat(request.args.get('date'))
        window_days = int(request.args.get('window_days', 30))

        # Search for imagery
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            imagery = loop.run_until_complete(
                satellite_engine.find_satellite_imagery(latitude, longitude, target_date, window_days)
            )
        finally:
            loop.close()

        # Convert to JSON format
        imagery_list = [
            {
                'source': img.source,
                'date': img.date.isoformat(),
                'latitude': img.latitude,
                'longitude': img.longitude,
                'resolution': img.resolution,
                'bands': img.bands,
                'url': img.url,
                'metadata': img.metadata
            }
            for img in imagery
        ]

        return jsonify({
            'imagery': imagery_list,
            'count': len(imagery_list),
            'search_parameters': {
                'latitude': latitude,
                'longitude': longitude,
                'target_date': target_date.isoformat(),
                'window_days': window_days
            }
        })

    except Exception as e:
        logger.error(f"Error searching satellite imagery: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/satellite/environmental', methods=['GET'])
def get_environmental_factors():
    """
    Get environmental factors for a location and time.

    Query parameters:
    - latitude: Target latitude (required)
    - longitude: Target longitude (required)
    - date: Target date in ISO format (required)
    """
    try:
        # Validate parameters
        if not all([request.args.get('latitude'), request.args.get('longitude'), request.args.get('date')]):
            return jsonify({'error': 'latitude, longitude, and date are required'}), 400

        latitude = float(request.args.get('latitude'))
        longitude = float(request.args.get('longitude'))
        target_date = datetime.fromisoformat(request.args.get('date'))

        # Get environmental factors
        factors = satellite_engine.get_environmental_factors(latitude, longitude, target_date)

        return jsonify({
            'environmental_factors': factors,
            'location': {
                'latitude': latitude,
                'longitude': longitude
            },
            'date': target_date.isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting environmental factors: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == "__main__":
    # Development server
    app.run(debug=True, host='0.0.0.0', port=5000)

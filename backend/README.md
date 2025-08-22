# SeismoGuard Backend

A production-ready seismic detection backend with machine learning capabilities, pressure-based noise masking, and multi-source data integration.

## 🌟 Features

### Core Functionality
- **Multi-source Data Fetching**: IRIS, NASA PDS, URL, and file upload support
- **Pressure-based Noise Masking**: Automatic wind noise removal for InSight data
- **Planet-specific Processing**: Optimized configurations for Earth, Mars, and Moon
- **RESTful API**: Flask-based API with CORS support for frontend integration

### Detection Algorithms
- **Classical Methods**: STA/LTA, Z-score detection
- **Machine Learning**: CNN-based classifiers and LSTM models
- **Anomaly Detection**: Autoencoder-based anomaly detection
- **Ensemble Methods**: Multiple algorithm combination

### Advanced Features
- **Teacher-Student Distillation**: Model compression for edge deployment
- **TFLite Conversion**: Quantized models for mobile/embedded systems
- **Representative Dataset Generation**: Real data calibration for quantization
- **Comprehensive Visualization**: Plots, spectrograms, and detection reports

## 🚀 Quick Start

### Installation

1. **Clone and navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   python run_server.py
   ```

The server will start on `http://127.0.0.1:5000` and is ready for frontend integration.

### Frontend Integration

The backend is designed to work seamlessly with the existing SeismoGuard frontend:

```javascript
// Frontend ApiClient automatically connects to backend
const api = new ApiClient("http://127.0.0.1:5000");

// Upload seismic data with optional pressure masking
const formData = new FormData();
formData.append('seismic', seismicFile);
formData.append('pressure', pressureFile);  // Optional
formData.append('planet', 'mars');

const response = await api.detect(formData);
```

## 📡 API Endpoints

### Main Detection Endpoint
```http
POST /detect
```

**Parameters:**
- `seismic`: Seismic data file (required if no other source)
- `pressure`: Pressure data file (optional, for noise masking)
- `source`: Data source (`iris`, `url`, `pds_search`)
- `planet`: Planet configuration (`earth`, `mars`, `moon`)

**IRIS Parameters:**
- `network`, `station`, `channel`, `starttime`, `endtime`

**URL Parameters:**
- `url`: URL to seismic data file

**PDS Parameters:**
- `mission`: Mission name (default: `insight`)
- `instrument`: Instrument name (default: `SEIS`)

**Response:**
```json
{
  "events": [
    {
      "time": "2023-01-01T12:00:00",
      "magnitude": 4.2,
      "confidence": 0.85,
      "algorithm": "sta_lta"
    }
  ],
  "diagnostics": {
    "total_events": 1,
    "algorithms_used": ["sta_lta"],
    "processing": {...}
  }
}
```

### Enhanced API Endpoints

#### Real-time Data & Monitoring
- `GET /earthquakes/recent` - Get recent earthquakes from multiple sources
- `GET /data-sources/status` - Status of all configured data sources
- `POST /compare/events` - Compare detected events with earthquake catalogs
- `GET /stream/status` - WebSocket streaming server status
- `POST /stream/broadcast` - Broadcast detection events to subscribers

#### Advanced IRIS Integration
- `GET /iris/stations/search` - Search for available IRIS stations
- `GET /iris/events/search` - Search earthquake events using IRIS
- `GET /iris/stations/nearby` - Find stations near a location
- `GET /iris/stations/{network}/{station}/availability` - Station data availability

#### Machine Learning APIs
- `POST /ml/classify` - Classify seismic data using external ML APIs
- `POST /ml/features` - Extract advanced features using ML
- `POST /ml/ensemble` - Ensemble analysis with multiple ML models
- `GET /ml/status` - Status of configured ML APIs

#### Satellite Data Correlation
- `POST /satellite/correlate` - Correlate seismic events with satellite data
- `GET /satellite/imagery/search` - Search for satellite imagery
- `GET /satellite/environmental` - Get environmental factors

#### Basic Endpoints
- `GET /health` - Health check
- `GET /algorithms` - Available detection algorithms
- `GET /planet-presets` - Planet processing configurations

## 🛠️ CLI Usage

The backend includes a comprehensive CLI for standalone operation:

### Basic Detection
```bash
# Process local file
python -m app.cli --file data/sample.mseed --planet mars

# With pressure masking
python -m app.cli --file seis.mseed --pressure pressure.mseed
```

### Data Fetching
```bash
# Fetch from IRIS
python -m app.cli --iris IU ANMO BHZ 2023-01-01T00:00:00 2023-01-01T01:00:00

# Search NASA PDS
python -m app.cli --pds_search insight SEIS 5 --planet mars

# Fetch from URL
python -m app.cli --url https://example.com/seismic_data.mseed
```

### Model Training
```bash
# Train synthetic models
python -m app.cli --train --epochs 10

# Run distillation pipeline
python -m app.cli --distill --epochs 8

# Test TFLite model
python -m app.cli --test_tflite models/compact_quant.tflite
```

## 🧪 Testing

### Run All Tests
```bash
python test_backend.py
```

### Integration Test
```bash
python test_backend.py --integration
```

### Specific Test Categories
```bash
# API tests only
python -m unittest test_backend.TestAPIEndpoints

# Detection algorithm tests
python -m unittest test_backend.TestDetectionAlgorithms
```

## 📁 Project Structure

```
backend/
├── app/                    # Main application package
│   ├── __init__.py        # Package initialization
│   ├── api.py             # Flask API with CORS
│   ├── cli.py             # Command-line interface
│   ├── detector_manager.py # Detection coordination
│   ├── detector_ml.py     # ML model definitions
│   ├── ml_train.py        # Training utilities
│   ├── noise_masker.py    # Pressure-based masking
│   ├── preprocess.py      # Data preprocessing
│   ├── io_utils.py        # I/O utilities
│   ├── data_fetcher.py    # Multi-source data fetching
│   ├── tflite_utils.py    # TFLite conversion
│   ├── quant_rep_gen.py   # Representative datasets
│   ├── distill.py         # Knowledge distillation
│   └── visualizer.py      # Plotting and reports
├── run_server.py          # Server startup script
├── test_backend.py        # Comprehensive test suite
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Planet Presets
The backend includes optimized processing configurations:

- **Earth**: 0.5-20 Hz (typical earthquake frequencies)
- **Mars**: 0.1-5 Hz (InSight SEIS optimized)
- **Moon**: 0.03-2 Hz (Apollo-era frequencies)

### Detection Algorithms
- **STA/LTA**: Short-term/Long-term average ratio
- **Z-score**: Statistical anomaly detection
- **CNN Classifier**: Deep learning event classification
- **LSTM**: Temporal pattern recognition
- **Autoencoder**: Unsupervised anomaly detection

## 🌍 Data Sources

### IRIS FDSN Web Services
Access to global seismic networks:
```python
fetch_from_iris('IU', 'ANMO', 'BHZ', start_time, end_time)
```

### NASA PDS (Planetary Data System)
InSight mission data with automatic pressure fetching:
```python
fetch_from_pds_search(mission='insight', instrument='SEIS')
```

### URL-based Fetching
Direct download from web resources:
```python
fetch_from_url('https://example.com/data.mseed')
```

### File Upload
Support for multiple formats:
- MiniSEED (.mseed, .msd)
- SAC (.sac)
- Compressed files (.gz, .zip)

## 🧠 Machine Learning Pipeline

### Model Architecture
1. **Teacher Model**: Full CNN for maximum accuracy
2. **Student Model**: Compact CNN for deployment
3. **Distillation**: Knowledge transfer from teacher to student
4. **TFLite Conversion**: Quantized model for edge devices

### Training Pipeline
```bash
# Generate synthetic data and train models
python -m app.cli --train

# Run complete distillation pipeline
python -m app.cli --distill
```

### Representative Dataset
Automatic generation of calibration data from:
- Local seismic files
- NASA PDS InSight data
- Synthetic earthquake signals

## 🔇 Noise Masking

### Pressure-based Wind Noise Removal
Automatic detection and masking of wind-contaminated segments:

1. **Gust Detection**: Sliding-window standard deviation analysis
2. **Mask Dilation**: Temporal expansion of contaminated periods
3. **Seismic Masking**: Zero-out, NaN, or attenuation of noisy segments

### Usage
```python
# Automatic masking when pressure data available
masked_seismic, mask = mask_seismic_with_pressure(
    seismic_data, pressure_data, sampling_rate
)
```

## 📊 Visualization and Reporting

### Automatic Plot Generation
- Seismic traces with event markers
- Detection summary statistics
- Noise masking visualization
- Spectrograms and frequency analysis

### Comprehensive Reports
```python
# Generate complete detection report
report_dir = create_detection_report(trace, events_df, diagnostics)
```

## 🚀 Production Deployment

### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app.api:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "run_server.py"]
```

### Environment Variables
- `FLASK_ENV`: Set to `production` for deployment
- `SEISMO_MODEL_PATH`: Custom model directory
- `SEISMO_DATA_PATH`: Data storage location

## 🤝 Frontend Integration

### Graceful Degradation
The backend is designed for non-disruptive integration:
- Frontend works with or without backend
- Automatic fallback to mock data
- Built-in caching and request throttling
- CORS configured for all common frontend ports

### ApiClient Compatibility
Perfect compatibility with existing frontend `ApiClient`:
```javascript
// Existing frontend code works unchanged
const response = await api.detect(formData, { cacheKey: 'mars_pds' });
```

## 📈 Performance

### Benchmarks
- **Classical Detection**: ~1000 samples/second
- **ML Inference**: ~100 samples/second
- **TFLite Quantized**: ~500 samples/second
- **API Response**: <2 seconds for typical requests

### Optimization Features
- Efficient numpy-based processing
- Batch inference for ML models
- Automatic model caching
- Parallel algorithm execution

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies installed
   ```bash
   pip install -r requirements.txt
   ```

2. **CORS Issues**: Check frontend URL in CORS configuration
   ```python
   CORS(app, origins=["http://localhost:3000", ...])
   ```

3. **Model Loading**: Ensure models directory exists
   ```bash
   mkdir -p models
   ```

4. **Memory Issues**: Reduce batch size for large datasets
   ```bash
   python -m app.cli --train --batch_size 16
   ```

### Debug Mode
```bash
python run_server.py --debug --verbose
```

## 📄 License

This project is part of the SeismoGuard application developed for the NASA Space Apps Challenge 2025.

## 🙏 Acknowledgments

- NASA for InSight mission data and PDS infrastructure
- IRIS for global seismic data access
- ObsPy community for seismic data processing tools
- TensorFlow team for ML framework and TFLite conversion

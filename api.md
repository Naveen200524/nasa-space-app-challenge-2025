🌍 Complete Dataset & API Integration Guide for SeismoGuard
📊 Seismic & Planetary Datasets
1. NASA/Space Agency Datasets
Apollo Lunar Seismic Data
const apolloDatasets = {
    'Apollo PSE': {
        url: 'https://pds-geosciences.wustl.edu/lunar/urn-nasa-pds-apollo_pse/',
        format: 'SEED/miniSEED',
        description: 'Complete Apollo 11,12,14,15,16 passive seismic experiments',
        size: '~12GB',
        api: null,
        directDownload: true,
        integration: `
            // Integration code
            async function loadApolloData() {
                const response = await fetch('/data/apollo/apollo12.mseed');
                const buffer = await response.arrayBuffer();
                return parseMiniSEED(buffer);
            }
        `
    },
    
    'Lunar Seismic Profiling': {
        url: 'https://pds-geosciences.wustl.edu/missions/apollo/seismic_experiments_se.htm',
        format: 'CSV/ASCII',
        years: '1969-1977',
        events: '13,000+ moonquakes'
    },
    
    'Nakamura Moonquake Catalog': {
        url: 'https://pds-geosciences.wustl.edu/lunar/urn-nasa-pds-apollo_seismic_event_catalog/',
        format: 'CSV',
        catalogedEvents: {
            deep: 1743,
            shallow: 28,
            meteoroid: 1743,
            artificial: 9
        }
    }
};

Mars InSight Seismic Data
const marsDatasets = {
    'InSight SEIS': {
        url: 'https://pds-geosciences.wustl.edu/insight/',
        api: 'https://api.nasa.gov/insight_weather/',
        apiKey: 'DEMO_KEY', // Get from https://api.nasa.gov/
        format: 'miniSEED/SAC',
        continuous: true,
        integration: `
            async function getInSightData() {
                const API_KEY = 'YOUR_NASA_API_KEY';
                const response = await fetch(\`https://api.nasa.gov/insight_weather/?api_key=\${API_KEY}&feedtype=json&ver=1.0\`);
                return await response.json();
            }
        `
    },
    
    'Mars Quake Service (MQS)': {
        url: 'https://www.insight.ethz.ch/seismicity/mqs-mars-catalogue/',
        format: 'QuakeML/CSV',
        events: '1300+ marsquakes',
        grades: ['A', 'B', 'C', 'D'] // Quality grades
    },
    
    'IRIS Mars Data': {
        url: 'https://www.iris.edu/hq/sis/insight',
        webService: 'http://service.iris.edu/fdsnws/',
        protocols: ['dataselect', 'station', 'event']
    }
};

2. Earth Seismic Networks
IRIS (Incorporated Research Institutions for Seismology)
class IRISIntegration {
    constructor() {
        this.baseURL = 'http://service.iris.edu/fdsnws/';
        this.services = {
            event: 'event/1/',
            dataselect: 'dataselect/1/',
            station: 'station/1/',
            availability: 'availability/1/'
        };
    }
    
    async getRecentEarthquakes(params = {}) {
        const defaultParams = {
            format: 'json',
            starttime: '2024-01-01',
            endtime: '2024-12-31',
            minmagnitude: 4.0,
            orderby: 'time'
        };
        
        const queryParams = { ...defaultParams, ...params };
        const url = `${this.baseURL}${this.services.event}query?${new URLSearchParams(queryParams)}`;
        
        const response = await fetch(url);
        return await response.json();
    }
    
    async getWaveformData(network, station, starttime, endtime) {
        const params = {
            net: network,
            sta: station,
            start: starttime,
            end: endtime,
            format: 'miniseed'
        };
        
        const url = `${this.baseURL}${this.services.dataselect}query?${new URLSearchParams(params)}`;
        const response = await fetch(url);
        return await response.arrayBuffer();
    }
}

USGS Earthquake Data
const USGSIntegration = {
    apis: {
        realtime: 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/',
        historical: 'https://earthquake.usgs.gov/fdsnws/event/1/',
        
        feeds: {
            // Real-time feeds
            'past_hour_all': 'summary/all_hour.geojson',
            'past_day_M4.5+': 'summary/4.5_day.geojson',
            'past_week_M2.5+': 'summary/2.5_week.geojson',
            'past_month_M2.5+': 'summary/2.5_month.geojson'
        }
    },
    
    async getRealtimeQuakes(feed = 'past_day_M4.5+') {
        const url = `${this.apis.realtime}${this.feeds[feed]}`;
        const response = await fetch(url);
        const data = await response.json();
        
        return data.features.map(f => ({
            magnitude: f.properties.mag,
            location: f.properties.place,
            time: new Date(f.properties.time),
            coordinates: f.geometry.coordinates,
            depth: f.geometry.coordinates[2],
            type: f.properties.type,
            tsunami: f.properties.tsunami,
            url: f.properties.url
        }));
    }
};

3. Regional Seismic Networks
const regionalNetworks = {
    'European EMSC': {
        url: 'https://www.emsc-csem.org/Earthquake/',
        api: 'https://www.seismicportal.eu/fdsnws/',
        websocket: 'wss://www.seismicportal.eu/streaming',
        coverage: 'Euro-Mediterranean'
    },
    
    'Japan JMA': {
        url: 'https://www.jma.go.jp/jma/en/Activities/earthquake.html',
        api: 'https://api.p2pquake.net/v2/',
        realtime: true,
        earlyWarning: true
    },
    
    'GeoNet New Zealand': {
        api: 'https://api.geonet.org.nz/',
        endpoints: ['quake', 'intensity', 'volcano'],
        format: 'GeoJSON'
    },
    
    'ANSS ComCat': {
        url: 'https://earthquake.usgs.gov/data/comcat/',
        api: 'https://earthquake.usgs.gov/fdsnws/event/1/',
        historicalData: '1900-present'
    }
};

🛰️ Satellite & Remote Sensing APIs
Earth Observation Data
const satelliteAPIs = {
    'NASA Earthdata': {
        url: 'https://earthdata.nasa.gov/api',
        auth: 'Bearer token required',
        datasets: ['MODIS', 'Landsat', 'Sentinel'],
        
        async searchGranules(params) {
            const headers = {
                'Authorization': `Bearer ${this.token}`,
                'Content-Type': 'application/json'
            };
            
            const response = await fetch('https://cmr.earthdata.nasa.gov/search/granules.json', {
                method: 'POST',
                headers,
                body: JSON.stringify(params)
            });
            
            return await response.json();
        }
    },
    
    'Copernicus Open Access Hub': {
        url: 'https://scihub.copernicus.eu/dhus/',
        satellites: ['Sentinel-1', 'Sentinel-2', 'Sentinel-3'],
        dataTypes: ['SAR', 'Optical', 'InSAR for ground deformation']
    },
    
    'Planet Labs': {
        api: 'https://api.planet.com/data/v1/',
        auth: 'API Key',
        resolution: '3-5m daily',
        useCase: 'Monitor surface changes before/after seismic events'
    }
};

🤖 Machine Learning & AI APIs
Pre-trained Models & Services
const mlAPIs = {
    'Google Earth Engine': {
        url: 'https://earthengine.google.com/',
        auth: 'OAuth 2.0',
        capabilities: [
            'Petabytes of satellite imagery',
            'Built-in ML algorithms',
            'Time series analysis'
        ],
        
        integration: `
            // Requires Earth Engine Python/JS API
            ee.initialize();
            
            const collection = ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterDate('2024-01-01', '2024-12-31')
                .filterBounds(aoi);
            
            // Detect ground deformation
            const deformation = collection.map(calculateInSAR);
        `
    },
    
    'Hugging Face Transformers': {
        api: 'https://api-inference.huggingface.co/models/',
        models: {
            'facebook/wav2vec2-base': 'Audio classification',
            'microsoft/wavlm-base': 'Seismic signal processing',
            'timm/efficientnet': 'Spectrogram analysis'
        },
        
        async classifySignal(audioData) {
            const response = await fetch(
                'https://api-inference.huggingface.co/models/facebook/wav2vec2-base',
                {
                    headers: { Authorization: `Bearer ${HF_TOKEN}` },
                    method: 'POST',
                    body: audioData
                }
            );
            return await response.json();
        }
    },
    
    'AWS SageMaker': {
        endpoint: 'https://runtime.sagemaker.region.amazonaws.com/',
        pretrainedModels: [
            'Seismic Facies Classification',
            'First Break Picking',
            'Event Detection'
        ]
    },
    
    'Google Cloud AI Platform': {
        api: 'https://ml.googleapis.com/v1/',
        autoML: true,
        vertexAI: 'Custom model training'
    }
};

🌐 Real-time Data Streams
WebSocket & Streaming Services
class RealtimeDataStreams {
    constructor() {
        this.streams = {
            'IRIS Web Services': {
                ws: 'wss://www.iris.edu/ws/',
                subscribe: (channels) => {
                    const ws = new WebSocket('wss://www.iris.edu/ws/');
                    ws.onopen = () => {
                        ws.send(JSON.stringify({
                            action: 'subscribe',
                            channels: channels // e.g., ['IU.ANMO.00.BHZ']
                        }));
                    };
                    return ws;
                }
            },
            
            'Raspberry Shake Network': {
                api: 'https://raspberryshake.net/api/',
                ws: 'wss://dataview.raspberryshake.org/',
                community: true,
                stations: '20,000+ worldwide'
            },
            
            'SeedLink Protocol': {
                servers: [
                    'rtserve.iris.washington.edu:18000',
                    'geofon.gfz-potsdam.de:18000',
                    'rtserve.beg.utexas.edu:18000'
                ],
                protocol: 'TCP/IP real-time data'
            }
        };
    }
    
    connectToSeedLink(server, station) {
        // Requires SeedLink client library
        const client = new SeedLinkClient(server);
        client.connect();
        client.select(station);
        client.on('data', (packet) => {
            this.processRealtimeData(packet);
        });
    }
}

🗺️ Geological & Geophysical Data
Supporting Datasets
const geologicalData = {
    'USGS National Map': {
        api: 'https://viewer.nationalmap.gov/services/',
        layers: ['Geology', 'Faults', 'Earthquakes', 'Volcanoes'],
        format: 'WMS/WFS/REST'
    },
    
    'Global CMT Catalog': {
        url: 'https://www.globalcmt.org/',
        api: 'https://www.globalcmt.org/cgi-bin/globalcmt-cgi-bin/CMT5/form',
        data: 'Moment tensors for M>5.0 since 1976'
    },
    
    'ISC (International Seismological Centre)': {
        api: 'http://www.isc.ac.uk/iscbulletin/search/webservices/',
        catalog: '1900-present reviewed bulletin'
    },
    
    'UNAVCO GPS/GNSS': {
        api: 'https://www.unavco.org/data/web-services/',
        data: 'Crustal deformation measurements',
        useCase: 'Correlate with seismic activity'
    }
};

💾 Implementation Examples
Complete Integration Class
class DataIntegrationHub {
    constructor() {
        this.apis = new Map();
        this.cache = new Map();
        this.rateLimits = new Map();
        
        this.initializeAPIs();
    }
    
    initializeAPIs() {
        // NASA APIs
        this.apis.set('nasa_insight', {
            url: 'https://api.nasa.gov/insight_weather/',
            key: process.env.NASA_API_KEY,
            rateLimit: 1000, // per hour
            cache: 3600000 // 1 hour
        });
        
        // USGS APIs
        this.apis.set('usgs_realtime', {
            url: 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/',
            key: null, // No key required
            rateLimit: null,
            cache: 60000 // 1 minute for real-time
        });
        
        // IRIS APIs
        this.apis.set('iris_fedcatalog', {
            url: 'http://service.iris.edu/fedcatalog/',
            key: null,
            rateLimit: null,
            cache: 86400000 // 1 day for catalog
        });
    }
    
    async fetchWithCache(apiName, endpoint, params = {}) {
        const cacheKey = `${apiName}_${endpoint}_${JSON.stringify(params)}`;
        
        // Check cache
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.apis.get(apiName).cache) {
                return cached.data;
            }
        }
        
        // Check rate limit
        if (!this.checkRateLimit(apiName)) {
            throw new Error(`Rate limit exceeded for ${apiName}`);
        }
        
        // Fetch data
        const api = this.apis.get(apiName);
        const url = new URL(api.url + endpoint);
        
        if (api.key) {
            params.api_key = api.key;
        }
        
        Object.entries(params).forEach(([key, value]) => {
            url.searchParams.append(key, value);
        });
        
        const response = await fetch(url);
        const data = await response.json();
        
        // Update cache
        this.cache.set(cacheKey, {
            data: data,
            timestamp: Date.now()
        });
        
        // Update rate limit
        this.updateRateLimit(apiName);
        
        return data;
    }
    
    checkRateLimit(apiName) {
        const api = this.apis.get(apiName);
        if (!api.rateLimit) return true;
        
        const calls = this.rateLimits.get(apiName) || [];
        const now = Date.now();
        const recentCalls = calls.filter(time => now - time < 3600000); // Last hour
        
        return recentCalls.length < api.rateLimit;
    }
    
    updateRateLimit(apiName) {
        const api = this.apis.get(apiName);
        if (!api.rateLimit) return;
        
        const calls = this.rateLimits.get(apiName) || [];
        calls.push(Date.now());
        this.rateLimits.set(apiName, calls);
    }
    
    // Specific API methods
    async getMarsQuakes(startDate, endDate) {
        // Combine multiple sources
        const insights = await this.fetchWithCache('nasa_insight', '', {
            start_date: startDate,
            end_date: endDate
        });
        
        const iris = await this.fetchWithCache('iris_fedcatalog', 'query', {
            network: 'XB', // InSight network code
            starttime: startDate,
            endtime: endDate
        });
        
        return this.mergeDatasets([insights, iris]);
    }
    
    async getGlobalEarthquakes(magnitude = 4.0) {
        const usgs = await this.fetchWithCache('usgs_realtime', '4.5_day.geojson');
        
        const emsc = await fetch('https://www.seismicportal.eu/fdsnws/event/1/query?' + 
            new URLSearchParams({
                format: 'json',
                minmagnitude: magnitude,
                starttime: new Date(Date.now() - 86400000).toISOString()
            }));
        
        return this.mergeDatasets([usgs, await emsc.json()]);
    }
    
    mergeDatasets(datasets) {
        // Implement deduplication and merging logic
        const merged = new Map();
        
        datasets.forEach(dataset => {
            // Process each dataset format
            this.normalizeDataset(dataset).forEach(event => {
                const key = `${event.time}_${event.location}`;
                if (!merged.has(key) || merged.get(key).quality < event.quality) {
                    merged.set(key, event);
                }
            });
        });
        
        return Array.from(merged.values());
    }
    
    normalizeDataset(dataset) {
        // Convert different formats to common schema
        // Implementation depends on specific dataset format
        return dataset;
    }
}

// Initialize the hub
const dataHub = new DataIntegrationHub();

// Use in your application
async function loadAllData() {
    try {
        // Get Mars data
        const marsQuakes = await dataHub.getMarsQuakes('2024-01-01', '2024-12-31');
        
        // Get Earth comparison data
        const earthquakes = await dataHub.getGlobalEarthquakes(4.0);
        
        // Get lunar historical data
        const lunarEvents = await loadLocalDataset('/data/apollo_catalog.csv');
        
        // Combine for analysis
        return {
            mars: marsQuakes,
            earth: earthquakes,
            moon: lunarEvents
        };
    } catch (error) {
        console.error('Data loading failed:', error);
    }
}

🔌 Quick Integration Guide
1. Environment Variables Setup
# .env file
NASA_API_KEY=your_nasa_api_key_here
USGS_API_KEY=optional_key_here
PLANET_API_KEY=your_planet_key_here
HF_TOKEN=your_huggingface_token
AWS_ACCESS_KEY=your_aws_key
GCP_PROJECT_ID=your_gcp_project

2. Package Dependencies
{
  "dependencies": {
    "obspy-js": "^1.0.0",
    "seedlink-client": "^2.0.0",
    "geojson": "^0.5.0",
    "mseed-js": "^1.0.0",
    "sac-js": "^1.0.0",
    "@turf/turf": "^6.5.0",
    "d3-geo": "^3.0.0",
    "mqtt": "^4.3.0",
    "websocket": "^1.0.34"
  }
}

3. CORS Proxy Setup (for browser-based requests)
// Use a CORS proxy for APIs that don't support CORS
const CORS_PROXY = 'https://cors-anywhere.herokuapp.com/';

async function fetchWithCORS(url) {
    return fetch(CORS_PROXY + url);
}

📈 Priority Datasets for Hackathon
Top 5 Must-Have Integrations:
NASA InSight Mars Data - Recent, relevant, impressive
USGS Real-time Earthquakes - Live data, well-documented
Apollo Lunar Catalog - Historical significance
IRIS Web Services - Professional standard
Raspberry Shake Network - Community aspect
Bonus Points Integrations:
Machine Learning APIs - Shows advanced capabilities
Satellite Imagery - Visual impact
Real-time WebSockets - Live demo capability
GPS/GNSS Data - Multi-sensor fusion
This comprehensive integration guide gives you access to petabytes of seismic data and dozens of APIs to make SeismoGuard a truly powerful, data-rich application! 🚀


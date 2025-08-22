export class WaveformVisualizer {
    constructor(containerId) {
        this.container = document.querySelector(containerId);
        this.chart = null;
        this.currentPlanet = null;
        this.dataMode = 'realtime';
        this.animationId = null;
        this.onEventSelect = null;
        
        this.seismicData = this.generateMockWaveformData();
        // Track requests to avoid redundant backend calls across view switches
        this._requestedDetections = new Set();
        this.initialize();
    }
    
    initialize() {
        if (!this.container) return;

        try {
            this.createChart();
            this.startRealTimeUpdates();
        } catch (e) {
            console.warn('WaveformVisualizer initialization skipped (likely CDN not ready):', e);
        }
    }
    
    createChart() {
        const layout = {
            title: {
                text: 'Seismic Waveform Analysis',
                font: { color: '#00ffff', family: 'Orbitron', size: 16 }
            },
            xaxis: {
                title: 'Time (seconds)',
                color: '#ffffff',
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                tickfont: { color: '#b4b4b8' },
                titlefont: { color: '#b4b4b8' }
            },
            yaxis: {
                title: 'Amplitude (m/s²)',
                color: '#ffffff',
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                tickfont: { color: '#b4b4b8' },
                titlefont: { color: '#b4b4b8' }
            },
            plot_bgcolor: 'rgba(0, 0, 0, 0)',
            paper_bgcolor: 'rgba(0, 0, 0, 0)',
            font: { color: '#ffffff' },
            showlegend: true,
            legend: {
                font: { color: '#b4b4b8' },
                bgcolor: 'rgba(0, 0, 0, 0.3)'
            },
            hovermode: 'x unified'
        };
        
        const config = {
            displayModeBar: false,
            responsive: true
        };
        
        // Initial empty traces
        const data = [
            {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'Waveform',
                line: { color: '#00ffff', width: 2 }
            },
            {
                x: [],
                y: [],
                type: 'scatter',
                mode: 'markers',
                name: 'Seismic Events',
                marker: {
                    color: '#f97316',
                    size: 8,
                    symbol: 'diamond'
                }
            }
        ];
        
        Plotly.newPlot(this.container, data, layout, config);
        
        // Add click event listener
        this.container.on('plotly_click', (data) => {
            if (data.points.length > 0 && data.points[0].curveNumber === 1) {
                const point = data.points[0];
                const event = this.findEventByTime(point.x);
                if (event && this.onEventSelect) {
                    this.onEventSelect(event);
                }
            }
        });
    }
    
    generateMockWaveformData() {
        const data = {};
        const planets = ['earth', 'mars', 'moon'];
        
        planets.forEach(planet => {
            data[planet] = {
                realtime: this.generateRealtimeData(planet),
                historical: this.generateHistoricalData(planet)
            };
        });
        
        return data;
    }
    
    generateRealtimeData(planet) {
        const points = 1000;
        const timespan = 3600; // 1 hour in seconds
        const x = [];
        const y = [];
        const events = [];
        
        // Base characteristics for each planet
        const planetChar = {
            earth: { baseFreq: 0.1, amplitude: 1, noiseLevel: 0.1 },
            mars: { baseFreq: 0.05, amplitude: 0.3, noiseLevel: 0.05 },
            moon: { baseFreq: 0.02, amplitude: 0.1, noiseLevel: 0.02 }
        };
        
        const char = planetChar[planet] || planetChar.earth;
        
        for (let i = 0; i < points; i++) {
            const time = (i / points) * timespan;
            let amplitude = 0;
            
            // Base seismic noise
            amplitude += (Math.random() - 0.5) * char.noiseLevel;
            amplitude += Math.sin(time * char.baseFreq) * char.noiseLevel * 0.5;
            
            // Add seismic events randomly
            if (Math.random() < 0.005) { // 0.5% chance per point
                const eventMagnitude = Math.random() * 8 + 1;
                const eventAmplitude = eventMagnitude * char.amplitude;
                
                // Create event signature
                for (let j = 0; j < 50 && (i + j) < points; j++) {
                    const decay = Math.exp(-j * 0.1);
                    const eventTime = ((i + j) / points) * timespan;
                    const eventY = Math.sin(j * 0.5) * eventAmplitude * decay;
                    
                    if (j === 0) {
                        events.push({
                            id: `event_${Date.now()}_${Math.random()}`,
                            time: eventTime,
                            magnitude: eventMagnitude,
                            depth: Math.random() * 700 + 10,
                            duration: 50 + Math.random() * 200,
                            amplitude: eventAmplitude,
                            planet: planet
                        });
                    }
                }
                
                amplitude += Math.sin(Math.random() * Math.PI) * eventAmplitude;
            }
            
            x.push(time);
            y.push(amplitude);
        }
        
        return { x, y, events };
    }
    
    generateHistoricalData(planet) {
        const points = 2000;
        const timespan = 86400 * 7; // 1 week in seconds
        const x = [];
        const y = [];
        const events = [];
        
        const planetChar = {
            earth: { baseFreq: 0.01, amplitude: 2, noiseLevel: 0.2, eventRate: 0.01 },
            mars: { baseFreq: 0.005, amplitude: 0.5, noiseLevel: 0.08, eventRate: 0.003 },
            moon: { baseFreq: 0.002, amplitude: 0.2, noiseLevel: 0.03, eventRate: 0.001 }
        };
        
        const char = planetChar[planet] || planetChar.earth;
        
        for (let i = 0; i < points; i++) {
            const time = (i / points) * timespan;
            let amplitude = 0;
            
            // Base seismic noise with diurnal variation
            const diurnalFactor = 1 + 0.3 * Math.sin((time / 86400) * 2 * Math.PI);
            amplitude += (Math.random() - 0.5) * char.noiseLevel * diurnalFactor;
            amplitude += Math.sin(time * char.baseFreq) * char.noiseLevel * 0.3;
            
            // Add major seismic events
            if (Math.random() < char.eventRate) {
                const eventMagnitude = Math.random() * 8 + 2;
                const eventAmplitude = eventMagnitude * char.amplitude;
                
                events.push({
                    id: `historical_${Date.now()}_${Math.random()}`,
                    time: time,
                    magnitude: eventMagnitude,
                    depth: Math.random() * 700 + 10,
                    duration: 100 + Math.random() * 400,
                    amplitude: eventAmplitude,
                    planet: planet,
                    timestamp: Date.now() - (timespan - time) * 1000
                });
                
                // Create realistic earthquake signature
                const eventDuration = Math.min(100, points - i);
                for (let j = 0; j < eventDuration; j++) {
                    const phase = j / eventDuration * Math.PI;
                    const decay = Math.exp(-j * 0.02);
                    amplitude += Math.sin(phase * 3) * eventAmplitude * decay;
                }
            }
            
            x.push(time);
            y.push(amplitude);
        }
        
        return { x, y, events };
    }
    
    loadPlanet(planetName) {
        this.currentPlanet = planetName;
        this.updateChart();
        // Optional backend overlay of detections (non-blocking, fail-silent)
        try {
            const api = window.seismoGuardApp?.api;
            // Only attempt real data overlay for Mars (InSight) and avoid duplicate calls
            const key = `pds_search:${planetName}`;
            if (api && planetName === 'mars' && !this._requestedDetections.has(key)) {
                this._requestedDetections.add(key);
                this.overlayDetectionsFromPDS(api, planetName, { cacheKey: key }).catch(() => {});
            }
        } catch (_) { /* ignore */ }
    }
    
    setDataMode(mode) {
        this.dataMode = mode;
        this.updateChart();
        
        if (mode === 'realtime') {
            this.startRealTimeUpdates();
        } else {
            this.stopRealTimeUpdates();
        }
    }
    
    updateChart() {
        if (!this.currentPlanet || !this.seismicData[this.currentPlanet]) return;
        
        const data = this.seismicData[this.currentPlanet][this.dataMode];
        const eventX = data.events.map(e => e.time);
        const eventY = data.events.map(e => e.amplitude);
        
        const update = {
            x: [data.x, eventX],
            y: [data.y, eventY]
        };
        
        Plotly.restyle(this.container, update, [0, 1]);
        
        // Update title based on planet and mode
        const title = `${this.getPlanetDisplayName(this.currentPlanet)} - ${this.dataMode === 'realtime' ? 'Real-time' : 'Historical'} Seismic Data`;
        Plotly.relayout(this.container, { title: title });
    }
    
    getPlanetDisplayName(planetName) {
        const names = {
            earth: 'Earth Seismogram',
            mars: 'Mars Seismogram (InSight)',
            moon: 'Lunar Seismogram (Apollo)'
        };
        return names[planetName] || 'Seismogram';
    }
    
    highlightEvent(event) {
        // Add temporary highlight trace
        const highlightData = {
            x: [event.time],
            y: [event.amplitude],
            type: 'scatter',
            mode: 'markers',
            name: 'Selected Event',
            marker: {
                color: '#00ffff',
                size: 15,
                symbol: 'star',
                line: { color: '#ffffff', width: 2 }
            },
            showlegend: false
        };
        
        Plotly.addTraces(this.container, highlightData);
        
        // Remove highlight after 3 seconds
        setTimeout(() => {
            const currentTraces = this.container.data.length;
            if (currentTraces > 2) {
                Plotly.deleteTraces(this.container, currentTraces - 1);
            }
        }, 3000);
    }
    
    findEventByTime(time) {
        if (!this.currentPlanet || !this.seismicData[this.currentPlanet]) return null;
        
        const data = this.seismicData[this.currentPlanet][this.dataMode];
        return data.events.find(event => Math.abs(event.time - time) < 10); // 10 second tolerance
    }
    
    getCurrentData() {
        if (!this.currentPlanet || !this.seismicData[this.currentPlanet]) return null;
        
        return {
            planet: this.currentPlanet,
            mode: this.dataMode,
            data: this.seismicData[this.currentPlanet][this.dataMode],
            generatedAt: new Date().toISOString()
        };
    }
    
    startRealTimeUpdates() {
        if (this.dataMode !== 'realtime') return;
        
        const updateInterval = 1000; // Update every second
        
        const updateData = () => {
            if (this.dataMode === 'realtime' && this.currentPlanet) {
                // Simulate real-time data by slightly modifying the last few points
                const data = this.seismicData[this.currentPlanet].realtime;
                const lastIndex = data.x.length - 1;
                
                // Add small variations to simulate live data
                for (let i = Math.max(0, lastIndex - 10); i <= lastIndex; i++) {
                    data.y[i] += (Math.random() - 0.5) * 0.01;
                }
                
                // Sometimes add a new event
                if (Math.random() < 0.01) { // 1% chance per update
                    const eventMagnitude = Math.random() * 6 + 1;
                    const eventTime = data.x[lastIndex];
                    const eventAmplitude = eventMagnitude * 0.3;
                    
                    data.events.push({
                        id: `realtime_${Date.now()}_${Math.random()}`,
                        time: eventTime,
                        magnitude: eventMagnitude,
                        depth: Math.random() * 700 + 10,
                        duration: 30 + Math.random() * 120,
                        amplitude: eventAmplitude,
                        planet: this.currentPlanet,
                        timestamp: Date.now()
                    });
                }
                
                this.updateChart();
            }
            
            this.animationId = setTimeout(updateData, updateInterval);
        };
        
        this.stopRealTimeUpdates(); // Clear any existing updates
        this.animationId = setTimeout(updateData, updateInterval);
    }
    
    stopRealTimeUpdates() {
        if (this.animationId) {
            clearTimeout(this.animationId);
            this.animationId = null;
        }
    }

    // Best-effort detection overlay using backend /detect with PDS search
    async overlayDetectionsFromPDS(api, planet, { cacheKey } = {}) {
        try {
            const fd = new FormData();
            fd.append('source', 'pds_search');
            fd.append('planet', planet);
            fd.append('mission', 'insight');
            fd.append('instrument', 'SEIS');
            const resp = await api.detect(fd, { cacheKey });
            const events = Array.isArray(resp?.events) ? resp.events : [];
            if (!events.length) return;

            const d = this.seismicData[planet][this.dataMode];
            const t0 = d.x[d.x.length - 1] || 0;
            events.forEach((e, i) => {
                d.events.push({
                    id: `backend_${Date.now()}_${i}`,
                    time: t0 - i * 60,
                    magnitude: e.magnitude || 4,
                    depth: e.depth || 50,
                    duration: e.duration || 60,
                    amplitude: (e.amplitude || 0.5),
                    planet
                });
            });
            this.updateChart();
        } catch (err) {
            // Fail silently to preserve UX; optionally log for diagnostics
            console.debug('[WaveformVisualizer] overlayDetectionsFromPDS skipped:', err?.message || err);
        }
    }
    
    handleResize() {
        if (this.container) {
            Plotly.Plots.resize(this.container);
        }
    }
    
    destroy() {
        this.stopRealTimeUpdates();
        
        if (this.container) {
            Plotly.purge(this.container);
        }
    }
}
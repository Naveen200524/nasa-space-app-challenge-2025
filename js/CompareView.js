export class CompareView {
    constructor() {
        this.planetA = 'earth';
        this.planetB = 'mars';
        this.viewerA = null;
        this.viewerB = null;
        this.chartA = null;
        this.chartB = null;
        this.syncPlayback = true;
        this.initialized = false;
        
        this.planetData = this.generateComparisonData();
    }
    
    initialize() {
        // Ensure idempotent init: clean any previous instances and DOM children
        if (this.initialized) {
            this.destroy();
        }
        this.setupViewers();
        this.setupCharts();
        this.updateComparison();
        this.initialized = true;
    }
    
    setupViewers() {
        const viewerAContainer = document.getElementById('compare-viewer-a');
        const viewerBContainer = document.getElementById('compare-viewer-b');
        
        if (viewerAContainer && viewerBContainer) {
            this.viewerA = new ComparisonViewer(viewerAContainer, this.planetA);
            this.viewerB = new ComparisonViewer(viewerBContainer, this.planetB);
        }
    }
    
    setupCharts() {
        const chartAContainer = document.getElementById('compare-chart-a');
        const chartBContainer = document.getElementById('compare-chart-b');
        
        if (chartAContainer && chartBContainer) {
            this.chartA = this.createComparisonChart(chartAContainer, this.planetA);
            this.chartB = this.createComparisonChart(chartBContainer, this.planetB);
        }
    }
    
    createComparisonChart(container, planetName) {
        const data = this.planetData[planetName];
        
        const traces = [
            {
                x: data.time,
                y: data.amplitude,
                type: 'scatter',
                mode: 'lines',
                name: `${this.getPlanetDisplayName(planetName)} Activity`,
                line: { 
                    color: this.getPlanetColor(planetName),
                    width: 2 
                }
            },
            {
                x: data.events.map(e => e.time),
                y: data.events.map(e => e.amplitude),
                type: 'scatter',
                mode: 'markers',
                name: 'Events',
                marker: {
                    color: data.events.map(e => this.getMagnitudeColor(e.magnitude)),
                    size: data.events.map(e => 4 + e.magnitude),
                    symbol: 'circle'
                }
            }
        ];
        
        const layout = {
            title: {
                text: this.getPlanetDisplayName(planetName),
                font: { color: this.getPlanetColor(planetName), size: 14 }
            },
            xaxis: {
                title: 'Time (hours)',
                color: '#ffffff',
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                tickfont: { color: '#b4b4b8', size: 10 },
                titlefont: { color: '#b4b4b8', size: 11 }
            },
            yaxis: {
                title: 'Amplitude',
                color: '#ffffff',
                gridcolor: 'rgba(255, 255, 255, 0.1)',
                tickfont: { color: '#b4b4b8', size: 10 },
                titlefont: { color: '#b4b4b8', size: 11 }
            },
            plot_bgcolor: 'rgba(0, 0, 0, 0)',
            paper_bgcolor: 'rgba(0, 0, 0, 0)',
            font: { color: '#ffffff' },
            showlegend: false,
            margin: { l: 50, r: 30, t: 50, b: 50 }
        };
        
        const config = {
            displayModeBar: false,
            responsive: true
        };
        
        Plotly.newPlot(container, traces, layout, config);
        
        return {
            container: container,
            data: data,
            planet: planetName
        };
    }
    
    generateComparisonData() {
        const data = {};
        const planets = ['earth', 'mars', 'moon'];
        const timespan = 24; // 24 hours
        const points = 288; // Every 5 minutes
        
        planets.forEach(planet => {
            const planetChar = this.getPlanetCharacteristics(planet);
            const time = [];
            const amplitude = [];
            const events = [];
            
            for (let i = 0; i < points; i++) {
                const t = (i / points) * timespan;
                let amp = 0;
                
                // Base seismic noise with planetary characteristics
                amp += (Math.random() - 0.5) * planetChar.noiseLevel;
                amp += Math.sin(t * planetChar.frequency) * planetChar.noiseLevel * 0.5;
                
                // Add tidal effects for Moon
                if (planet === 'moon') {
                    amp += Math.sin(t * 0.5) * 0.02; // Tidal moonquakes
                }
                
                // Add thermal effects
                if (planet === 'moon' || planet === 'mars') {
                    const thermalCycle = Math.sin((t / 24) * 2 * Math.PI);
                    amp += thermalCycle * planetChar.thermalEffect;
                }
                
                // Random seismic events
                if (Math.random() < planetChar.eventRate / points) {
                    const eventMagnitude = Math.random() * planetChar.maxMagnitude + 1;
                    const eventAmp = eventMagnitude * planetChar.amplificationFactor;
                    
                    events.push({
                        time: t,
                        magnitude: eventMagnitude,
                        amplitude: eventAmp,
                        depth: Math.random() * planetChar.avgDepth + 10,
                        duration: 30 + eventMagnitude * 20,
                        planet: planet
                    });
                    
                    amp += eventAmp;
                }
                
                time.push(t);
                amplitude.push(amp);
            }
            
            data[planet] = { time, amplitude, events };
        });
        
        return data;
    }
    
    getPlanetCharacteristics(planet) {
        const characteristics = {
            earth: {
                noiseLevel: 0.1,
                frequency: 0.1,
                eventRate: 5,
                maxMagnitude: 8,
                amplificationFactor: 0.3,
                avgDepth: 300,
                thermalEffect: 0
            },
            mars: {
                noiseLevel: 0.05,
                frequency: 0.05,
                eventRate: 1,
                maxMagnitude: 5,
                amplificationFactor: 0.15,
                avgDepth: 150,
                thermalEffect: 0.01
            },
            moon: {
                noiseLevel: 0.02,
                frequency: 0.02,
                eventRate: 0.3,
                maxMagnitude: 4,
                amplificationFactor: 0.08,
                avgDepth: 50,
                thermalEffect: 0.005
            }
        };
        
        return characteristics[planet] || characteristics.earth;
    }
    
    getPlanetDisplayName(planet) {
        const names = {
            earth: 'Earth',
            mars: 'Mars',
            moon: 'Moon'
        };
        return names[planet] || planet;
    }
    
    getPlanetColor(planet) {
        const colors = {
            earth: '#00ffff',
            mars: '#f97316',
            moon: '#9ca3af'
        };
        return colors[planet] || '#ffffff';
    }
    
    getMagnitudeColor(magnitude) {
        if (magnitude < 2) return '#10b981';
        if (magnitude < 4) return '#f59e0b';
        if (magnitude < 6) return '#f97316';
        return '#ef4444';
    }
    
    setPlanetA(planetName) {
        this.planetA = planetName;
    // If not initialized yet, ignore (will be set during initialize)
    if (!this.initialized) return;
    this.updateViewerA();
    this.updateChartA();
    }
    
    setPlanetB(planetName) {
        this.planetB = planetName;
    if (!this.initialized) return;
    this.updateViewerB();
    this.updateChartB();
    }
    
    updateViewerA() {
        if (this.viewerA) {
            this.viewerA.loadPlanet(this.planetA);
        }
    }
    
    updateViewerB() {
        if (this.viewerB) {
            this.viewerB.loadPlanet(this.planetB);
        }
    }
    
    updateChartA() {
        if (this.chartA) {
            const container = this.chartA.container;
            try { if (window.Plotly) Plotly.purge(container); } catch(_) {}
            container.innerHTML = '';
            this.chartA = this.createComparisonChart(container, this.planetA);
        }
    }
    
    updateChartB() {
        if (this.chartB) {
            const container = this.chartB.container;
            try { if (window.Plotly) Plotly.purge(container); } catch(_) {}
            container.innerHTML = '';
            this.chartB = this.createComparisonChart(container, this.planetB);
        }
    }
    
    updateComparison() {
        this.updateViewerA();
        this.updateViewerB();
        this.updateChartA();
        this.updateChartB();
        this.generateComparisonInsights();
    }
    
    generateComparisonInsights() {
        const dataA = this.planetData[this.planetA];
        const dataB = this.planetData[this.planetB];
        
        const avgMagnitudeA = dataA.events.reduce((sum, e) => sum + e.magnitude, 0) / dataA.events.length;
        const avgMagnitudeB = dataB.events.reduce((sum, e) => sum + e.magnitude, 0) / dataB.events.length;
        
        const insights = {
            eventCount: {
                [this.planetA]: dataA.events.length,
                [this.planetB]: dataB.events.length
            },
            avgMagnitude: {
                [this.planetA]: avgMagnitudeA || 0,
                [this.planetB]: avgMagnitudeB || 0
            },
            maxEvent: {
                [this.planetA]: Math.max(...dataA.events.map(e => e.magnitude), 0),
                [this.planetB]: Math.max(...dataB.events.map(e => e.magnitude), 0)
            }
        };
        
        this.displayInsights(insights);
    }
    
    displayInsights(insights) {
        // Create or update insights panel
        let insightsPanel = document.getElementById('comparison-insights');
        
        if (!insightsPanel) {
            insightsPanel = document.createElement('div');
            insightsPanel.id = 'comparison-insights';
            insightsPanel.style.cssText = `
                position: fixed;
                bottom: 2rem;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 1.5rem;
                color: white;
                font-size: 0.9rem;
                z-index: 1000;
                max-width: 600px;
                text-align: center;
            `;
            document.body.appendChild(insightsPanel);
        }
        
        const planetAName = this.getPlanetDisplayName(this.planetA);
        const planetBName = this.getPlanetDisplayName(this.planetB);
        
        insightsPanel.innerHTML = `
            <h3 style="color: #00ffff; margin-bottom: 1rem;">Comparative Analysis</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; text-align: center;">
                <div>
                    <strong style="color: #f97316;">Event Count (24h)</strong><br>
                    <span style="color: ${this.getPlanetColor(this.planetA)}">${planetAName}: ${insights.eventCount[this.planetA]}</span><br>
                    <span style="color: ${this.getPlanetColor(this.planetB)}">${planetBName}: ${insights.eventCount[this.planetB]}</span>
                </div>
                <div>
                    <strong style="color: #8b5cf6;">Average Magnitude</strong><br>
                    <span style="color: ${this.getPlanetColor(this.planetA)}">${planetAName}: ${insights.avgMagnitude[this.planetA].toFixed(1)}</span><br>
                    <span style="color: ${this.getPlanetColor(this.planetB)}">${planetBName}: ${insights.avgMagnitude[this.planetB].toFixed(1)}</span>
                </div>
                <div>
                    <strong style="color: #10b981;">Maximum Event</strong><br>
                    <span style="color: ${this.getPlanetColor(this.planetA)}">${planetAName}: ${insights.maxEvent[this.planetA].toFixed(1)}</span><br>
                    <span style="color: ${this.getPlanetColor(this.planetB)}">${planetBName}: ${insights.maxEvent[this.planetB].toFixed(1)}</span>
                </div>
            </div>
        `;
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            if (insightsPanel) {
                insightsPanel.style.opacity = '0';
                setTimeout(() => {
                    document.body.removeChild(insightsPanel);
                }, 500);
            }
        }, 10000);
    }
    
    syncPlaybackMode(enabled) {
        this.syncPlayback = enabled;
        // Implementation for synchronized playback of events
    }
    
    handleResize() {
        this.viewerA?.handleResize?.();
        this.viewerB?.handleResize?.();
    }
    
    exportComparison() {
        const comparisonData = {
            planetA: {
                name: this.planetA,
                data: this.planetData[this.planetA]
            },
            planetB: {
                name: this.planetB,
                data: this.planetData[this.planetB]
            },
            generatedAt: new Date().toISOString(),
            analysisType: 'comparative_seismic_analysis'
        };
        
        const blob = new Blob([JSON.stringify(comparisonData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `planetary-comparison-${this.planetA}-vs-${this.planetB}-${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    destroy() {
        // Clean up viewers and charts
    if (this.viewerA) { this.viewerA.destroy(); this.viewerA = null; }
    if (this.viewerB) { this.viewerB.destroy(); this.viewerB = null; }
        
    if (this.chartA) { try { Plotly.purge(this.chartA.container); } catch(_) {} this.chartA.container.innerHTML = ''; this.chartA = null; }
    if (this.chartB) { try { Plotly.purge(this.chartB.container); } catch(_) {} this.chartB.container.innerHTML = ''; this.chartB = null; }
        
    // Also clear any leftover DOM nodes inside viewer containers
    const va = document.getElementById('compare-viewer-a');
    const vb = document.getElementById('compare-viewer-b');
    if (va) va.innerHTML = '';
    if (vb) vb.innerHTML = '';
        
        // Remove insights panel
        const insightsPanel = document.getElementById('comparison-insights');
        if (insightsPanel) {
            document.body.removeChild(insightsPanel);
        }
    this.initialized = false;
    }
}

// Simplified 3D viewer for comparison mode
class ComparisonViewer {
    constructor(container, planetName) {
        this.container = container;
        this.planetName = planetName;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.planet = null; // fallback sphere
        this.modelGroup = null; // loaded GLTF root
        this.loader = null;
        this.animationId = null;
        
        this.initialize();
    }
    
    initialize() {
        this.setupScene();
        this.loadPlanetModel(this.planetName).catch(() => {
            // Fallback to sphere on error
            this.createPlanet();
        });
        this.startAnimation();
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
    this.camera.position.set(0, 0, 2.5);
    this.camera.lookAt(0, 0, 0);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setClearColor(0x000000, 0);
        
    // Defensive: ensure container is empty before appending a new canvas
    while (this.container.firstChild) this.container.removeChild(this.container.firstChild);
    this.container.appendChild(this.renderer.domElement);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(-1, 1, 1);
        this.scene.add(directionalLight);
        
        this.loader = new THREE.GLTFLoader();
    }
    
    async loadPlanetModel(planetName) {
        const path = this.getModelPath(planetName);
        if (!path) throw new Error('No model path');
        // Clear previous
        this.clearModel();
        const gltf = await this.loader.loadAsync(path);
        const root = gltf.scene || gltf.scenes?.[0];
        if (!root) throw new Error('Invalid GLB');
        this.modelGroup = root;
        // Normalize size and center
        const box = new THREE.Box3().setFromObject(root);
        const size = new THREE.Vector3();
        box.getSize(size);
        const center = new THREE.Vector3();
        box.getCenter(center);
    root.position.sub(center); // center at origin
    root.position.y = 0; // ensure consistent Y alignment across models
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        const targetSize = 1.6; // fit within camera view comfortably
        const scale = targetSize / maxDim;
        root.scale.setScalar(scale);
    root.rotation.set(0, 0, 0);
        this.scene.add(root);
    }

    getModelPath(planet) {
        switch (planet) {
            case 'earth': return 'public/earth.glb';
            case 'mars':  return 'public/mars.glb';
            case 'moon':  return 'public/moon.glb';
            default: return null;
        }
    }

    createPlanet() {
        const geometry = new THREE.SphereGeometry(0.8, 32, 16);
        const material = new THREE.MeshPhongMaterial({
            color: this.getPlanetColor(),
            shininess: 30
        });
        
        this.planet = new THREE.Mesh(geometry, material);
        this.scene.add(this.planet);
    }
    
    getPlanetColor() {
        const colors = {
            earth: 0x4a90e2,
            mars: 0xe74c3c,
            moon: 0xbdc3c7
        };
        return colors[this.planetName] || 0xffffff;
    }
    
    loadPlanet(planetName) {
        this.planetName = planetName;
        // Try load GLB; on fail, fall back to recoloring sphere
        this.loadPlanetModel(planetName).catch(() => {
            if (!this.planet) this.createPlanet();
            if (this.planet) this.planet.material.color.setHex(this.getPlanetColor());
        });
    }
    
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            // Static models (no rotation) per request
            
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    handleResize() {
        if (!this.renderer || !this.camera) return;
        const w = this.container.clientWidth || 1;
        const h = this.container.clientHeight || 1;
        this.renderer.setSize(w, h);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }
    
    clearModel() {
        if (this.modelGroup) {
            this.scene.remove(this.modelGroup);
            this.disposeObject(this.modelGroup);
            this.modelGroup = null;
        }
        if (this.planet) {
            this.scene.remove(this.planet);
            this.disposeObject(this.planet);
            this.planet = null;
        }
    }
    
    disposeObject(obj) {
        obj.traverse?.((child) => {
            if (child.geometry) child.geometry.dispose?.();
            if (child.material) {
                if (Array.isArray(child.material)) child.material.forEach(m => m.dispose?.());
                else child.material.dispose?.();
            }
            if (child.texture) child.texture.dispose?.();
        });
    }
    
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.renderer) {
            this.container.removeChild(this.renderer.domElement);
            this.renderer.dispose();
        }
        this.clearModel();
    }
}
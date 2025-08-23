import { PlanetCarousel } from './PlanetCarousel.js';
import { PlanetDashboard } from './PlanetDashboard.js';
import { WaveformVisualizer } from './WaveformVisualizer.js';
import { Timeline } from './Timeline.js';
import { OverlayManager } from './OverlayManager.js';
import { CompareView } from './CompareView.js';
import { AnnotationManager } from './AnnotationManager.js';
import { ApiClient } from './apiClient.js';

class SeismoGuardApp {
    constructor() {
        this.currentView = 'home';
        this.currentPlanet = 'earth';
        this.isLoading = true;

        try {
            this.initializeComponents();
            this.setupEventListeners();
        } catch (err) {
            console.error('App initialization error:', err);
            this.showStartupError(err);
        } finally {
            // Never leave the loading screen stuck
            this.hideLoadingScreen();
        }
    }
    
    initializeComponents() {
        // Initialize optional backend client (non-disruptive)
        try {
            this.api = new ApiClient("http://127.0.0.1:5000");
            // Kick a tiny health ping for visibility (non-blocking)
            this.api.health().then(h => console.log('[backend] health ok', h?.status)).catch(() => {
                console.warn('[backend] not reachable; operating in offline mode');
            });
        } catch (_) {
            this.api = null; // graceful fallback
        }

    this.carousel = new PlanetCarousel('#planet-carousel');
    this.dashboard = new PlanetDashboard('#planet-viewer');
    this.waveform = new WaveformVisualizer('#waveform-chart', this.api);
        this.timeline = new Timeline('#timeline-container');
        this.overlay = new OverlayManager();
        this.compare = new CompareView();
        this.annotations = new AnnotationManager();
        
        // Initialize carousel (never block the app)
        this.carousel.initialize().then(() => {
            console.log('Carousel initialized successfully');
        }).catch(error => {
            console.error('Failed to initialize carousel:', error);
        }).finally(() => {
            // Ensure loading screen is dismissed even if carousel struggles
            this.hideLoadingScreen();
        });

        // Scroll reveal on desktop: mark key panels and observe
        try {
            document.querySelectorAll('.left-panel, .right-panel, .timeline-panel').forEach(el => {
                el.classList.add('reveal');
            });
            const io = new IntersectionObserver((entries) => {
                entries.forEach(e => {
                    if (e.isIntersecting) {
                        e.target.classList.add('is-visible');
                        // Nudge charts/viewers to resize when they appear
                        if (e.target.querySelector('#waveform-chart')) {
                            this.waveform?.handleResize();
                        }
                        if (e.target.querySelector('#planet-viewer')) {
                            this.dashboard?.handleResize?.();
                        }
                    }
                });
            }, { threshold: 0.15 });
            document.querySelectorAll('.reveal').forEach(el => io.observe(el));
            this._revealObserver = io;
        } catch(_) {}
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const view = e.target.getAttribute('data-view');
                this.switchView(view);
            });
        });
        
        // Mobile menu toggle
        const mobileToggle = document.querySelector('.mobile-menu-toggle');
        const navMenu = document.querySelector('.nav-menu');
        mobileToggle?.addEventListener('click', () => {
            navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
        });
        
        // Carousel interactions
        const exploreButton = document.getElementById('explore-button');
        exploreButton?.addEventListener('click', () => {
            this.switchToDashboard();
        });
        
        // Dashboard controls
        const backButton = document.getElementById('back-to-carousel');
        backButton?.addEventListener('click', () => {
            this.switchView('home');
        });
        
        const toggleMarkers = document.getElementById('toggle-markers');
        toggleMarkers?.addEventListener('click', () => {
            this.dashboard.toggleEventMarkers();
            toggleMarkers.classList.toggle('active');
        });
        
        const dataMode = document.getElementById('data-mode');
        dataMode?.addEventListener('change', (e) => {
            this.waveform.setDataMode(e.target.value);
        });
        
        const exportData = document.getElementById('export-data');
        exportData?.addEventListener('click', () => {
            this.exportCurrentData();
        });
        
        // Timeline controls
        const playTimeline = document.getElementById('play-timeline');
        const resetTimeline = document.getElementById('reset-timeline');
        
        playTimeline?.addEventListener('click', () => {
            this.timeline.togglePlayback();
            playTimeline.textContent = this.timeline.isPlaying ? '⏸ Pause' : '▶ Play';
        });
        
        resetTimeline?.addEventListener('click', () => {
            this.timeline.reset();
            playTimeline.textContent = '▶ Play';
        });
        
        // Planet selection from carousel
        this.carousel.onPlanetSelect = (planet) => {
            this.selectPlanet(planet);
        };
        
        // Seismic event selection
        this.waveform.onEventSelect = (event) => {
            this.showEventDetails(event);
        };
        
        this.timeline.onEventSelect = (event) => {
            this.showEventDetails(event);
            this.waveform.highlightEvent(event);
        };
        
        // Compare view controls
        const comparePlanetA = document.getElementById('compare-planet-a');
        const comparePlanetB = document.getElementById('compare-planet-b');
        
        comparePlanetA?.addEventListener('change', (e) => {
            this.compare.setPlanetA(e.target.value);
        });
        
        comparePlanetB?.addEventListener('change', (e) => {
            this.compare.setPlanetB(e.target.value);
        });

        // When waveform ingests backend data, reflect into timeline
        window.addEventListener('waveform:dataUpdated', (e) => {
            try {
                if (this.currentView === 'dashboard') {
                    this.timeline.loadExternalEvents(e.detail?.events || []);
                }
            } catch(_) {}
        });
    }
    
    switchView(viewName) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-view="${viewName}"]`)?.classList.add('active');
        
        // Hide all views
        document.querySelectorAll('.view-section').forEach(section => {
            section.classList.remove('active');
        });
        
        // Show target view
        const targetView = this.getViewElement(viewName);
        if (targetView) {
            targetView.classList.add('active');
            this.currentView = viewName;
            
            // Initialize view-specific components
            switch (viewName) {
                case 'home':
                    this.carousel.startAnimation();
                    break;
                case 'compare':
                    // Re-init compare view cleanly to avoid duplicate canvases/charts
                    try { this.compare.destroy(); } catch(_) {}
                    this.compare.initialize();
                    break;
                case 'data':
                    this.showDataExplorer();
                    break;
                case 'about':
                    this.showAboutInfo();
                    break;
            }
        }
    }
    
    getViewElement(viewName) {
        switch (viewName) {
            case 'home': return document.getElementById('carousel-view');
            case 'dashboard': return document.getElementById('dashboard-view');
            case 'compare': return document.getElementById('compare-view');
            default: return document.getElementById('carousel-view');
        }
    }
    
    switchToDashboard() {
        this.switchView('dashboard');
        this.dashboard.loadPlanet(this.currentPlanet);
        this.waveform.loadPlanet(this.currentPlanet).then(() => {
            try {
                // If waveform has backend events for current planet, reflect into timeline
                const d = this.waveform.seismicData[this.currentPlanet][this.waveform.dataMode];
                if (Array.isArray(d.events) && d.events.length) {
                    this.timeline.loadExternalEvents(d.events);
                }
            } catch(_) {}
        });
        this.timeline.loadPlanet(this.currentPlanet);
    // Ensure user lands at top of dashboard for stacked layout
    try { document.getElementById('dashboard-view')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); } catch(_) {}
        
        // Show educational overlay for the selected planet
        this.overlay.showPlanetInfo(this.currentPlanet);
    }
    
    selectPlanet(planetName) {
        this.currentPlanet = planetName;
        
        // Update carousel info
        const planetInfo = this.getPlanetInfo(planetName);
        document.getElementById('current-planet-name').textContent = planetInfo.name;
        document.getElementById('current-planet-desc').textContent = planetInfo.description;
        
        // Update dashboard title
        document.getElementById('dashboard-planet-name').textContent = planetInfo.name;
        
        // If in dashboard view, update components
        if (this.currentView === 'dashboard') {
            this.dashboard.loadPlanet(planetName);
            this.waveform.loadPlanet(planetName);
            this.timeline.loadPlanet(planetName);
        }
    }
    
    getPlanetInfo(planetName) {
        const planetData = {
            earth: {
                name: 'Earth',
                description: 'Our home planet experiences thousands of earthquakes daily, from subtle tremors to major seismic events that reshape the landscape.'
            },
            mars: {
                name: 'Mars',
                description: 'The Red Planet\'s marsquakes are less frequent but longer-lasting, providing insights into the planet\'s internal structure and geological activity.'
            },
            moon: {
                name: 'Moon',
                description: 'Our lunar companion experiences moonquakes caused by thermal expansion, meteorite impacts, and tidal forces from Earth\'s gravity.'
            }
        };
        
        return planetData[planetName] || planetData.earth;
    }
    
    showEventDetails(event) {
        // Update metadata display
        document.getElementById('latest-magnitude').textContent = event.magnitude || '--';
        document.getElementById('latest-depth').textContent = event.depth ? `${event.depth} km` : '--';
        document.getElementById('latest-duration').textContent = event.duration ? `${event.duration}s` : '--';
        
        // Show event on planet surface
        this.dashboard.highlightEvent(event);
        
        // Open annotations panel if event has notes
        if (event.hasAnnotations) {
            this.annotations.showPanel(event);
        }
    }
    
    exportCurrentData() {
        const exportData = {
            planet: this.currentPlanet,
            timestamp: new Date().toISOString(),
            waveformData: this.waveform.getCurrentData(),
            timelineEvents: this.timeline.getCurrentEvents(),
            annotations: this.annotations.getAnnotations()
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `seismic-data-${this.currentPlanet}-${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
    }
    
    showDataExplorer() {
        // Show data exploration interface
        this.overlay.showDataExplorer();
    }
    
    showAboutInfo() {
        // Show about information
        this.overlay.showAboutInfo();
    }

    showStartupError(err) {
        console.error('Startup error:', err);
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            const text = loadingScreen.querySelector('.loading-text');
            if (text) text.textContent = 'Startup error – check console logs';
            const spinner = loadingScreen.querySelector('.loading-spinner');
            if (spinner) spinner.style.borderTopColor = '#ef4444';
        }
    }

    hideLoadingScreen() {
        setTimeout(() => {
            const loadingScreen = document.getElementById('loading-screen');
            if (loadingScreen) {
                loadingScreen.style.opacity = '0';
                setTimeout(() => {
                    loadingScreen.style.display = 'none';
                }, 500);
            }
        }, 2000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.seismoGuardApp = new SeismoGuardApp();
});

// Handle window resize
window.addEventListener('resize', () => {
    if (window.seismoGuardApp) {
        window.seismoGuardApp.carousel?.handleResize();
        window.seismoGuardApp.dashboard?.handleResize();
        window.seismoGuardApp.waveform?.handleResize();
    window.seismoGuardApp.compare?.handleResize?.();
    }
});
export class OverlayManager {
    constructor() {
        this.overlay = document.getElementById('educational-overlay');
        this.overlayTitle = document.getElementById('overlay-title');
        this.overlayText = document.getElementById('overlay-text');
        this.overlayIcon = document.getElementById('overlay-icon');
        this.closeButton = this.overlay?.querySelector('.overlay-close');
        
        this.setupEventListeners();
        
        this.planetFacts = {
            earth: {
                title: "Earth Seismology",
                text: "Earth experiences over 500,000 detectable earthquakes annually. The planet's tectonic plates create a dynamic system where energy is constantly building and releasing. Most earthquakes occur along plate boundaries, with the Pacific Ring of Fire being the most seismically active region.",
                icon: "🌍"
            },
            mars: {
                title: "Martian Marsquakes",
                text: "Mars experiences unique seismic activity detected by NASA's InSight lander. Marsquakes are generally weaker but longer-lasting than Earth's earthquakes due to Mars' thinner atmosphere and different geological structure. The planet's seismic activity provides insights into its internal composition and geological history.",
                icon: "🔴"
            },
            moon: {
                title: "Lunar Moonquakes",
                text: "The Moon experiences four types of moonquakes: deep moonquakes, shallow moonquakes, thermal moonquakes, and impact moonquakes. Apollo missions detected over 13,000 seismic events. Unlike Earth, the Moon's seismic waves can continue for over an hour due to the lack of water to dampen vibrations.",
                icon: "🌙"
            }
        };
    }
    
    setupEventListeners() {
        // Close button
        this.closeButton?.addEventListener('click', () => {
            this.hideOverlay();
        });
        
        // Click outside to close
        this.overlay?.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.hideOverlay();
            }
        });
        
        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.overlay?.classList.contains('active')) {
                this.hideOverlay();
            }
        });
    }
    
    showPlanetInfo(planetName) {
        const info = this.planetFacts[planetName];
        if (!info || !this.overlay) return;
        
        this.overlayTitle.textContent = info.title;
        this.overlayText.textContent = info.text;
        this.overlayIcon.textContent = info.icon;
        
        this.showOverlay();
        
        // Auto-hide after 8 seconds
        setTimeout(() => {
            this.hideOverlay();
        }, 8000);
    }
    
    showDataExplorer() {
        if (!this.overlay) return;
        
        this.overlayTitle.textContent = "Seismic Data Explorer";
        this.overlayText.innerHTML = `
            <div style="text-align: left; line-height: 1.8;">
                <h3 style="color: #00ffff; margin-bottom: 1rem;">Available Datasets</h3>
                
                <div style="margin-bottom: 1.5rem;">
                    <strong>🌍 Earth Seismic Network</strong><br>
                    Global seismograph network data from USGS and international monitoring stations. 
                    Real-time earthquake detection and historical records dating back decades.
                </div>
                
                <div style="margin-bottom: 1.5rem;">
                    <strong>🔴 Mars InSight Mission</strong><br>
                    Seismic data from NASA's InSight lander on Mars (2018-2022). 
                    Over 1,300 marsquakes detected, providing unprecedented insights into Martian geology.
                </div>
                
                <div style="margin-bottom: 1.5rem;">
                    <strong>🌙 Apollo Lunar Seismic Array</strong><br>
                    Historical moonquake data from Apollo missions 12, 14, 15, and 16 (1969-1977). 
                    Passive Seismic Experiment data revealing lunar internal structure.
                </div>
                
                <p style="color: #8b5cf6; font-style: italic;">
                    Toggle between real-time monitoring and historical analysis modes to explore different aspects of planetary seismology.
                </p>
            </div>
        `;
        this.overlayIcon.textContent = "📊";
        
        this.showOverlay();
    }
    
    showAboutInfo() {
        if (!this.overlay) return;
        
        this.overlayTitle.textContent = "About SeismoGuard";
        this.overlayText.innerHTML = `
            <div style="text-align: left; line-height: 1.8;">
                <p style="margin-bottom: 1.5rem;">
                    SeismoGuard is an interactive planetary seismology visualization platform that brings 
                    the fascinating world of planetary earthquakes to your fingertips.
                </p>
                
                <h3 style="color: #00ffff; margin-bottom: 1rem;">Features</h3>
                <ul style="margin-bottom: 1.5rem; padding-left: 1.5rem;">
                    <li>3D planetary carousel with Earth, Mars, and Moon</li>
                    <li>Real-time seismic waveform visualization</li>
                    <li>Interactive timeline for event exploration</li>
                    <li>Comparative analysis between planets</li>
                    <li>Educational overlays and annotations</li>
                </ul>
                
                <h3 style="color: #8b5cf6; margin-bottom: 1rem;">Educational Mission</h3>
                <p style="margin-bottom: 1.5rem;">
                    Our mission is to make planetary science accessible and engaging, inspiring 
                    curiosity about the seismic forces that shape worlds across our solar system.
                </p>
                
                <p style="color: #f97316; font-style: italic;">
                    Explore. Discover. Understand the seismic symphony of the cosmos.
                </p>
            </div>
        `;
        this.overlayIcon.textContent = "🚀";
        
        this.showOverlay();
    }
    
    showEventDetails(event) {
        if (!this.overlay || !event) return;
        
        this.overlayTitle.textContent = `Seismic Event Details`;
        this.overlayText.innerHTML = `
            <div style="text-align: left; line-height: 1.8;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem;">
                    <div>
                        <strong style="color: #00ffff;">Magnitude:</strong><br>
                        ${event.magnitude?.toFixed(1) || 'Unknown'}
                    </div>
                    <div>
                        <strong style="color: #8b5cf6;">Depth:</strong><br>
                        ${event.depth?.toFixed(0) || 'Unknown'} km
                    </div>
                    <div>
                        <strong style="color: #f97316;">Duration:</strong><br>
                        ${event.duration?.toFixed(0) || 'Unknown'} seconds
                    </div>
                    <div>
                        <strong style="color: #10b981;">Planet:</strong><br>
                        ${event.planet?.charAt(0).toUpperCase() + event.planet?.slice(1) || 'Unknown'}
                    </div>
                </div>
                
                ${event.latitude && event.longitude ? `
                    <div style="margin-bottom: 1.5rem;">
                        <strong style="color: #00ffff;">Location:</strong><br>
                        Latitude: ${event.latitude.toFixed(2)}°<br>
                        Longitude: ${event.longitude.toFixed(2)}°
                    </div>
                ` : ''}
                
                <div style="background: rgba(0, 255, 255, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #00ffff;">
                    <strong>Seismic Analysis:</strong><br>
                    This event registered as a ${this.getMagnitudeDescription(event.magnitude)} 
                    ${event.planet}quake. The seismic waves provide valuable data about the internal 
                    structure and geological processes of ${event.planet === 'earth' ? 'our planet' : event.planet}.
                </div>
            </div>
        `;
        this.overlayIcon.textContent = "📊";
        
        this.showOverlay();
    }
    
    getMagnitudeDescription(magnitude) {
        if (magnitude < 2) return "micro";
        if (magnitude < 4) return "minor";
        if (magnitude < 6) return "moderate";
        if (magnitude < 7) return "strong";
        if (magnitude < 8) return "major";
        return "great";
    }
    
    showOverlay() {
        if (this.overlay) {
            this.overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    }
    
    hideOverlay() {
        if (this.overlay) {
            this.overlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    }
    
    showTooltip(element, content) {
        // Create temporary tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'custom-tooltip';
        tooltip.innerHTML = content;
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: #ffffff;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            z-index: 10000;
            pointer-events: none;
            border: 1px solid rgba(0, 255, 255, 0.3);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        `;
        
        document.body.appendChild(tooltip);
        
        // Position tooltip
        const rect = element.getBoundingClientRect();
        tooltip.style.left = `${rect.left + rect.width / 2}px`;
        tooltip.style.top = `${rect.top - tooltip.offsetHeight - 8}px`;
        tooltip.style.transform = 'translateX(-50%)';
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            document.body.removeChild(tooltip);
        }, 3000);
        
        return tooltip;
    }
    
    showLoadingMessage(message) {
        if (!this.overlay) return;
        
        this.overlayTitle.textContent = "Loading...";
        this.overlayText.textContent = message || "Processing seismic data...";
        this.overlayIcon.innerHTML = `
            <div style="width: 40px; height: 40px; border: 3px solid rgba(0, 255, 255, 0.3); 
                        border-top: 3px solid #00ffff; border-radius: 50%; 
                        animation: spin 1s linear infinite;"></div>
        `;
        
        this.showOverlay();
    }
    
    showSuccessMessage(title, message) {
        if (!this.overlay) return;
        
        this.overlayTitle.textContent = title;
        this.overlayText.textContent = message;
        this.overlayIcon.textContent = "✅";
        
        this.showOverlay();
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            this.hideOverlay();
        }, 3000);
    }
    
    showErrorMessage(title, message) {
        if (!this.overlay) return;
        
        this.overlayTitle.textContent = title;
        this.overlayText.textContent = message;
        this.overlayIcon.textContent = "❌";
        
        this.showOverlay();
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideOverlay();
        }, 5000);
    }
}
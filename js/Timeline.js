export class Timeline {
    constructor(containerId) {
        this.container = document.querySelector(containerId);
        this.currentPlanet = null;
        this.events = [];
        this.isPlaying = false;
        this.currentTime = 0;
        this.playbackSpeed = 1;
        this.animationId = null;
        this.onEventSelect = null;
        
        this.initialize();
    }
    
    initialize() {
        if (!this.container) return;
        
        this.createTimelineStructure();
        this.setupInteractions();
    }
    
    createTimelineStructure() {
        this.container.innerHTML = `
            <div class="timeline-header">
                <div class="timeline-info">
                    <span class="timeline-duration">Duration: <span id="total-duration">--</span></span>
                    <span class="timeline-events">Events: <span id="event-count">--</span></span>
                </div>
                <div class="timeline-speed">
                    <label>Speed: </label>
                    <select id="playback-speed">
                        <option value="0.25">0.25x</option>
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="2">2x</option>
                        <option value="4">4x</option>
                    </select>
                </div>
            </div>
            <div class="timeline-track">
                <div class="timeline-background"></div>
                <div class="timeline-progress"></div>
                <div class="timeline-handle"></div>
                <div class="timeline-events-layer"></div>
            </div>
            <div class="timeline-time-labels"></div>
        `;
        
        this.addTimelineStyles();
    }
    
    addTimelineStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .timeline-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                font-size: 0.9rem;
                color: #b4b4b8;
            }
            
            .timeline-info {
                display: flex;
                gap: 2rem;
            }
            
            .timeline-speed select {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: #ffffff;
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.8rem;
            }
            
            .timeline-track {
                position: relative;
                height: 60px;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 8px;
                margin-bottom: 0.5rem;
                cursor: pointer;
                overflow: hidden;
            }
            
            .timeline-background {
                position: absolute;
                top: 50%;
                left: 0;
                right: 0;
                height: 4px;
                background: rgba(255, 255, 255, 0.2);
                transform: translateY(-50%);
            }
            
            .timeline-progress {
                position: absolute;
                top: 50%;
                left: 0;
                height: 4px;
                background: linear-gradient(90deg, #00ffff, #8b5cf6);
                transform: translateY(-50%);
                width: 0%;
                transition: width 0.1s ease;
            }
            
            .timeline-handle {
                position: absolute;
                top: 50%;
                left: 0%;
                width: 16px;
                height: 16px;
                background: #00ffff;
                border: 2px solid #ffffff;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                cursor: grab;
                box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
                transition: left 0.1s ease;
            }
            
            .timeline-handle:active {
                cursor: grabbing;
                transform: translate(-50%, -50%) scale(1.2);
            }
            
            .timeline-events-layer {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                pointer-events: none;
            }
            
            .timeline-event-marker {
                position: absolute;
                top: 50%;
                width: 3px;
                height: 30px;
                background: linear-gradient(to bottom, transparent, var(--event-color), transparent);
                transform: translateY(-50%);
                pointer-events: all;
                cursor: pointer;
                border-radius: 2px;
                transition: all 0.2s ease;
            }
            
            .timeline-event-marker:hover {
                height: 45px;
                box-shadow: 0 0 10px var(--event-color);
                z-index: 10;
            }
            
            .timeline-event-marker.major {
                width: 5px;
                height: 40px;
            }
            
            .timeline-time-labels {
                display: flex;
                justify-content: space-between;
                font-size: 0.8rem;
                color: #9ca3af;
                padding: 0 0.5rem;
            }
            
            .event-tooltip {
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0, 0, 0, 0.9);
                color: #ffffff;
                padding: 0.5rem;
                border-radius: 6px;
                font-size: 0.8rem;
                white-space: nowrap;
                margin-bottom: 8px;
                opacity: 0;
                visibility: hidden;
                transition: all 0.2s ease;
                z-index: 20;
                border: 1px solid rgba(0, 255, 255, 0.3);
            }
            
            .timeline-event-marker:hover .event-tooltip {
                opacity: 1;
                visibility: visible;
            }
            
            .event-tooltip::after {
                content: '';
                position: absolute;
                top: 100%;
                left: 50%;
                border: 4px solid transparent;
                border-top-color: rgba(0, 0, 0, 0.9);
                transform: translateX(-50%);
            }
        `;
        
        document.head.appendChild(style);
    }
    
    setupInteractions() {
        const track = this.container.querySelector('.timeline-track');
        const handle = this.container.querySelector('.timeline-handle');
        const speedSelect = this.container.querySelector('#playback-speed');
        
        let isDragging = false;
        
        // Speed control
        speedSelect?.addEventListener('change', (e) => {
            this.playbackSpeed = parseFloat(e.target.value);
        });
        
        // Track click
        track?.addEventListener('click', (e) => {
            if (!isDragging) {
                const rect = track.getBoundingClientRect();
                const progress = (e.clientX - rect.left) / rect.width;
                this.seekToProgress(progress);
            }
        });
        
        // Handle drag
        const startDrag = (e) => {
            isDragging = true;
            document.addEventListener('mousemove', onDrag);
            document.addEventListener('mouseup', stopDrag);
            document.addEventListener('touchmove', onDrag);
            document.addEventListener('touchend', stopDrag);
        };
        
        const onDrag = (e) => {
            if (!isDragging) return;
            
            const clientX = e.clientX || e.touches[0].clientX;
            const rect = track.getBoundingClientRect();
            const progress = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
            this.seekToProgress(progress);
        };
        
        const stopDrag = () => {
            isDragging = false;
            document.removeEventListener('mousemove', onDrag);
            document.removeEventListener('mouseup', stopDrag);
            document.removeEventListener('touchmove', onDrag);
            document.removeEventListener('touchend', stopDrag);
        };
        
        handle?.addEventListener('mousedown', startDrag);
        handle?.addEventListener('touchstart', startDrag);
    }
    
    loadPlanet(planetName) {
        this.currentPlanet = planetName;
        this.events = this.generateEventsForPlanet(planetName);
        this.currentTime = 0;
        this.updateDisplay();
        this.renderEvents();
        this.updateTimeLabels();
    }
    
    generateEventsForPlanet(planetName) {
        const events = [];
        const timespan = 3600; // 1 hour in seconds
        
        const planetCharacteristics = {
            earth: { eventRate: 0.3, maxMagnitude: 8, avgDepth: 300 },
            mars: { eventRate: 0.1, maxMagnitude: 5, avgDepth: 150 },
            moon: { eventRate: 0.05, maxMagnitude: 3, avgDepth: 50 }
        };
        
        const char = planetCharacteristics[planetName] || planetCharacteristics.earth;
        
        // Generate events throughout the timespan
        for (let i = 0; i < timespan; i += 60) { // Every minute
            if (Math.random() < char.eventRate) {
                const time = i + Math.random() * 60;
                const magnitude = Math.random() * char.maxMagnitude + 1;
                
                events.push({
                    id: `timeline_${planetName}_${time}_${Math.random()}`,
                    time: time,
                    magnitude: magnitude,
                    depth: Math.random() * char.avgDepth + 10,
                    duration: 10 + magnitude * 20 + Math.random() * 100,
                    planet: planetName,
                    timestamp: Date.now() - (timespan - time) * 1000,
                    latitude: (Math.random() - 0.5) * 180,
                    longitude: (Math.random() - 0.5) * 360
                });
            }
        }
        
        // Sort events by time
        events.sort((a, b) => a.time - b.time);
        
        return events;
    }
    
    renderEvents() {
        const eventsLayer = this.container.querySelector('.timeline-events-layer');
        if (!eventsLayer) return;
        
        eventsLayer.innerHTML = '';
        const timespan = Math.max(3600, ...this.events.map(e => e.time));
        
        this.events.forEach(event => {
            const position = (event.time / timespan) * 100;
            const color = this.getMagnitudeColor(event.magnitude);
            const isMajor = event.magnitude > 5;
            
            const marker = document.createElement('div');
            marker.className = `timeline-event-marker ${isMajor ? 'major' : ''}`;
            marker.style.left = `${position}%`;
            marker.style.setProperty('--event-color', color);
            
            // Add tooltip
            const tooltip = document.createElement('div');
            tooltip.className = 'event-tooltip';
            tooltip.innerHTML = `
                <strong>Magnitude ${event.magnitude.toFixed(1)}</strong><br>
                Depth: ${event.depth.toFixed(0)} km<br>
                Duration: ${event.duration.toFixed(0)}s<br>
                Time: ${this.formatTime(event.time)}
            `;
            marker.appendChild(tooltip);
            
            // Click event
            marker.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectEvent(event);
            });
            
            eventsLayer.appendChild(marker);
        });
        
        // Update info
        const eventCount = this.container.querySelector('#event-count');
        const totalDuration = this.container.querySelector('#total-duration');
        
        if (eventCount) eventCount.textContent = this.events.length;
        if (totalDuration) totalDuration.textContent = this.formatTime(timespan);
    }
    
    updateTimeLabels() {
        const labelsContainer = this.container.querySelector('.timeline-time-labels');
        if (!labelsContainer) return;
        
        const timespan = Math.max(3600, ...this.events.map(e => e.time));
        const labelCount = 6;
        
        labelsContainer.innerHTML = '';
        
        for (let i = 0; i <= labelCount; i++) {
            const time = (i / labelCount) * timespan;
            const label = document.createElement('span');
            label.textContent = this.formatTime(time);
            labelsContainer.appendChild(label);
        }
    }
    
    getMagnitudeColor(magnitude) {
        if (magnitude < 2) return '#10b981'; // Green
        if (magnitude < 4) return '#f59e0b'; // Yellow
        if (magnitude < 6) return '#f97316'; // Orange
        return '#ef4444'; // Red
    }
    
    formatTime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        } else {
            return `${minutes}:${secs.toString().padStart(2, '0')}`;
        }
    }
    
    seekToProgress(progress) {
        const timespan = Math.max(3600, ...this.events.map(e => e.time));
        this.currentTime = progress * timespan;
        this.updateDisplay();
        
        // Find nearest event
        const nearestEvent = this.findNearestEvent(this.currentTime);
        if (nearestEvent) {
            this.selectEvent(nearestEvent);
        }
    }
    
    findNearestEvent(time) {
        let nearest = null;
        let minDistance = Infinity;
        
        this.events.forEach(event => {
            const distance = Math.abs(event.time - time);
            if (distance < minDistance) {
                minDistance = distance;
                nearest = event;
            }
        });
        
        return minDistance < 30 ? nearest : null; // 30 second tolerance
    }
    
    selectEvent(event) {
        this.currentTime = event.time;
        this.updateDisplay();
        
        if (this.onEventSelect) {
            this.onEventSelect(event);
        }
        
        // Highlight selected event
        const markers = this.container.querySelectorAll('.timeline-event-marker');
        markers.forEach(marker => marker.classList.remove('selected'));
        
        const eventIndex = this.events.indexOf(event);
        if (eventIndex !== -1 && markers[eventIndex]) {
            markers[eventIndex].classList.add('selected');
        }
    }
    
    updateDisplay() {
        const timespan = Math.max(3600, ...this.events.map(e => e.time));
        const progress = this.currentTime / timespan;
        
        const progressBar = this.container.querySelector('.timeline-progress');
        const handle = this.container.querySelector('.timeline-handle');
        
        if (progressBar) progressBar.style.width = `${progress * 100}%`;
        if (handle) handle.style.left = `${progress * 100}%`;
    }
    
    togglePlayback() {
        this.isPlaying = !this.isPlaying;
        
        if (this.isPlaying) {
            this.startPlayback();
        } else {
            this.stopPlayback();
        }
        
        return this.isPlaying;
    }
    
    startPlayback() {
        if (this.animationId) return;
        
        const timespan = Math.max(3600, ...this.events.map(e => e.time));
        const startTime = Date.now();
        const startProgress = this.currentTime / timespan;
        
        const animate = () => {
            if (!this.isPlaying) return;
            
            const elapsed = (Date.now() - startTime) / 1000 * this.playbackSpeed;
            const newProgress = startProgress + (elapsed / timespan);
            
            if (newProgress >= 1) {
                this.currentTime = timespan;
                this.updateDisplay();
                this.isPlaying = false;
                return;
            }
            
            this.currentTime = newProgress * timespan;
            this.updateDisplay();
            
            // Check for events at current time
            const activeEvents = this.events.filter(event => 
                Math.abs(event.time - this.currentTime) < 5 // 5 second window
            );
            
            activeEvents.forEach(event => {
                if (this.onEventSelect) {
                    this.onEventSelect(event);
                }
            });
            
            this.animationId = requestAnimationFrame(animate);
        };
        
        this.animationId = requestAnimationFrame(animate);
    }
    
    stopPlayback() {
        this.isPlaying = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    reset() {
        this.stopPlayback();
        this.currentTime = 0;
        this.updateDisplay();
        
        // Clear event selections
        const markers = this.container.querySelectorAll('.timeline-event-marker');
        markers.forEach(marker => marker.classList.remove('selected'));
    }
    
    getCurrentEvents() {
        return this.events.map(event => ({
            ...event,
            planet: this.currentPlanet,
            exportedAt: new Date().toISOString()
        }));
    }
    
    destroy() {
        this.stopPlayback();
        
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}
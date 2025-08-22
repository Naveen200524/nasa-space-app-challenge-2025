export class PlanetDashboard {
    constructor(containerId) {
        this.container = document.querySelector(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.planet = null;
        this.eventMarkers = [];
        this.showMarkers = true;
        this.currentPlanet = null;
        this.animationId = null;
        
        this.seismicEvents = this.generateMockSeismicEvents();
        
        this.initialize();
    }
    
    initialize() {
        if (!this.container) return;
        
        this.setupScene();
        this.setupLighting();
        this.setupControls();
        this.startAnimation();
    }
    
    setupScene() {
        // Scene
        this.scene = new THREE.Scene();
        
        // Camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 3);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true, 
            alpha: true 
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x000000, 0);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(-5, 3, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Point light for rim lighting
        const pointLight = new THREE.PointLight(0x00ffff, 0.5, 10);
        pointLight.position.set(0, 0, 3);
        this.scene.add(pointLight);
    }
    
    setupControls() {
        // Mouse interaction for planet rotation
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        
        const onMouseDown = (event) => {
            isDragging = true;
            previousMousePosition = {
                x: event.clientX,
                y: event.clientY
            };
        };
        
        const onMouseMove = (event) => {
            if (!isDragging || !this.planet) return;
            
            const deltaMove = {
                x: event.clientX - previousMousePosition.x,
                y: event.clientY - previousMousePosition.y
            };
            
            this.planet.rotation.y += deltaMove.x * 0.01;
            this.planet.rotation.x += deltaMove.y * 0.01;
            
            previousMousePosition = {
                x: event.clientX,
                y: event.clientY
            };
        };
        
        const onMouseUp = () => {
            isDragging = false;
        };
        
        this.container.addEventListener('mousedown', onMouseDown);
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
        
        // Touch support
        this.container.addEventListener('touchstart', (event) => {
            if (event.touches.length === 1) {
                onMouseDown({
                    clientX: event.touches[0].clientX,
                    clientY: event.touches[0].clientY
                });
            }
        });
        
        this.container.addEventListener('touchmove', (event) => {
            if (event.touches.length === 1) {
                event.preventDefault();
                onMouseMove({
                    clientX: event.touches[0].clientX,
                    clientY: event.touches[0].clientY
                });
            }
        });
        
        this.container.addEventListener('touchend', onMouseUp);
        
        // Zoom with mouse wheel
        this.container.addEventListener('wheel', (event) => {
            event.preventDefault();
            const delta = event.deltaY * 0.001;
            this.camera.position.z = Math.max(2, Math.min(8, this.camera.position.z + delta));
        });
    }
    
    loadPlanet(planetName) {
        this.currentPlanet = planetName;
        
        // Remove existing planet
        if (this.planet) {
            this.scene.remove(this.planet);
            this.planet.geometry.dispose();
            this.planet.material.dispose();
        }
        
        // Clear existing markers
        this.clearEventMarkers();
        
        // Create new planet
        this.createPlanet(planetName);
        
        // Add seismic event markers
        if (this.showMarkers) {
            this.addEventMarkers(planetName);
        }
    }
    
    createPlanet(planetName) {
        const planetData = this.getPlanetData(planetName);
        
        const geometry = new THREE.SphereGeometry(1.2, 64, 32);
        const material = new THREE.MeshPhongMaterial({
            map: planetData.texture,
            shininess: planetData.shininess,
            transparent: true,
            opacity: 0.95
        });
        
        this.planet = new THREE.Mesh(geometry, material);
        this.planet.castShadow = true;
        this.planet.receiveShadow = true;
        
        // Add atmosphere for Earth
        if (planetName === 'earth') {
            const atmosphereGeometry = new THREE.SphereGeometry(1.25, 64, 32);
            const atmosphereMaterial = new THREE.MeshBasicMaterial({
                color: 0x87ceeb,
                transparent: true,
                opacity: 0.15,
                side: THREE.BackSide
            });
            const atmosphere = new THREE.Mesh(atmosphereGeometry, atmosphereMaterial);
            this.planet.add(atmosphere);
        }
        
        this.scene.add(this.planet);
    }
    
    getPlanetData(planetName) {
        const data = {
            earth: {
                texture: this.createEarthTexture(),
                shininess: 30
            },
            mars: {
                texture: this.createMarsTexture(),
                shininess: 5
            },
            moon: {
                texture: this.createMoonTexture(),
                shininess: 1
            }
        };
        
        return data[planetName] || data.earth;
    }
    
    createEarthTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Create detailed Earth texture
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        gradient.addColorStop(0, '#1e3a8a');
        gradient.addColorStop(0.2, '#1e40af');
        gradient.addColorStop(0.4, '#3b82f6');
        gradient.addColorStop(0.6, '#059669');
        gradient.addColorStop(0.8, '#16a34a');
        gradient.addColorStop(1, '#a3a3a3');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Add continents
        ctx.fillStyle = '#166534';
        for (let i = 0; i < 20; i++) {
            ctx.globalAlpha = 0.7;
            ctx.fillRect(
                Math.random() * canvas.width,
                Math.random() * canvas.height,
                Math.random() * 200 + 50,
                Math.random() * 100 + 30
            );
        }
        
        // Add clouds
        ctx.fillStyle = '#ffffff';
        for (let i = 0; i < 30; i++) {
            ctx.globalAlpha = 0.3;
            ctx.fillRect(
                Math.random() * canvas.width,
                Math.random() * canvas.height,
                Math.random() * 80 + 20,
                Math.random() * 40 + 10
            );
        }
        
        return new THREE.CanvasTexture(canvas);
    }
    
    createMarsTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Mars surface
        const gradient = ctx.createRadialGradient(
            canvas.width/2, canvas.height/2, 0,
            canvas.width/2, canvas.height/2, canvas.width/2
        );
        gradient.addColorStop(0, '#dc2626');
        gradient.addColorStop(0.3, '#ea580c');
        gradient.addColorStop(0.5, '#f97316');
        gradient.addColorStop(0.7, '#ca8a04');
        gradient.addColorStop(1, '#92400e');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Add surface features
        for (let i = 0; i < 50; i++) {
            ctx.fillStyle = Math.random() > 0.5 ? '#b91c1c' : '#d97706';
            ctx.globalAlpha = 0.6;
            ctx.fillRect(
                Math.random() * canvas.width,
                Math.random() * canvas.height,
                Math.random() * 60 + 20,
                Math.random() * 60 + 20
            );
        }
        
        return new THREE.CanvasTexture(canvas);
    }
    
    createMoonTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        
        // Moon surface
        const gradient = ctx.createRadialGradient(
            canvas.width/2, canvas.height/2, 0,
            canvas.width/2, canvas.height/2, canvas.width/2
        );
        gradient.addColorStop(0, '#f9fafb');
        gradient.addColorStop(0.3, '#e5e7eb');
        gradient.addColorStop(0.5, '#d1d5db');
        gradient.addColorStop(0.7, '#9ca3af');
        gradient.addColorStop(1, '#4b5563');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Add craters
        for (let i = 0; i < 100; i++) {
            const x = Math.random() * canvas.width;
            const y = Math.random() * canvas.height;
            const radius = Math.random() * 30 + 5;
            
            ctx.fillStyle = '#374151';
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
            
            // Inner crater
            ctx.fillStyle = '#6b7280';
            ctx.globalAlpha = 0.4;
            ctx.beginPath();
            ctx.arc(x, y, radius * 0.3, 0, Math.PI * 2);
            ctx.fill();
        }
        
        return new THREE.CanvasTexture(canvas);
    }
    
    generateMockSeismicEvents() {
        const events = [];
        
        for (let i = 0; i < 50; i++) {
            events.push({
                id: `event_${i}`,
                timestamp: Date.now() - Math.random() * 86400000 * 30, // Last 30 days
                magnitude: Math.random() * 8 + 1,
                depth: Math.random() * 700 + 10,
                duration: Math.random() * 300 + 10,
                latitude: (Math.random() - 0.5) * 180,
                longitude: (Math.random() - 0.5) * 360,
                planet: ['earth', 'mars', 'moon'][Math.floor(Math.random() * 3)]
            });
        }
        
        return events.sort((a, b) => b.timestamp - a.timestamp);
    }
    
    addEventMarkers(planetName) {
        const planetEvents = this.seismicEvents.filter(event => event.planet === planetName);
        
        planetEvents.forEach(event => {
            const marker = this.createEventMarker(event);
            this.eventMarkers.push(marker);
            this.planet.add(marker);
        });
    }
    
    createEventMarker(event) {
        // Convert lat/lon to 3D coordinates on sphere surface
        const phi = (90 - event.latitude) * (Math.PI / 180);
        const theta = (event.longitude + 180) * (Math.PI / 180);
        
        const x = -(1.21 * Math.sin(phi) * Math.cos(theta));
        const y = 1.21 * Math.cos(phi);
        const z = 1.21 * Math.sin(phi) * Math.sin(theta);
        
        // Create marker based on magnitude
        const markerColor = this.getMagnitudeColor(event.magnitude);
        const markerSize = 0.02 + (event.magnitude / 10) * 0.05;
        
        const geometry = new THREE.SphereGeometry(markerSize, 16, 8);
        const material = new THREE.MeshBasicMaterial({
            color: markerColor,
            transparent: true,
            opacity: 0.8
        });
        
        const marker = new THREE.Mesh(geometry, material);
        marker.position.set(x, y, z);
        
        // Add pulsing animation
        marker.userData = {
            event: event,
            originalScale: markerSize,
            animationPhase: Math.random() * Math.PI * 2
        };
        
        // Add glow ring
        const ringGeometry = new THREE.RingGeometry(markerSize * 1.5, markerSize * 2, 8);
        const ringMaterial = new THREE.MeshBasicMaterial({
            color: markerColor,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        const ring = new THREE.Mesh(ringGeometry, ringMaterial);
        ring.lookAt(0, 0, 0);
        marker.add(ring);
        
        return marker;
    }
    
    getMagnitudeColor(magnitude) {
        if (magnitude < 3) return 0x10b981; // Green
        if (magnitude < 5) return 0xf59e0b; // Yellow
        if (magnitude < 7) return 0xf97316; // Orange
        return 0xef4444; // Red
    }
    
    toggleEventMarkers() {
        this.showMarkers = !this.showMarkers;
        
        if (this.showMarkers) {
            this.addEventMarkers(this.currentPlanet);
        } else {
            this.clearEventMarkers();
        }
    }
    
    clearEventMarkers() {
        this.eventMarkers.forEach(marker => {
            if (this.planet) {
                this.planet.remove(marker);
            }
            marker.geometry.dispose();
            marker.material.dispose();
        });
        this.eventMarkers = [];
    }
    
    highlightEvent(event) {
        // Find and highlight the corresponding marker
        const marker = this.eventMarkers.find(m => m.userData.event.id === event.id);
        if (marker) {
            // Create highlight effect
            const highlightGeometry = new THREE.SphereGeometry(0.1, 16, 8);
            const highlightMaterial = new THREE.MeshBasicMaterial({
                color: 0x00ffff,
                transparent: true,
                opacity: 0.6
            });
            
            const highlight = new THREE.Mesh(highlightGeometry, highlightMaterial);
            highlight.position.copy(marker.position);
            this.planet.add(highlight);
            
            // Animate highlight
            let scale = 0;
            const animateHighlight = () => {
                scale += 0.05;
                highlight.scale.setScalar(1 + Math.sin(scale) * 0.3);
                highlight.material.opacity = 0.6 - Math.sin(scale * 0.5) * 0.3;
                
                if (scale < Math.PI * 4) {
                    requestAnimationFrame(animateHighlight);
                } else {
                    this.planet.remove(highlight);
                    highlight.geometry.dispose();
                    highlight.material.dispose();
                }
            };
            animateHighlight();
        }
    }
    
    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);
            
            if (this.planet) {
                // Gentle planet rotation
                this.planet.rotation.y += 0.002;
                
                // Animate event markers
                this.eventMarkers.forEach(marker => {
                    const phase = marker.userData.animationPhase + Date.now() * 0.003;
                    const pulseScale = 1 + Math.sin(phase) * 0.2;
                    marker.scale.setScalar(pulseScale);
                    marker.material.opacity = 0.6 + Math.sin(phase * 2) * 0.3;
                });
            }
            
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    handleResize() {
        if (!this.renderer || !this.camera) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    destroy() {
        this.stopAnimation();
        this.clearEventMarkers();
        
        if (this.renderer) {
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }
        
        if (this.planet) {
            this.planet.geometry.dispose();
            this.planet.material.dispose();
        }
    }
}
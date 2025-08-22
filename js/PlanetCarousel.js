export class PlanetCarousel {
    constructor(containerId) {
        this.container = document.querySelector(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.planets = [];
        this.currentPlanetIndex = 0;
        this.isAnimating = false;
        this.animationId = null;
        this.onPlanetSelect = null;

        // GLTF Loader for 3D models
        this.loader = new THREE.GLTFLoader();

        // Planet metadata and model paths
        this.planetData = {
            earth: {
                name: 'Earth',
                radius: 1.2,
                texture: this.createEarthTexture(), // used as placeholder while model loads
                modelPath: '/public/models/earth.glb',
                position: { x: -3, y: 0, z: 0 },
                rotation: { x: 0, y: 0, z: 0.1 }
            },
            mars: {
                name: 'Mars',
                radius: 1.0,
                texture: this.createMarsTexture(),
                modelPath: '/public/models/mars.glb',
                position: { x: 0, y: 0, z: 0 },
                rotation: { x: 0, y: 0, z: 0.15 }
            },
            moon: {
                name: 'Moon',
                radius: 0.8,
                texture: this.createMoonTexture(),
                modelPath: '/public/models/moon.glb',
                position: { x: 3, y: 0, z: 0 },
                rotation: { x: 0, y: 0, z: 0 }
            }
        };
    }

    async initialize() {
        if (!this.container) {
            throw new Error('Container not found');
        }

        this.setupScene();
        this.createPlanets();
        this.setupLighting();
        this.setupControls();
        this.startAnimation();

        return Promise.resolve();
    }

    setupScene() {
        // Scene
        this.scene = new THREE.Scene();

        // Camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 0, 6);

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

        // Add starfield background
        this.createStarfield();
    }

    createStarfield() {
        const starsGeometry = new THREE.BufferGeometry();
        const starsMaterial = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 2,
            transparent: true,
            opacity: 0.8
        });

        const starsVertices = [];
        for (let i = 0; i < 1000; i++) {
            const x = (Math.random() - 0.5) * 2000;
            const y = (Math.random() - 0.5) * 2000;
            const z = (Math.random() - 0.5) * 2000;
            starsVertices.push(x, y, z);
        }

        starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starsVertices, 3));
        const stars = new THREE.Points(starsGeometry, starsMaterial);
        this.scene.add(stars);
    }

    createPlanets() {
        Object.entries(this.planetData).forEach(([key, data], index) => {
            // Create an invisible-but-raycastable sphere as the root for each planet
            const colliderGeometry = new THREE.SphereGeometry(data.radius, 32, 16);
            const colliderMaterial = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.0, depthWrite: false });
            const planet = new THREE.Mesh(colliderGeometry, colliderMaterial);
            planet.position.set(data.position.x, data.position.y, data.position.z);
            planet.rotation.set(data.rotation.x, data.rotation.y, data.rotation.z);
            planet.castShadow = false;
            planet.receiveShadow = false;

            // Add glow effect around the collider sphere
            const glowGeometry = new THREE.SphereGeometry(data.radius * 1.1, 32, 16);
            const glowMaterial = new THREE.MeshBasicMaterial({
                color: this.getPlanetGlowColor(key),
                transparent: true,
                opacity: 0.2,
                side: THREE.BackSide
            });
            const glow = new THREE.Mesh(glowGeometry, glowMaterial);
            planet.add(glow);

            // Add a visible placeholder sphere while model loads
            const placeholderGeometry = new THREE.SphereGeometry(data.radius, 64, 32);
            const placeholderMaterial = new THREE.MeshPhongMaterial({
                map: data.texture,
                shininess: key === 'earth' ? 30 : 5,
                transparent: true,
                opacity: 0.9
            });
            const placeholder = new THREE.Mesh(placeholderGeometry, placeholderMaterial);
            placeholder.castShadow = true;
            placeholder.receiveShadow = true;
            planet.add(placeholder);

            // Store metadata
            planet.userData = {
                planetName: key,
                originalPosition: { ...data.position },
                isSelected: index === 1, // Mars starts selected
                loaded: false,
                placeholder
            };

            // Begin loading the GLTF model (robust path handling)
            this.loadAndAttachModel(planet, data, key);

            this.planets.push(planet);
            this.scene.add(planet);
        });

        // Set initial selection after all roots created
        this.selectPlanet(1);
    }

    // Try loading a model from multiple candidate paths and extensions
    loadAndAttachModel(planet, data, key) {
        const original = data.modelPath;
        const baseNoExt = original.replace(/\.(glb|gltf)$/i, '');
        const baseCandidates = Array.from(new Set([
            baseNoExt,
            baseNoExt.replace('/public/models/', '/public/'),
            baseNoExt.replace('/public/models/', '/models/'),
            baseNoExt.replace('/public/models/', '/'),
        ]));
        const extensions = original.toLowerCase().endsWith('.gltf') ? ['.gltf', '.glb'] : ['.glb', '.gltf'];
        const urlQueue = [];
        for (const b of baseCandidates) {
            for (const ext of extensions) urlQueue.push(b + ext);
        }
        let idx = 0;
        const tryNext = () => {
            if (idx >= urlQueue.length) {
                console.error(`All candidate model paths failed for ${key}:`, urlQueue);
                return;
            }
            const url = urlQueue[idx++];
            this.loader.load(
                url,
                (gltf) => {
                    try {
                        const model = gltf.scene || gltf.scenes?.[0];
                        if (!model) throw new Error('Model scene not found in GLTF');
                        const box = new THREE.Box3().setFromObject(model);
                        const size = new THREE.Vector3();
                        box.getSize(size);
                        const center = new THREE.Vector3();
                        box.getCenter(center);
                        model.position.sub(center);
                        const maxDim = Math.max(size.x, size.y, size.z) || 1;
                        const desiredDiameter = data.radius * 2;
                        const scale = desiredDiameter / maxDim;
                        model.scale.setScalar(scale);
                        model.traverse((obj) => {
                            if (obj.isMesh) {
                                obj.castShadow = true;
                                obj.receiveShadow = true;
                            }
                        });
                        planet.add(model);
                        if (planet.userData.placeholder) {
                            planet.remove(planet.userData.placeholder);
                            planet.userData.placeholder.geometry.dispose();
                            planet.userData.placeholder.material.dispose();
                            planet.userData.placeholder = null;
                        }
                        planet.userData.loaded = true;
                    } catch (e) {
                        console.error(`Failed to process model for ${key} from ${url}:`, e);
                        tryNext();
                    }
                },
                undefined,
                () => {
                    // Try next URL
                    tryNext();
                }
            );
        };
        tryNext();
    }

    getPlanetGlowColor(planetName) {
        const colors = {
            earth: 0x00ffff,
            mars: 0xff6b35,
            moon: 0xc0c0c0
        };
        return colors[planetName] || 0xffffff;
    }

    createEarthTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 256;
        const ctx = canvas.getContext('2d');

        // Create Earth-like texture with blue oceans and green/brown continents
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        gradient.addColorStop(0, '#1e40af');
        gradient.addColorStop(0.3, '#3b82f6');
        gradient.addColorStop(0.6, '#059669');
        gradient.addColorStop(0.8, '#ca8a04');
        gradient.addColorStop(1, '#a3a3a3');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Add some texture variation
        for (let i = 0; i < 100; i++) {
            ctx.fillStyle = Math.random() > 0.5 ? '#10b981' : '#0ea5e9';
            ctx.globalAlpha = 0.3;
            ctx.fillRect(
                Math.random() * canvas.width,
                Math.random() * canvas.height,
                Math.random() * 50,
                Math.random() * 50
            );
        }

        return new THREE.CanvasTexture(canvas);
    }

    createMarsTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 256;
        const ctx = canvas.getContext('2d');

        // Create Mars-like texture with red/orange colors
        const gradient = ctx.createRadialGradient(
            canvas.width/2, canvas.height/2, 0,
            canvas.width/2, canvas.height/2, canvas.width/2
        );
        gradient.addColorStop(0, '#dc2626');
        gradient.addColorStop(0.3, '#ea580c');
        gradient.addColorStop(0.6, '#f97316');
        gradient.addColorStop(0.8, '#ca8a04');
        gradient.addColorStop(1, '#92400e');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Add surface details
        for (let i = 0; i < 80; i++) {
            ctx.fillStyle = Math.random() > 0.5 ? '#b91c1c' : '#d97706';
            ctx.globalAlpha = 0.4;
            ctx.fillRect(
                Math.random() * canvas.width,
                Math.random() * canvas.height,
                Math.random() * 30,
                Math.random() * 30
            );
        }

        return new THREE.CanvasTexture(canvas);
    }

    createMoonTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 256;
        const ctx = canvas.getContext('2d');

        // Create Moon-like texture with gray colors and craters
        const gradient = ctx.createRadialGradient(
            canvas.width/2, canvas.height/2, 0,
            canvas.width/2, canvas.height/2, canvas.width/2
        );
        gradient.addColorStop(0, '#f3f4f6');
        gradient.addColorStop(0.3, '#d1d5db');
        gradient.addColorStop(0.6, '#9ca3af');
        gradient.addColorStop(0.8, '#6b7280');
        gradient.addColorStop(1, '#374151');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Add craters
        for (let i = 0; i < 60; i++) {
            const x = Math.random() * canvas.width;
            const y = Math.random() * canvas.height;
            const radius = Math.random() * 20 + 5;

            ctx.fillStyle = '#4b5563';
            ctx.globalAlpha = 0.6;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fill();
        }

        return new THREE.CanvasTexture(canvas);
    }

    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);

        // Directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(-5, 3, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);

        // Point lights for each planet
        this.planets.forEach((planet) => {
            const light = new THREE.PointLight(
                this.getPlanetGlowColor(planet.userData.planetName),
                0.3,
                10
            );
            light.position.copy(planet.position);
            light.position.z += 2;
            this.scene.add(light);
        });
    }

    setupControls() {
        // Mouse/touch interaction
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        const onMouseClick = (event) => {
            if (this.isAnimating) return;

            const rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, this.camera);
            const intersects = raycaster.intersectObjects(this.planets, true);

            if (intersects.length > 0) {
                let selected = intersects[0].object;
                while (selected && !this.planets.includes(selected)) {
                    selected = selected.parent;
                }
                if (!selected) return;
                const planetIndex = this.planets.indexOf(selected);
                if (planetIndex !== -1) {
                    this.selectPlanet(planetIndex);

                    // Trigger planet selection callback
                    if (this.onPlanetSelect) {
                        this.onPlanetSelect(selected.userData.planetName);
                    }
                }
            }
        };

        this.renderer.domElement.addEventListener('click', onMouseClick);
        this.renderer.domElement.addEventListener('touchend', onMouseClick);

        // Hover effects
        const onMouseMove = (event) => {
            const rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, this.camera);
            const intersects = raycaster.intersectObjects(this.planets, true);

            const hoveredRoots = new Set();
            for (const hit of intersects) {
                let node = hit.object;
                while (node && !this.planets.includes(node)) node = node.parent;
                if (node) hoveredRoots.add(node);
            }

            this.planets.forEach(planet => {
                const isHovered = hoveredRoots.has(planet);
                planet.scale.setScalar(isHovered ? 1.1 : 1.0);
                if (isHovered) this.renderer.domElement.style.cursor = 'pointer';
            });
            if (hoveredRoots.size === 0) this.renderer.domElement.style.cursor = 'default';
        };

        this.renderer.domElement.addEventListener('mousemove', onMouseMove);
    }

    selectPlanet(index) {
        if (index === this.currentPlanetIndex) return;

        this.isAnimating = true;
        this.currentPlanetIndex = index;

        // Reset all planets
        this.planets.forEach((planet, i) => {
            planet.userData.isSelected = i === index;

            // Animate to new positions
            const targetPosition = i === index ? { x: 0, y: 0, z: 0 } :
                i < index ? { x: -3 - (index - i), y: 0, z: -1 } :
                { x: 3 + (i - index), y: 0, z: -1 };

            this.animatePlanetPosition(planet, targetPosition);
        });

        setTimeout(() => {
            this.isAnimating = false;
        }, 1000);
    }

    animatePlanetPosition(planet, targetPosition) {
        const startPosition = { ...planet.position };
        const duration = 1000; // 1 second
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeProgress = this.easeInOutCubic(progress);

            planet.position.x = startPosition.x + (targetPosition.x - startPosition.x) * easeProgress;
            planet.position.y = startPosition.y + (targetPosition.y - startPosition.y) * easeProgress;
            planet.position.z = startPosition.z + (targetPosition.z - startPosition.z) * easeProgress;

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        animate();
    }

    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
    }

    startAnimation() {
        const animate = () => {
            this.animationId = requestAnimationFrame(animate);

            // Rotate planets
            this.planets.forEach((planet, index) => {
                planet.rotation.y += 0.005 + (index * 0.001);

                // Add subtle floating motion
                planet.position.y = planet.userData.originalPosition.y +
                    Math.sin(Date.now() * 0.001 + index) * 0.05;
            });

            // Slowly rotate the camera around the scene
            const time = Date.now() * 0.0002;
            this.camera.position.x = Math.cos(time) * 6;
            this.camera.position.z = Math.sin(time) * 6;
            this.camera.lookAt(this.scene.position);

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

        // Dispose scene objects
        if (this.planets) {
            this.planets.forEach((planet) => {
                // Dispose collider geometry/material
                if (planet.geometry) planet.geometry.dispose?.();
                if (planet.material) {
                    const mats = Array.isArray(planet.material) ? planet.material : [planet.material];
                    mats.forEach(m => m && m.dispose && m.dispose());
                }
                // Dispose any child meshes/materials/textures (models or placeholders)
                planet.traverse((obj) => {
                    if (obj.isMesh) {
                        if (obj.geometry) obj.geometry.dispose?.();
                        if (obj.material) {
                            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
                            mats.forEach((mat) => {
                                if (!mat) return;
                                // dispose common textures if present
                                ['map','normalMap','roughnessMap','metalnessMap','aoMap','emissiveMap','bumpMap','alphaMap','displacementMap'].forEach((key) => {
                                    if (mat[key] && mat[key].dispose) mat[key].dispose();
                                });
                                mat.dispose?.();
                            });
                        }
                    }
                });
                this.scene && this.scene.remove(planet);
            });
            this.planets = [];
        }

        if (this.renderer) {
            this.renderer.dispose();
            if (this.renderer.domElement && this.container.contains(this.renderer.domElement)) {
                this.container.removeChild(this.renderer.domElement);
            }
        }
    }
}
/**
 * HETROFL 3D Workflow Visualization
 * Production-ready 3D scene for federated learning workflow visualization
 * Enhanced with advanced particle systems, animations, interactions, and optimizations
 * 
 * @version 3.0.0
 * @author HETROFL Team
 * @license MIT
 */

// Performance monitoring and optimization utilities
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            fps: 0,
            frameTime: 0,
            memoryUsage: 0,
            drawCalls: 0,
            triangles: 0
        };
        this.fpsHistory = [];
        this.maxHistoryLength = 60;
        this.lastTime = performance.now();
        this.frameCount = 0;
    }
    
    update(renderer) {
        const currentTime = performance.now();
        this.frameCount++;
        
        if (currentTime - this.lastTime >= 1000) {
            this.metrics.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime));
            this.metrics.frameTime = (currentTime - this.lastTime) / this.frameCount;
            
            this.fpsHistory.push(this.metrics.fps);
            if (this.fpsHistory.length > this.maxHistoryLength) {
                this.fpsHistory.shift();
            }
            
            this.frameCount = 0;
            this.lastTime = currentTime;
            
            // Memory usage (if available)
            if (performance.memory) {
                this.metrics.memoryUsage = Math.round(performance.memory.usedJSHeapSize / 1048576);
            }
            
            // Renderer info
            if (renderer && renderer.info) {
                this.metrics.drawCalls = renderer.info.render.calls;
                this.metrics.triangles = renderer.info.render.triangles;
            }
        }
    }
    
    getAverageFPS() {
        if (this.fpsHistory.length === 0) return 0;
        return Math.round(this.fpsHistory.reduce((a, b) => a + b, 0) / this.fpsHistory.length);
    }
    
    shouldReduceQuality() {
        const avgFPS = this.getAverageFPS();
        return avgFPS < 30 && this.fpsHistory.length >= 10;
    }
}

// Settings manager for quality and preferences
class SettingsManager {
    constructor() {
        this.settings = {
            quality: 'auto', // auto, low, medium, high, ultra
            particles: true,
            fog: true,
            glow: true,
            shadows: true,
            antialiasing: true,
            pixelRatio: 'auto',
            reducedMotion: false,
            autoQuality: true
        };
        
        this.qualityPresets = {
            low: {
                particles: false,
                fog: false,
                glow: false,
                shadows: false,
                antialiasing: false,
                pixelRatio: 1,
                maxParticles: 100
            },
            medium: {
                particles: true,
                fog: true,
                glow: false,
                shadows: false,
                antialiasing: true,
                pixelRatio: 1,
                maxParticles: 500
            },
            high: {
                particles: true,
                fog: true,
                glow: true,
                shadows: true,
                antialiasing: true,
                pixelRatio: 1.5,
                maxParticles: 1000
            },
            ultra: {
                particles: true,
                fog: true,
                glow: true,
                shadows: true,
                antialiasing: true,
                pixelRatio: 2,
                maxParticles: 2000
            }
        };
        
        this.loadSettings();
        this.detectCapabilities();
    }
    
    loadSettings() {
        try {
            const saved = localStorage.getItem('hetrofl-3d-settings');
            if (saved) {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
            }
        } catch (e) {
            console.warn('Failed to load settings:', e);
        }
        
        // Check for reduced motion preference
        if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            this.settings.reducedMotion = true;
        }
    }
    
    saveSettings() {
        try {
            localStorage.setItem('hetrofl-3d-settings', JSON.stringify(this.settings));
        } catch (e) {
            console.warn('Failed to save settings:', e);
        }
    }
    
    detectCapabilities() {
        // Detect device capabilities
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        
        if (!gl) {
            this.settings.quality = 'low';
            return;
        }
        
        // Check for mobile device
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        // Auto-detect quality based on device
        if (this.settings.quality === 'auto') {
            if (isMobile) {
                this.settings.quality = 'medium';
            } else {
                // Desktop - check for high-end GPU
                const renderer = gl.getParameter(gl.RENDERER);
                if (renderer && (renderer.includes('RTX') || renderer.includes('GTX') || renderer.includes('Radeon'))) {
                    this.settings.quality = 'high';
                } else {
                    this.settings.quality = 'medium';
                }
            }
        }
    }
    
    applyQualityPreset(quality) {
        if (this.qualityPresets[quality]) {
            this.settings = { ...this.settings, ...this.qualityPresets[quality], quality };
            this.saveSettings();
        }
    }
    
    get(key) {
        return this.settings[key];
    }
    
    set(key, value) {
        this.settings[key] = value;
        this.saveSettings();
    }
}

// Loading manager with progress tracking
class LoadingManager {
    constructor() {
        this.loadingElement = null;
        this.progressBar = null;
        this.progressText = null;
        this.totalItems = 0;
        this.loadedItems = 0;
        this.isLoading = false;
        this.loadingPromises = [];
        
        this.createLoadingScreen();
    }
    
    createLoadingScreen() {
        // Create loading overlay
        this.loadingElement = document.createElement('div');
        this.loadingElement.className = 'workflow-3d-loading';
        this.loadingElement.innerHTML = `
            <div class="loading-content">
                <div class="loading-logo">
                    <i class="fas fa-cube fa-spin"></i>
                </div>
                <h3>Loading HETROFL 3D Workflow</h3>
                <div class="loading-progress">
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    <div class="progress-text">0%</div>
                </div>
                <div class="loading-status">Initializing...</div>
            </div>
        `;
        
        this.progressBar = this.loadingElement.querySelector('.progress-fill');
        this.progressText = this.loadingElement.querySelector('.progress-text');
        this.statusText = this.loadingElement.querySelector('.loading-status');
    }
    
    show(container) {
        if (container && !container.contains(this.loadingElement)) {
            container.appendChild(this.loadingElement);
        }
        this.isLoading = true;
    }
    
    hide() {
        if (this.loadingElement && this.loadingElement.parentNode) {
            this.loadingElement.parentNode.removeChild(this.loadingElement);
        }
        this.isLoading = false;
    }
    
    updateProgress(loaded, total, status = '') {
        this.loadedItems = loaded;
        this.totalItems = total;
        
        const progress = total > 0 ? (loaded / total) * 100 : 0;
        
        if (this.progressBar) {
            this.progressBar.style.width = `${progress}%`;
        }
        
        if (this.progressText) {
            this.progressText.textContent = `${Math.round(progress)}%`;
        }
        
        if (this.statusText && status) {
            this.statusText.textContent = status;
        }
    }
    
    async loadAssets(assets) {
        this.totalItems = assets.length;
        this.loadedItems = 0;
        
        const promises = assets.map(async (asset, index) => {
            try {
                await this.loadAsset(asset);
                this.loadedItems++;
                this.updateProgress(this.loadedItems, this.totalItems, `Loaded ${asset.name || asset.url}`);
            } catch (error) {
                console.warn(`Failed to load asset: ${asset.name || asset.url}`, error);
                this.loadedItems++;
                this.updateProgress(this.loadedItems, this.totalItems, `Failed: ${asset.name || asset.url}`);
            }
        });
        
        await Promise.all(promises);
    }
    
    async loadAsset(asset) {
        return new Promise((resolve, reject) => {
            if (asset.type === 'texture') {
                const loader = new THREE.TextureLoader();
                loader.load(asset.url, resolve, undefined, reject);
            } else if (asset.type === 'model') {
                // Model loading would go here
                setTimeout(resolve, 100); // Simulate loading
            } else {
                resolve();
            }
        });
    }
}

// Help system with interactive tutorials
class HelpSystem {
    constructor() {
        this.isActive = false;
        this.currentStep = 0;
        this.steps = [
            {
                target: '.workflow-3d-controls',
                title: 'Animation Controls',
                content: 'Use these controls to start, pause, and reset the workflow animation. Press Space to quickly start/pause.',
                position: 'bottom'
            },
            {
                target: '.control-group:nth-child(2)',
                title: 'View Controls',
                content: 'Adjust camera position and zoom. Use mouse wheel to zoom, drag to rotate the view.',
                position: 'bottom'
            },
            {
                target: '.control-group:nth-child(3)',
                title: 'Visual Effects',
                content: 'Toggle various visual effects like particles, fog, and glow to customize your experience.',
                position: 'bottom'
            },
            {
                target: '#workflow-3d-container canvas',
                title: 'Interactive 3D Scene',
                content: 'Click on nodes to see details. The animation shows data flow through the federated learning process.',
                position: 'center'
            }
        ];
        
        this.createHelpOverlay();
    }
    
    createHelpOverlay() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'help-overlay';
        this.overlay.innerHTML = `
            <div class="help-backdrop"></div>
            <div class="help-tooltip">
                <div class="help-content">
                    <h4 class="help-title"></h4>
                    <p class="help-text"></p>
                </div>
                <div class="help-navigation">
                    <button class="help-btn help-prev">Previous</button>
                    <span class="help-progress">1 / 4</span>
                    <button class="help-btn help-next">Next</button>
                    <button class="help-btn help-close">Close</button>
                </div>
            </div>
        `;
        
        this.setupHelpEvents();
    }
    
    setupHelpEvents() {
        const prevBtn = this.overlay.querySelector('.help-prev');
        const nextBtn = this.overlay.querySelector('.help-next');
        const closeBtn = this.overlay.querySelector('.help-close');
        
        prevBtn.addEventListener('click', () => this.previousStep());
        nextBtn.addEventListener('click', () => this.nextStep());
        closeBtn.addEventListener('click', () => this.hide());
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (!this.isActive) return;
            
            switch (e.key) {
                case 'ArrowLeft':
                    this.previousStep();
                    break;
                case 'ArrowRight':
                    this.nextStep();
                    break;
                case 'Escape':
                    this.hide();
                    break;
            }
        });
    }
    
    show() {
        document.body.appendChild(this.overlay);
        this.isActive = true;
        this.currentStep = 0;
        this.showStep(0);
    }
    
    hide() {
        if (this.overlay.parentNode) {
            this.overlay.parentNode.removeChild(this.overlay);
        }
        this.isActive = false;
    }
    
    showStep(stepIndex) {
        if (stepIndex < 0 || stepIndex >= this.steps.length) return;
        
        this.currentStep = stepIndex;
        const step = this.steps[stepIndex];
        
        // Update content
        this.overlay.querySelector('.help-title').textContent = step.title;
        this.overlay.querySelector('.help-text').textContent = step.content;
        this.overlay.querySelector('.help-progress').textContent = `${stepIndex + 1} / ${this.steps.length}`;
        
        // Position tooltip
        this.positionTooltip(step);
        
        // Update navigation buttons
        this.overlay.querySelector('.help-prev').disabled = stepIndex === 0;
        this.overlay.querySelector('.help-next').disabled = stepIndex === this.steps.length - 1;
    }
    
    positionTooltip(step) {
        const tooltip = this.overlay.querySelector('.help-tooltip');
        const target = document.querySelector(step.target);
        
        if (!target) return;
        
        const targetRect = target.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        
        let left, top;
        
        switch (step.position) {
            case 'bottom':
                left = targetRect.left + (targetRect.width - tooltipRect.width) / 2;
                top = targetRect.bottom + 10;
                break;
            case 'top':
                left = targetRect.left + (targetRect.width - tooltipRect.width) / 2;
                top = targetRect.top - tooltipRect.height - 10;
                break;
            case 'center':
                left = (window.innerWidth - tooltipRect.width) / 2;
                top = (window.innerHeight - tooltipRect.height) / 2;
                break;
            default:
                left = targetRect.right + 10;
                top = targetRect.top + (targetRect.height - tooltipRect.height) / 2;
        }
        
        tooltip.style.left = `${Math.max(10, Math.min(left, window.innerWidth - tooltipRect.width - 10))}px`;
        tooltip.style.top = `${Math.max(10, Math.min(top, window.innerHeight - tooltipRect.height - 10))}px`;
    }
    
    nextStep() {
        if (this.currentStep < this.steps.length - 1) {
            this.showStep(this.currentStep + 1);
        }
    }
    
    previousStep() {
        if (this.currentStep > 0) {
            this.showStep(this.currentStep - 1);
        }
    }
}

// Error handler with user-friendly messages
class ErrorHandler {
    constructor() {
        this.errorContainer = null;
        this.createErrorContainer();
    }
    
    createErrorContainer() {
        this.errorContainer = document.createElement('div');
        this.errorContainer.className = 'error-container';
        this.errorContainer.innerHTML = `
            <div class="error-content">
                <div class="error-icon">
                    <i class="fas fa-exclamation-triangle"></i>
                </div>
                <div class="error-message">
                    <h4>Something went wrong</h4>
                    <p class="error-details"></p>
                    <div class="error-actions">
                        <button class="btn btn-primary error-retry">Try Again</button>
                        <button class="btn btn-secondary error-dismiss">Dismiss</button>
                    </div>
                </div>
            </div>
        `;
        
        this.setupErrorEvents();
    }
    
    setupErrorEvents() {
        const retryBtn = this.errorContainer.querySelector('.error-retry');
        const dismissBtn = this.errorContainer.querySelector('.error-dismiss');
        
        retryBtn.addEventListener('click', () => {
            this.hide();
            if (this.retryCallback) {
                this.retryCallback();
            }
        });
        
        dismissBtn.addEventListener('click', () => this.hide());
    }
    
    show(message, details = '', retryCallback = null) {
        this.errorContainer.querySelector('h4').textContent = message;
        this.errorContainer.querySelector('.error-details').textContent = details;
        this.retryCallback = retryCallback;
        
        if (!document.body.contains(this.errorContainer)) {
            document.body.appendChild(this.errorContainer);
        }
        
        this.errorContainer.classList.add('show');
    }
    
    hide() {
        this.errorContainer.classList.remove('show');
        setTimeout(() => {
            if (this.errorContainer.parentNode) {
                this.errorContainer.parentNode.removeChild(this.errorContainer);
            }
        }, 300);
    }
}

class ParticleSystem {
    constructor(scene, settings = null) {
        this.scene = scene;
        this.settings = settings;
        this.particles = [];
        this.particlePool = [];
        this.maxParticles = settings ? settings.get('maxParticles') || 1000 : 1000;
        this.instancedMesh = null;
        this.particleGeometry = null;
        this.particleMaterial = null;
        this.enabled = settings ? settings.get('particles') : true;
        this.init();
    }
    
    init() {
        // Create instanced geometry for performance
        this.particleGeometry = new THREE.SphereGeometry(0.02, 8, 8);
        this.particleMaterial = new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0.8
        });
        
        // Create particle pool for object reuse
        for (let i = 0; i < this.maxParticles; i++) {
            const particle = new THREE.Mesh(this.particleGeometry, this.particleMaterial.clone());
            particle.visible = false;
            this.particlePool.push(particle);
            this.scene.add(particle);
        }
    }
    
    createDataFlowParticle(curve, color = 0x4361ee, speed = 0.01) {
        const particle = this.getParticleFromPool();
        if (!particle) return null;
        
        particle.material.color.setHex(color);
        particle.material.opacity = 0.8;
        particle.visible = true;
        particle.scale.setScalar(1);
        
        const particleData = {
            mesh: particle,
            curve: curve,
            progress: 0,
            speed: speed + Math.random() * 0.005,
            life: 1.0,
            maxLife: 1.0,
            trail: []
        };
        
        this.particles.push(particleData);
        return particleData;
    }
    
    createBurstEffect(position, color = 0x05c46b, count = 20) {
        for (let i = 0; i < count; i++) {
            const particle = this.getParticleFromPool();
            if (!particle) break;
            
            particle.material.color.setHex(color);
            particle.material.opacity = 1.0;
            particle.visible = true;
            particle.position.copy(position);
            
            const direction = new THREE.Vector3(
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2,
                (Math.random() - 0.5) * 2
            ).normalize();
            
            const particleData = {
                mesh: particle,
                velocity: direction.multiplyScalar(0.1 + Math.random() * 0.1),
                life: 1.0,
                maxLife: 1.0,
                isBurst: true
            };
            
            this.particles.push(particleData);
        }
    }
    
    createAmbientParticles(count = 50) {
        for (let i = 0; i < count; i++) {
            const particle = this.getParticleFromPool();
            if (!particle) break;
            
            particle.material.color.setHex(0x2d3748);
            particle.material.opacity = 0.3;
            particle.visible = true;
            particle.position.set(
                (Math.random() - 0.5) * 60,
                (Math.random() - 0.5) * 30,
                (Math.random() - 0.5) * 60
            );
            
            const particleData = {
                mesh: particle,
                velocity: new THREE.Vector3(
                    (Math.random() - 0.5) * 0.01,
                    (Math.random() - 0.5) * 0.01,
                    (Math.random() - 0.5) * 0.01
                ),
                life: Infinity,
                isAmbient: true,
                initialPosition: particle.position.clone()
            };
            
            this.particles.push(particleData);
        }
    }
    
    getParticleFromPool() {
        return this.particlePool.find(p => !p.visible) || null;
    }
    
    update(deltaTime) {
        this.particles = this.particles.filter(particleData => {
            const particle = particleData.mesh;
            
            if (particleData.curve) {
                // Data flow particle
                particleData.progress += particleData.speed;
                if (particleData.progress >= 1) {
                    particle.visible = false;
                    return false;
                }
                
                const position = particleData.curve.getPoint(particleData.progress);
                particle.position.copy(position);
                
                // Add trail effect
                if (particleData.trail.length > 10) {
                    particleData.trail.shift();
                }
                particleData.trail.push(position.clone());
                
            } else if (particleData.isBurst) {
                // Burst particle
                particleData.life -= deltaTime * 2;
                if (particleData.life <= 0) {
                    particle.visible = false;
                    return false;
                }
                
                particle.position.add(particleData.velocity);
                particle.material.opacity = particleData.life;
                particle.scale.setScalar(particleData.life);
                
            } else if (particleData.isAmbient) {
                // Ambient floating particle
                particle.position.add(particleData.velocity);
                
                // Boundary check
                const distance = particle.position.distanceTo(particleData.initialPosition);
                if (distance > 5) {
                    particleData.velocity.multiplyScalar(-1);
                }
            }
            
            return true;
        });
    }
    
    clear() {
        this.particles.forEach(particleData => {
            particleData.mesh.visible = false;
        });
        this.particles = [];
    }
}

class AnimationManager {
    constructor() {
        this.timeline = null;
        this.animations = new Map();
        this.isGSAPAvailable = typeof gsap !== 'undefined';
        this.cameraPresets = {
            perspective: { position: [0, 10, 20], target: [0, 0, 0] },
            top: { position: [0, 30, 0], target: [0, 0, 0] },
            side: { position: [30, 5, 0], target: [0, 0, 0] },
            overview: { position: [15, 15, 15], target: [0, 0, 0] }
        };
    }
    
    createTimeline() {
        if (this.isGSAPAvailable) {
            this.timeline = gsap.timeline({ paused: true });
            return this.timeline;
        }
        return null;
    }
    
    animateCamera(camera, preset, duration = 2) {
        const targetPreset = this.cameraPresets[preset];
        if (!targetPreset) return;
        
        if (this.isGSAPAvailable) {
            gsap.to(camera.position, {
                duration: duration,
                x: targetPreset.position[0],
                y: targetPreset.position[1],
                z: targetPreset.position[2],
                ease: "power2.inOut"
            });
            
            // Animate look-at target
            const lookAtTarget = new THREE.Vector3(...targetPreset.target);
            gsap.to(camera, {
                duration: duration,
                onUpdate: () => camera.lookAt(lookAtTarget),
                ease: "power2.inOut"
            });
        } else {
            // Fallback animation
            this.animateCameraFallback(camera, targetPreset, duration);
        }
    }
    
    animateCameraFallback(camera, preset, duration) {
        const startPos = camera.position.clone();
        const targetPos = new THREE.Vector3(...preset.position);
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = (Date.now() - startTime) / 1000;
            const progress = Math.min(elapsed / duration, 1);
            const eased = this.easeInOutQuad(progress);
            
            camera.position.lerpVectors(startPos, targetPos, eased);
            camera.lookAt(...preset.target);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    animateNodeEntrance(node, delay = 0) {
        node.scale.setScalar(0);
        node.visible = true;
        
        if (this.isGSAPAvailable) {
            gsap.to(node.scale, {
                duration: 1,
                x: 1, y: 1, z: 1,
                delay: delay,
                ease: "back.out(1.7)"
            });
            
            gsap.to(node.rotation, {
                duration: 1,
                y: Math.PI * 2,
                delay: delay,
                ease: "power2.out"
            });
        } else {
            this.animateNodeEntranceFallback(node, delay);
        }
    }
    
    animateNodeEntranceFallback(node, delay) {
        setTimeout(() => {
            const startTime = Date.now();
            const duration = 1000;
            
            const animate = () => {
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = this.easeBackOut(progress);
                
                node.scale.setScalar(eased);
                node.rotation.y = progress * Math.PI * 2;
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                }
            };
            
            animate();
        }, delay * 1000);
    }
    
    animateNodePulse(node, color = 0x4361ee) {
        if (this.isGSAPAvailable) {
            gsap.to(node.scale, {
                duration: 1,
                x: 1.2, y: 1.2, z: 1.2,
                yoyo: true,
                repeat: -1,
                ease: "power2.inOut"
            });
        }
    }
    
    animateConnectionFlow(connection, duration = 2) {
        if (!connection.userData.curve) return;
        
        const points = connection.userData.curve.getPoints(50);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        // Create animated line material
        const material = new THREE.LineBasicMaterial({
            color: 0x4361ee,
            transparent: true,
            opacity: 0
        });
        
        if (this.isGSAPAvailable) {
            gsap.to(material, {
                duration: duration,
                opacity: 1,
                ease: "power2.inOut"
            });
        }
    }
    
    // Easing functions for fallback animations
    easeInOutQuad(t) {
        return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
    }
    
    easeBackOut(t) {
        const c1 = 1.70158;
        const c3 = c1 + 1;
        return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
    }
    
    stopAll() {
        if (this.isGSAPAvailable && this.timeline) {
            this.timeline.kill();
        }
        this.animations.clear();
    }
}

class HETROFL3DWorkflow {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.animationId = null;
        
        // Production systems
        this.settingsManager = new SettingsManager();
        this.performanceMonitor = new PerformanceMonitor();
        this.loadingManager = new LoadingManager();
        this.helpSystem = new HelpSystem();
        this.errorHandler = new ErrorHandler();
        
        // Advanced systems
        this.particleSystem = null;
        this.animationManager = null;
        this.raycaster = null;
        this.mouse = new THREE.Vector2();
        this.selectedNode = null;
        this.hoveredNode = null;
        
        // Scene objects
        this.nodes = {};
        this.connections = [];
        this.dataParticles = [];
        this.environmentEffects = {};
        
        // Animation state
        this.isAnimating = false;
        this.animationSpeed = 1.0;
        this.currentStep = 0;
        this.workflowSteps = [];
        this.isPaused = false;
        
        // Performance monitoring
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.fps = 60;
        this.deltaTime = 0;
        this.qualityLevel = 'medium';
        
        // Resource management
        this.disposables = [];
        this.isDisposed = false;
        
        // Data integration
        this.realTimeData = {};
        this.dataUpdateInterval = null;
        
        this.init();
    }
    
    async init() {
        try {
            // Show loading screen
            this.loadingManager.show(this.container);
            this.loadingManager.updateProgress(0, 10, 'Checking WebGL support...');
            
            // Check WebGL support
            if (!this.checkWebGLSupport()) {
                throw new Error('WebGL not supported');
            }
            
            // Initialize with progressive loading
            await this.initializeWithProgress();
            
            // Hide loading screen
            this.loadingManager.hide();
            
            console.log('HETROFL 3D Workflow initialized successfully');
            console.log(`- Quality: ${this.settingsManager.get('quality')}`);
            console.log(`- Renderer: ${this.renderer.capabilities.isWebGL2 ? 'WebGL2' : 'WebGL1'}`);
            console.log(`- Nodes created: ${Object.values(this.nodes).reduce((total, arr) => total + arr.length, 0)}`);
            console.log(`- Connections created: ${this.connections.length}`);
            console.log(`- Particle system: ${this.particleSystem ? 'Enabled' : 'Disabled'}`);
            console.log(`- Animation manager: ${this.animationManager ? 'Enabled' : 'Disabled'}`);
            
        } catch (error) {
            console.error('Failed to initialize 3D workflow:', error);
            this.loadingManager.hide();
            this.errorHandler.show(
                'Failed to initialize 3D visualization',
                error.message,
                () => this.init()
            );
            this.showFallbackMessage();
        }
    }
    
    async initializeWithProgress() {
        const steps = [
            { fn: () => this.createScene(), name: 'Creating scene...' },
            { fn: () => this.createCamera(), name: 'Setting up camera...' },
            { fn: () => this.createRenderer(), name: 'Initializing renderer...' },
            { fn: () => this.createLights(), name: 'Adding lights...' },
            { fn: () => this.createControls(), name: 'Setting up controls...' },
            { fn: () => this.setupEventListeners(), name: 'Binding events...' },
            { fn: () => this.setupInteractionSystem(), name: 'Enabling interactions...' },
            { fn: () => this.createAdvancedSystems(), name: 'Loading advanced systems...' },
            { fn: () => this.createEnvironmentalEffects(), name: 'Creating effects...' },
            { fn: () => this.createWorkflowNodes(), name: 'Building workflow nodes...' },
            { fn: () => this.createConnections(), name: 'Connecting nodes...' },
            { fn: () => this.setupDataIntegration(), name: 'Integrating data...' },
            { fn: () => this.setupProductionFeatures(), name: 'Finalizing...' },
            { fn: () => this.animate(), name: 'Starting animation...' }
        ];
        
        for (let i = 0; i < steps.length; i++) {
            this.loadingManager.updateProgress(i, steps.length, steps[i].name);
            await new Promise(resolve => {
                setTimeout(() => {
                    steps[i].fn();
                    resolve();
                }, 50); // Small delay for smooth progress
            });
        }
        
        this.loadingManager.updateProgress(steps.length, steps.length, 'Complete!');
    }
    
    setupProductionFeatures() {
        // Apply quality settings
        this.applyQualitySettings();
        
        // Setup performance monitoring
        this.setupPerformanceMonitoring();
        
        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();
        
        // Setup fullscreen support
        this.setupFullscreenSupport();
        
        // Setup accessibility features
        this.setupAccessibilityFeatures();
        
        // Start data integration
        this.startDataIntegration();
    }
    
    applyQualitySettings() {
        const quality = this.settingsManager.get('quality');
        
        // Apply renderer settings
        this.renderer.setPixelRatio(
            this.settingsManager.get('pixelRatio') === 'auto' 
                ? Math.min(window.devicePixelRatio, 2)
                : this.settingsManager.get('pixelRatio')
        );
        
        // Apply antialiasing
        if (!this.settingsManager.get('antialiasing')) {
            this.renderer.antialias = false;
        }
        
        // Apply shadows
        this.renderer.shadowMap.enabled = this.settingsManager.get('shadows');
        
        // Apply fog
        if (this.settingsManager.get('fog')) {
            this.scene.fog = new THREE.Fog(0x06090f, 10, 100);
        } else {
            this.scene.fog = null;
        }
        
        // Update particle system
        if (this.particleSystem) {
            this.particleSystem.enabled = this.settingsManager.get('particles');
            this.particleSystem.maxParticles = this.settingsManager.get('maxParticles') || 1000;
        }
    }
    
    setupPerformanceMonitoring() {
        // Auto quality adjustment
        if (this.settingsManager.get('autoQuality')) {
            setInterval(() => {
                if (this.performanceMonitor.shouldReduceQuality()) {
                    this.reduceQuality();
                }
            }, 5000);
        }
        
        // Performance stats display (development mode)
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            this.createPerformanceDisplay();
        }
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (event) => {
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
            
            switch (event.key) {
                case ' ': // Space - toggle animation
                    event.preventDefault();
                    this.toggleAnimation();
                    break;
                case 'r': // R - reset view
                    this.resetCamera();
                    break;
                case 'f': // F - toggle fullscreen
                    this.toggleFullscreen();
                    break;
                case 'h': // H - show help
                    this.helpSystem.show();
                    break;
                case '1':
                case '2':
                case '3':
                case '4':
                    const qualities = ['low', 'medium', 'high', 'ultra'];
                    this.setQuality(qualities[parseInt(event.key) - 1]);
                    break;
            }
        });
    }
    
    setupFullscreenSupport() {
        // Add fullscreen button to controls
        const fullscreenBtn = document.createElement('button');
        fullscreenBtn.className = 'control-btn';
        fullscreenBtn.innerHTML = '<i class="fas fa-expand"></i> Fullscreen';
        fullscreenBtn.title = 'Toggle fullscreen (F)';
        fullscreenBtn.addEventListener('click', () => this.toggleFullscreen());
        
        const controlsPanel = document.querySelector('.workflow-3d-controls .control-group:last-child .control-buttons');
        if (controlsPanel) {
            controlsPanel.appendChild(fullscreenBtn);
        }
    }
    
    setupAccessibilityFeatures() {
        // Add ARIA labels
        if (this.renderer.domElement) {
            this.renderer.domElement.setAttribute('role', 'img');
            this.renderer.domElement.setAttribute('aria-label', 'Interactive 3D visualization of HETROFL federated learning workflow');
        }
        
        // Add focus management
        this.container.setAttribute('tabindex', '0');
        this.container.addEventListener('focus', () => {
            this.container.style.outline = '2px solid #4361ee';
        });
        this.container.addEventListener('blur', () => {
            this.container.style.outline = 'none';
        });
        
        // Respect reduced motion preference
        if (this.settingsManager.get('reducedMotion')) {
            this.animationSpeed = 0.1;
        }
    }
    
    checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            return !!context;
        } catch (e) {
            return false;
        }
    }
    
    createScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x06090f); // Match CSS --dark-color
        
        // Add fog for depth perception
        this.scene.fog = new THREE.Fog(0x06090f, 10, 100);
    }
    
    createCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        this.camera.position.set(0, 10, 20);
        this.camera.lookAt(0, 0, 0);
    }
    
    createRenderer() {
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true,
            powerPreference: "high-performance"
        });
        
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    createLights() {
        // Ambient light for overall illumination
        const ambientLight = new THREE.AmbientLight(0x4361ee, 0.3);
        this.scene.add(ambientLight);
        
        // Main directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 50;
        this.scene.add(directionalLight);
        
        // Accent lights for visual interest
        const accentLight1 = new THREE.PointLight(0x05c46b, 0.5, 30);
        accentLight1.position.set(-15, 5, 10);
        this.scene.add(accentLight1);
        
        const accentLight2 = new THREE.PointLight(0xff5e57, 0.5, 30);
        accentLight2.position.set(15, 5, -10);
        this.scene.add(accentLight2);
        
        // Dynamic lighting that responds to workflow state
        this.environmentEffects.dynamicLight = new THREE.PointLight(0x4361ee, 0.3, 50);
        this.environmentEffects.dynamicLight.position.set(0, 15, 0);
        this.scene.add(this.environmentEffects.dynamicLight);
    }
    
    setupInteractionSystem() {
        // Initialize raycaster for mouse interactions
        this.raycaster = new THREE.Raycaster();
        
        // Mouse event listeners
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            this.onMouseMove(event);
        });
        
        this.renderer.domElement.addEventListener('click', (event) => {
            this.onMouseClick(event);
        });
        
        this.renderer.domElement.addEventListener('dblclick', (event) => {
            this.onMouseDoubleClick(event);
        });
        
        // Touch event listeners for mobile
        this.renderer.domElement.addEventListener('touchstart', (event) => {
            this.onTouchStart(event);
        });
        
        this.renderer.domElement.addEventListener('touchmove', (event) => {
            this.onTouchMove(event);
        });
        
        // Keyboard event listeners
        document.addEventListener('keydown', (event) => {
            this.onKeyDown(event);
        });
    }
    
    createAdvancedSystems() {
        // Initialize particle system
        this.particleSystem = new ParticleSystem(this.scene);
        
        // Initialize animation manager
        this.animationManager = new AnimationManager();
        
        // Create ambient particles
        this.particleSystem.createAmbientParticles(30);
        
        console.log('Advanced systems initialized');
    }
    
    createEnvironmentalEffects() {
        // Dynamic background gradient
        this.createDynamicBackground();
        
        // Enhanced fog effects
        this.scene.fog = new THREE.FogExp2(0x06090f, 0.01);
        
        // Atmospheric particles
        this.createAtmosphericEffects();
        
        console.log('Environmental effects created');
    }
    
    createDynamicBackground() {
        // Create a large sphere for dynamic background
        const backgroundGeometry = new THREE.SphereGeometry(200, 32, 32);
        const backgroundMaterial = new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0 },
                color1: { value: new THREE.Color(0x06090f) },
                color2: { value: new THREE.Color(0x1a2238) },
                color3: { value: new THREE.Color(0x4361ee) }
            },
            vertexShader: `
                varying vec3 vPosition;
                void main() {
                    vPosition = position;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform float time;
                uniform vec3 color1;
                uniform vec3 color2;
                uniform vec3 color3;
                varying vec3 vPosition;
                
                void main() {
                    float noise = sin(vPosition.x * 0.01 + time) * 
                                 cos(vPosition.y * 0.01 + time) * 
                                 sin(vPosition.z * 0.01 + time);
                    
                    vec3 color = mix(color1, color2, (noise + 1.0) * 0.5);
                    color = mix(color, color3, abs(sin(time * 0.5)) * 0.1);
                    
                    gl_FragColor = vec4(color, 1.0);
                }
            `,
            side: THREE.BackSide
        });
        
        this.environmentEffects.background = new THREE.Mesh(backgroundGeometry, backgroundMaterial);
        this.scene.add(this.environmentEffects.background);
    }
    
    createAtmosphericEffects() {
        // Create floating energy fields
        const fieldGeometry = new THREE.RingGeometry(5, 8, 16);
        const fieldMaterial = new THREE.MeshBasicMaterial({
            color: 0x4361ee,
            transparent: true,
            opacity: 0.1,
            side: THREE.DoubleSide
        });
        
        for (let i = 0; i < 3; i++) {
            const field = new THREE.Mesh(fieldGeometry, fieldMaterial.clone());
            field.position.set(
                (Math.random() - 0.5) * 40,
                (Math.random() - 0.5) * 20,
                (Math.random() - 0.5) * 40
            );
            field.rotation.x = Math.random() * Math.PI;
            field.rotation.y = Math.random() * Math.PI;
            this.scene.add(field);
            
            // Store for animation
            if (!this.environmentEffects.fields) {
                this.environmentEffects.fields = [];
            }
            this.environmentEffects.fields.push(field);
        }
    }
    
    setupDataIntegration() {
        // Start real-time data fetching
        this.startDataUpdates();
        
        // Initialize workflow steps
        this.workflowSteps = [
            { name: 'Data Loading', duration: 2000, nodes: ['dataSources'] },
            { name: 'Local Training', duration: 3000, nodes: ['localModels'] },
            { name: 'Parameter Extraction', duration: 1500, nodes: ['localModels'] },
            { name: 'Federated Aggregation', duration: 2500, nodes: ['aggregation'] },
            { name: 'Global Model Update', duration: 2000, nodes: ['globalModel'] },
            { name: 'Model Evaluation', duration: 1500, nodes: ['evaluation'] },
            { name: 'Distribution', duration: 2000, nodes: ['globalModel', 'localModels'] }
        ];
    }
    
    startDataUpdates() {
        // Fetch real-time data every 5 seconds
        this.dataUpdateInterval = setInterval(() => {
            this.fetchRealTimeData();
        }, 5000);
        
        // Initial fetch
        this.fetchRealTimeData();
    }
    
    async fetchRealTimeData() {
        try {
            // Fetch latest metrics from Flask API
            const response = await fetch('/api/metrics/latest');
            if (response.ok) {
                const data = await response.json();
                this.updateNodesWithData(data);
            }
        } catch (error) {
            console.warn('Failed to fetch real-time data:', error);
        }
    }
    
    updateNodesWithData(data) {
        // Update node colors and sizes based on performance data
        if (data && data.models) {
            Object.entries(data.models).forEach(([modelName, metrics]) => {
                const accuracy = metrics.accuracy || 0;
                const color = this.getPerformanceColor(accuracy);
                const scale = 0.8 + (accuracy * 0.4); // Scale between 0.8 and 1.2
                
                // Find corresponding node and update
                Object.values(this.nodes).forEach(nodeArray => {
                    nodeArray.forEach(node => {
                        if (node.userData.name.toLowerCase().includes(modelName.toLowerCase())) {
                            // Update color
                            if (node.children[0] && node.children[0].material) {
                                node.children[0].material.color.setHex(color);
                            }
                            
                            // Update scale
                            if (this.animationManager && this.animationManager.isGSAPAvailable) {
                                gsap.to(node.scale, {
                                    duration: 1,
                                    x: scale, y: scale, z: scale,
                                    ease: "power2.inOut"
                                });
                            } else {
                                node.scale.setScalar(scale);
                            }
                        }
                    });
                });
            });
        }
    }
    
    getPerformanceColor(accuracy) {
        if (accuracy >= 0.9) return 0x05c46b; // Green for excellent
        if (accuracy >= 0.8) return 0x3dc7ff; // Blue for good
        if (accuracy >= 0.7) return 0xffa801; // Orange for fair
        return 0xff5e57; // Red for poor
    }
    
    createControls() {
        // Check if OrbitControls is available (would be loaded separately)
        if (typeof THREE.OrbitControls !== 'undefined') {
            this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.enableZoom = true;
            this.controls.enablePan = true;
            this.controls.maxDistance = 50;
            this.controls.minDistance = 5;
        } else {
            // Fallback to basic mouse controls
            this.setupBasicControls();
        }
    }
    
    setupBasicControls() {
        let isMouseDown = false;
        let mouseX = 0;
        let mouseY = 0;
        let targetRotationX = 0;
        let targetRotationY = 0;
        let currentRotationX = 0;
        let currentRotationY = 0;
        
        this.renderer.domElement.addEventListener('mousedown', (event) => {
            isMouseDown = true;
            mouseX = event.clientX;
            mouseY = event.clientY;
        });
        
        this.renderer.domElement.addEventListener('mousemove', (event) => {
            if (!isMouseDown) return;
            
            const deltaX = event.clientX - mouseX;
            const deltaY = event.clientY - mouseY;
            
            targetRotationY += deltaX * 0.01;
            targetRotationX += deltaY * 0.01;
            
            mouseX = event.clientX;
            mouseY = event.clientY;
        });
        
        this.renderer.domElement.addEventListener('mouseup', () => {
            isMouseDown = false;
        });
        
        // Smooth camera rotation
        const updateCameraRotation = () => {
            currentRotationX += (targetRotationX - currentRotationX) * 0.05;
            currentRotationY += (targetRotationY - currentRotationY) * 0.05;
            
            const radius = 20;
            this.camera.position.x = Math.sin(currentRotationY) * radius;
            this.camera.position.z = Math.cos(currentRotationY) * radius;
            this.camera.position.y = 10 + Math.sin(currentRotationX) * 10;
            this.camera.lookAt(0, 0, 0);
            
            requestAnimationFrame(updateCameraRotation);
        };
        updateCameraRotation();
        
        // Zoom with mouse wheel
        this.renderer.domElement.addEventListener('wheel', (event) => {
            const zoomSpeed = 0.1;
            const direction = event.deltaY > 0 ? 1 : -1;
            
            this.camera.position.multiplyScalar(1 + direction * zoomSpeed);
            
            // Clamp zoom
            const distance = this.camera.position.length();
            if (distance < 5) {
                this.camera.position.normalize().multiplyScalar(5);
            } else if (distance > 50) {
                this.camera.position.normalize().multiplyScalar(50);
            }
        });
    }
    
    createWorkflowNodes() {
        // Node configurations
        const nodeConfigs = {
            dataSources: [
                { name: 'Dataset A', position: [-15, 2, 5], color: 0xffa801, type: 'data' },
                { name: 'Dataset B', position: [-15, 0, 0], color: 0xffa801, type: 'data' },
                { name: 'Dataset C', position: [-15, -2, -5], color: 0xffa801, type: 'data' }
            ],
            localModels: [
                { name: 'XGBoost', position: [-5, 2, 5], color: 0x05c46b, type: 'local' },
                { name: 'Random Forest', position: [-5, 0, 0], color: 0x05c46b, type: 'local' },
                { name: 'CatBoost', position: [-5, -2, -5], color: 0x05c46b, type: 'local' }
            ],
            aggregation: [
                { name: 'Aggregation', position: [5, 0, 0], color: 0x3dc7ff, type: 'aggregation' }
            ],
            globalModel: [
                { name: 'Global Model', position: [15, 0, 0], color: 0x4361ee, type: 'global' }
            ],
            evaluation: [
                { name: 'Evaluation', position: [15, 4, 0], color: 0xff5e57, type: 'evaluation' }
            ]
        };
        
        // Create nodes for each category
        Object.entries(nodeConfigs).forEach(([category, nodes]) => {
            this.nodes[category] = [];
            nodes.forEach((config, index) => {
                const node = this.createNode(config);
                this.nodes[category].push(node);
                this.scene.add(node);
            });
        });
    }
    
    createNode(config) {
        const group = new THREE.Group();
        group.userData = { 
            ...config, 
            active: false, 
            hovered: false,
            selected: false,
            performance: { accuracy: 0.5, loss: 0.5 }
        };
        
        // Main node geometry with enhanced details
        const geometry = this.getAdvancedNodeGeometry(config.type);
        const material = this.createAdvancedMaterial(config);
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = { isNode: true, nodeGroup: group };
        group.add(mesh);
        
        // Holographic outline effect
        const outlineGeometry = geometry.clone();
        const outlineMaterial = new THREE.MeshBasicMaterial({
            color: config.color,
            transparent: true,
            opacity: 0.3,
            side: THREE.BackSide
        });
        const outlineMesh = new THREE.Mesh(outlineGeometry, outlineMaterial);
        outlineMesh.scale.multiplyScalar(1.05);
        group.add(outlineMesh);
        
        // Energy field effect
        const fieldGeometry = new THREE.RingGeometry(1.5, 2, 16);
        const fieldMaterial = new THREE.MeshBasicMaterial({
            color: config.color,
            transparent: true,
            opacity: 0.1,
            side: THREE.DoubleSide
        });
        const fieldMesh = new THREE.Mesh(fieldGeometry, fieldMaterial);
        fieldMesh.rotation.x = Math.PI / 2;
        group.add(fieldMesh);
        
        // Performance indicator (small sphere above node)
        const indicatorGeometry = new THREE.SphereGeometry(0.2, 8, 8);
        const indicatorMaterial = new THREE.MeshBasicMaterial({
            color: 0x05c46b,
            transparent: true,
            opacity: 0.8
        });
        const indicator = new THREE.Mesh(indicatorGeometry, indicatorMaterial);
        indicator.position.y = 2.5;
        group.add(indicator);
        group.userData.performanceIndicator = indicator;
        
        // Information panel (initially hidden)
        this.createNodeInfoPanel(group, config);
        
        // Position the group
        group.position.set(...config.position);
        
        // Add floating animation
        this.addFloatingAnimation(group);
        
        // Add entrance animation
        if (this.animationManager) {
            this.animationManager.animateNodeEntrance(group, Math.random() * 2);
        }
        
        return group;
    }
    
    getAdvancedNodeGeometry(type) {
        switch (type) {
            case 'data':
                // Enhanced cylinder with beveled edges
                const dataGeometry = new THREE.CylinderGeometry(0.8, 0.8, 1.5, 12);
                return dataGeometry;
            case 'local':
                // Rounded box geometry
                const localGeometry = new THREE.BoxGeometry(1.5, 1.5, 1.5, 2, 2, 2);
                return localGeometry;
            case 'aggregation':
                // Complex octahedron with additional details
                const aggGeometry = new THREE.OctahedronGeometry(1.2, 1);
                return aggGeometry;
            case 'global':
                // Icosphere for smoother appearance
                const globalGeometry = new THREE.IcosahedronGeometry(1.5, 1);
                return globalGeometry;
            case 'evaluation':
                // Double cone (diamond shape)
                const evalGeometry = new THREE.ConeGeometry(1, 2, 8);
                return evalGeometry;
            default:
                return new THREE.BoxGeometry(1, 1, 1);
        }
    }
    
    createAdvancedMaterial(config) {
        // PBR material with enhanced properties
        const material = new THREE.MeshStandardMaterial({
            color: config.color,
            transparent: true,
            opacity: 0.9,
            metalness: 0.3,
            roughness: 0.4,
            emissive: new THREE.Color(config.color).multiplyScalar(0.1),
            emissiveIntensity: 0.2
        });
        
        // Add normal map for surface detail (if available)
        if (this.textureLoader) {
            // Placeholder for normal maps
            // material.normalMap = this.textureLoader.load('/static/textures/normal.jpg');
        }
        
        return material;
    }
    
    createNodeInfoPanel(group, config) {
        // Create a simple text sprite for node information
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 256;
        canvas.height = 128;
        
        // Draw background
        context.fillStyle = 'rgba(26, 34, 52, 0.9)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw border
        context.strokeStyle = '#4361ee';
        context.lineWidth = 2;
        context.strokeRect(0, 0, canvas.width, canvas.height);
        
        // Draw text
        context.fillStyle = '#ffffff';
        context.font = '16px Arial';
        context.textAlign = 'center';
        context.fillText(config.name, canvas.width / 2, 30);
        context.font = '12px Arial';
        context.fillText(`Type: ${config.type}`, canvas.width / 2, 50);
        context.fillText('Accuracy: --', canvas.width / 2, 70);
        context.fillText('Status: Ready', canvas.width / 2, 90);
        
        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ 
            map: texture,
            transparent: true,
            opacity: 0
        });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(4, 2, 1);
        sprite.position.set(0, 4, 0);
        
        group.add(sprite);
        group.userData.infoPanel = sprite;
    }
    
    getNodeGeometry(type) {
        switch (type) {
            case 'data':
                return new THREE.CylinderGeometry(0.8, 0.8, 1.5, 8);
            case 'local':
                return new THREE.BoxGeometry(1.5, 1.5, 1.5);
            case 'aggregation':
                return new THREE.OctahedronGeometry(1.2);
            case 'global':
                return new THREE.SphereGeometry(1.5, 16, 16);
            case 'evaluation':
                return new THREE.ConeGeometry(1, 2, 8);
            default:
                return new THREE.BoxGeometry(1, 1, 1);
        }
    }
    
    addFloatingAnimation(node) {
        const initialY = node.position.y;
        const amplitude = 0.3;
        const frequency = 0.002;
        const phase = Math.random() * Math.PI * 2;
        
        node.userData.floatAnimation = {
            initialY,
            amplitude,
            frequency,
            phase
        };
    }
    
    createConnections() {
        // Define connection paths
        const connectionPaths = [
            // Data to Local Models
            { from: 'dataSources', to: 'localModels', indices: [[0,0], [1,1], [2,2]] },
            // Local Models to Aggregation
            { from: 'localModels', to: 'aggregation', indices: [[0,0], [1,0], [2,0]] },
            // Aggregation to Global Model
            { from: 'aggregation', to: 'globalModel', indices: [[0,0]] },
            // Global Model to Evaluation
            { from: 'globalModel', to: 'evaluation', indices: [[0,0]] },
            // Global Model back to Local Models
            { from: 'globalModel', to: 'localModels', indices: [[0,0], [0,1], [0,2]] }
        ];
        
        connectionPaths.forEach(path => {
            path.indices.forEach(([fromIdx, toIdx]) => {
                const fromNode = this.nodes[path.from][fromIdx];
                const toNode = this.nodes[path.to][toIdx];
                
                if (fromNode && toNode) {
                    const connection = this.createConnection(fromNode, toNode);
                    this.connections.push(connection);
                    this.scene.add(connection);
                }
            });
        });
    }
    
    createConnection(fromNode, toNode) {
        const fromPos = fromNode.position;
        const toPos = toNode.position;
        
        // Create curved path
        const curve = new THREE.QuadraticBezierCurve3(
            fromPos.clone(),
            new THREE.Vector3(
                (fromPos.x + toPos.x) / 2,
                Math.max(fromPos.y, toPos.y) + 2,
                (fromPos.z + toPos.z) / 2
            ),
            toPos.clone()
        );
        
        const points = curve.getPoints(50);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        
        const material = new THREE.LineBasicMaterial({
            color: 0x2d3748,
            transparent: true,
            opacity: 0.6,
            linewidth: 2
        });
        
        const line = new THREE.Line(geometry, material);
        line.userData = { 
            active: false,
            curve: curve,
            fromNode: fromNode,
            toNode: toNode
        };
        
        return line;
    }
    
    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
        
        // Visibility change for performance
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseAnimation();
            } else {
                this.resumeAnimation();
            }
        });
    }
    
    handleResize() {
        if (!this.container || !this.camera || !this.renderer) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        // Calculate delta time
        const currentTime = performance.now();
        this.deltaTime = (currentTime - this.lastTime) / 1000;
        this.lastTime = currentTime;
        
        // Update controls if available
        if (this.controls && this.controls.update) {
            this.controls.update();
        }
        
        // Update floating animations
        this.updateFloatingAnimations();
        
        // Update data particles
        this.updateDataParticles();
        
        // Update environmental effects
        this.updateEnvironmentalEffects();
        
        // Performance monitoring
        this.updatePerformanceStats();
        
        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }
    
    updateEnvironmentalEffects() {
        const time = Date.now() * 0.001;
        
        // Update dynamic background
        if (this.environmentEffects.background && this.environmentEffects.background.material.uniforms) {
            this.environmentEffects.background.material.uniforms.time.value = time;
        }
        
        // Animate energy fields
        if (this.environmentEffects.fields) {
            this.environmentEffects.fields.forEach((field, index) => {
                field.rotation.z += 0.01 * (index + 1);
                field.material.opacity = 0.1 + Math.sin(time + index) * 0.05;
            });
        }
        
        // Update dynamic lighting
        if (this.environmentEffects.dynamicLight) {
            this.environmentEffects.dynamicLight.intensity = 0.3 + Math.sin(time * 2) * 0.1;
            this.environmentEffects.dynamicLight.color.setHSL(
                (time * 0.1) % 1,
                0.5,
                0.5
            );
        }
    }
    
    updateFloatingAnimations() {
        const time = Date.now();
        
        Object.values(this.nodes).forEach(nodeArray => {
            nodeArray.forEach(node => {
                if (node.userData.floatAnimation) {
                    const anim = node.userData.floatAnimation;
                    node.position.y = anim.initialY + 
                        Math.sin(time * anim.frequency + anim.phase) * anim.amplitude;
                }
            });
        });
    }
    
    updateDataParticles() {
        // Update existing particles
        this.dataParticles = this.dataParticles.filter(particle => {
            particle.progress += 0.01 * this.animationSpeed;
            
            if (particle.progress >= 1) {
                this.scene.remove(particle.mesh);
                return false;
            }
            
            // Update position along curve
            const position = particle.curve.getPoint(particle.progress);
            particle.mesh.position.copy(position);
            
            return true;
        });
        
        // Update particle system
        if (this.particleSystem) {
            this.particleSystem.update(this.deltaTime);
        }
    }
    
    // Interaction event handlers
    onMouseMove(event) {
        // Calculate mouse position in normalized device coordinates
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        // Check for intersections with nodes
        const intersectableObjects = [];
        Object.values(this.nodes).forEach(nodeArray => {
            nodeArray.forEach(node => {
                if (node.children[0]) {
                    intersectableObjects.push(node.children[0]);
                }
            });
        });
        
        const intersects = this.raycaster.intersectObjects(intersectableObjects);
        
        // Handle hover effects
        if (intersects.length > 0) {
            const hoveredObject = intersects[0].object;
            const nodeGroup = hoveredObject.userData.nodeGroup;
            
            if (this.hoveredNode !== nodeGroup) {
                // Remove previous hover effect
                if (this.hoveredNode) {
                    this.removeHoverEffect(this.hoveredNode);
                }
                
                // Add new hover effect
                this.hoveredNode = nodeGroup;
                this.addHoverEffect(nodeGroup);
                
                // Change cursor
                this.renderer.domElement.style.cursor = 'pointer';
            }
        } else {
            // Remove hover effect
            if (this.hoveredNode) {
                this.removeHoverEffect(this.hoveredNode);
                this.hoveredNode = null;
                this.renderer.domElement.style.cursor = 'default';
            }
        }
    }
    
    onMouseClick(event) {
        if (this.hoveredNode) {
            this.selectNode(this.hoveredNode);
        } else {
            this.deselectNode();
        }
    }
    
    onMouseDoubleClick(event) {
        if (this.hoveredNode) {
            // Focus camera on selected node
            this.focusOnNode(this.hoveredNode);
        }
    }
    
    onTouchStart(event) {
        if (event.touches.length === 1) {
            const touch = event.touches[0];
            this.onMouseMove({
                clientX: touch.clientX,
                clientY: touch.clientY
            });
        }
    }
    
    onTouchMove(event) {
        if (event.touches.length === 1) {
            const touch = event.touches[0];
            this.onMouseMove({
                clientX: touch.clientX,
                clientY: touch.clientY
            });
        }
    }
    
    onKeyDown(event) {
        switch (event.code) {
            case 'Space':
                event.preventDefault();
                this.isAnimating ? this.pauseAnimation() : this.startWorkflowAnimation();
                break;
            case 'KeyR':
                this.resetWorkflow();
                break;
            case 'Digit1':
                this.animationManager.animateCamera(this.camera, 'perspective');
                break;
            case 'Digit2':
                this.animationManager.animateCamera(this.camera, 'top');
                break;
            case 'Digit3':
                this.animationManager.animateCamera(this.camera, 'side');
                break;
            case 'Digit4':
                this.animationManager.animateCamera(this.camera, 'overview');
                break;
            case 'Escape':
                this.deselectNode();
                break;
        }
    }
    
    addHoverEffect(node) {
        node.userData.hovered = true;
        
        // Scale up slightly
        if (this.animationManager && this.animationManager.isGSAPAvailable) {
            gsap.to(node.scale, {
                duration: 0.3,
                x: 1.1, y: 1.1, z: 1.1,
                ease: "power2.out"
            });
            
            // Brighten the material
            if (node.children[0] && node.children[0].material) {
                gsap.to(node.children[0].material, {
                    duration: 0.3,
                    emissiveIntensity: 0.5,
                    ease: "power2.out"
                });
            }
        } else {
            node.scale.setScalar(1.1);
        }
        
        // Show info panel
        if (node.userData.infoPanel) {
            if (this.animationManager && this.animationManager.isGSAPAvailable) {
                gsap.to(node.userData.infoPanel.material, {
                    duration: 0.3,
                    opacity: 1,
                    ease: "power2.out"
                });
            } else {
                node.userData.infoPanel.material.opacity = 1;
            }
        }
    }
    
    removeHoverEffect(node) {
        node.userData.hovered = false;
        
        if (!node.userData.selected) {
            // Scale back to normal
            if (this.animationManager && this.animationManager.isGSAPAvailable) {
                gsap.to(node.scale, {
                    duration: 0.3,
                    x: 1, y: 1, z: 1,
                    ease: "power2.out"
                });
                
                // Restore material
                if (node.children[0] && node.children[0].material) {
                    gsap.to(node.children[0].material, {
                        duration: 0.3,
                        emissiveIntensity: 0.2,
                        ease: "power2.out"
                    });
                }
            } else {
                node.scale.setScalar(1);
            }
            
            // Hide info panel
            if (node.userData.infoPanel) {
                if (this.animationManager && this.animationManager.isGSAPAvailable) {
                    gsap.to(node.userData.infoPanel.material, {
                        duration: 0.3,
                        opacity: 0,
                        ease: "power2.out"
                    });
                } else {
                    node.userData.infoPanel.material.opacity = 0;
                }
            }
        }
    }
    
    selectNode(node) {
        // Deselect previous node
        if (this.selectedNode && this.selectedNode !== node) {
            this.deselectNode();
        }
        
        this.selectedNode = node;
        node.userData.selected = true;
        
        // Add selection effects
        if (this.animationManager && this.animationManager.isGSAPAvailable) {
            gsap.to(node.scale, {
                duration: 0.5,
                x: 1.2, y: 1.2, z: 1.2,
                ease: "back.out(1.7)"
            });
        }
        
        // Create burst effect
        if (this.particleSystem) {
            this.particleSystem.createBurstEffect(node.position, node.userData.color, 15);
        }
        
        // Update UI to show node details
        this.updateNodeDetailsUI(node);
        
        console.log('Node selected:', node.userData.name);
    }
    
    deselectNode() {
        if (this.selectedNode) {
            this.selectedNode.userData.selected = false;
            
            // Remove selection effects
            if (this.animationManager && this.animationManager.isGSAPAvailable) {
                gsap.to(this.selectedNode.scale, {
                    duration: 0.3,
                    x: 1, y: 1, z: 1,
                    ease: "power2.out"
                });
            }
            
            this.selectedNode = null;
        }
    }
    
    focusOnNode(node) {
        const targetPosition = node.position.clone();
        targetPosition.add(new THREE.Vector3(5, 5, 5));
        
        if (this.animationManager && this.animationManager.isGSAPAvailable) {
            gsap.to(this.camera.position, {
                duration: 2,
                x: targetPosition.x,
                y: targetPosition.y,
                z: targetPosition.z,
                ease: "power2.inOut",
                onUpdate: () => {
                    this.camera.lookAt(node.position);
                }
            });
        }
    }
    
    updateNodeDetailsUI(node) {
        // Update the info panel with current node details
        const infoPanel = document.querySelector('.workflow-3d-info .info-content');
        if (infoPanel) {
            infoPanel.innerHTML = `
                <div class="selected-node-info">
                    <h4>${node.userData.name}</h4>
                    <p><strong>Type:</strong> ${node.userData.type}</p>
                    <p><strong>Status:</strong> ${node.userData.active ? 'Active' : 'Idle'}</p>
                    <p><strong>Performance:</strong> ${(node.userData.performance.accuracy * 100).toFixed(1)}%</p>
                </div>
            `;
        }
    }
    
    updatePerformanceStats() {
        this.frameCount++;
        const currentTime = performance.now();
        
        if (currentTime - this.lastTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = currentTime;
            
            // Adjust quality based on performance
            if (this.fps < 30) {
                this.renderer.setPixelRatio(1);
            } else if (this.fps > 50) {
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            }
        }
    }
    
    // Public API methods
    startWorkflowAnimation() {
        if (this.isAnimating) return;
        
        this.isAnimating = true;
        this.currentStep = 0;
        this.runAnimationSequence();
    }
    
    pauseAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }
    
    resumeAnimation() {
        if (!this.animationId) {
            this.animate();
        }
    }
    
    resetWorkflow() {
        this.isAnimating = false;
        this.currentStep = 0;
        
        // Reset all nodes
        Object.values(this.nodes).forEach(nodeArray => {
            nodeArray.forEach(node => {
                node.userData.active = false;
                node.children[0].material.opacity = 0.3;
                node.children[1].material.opacity = 0.1;
            });
        });
        
        // Reset all connections
        this.connections.forEach(connection => {
            connection.userData.active = false;
            connection.material.opacity = 0.3;
        });
        
        // Clear particles
        this.dataParticles.forEach(particle => {
            this.scene.remove(particle.mesh);
        });
        this.dataParticles = [];
    }
    
    setAnimationSpeed(speed) {
        this.animationSpeed = Math.max(0.1, Math.min(3.0, speed));
    }
    
    runAnimationSequence() {
        console.log('Starting advanced workflow animation sequence');
        
        if (this.animationManager && this.animationManager.isGSAPAvailable) {
            this.runGSAPAnimationSequence();
        } else {
            this.runFallbackAnimationSequence();
        }
    }
    
    runGSAPAnimationSequence() {
        const timeline = this.animationManager.createTimeline();
        if (!timeline) return;
        
        // Step 1: Data Loading
        timeline.to({}, { duration: 0.5 })
            .call(() => {
                this.activateWorkflowStep(0);
                this.activateNodeCategory('dataSources');
                this.createDataFlowParticles('dataSources', 'localModels');
            })
            
        // Step 2: Local Training
        .to({}, { duration: 1.5 })
            .call(() => {
                this.activateWorkflowStep(1);
                this.activateNodeCategory('localModels');
                this.createTrainingEffects('localModels');
            })
            
        // Step 3: Parameter Extraction
        .to({}, { duration: 1 })
            .call(() => {
                this.activateWorkflowStep(2);
                this.createParameterExtractionEffects('localModels');
            })
            
        // Step 4: Federated Aggregation
        .to({}, { duration: 2 })
            .call(() => {
                this.activateWorkflowStep(3);
                this.activateNodeCategory('aggregation');
                this.createDataFlowParticles('localModels', 'aggregation');
                this.createAggregationEffects();
            })
            
        // Step 5: Global Model Update
        .to({}, { duration: 1.5 })
            .call(() => {
                this.activateWorkflowStep(4);
                this.activateNodeCategory('globalModel');
                this.createDataFlowParticles('aggregation', 'globalModel');
            })
            
        // Step 6: Model Evaluation
        .to({}, { duration: 1 })
            .call(() => {
                this.activateWorkflowStep(5);
                this.activateNodeCategory('evaluation');
                this.createDataFlowParticles('globalModel', 'evaluation');
            })
            
        // Step 7: Distribution
        .to({}, { duration: 1.5 })
            .call(() => {
                this.activateWorkflowStep(6);
                this.createDataFlowParticles('globalModel', 'localModels');
                this.createDistributionEffects();
            })
            
        // Complete
        .call(() => {
            this.completeWorkflowAnimation();
        });
        
        timeline.play();
    }
    
    runFallbackAnimationSequence() {
        let stepIndex = 0;
        const executeStep = () => {
            if (stepIndex >= this.workflowSteps.length) {
                this.completeWorkflowAnimation();
                return;
            }
            
            const step = this.workflowSteps[stepIndex];
            this.activateWorkflowStep(stepIndex);
            
            // Activate corresponding nodes
            step.nodes.forEach(nodeCategory => {
                this.activateNodeCategory(nodeCategory);
            });
            
            stepIndex++;
            setTimeout(executeStep, step.duration);
        };
        
        executeStep();
    }
    
    activateWorkflowStep(stepIndex) {
        // Update UI to show current step
        document.querySelectorAll('.workflow-step').forEach((step, index) => {
            step.classList.toggle('active', index === stepIndex);
        });
        
        this.currentStep = stepIndex;
        console.log(`Workflow step ${stepIndex + 1}: ${this.workflowSteps[stepIndex] ? this.workflowSteps[stepIndex].name : 'Unknown'}`);
    }
    
    activateNodeCategory(category) {
        if (!this.nodes[category]) return;
        
        this.nodes[category].forEach(node => {
            node.userData.active = true;
            
            // Visual activation effects
            if (this.animationManager && this.animationManager.isGSAPAvailable) {
                // Pulse animation
                gsap.to(node.scale, {
                    duration: 0.5,
                    x: 1.3, y: 1.3, z: 1.3,
                    yoyo: true,
                    repeat: 1,
                    ease: "power2.inOut"
                });
                
                // Brighten material
                if (node.children[0] && node.children[0].material) {
                    gsap.to(node.children[0].material, {
                        duration: 0.5,
                        emissiveIntensity: 0.8,
                        yoyo: true,
                        repeat: 1,
                        ease: "power2.inOut"
                    });
                }
            }
            
            // Create burst effect
            if (this.particleSystem) {
                this.particleSystem.createBurstEffect(node.position, node.userData.color, 10);
            }
        });
    }
    
    createDataFlowParticles(fromCategory, toCategory) {
        if (!this.nodes[fromCategory] || !this.nodes[toCategory]) return;
        
        this.nodes[fromCategory].forEach((fromNode, fromIndex) => {
            this.nodes[toCategory].forEach((toNode, toIndex) => {
                // Create multiple particles for each connection
                for (let i = 0; i < 3; i++) {
                    setTimeout(() => {
                        const curve = this.createConnectionCurve(fromNode.position, toNode.position);
                        if (this.particleSystem) {
                            this.particleSystem.createDataFlowParticle(curve, fromNode.userData.color, 0.02);
                        }
                    }, i * 200);
                }
            });
        });
    }
    
    createConnectionCurve(fromPos, toPos) {
        return new THREE.QuadraticBezierCurve3(
            fromPos.clone(),
            new THREE.Vector3(
                (fromPos.x + toPos.x) / 2,
                Math.max(fromPos.y, toPos.y) + 3,
                (fromPos.z + toPos.z) / 2
            ),
            toPos.clone()
        );
    }
    
    createTrainingEffects(category) {
        if (!this.nodes[category]) return;
        
        this.nodes[category].forEach(node => {
            // Create training particle effects around the node
            for (let i = 0; i < 20; i++) {
                setTimeout(() => {
                    const offset = new THREE.Vector3(
                        (Math.random() - 0.5) * 4,
                        (Math.random() - 0.5) * 4,
                        (Math.random() - 0.5) * 4
                    );
                    const position = node.position.clone().add(offset);
                    
                    if (this.particleSystem) {
                        this.particleSystem.createBurstEffect(position, 0x05c46b, 5);
                    }
                }, i * 100);
            }
        });
    }
    
    createParameterExtractionEffects(category) {
        if (!this.nodes[category]) return;
        
        this.nodes[category].forEach(node => {
            // Create spiral particle effect
            const spiralParticles = 30;
            for (let i = 0; i < spiralParticles; i++) {
                setTimeout(() => {
                    const angle = (i / spiralParticles) * Math.PI * 4;
                    const radius = 2 + (i / spiralParticles) * 2;
                    const position = node.position.clone().add(new THREE.Vector3(
                        Math.cos(angle) * radius,
                        (i / spiralParticles) * 3,
                        Math.sin(angle) * radius
                    ));
                    
                    if (this.particleSystem) {
                        this.particleSystem.createBurstEffect(position, 0x3dc7ff, 3);
                    }
                }, i * 50);
            }
        });
    }
    
    createAggregationEffects() {
        if (!this.nodes.aggregation || !this.nodes.aggregation[0]) return;
        
        const aggregationNode = this.nodes.aggregation[0];
        
        // Create convergence effect
        for (let i = 0; i < 50; i++) {
            setTimeout(() => {
                const startPos = new THREE.Vector3(
                    aggregationNode.position.x + (Math.random() - 0.5) * 20,
                    aggregationNode.position.y + (Math.random() - 0.5) * 10,
                    aggregationNode.position.z + (Math.random() - 0.5) * 20
                );
                
                const curve = new THREE.QuadraticBezierCurve3(
                    startPos,
                    startPos.clone().lerp(aggregationNode.position, 0.5).add(new THREE.Vector3(0, 2, 0)),
                    aggregationNode.position
                );
                
                if (this.particleSystem) {
                    this.particleSystem.createDataFlowParticle(curve, 0x4361ee, 0.03);
                }
            }, i * 30);
        }
    }
    
    createDistributionEffects() {
        if (!this.nodes.globalModel || !this.nodes.globalModel[0]) return;
        
        const globalNode = this.nodes.globalModel[0];
        
        // Create distribution wave effect
        for (let i = 0; i < 60; i++) {
            setTimeout(() => {
                const angle = (i / 60) * Math.PI * 2;
                const radius = 5 + (i / 60) * 15;
                const endPos = globalNode.position.clone().add(new THREE.Vector3(
                    Math.cos(angle) * radius,
                    (Math.random() - 0.5) * 5,
                    Math.sin(angle) * radius
                ));
                
                const curve = new THREE.QuadraticBezierCurve3(
                    globalNode.position,
                    globalNode.position.clone().lerp(endPos, 0.5).add(new THREE.Vector3(0, 3, 0)),
                    endPos
                );
                
                if (this.particleSystem) {
                    this.particleSystem.createDataFlowParticle(curve, 0xff5e57, 0.025);
                }
            }, i * 25);
        }
    }
    
    completeWorkflowAnimation() {
        console.log('Workflow animation completed');
        this.isAnimating = false;
        
        // Reset all workflow steps
        document.querySelectorAll('.workflow-step').forEach(step => {
            step.classList.remove('active');
        });
        
        // Create completion burst effect
        Object.values(this.nodes).forEach(nodeArray => {
            nodeArray.forEach(node => {
                if (this.particleSystem) {
                    this.particleSystem.createBurstEffect(node.position, 0xffa801, 20);
                }
            });
        });
        
        // Auto-restart after delay if continuous mode is enabled
        setTimeout(() => {
            if (this.isAnimating) {
                this.runAnimationSequence();
            }
        }, 3000);
    }
    
    showFallbackMessage() {
        this.container.innerHTML = `
            <div class="fallback-message">
                <h3>3D Visualization Unavailable</h3>
                <p>Your browser doesn't support WebGL or there was an error loading the 3D scene.</p>
                <p>Please try updating your browser or enabling hardware acceleration.</p>
            </div>
        `;
    }
    
    // Camera preset methods
    setCameraView(viewType) {
        if (this.animationManager) {
            this.animationManager.animateCamera(this.camera, viewType);
        }
        
        // Update UI to reflect active view
        document.querySelectorAll('[id^="view-"]').forEach(btn => {
            btn.classList.remove('active');
        });
        const viewButton = document.getElementById(`view-${viewType}`);
        if (viewButton) {
            viewButton.classList.add('active');
        }
    }
    
    // Export capabilities
    captureScreenshot(filename = 'hetrofl-3d-workflow.png') {
        try {
            const canvas = this.renderer.domElement;
            const link = document.createElement('a');
            link.download = filename;
            link.href = canvas.toDataURL('image/png');
            link.click();
            console.log('Screenshot captured:', filename);
        } catch (error) {
            console.error('Failed to capture screenshot:', error);
        }
    }
    
    startVideoCapture() {
        try {
            const canvas = this.renderer.domElement;
            const stream = canvas.captureStream(30); // 30 FPS
            
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp9'
            });
            
            this.recordedChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.recordedChunks, {
                    type: 'video/webm'
                });
                
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.download = 'hetrofl-3d-workflow.webm';
                link.href = url;
                link.click();
                
                URL.revokeObjectURL(url);
                console.log('Video capture completed');
            };
            
            this.mediaRecorder.start();
            console.log('Video capture started');
            
        } catch (error) {
            console.error('Failed to start video capture:', error);
        }
    }
    
    stopVideoCapture() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
    }
    
    // Performance optimization methods
    setQualityLevel(level) {
        switch (level) {
            case 'low':
                this.renderer.setPixelRatio(1);
                this.particleSystem.maxParticles = 500;
                this.scene.fog.density = 0.02;
                break;
            case 'medium':
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
                this.particleSystem.maxParticles = 750;
                this.scene.fog.density = 0.015;
                break;
            case 'high':
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                this.particleSystem.maxParticles = 1000;
                this.scene.fog.density = 0.01;
                break;
        }
        console.log(`Quality level set to: ${level}`);
    }
    
    // Accessibility features
    enableKeyboardNavigation() {
        document.addEventListener('keydown', (event) => {
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                return; // Don't interfere with form inputs
            }
            
            switch (event.code) {
                case 'Tab':
                    event.preventDefault();
                    this.focusNextNode();
                    break;
                case 'Enter':
                    if (this.hoveredNode) {
                        this.selectNode(this.hoveredNode);
                    }
                    break;
                case 'ArrowUp':
                case 'ArrowDown':
                case 'ArrowLeft':
                case 'ArrowRight':
                    event.preventDefault();
                    this.navigateWithArrows(event.code);
                    break;
            }
        });
    }
    
    focusNextNode() {
        const allNodes = [];
        Object.values(this.nodes).forEach(nodeArray => {
            allNodes.push(...nodeArray);
        });
        
        if (allNodes.length === 0) return;
        
        let currentIndex = this.hoveredNode ? allNodes.indexOf(this.hoveredNode) : -1;
        currentIndex = (currentIndex + 1) % allNodes.length;
        
        if (this.hoveredNode) {
            this.removeHoverEffect(this.hoveredNode);
        }
        
        this.hoveredNode = allNodes[currentIndex];
        this.addHoverEffect(this.hoveredNode);
        this.focusOnNode(this.hoveredNode);
    }
    
    navigateWithArrows(direction) {
        if (!this.hoveredNode) {
            this.focusNextNode();
            return;
        }
        
        const allNodes = [];
        Object.values(this.nodes).forEach(nodeArray => {
            allNodes.push(...nodeArray);
        });
        
        let targetNode = null;
        let minDistance = Infinity;
        const currentPos = this.hoveredNode.position;
        
        allNodes.forEach(node => {
            if (node === this.hoveredNode) return;
            
            const pos = node.position;
            const distance = currentPos.distanceTo(pos);
            
            let isValidDirection = false;
            switch (direction) {
                case 'ArrowUp':
                    isValidDirection = pos.y > currentPos.y;
                    break;
                case 'ArrowDown':
                    isValidDirection = pos.y < currentPos.y;
                    break;
                case 'ArrowLeft':
                    isValidDirection = pos.x < currentPos.x;
                    break;
                case 'ArrowRight':
                    isValidDirection = pos.x > currentPos.x;
                    break;
            }
            
            if (isValidDirection && distance < minDistance) {
                minDistance = distance;
                targetNode = node;
            }
        });
        
        if (targetNode) {
            this.removeHoverEffect(this.hoveredNode);
            this.hoveredNode = targetNode;
            this.addHoverEffect(this.hoveredNode);
        }
    }
    
    // Cleanup
    destroy() {
        // Stop data updates
        if (this.dataUpdateInterval) {
            clearInterval(this.dataUpdateInterval);
        }
        
        // Stop video capture if active
        this.stopVideoCapture();
        
        // Stop animations
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        // Clean up animation manager
        if (this.animationManager) {
            this.animationManager.stopAll();
        }
        
        // Clean up particle system
        if (this.particleSystem) {
            this.particleSystem.clear();
        }
        
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        // Clean up geometries and materials
        this.scene.traverse((object) => {
            if (object.geometry) {
                object.geometry.dispose();
            }
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose());
                } else {
                    object.material.dispose();
                }
            }
        });
        
        console.log('HETROFL 3D Workflow destroyed');
    }
    
    // Production optimization methods
    reduceQuality() {
        const currentQuality = this.settingsManager.get('quality');
        const qualities = ['ultra', 'high', 'medium', 'low'];
        const currentIndex = qualities.indexOf(currentQuality);
        
        if (currentIndex < qualities.length - 1) {
            const newQuality = qualities[currentIndex + 1];
            this.setQuality(newQuality);
            console.log(`Quality reduced to ${newQuality} due to performance`);
        }
    }
    
    setQuality(quality) {
        this.settingsManager.applyQualityPreset(quality);
        this.applyQualitySettings();
        
        // Show notification
        this.showNotification(`Quality set to ${quality.toUpperCase()}`);
    }
    
    toggleAnimation() {
        if (this.isAnimating) {
            this.pauseAnimation();
        } else {
            this.resumeAnimation();
        }
    }
    
    pauseAnimation() {
        this.isPaused = true;
        this.isAnimating = false;
        
        // Update UI
        const playBtn = document.getElementById('start-3d-animation');
        if (playBtn) {
            playBtn.innerHTML = '<i class="fas fa-play"></i> Resume';
        }
    }
    
    resumeAnimation() {
        this.isPaused = false;
        this.isAnimating = true;
        
        // Update UI
        const playBtn = document.getElementById('start-3d-animation');
        if (playBtn) {
            playBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';
        }
    }
    
    resetCamera() {
        if (this.camera && this.controls) {
            if (this.animationManager && this.animationManager.isGSAPAvailable) {
                gsap.to(this.camera.position, {
                    duration: 1,
                    x: 0, y: 10, z: 20,
                    ease: "power2.inOut"
                });
                gsap.to(this.controls.target, {
                    duration: 1,
                    x: 0, y: 0, z: 0,
                    ease: "power2.inOut",
                    onUpdate: () => this.controls.update()
                });
            } else {
                this.camera.position.set(0, 10, 20);
                this.controls.target.set(0, 0, 0);
                this.controls.update();
            }
        }
    }
    
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            this.container.requestFullscreen().catch(err => {
                console.warn('Failed to enter fullscreen:', err);
            });
        } else {
            document.exitFullscreen();
        }
    }
    
    createPerformanceDisplay() {
        const perfDisplay = document.createElement('div');
        perfDisplay.className = 'performance-display';
        perfDisplay.innerHTML = `
            <div class="perf-item">FPS: <span id="fps-counter">60</span></div>
            <div class="perf-item">Memory: <span id="memory-counter">0</span>MB</div>
            <div class="perf-item">Draw Calls: <span id="draw-calls">0</span></div>
        `;
        
        this.container.appendChild(perfDisplay);
        
        // Update performance display
        setInterval(() => {
            document.getElementById('fps-counter').textContent = this.performanceMonitor.metrics.fps;
            document.getElementById('memory-counter').textContent = this.performanceMonitor.metrics.memoryUsage;
            document.getElementById('draw-calls').textContent = this.performanceMonitor.metrics.drawCalls;
        }, 1000);
    }
    
    startDataIntegration() {
        // Connect to real HETROFL data
        this.dataUpdateInterval = setInterval(() => {
            this.fetchRealTimeData();
        }, 5000);
    }
    
    async fetchRealTimeData() {
        try {
            const response = await fetch('/api/workflow-status');
            if (response.ok) {
                const data = await response.json();
                this.updateVisualizationWithRealData(data);
            }
        } catch (error) {
            console.warn('Failed to fetch real-time data:', error);
        }
    }
    
    updateVisualizationWithRealData(data) {
        // Update node states based on real data
        if (data.training_status) {
            this.updateNodeState('globalModel', data.training_status.global_model);
            this.updateNodeState('localModels', data.training_status.local_models);
        }
        
        // Update metrics display
        if (data.metrics) {
            this.updateMetricsDisplay(data.metrics);
        }
    }
    
    updateNodeState(nodeType, state) {
        const nodes = this.nodes[nodeType];
        if (!nodes) return;
        
        nodes.forEach((node, index) => {
            if (state && state[index]) {
                const nodeState = state[index];
                
                // Update node color based on state
                if (node.children[0] && node.children[0].material) {
                    const material = node.children[0].material;
                    
                    switch (nodeState.status) {
                        case 'training':
                            material.color.setHex(0x4361ee); // Blue
                            break;
                        case 'complete':
                            material.color.setHex(0x05c46b); // Green
                            break;
                        case 'error':
                            material.color.setHex(0xff5e57); // Red
                            break;
                        default:
                            material.color.setHex(0x6c757d); // Gray
                    }
                }
                
                // Update node info
                if (node.userData.info) {
                    node.userData.info = { ...node.userData.info, ...nodeState };
                }
            }
        });
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `workflow-notification ${type}`;
        notification.textContent = message;
        
        this.container.appendChild(notification);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }
    
    // Memory management and cleanup
    dispose() {
        this.isDisposed = true;
        
        // Clear intervals
        if (this.dataUpdateInterval) {
            clearInterval(this.dataUpdateInterval);
        }
        
        // Dispose all tracked resources
        this.disposables.forEach(resource => {
            if (resource && typeof resource.dispose === 'function') {
                resource.dispose();
            }
        });
        
        // Call parent dispose
        this.destroy();
    }
    
    addDisposable(resource) {
        if (resource && typeof resource.dispose === 'function') {
            this.disposables.push(resource);
        }
    }
}

// Global instance
let hetrofl3D = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    const container = document.getElementById('workflow-3d-container');
    if (container) {
        hetrofl3D = new HETROFL3DWorkflow('workflow-3d-container');
        
        // Expose to global scope for external control
        window.hetrofl3D = hetrofl3D;
    }
});

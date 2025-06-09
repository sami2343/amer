# HETROFL 3D Workflow - Performance Optimization Guide

## Overview

This guide provides comprehensive information on optimizing the HETROFL 3D Workflow visualization for maximum performance across different devices and scenarios.

## Performance Monitoring

### Built-in Performance Metrics

The system includes real-time performance monitoring:

```javascript
// Access performance metrics
const metrics = workflow.performanceMonitor.metrics;
console.log(`FPS: ${metrics.fps}`);
console.log(`Memory: ${metrics.memoryUsage}MB`);
console.log(`Draw Calls: ${metrics.drawCalls}`);
```

### Performance Display (Development)

Enable performance overlay for development:

```javascript
// Automatically enabled on localhost
if (window.location.hostname === 'localhost') {
    workflow.createPerformanceDisplay();
}
```

### Key Performance Indicators

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| FPS | 60+ | 30-59 | <30 |
| Memory Usage | <500MB | 500-1000MB | >1000MB |
| Draw Calls | <100 | 100-200 | >200 |
| Triangles | <50K | 50K-100K | >100K |

## Quality Settings Optimization

### Automatic Quality Scaling

The system automatically adjusts quality based on performance:

```javascript
// Enable auto quality adjustment
workflow.settingsManager.set('autoQuality', true);

// Manual quality check
if (workflow.performanceMonitor.shouldReduceQuality()) {
    workflow.reduceQuality();
}
```

### Quality Presets Performance Impact

| Setting | Performance Impact | Visual Quality | Recommended For |
|---------|-------------------|----------------|-----------------|
| **Low** | Minimal | Basic | Mobile, older devices |
| **Medium** | Moderate | Good | Most desktop computers |
| **High** | High | Excellent | Gaming PCs, workstations |
| **Ultra** | Maximum | Outstanding | High-end systems only |

### Custom Quality Configuration

```javascript
// Custom quality settings
const customSettings = {
    particles: true,
    fog: false,           // Disable fog for better performance
    glow: true,
    shadows: false,       // Disable shadows for mobile
    antialiasing: true,
    pixelRatio: 1,        // Lower pixel ratio for performance
    maxParticles: 500     // Reduce particle count
};

workflow.settingsManager.settings = { ...workflow.settingsManager.settings, ...customSettings };
```

## Device-Specific Optimizations

### Mobile Devices

#### Automatic Mobile Detection
```javascript
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

if (isMobile) {
    workflow.settingsManager.applyQualityPreset('medium');
    workflow.settingsManager.set('shadows', false);
    workflow.settingsManager.set('maxParticles', 200);
}
```

#### Mobile-Specific Settings
- **Pixel Ratio**: Limited to 1.5 to prevent excessive memory usage
- **Particles**: Reduced count (200-500 vs 1000+)
- **Shadows**: Disabled by default
- **Antialiasing**: Reduced quality
- **Texture Resolution**: Automatically downscaled

### Desktop Optimization

#### High-End Systems
```javascript
// Detect high-end GPU
const canvas = document.createElement('canvas');
const gl = canvas.getContext('webgl');
const renderer = gl.getParameter(gl.RENDERER);

if (renderer.includes('RTX') || renderer.includes('GTX') || renderer.includes('Radeon')) {
    workflow.settingsManager.applyQualityPreset('high');
}
```

#### Low-End Systems
```javascript
// Conservative settings for older hardware
workflow.settingsManager.applyQualityPreset('low');
workflow.settingsManager.set('pixelRatio', 1);
workflow.settingsManager.set('maxParticles', 100);
```

## Memory Management

### Automatic Memory Management

The system includes comprehensive memory management:

```javascript
// Automatic cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.hetrofl3D) {
        window.hetrofl3D.dispose();
    }
});

// Periodic memory cleanup
setInterval(() => {
    if (workflow.performanceMonitor.metrics.memoryUsage > 800) {
        workflow.cleanupUnusedResources();
    }
}, 30000);
```

### Manual Memory Management

```javascript
// Track disposable resources
workflow.addDisposable(geometry);
workflow.addDisposable(material);
workflow.addDisposable(texture);

// Manual cleanup
workflow.dispose();
```

### Memory Optimization Techniques

1. **Geometry Instancing**: Reuse geometries for similar objects
2. **Material Sharing**: Share materials between objects
3. **Texture Atlasing**: Combine textures to reduce memory
4. **LOD (Level of Detail)**: Use simpler models at distance
5. **Culling**: Don't render objects outside view

## Rendering Optimizations

### Frustum Culling

```javascript
// Automatic frustum culling
workflow.renderer.setFrustumCulling(true);

// Manual culling for specific objects
object.frustumCulled = true;
```

### Level of Detail (LOD)

```javascript
// Implement LOD for complex models
const lod = new THREE.LOD();
lod.addLevel(highDetailMesh, 0);
lod.addLevel(mediumDetailMesh, 50);
lod.addLevel(lowDetailMesh, 100);
```

### Instanced Rendering

```javascript
// Use instanced meshes for repeated objects
const instancedGeometry = new THREE.InstancedBufferGeometry();
const instancedMesh = new THREE.InstancedMesh(geometry, material, count);
```

## Network Optimization

### Data Fetching Optimization

```javascript
// Optimize API calls
const dataCache = new Map();
const CACHE_DURATION = 5000; // 5 seconds

async function fetchOptimizedData() {
    const now = Date.now();
    const cached = dataCache.get('workflow-status');
    
    if (cached && (now - cached.timestamp) < CACHE_DURATION) {
        return cached.data;
    }
    
    const data = await fetch('/api/workflow-status').then(r => r.json());
    dataCache.set('workflow-status', { data, timestamp: now });
    return data;
}
```

### Bandwidth Optimization

- **Data Compression**: Use gzip compression for API responses
- **Delta Updates**: Only send changed data
- **Batch Requests**: Combine multiple API calls
- **WebSocket**: Use WebSocket for real-time updates

## Browser-Specific Optimizations

### Chrome/Chromium
```javascript
// Enable hardware acceleration
if (navigator.userAgent.includes('Chrome')) {
    workflow.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
}
```

### Firefox
```javascript
// Firefox-specific optimizations
if (navigator.userAgent.includes('Firefox')) {
    workflow.settingsManager.set('antialiasing', false); // Better performance
}
```

### Safari
```javascript
// Safari optimizations
if (navigator.userAgent.includes('Safari') && !navigator.userAgent.includes('Chrome')) {
    workflow.settingsManager.set('shadows', false); // Safari shadow performance
}
```

## Performance Profiling

### Browser DevTools

1. **Performance Tab**: Record performance during animation
2. **Memory Tab**: Monitor memory usage and leaks
3. **Network Tab**: Check API call frequency and size
4. **Console**: Monitor FPS and custom metrics

### Custom Profiling

```javascript
// Performance timing
const startTime = performance.now();
workflow.startWorkflowAnimation();
const endTime = performance.now();
console.log(`Animation start took ${endTime - startTime}ms`);

// Memory profiling
if (performance.memory) {
    console.log(`Used: ${performance.memory.usedJSHeapSize / 1048576}MB`);
    console.log(`Total: ${performance.memory.totalJSHeapSize / 1048576}MB`);
}
```

## Optimization Checklist

### Initial Setup
- [ ] Enable auto quality detection
- [ ] Set appropriate quality preset for target devices
- [ ] Configure particle limits
- [ ] Enable performance monitoring

### Runtime Optimization
- [ ] Monitor FPS and adjust quality dynamically
- [ ] Clean up unused resources periodically
- [ ] Cache API responses appropriately
- [ ] Use efficient rendering techniques

### Mobile Optimization
- [ ] Reduce particle count
- [ ] Disable expensive effects (shadows, high-res textures)
- [ ] Limit pixel ratio
- [ ] Use touch-optimized controls

### Desktop Optimization
- [ ] Enable high-quality effects on capable hardware
- [ ] Use full pixel ratio on high-DPI displays
- [ ] Enable advanced lighting and shadows
- [ ] Maximize particle counts

## Troubleshooting Performance Issues

### Low FPS

**Symptoms**: Choppy animation, slow response
**Solutions**:
1. Reduce quality setting
2. Disable particle effects
3. Lower pixel ratio
4. Disable shadows and glow
5. Close other browser tabs

### High Memory Usage

**Symptoms**: Browser slowdown, crashes
**Solutions**:
1. Call `workflow.dispose()` when done
2. Reduce texture sizes
3. Limit particle count
4. Clear browser cache
5. Restart browser

### Slow Loading

**Symptoms**: Long initialization time
**Solutions**:
1. Preload critical assets
2. Use CDN for libraries
3. Compress textures
4. Implement progressive loading
5. Cache static assets

### Mobile Performance Issues

**Symptoms**: Poor performance on mobile devices
**Solutions**:
1. Force low quality on mobile
2. Disable expensive effects
3. Reduce animation complexity
4. Use lower resolution textures
5. Implement touch-specific optimizations

## Performance Best Practices

### Development
1. **Profile Early**: Use performance tools during development
2. **Test on Target Devices**: Test on actual mobile devices
3. **Monitor Memory**: Watch for memory leaks
4. **Optimize Assets**: Compress textures and models
5. **Use Efficient Algorithms**: Choose performance over complexity

### Production
1. **Enable Compression**: Use gzip for all assets
2. **Use CDN**: Serve libraries from CDN
3. **Monitor Real Users**: Track performance metrics
4. **Graceful Degradation**: Provide fallbacks for older devices
5. **Regular Updates**: Keep libraries updated

### User Experience
1. **Loading Feedback**: Show progress during loading
2. **Quality Options**: Let users choose quality
3. **Performance Warnings**: Warn about performance issues
4. **Accessibility**: Ensure good performance for all users
5. **Error Handling**: Handle performance-related errors gracefully

## Benchmarking

### Performance Targets

| Device Type | Target FPS | Max Memory | Load Time |
|-------------|------------|------------|-----------|
| High-end Desktop | 60 FPS | 1GB | <3s |
| Mid-range Desktop | 45 FPS | 512MB | <5s |
| High-end Mobile | 30 FPS | 256MB | <8s |
| Mid-range Mobile | 24 FPS | 128MB | <10s |

### Testing Scenarios

1. **Idle State**: No animation, static view
2. **Active Animation**: Full workflow animation
3. **Heavy Interaction**: Rapid camera movement, node selection
4. **Stress Test**: Maximum particles, all effects enabled
5. **Extended Use**: 30+ minutes of continuous use

---

**Remember**: Performance optimization is an ongoing process. Regular monitoring and adjustment ensure the best user experience across all devices.

**Version**: 3.0.0  
**Last Updated**: 2024
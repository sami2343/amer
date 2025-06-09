# HETROFL 3D Workflow Visualization

## Overview
This directory contains the 3D visualization infrastructure for the HETROFL (Heterogeneous Federated Learning) system. The implementation provides a professional 3D scene for visualizing federated learning workflows with real-time animations and interactive controls.

## File Structure
```
hetrofl_system/gui/static/
├── js/
│   ├── three.min.js          # Three.js r150+ WebGL library
│   ├── gsap.min.js           # GSAP 3.12+ animation library
│   └── workflow-3d.js        # Main 3D workflow implementation
└── css/
    └── workflow-3d.css       # 3D-specific styles with glassmorphism effects
```

## Features

### Core 3D Infrastructure
- **WebGL Rendering**: High-performance 3D rendering with Three.js
- **Responsive Design**: Adapts to different screen sizes and devices
- **Performance Monitoring**: Real-time FPS tracking and optimization
- **Fallback Support**: Graceful degradation for unsupported browsers

### Visual Elements
- **3D Nodes**: Different geometries for different model types
  - Data Sources: Cylinders (orange)
  - Local Models: Cubes (green)
  - Aggregation: Octahedron (cyan)
  - Global Model: Sphere (blue)
  - Evaluation: Cone (red)
- **Connections**: Curved 3D paths between nodes
- **Animations**: Floating nodes and data particle flows
- **Lighting**: Professional lighting setup with ambient and directional lights

### User Interface
- **Glassmorphism Panels**: Modern translucent UI elements
- **Interactive Controls**: Animation start/pause/reset, speed control
- **Camera Views**: Multiple viewing angles (3D, top, side)
- **Performance Stats**: Real-time monitoring display
- **Legend**: Visual guide for node types

## Technical Specifications

### Requirements
- **WebGL Support**: WebGL 1.0 or 2.0 compatible browser
- **JavaScript**: ES6+ support
- **Performance**: Target 60fps on modern devices

### Browser Compatibility
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

### Performance Optimizations
- Automatic quality adjustment based on FPS
- Efficient geometry reuse
- Optimized material sharing
- Particle system management

## Usage

### Basic Integration
```html
<!-- Include required libraries -->
<script src="js/three.min.js"></script>
<script src="js/gsap.min.js"></script>
<script src="js/workflow-3d.js"></script>

<!-- Include styles -->
<link rel="stylesheet" href="css/workflow-3d.css">

<!-- Create container -->
<div id="workflow-3d-container"></div>

<!-- Initialize -->
<script>
const workflow = new HETROFL3DWorkflow('workflow-3d-container');
</script>
```

### API Methods
```javascript
// Animation control
workflow.startWorkflowAnimation();
workflow.pauseAnimation();
workflow.resetWorkflow();

// Speed control
workflow.setAnimationSpeed(1.5); // 0.1 to 3.0

// Cleanup
workflow.destroy();
```

## Configuration

### Scene Settings
- **Background**: Matches CSS `--dark-color` (#06090f)
- **Fog**: Depth perception enhancement
- **Camera**: Perspective with 75° FOV
- **Lighting**: Ambient + directional + accent lights

### Node Positions
- Data Sources: Left side (-15, y, z)
- Local Models: Left-center (-5, y, z)
- Aggregation: Center (5, 0, 0)
- Global Model: Right-center (15, 0, 0)
- Evaluation: Top-right (15, 4, 0)

## Customization

### Adding New Node Types
1. Define geometry in `getNodeGeometry()`
2. Add color configuration
3. Update legend in CSS
4. Implement positioning logic

### Modifying Animations
1. Update `addFloatingAnimation()` for node movement
2. Modify `createConnection()` for path curves
3. Enhance `updateDataParticles()` for flow effects

## Troubleshooting

### Common Issues
1. **Black Screen**: Check WebGL support and console errors
2. **Poor Performance**: Reduce particle count or disable shadows
3. **Controls Not Working**: Verify event listeners are attached
4. **Missing Nodes**: Check node creation and positioning logic

### Debug Mode
Enable debug logging by setting:
```javascript
window.HETROFL_DEBUG = true;
```

## Future Enhancements
- OrbitControls integration for better camera interaction
- VR/AR support for immersive visualization
- Advanced particle systems for data flow
- Real-time data integration
- Export capabilities for presentations

## Dependencies
- Three.js r150+: 3D rendering engine
- GSAP 3.12+: Professional animation library
- Modern browser with WebGL support

## License
Part of the HETROFL system - see main project license.
# HETROFL 3D Workflow - User Guide

## Welcome to HETROFL 3D Visualization! 🚀

This guide will help you navigate and use the interactive 3D visualization of the HETROFL federated learning system.

## Getting Started

### First Time Setup

1. **Open the Workflow Page**: Navigate to the "Workflow" section in HETROFL
2. **Wait for Loading**: The 3D scene will load automatically with a progress indicator
3. **Interactive Tutorial**: Click the "Help" button for a guided tour
4. **Adjust Quality**: Use the quality dropdown to optimize for your device

### System Requirements

- **Modern Browser**: Chrome 60+, Firefox 55+, Safari 12+, Edge 79+
- **Graphics**: WebGL-capable graphics card
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Internet**: Stable connection for real-time updates

## Interface Overview

### Main Controls Panel

Located in the top-left corner of the 3D view:

#### Animation Controls
- **▶️ Start**: Begin the workflow animation
- **⏸️ Pause**: Pause the current animation
- **🔄 Reset**: Reset the workflow to initial state
- **Speed Slider**: Adjust animation speed (0.1x to 3x)

#### Camera Views
- **🧊 3D**: Full 3D perspective view (default)
- **👁️ Top**: Top-down view of the workflow
- **↔️ Side**: Side view for better depth perception
- **🔍 Overview**: Zoomed-out overview of entire system

#### Visual Effects
- **✨ Particles**: Toggle particle effects for data flow
- **☀️ Glow**: Toggle glow effects on nodes
- **☁️ Fog**: Toggle atmospheric fog
- **🌓 Shadows**: Toggle shadow effects

#### Tools
- **❓ Help**: Show interactive tutorial
- **🔍 Fullscreen**: Enter fullscreen mode
- **🏠 Home**: Reset camera to default position

### Quality Controls

Located in the top-center:

- **Low**: Best performance, minimal effects
- **Med**: Balanced performance and quality
- **High**: Best visuals, requires good hardware

### Export Controls

Located in the top-right:

- **📷 Screenshot**: Capture current view as image
- **🎥 Recording**: Start/stop video recording

## Understanding the Visualization

### Node Types

The 3D scene shows different types of nodes representing parts of the federated learning system:

#### 🗄️ Data Sources (Blue Cubes)
- Represent distributed datasets
- Show data availability and quality
- Animate when data is being processed

#### 🤖 Local Models (Green Spheres)
- Individual machine learning models
- Different shapes for different model types:
  - **Sphere**: Neural Networks
  - **Cylinder**: Random Forest
  - **Cone**: XGBoost
  - **Octahedron**: CatBoost

#### 🔄 Aggregation (Purple Pyramid)
- Central aggregation server
- Combines updates from local models
- Pulses during aggregation process

#### 🌐 Global Model (Gold Dodecahedron)
- The federated global model
- Shows overall system performance
- Glows when training is complete

#### 📊 Evaluation (Orange Torus)
- Model evaluation and testing
- Shows performance metrics
- Changes color based on results

### Data Flow Animation

Watch the animated particles to understand data flow:

- **Blue Particles**: Raw data flowing to local models
- **Green Particles**: Model updates flowing to aggregation
- **Purple Particles**: Aggregated updates flowing to global model
- **Gold Particles**: Global model updates flowing back to local models

### Node States

Nodes change appearance based on their current state:

- **Gray**: Idle/Waiting
- **Blue**: Processing/Training
- **Green**: Complete/Success
- **Red**: Error/Failed
- **Pulsing**: Active operation

## Interaction Guide

### Mouse Controls

- **Left Click + Drag**: Rotate the camera around the scene
- **Right Click + Drag**: Pan the camera
- **Mouse Wheel**: Zoom in/out
- **Click on Node**: Select and view details
- **Double Click**: Focus camera on selected node

### Touch Controls (Mobile/Tablet)

- **Single Finger Drag**: Rotate camera
- **Two Finger Pinch**: Zoom in/out
- **Two Finger Drag**: Pan camera
- **Tap Node**: Select node
- **Double Tap**: Focus on node

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Start/Pause animation |
| `R` | Reset camera position |
| `F` | Toggle fullscreen |
| `H` | Show help tutorial |
| `1` | Low quality |
| `2` | Medium quality |
| `3` | High quality |
| `4` | Ultra quality |
| `Esc` | Deselect node/Close help |

## Features in Detail

### Real-Time Monitoring

The visualization connects to the HETROFL system to show:

- **Live Training Status**: See which models are currently training
- **Performance Metrics**: Real-time accuracy, loss, and other metrics
- **Data Flow**: Visualize actual data movement between components
- **System Health**: Monitor overall system status

### Interactive Help System

Click the "Help" button to start an interactive tutorial:

1. **Navigation**: Use arrow keys or click Next/Previous
2. **Highlights**: Important areas are highlighted
3. **Context**: Each step explains specific features
4. **Skip**: Press Escape to exit at any time

### Quality Optimization

The system automatically adjusts quality based on your device:

#### Automatic Features
- **Device Detection**: Identifies mobile vs desktop
- **Performance Monitoring**: Tracks FPS and adjusts quality
- **Memory Management**: Prevents memory leaks
- **Battery Optimization**: Reduces effects on mobile devices

#### Manual Override
- Use quality buttons to override automatic settings
- Higher quality = better visuals but requires more resources
- Lower quality = better performance on older devices

### Accessibility Features

The visualization is designed to be accessible:

- **Screen Reader Support**: All controls have proper labels
- **Keyboard Navigation**: Complete keyboard control
- **High Contrast**: Supports high contrast mode
- **Reduced Motion**: Respects motion preferences
- **Focus Indicators**: Clear focus indicators for navigation

## Troubleshooting

### Performance Issues

**Symptoms**: Low frame rate, stuttering, lag
**Solutions**:
1. Lower quality setting to "Low" or "Medium"
2. Disable particle effects
3. Close other browser tabs
4. Update graphics drivers
5. Try a different browser

### Loading Problems

**Symptoms**: Stuck on loading screen, blank display
**Solutions**:
1. Refresh the page
2. Check internet connection
3. Clear browser cache
4. Disable browser extensions
5. Try incognito/private mode

### Display Issues

**Symptoms**: Missing elements, visual glitches
**Solutions**:
1. Check WebGL support: visit `webglreport.com`
2. Update browser to latest version
3. Enable hardware acceleration
4. Try different quality settings

### Mobile Issues

**Symptoms**: Poor performance on mobile
**Solutions**:
1. Use "Low" quality setting
2. Close other apps
3. Ensure good WiFi connection
4. Try landscape orientation
5. Use latest mobile browser

## Tips for Best Experience

### Performance Tips
- Use "High" quality only on powerful devices
- Enable auto-quality for adaptive performance
- Close unnecessary browser tabs
- Use wired internet connection for real-time data

### Viewing Tips
- Use fullscreen mode for immersive experience
- Try different camera angles for better understanding
- Watch the particle animations to understand data flow
- Select nodes to see detailed information

### Learning Tips
- Start with the interactive tutorial
- Experiment with different quality settings
- Try all camera views to understand the system
- Watch a complete animation cycle
- Use keyboard shortcuts for efficient navigation

## Advanced Features

### Video Recording
1. Click the video button to start recording
2. Perform actions you want to capture
3. Click again to stop and download
4. Supported formats: WebM, MP4 (browser dependent)

### Screenshot Capture
1. Position the view as desired
2. Click the camera button
3. Image is automatically downloaded
4. High-resolution capture based on current quality

### Settings Persistence
- Quality preferences are saved automatically
- Effect toggles are remembered
- Camera position can be bookmarked
- Settings sync across browser sessions

## Getting Help

### In-App Help
- **Help Button**: Interactive tutorial system
- **Tooltips**: Hover over controls for quick help
- **Keyboard Shortcuts**: Press F1 for shortcut list

### Documentation
- **User Guide**: This document
- **Technical Guide**: For developers and advanced users
- **API Documentation**: For integration developers

### Support Channels
- **GitHub Issues**: Report bugs and request features
- **Email Support**: Technical assistance
- **Community Forum**: User discussions and tips

## Frequently Asked Questions

### Q: Why is the visualization running slowly?
A: Try lowering the quality setting or disabling particle effects. The system will also automatically adjust quality if performance drops.

### Q: Can I use this on my phone?
A: Yes! The visualization is optimized for mobile devices. Use landscape orientation for the best experience.

### Q: How do I save my current view?
A: Use the screenshot button to capture the current view, or the video button to record animations.

### Q: What browsers are supported?
A: Modern browsers with WebGL support: Chrome 60+, Firefox 55+, Safari 12+, Edge 79+.

### Q: Can I integrate this into my own application?
A: Yes! See the technical documentation for API details and integration examples.

### Q: Is my data secure?
A: The visualization only displays aggregated, non-sensitive information. No raw data is transmitted or stored.

---

**Need more help?** Contact our support team or check the technical documentation for advanced features.

**Version**: 3.0.0  
**Last Updated**: 2024
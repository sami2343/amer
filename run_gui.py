#!/usr/bin/env python3
"""
HETROFL System - Run GUI only
This script runs only the GUI component of the HETROFL system.

Usage:
    python run_gui.py [--host HOST] [--port PORT] [--debug]
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main application entry point for GUI only."""
    parser = argparse.ArgumentParser(
        description='HETROFL - GUI Only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_gui.py                    # Run with default settings
  python run_gui.py --host 0.0.0.0     # Allow external connections
  python run_gui.py --port 8080        # Use custom port
  python run_gui.py --debug            # Enable debug mode
        """
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host address to bind to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    print(f"Starting HETROFL GUI on http://{args.host}:{args.port}")
    
    # Update GUI configuration
    from hetrofl_system.config import GUI_CONFIG
    GUI_CONFIG['host'] = args.host
    GUI_CONFIG['port'] = args.port
    GUI_CONFIG['debug'] = args.debug
    
    # Run the GUI
    from hetrofl_system.gui.app import app, socketio, initialize_system
    
    # Initialize system
    initialize_system()
    
    # Run Flask app
    socketio.run(
        app,
        host=GUI_CONFIG['host'],
        port=GUI_CONFIG['port'],
        debug=GUI_CONFIG['debug']
    )

if __name__ == "__main__":
    main() 
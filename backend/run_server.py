#!/usr/bin/env python3
"""
SeismoGuard Backend Server Startup Script

This script starts the Flask API server with proper configuration for
integration with the SeismoGuard frontend application.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.api import app
from app.io_utils import ensure_directory, clean_temp_files


def setup_logging(level=logging.INFO):
    """Configure logging for the server."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('backend.log')
        ]
    )


def setup_directories():
    """Ensure all required directories exist."""
    directories = [
        'models',
        'uploads',
        'outputs',
        'cache',
        'data',
        'logs'
    ]
    
    for directory in directories:
        ensure_directory(directory)
        print(f"✓ Directory ready: {directory}")


def cleanup_temp_files():
    """Clean up old temporary files."""
    try:
        deleted = clean_temp_files('uploads', max_age_hours=24)
        if deleted > 0:
            print(f"✓ Cleaned up {deleted} old temporary files")
    except Exception as e:
        print(f"Warning: Could not clean temp files: {e}")


def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        'flask',
        'flask_cors',
        'obspy',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'tensorflow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (missing)")
    
    if missing_packages:
        print(f"\nError: Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def test_api_endpoints():
    """Test basic API functionality."""
    try:
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ Health endpoint working")
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                return False
            
            # Test algorithms endpoint
            response = client.get('/algorithms')
            if response.status_code == 200:
                print("✓ Algorithms endpoint working")
            else:
                print(f"✗ Algorithms endpoint failed: {response.status_code}")
                return False
            
            # Test planet presets endpoint
            response = client.get('/planet-presets')
            if response.status_code == 200:
                print("✓ Planet presets endpoint working")
            else:
                print(f"✗ Planet presets endpoint failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False


def print_startup_info():
    """Print startup information and instructions."""
    print("\n" + "="*60)
    print("🌍 SeismoGuard Backend Server")
    print("="*60)
    print("A production-ready seismic detection backend")
    print("Supports multiple data sources and ML-based detection")
    print("="*60)
    
    print("\n📡 Server Information:")
    print("  • Host: 0.0.0.0 (all interfaces)")
    print("  • Port: 5000")
    print("  • Frontend URL: http://127.0.0.1:5000")
    print("  • CORS: Enabled for frontend integration")
    
    print("\n🔗 API Endpoints:")
    print("  • GET  /health          - Health check")
    print("  • POST /detect          - Main detection endpoint")
    print("  • GET  /algorithms      - Available algorithms")
    print("  • GET  /planet-presets  - Planet configurations")
    
    print("\n📊 Data Sources Supported:")
    print("  • File upload (seismic + optional pressure)")
    print("  • IRIS FDSN web services")
    print("  • URL-based data fetching")
    print("  • NASA PDS search (InSight data)")
    
    print("\n🧠 Detection Features:")
    print("  • Classical algorithms (STA/LTA, Z-score)")
    print("  • Machine learning models (CNN, LSTM)")
    print("  • Pressure-based noise masking")
    print("  • Planet-specific processing")
    
    print("\n🚀 Frontend Integration:")
    print("  • Compatible with existing ApiClient")
    print("  • Graceful fallback when offline")
    print("  • Automatic caching and backoff")
    print("  • Non-disruptive operation")


def main():
    """Main server startup function."""
    parser = argparse.ArgumentParser(description="SeismoGuard Backend Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-checks', action='store_true', help='Skip startup checks')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    print_startup_info()
    
    if not args.no_checks:
        print("\n🔍 Running startup checks...")
        
        # Check dependencies
        print("\nChecking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        # Setup directories
        print("\nSetting up directories...")
        setup_directories()
        
        # Cleanup old files
        print("\nCleaning up temporary files...")
        cleanup_temp_files()
        
        # Test API
        print("\nTesting API endpoints...")
        if not test_api_endpoints():
            print("Warning: Some API tests failed, but continuing...")
        
        print("\n✅ All startup checks completed!")
    
    print(f"\n🚀 Starting server on {args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        # Start the Flask development server
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

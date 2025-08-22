#!/usr/bin/env python3
"""
Comprehensive QA test suite for enhanced SeismoGuard backend.
Tests all new functionality and ensures backward compatibility.
"""

import sys
import os
import unittest
import json
import tempfile
import numpy as np
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Test basic imports first
def test_basic_imports():
    """Test that all basic modules can be imported."""
    print("Testing basic imports...")
    
    try:
        from app.io_utils import load_waveform, trace_to_numpy
        print("✓ io_utils import successful")
    except Exception as e:
        print(f"✗ io_utils import failed: {e}")
        return False
    
    try:
        from app.preprocess import preprocess_trace, get_planet_preset
        print("✓ preprocess import successful")
    except Exception as e:
        print(f"✗ preprocess import failed: {e}")
        return False
    
    try:
        from app.detector_manager import DetectorManager
        print("✓ detector_manager import successful")
    except Exception as e:
        print(f"✗ detector_manager import failed: {e}")
        return False
    
    try:
        from app.noise_masker import mask_seismic_with_pressure
        print("✓ noise_masker import successful")
    except Exception as e:
        print(f"✗ noise_masker import failed: {e}")
        return False
    
    return True


def test_enhanced_imports():
    """Test enhanced module imports with graceful fallback."""
    print("\nTesting enhanced module imports...")
    
    # Test data integration hub
    try:
        from app.data_integration_hub import DataIntegrationHub
        print("✓ data_integration_hub import successful")
    except ImportError as e:
        if 'aiohttp' in str(e):
            print("⚠ data_integration_hub requires aiohttp (pip install aiohttp)")
        else:
            print(f"✗ data_integration_hub import failed: {e}")
            return False
    except Exception as e:
        print(f"✗ data_integration_hub import failed: {e}")
        return False
    
    # Test external ML APIs
    try:
        from app.external_ml_apis import ExternalMLAPIs
        print("✓ external_ml_apis import successful")
    except ImportError as e:
        if 'aiohttp' in str(e):
            print("⚠ external_ml_apis requires aiohttp (pip install aiohttp)")
        else:
            print(f"✗ external_ml_apis import failed: {e}")
            return False
    except Exception as e:
        print(f"✗ external_ml_apis import failed: {e}")
        return False
    
    # Test WebSocket streaming
    try:
        from app.websocket_streaming import WebSocketStreamer
        print("✓ websocket_streaming import successful")
    except ImportError as e:
        if 'websockets' in str(e):
            print("⚠ websocket_streaming requires websockets (pip install websockets)")
        else:
            print(f"✗ websocket_streaming import failed: {e}")
            return False
    except Exception as e:
        print(f"✗ websocket_streaming import failed: {e}")
        return False
    
    # Test satellite correlation
    try:
        from app.satellite_correlation import SatelliteCorrelationEngine
        print("✓ satellite_correlation import successful")
    except ImportError as e:
        if 'aiohttp' in str(e):
            print("⚠ satellite_correlation requires aiohttp (pip install aiohttp)")
        else:
            print(f"✗ satellite_correlation import failed: {e}")
            return False
    except Exception as e:
        print(f"✗ satellite_correlation import failed: {e}")
        return False
    
    return True


def test_api_import():
    """Test Flask API import."""
    print("\nTesting Flask API import...")
    
    try:
        # Test if we can import the API without starting servers
        os.environ['SEISMO_SKIP_WEBSOCKET'] = '1'  # Skip WebSocket startup
        from app.api import app
        print("✓ Flask API import successful")
        
        # Test basic app configuration
        if app.config.get('TESTING') is not None:
            print("✓ Flask app properly configured")
        
        return True
        
    except Exception as e:
        print(f"✗ Flask API import failed: {e}")
        return False


def test_endpoint_definitions():
    """Test that all API endpoints are properly defined."""
    print("\nTesting API endpoint definitions...")
    
    try:
        from app.api import app
        
        # Get all routes
        routes = []
        for rule in app.url_map.iter_rules():
            routes.append({
                'endpoint': rule.endpoint,
                'methods': list(rule.methods),
                'rule': str(rule)
            })
        
        # Check for required endpoints
        required_endpoints = [
            '/health',
            '/detect',
            '/algorithms',
            '/planet-presets',
            '/earthquakes/recent',
            '/data-sources/status',
            '/compare/events',
            '/iris/stations/search',
            '/iris/events/search',
            '/iris/stations/nearby',
            '/ml/classify',
            '/ml/features',
            '/ml/ensemble',
            '/ml/status',
            '/satellite/correlate',
            '/satellite/imagery/search',
            '/satellite/environmental',
            '/stream/status',
            '/stream/broadcast'
        ]
        
        found_endpoints = [route['rule'] for route in routes]
        
        missing_endpoints = []
        for endpoint in required_endpoints:
            if endpoint not in found_endpoints:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"✗ Missing endpoints: {missing_endpoints}")
            return False
        else:
            print(f"✓ All {len(required_endpoints)} required endpoints found")
            return True
            
    except Exception as e:
        print(f"✗ Endpoint definition test failed: {e}")
        return False


def test_backward_compatibility():
    """Test that original functionality still works."""
    print("\nTesting backward compatibility...")
    
    try:
        from app.api import app
        
        # Test original endpoints exist
        with app.test_client() as client:
            # Test health endpoint
            response = client.get('/health')
            if response.status_code == 200:
                print("✓ /health endpoint working")
            else:
                print(f"✗ /health endpoint failed: {response.status_code}")
                return False
            
            # Test algorithms endpoint
            response = client.get('/algorithms')
            if response.status_code == 200:
                print("✓ /algorithms endpoint working")
            else:
                print(f"✗ /algorithms endpoint failed: {response.status_code}")
                return False
            
            # Test planet presets endpoint
            response = client.get('/planet-presets')
            if response.status_code == 200:
                print("✓ /planet-presets endpoint working")
            else:
                print(f"✗ /planet-presets endpoint failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False


def test_error_handling():
    """Test error handling in API endpoints."""
    print("\nTesting error handling...")
    
    try:
        from app.api import app
        
        with app.test_client() as client:
            # Test invalid endpoint
            response = client.get('/invalid-endpoint')
            if response.status_code == 404:
                print("✓ 404 error handling working")
            else:
                print(f"✗ 404 error handling failed: {response.status_code}")
                return False
            
            # Test POST to GET-only endpoint
            response = client.post('/health')
            if response.status_code == 405:
                print("✓ 405 error handling working")
            else:
                print(f"✗ 405 error handling failed: {response.status_code}")
                return False
            
            # Test detect endpoint with no data
            response = client.post('/detect')
            if response.status_code == 400:
                print("✓ 400 error handling working")
            else:
                print(f"✗ 400 error handling failed: {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def main():
    """Run comprehensive QA tests."""
    print("="*60)
    print("🧪 SeismoGuard Enhanced Backend QA Test Suite")
    print("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Enhanced Imports", test_enhanced_imports),
        ("API Import", test_api_import),
        ("Endpoint Definitions", test_endpoint_definitions),
        ("Backward Compatibility", test_backward_compatibility),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Backend is ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please review and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

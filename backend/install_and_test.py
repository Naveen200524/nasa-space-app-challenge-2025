#!/usr/bin/env python3
"""
SeismoGuard Backend Installation and Testing Script
Automatically installs dependencies and runs comprehensive tests.
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} successful")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is too old. Need Python 3.8+")
        return False


def install_core_dependencies():
    """Install core dependencies required for basic functionality."""
    print("\n📦 Installing Core Dependencies...")
    
    core_packages = [
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "obspy>=1.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "matplotlib>=3.7.0",
        "requests>=2.31.0"
    ]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    return True


def install_enhanced_dependencies():
    """Install enhanced dependencies for advanced features."""
    print("\n🚀 Installing Enhanced Dependencies...")
    
    enhanced_packages = [
        "aiohttp>=3.8.5",
        "websockets>=11.0.3",
        "psutil>=5.9.5"
    ]
    
    success = True
    for package in enhanced_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Enhanced feature dependency {package} failed - continuing with core features")
            success = False
    
    return success


def test_basic_functionality():
    """Test basic backend functionality."""
    print("\n🧪 Testing Basic Functionality...")
    
    # Test basic imports
    test_script = '''
import sys
sys.path.insert(0, ".")

try:
    from app.api import app
    print("✅ Flask API import successful")
    
    # Test basic endpoints
    with app.test_client() as client:
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            sys.exit(1)
        
        response = client.get("/algorithms")
        if response.status_code == 200:
            print("✅ Algorithms endpoint working")
        else:
            print(f"❌ Algorithms endpoint failed: {response.status_code}")
            sys.exit(1)
    
    print("✅ All basic tests passed")
    
except Exception as e:
    print(f"❌ Basic functionality test failed: {e}")
    sys.exit(1)
'''
    
    with open("test_basic.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python test_basic.py", "Testing basic functionality")
    
    # Cleanup
    if os.path.exists("test_basic.py"):
        os.remove("test_basic.py")
    
    return success


def test_enhanced_features():
    """Test enhanced features if dependencies are available."""
    print("\n🔬 Testing Enhanced Features...")
    
    test_script = '''
import sys
sys.path.insert(0, ".")

try:
    from app.api import app
    
    # Test enhanced endpoints
    with app.test_client() as client:
        # Test data sources status
        response = client.get("/data-sources/status")
        print(f"Data sources status: {response.status_code}")
        
        # Test ML status
        response = client.get("/ml/status")
        print(f"ML status: {response.status_code}")
        
        # Test stream status
        response = client.get("/stream/status")
        print(f"Stream status: {response.status_code}")
    
    print("✅ Enhanced features test completed")
    
except Exception as e:
    print(f"⚠️  Enhanced features test: {e}")
'''
    
    with open("test_enhanced.py", "w") as f:
        f.write(test_script)
    
    success = run_command("python test_enhanced.py", "Testing enhanced features")
    
    # Cleanup
    if os.path.exists("test_enhanced.py"):
        os.remove("test_enhanced.py")
    
    return success


def create_startup_verification():
    """Create a startup verification script."""
    print("\n📝 Creating startup verification...")
    
    verification_script = '''#!/usr/bin/env python3
"""
SeismoGuard Backend Startup Verification
Run this script to verify the backend is working correctly.
"""

import requests
import time
import sys

def test_backend():
    """Test backend endpoints."""
    base_url = "http://127.0.0.1:5000"
    
    print("🔍 Testing SeismoGuard Backend...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is healthy")
            data = response.json()
            print(f"   Service: {data.get('service', 'Unknown')}")
            print(f"   Algorithms: {len(data.get('available_algorithms', []))}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("   Make sure the backend is running: python run_server.py")
        return False
    
    # Test algorithms endpoint
    try:
        response = requests.get(f"{base_url}/algorithms", timeout=5)
        if response.status_code == 200:
            print("✅ Algorithms endpoint working")
        else:
            print(f"⚠️  Algorithms endpoint issue: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Algorithms endpoint error: {e}")
    
    # Test enhanced features
    enhanced_endpoints = [
        "/data-sources/status",
        "/ml/status", 
        "/stream/status"
    ]
    
    for endpoint in enhanced_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ Enhanced endpoint {endpoint} working")
            elif response.status_code == 503:
                print(f"⚠️  Enhanced endpoint {endpoint} unavailable (missing dependencies)")
            else:
                print(f"⚠️  Enhanced endpoint {endpoint} issue: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Enhanced endpoint {endpoint} error: {e}")
    
    print("\\n🎉 Backend verification complete!")
    return True

if __name__ == "__main__":
    if test_backend():
        sys.exit(0)
    else:
        sys.exit(1)
'''
    
    with open("verify_backend.py", "w") as f:
        f.write(verification_script)
    
    print("✅ Created verify_backend.py - run this after starting the server")
    return True


def main():
    """Main installation and testing process."""
    print("="*60)
    print("🌍 SeismoGuard Backend Installation & Testing")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install core dependencies
    if not install_core_dependencies():
        print("\n❌ Core dependency installation failed")
        return 1
    
    # Install enhanced dependencies (optional)
    enhanced_success = install_enhanced_dependencies()
    if enhanced_success:
        print("✅ All dependencies installed successfully")
    else:
        print("⚠️  Some enhanced dependencies failed - core features will work")
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality tests failed")
        return 1
    
    # Test enhanced features
    test_enhanced_features()
    
    # Create verification script
    create_startup_verification()
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 Installation Complete!")
    print("="*60)
    print("✅ Core dependencies installed")
    print("✅ Basic functionality verified")
    if enhanced_success:
        print("✅ Enhanced features available")
    else:
        print("⚠️  Enhanced features partially available")
    
    print("\n🚀 Next Steps:")
    print("1. Start the backend: python run_server.py")
    print("2. Verify it's working: python verify_backend.py")
    print("3. Access the API at: http://127.0.0.1:5000")
    print("4. Check health: http://127.0.0.1:5000/health")
    
    print("\n📚 Documentation:")
    print("- Backend features: README.md")
    print("- QA report: QUALITY_ASSURANCE_REPORT.md")
    print("- API endpoints: http://127.0.0.1:5000/health")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

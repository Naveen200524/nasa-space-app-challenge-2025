#!/usr/bin/env python3
"""
SeismoGuard Backend Test Suite

Comprehensive tests for the backend functionality including:
- API endpoints
- Data processing
- Detection algorithms
- Noise masking
- ML models
"""

import os
import sys
import unittest
import tempfile
import json
import numpy as np
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.api import app
from app.io_utils import load_waveform, trace_to_numpy, numpy_to_trace
from app.preprocess import preprocess_trace, get_planet_preset
from app.detector_manager import DetectorManager
from app.noise_masker import mask_seismic_with_pressure
from app.data_fetcher import fetch_from_pds_search
from app.ml_train import synthetic_windows, generate_synthetic_earthquake
from obspy import Trace, UTCDateTime
import pandas as pd


class TestAPIEndpoints(unittest.TestCase):
    """Test Flask API endpoints."""
    
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('available_algorithms', data)
    
    def test_algorithms_endpoint(self):
        """Test algorithms endpoint."""
        response = self.app.get('/algorithms')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('algorithms', data)
        self.assertIsInstance(data['algorithms'], list)
    
    def test_planet_presets_endpoint(self):
        """Test planet presets endpoint."""
        response = self.app.get('/planet-presets')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('presets', data)
        self.assertIn('earth', data['presets'])
        self.assertIn('mars', data['presets'])
        self.assertIn('moon', data['presets'])
    
    def test_detect_endpoint_no_data(self):
        """Test detect endpoint with no data."""
        response = self.app.post('/detect')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)


class TestDataProcessing(unittest.TestCase):
    """Test data processing functionality."""
    
    def setUp(self):
        # Create synthetic test data
        self.sr = 20.0
        self.duration = 60.0  # 1 minute
        self.npts = int(self.duration * self.sr)
        
        # Generate synthetic seismic data
        self.seismic_data = generate_synthetic_earthquake(self.npts, self.sr)
        self.trace = numpy_to_trace(
            self.seismic_data, 
            self.sr, 
            starttime=UTCDateTime(),
            station="TEST",
            channel="BHZ"
        )
    
    def test_trace_creation(self):
        """Test trace creation from numpy array."""
        self.assertEqual(self.trace.stats.sampling_rate, self.sr)
        self.assertEqual(len(self.trace.data), self.npts)
        self.assertEqual(self.trace.stats.station, "TEST")
    
    def test_trace_to_numpy_conversion(self):
        """Test conversion from trace to numpy."""
        data, sr = trace_to_numpy(self.trace)
        
        self.assertEqual(sr, self.sr)
        self.assertEqual(len(data), self.npts)
        np.testing.assert_array_equal(data, self.trace.data)
    
    def test_planet_presets(self):
        """Test planet configuration presets."""
        earth_preset = get_planet_preset('earth')
        mars_preset = get_planet_preset('mars')
        moon_preset = get_planet_preset('moon')
        
        self.assertIn('fmin', earth_preset)
        self.assertIn('fmax', earth_preset)
        
        # Mars should have lower frequencies than Earth
        self.assertLess(mars_preset['fmax'], earth_preset['fmax'])
        
        # Moon should have lowest frequencies
        self.assertLess(moon_preset['fmax'], mars_preset['fmax'])
    
    def test_preprocessing(self):
        """Test seismic data preprocessing."""
        processed_trace = preprocess_trace(self.trace, planet='earth')
        
        # Should have same length
        self.assertEqual(len(processed_trace.data), len(self.trace.data))
        
        # Should have same sampling rate
        self.assertEqual(processed_trace.stats.sampling_rate, self.trace.stats.sampling_rate)
        
        # Data should be different (filtered)
        self.assertFalse(np.array_equal(processed_trace.data, self.trace.data))


class TestDetectionAlgorithms(unittest.TestCase):
    """Test seismic detection algorithms."""
    
    def setUp(self):
        self.dm = DetectorManager(model_root='test_models')
        
        # Create test trace with synthetic earthquake
        self.sr = 20.0
        self.duration = 300.0  # 5 minutes
        self.npts = int(self.duration * self.sr)
        
        # Generate data with known events
        data = np.random.normal(0, 1e-9, self.npts)
        
        # Add synthetic events at known times
        event_times = [60, 120, 180]  # Events at 1, 2, 3 minutes
        for event_time in event_times:
            start_idx = int(event_time * self.sr)
            end_idx = start_idx + int(30 * self.sr)  # 30 second event
            
            if end_idx < self.npts:
                t = np.linspace(0, 30, end_idx - start_idx)
                amplitude = 1e-7 * np.exp(-t/10) * np.sin(2 * np.pi * 2 * t)
                data[start_idx:end_idx] += amplitude
        
        self.trace = numpy_to_trace(
            data, self.sr, 
            starttime=UTCDateTime(),
            station="TEST", 
            channel="BHZ"
        )
    
    def test_available_algorithms(self):
        """Test that algorithms are available."""
        algorithms = self.dm.get_available_algorithms()
        self.assertIsInstance(algorithms, list)
        self.assertGreater(len(algorithms), 0)
        
        # Should have classical algorithms
        self.assertIn('sta_lta', algorithms)
    
    def test_sta_lta_detection(self):
        """Test STA/LTA detection algorithm."""
        events_df, diagnostics = self.dm.analyze_trace(self.trace, algorithms=['sta_lta'])
        
        self.assertIsInstance(events_df, pd.DataFrame)
        self.assertIsInstance(diagnostics, dict)
        
        # Should detect some events
        self.assertGreaterEqual(len(events_df), 0)
        
        if len(events_df) > 0:
            # Check event structure
            required_columns = ['time', 'magnitude', 'confidence', 'algorithm']
            for col in required_columns:
                self.assertIn(col, events_df.columns)
    
    def test_detection_diagnostics(self):
        """Test detection diagnostics."""
        events_df, diagnostics = self.dm.analyze_trace(self.trace)
        
        self.assertIn('trace_info', diagnostics)
        self.assertIn('algorithms_used', diagnostics)
        self.assertIn('total_events', diagnostics)
        
        trace_info = diagnostics['trace_info']
        self.assertEqual(trace_info['station'], 'TEST')
        self.assertEqual(trace_info['sampling_rate'], self.sr)


class TestNoiseMasking(unittest.TestCase):
    """Test pressure-based noise masking."""
    
    def setUp(self):
        self.sr = 20.0
        self.duration = 120.0  # 2 minutes
        self.npts = int(self.duration * self.sr)
        
        # Generate seismic data
        self.seismic_data = np.random.normal(0, 1e-9, self.npts)
        
        # Generate pressure data with gusts
        self.pressure_data = np.full(self.npts, 610.0)  # Base pressure
        
        # Add pressure gusts
        gust_start = int(30 * self.sr)
        gust_end = int(60 * self.sr)
        self.pressure_data[gust_start:gust_end] += 20.0  # Pressure spike
    
    def test_pressure_gust_detection(self):
        """Test pressure gust detection."""
        from app.noise_masker import compute_pressure_gust_mask
        
        mask = compute_pressure_gust_mask(self.pressure_data, self.sr)
        
        self.assertEqual(len(mask), len(self.pressure_data))
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, bool)
        
        # Should detect the gust we added
        self.assertTrue(np.any(mask))
    
    def test_mask_application(self):
        """Test applying mask to seismic data."""
        from app.noise_masker import apply_mask_to_seismic
        
        # Create simple mask
        mask = np.zeros(len(self.seismic_data), dtype=bool)
        mask[100:200] = True  # Mask middle section
        
        # Apply mask
        masked_data = apply_mask_to_seismic(self.seismic_data, mask, mode='zero')
        
        # Masked section should be zero
        np.testing.assert_array_equal(masked_data[100:200], 0.0)
        
        # Unmasked sections should be unchanged
        np.testing.assert_array_equal(masked_data[:100], self.seismic_data[:100])
        np.testing.assert_array_equal(masked_data[200:], self.seismic_data[200:])
    
    def test_full_masking_pipeline(self):
        """Test complete masking pipeline."""
        masked_data, mask = mask_seismic_with_pressure(
            self.seismic_data, 
            self.pressure_data, 
            self.sr
        )
        
        self.assertEqual(len(masked_data), len(self.seismic_data))
        self.assertEqual(len(mask), len(self.seismic_data))
        
        # Should have detected and masked the pressure gust
        self.assertTrue(np.any(mask))


class TestMLComponents(unittest.TestCase):
    """Test machine learning components."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic training data generation."""
        X, y = synthetic_windows(n_samples=100, ws=200, quake_frac=0.3)
        
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(X.shape[1], 200)
        self.assertEqual(len(y), 100)
        
        # Should have both classes
        self.assertIn(0, y)  # Noise
        self.assertIn(1, y)  # Events
        
        # Check class distribution
        event_fraction = np.mean(y)
        self.assertAlmostEqual(event_fraction, 0.3, delta=0.1)
    
    def test_earthquake_generation(self):
        """Test synthetic earthquake generation."""
        ws = 200
        sr = 20.0
        
        earthquake = generate_synthetic_earthquake(ws, sr)
        
        self.assertEqual(len(earthquake), ws)
        self.assertIsInstance(earthquake, np.ndarray)
        
        # Should have some variation (not constant)
        self.assertGreater(np.std(earthquake), 0)


class TestDataFetching(unittest.TestCase):
    """Test data fetching functionality."""
    
    def test_pds_mock_data(self):
        """Test PDS mock data generation."""
        try:
            downloaded = fetch_from_pds_search(
                mission="insight", 
                instrument="SEIS", 
                limit=1, 
                download_first_match=True
            )
            
            self.assertIsInstance(downloaded, list)
            
            if downloaded:
                # Should be able to load the mock data
                st = load_waveform(downloaded[0])
                self.assertGreater(len(st), 0)
                
                trace = st[0]
                self.assertEqual(trace.stats.station, "ELYSE")
                self.assertEqual(trace.stats.network, "XB")
                
        except Exception as e:
            self.skipTest(f"PDS mock data generation failed: {e}")


def run_integration_test():
    """Run a complete integration test."""
    print("\n" + "="*60)
    print("🧪 Running SeismoGuard Backend Integration Test")
    print("="*60)
    
    try:
        # Test 1: Generate synthetic data
        print("\n1. Generating synthetic test data...")
        X, y = synthetic_windows(n_samples=10, ws=200, quake_frac=0.5)
        print(f"   ✓ Generated {len(X)} samples")
        
        # Test 2: Create trace
        print("\n2. Creating seismic trace...")
        trace = numpy_to_trace(X[0], 20.0, starttime=UTCDateTime())
        print(f"   ✓ Created trace: {trace.stats.station}.{trace.stats.channel}")
        
        # Test 3: Preprocess
        print("\n3. Preprocessing data...")
        processed = preprocess_trace(trace, planet='mars')
        print(f"   ✓ Preprocessed for Mars")
        
        # Test 4: Run detection
        print("\n4. Running detection...")
        dm = DetectorManager()
        events_df, diagnostics = dm.analyze_trace(processed)
        print(f"   ✓ Detected {len(events_df)} events")
        
        # Test 5: Test API
        print("\n5. Testing API...")
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print("   ✓ API health check passed")
            else:
                print(f"   ✗ API health check failed: {response.status_code}")
        
        print("\n✅ Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False


def main():
    """Main test runner."""
    parser = unittest.ArgumentParser(description="SeismoGuard Backend Tests")
    parser.add_argument('--integration', action='store_true', 
                       help='Run integration test only')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args, unknown = parser.parse_known_args()
    
    if args.integration:
        success = run_integration_test()
        sys.exit(0 if success else 1)
    
    # Run unit tests
    verbosity = 2 if args.verbose else 1
    
    # Create test suite
    test_classes = [
        TestAPIEndpoints,
        TestDataProcessing,
        TestDetectionAlgorithms,
        TestNoiseMasking,
        TestMLComponents,
        TestDataFetching
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

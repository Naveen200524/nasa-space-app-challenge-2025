"""
CLI wrapper with noise masking and distillation helpers.
"""

import argparse
import os
import sys
import logging
from obspy import UTCDateTime
from .data_fetcher import fetch_from_iris, fetch_from_url, fetch_from_pds_search
from .io_utils import load_waveform, trace_to_numpy
from .detector_manager import DetectorManager
from .noise_masker import mask_seismic_with_pressure
from .distill import run_distillation_pipeline
from .tflite_utils import demo_tflite_inference, get_tflite_model_info
from .preprocess import preprocess_trace
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_file(path: str, planet: str = 'earth', pressure_path: str = None, 
            output_dir: str = 'outputs'):
    """
    Run detection on a local file.
    
    Args:
        path: Path to seismic data file
        planet: Planet configuration
        pressure_path: Optional path to pressure data file
        output_dir: Output directory for results
    """
    try:
        logger.info(f"Processing file: {path}")
        
        # Load seismic data
        st = load_waveform(path)
        tr = st[0]
        
        logger.info(f"Loaded trace: {tr.stats.station}.{tr.stats.channel}, "
                   f"Duration: {tr.stats.endtime - tr.stats.starttime:.1f}s")
        
        # Load pressure data if provided
        pressure_trace = None
        if pressure_path:
            logger.info(f"Loading pressure data: {pressure_path}")
            pst = load_waveform(pressure_path)
            pressure_trace = pst[0]
        
        # Preprocess seismic data
        logger.info(f"Preprocessing for planet: {planet}")
        processed_trace = preprocess_trace(tr, planet=planet)
        
        # Apply noise masking if pressure data available
        if pressure_trace is not None:
            logger.info("Applying pressure-based noise masking")
            try:
                # Align sampling rates
                if abs(pressure_trace.stats.sampling_rate - processed_trace.stats.sampling_rate) > 1e-6:
                    pressure_trace.resample(processed_trace.stats.sampling_rate)
                
                # Convert to numpy
                s, _ = trace_to_numpy(processed_trace)
                p, _ = trace_to_numpy(pressure_trace)
                
                # Apply masking
                masked, mask = mask_seismic_with_pressure(
                    s, p, sr=processed_trace.stats.sampling_rate,
                    win_sec=1.0, std_thresh=3.0, dilate_sec=2.0, mode='zero'
                )
                
                # Update trace data
                masked = np.nan_to_num(masked, nan=0.0)
                processed_trace.data = masked.astype(processed_trace.data.dtype)
                
                mask_percentage = np.sum(mask) / len(mask) * 100
                logger.info(f"Noise masking applied: {mask_percentage:.1f}% of samples masked")
                
            except Exception as e:
                logger.warning(f"Noise masking failed: {e}")
        
        # Run detection
        logger.info("Running seismic event detection")
        dm = DetectorManager(model_root='models')
        events_df, diagnostics = dm.analyze_trace(processed_trace, planet=planet)
        
        if events_df is None or events_df.empty:
            logger.info("No events detected")
            print("No events detected.")
            print(f"Diagnostics: {diagnostics}")
        else:
            logger.info(f"Detected {len(events_df)} events")
            print(f"Detected {len(events_df)} events:")
            
            for _, event in events_df.iterrows():
                print(f"  Time: {event['time']}, Magnitude: {event.get('magnitude', 'N/A'):.2f}, "
                      f"Confidence: {event.get('confidence', 'N/A'):.2f}, "
                      f"Algorithm: {event.get('algorithm', 'N/A')}")
            
            # Save results
            os.makedirs(output_dir, exist_ok=True)
            
            # Save events CSV
            csv_out = os.path.join(output_dir, os.path.basename(path) + '.events.csv')
            events_df.to_csv(csv_out, index=False)
            logger.info(f"Events saved to: {csv_out}")
            
            # Try to create visualization
            try:
                from .visualizer import plot_trace_with_events
                plot_out = os.path.join(output_dir, os.path.basename(path) + '.png')
                plot_trace_with_events(processed_trace, events_df, outpath=plot_out)
                logger.info(f"Plot saved to: {plot_out}")
            except ImportError:
                logger.warning("Visualization module not available")
            except Exception as e:
                logger.warning(f"Could not create plot: {e}")
            
            print(f"Results saved to: {output_dir}")
    
    except Exception as e:
        logger.error(f"Error processing file {path}: {e}")
        print(f"Error: {e}")


def run_iris_fetch(network: str, station: str, channel: str, 
                  starttime: str, endtime: str, planet: str = 'earth'):
    """
    Fetch data from IRIS and run detection.
    """
    try:
        logger.info(f"Fetching from IRIS: {network}.{station}.{channel}")
        
        t1 = UTCDateTime(starttime)
        t2 = UTCDateTime(endtime)
        
        path = fetch_from_iris(network, station, channel, t1, t2)
        run_file(path, planet=planet)
        
    except Exception as e:
        logger.error(f"IRIS fetch failed: {e}")
        print(f"Error: {e}")


def run_url_fetch(url: str, planet: str = 'earth'):
    """
    Fetch data from URL and run detection.
    """
    try:
        logger.info(f"Fetching from URL: {url}")
        
        path = fetch_from_url(url)
        run_file(path, planet=planet)
        
    except Exception as e:
        logger.error(f"URL fetch failed: {e}")
        print(f"Error: {e}")


def run_pds_search(mission: str = 'insight', instrument: str = 'SEIS', 
                  limit: int = 20, planet: str = 'earth'):
    """
    Search PDS and run detection on downloaded data.
    """
    try:
        logger.info(f"Searching PDS: {mission} {instrument}")
        
        downloaded = fetch_from_pds_search(
            mission=mission, 
            instrument=instrument, 
            limit=limit
        )
        
        if not downloaded:
            print("No products downloaded from PDS.")
            return
        
        print(f"Downloaded {len(downloaded)} files from PDS")
        
        for i, path in enumerate(downloaded):
            print(f"\nProcessing file {i+1}/{len(downloaded)}: {os.path.basename(path)}")
            run_file(path, planet=planet)
    
    except Exception as e:
        logger.error(f"PDS search failed: {e}")
        print(f"Error: {e}")


def train_models(epochs: int = 6, ws: int = 200):
    """
    Train synthetic ML models.
    """
    try:
        from .ml_train import train_classifier, train_autoencoder
        
        logger.info("Training classifier...")
        train_classifier(epochs=epochs, ws=ws)
        
        logger.info("Training autoencoder...")
        train_autoencoder(epochs=epochs, ws=ws)
        
        print("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Error: {e}")


def run_distillation(ws: int = 200, epochs: int = 5, batch_size: int = 32):
    """
    Run distillation pipeline.
    """
    try:
        logger.info("Starting distillation pipeline...")
        
        distilled_path, tflite_path = run_distillation_pipeline(
            ws=ws, 
            epochs=epochs, 
            batch_size=batch_size,
            tflite_out='models/compact_quant.tflite'
        )
        
        print("Distillation completed successfully!")
        print(f"Distilled model: {distilled_path}")
        if tflite_path:
            print(f"TFLite model: {tflite_path}")
        
    except Exception as e:
        logger.error(f"Distillation failed: {e}")
        print(f"Error: {e}")


def test_tflite(model_path: str):
    """
    Test TFLite model inference.
    """
    try:
        if not os.path.exists(model_path):
            print(f"TFLite model not found: {model_path}")
            return
        
        # Get model info
        info = get_tflite_model_info(model_path)
        print(f"TFLite Model Info:")
        print(f"  File size: {info.get('file_size_kb', 0):.1f} KB")
        print(f"  Input shape: {info.get('input_shape', 'unknown')}")
        print(f"  Output shape: {info.get('output_shape', 'unknown')}")
        
        # Generate test input
        input_shape = info.get('input_shape', [1, 200, 1])
        if len(input_shape) >= 2:
            test_input = np.random.normal(0, 1, input_shape[1]).astype(np.float32)
        else:
            test_input = np.random.normal(0, 1, 200).astype(np.float32)
        
        # Run inference
        output = demo_tflite_inference(model_path, test_input)
        print(f"Test inference successful!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output: {output}")
        
    except Exception as e:
        logger.error(f"TFLite test failed: {e}")
        print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SeismoGuard Backend CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process local file
  python -m app.cli --file data/sample.mseed --planet mars
  
  # Process with pressure masking
  python -m app.cli --file data/seis.mseed --pressure data/pressure.mseed
  
  # Fetch from IRIS
  python -m app.cli --iris IU ANMO BHZ 2023-01-01T00:00:00 2023-01-01T01:00:00
  
  # Search PDS
  python -m app.cli --pds_search insight SEIS 5 --planet mars
  
  # Train models
  python -m app.cli --train --epochs 10
  
  # Run distillation
  python -m app.cli --distill --epochs 8
  
  # Test TFLite model
  python -m app.cli --test_tflite models/compact_quant.tflite
        """
    )
    
    # Input options
    parser.add_argument('--file', help='Local seismic data file')
    parser.add_argument('--pressure', help='Local pressure data file for noise masking')
    parser.add_argument('--iris', nargs=5, metavar=('NET', 'STA', 'CHA', 'START', 'END'),
                       help='Fetch from IRIS: network station channel starttime endtime')
    parser.add_argument('--url', help='Fetch from HTTP(S) URL')
    parser.add_argument('--pds_search', nargs='*', 
                       help='PDS search: [mission] [instrument] [limit]')
    
    # Processing options
    parser.add_argument('--planet', choices=['moon', 'mars', 'earth'], default='earth',
                       help='Planet configuration for processing')
    parser.add_argument('--output', default='outputs', help='Output directory')
    
    # Training options
    parser.add_argument('--train', action='store_true', help='Train synthetic ML models')
    parser.add_argument('--distill', action='store_true', 
                       help='Run distillation -> TFLite pipeline')
    parser.add_argument('--epochs', type=int, default=6, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--window_size', type=int, default=200, help='Window size for training')
    
    # Testing options
    parser.add_argument('--test_tflite', help='Test TFLite model inference')
    
    # Utility options
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle different modes
    if args.train:
        train_models(epochs=args.epochs, ws=args.window_size)
    
    elif args.distill:
        run_distillation(ws=args.window_size, epochs=args.epochs, batch_size=args.batch_size)
    
    elif args.test_tflite:
        test_tflite(args.test_tflite)
    
    elif args.file:
        run_file(args.file, planet=args.planet, pressure_path=args.pressure, 
                output_dir=args.output)
    
    elif args.iris:
        network, station, channel, starttime, endtime = args.iris
        run_iris_fetch(network, station, channel, starttime, endtime, planet=args.planet)
    
    elif args.url:
        run_url_fetch(args.url, planet=args.planet)
    
    elif args.pds_search is not None:
        mission = args.pds_search[0] if len(args.pds_search) >= 1 else 'insight'
        instrument = args.pds_search[1] if len(args.pds_search) >= 2 else 'SEIS'
        limit = int(args.pds_search[2]) if len(args.pds_search) >= 3 else 20
        run_pds_search(mission=mission, instrument=instrument, limit=limit, planet=args.planet)
    
    else:
        parser.print_help()
        print("\nError: No input provided. Use --file, --iris, --url, --pds_search, --train, or --distill.")
        sys.exit(1)


if __name__ == "__main__":
    main()

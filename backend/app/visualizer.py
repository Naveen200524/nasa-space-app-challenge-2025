"""
Visualization utilities for seismic data and detection results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from obspy import Trace
from typing import Optional, List, Tuple
import os
from datetime import datetime


def plot_trace_with_events(trace: Trace, events_df: pd.DataFrame, 
                          outpath: Optional[str] = None, 
                          figsize: Tuple[int, int] = (12, 6)) -> str:
    """
    Plot seismic trace with detected events.
    
    Args:
        trace: ObsPy Trace object
        events_df: DataFrame with detected events
        outpath: Output file path (optional)
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot trace
        times = trace.times('matplotlib')
        data = trace.data
        
        ax.plot(times, data, 'b-', linewidth=0.5, alpha=0.8, label='Seismic Data')
        
        # Plot events
        if not events_df.empty:
            for _, event in events_df.iterrows():
                event_time = event['time']
                if hasattr(event_time, 'matplotlib_date'):
                    event_mpl_time = event_time.matplotlib_date
                else:
                    # Convert string to matplotlib date if needed
                    from obspy import UTCDateTime
                    event_utc = UTCDateTime(str(event_time))
                    event_mpl_time = event_utc.matplotlib_date
                
                # Find corresponding amplitude
                time_diff = np.abs(times - event_mpl_time)
                closest_idx = np.argmin(time_diff)
                amplitude = data[closest_idx]
                
                # Plot event marker
                magnitude = event.get('magnitude', 0)
                confidence = event.get('confidence', 0.5)
                algorithm = event.get('algorithm', 'unknown')
                
                # Size marker based on magnitude
                marker_size = max(20, magnitude * 10)
                
                # Color based on confidence
                color = plt.cm.Reds(0.5 + confidence * 0.5)
                
                ax.scatter(event_mpl_time, amplitude, s=marker_size, c=[color], 
                          marker='o', edgecolors='red', linewidth=1, 
                          alpha=0.8, zorder=5)
                
                # Add text annotation
                ax.annotate(f'M{magnitude:.1f}', 
                           xy=(event_mpl_time, amplitude),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
        
        # Format plot
        ax.set_xlabel('Time (UTC)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Seismic Data: {trace.stats.station}.{trace.stats.channel}\n'
                    f'{trace.stats.starttime} - {trace.stats.endtime}')
        
        # Format time axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        if not events_df.empty:
            ax.legend(['Seismic Data', 'Detected Events'])
        
        plt.tight_layout()
        
        # Save plot
        if outpath is None:
            outpath = f"seismic_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return outpath
        
    except Exception as e:
        plt.close()
        raise Exception(f"Failed to create plot: {str(e)}")


def plot_detection_summary(events_df: pd.DataFrame, 
                          outpath: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> str:
    """
    Create summary plots of detection results.
    
    Args:
        events_df: DataFrame with detected events
        outpath: Output file path (optional)
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    try:
        if events_df.empty:
            raise ValueError("No events to plot")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Magnitude distribution
        if 'magnitude' in events_df.columns:
            axes[0, 0].hist(events_df['magnitude'], bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_xlabel('Magnitude')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Magnitude Distribution')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Confidence distribution
        if 'confidence' in events_df.columns:
            axes[0, 1].hist(events_df['confidence'], bins=20, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Confidence')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Confidence Distribution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Algorithm comparison
        if 'algorithm' in events_df.columns:
            alg_counts = events_df['algorithm'].value_counts()
            axes[1, 0].bar(alg_counts.index, alg_counts.values, alpha=0.7, color='orange')
            axes[1, 0].set_xlabel('Algorithm')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Events by Algorithm')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Time series of events
        if 'time' in events_df.columns:
            times = pd.to_datetime(events_df['time'])
            axes[1, 1].plot(times, range(len(times)), 'o-', alpha=0.7, color='red')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Cumulative Events')
            axes[1, 1].set_title('Events Over Time')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if outpath is None:
            outpath = f"detection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return outpath
        
    except Exception as e:
        plt.close()
        raise Exception(f"Failed to create summary plot: {str(e)}")


def plot_noise_masking_results(original_data: np.ndarray, masked_data: np.ndarray,
                              mask: np.ndarray, sr: float,
                              outpath: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> str:
    """
    Plot noise masking results.
    
    Args:
        original_data: Original seismic data
        masked_data: Masked seismic data
        mask: Boolean mask array
        sr: Sampling rate
        outpath: Output file path (optional)
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    try:
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Time axis
        times = np.arange(len(original_data)) / sr
        
        # Original data
        axes[0].plot(times, original_data, 'b-', linewidth=0.5, alpha=0.8)
        axes[0].set_ylabel('Original Amplitude')
        axes[0].set_title('Noise Masking Results')
        axes[0].grid(True, alpha=0.3)
        
        # Mask
        axes[1].fill_between(times, 0, mask.astype(int), alpha=0.7, color='red')
        axes[1].set_ylabel('Mask')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].grid(True, alpha=0.3)
        
        # Masked data
        axes[2].plot(times, masked_data, 'g-', linewidth=0.5, alpha=0.8)
        axes[2].set_ylabel('Masked Amplitude')
        axes[2].set_xlabel('Time (s)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if outpath is None:
            outpath = f"noise_masking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return outpath
        
    except Exception as e:
        plt.close()
        raise Exception(f"Failed to create masking plot: {str(e)}")


def plot_spectrogram(trace: Trace, outpath: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 6)) -> str:
    """
    Create spectrogram of seismic trace.
    
    Args:
        trace: ObsPy Trace object
        outpath: Output file path (optional)
        figsize: Figure size
        
    Returns:
        Path to saved plot
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Time series
        times = trace.times()
        data = trace.data
        
        ax1.plot(times, data, 'b-', linewidth=0.5)
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Seismic Data and Spectrogram: {trace.stats.station}.{trace.stats.channel}')
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram
        from scipy import signal
        
        nperseg = min(1024, len(data) // 8)
        f, t, Sxx = signal.spectrogram(data, trace.stats.sampling_rate, nperseg=nperseg)
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        
        im = ax2.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylim(0, min(20, trace.stats.sampling_rate / 2))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Power (dB)')
        
        plt.tight_layout()
        
        # Save plot
        if outpath is None:
            outpath = f"spectrogram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return outpath
        
    except Exception as e:
        plt.close()
        raise Exception(f"Failed to create spectrogram: {str(e)}")


def create_detection_report(trace: Trace, events_df: pd.DataFrame, 
                           diagnostics: dict, output_dir: str = 'outputs') -> str:
    """
    Create a comprehensive detection report with multiple plots.
    
    Args:
        trace: ObsPy Trace object
        events_df: DataFrame with detected events
        diagnostics: Detection diagnostics
        output_dir: Output directory
        
    Returns:
        Path to report directory
    """
    try:
        # Create report directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(output_dir, f'detection_report_{timestamp}')
        os.makedirs(report_dir, exist_ok=True)
        
        # Create plots
        plots_created = []
        
        # Main trace plot
        try:
            plot_path = plot_trace_with_events(
                trace, events_df, 
                os.path.join(report_dir, 'trace_with_events.png')
            )
            plots_created.append(plot_path)
        except Exception as e:
            print(f"Warning: Could not create trace plot: {e}")
        
        # Summary plots
        if not events_df.empty:
            try:
                summary_path = plot_detection_summary(
                    events_df,
                    os.path.join(report_dir, 'detection_summary.png')
                )
                plots_created.append(summary_path)
            except Exception as e:
                print(f"Warning: Could not create summary plot: {e}")
        
        # Spectrogram
        try:
            spec_path = plot_spectrogram(
                trace,
                os.path.join(report_dir, 'spectrogram.png')
            )
            plots_created.append(spec_path)
        except Exception as e:
            print(f"Warning: Could not create spectrogram: {e}")
        
        # Save data files
        events_df.to_csv(os.path.join(report_dir, 'events.csv'), index=False)
        
        # Save diagnostics
        import json
        with open(os.path.join(report_dir, 'diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        # Create summary text
        with open(os.path.join(report_dir, 'summary.txt'), 'w') as f:
            f.write(f"SeismoGuard Detection Report\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write(f"Trace Information:\n")
            f.write(f"  Station: {trace.stats.station}\n")
            f.write(f"  Channel: {trace.stats.channel}\n")
            f.write(f"  Start Time: {trace.stats.starttime}\n")
            f.write(f"  End Time: {trace.stats.endtime}\n")
            f.write(f"  Duration: {trace.stats.endtime - trace.stats.starttime:.1f} seconds\n")
            f.write(f"  Sampling Rate: {trace.stats.sampling_rate} Hz\n\n")
            
            f.write(f"Detection Results:\n")
            f.write(f"  Total Events: {len(events_df)}\n")
            
            if not events_df.empty:
                f.write(f"  Magnitude Range: {events_df['magnitude'].min():.2f} - {events_df['magnitude'].max():.2f}\n")
                f.write(f"  Average Confidence: {events_df['confidence'].mean():.2f}\n")
                
                if 'algorithm' in events_df.columns:
                    alg_counts = events_df['algorithm'].value_counts()
                    f.write(f"  Algorithms Used:\n")
                    for alg, count in alg_counts.items():
                        f.write(f"    {alg}: {count} events\n")
            
            f.write(f"\nFiles Created:\n")
            for plot_path in plots_created:
                f.write(f"  {os.path.basename(plot_path)}\n")
            f.write(f"  events.csv\n")
            f.write(f"  diagnostics.json\n")
        
        print(f"Detection report created: {report_dir}")
        return report_dir
        
    except Exception as e:
        raise Exception(f"Failed to create detection report: {str(e)}")

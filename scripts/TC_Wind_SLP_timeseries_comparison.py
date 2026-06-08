# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:25:29 2026

@author: user
"""

# -*- coding: utf-8 -*-
"""
Create 4x2 small multiples comparing OBS, ELLIP (control), and ELLIP (1010) for pressure

This script generates a publication-ready figure showing pressure time series
for all eight TCs at a specified station, comparing:
- Observed pressure (black)
- ELLIP model with climatological ambient pressure (green solid)
- ELLIP model with fixed 1010 hPa ambient pressure (red dashed)

Usage:
    python scripts/TC_SLP_WS_timeseries_comparison.py --station HKO_AWS
    python scripts/TC_SLP_WS_timeseries_comparison.py --station CCH_AWS --tc HATO MANGKHUT
"""

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import argparse

# Import configuration
try:
    from config import OUTPUT_DIR
except ImportError:
    # Fallback for when run as standalone script
    OUTPUT_DIR = Path.cwd() / "results"

# Set font sizes for A4 paper readability
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'xtick.labelsize': 20,
    'ytick.labelsize': 24,
    'legend.fontsize': 20,
    'figure.titlesize': 28,
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
})

# Path to results_with_wrf folder
folder = OUTPUT_DIR / "results_with_wrf"


def load_and_combine_data(folder):
    """Load all prediction CSV files and combine into a single DataFrame."""
    all_dfs = []
    
    for file in folder.glob("predictions_*.csv"):
        fname = file.name
        
        # Determine suffix
        if "_1010" in fname:  # Fixed: removed trailing underscore
            suffix = "1010"
        elif "_nogd_" in fname or "restricted" in fname:
            continue  # Skip other sensitivity tests
        else:
            suffix = "None"
        
        tc_name = fname.split("_")[1]
        
        df = pd.read_csv(file)
        df['tc_name'] = tc_name
        df['suffix'] = suffix
        all_dfs.append(df)
    
    if not all_dfs:
        raise FileNotFoundError(f"No prediction files found in {folder}")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Pressure columns
    p_columns = [col for col in combined.columns if isinstance(col, str) and col.startswith('P_')]
    
    # Convert to numeric and filter invalid values
    for col in p_columns:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')
        combined = combined[(combined[col] >= 800) & (combined[col] <= 1100)]
    
    return combined


# Define TC order (UPPERCASE to match your data)
TC_ORDER = ['HATO', 'MANGKHUT', 'VICENTE', 'NURI', 'KOINU', 'SAOLA', 'WIPHA', 'RAGASA']

# Define x-axis restrictions for specific TCs
XAXIS_RESTRICTIONS = {
    'VICENTE': {
        'start': datetime(2012, 7, 22, 20, 0),
        'end': None
    },
    'KOINU': {
        'start': datetime(2023, 10, 7, 5, 0),
        'end': None
    },
    'SAOLA': {
        'start': datetime(2023, 8, 31, 10, 0),
        'end': None
    },
    'WIPHA': {
        'start': None,
        'end': datetime(2025, 7, 21, 4, 0)
    },
}


def apply_xaxis_restriction(tc_name, time_min, time_max):
    """Apply x-axis restrictions for specific TCs."""
    
    if tc_name in XAXIS_RESTRICTIONS:
        restriction = XAXIS_RESTRICTIONS[tc_name]
        
        if restriction['start'] is not None:
            start_dt = pd.to_datetime(restriction['start'])
            if start_dt > time_min:
                time_min = start_dt
        if restriction['end'] is not None:
            end_dt = pd.to_datetime(restriction['end'])
            if end_dt < time_max:
                time_max = end_dt
    
    return time_min, time_max


def create_pressure_comparison_plot(station='HKO_AWS', tc_list=None):
    """
    Create 4x2 small multiples comparing OBS, ELLIP (control), and ELLIP (1010)
    
    Parameters
    ----------
    station : str
        Station ID (e.g., 'HKO_AWS', 'CCH_AWS', 'LFS_AWS')
    tc_list : list, optional
        List of TC names to plot (default: all 8 TCs in TC_ORDER)
    """
    
    # Load data
    combined = load_and_combine_data(folder)
    
    # Filter data for this station
    data = combined[combined['ID'] == station].copy()
    
    if len(data) == 0:
        print(f"No data found for station {station}")
        return
    
    # Determine which TCs to plot
    if tc_list is None:
        tcs_to_plot = [tc for tc in TC_ORDER if tc in data['tc_name'].unique()]
    else:
        tcs_to_plot = [tc for tc in tc_list if tc in data['tc_name'].unique()]
    
    if len(tcs_to_plot) == 0:
        print(f"No TCs found for station {station}")
        return
    
    print(f"Available TCs: {tcs_to_plot}")
    
    # Set up
    obs_col = 'P_obs'
    y_label = 'Pressure (hPa)'
    
    # Station name mapping for display
    station_names = {
        'HKO_AWS': 'Hong Kong Observatory (HKO)',
        'CCH_AWS': 'Cheung Chau (CCH)',
        'LFS_AWS': 'Lau Fau Shan (LFS)'
    }
    location = station_names.get(station, station)
    
    # Create 4x2 grid - TALLER figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 16), sharex=False, sharey=False)
    axes = axes.flatten()
    
    # Store y-limits for each TC
    y_limits = {}
    
    # Plot each TC
    for idx, tc_name in enumerate(tcs_to_plot[:8]):
        ax = axes[idx]
        
        # Get control test data for this TC
        control_data = data[(data['tc_name'] == tc_name) & (data['suffix'] == 'None')].copy()
        # Get 1010 test data for this TC
        test1010_data = data[(data['tc_name'] == tc_name) & (data['suffix'] == '1010')].copy()
        
        if len(control_data) == 0 and len(test1010_data) == 0:
            ax.text(0.5, 0.5, f'No data for {tc_name}', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(tc_name.capitalize(), fontweight='bold')
            continue
        
        # Convert Time to datetime and sort
        if len(control_data) > 0:
            control_data['Time'] = pd.to_datetime(control_data['Time'])
            control_data = control_data.sort_values('Time')
        if len(test1010_data) > 0:
            test1010_data['Time'] = pd.to_datetime(test1010_data['Time'])
            test1010_data = test1010_data.sort_values('Time')
        
        # Determine overall time range for x-axis limits
        all_times = []
        if len(control_data) > 0:
            all_times.extend(control_data['Time'].tolist())
        if len(test1010_data) > 0:
            all_times.extend(test1010_data['Time'].tolist())
        
        if len(all_times) > 0:
            time_min = min(all_times)
            time_max = max(all_times)
        else:
            time_min = pd.Timestamp.now()
            time_max = pd.Timestamp.now()
        
        print(f"  {tc_name}: time range {time_min} to {time_max}")
        
        # Apply x-axis restrictions
        time_min, time_max = apply_xaxis_restriction(tc_name, time_min, time_max)
        print(f"    After restriction: {time_min} to {time_max}")
        
        # Plot OBS (black solid line, bold)
        if len(control_data) > 0 and obs_col in control_data.columns:
            obs_data = control_data.dropna(subset=[obs_col])
            if len(obs_data) > 0:
                ax.plot(obs_data['Time'], obs_data[obs_col],
                        color='black', linewidth=5,
                        label='OBS', zorder=10)
                
                # Store y-limits
                y_min = obs_data[obs_col].min() - 5
                y_max = obs_data[obs_col].max() + 5
                y_limits[tc_name] = (y_min, y_max)
        
        # Plot ELLIP (control) - green solid line
        ellip_control_col = 'P_ellip'
        if len(control_data) > 0 and ellip_control_col in control_data.columns:
            ellip_control_data = control_data.dropna(subset=[ellip_control_col])
            if len(ellip_control_data) > 0:
                ax.plot(ellip_control_data['Time'], ellip_control_data[ellip_control_col],
                        color='green', linewidth=3, linestyle='-',
                        marker='o', markersize=6, markevery=max(1, len(ellip_control_data)//8),
                        label='ELLIP (Control)', alpha=0.8)
        
        # Plot ELLIP (1010) - RED dashed line
        ellip_1010_col = 'P_ellip'
        if len(test1010_data) > 0 and ellip_1010_col in test1010_data.columns:
            ellip_1010_data = test1010_data.dropna(subset=[ellip_1010_col])
            if len(ellip_1010_data) > 0:
                ax.plot(ellip_1010_data['Time'], ellip_1010_data[ellip_1010_col],
                        color='red', linewidth=3, linestyle='--',
                        marker='s', markersize=6, markevery=max(1, len(ellip_1010_data)//8),
                        label='ELLIP (P$_{ambi}$=1010 hPa)', alpha=0.8)
        
        # Set x-axis limits
        ax.set_xlim(time_min, time_max)
        
        # Set y-axis limits (individual for each TC)
        if tc_name in y_limits:
            ax.set_ylim(y_limits[tc_name])
        
        # Format x-axis ticks with 8-hour intervals
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=16, rotation=0)
        
        # Customize subplot
        display_name = tc_name.capitalize()
        ax.set_title(display_name, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Only show y-label for leftmost columns
        if idx % 4 == 0:
            ax.set_ylabel(y_label, fontweight='bold')
    
    # Hide any unused subplots
    for idx in range(len(tcs_to_plot[:8]), 8):
        axes[idx].set_visible(False)
    
    # Collect legend handles (deduplicate)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    
    # Legend order: OBS, ELLIP (Control), ELLIP (1010)
    order = ['OBS', 'ELLIP (Control)', 'ELLIP (P$_{ambi}$=1010 hPa)']
    ordered_handles, ordered_labels = [], []
    for label_key in order:
        if label_key in labels:
            idx = labels.index(label_key)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
    
    # Add legend
    fig.legend(ordered_handles, ordered_labels, loc='lower center', 
               ncol=len(ordered_handles), bbox_to_anchor=(0.5, 0.01),
               frameon=True, fontsize=20)
    
    # Add main title
    fig.suptitle(f'Pressure Time Series at {location} - Control vs. P$_{{ambi}}$=1010 hPa', 
                 fontsize=28, fontweight='bold', y=0.99)
    
    # Adjust layout with more bottom padding for legend
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.06, right=0.98, 
                       hspace=0.25, wspace=0.15)
    
    # Save figure - ensure plots directory exists
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    output_filename = f"pressure_comparison_ELLIP_control_vs_1010_{station}.png"
    output_path = plots_dir / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    return


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot pressure time series comparison for ELLIP model')
    parser.add_argument('--station', type=str, default='HKO_AWS',
                        help='Station ID (e.g., HKO_AWS, CCH_AWS, LFS_AWS)')
    parser.add_argument('--tc', type=str, nargs='+', 
                        help='TC names to plot (e.g., --tc HATO MANGKHUT)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (overrides config)')
    args = parser.parse_args()
    
    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output_dir)
        folder = OUTPUT_DIR / "results_with_wrf"
    
    print(f"Creating pressure comparison plot for station {args.station}...")
    create_pressure_comparison_plot(station=args.station, tc_list=args.tc)
    
    print("\nPlot completed!")
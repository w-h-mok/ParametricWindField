# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 12:33:06 2026

@author: user
"""

# -*- coding: utf-8 -*-
"""
Generate all result tables for the TC parametric model paper.

This script reads the prediction CSV files from main.py and generates
Tables 1-16 as presented in the paper.

Usage:
    python scripts/table_generator.py
    python scripts/table_generator.py --tables 1 2 3 4
    python scripts/table_generator.py --output-dir ./tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Import configuration
try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path.cwd() / "results"


# ============================================================================
# CONSTANTS
# ============================================================================

TC_NAMES = ['VICENTE', 'HATO', 'MANGKHUT', 'NURI', 'KOINU', 'SAOLA', 'WIPHA', 'RAGASA']
SUFFIXES = ['None', '1010', 'nogd', 'restricted']
PRESSURE_MODELS = ['CIRC-EM', 'ELLIP-EM', 'JTWC-EM', 'WRF']
WIND_MODELS = ['CIRC-EM', 'CIRC-GB', 'ELLIP-EM', 'ELLIP-GB', 'JTWC-EM', 'JTWC-GB', 'WRF']
PRESSURE_STATIONS = ['CCH_AWS', 'HKO_AWS', 'LFS_AWS']
WIND_STATION = 'WGL_AWS'

# Station name mapping
STATION_NAMES = {
    'CCH_AWS': 'CCH', 'HKO_AWS': 'HKO', 'LFS_AWS': 'LFS',
    'HKA_AWS': 'HKA', 'KP_AWS': 'KP', 'SHA_AWS': 'SHA',
    'SLW_AWS': 'SLW', 'TKL_AWS': 'TKL', 'TMS_AWS': 'TMS',
    'WGL_AWS': 'WGL', 'MC_TG': 'MC_TG'
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_predictions(folder):
    """Load all prediction CSV files and return a combined DataFrame."""
    file_paths = list(folder.glob("predictions_*.csv"))
    
    if not file_paths:
        raise FileNotFoundError(f"No prediction files found in {folder}")
    
    all_dfs = []
    
    for file_path in file_paths:
        filename = file_path.name
        tc = filename.split('_')[1]
        if tc not in TC_NAMES:
            continue
        
        # Determine suffix
        suffix = 'None'
        for s in ['1010', 'nogd', 'restricted']:
            if f'_{s}_' in filename or filename.endswith(f'_{s}.csv'):
                suffix = s
                break
        
        df = pd.read_csv(file_path)
        
        # Convert WRF wind speed from m/s to knots
        if 'WS_wrf' in df.columns:
            df['WS_wrf'] = df['WS_wrf'] / 0.514444444
        
        df['tc_name'] = tc
        df['suffix'] = suffix
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True)


def get_station_data(df, station, suffix, tc_list=None):
    """Filter data for a specific station and suffix."""
    mask = (df['ID'] == station) & (df['suffix'] == suffix)
    if tc_list:
        mask = mask & (df['tc_name'].isin(tc_list))
    return df[mask].copy()


# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def compute_correlation(y_true, y_pred):
    """Compute Pearson correlation coefficient."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(valid_mask) < 2:
        return np.nan
    return np.corrcoef(y_true[valid_mask], y_pred[valid_mask])[0, 1]


def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error."""
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(valid_mask) == 0:
        return np.nan
    return np.mean(np.abs(y_true[valid_mask] - y_pred[valid_mask]))


def compute_metrics_for_models(df, obs_col, model_cols):
    """Compute correlation and MAE for multiple models."""
    results = {}
    
    for model_name, model_col in model_cols.items():
        model_data = df[model_col].dropna()
        obs_data = df[obs_col].dropna()
        
        common_idx = model_data.index.intersection(obs_data.index)
        if len(common_idx) > 1:
            y_true = obs_data.loc[common_idx]
            y_pred = model_data.loc[common_idx]
            results[model_name] = {
                'corr': compute_correlation(y_true, y_pred),
                'mae': compute_mae(y_true, y_pred)
            }
        else:
            results[model_name] = {'corr': np.nan, 'mae': np.nan}
    
    return results


# ============================================================================
# TABLE GENERATORS
# ============================================================================

def print_table_1_2(df):
    """Print Tables 1 and 2: Pressure and Wind Speed summary metrics."""
    
    # Table 1: Pressure
    print("\n" + "=" * 100)
    print("Table 1: Pressure - Correlation and Mean Absolute Error")
    print("(CCH_AWS, HKO_AWS, LFS_AWS)")
    print("=" * 100)
    
    pressure_cols = {
        'CIRC-EM': 'P_circ',
        'ELLIP-EM': 'P_ellip',
        'JTWC-EM': 'P_r34',
        'WRF': 'P_wrf'
    }
    
    header = "Suffix"
    for model in PRESSURE_MODELS:
        header += f"\tP_{model}_corr"
    for model in PRESSURE_MODELS:
        header += f"\tP_{model}_mae"
    print(header)
    
    for suffix in SUFFIXES:
        all_corrs, all_maes = [], []
        for station in PRESSURE_STATIONS:
            station_data = get_station_data(df, station, suffix)
            if len(station_data) == 0:
                continue
            metrics = compute_metrics_for_models(station_data, 'P_obs', pressure_cols)
            for model in PRESSURE_MODELS:
                all_corrs.append(metrics.get(model, {}).get('corr', np.nan))
                all_maes.append(metrics.get(model, {}).get('mae', np.nan))
        
        row = suffix
        for i, model in enumerate(PRESSURE_MODELS):
            val = np.nanmean([all_corrs[j] for j in range(i, len(all_corrs), len(PRESSURE_MODELS))])
            row += f"\t{val:.4f}" if not np.isnan(val) else "\t"
        for i, model in enumerate(PRESSURE_MODELS):
            val = np.nanmean([all_maes[j] for j in range(i, len(all_maes), len(PRESSURE_MODELS))])
            row += f"\t{val:.4f}" if not np.isnan(val) else "\t"
        print(row)
    
    # Table 2: Wind Speed
    print("\n" + "=" * 100)
    print("Table 2: Wind Speed - Correlation and Mean Absolute Error (WGL_AWS)")
    print("=" * 100)
    
    wind_cols = {
        'CIRC-EM': 'WS_circ',
        'CIRC-GB': 'WS_circ_P_only',
        'ELLIP-EM': 'WS_ellip',
        'ELLIP-GB': 'WS_ellip_P_only',
        'JTWC-EM': 'WS_r34',
        'JTWC-GB': 'WS_r34_P_only',
        'WRF': 'WS_wrf'
    }
    
    header = "Metric_Suffix"
    for model in WIND_MODELS:
        header += f"\t{model}"
    print(header)
    
    for suffix in SUFFIXES:
        station_data = get_station_data(df, WIND_STATION, suffix)
        metrics = compute_metrics_for_models(station_data, 'WS_obs', wind_cols)
        
        row_corr = f"Correlation_{suffix}"
        row_mae = f"MAE_{suffix}"
        for model in WIND_MODELS:
            corr = metrics.get(model, {}).get('corr', np.nan)
            mae = metrics.get(model, {}).get('mae', np.nan)
            row_corr += f"\t{corr:.4f}" if not np.isnan(corr) else "\t"
            row_mae += f"\t{mae:.4f}" if not np.isnan(mae) else "\t"
        print(row_corr)
        print(row_mae)


def print_table_3_4(df):
    """Print Tables 3 and 4: Per-TC correlations for pressure and wind."""
    
    # Table 3: Pressure correlation by TC at HKO
    print("\n" + "=" * 100)
    print("Table 3: HKO_AWS - Pressure Correlation by TC (None suffix)")
    print("=" * 100)
    
    pressure_cols = {'CIRC-EM': 'P_circ', 'ELLIP-EM': 'P_ellip', 'JTWC-EM': 'P_r34', 'WRF': 'P_wrf'}
    
    header = "TC"
    for model in PRESSURE_MODELS:
        header += f"\t{model}"
    print(header)
    
    all_corrs = {model: [] for model in PRESSURE_MODELS}
    station_data = get_station_data(df, 'HKO_AWS', 'None')
    
    for tc in TC_NAMES:
        tc_data = station_data[station_data['tc_name'] == tc]
        if len(tc_data) == 0:
            print(f"{tc}\t\t\t\t")
            continue
        
        metrics = compute_metrics_for_models(tc_data, 'P_obs', pressure_cols)
        row = tc
        for model in PRESSURE_MODELS:
            corr = metrics.get(model, {}).get('corr', np.nan)
            if not np.isnan(corr):
                all_corrs[model].append(corr)
                row += f"\t{corr:.4f}"
            else:
                row += "\t"
        print(row)
    
    row = "Average"
    for model in PRESSURE_MODELS:
        row += f"\t{np.mean(all_corrs[model]):.4f}" if all_corrs[model] else "\t"
    print(row)
    
    # Table 4: Wind correlation by TC at WGL
    print("\n" + "=" * 100)
    print("Table 4: WGL_AWS - Wind Speed Correlation by TC (None suffix)")
    print("=" * 100)
    
    wind_cols = {
        'CIRC-EM': 'WS_circ', 'ELLIP-EM': 'WS_ellip', 'JTWC-EM': 'WS_r34',
        'CIRC-GB': 'WS_circ_P_only', 'ELLIP-GB': 'WS_ellip_P_only',
        'JTWC-GB': 'WS_r34_P_only', 'WRF': 'WS_wrf'
    }
    
    header = "TC"
    for model in WIND_MODELS:
        header += f"\t{model}"
    print(header)
    
    all_corrs = {model: [] for model in WIND_MODELS}
    station_data = get_station_data(df, 'WGL_AWS', 'None')
    
    for tc in TC_NAMES:
        tc_data = station_data[station_data['tc_name'] == tc]
        if len(tc_data) == 0:
            print(f"{tc}\t\t\t\t\t\t\t")
            continue
        
        metrics = compute_metrics_for_models(tc_data, 'WS_obs', wind_cols)
        row = tc
        for model in WIND_MODELS:
            corr = metrics.get(model, {}).get('corr', np.nan)
            if not np.isnan(corr):
                all_corrs[model].append(corr)
                row += f"\t{corr:.4f}"
            else:
                row += "\t"
        print(row)
    
    row = "Average"
    for model in WIND_MODELS:
        row += f"\t{np.mean(all_corrs[model]):.4f}" if all_corrs[model] else "\t"
    print(row)


def print_table_5_6(df):
    """Print Tables 5 and 6: Min pressure and max wind summary."""
    
    # Table 5: Min pressure
    print("\n" + "=" * 100)
    print("Table 5: Pressure (Min Values) - Correlation and Mean Absolute Error")
    print("(HKO_AWS, CCH_AWS, LFS_AWS)")
    print("=" * 100)
    
    pressure_cols = {'CIRC-EM': 'P_circ', 'ELLIP-EM': 'P_ellip', 'JTWC-EM': 'P_r34', 'WRF': 'P_wrf'}
    
    header = "Suffix"
    for model in PRESSURE_MODELS:
        header += f"\tP_{model}_corr"
    for model in PRESSURE_MODELS:
        header += f"\tP_{model}_mae"
    print(header)
    
    for suffix in SUFFIXES:
        all_corrs, all_maes = [], []
        for station in PRESSURE_STATIONS:
            station_data = get_station_data(df, station, suffix)
            if len(station_data) == 0:
                continue
            
            # Get min pressure per TC
            obs_mins = []
            model_mins = {model: [] for model in PRESSURE_MODELS}
            
            for tc in TC_NAMES:
                tc_data = station_data[station_data['tc_name'] == tc]
                if len(tc_data) == 0:
                    continue
                obs_mins.append(tc_data['P_obs'].min())
                for model in PRESSURE_MODELS:
                    col = pressure_cols[model]
                    if col in tc_data.columns:
                        model_mins[model].append(tc_data[col].min())
            
            if len(obs_mins) > 1:
                for model in PRESSURE_MODELS:
                    corr = compute_correlation(np.array(obs_mins), np.array(model_mins[model]))
                    mae = compute_mae(np.array(obs_mins), np.array(model_mins[model]))
                    all_corrs.append(corr)
                    all_maes.append(mae)
        
        row = suffix
        for i, model in enumerate(PRESSURE_MODELS):
            val = np.nanmean([all_corrs[j] for j in range(i, len(all_corrs), len(PRESSURE_MODELS))])
            row += f"\t{val:.4f}" if not np.isnan(val) else "\t"
        for i, model in enumerate(PRESSURE_MODELS):
            val = np.nanmean([all_maes[j] for j in range(i, len(all_maes), len(PRESSURE_MODELS))])
            row += f"\t{val:.4f}" if not np.isnan(val) else "\t"
        print(row)
    
    # Table 6: Max wind
    print("\n" + "=" * 100)
    print("Table 6: Wind Speed (Max Values) - Correlation and Mean Absolute Error")
    print("(WGL_AWS)")
    print("=" * 100)
    
    wind_cols = {
        'CIRC-EM': 'WS_circ', 'CIRC-GB': 'WS_circ_P_only',
        'ELLIP-EM': 'WS_ellip', 'ELLIP-GB': 'WS_ellip_P_only',
        'JTWC-EM': 'WS_r34', 'JTWC-GB': 'WS_r34_P_only',
        'WRF': 'WS_wrf'
    }
    
    header = "Metric_Suffix"
    for model in WIND_MODELS:
        header += f"\t{model}"
    print(header)
    
    for suffix in SUFFIXES:
        station_data = get_station_data(df, WIND_STATION, suffix)
        
        # Get max wind per TC
        obs_maxs = []
        model_maxs = {model: [] for model in WIND_MODELS}
        
        for tc in TC_NAMES:
            tc_data = station_data[station_data['tc_name'] == tc]
            if len(tc_data) == 0:
                continue
            obs_maxs.append(tc_data['WS_obs'].max())
            for model in WIND_MODELS:
                col = wind_cols[model]
                if col in tc_data.columns:
                    val = tc_data[col].max()
                    model_maxs[model].append(val if not np.isnan(val) else np.nan)
        
        row_corr = f"Correlation_{suffix}"
        row_mae = f"MAE_{suffix}"
        
        if len(obs_maxs) > 1:
            for model in WIND_MODELS:
                valid_mask = ~np.isnan(model_maxs[model])
                if np.sum(valid_mask) > 1:
                    obs_valid = np.array(obs_maxs)[valid_mask]
                    mod_valid = np.array(model_maxs[model])[valid_mask]
                    corr = compute_correlation(obs_valid, mod_valid)
                    mae = compute_mae(obs_valid, mod_valid)
                    row_corr += f"\t{corr:.4f}" if not np.isnan(corr) else "\t"
                    row_mae += f"\t{mae:.4f}" if not np.isnan(mae) else "\t"
                else:
                    row_corr += "\t"
                    row_mae += "\t"
        else:
            row_corr += "\t" * len(WIND_MODELS)
            row_mae += "\t" * len(WIND_MODELS)
        
        print(row_corr)
        print(row_mae)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate tables for TC parametric model paper')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Directory containing prediction CSV files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for tables (not yet implemented)')
    parser.add_argument('--tables', type=str, nargs='+',
                        choices=['1', '2', '3', '4', '5', '6'],
                        default=['1', '2', '3', '4', '5', '6'],
                        help='Which tables to generate')
    
    args = parser.parse_args()
    
    # Set input directory
    if args.input_dir:
        folder = Path(args.input_dir)
    else:
        folder = OUTPUT_DIR / "results_with_wrf"
    
    print(f"Loading data from {folder}...")
    df = load_predictions(folder)
    print(f"Loaded {len(df)} rows")
    
    # Generate requested tables
    if '1' in args.tables or '2' in args.tables:
        print_table_1_2(df)
    
    if '3' in args.tables or '4' in args.tables:
        print_table_3_4(df)
    
    if '5' in args.tables or '6' in args.tables:
        print_table_5_6(df)
    
    print("\nTable generation complete!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
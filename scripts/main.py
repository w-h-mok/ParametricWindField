# -*- coding: utf-8 -*-
"""
Parametric tropical cyclone wind and pressure field modeling.

Usage:
    python main.py --tc HATO --test-case a --window 0
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from pathlib import Path
from scipy.optimize import least_squares, minimize
from scipy.interpolate import CubicSpline
import argparse
import os

# Import configuration
try:
    from config import (
        DATA_ROOT, METEODATA_DIR, GUANGDONG_DIR, IBTRACS_PATH, OUTPUT_DIR,
        P_AMBI_DEFAULT, CIRC_BOUNDS, ELLIP_BOUNDS, R34_BOUNDS,
        MS_TO_KTS, R_EARTH_NM
    )
except ImportError:
    # Fallback defaults if config.py not found
    DATA_ROOT = Path.cwd()
    METEODATA_DIR = DATA_ROOT / "Meteodata"
    GUANGDONG_DIR = DATA_ROOT / "GuangDongSurfObs/GuangDongSurfObs"
    IBTRACS_PATH = DATA_ROOT / "ibtracs.WP.list.v04r01.csv"
    OUTPUT_DIR = DATA_ROOT / "results"
    P_AMBI_DEFAULT = 1010.0
    CIRC_BOUNDS = ([5, 0.5], [100, 2.5])
    ELLIP_BOUNDS = ([5, 0.5, 0, 0.5], [100, 1.0, 3.14159, 2.5])
    R34_BOUNDS = ([0.5, 5], [2.5, 100])
    MS_TO_KTS = 1.94384
    R_EARTH_NM = 3440.0


################################
####### LOADING DATASETS #######
################################

def load_envr_data(NAME, data_dir=None):
    """
    Load ENVR pressure data by searching for files containing NAME in the target directory.
    
    Parameters
    ----------
    NAME : str
        Tropical cyclone name (e.g., 'HATO')
    data_dir : Path, optional
        Directory containing Meteodata files. Defaults to METEODATA_DIR from config.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Time, ID, Lat, Lon, Pressure, WS, WD
    """
    if data_dir is None:
        data_dir = METEODATA_DIR
    
    base_dir = Path(data_dir)
    
    # Search for pressure files
    matching_files = list(base_dir.glob(f"A_PRE_SLP*{NAME}*.csv"))
    
    if not matching_files:
        raise FileNotFoundError(f"No ENVR pressure files found containing '{NAME}' in {base_dir}")
    elif len(matching_files) > 1:
        print(f"Warning: Multiple files found, using the first one: {matching_files[0]}")
    
    filepath = matching_files[0]
    
    df_p = pd.read_csv(filepath, skiprows=5)[3:]
    df_loc = pd.read_csv(filepath, skiprows=5)[:2].transpose()
    
    # Search for wind files
    matching_files = list(base_dir.glob(f"A_WIND*{NAME}*.csv"))
    
    if not matching_files:
        raise FileNotFoundError(f"No ENVR wind files found containing '{NAME}' in {base_dir}")
    elif len(matching_files) > 1:
        print(f"Warning: Multiple files found, using the first one: {matching_files[0]}")
    
    filepath = matching_files[0]

    # First count total number of rows
    with open(filepath, 'r') as f:
        NR = sum(1 for _ in f)
    
    # Calculate the split point
    split_row = (NR - 3) // 2
    
    # Read df_ws (first section)
    df_ws = pd.read_csv(
        filepath,
        skiprows=5,
        nrows=split_row - 7,
        header=0
    )[3:]
    
    # Read df_wd (second section)
    df_wd = pd.read_csv(
        filepath,
        skiprows=split_row + 5,
        nrows=NR - (split_row + 9),
        header=0
    )[3:]
    
    # Melt all three datasets
    melted_wd = pd.melt(
        df_wd.reset_index(),
        id_vars=['Station ID'],
        var_name='Station_Type',
        value_name='WD'
    )
    
    melted_ws = pd.melt(
        df_ws.reset_index(),
        id_vars=['Station ID'],
        var_name='Station_Type',
        value_name='WS'
    )
    
    melted_p = pd.melt(
        df_p.reset_index(),
        id_vars=['Station ID'],
        var_name='Station_Type',
        value_name='Pressure'
    )
    
    from functools import reduce
    df_long = reduce(
        lambda left, right: pd.merge(left, right, on=['Station ID', 'Station_Type'], how='outer'),
        [melted_wd, melted_ws, melted_p]
    )
    
    # Merge with locations
    df_loc = df_loc.reset_index()
    df_loc.columns = ['Station_Type', 'Lat', 'Lon']
    df_merged = pd.merge(df_long, df_loc, on='Station_Type')
    
    # Final formatting
    df_merged = df_merged.rename(columns={'Station ID': 'Time', 'Station_Type': 'ID'})
    df_merged['Time'] = pd.to_datetime(df_merged['Time']) 
    df_merged[['Lat', 'Lon', 'Pressure', 'WS', 'WD']] = df_merged[['Lat', 'Lon', 'Pressure', 'WS', 'WD']].astype(float)
    df_merged['Pressure'] /= 100  # Convert to hPa
    df_merged.WS *= MS_TO_KTS
    df_merged = df_merged[(df_merged.Pressure > 0) & (df_merged.WS >= 0) & (df_merged.WD >= 0)]
        
    return df_merged[['Time', 'ID', 'Lat', 'Lon', 'Pressure', 'WS', 'WD']]


def load_guangdong_data(date_range, base_dir=None):
    """
    Load Guangdong surface observations.
    
    Parameters
    ----------
    date_range : pd.DatetimeIndex
        Range of dates to load
    base_dir : Path, optional
        Base directory for Guangdong data. Defaults to GUANGDONG_DIR from config.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with observations
    """
    if base_dir is None:
        base_dir = GUANGDONG_DIR
    
    year = date_range[0].year
    base_path = Path(base_dir) / str(year) / str(year)

    all_dfs = []
    for time in date_range:
        date_str = time.strftime('%Y%m%d')
        hour_str = time.strftime('%H')
        file_path = base_path / date_str / f"SURF_HOR_{date_str}{hour_str}.txt"
        
        if not file_path.exists():
            continue
            
        try:
            df = pd.read_csv(file_path, delim_whitespace=True)
            df = df[['Station_Id_C', 'Lat', 'Lon', 'Alti', 'Year', 'Mon', 'Day', 'Hour', 
                      'PRS', 'PRS_Sea', 'WIN_S_Avg_2mi', 'WIN_D_Avg_2mi']]
            df['Time'] = pd.to_datetime(
                df['Year'].astype(str) + '/' + df['Mon'].astype(str) + '/' + 
                df['Day'].astype(str) + ' ' + df['Hour'].astype(str) + ':00:00'
            ) 
            
            # Pressure correction
            df[['PRS', 'PRS_Sea']] = df[['PRS', 'PRS_Sea']].replace(999999.0, np.nan)
            df['PRS_Sea_c'] = df['PRS'] + df['Alti'] * 0.1176
            df['PRS_Sea_c'] = np.nan
            df['Pressure'] = df['PRS_Sea'].combine_first(df['PRS_Sea_c'])
            df['WIN_S_Avg_2mi'] = df['WIN_S_Avg_2mi'] * MS_TO_KTS  # m/s to knots
            all_dfs.append(df[['Time', 'Station_Id_C', 'Lat', 'Lon', 'Pressure', 
                               'WIN_S_Avg_2mi', 'WIN_D_Avg_2mi']].rename(
                                   columns={'Station_Id_C': 'ID', 'WIN_S_Avg_2mi': 'WS', 'WIN_D_Avg_2mi': 'WD'}))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    try:
        return pd.concat(all_dfs, ignore_index=True)
    except:
        return pd.DataFrame()


def load_track_data(date_range, tc_name, ibtracs_path=None):
    """
    Load IBTrACS TC track data.
    
    Parameters
    ----------
    date_range : pd.DatetimeIndex
        Range of dates to filter
    tc_name : str
        Tropical cyclone name
    ibtracs_path : Path, optional
        Path to IBTrACS CSV file. Defaults to IBTRACS_PATH from config.
    
    Returns
    -------
    pd.DataFrame
        Track data resampled to hourly intervals
    """
    if ibtracs_path is None:
        ibtracs_path = IBTRACS_PATH
    
    df = pd.read_csv(ibtracs_path, skiprows=[1], na_values=['', 'NA', 'NaN', '-999.0'])
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df = df.replace(' ', np.nan)
    
    df_track = (
        df.loc[
            (df['ISO_TIME'] >= date_range[0]) & 
            (df['ISO_TIME'] <= date_range[-1]) &
            (df['NAME'] == tc_name),
            ['ISO_TIME', 'LAT', 'LON', 'USA_PRES', 'USA_WIND',
             "USA_R34_NE", "USA_R50_NE", "USA_R64_NE",
             "USA_R34_SE", "USA_R50_SE", "USA_R64_SE",
             "USA_R34_SW", "USA_R50_SW", "USA_R64_SW",
             "USA_R34_NW", "USA_R50_NW", "USA_R64_NW",
             ]
        ]
        .rename(columns={'ISO_TIME': 'Time', 'LAT': 'Lat', 'LON': 'Lon', 'USA_PRES': 'Pressure'})
        .astype({'Lat': float, 'Lon': float, 'Pressure': float})
    )
    df_track[df_track.columns[4:]] = df_track[df_track.columns[4:]].astype(float)
    df_track = df_track[df_track.USA_WIND >= 34]
    return df_track.set_index('Time').resample('H').interpolate(method='linear', limit_area='inside').reset_index()


################################
##### DATASET PREPARATION ######
################################

def prepare_obs_dataset(df_obs, df_track, target_time, time_window=0):
    """
    Extract observations within time window and compute distances/azimuths.
    
    Args:
        df_obs: Observations DataFrame
        df_track: Track DataFrame
        target_time: Central time point
        time_window: Hours before target_time to include (0 = only target_time)
        
    Returns:
        tuple: (combined_df, final_tc_center)
        - combined_df: Concatenated DataFrame of all time steps
        - final_tc_center: TC center at target_time
    """
    # Generate time points to process
    time_points = pd.date_range(end=target_time, periods=time_window+1, freq='H')
    
    all_dfs = []
    final_tc_center = None
    
    for time in time_points:
        # Get TC center for this time
        try:
            tc_center = df_track[df_track['Time'] == time].iloc[0]
        except:
            continue
        
        # Get observations for this time
        df_selected = df_obs[df_obs['Time'] == time].copy()
        
        # Round coordinates (0.2 degree resolution)
        df_selected['Lat_rounded'] = np.round(df_selected['Lat'] * 5) / 5
        df_selected['Lon_rounded'] = np.round(df_selected['Lon'] * 5) / 5
        
        # Compute distance and azimuth
        def _calc_dist_azimuth(row, center_lat, center_lon):
            dist = geodesic((row['Lat'], row['Lon']), (center_lat, center_lon)).nm
            dlon = np.radians(row['Lon'] - center_lon)
            y = np.sin(dlon) * np.cos(np.radians(row['Lat']))
            x = (np.cos(np.radians(center_lat)) * np.sin(np.radians(row['Lat'])) - 
                 np.sin(np.radians(center_lat)) * np.cos(np.radians(row['Lat'])) * np.cos(dlon))
            azimuth = (np.degrees(np.arctan2(y, x)) + 360) % 360
            return pd.Series([dist, azimuth], index=['distance', 'azimuth'])
        
        df_selected[['distance', 'azimuth']] = df_selected.apply(
            lambda row: _calc_dist_azimuth(row, tc_center['Lat'], tc_center['Lon']),
            axis=1
        )
        
        # Store results
        all_dfs.append(df_selected)
        final_tc_center = tc_center  # Will end with target_time's center
    
    # Combine all time steps
    combined_df = pd.concat(all_dfs, ignore_index=True).dropna().drop_duplicates(subset=['Time', 'Lat', 'Lon'], keep='first')    
    return combined_df, final_tc_center


################################
######## MODEL FUNCTIONS #######
################################

def ellip_rmax(params, azimuth):
    rmax, e, phi, B = params
    rmin = e * rmax
    r_max = (rmax - rmin)/2 * np.cos(azimuth + phi) + (rmin + rmax)/2
    return r_max


def circ_rmax(params, azimuth):
    rmax, B = params
    return azimuth * 0 + rmax


def R34_rmax(params, azimuth, R_quadrants):
    (b, r_max_max), a_NE = params, np.pi/4
    
    # Calculate Rmax for each quadrant
    R_values = np.where(~np.isnan(R_quadrants[:, 2]), R_quadrants[:, 2], 
            np.where(~np.isnan(R_quadrants[:, 1]), R_quadrants[:, 1], 
            R_quadrants[:, 0]))
    
    R_values = R_values / R_values.mean() * r_max_max
    
    # Create angles for each quadrant (0-360° range)
    angles = np.array([
        a_NE % (2 * np.pi),
        (a_NE + 0.5 * np.pi) % (2 * np.pi),
        (a_NE + 1.0 * np.pi) % (2 * np.pi),
        (a_NE + 1.5 * np.pi) % (2 * np.pi)
    ])
    
    # Sort angles and corresponding R_values to ensure increasing order
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    R_sorted = R_values[sort_idx]
    
    # Add periodic point (repeat first point at +360°)
    angles = np.append(angles, angles[0] + 2 * np.pi)
    R_sorted = np.append(R_sorted, R_sorted[0])

    # Create cubic spline interpolator
    cs = CubicSpline(angles, R_sorted, bc_type='periodic')
    
    # Normalize azimuths and interpolate r_max
    az_norm = azimuth % (2 * np.pi)
    r_max = cs(az_norm)

    return r_max


# Pressure models
def circular_pmodel(params, r, azimuth, p_ambi, p_min):
    r_max, B = params
    return p_min + (p_ambi - p_min) * np.exp(-(r_max / r)**B)


def elliptical_pmodel(params, r, azimuth, p_ambi, p_min):
    rmax, e, phi, B = params
    r_max = ellip_rmax([rmax, e, phi, B], azimuth)
    return p_min + (p_ambi - p_min) * np.exp(-(r_max / r)**B)


def R34_pmodel(params, r, azimuth, p_ambi, p_min, R_quadrants):
    (b, r_max_max), a_NE = params, np.pi/4
    
    # Normalize azimuths and interpolate r_max
    az_norm = azimuth % (2 * np.pi)
    r_max = R34_rmax(params, az_norm, R_quadrants)

    with np.errstate(divide='ignore', invalid='ignore'):
        pressure = p_min + (p_ambi - p_min) * np.exp(-(r_max / r) ** b)

    return np.where(np.isfinite(pressure), pressure, p_min)


# Wind speed models (EM method)
def find_optimal_c(rmax_func, params, df_track, target_time, R_quadrants, v_max):
    """
    Find optimal c parameter for EM wind speed model.
    """
    # Calculate Rmax for each quadrant at the 4 cardinal directions
    angles = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    Rmax_values = rmax_func(params, angles)
    
    R_q_with_rmax = np.column_stack((R_quadrants, Rmax_values)).flatten()
    wss = np.tile(np.array([34., 50., 64., v_max], dtype=np.float64), 4)
    Rmax_wswd = Rmax_values.repeat(4)
    mask = ~np.isnan(R_q_with_rmax)
    
    R_q_with_rmax = R_q_with_rmax[mask]
    wss = wss[mask]
    Rmax_wswd = Rmax_wswd[mask]
    
    def v_error(c):
        return np.sum(np.power(((Rmax_wswd / R_q_with_rmax) ** c) * v_max / wss - 1, 2))
    
    optimal_c = minimize(v_error, x0=1.0, bounds=[(0.4, 5.0)], method='L-BFGS-B').x[0]
    return optimal_c


def circular_vmodel(params, r, azimuth, R_quadrants, v_max):
    """EM wind speed for circular model."""
    r_max, B = params
    c = find_optimal_c(circ_rmax, params, None, None, R_quadrants, v_max)  # Simplified
    return np.where(r >= r_max, v_max * (r_max / r) ** c, v_max * (r / r_max) ** 0.5)


def elliptical_vmodel(params, r, azimuth, R_quadrants, v_max):
    """EM wind speed for elliptical model."""
    rmax, e, phi, B = params
    r_max = ellip_rmax([rmax, e, phi, B], azimuth)
    c = find_optimal_c(ellip_rmax, params, None, None, R_quadrants, v_max)
    return np.where(r >= r_max, v_max * (r_max / r) ** c, v_max * (r / r_max) ** 0.5)


def R34_vmodel(params, r, azimuth, R_quadrants, v_max):
    """EM wind speed for R34 model."""
    (b, r_max_max), a_NE = params, np.pi/4
    c = find_optimal_c(R34_rmax, params, None, None, R_quadrants, v_max)
    az_norm = azimuth % (2 * np.pi)
    r_max = R34_rmax(params, az_norm, R_quadrants)
    return np.where(r >= r_max, v_max * (r_max / r) ** c, v_max * (r / r_max) ** 0.5)


# Wind speed models (GB method)
def circular_vmodel_gb(params, r, azimuth, tc_center_local, p_ambi, p_min, v_max):
    r_max, B = params
    f = np.sin(np.pi * tc_center_local['Lat'] / 180) * 7.292e-5 * 2
    
    v_outside = np.sqrt((B / 1.2) * ((r_max / r) ** B)
            * (p_ambi - p_min) * np.exp(-(r_max / r) ** B) * 100
            + (r * 1852 * f / 2) ** 2) - r * 1852 * f / 2
    
    return np.where(r >= r_max, v_outside * MS_TO_KTS, v_max * (r / r_max) ** 0.5)


def elliptical_vmodel_gb(params, r, azimuth, tc_center_local, p_ambi, p_min, v_max):
    rmax, e, phi, B = params
    r_max = ellip_rmax([rmax, e, phi, B], azimuth)
    f = np.sin(np.pi * tc_center_local['Lat'] / 180) * 7.292e-5 * 2
    
    v_outside = np.sqrt((B / 1.2) * ((r_max / r) ** B)
            * (p_ambi - p_min) * np.exp(-(r_max / r) ** B) * 100
            + (r * 1852 * f / 2) ** 2) - r * 1852 * f / 2
    
    return np.where(r >= r_max, v_outside * MS_TO_KTS, v_max * (r / r_max) ** 0.5)


def R34_vmodel_gb(params, r, azimuth, tc_center_local, p_ambi, p_min, v_max, R_quadrants):
    (b, r_max_max), a_NE = params, np.pi/4
    f = np.sin(np.pi * tc_center_local['Lat'] / 180) * 7.292e-5 * 2
    
    r_max = R34_rmax(params, azimuth, R_quadrants)
    
    v_outside = (np.sqrt((b / 1.2) * ((r_max / r) ** b)
            * (p_ambi - p_min) * np.exp(-(r_max / r) ** b) * 100
            + (r * 1852 * f / 2) ** 2) - r * 1852 * f / 2)
    
    return np.where(r >= r_max, v_outside * MS_TO_KTS, v_max * (r / r_max) ** 0.5)


def find_clim_p_ambi(target_range):
    """Get climatological ambient pressure based on month."""
    month = target_range[0].month
    month_pressures = {
        4: 1013.0, 5: 1009.3, 6: 1006.1,
        7: 1005.6, 8: 1005.2, 9: 1008.8,
        10: 1014.0, 11: 1017.3
    }
    return month_pressures.get(month, 1014.0)


def fit_model(model_func, initial_params, bounds, r, azimuth_rad, observed_p, **kwargs):
    """Generic curve fitting for any pressure model."""
    def residuals(params, r, azimuth, observed):
        return model_func(params, r, azimuth, **kwargs) - observed
    
    result = least_squares(
        residuals,
        initial_params,
        args=(r, azimuth_rad, observed_p),
        bounds=bounds,
    )
    return result.x


def generate_predictions(df, df_params, df_track, p_ambi_func):
    """
    Generate model predictions for given observations, matching parameters by time.
    """
    # Initialize output DataFrame with observations
    predictions = df[['Time', 'ID', 'Lat', 'Lon']].copy()
    if 'WS' in df:
        predictions['WS_obs'] = df['WS']
    if 'Pressure' in df:
        predictions['P_obs'] = df['Pressure']
    
    # Initialize prediction columns
    for prefix in ['WS', 'P']:
        for model in ['circ', 'ellip', 'r34']:
            predictions[f'{prefix}_{model}'] = np.nan
            if prefix == 'WS':
                predictions[f'{prefix}_{model}_P_only'] = np.nan
    
    # Process each unique time in the input data
    for target_time in df['Time'].unique():
        # Get parameters for this time
        time_params = df_params[df_params['Time'] == target_time]
        if time_params.empty:
            continue
            
        tc_center = df_track[df_track['Time'] == target_time].iloc[0]
        p_min = tc_center['Pressure']
        v_max = tc_center['USA_WIND']
        
        # Get R_quadrants for this time
        R_quadrants = df_track[df_track['Time'] == target_time][[
            "USA_R34_NE", "USA_R50_NE", "USA_R64_NE",
            "USA_R34_SE", "USA_R50_SE", "USA_R64_SE",
            "USA_R34_SW", "USA_R50_SW", "USA_R64_SW",
            "USA_R34_NW", "USA_R50_NW", "USA_R64_NW"
        ]].values.reshape([4, 3])
        
        params = time_params.iloc[0]
        time_mask = df['Time'] == target_time
        
        # Get coordinates for stations at this time
        r = df.loc[time_mask, 'distance'].values
        azimuth_rad = np.radians(df.loc[time_mask, 'azimuth'].values)
        
        p_ambi = p_ambi_func(target_time) if callable(p_ambi_func) else p_ambi_func
        
        # Circular Model predictions
        circ_params = [params['Circular_rmax'], params['Circular_b']]
        predictions.loc[time_mask, 'P_circ'] = circular_pmodel(
            circ_params, r, azimuth_rad, p_ambi, p_min)
        
        # Elliptical Model predictions
        ellip_params = [params['Elliptical_rmax'], params['Elliptical_e'], 
                        params['Elliptical_phi'], params['Elliptical_b']]
        predictions.loc[time_mask, 'P_ellip'] = elliptical_pmodel(
            ellip_params, r, azimuth_rad, p_ambi, p_min)
        
        # R34 Model predictions
        r34_params = [params['R34_b'], params['R34_rmax_max']]
        predictions.loc[time_mask, 'P_r34'] = R34_pmodel(
            r34_params, r, azimuth_rad, p_ambi, p_min, R_quadrants)
    
    return predictions


################################
######## MAIN FUNCTION #########
################################

def run_analysis(tc_name, window, test_case, output_dir=None, data_dir=None, gd_dir=None, ibtracs_path=None):
    """
    Run analysis for a specific TC and test case.
    
    Parameters
    ----------
    tc_name : str
        Tropical cyclone name
    window : int
        Time window in hours
    test_case : str
        'a'=control, 'b'=fixed P_ambi, 'c'=ENVR only, 'd'=limited observations
    output_dir : Path, optional
        Directory to save results
    data_dir : Path, optional
        Directory for ENVR data
    gd_dir : Path, optional
        Directory for Guangdong data
    ibtracs_path : Path, optional
        Path to IBTrACS CSV file
    
    Returns
    -------
    tuple
        (predictions, params, rmse_results)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if data_dir is None:
        data_dir = METEODATA_DIR
    if gd_dir is None:
        gd_dir = GUANGDONG_DIR
    if ibtracs_path is None:
        ibtracs_path = IBTRACS_PATH
    
    print(f"Processing {tc_name}, test case {test_case}...")
    
    # 1. Load data
    df_envr = load_envr_data(tc_name, data_dir=data_dir)
    date_range = pd.date_range(df_envr.iloc[0].Time, df_envr.iloc[-1].Time, freq='H')
    df_track = load_track_data(date_range, tc_name, ibtracs_path=ibtracs_path)
    df_track = df_track[~df_track.USA_R34_NW.isna()]
    date_range = pd.date_range(df_track.iloc[0].Time, df_track.iloc[-1].Time, freq='H')
    
    if test_case != 'c':  # For all except 'c' (no Guangdong data)
        df_gd = load_guangdong_data(date_range, base_dir=gd_dir)
    else:
        df_gd = pd.DataFrame()
    
    # Set ambient pressure based on test case
    target_range = date_range
    if test_case == 'b':
        p_ambi = P_AMBI_DEFAULT
    else:
        p_ambi = find_clim_p_ambi(target_range)
    
    # Prepare multi_time_data based on test case
    if test_case == 'd':
        # For test case d: Use restricted IDs for parameter fitting only
        full_data = pd.concat([df_envr, df_gd]) if not df_gd.empty else df_envr.copy()
        
        # Create restricted dataset for fitting
        restricted_ids = ["TKL_AWS", "EPC_AWS", "HKA_AWS", "SHA_AWS", "CCH_AWS", "KP_AWS", "MC_TG"]
        restricted_envr = df_envr[df_envr['ID'].isin(restricted_ids)]
        
        if not df_gd.empty:
            restricted_gd = df_gd[df_gd['ID'].isin(restricted_ids)]
            df_obs_for_fitting = pd.concat([restricted_envr, restricted_gd])
        else:
            df_obs_for_fitting = restricted_envr
        
        # For multi_time_data (used in predictions), use the full dataset
        multi_time_data = pd.concat([
            prepare_obs_dataset(full_data, df_track, t, time_window=0)[0]
            for t in date_range
        ])
    else:
        # For test cases a, b, c
        if test_case == 'c':
            df_obs_for_fitting = df_envr
        else:
            df_obs_for_fitting = pd.concat([df_envr, df_gd]) if not df_gd.empty else df_envr
        
        multi_time_data = pd.concat([
            prepare_obs_dataset(df_obs_for_fitting, df_track, t, time_window=0)[0]
            for t in date_range
        ])
    
    # Store parameters
    parameters = {
        'Time': [],
        'Circular_b': [], 'Circular_rmax': [], 'Circular_c': [],
        'Elliptical_b': [], 'Elliptical_rmax': [], 'Elliptical_e': [], 
        'Elliptical_phi': [], 'Elliptical_c': [],
        'R34_b': [], 'R34_c': [], 'R34_rmax_max': [], 
        'R34_rmax_NE': [], 'R34_rmax_SE': [], 'R34_rmax_SW': [], 'R34_rmax_NW': []
    }
    
    station_list = ["CCH_AWS", "HKO_AWS", "LFS_AWS"]
    
    for target_time in date_range:
        print(f"  Processing time {target_time}...")
        
        # For test case d, use the restricted dataset for fitting
        if test_case == 'd':
            df_temp_for_fitting = df_obs_for_fitting
        else:
            df_temp_for_fitting = df_obs_for_fitting.copy()
        
        # Prepare dataset for fitting
        df_selected, tc_center = prepare_obs_dataset(
            df_temp_for_fitting[~df_temp_for_fitting.ID.isin(station_list)], 
            df_track, 
            target_time, 
            time_window=window
        )
        df_selected = df_selected[~df_selected.Pressure.isna()]
        
        # Common variables
        p_min = tc_center['Pressure']
        v_max = tc_center['USA_WIND']
        
        # Get R_quadrants
        df_track_sel = df_track[df_track['Time'] == target_time]
        R_quadrants = df_track_sel[[
            "USA_R34_NE", "USA_R50_NE", "USA_R64_NE",
            "USA_R34_SE", "USA_R50_SE", "USA_R64_SE",
            "USA_R34_SW", "USA_R50_SW", "USA_R64_SW",
            "USA_R34_NW", "USA_R50_NW", "USA_R64_NW"
        ]].values.reshape([4, 3])
        
        # Get bound limits from R_quadrants
        rm_bound_min = 5.0  # Default minimum
        R_non_nan = R_quadrants[:, ~np.all(np.isnan(R_quadrants), axis=0)]
        if R_non_nan.size > 0:
            rm_bound_max = np.nanmax(R_non_nan[:, -1]) * 1.5
        else:
            rm_bound_max = 100.0
        
        r = df_selected['distance'].values
        azimuth_rad = np.radians(df_selected['azimuth'].values)
        observed_p = df_selected['Pressure'].values
        
        # --- Model Processing ---
        # 1. Circular Model
        circ_params = fit_model(
            circular_pmodel,
            initial_params=[rm_bound_min, 1.5],
            bounds=([rm_bound_min, 0.5], [rm_bound_max, 2.5]),
            r=r, azimuth_rad=azimuth_rad, 
            observed_p=observed_p,
            p_ambi=p_ambi, p_min=p_min
        )
        
        parameters['Time'].append(target_time)
        parameters['Circular_b'].append(circ_params[1])
        parameters['Circular_rmax'].append(circ_params[0])
        parameters['Circular_c'].append(0.5)  # Placeholder
        
        # 2. Elliptical Model
        ellip_params = fit_model(
            elliptical_pmodel,
            initial_params=[rm_bound_min, 0.99, 0.0, 1.5],
            bounds=([rm_bound_min, 0.5, 0, 0.5], [rm_bound_max, 1.0, np.pi, 2.5]),
            r=r, azimuth_rad=azimuth_rad, 
            observed_p=observed_p,
            p_ambi=p_ambi, p_min=p_min
        )

        parameters['Elliptical_rmax'].append(ellip_params[0])
        parameters['Elliptical_e'].append(ellip_params[1])
        parameters['Elliptical_phi'].append(ellip_params[2])
        parameters['Elliptical_b'].append(ellip_params[3])
        parameters['Elliptical_c'].append(0.5)  # Placeholder
        
        # 3. R34 Model
        r34_params = fit_model(
            R34_pmodel,
            initial_params=[1, rm_bound_min],
            bounds=([0.5, rm_bound_min], [2.5, rm_bound_max]),
            r=r, azimuth_rad=azimuth_rad, 
            observed_p=observed_p,
            p_ambi=p_ambi, p_min=p_min, R_quadrants=R_quadrants
        )
        
        b, r_max_max = r34_params
        
        # Calculate Rmax for each quadrant
        Rmax_values = np.where(~np.isnan(R_quadrants[:, 2]), R_quadrants[:, 2], 
                np.where(~np.isnan(R_quadrants[:, 1]), R_quadrants[:, 1], 
                R_quadrants[:, 0]))
        
        Rmax_values = Rmax_values / Rmax_values.mean() * r_max_max
        
        parameters['R34_b'].append(b)
        parameters['R34_c'].append(0.5)  # Placeholder
        parameters['R34_rmax_max'].append(r_max_max)
        parameters['R34_rmax_NE'].append(Rmax_values[0])
        parameters['R34_rmax_SE'].append(Rmax_values[1])
        parameters['R34_rmax_SW'].append(Rmax_values[2])
        parameters['R34_rmax_NW'].append(Rmax_values[3])
    
    df_params = pd.DataFrame(parameters)
    
    # Generate predictions
    if test_case == 'd':
        full_data = pd.concat([df_envr, df_gd]) if not df_gd.empty else df_envr.copy()
        multi_time_data_for_predictions = pd.concat([
            prepare_obs_dataset(full_data, df_track, t, time_window=0)[0]
            for t in date_range
        ])
        df_predictions = generate_predictions(multi_time_data_for_predictions, df_params, df_track, 
                                              lambda t: p_ambi)
    else:
        df_predictions = generate_predictions(multi_time_data, df_params, df_track, 
                                              lambda t: p_ambi)
    
    # Save results with appropriate suffix
    suffix = ''
    if test_case == 'b':
        suffix = '_1010'
    elif test_case == 'c':
        suffix = '_nogd'
    elif test_case == 'd':
        suffix = '_restricted'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    df_predictions.to_csv(os.path.join(output_dir, f"predictions_{tc_name}_w{window}{suffix}.csv"), index=False)
    df_params.to_csv(os.path.join(output_dir, f"params_{tc_name}_w{window}{suffix}.csv"), index=False)
    df_track.to_csv(os.path.join(output_dir, f"tracks_{tc_name}{suffix}.csv"), index=False)
    
    print(f"  Completed {tc_name}, test case {test_case}")
    print(f"  Saved to: {output_dir}")
    
    return df_predictions, df_params, None


##############################
######## MAIN EXECUTION #######
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TC parametric model analysis')
    parser.add_argument('--tc', type=str, required=True, help='TC name (e.g., HATO)')
    parser.add_argument('--test-case', type=str, choices=['a', 'b', 'c', 'd'], 
                        default='a', help='Test case: a=control, b=fixed P_ambi, c=ENVR only, d=limited obs')
    parser.add_argument('--window', type=int, default=0, help='Time window in hours')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR
    
    results = run_analysis(
        tc_name=args.tc,
        window=args.window,
        test_case=args.test_case,
        output_dir=output_dir
    )
    
    print(f"\nAnalysis complete for {args.tc}, test case {args.test_case}")
    print(f"Results saved to: {output_dir}")

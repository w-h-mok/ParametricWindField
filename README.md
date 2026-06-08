

# TC-Parametric-Model

Code for real-time parametric modelling of tropical cyclone wind and pressure fields.

## Overview

This repository contains the code used in the paper "Real-Time Parametric Modelling of Tropical Cyclones using Local Measurements from Ground-Level Stations and Real-Time Tropical Cyclone Reports" by Mok et al. (2026). The code implements:

- CIRC (circular), ELLIP (elliptical), and JTWC (R34-based) pressure models
- Empirical (EM) and gradient balance (GB) wind speed derivations
- Model fitting using trust-region reflective algorithm
- Four sensitivity tests (control, fixed P_ambi, ENVR-only, limited observations)

## Scripts Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `main.py` | Run parametric TC models, fit parameters, generate predictions for all sensitivity tests | ENVR/GD data, JTWC track data | `predictions_*.csv`, `params_*.csv`, `tracks_*.csv` |
| `table_generator.py` | Generate all result tables for the paper | `predictions_*.csv` files | Console output (tables) |
| `pressure_sensitivity_timeseries_comparison.py` | Plot pressure time series (control vs P_ambi=1010) | `predictions_*.csv` files | PNG figure |
| `TC_wind_SLP_timeseries_comparison.py` | Plot wind/SLP time series comparison | `predictions_*.csv` files | PNG figure |

## Data Flow

```
                    +-------------------------+
                    |     ENVR Data           |
                    |     Guangdong Data      |
                    |     IBTrACS Data        |
                    +-------------------------+
                                 |
                                 v
                    +-------------------------+
                    |        main.py          |
                    +-------------------------+
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
    +-----------------+  +-----------------+  +-----------------+
    | predictions_*.csv|  | params_*.csv    |  | tracks_*.csv    |
    +-----------------+  +-----------------+  +-----------------+
              |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
    +-----------------+  +-----------------+  +-----------------+
    | table_generator |  | pressure_sens   |  | TC_wind_SLP_    |
    | .py             |  | itivity_times   |  | timeseries_com  |
    | (console tables)|  | eries_compar    |  | parison.py      |
    +-----------------+  | ison.py (PNG)   |  | (PNG)           |
                         +-----------------+  +-----------------+
```

## Installation

### Using conda
```bash
conda env create -f environment.yml
conda activate tc_model
```

### Using pip
```bash
pip install -r requirements.txt
```

## Configuration

1. Edit `config.py` to set `DATA_ROOT` to your data directory
2. Ensure all data files are in the expected subdirectories

## Data Requirements

This code requires the following data (not included in this repository):

1. **ENVR dataset**: Provided by the Institute for the Environment, HKUST
2. **Guangdong dataset**: Proprietary; not included
3. **JTWC track data**: Downloaded from IBTrACS

## Test Cases in `main.py`

The `--test-case` argument controls which sensitivity test is run:

| Test Case | Name | Description | Suffix | Used in Paper |
|-----------|------|-------------|--------|---------------|
| `a` | Control | Full dataset (ENVR + Guangdong data) with climatological ambient pressure | None (no suffix) | Tables 1-8, 11-12 |
| `b` | Fixed P_ambi | Same as control but with fixed ambient pressure (1010 hPa) | `_1010` | Tables 1-2, 5-6 |
| `c` | ENVR-only | Only ENVR data (no Guangdong dataset) | `_nogd` | Tables 1-2, 5-6 |
| `d` | Limited observations | Only 7 selected stations (TKL, EPC, HKA, SHA, CCH, KP, Macau) | `_restricted` | Tables 1-2, 5-6, 12-13 |

### When to Use Each Test Case

| Test Case | Purpose |
|-----------|---------|
| `a` | Baseline performance evaluation (main results in paper) |
| `b` | Demonstrate importance of ambient pressure selection |
| `c` | Assess impact of removing Guangdong data |
| `d` | Simulate limited observation networks (real-world application) |

## Suffix Convention

All prediction files follow the naming pattern: `predictions_{TC}_{suffix}_w{window}.csv`

Where `{suffix}` can be:

| Suffix | Corresponding Test Case | Description |
|--------|------------------------|-------------|
| (no suffix) | `a` | Control (full data, climatological P_ambi) |
| `_1010` | `b` | Fixed ambient pressure (1010 hPa) |
| `_nogd` | `c` | ENVR only (no Guangdong data) |
| `_restricted` | `d` | Limited observations (7 stations only) |

## Execution Order

**Recommended order:**

1. First, run `main.py` to generate prediction files:
   ```bash
   python scripts/main.py --tc HATO --test-case a --window 0
   ```

2. Then, run `table_generator.py` to generate result tables:
   ```bash
   python scripts/table_generator.py
   ```

3. Finally, run plotting scripts to generate figures:
   ```bash
   python scripts/pressure_sensitivity_timeseries_comparison.py --station HKO_AWS
   python scripts/TC_wind_SLP_timeseries_comparison.py --station HKO_AWS
   ```

## Usage

### Run the full analysis for a TC
```bash
python scripts/main.py --tc HATO --test-case a --window 0
```

### Generate result tables
```bash
python scripts/table_generator.py
```

### Generate plots
```bash
python scripts/pressure_sensitivity_timeseries_comparison.py --station HKO_AWS
python scripts/TC_wind_SLP_timeseries_comparison.py --station HKO_AWS
```

## Output Files

- `predictions_{TC}_w{window}{suffix}.csv`: Model predictions at AWS locations
- `params_{TC}_w{window}{suffix}.csv`: Fitted model parameters
- `tracks_{TC}{suffix}.csv`: TC track data

## License

MIT License

## Citation

If you use this code, please cite:

Mok, W. H., et al. (2026). *TC-Parametric-Model* (Version 1.0.0) [Software]. Zenodo. [placeholder: https://doi.org/10.5281/zenodo.XXXXXX]

## Contact

Wan Hin MOK - whmok@ust.hk
```



"""Configuration template for TC parametric model.

Instructions:
1. Copy this file to config.py
2. Edit DATA_ROOT to point to your data directory
3. Adjust subdirectory paths if needed
"""

from pathlib import Path

# ============================================================
# PATHS - EDIT THIS SECTION
# ============================================================
# Set this to the root directory containing your data
DATA_ROOT = Path("/path/to/your/data/")  # <-- CHANGE THIS

# ============================================================
# SUBDIRECTORIES (adjust if your structure differs)
# ============================================================
PROJECT_DIR = DATA_ROOT / "MPhil Project/Data/TC parametric model"
METEODATA_DIR = PROJECT_DIR / "Meteodata"
GUANGDONG_DIR = DATA_ROOT / "MPhil Project/Data/GuangDongSurfObs/GuangDongSurfObs"
IBTRACS_PATH = DATA_ROOT / "MPhil Project/Data/TC analysis/ibtracs.WP.list.v04r01.csv"
OUTPUT_DIR = PROJECT_DIR / "Results/20260402"

# ============================================================
# MODEL PARAMETERS AND BOUNDS
# ============================================================
P_AMBI_DEFAULT = 1010.0
CIRC_BOUNDS = ([5, 0.5], [100, 2.5])
ELLIP_BOUNDS = ([5, 0.5, 0, 0.5], [100, 1.0, 3.14159, 2.5])
R34_BOUNDS = ([0.5, 5], [2.5, 100])

# ============================================================
# CONSTANTS
# ============================================================
MS_TO_KTS = 1.94384
R_EARTH_NM = 3440.0
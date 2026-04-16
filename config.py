"""
Centralized configuration for the Arctic Sea Ice Analysis project.
All paths, URLs, and analysis parameters are defined here.
"""

from pathlib import Path

# ── Directory structure ─────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── NSIDC Sea Ice Index v4 (G02135) ────────────────────────────────────────
NSIDC_BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/G02135"
NSIDC_MONTHLY_DATA_URL = f"{NSIDC_BASE_URL}/north/monthly/data"
NSIDC_DAILY_DATA_URL = f"{NSIDC_BASE_URL}/north/daily/data"
NSIDC_DAILY_FILE = "N_seaice_extent_daily_v4.0.csv"
NSIDC_CLIMATOLOGY_FILE = "N_seaice_extent_climatology_1981-2010_v4.0.csv"

# ── NASA GISTEMP v4 ────────────────────────────────────────────────────────
GISTEMP_URL = (
    "https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv"
)
GISTEMP_FILE = "ZonAnn.Ts+dSST.csv"

# ── PolarWatch ERDDAP (gridded sea ice concentration) ──────────────────────
POLARWATCH_ERDDAP = "https://polarwatch.noaa.gov/erddap/griddap"
POLARWATCH_DATASET_NH = "nsidcG10016v2nhmday"

# ── Analysis parameters ─────────────────────────────────────────────────────
SATELLITE_ERA_START = 1979
BLUE_OCEAN_THRESHOLD_KM2 = 1.0       # million km²
BASELINE_PERIOD = (1981, 2010)
TEMP_BASELINE_PERIOD = (1991, 2020)   # paper uses 1991–2020 for temperature
SEPTEMBER = 9

# GISTEMP columns used as proxies
GISTEMP_ARCTIC_COL = "64N-90N"        # proxy for Arctic 60–90°N
GISTEMP_GLOBAL_COL = "Glob"

# ── Plot styling ────────────────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

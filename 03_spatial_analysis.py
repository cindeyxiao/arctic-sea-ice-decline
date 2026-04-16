"""
Step 3 — Spatial Analysis of Sea Ice Concentration
====================================================
Produces polar-projection maps and heatmaps of ice concentration.

Figures generated (when gridded data is available):
  • fig06_concentration_map.png       — Polar map of September concentration
  • fig07_concentration_anomaly.png   — Concentration anomaly map
  • fig08_regional_trend_map.png      — Pixel-wise linear trend heatmap

Data requirements
-----------------
This script works with monthly sea ice concentration NetCDF files.
It will attempt to load data from two sources in order:

  1. PolarWatch ERDDAP NetCDF files (downloaded by 01_download_data.py)
  2. NSIDC CDR (G02202) NetCDF files (manual download — see README)

If no gridded data is found, the script generates a demonstration using
the NSIDC monthly extent CSVs (regional bar chart + polar schematic).
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from config import (
    BASELINE_PERIOD,
    DATA_DIR,
    FIGURES_DIR,
    PLOT_STYLE,
    SATELLITE_ERA_START,
)

plt.rcParams.update(PLOT_STYLE)

HAS_CARTOPY = True
HAS_XARRAY = True

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    HAS_CARTOPY = False
    print("  Cartopy not installed — polar maps will be skipped.")
    print("  Install with: pip install cartopy")

try:
    import xarray as xr
except ImportError:
    HAS_XARRAY = False
    print("  xarray not installed — gridded analysis will be skipped.")
    print("  Install with: pip install xarray netCDF4")


# ── Data loading helpers ────────────────────────────────────────────────────

def find_netcdf_files() -> list[Path]:
    """Locate any sea ice concentration NetCDF files in the data directory."""
    patterns = ["polarwatch_sic_*.nc", "*.nc"]
    found = []
    for pat in patterns:
        found.extend(DATA_DIR.glob(pat))
    return sorted(set(found))


def load_concentration_grid(nc_path: Path) -> tuple:
    """
    Load a single NetCDF file and return (lat, lon, concentration_2d).
    Handles variable-name differences between PolarWatch and NSIDC CDR.
    """
    ds = xr.open_dataset(nc_path)

    # Find the concentration variable
    conc_var = None
    for candidate in [
        "cdr_seaice_conc_monthly",
        "seaice_conc_monthly_cdr",
        "goddard_merged_seaice_conc_monthly",
        "sea_ice_concentration",
        "conc",
        "sic",
    ]:
        if candidate in ds.data_vars:
            conc_var = candidate
            break

    if conc_var is None:
        available = list(ds.data_vars)
        ds.close()
        raise KeyError(
            f"Cannot identify concentration variable. Available: {available}"
        )

    conc = ds[conc_var]

    # Squeeze singleton dimensions (time, etc.)
    while conc.ndim > 2:
        conc = conc.isel({conc.dims[0]: 0})

    conc_vals = conc.values.astype(float)

    # NSIDC CDR uses 0–1 scale; PolarWatch may use 0–100 or 0–1
    if np.nanmax(conc_vals) > 1.5:
        conc_vals /= 100.0

    # Get lat/lon (may be 1D or 2D)
    lat = lon = None
    for lat_name in ["latitude", "lat", "ygrid"]:
        if lat_name in ds.coords or lat_name in ds.data_vars:
            lat = ds[lat_name].values
            break
    for lon_name in ["longitude", "lon", "xgrid"]:
        if lon_name in ds.coords or lon_name in ds.data_vars:
            lon = ds[lon_name].values
            break

    ds.close()

    return lat, lon, conc_vals


# ── Polar projection map of concentration ───────────────────────────────────

def plot_concentration_polar(lat, lon, conc, title_suffix=""):
    """
    Plot sea ice concentration on a North Polar Stereographic map.
    lat/lon may be 1-D (regular grid) or 2-D (curvilinear).
    """
    if not HAS_CARTOPY:
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="black",
                   linewidth=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)

    # Mask out fill/missing values
    conc_masked = np.where((conc < 0) | (conc > 1), np.nan, conc)

    cmap = plt.cm.Blues_r
    cmap.set_bad(color="white")

    if lat is not None and lon is not None:
        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lon2d, lat2d = lon, lat

        im = ax.pcolormesh(
            lon2d, lat2d, conc_masked,
            cmap=cmap, vmin=0, vmax=1,
            transform=ccrs.PlateCarree(), shading="auto"
        )
    else:
        im = ax.imshow(
            conc_masked, cmap=cmap, vmin=0, vmax=1,
            extent=[-180, 180, 50, 90],
            transform=ccrs.PlateCarree(), origin="lower"
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("Sea Ice Concentration (fraction)")
    ax.set_title(f"Arctic Sea Ice Concentration{title_suffix}", fontsize=13)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_concentration_map.png")
    plt.close(fig)
    print("  Fig 06 saved.")


# ── Anomaly map ─────────────────────────────────────────────────────────────

def plot_anomaly_map(lat, lon, conc_recent, conc_baseline):
    """Plot a concentration anomaly map (recent minus baseline)."""
    if not HAS_CARTOPY:
        return

    anomaly = conc_recent - conc_baseline

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="black",
                   linewidth=0.4)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5)

    anomaly_masked = np.where(
        np.isnan(conc_recent) | np.isnan(conc_baseline), np.nan, anomaly
    )
    vmax = np.nanmax(np.abs(anomaly_masked))
    if vmax == 0 or np.isnan(vmax):
        vmax = 0.5

    cmap = plt.cm.RdBu
    cmap.set_bad(color="white")

    if lat is not None and lon is not None:
        if lat.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lon2d, lat2d = lon, lat

        im = ax.pcolormesh(
            lon2d, lat2d, anomaly_masked,
            cmap=cmap, vmin=-vmax, vmax=vmax,
            transform=ccrs.PlateCarree(), shading="auto"
        )
    else:
        im = ax.imshow(
            anomaly_masked, cmap=cmap, vmin=-vmax, vmax=vmax,
            extent=[-180, 180, 50, 90],
            transform=ccrs.PlateCarree(), origin="lower"
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("Concentration Anomaly (fraction)")
    ax.set_title("Sea Ice Concentration Anomaly (Recent − Baseline)", fontsize=13)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig07_concentration_anomaly.png")
    plt.close(fig)
    print("  Fig 07 saved.")


# ── Trend heatmap (pixel-wise linear trend) ─────────────────────────────────

def plot_trend_heatmap(nc_files: list[Path]):
    """
    If multiple years of gridded data are available, compute per-pixel
    linear trends and plot as a polar heatmap.
    """
    if not HAS_CARTOPY or not HAS_XARRAY or len(nc_files) < 3:
        print("  Fig 08 skipped (need ≥3 years of gridded data for trends).")
        return

    grids = []
    lat = lon = None
    for f in sorted(nc_files):
        try:
            lt, ln, c = load_concentration_grid(f)
            if lat is None:
                lat, lon = lt, ln
            grids.append(c)
        except Exception as exc:
            warnings.warn(f"Skipping {f.name}: {exc}")

    if len(grids) < 3:
        print("  Fig 08 skipped (insufficient gridded data).")
        return

    stack = np.array(grids)  # (n_years, ny, nx)
    n = stack.shape[0]
    x = np.arange(n)

    # Vectorised per-pixel linear regression (slope only)
    x_mean = x.mean()
    s_mean = np.nanmean(stack, axis=0)
    numer = np.nansum((x[:, None, None] - x_mean) * (stack - s_mean), axis=0)
    denom = np.sum((x - x_mean) ** 2)
    slope = numer / denom  # concentration change per year-index

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#d9d9d9", edgecolor="black",
                   linewidth=0.4)

    vmax = np.nanpercentile(np.abs(slope), 95)
    if vmax == 0 or np.isnan(vmax):
        vmax = 0.05

    cmap = plt.cm.RdBu
    cmap.set_bad("white")

    if lat is not None and lon is not None:
        if lat.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        else:
            lon2d, lat2d = lon, lat
        im = ax.pcolormesh(
            lon2d, lat2d, slope, cmap=cmap, vmin=-vmax, vmax=vmax,
            transform=ccrs.PlateCarree(), shading="auto"
        )
    else:
        im = ax.imshow(
            slope, cmap=cmap, vmin=-vmax, vmax=vmax,
            extent=[-180, 180, 50, 90],
            transform=ccrs.PlateCarree(), origin="lower"
        )

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label("Concentration Trend (fraction yr⁻¹)")
    ax.set_title("Per-Pixel Linear Trend in September Sea Ice Concentration",
                 fontsize=12)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig08_regional_trend_map.png")
    plt.close(fig)
    print("  Fig 08 saved.")


# ── Fallback: extent-based regional bar chart ───────────────────────────────

def plot_extent_decline_bar():
    """
    When gridded data is unavailable, visualise the September extent
    decline using the CSV time-series as a colour-coded bar chart.
    """
    path = DATA_DIR / "N_09_extent_v4.0.csv"
    if not path.exists():
        print("  Fallback bar chart skipped (no September CSV).")
        return

    df = pd.read_csv(path, skipinitialspace=True, comment="#")
    df.columns = df.columns.str.strip().str.lower()
    rename = {}
    for col in df.columns:
        if "year" in col:
            rename[col] = "year"
        elif "extent" in col:
            rename[col] = "extent"
    df.rename(columns=rename, inplace=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["extent"] = pd.to_numeric(df["extent"], errors="coerce")
    df.dropna(subset=["year", "extent"], inplace=True)
    df = df[df["year"] >= SATELLITE_ERA_START].sort_values("year")

    baseline_mean = df[
        df["year"].between(*BASELINE_PERIOD)
    ]["extent"].mean()

    colors = plt.cm.RdBu(
        (df["extent"] - df["extent"].min())
        / (df["extent"].max() - df["extent"].min())
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df["year"], df["extent"], color=colors, width=0.8)
    ax.axhline(baseline_mean, color="black", linestyle="--", linewidth=1,
               label=f"Baseline mean ({BASELINE_PERIOD[0]}–{BASELINE_PERIOD[1]})")
    ax.axhline(1.0, color="red", linestyle=":", linewidth=1.2,
               label='"Blue Ocean" threshold (1 M km²)')
    ax.set_xlabel("Year")
    ax.set_ylabel("September Extent (million km²)")
    ax.set_title("September Arctic Sea Ice Extent — Year-by-Year Decline")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_september_extent_bars.png")
    plt.close(fig)
    print("  Fig 06 (fallback bar chart) saved.")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ARCTIC SEA ICE — SPATIAL ANALYSIS")
    print("=" * 64)

    nc_files = find_netcdf_files() if HAS_XARRAY else []
    print(f"\n  Found {len(nc_files)} NetCDF file(s) in {DATA_DIR.name}/")

    if nc_files and HAS_CARTOPY:
        print("\nGenerating polar-projection maps ...\n")

        # Use the most recent file for the concentration map
        lat, lon, conc = load_concentration_grid(nc_files[-1])
        plot_concentration_polar(lat, lon, conc, f" — {nc_files[-1].stem}")

        # If we have at least 2 files, compute an anomaly
        if len(nc_files) >= 2:
            _, _, conc_early = load_concentration_grid(nc_files[0])
            plot_anomaly_map(lat, lon, conc, conc_early)

        plot_trend_heatmap(nc_files)

    else:
        print("\n  No gridded NetCDF data found.")
        print("  Generating fallback visualisation from extent CSVs ...\n")
        print("  ┌─────────────────────────────────────────────────────┐")
        print("  │  For full polar maps, download gridded data from:   │")
        print("  │                                                     │")
        print("  │  • PolarWatch ERDDAP (no login, 2021-present):     │")
        print("  │    polarwatch.noaa.gov/erddap/griddap              │")
        print("  │                                                     │")
        print("  │  • NSIDC CDR G02202 (free Earthdata login):        │")
        print("  │    nsidc.org/data/g02202                            │")
        print("  │                                                     │")
        print("  │  Place .nc files in the data/ folder and re-run.   │")
        print("  └─────────────────────────────────────────────────────┘\n")

        plot_extent_decline_bar()

    print(f"\nFigures saved to {FIGURES_DIR.resolve()}")
    print("Proceed to 04_arctic_amplification.py")


if __name__ == "__main__":
    main()

"""
Step 1 — Data Acquisition
=========================
Downloads all required datasets for the Arctic sea ice analysis:

  1. NSIDC Sea Ice Index v4 (G02135): monthly extent CSVs + daily extent
  2. NASA GISTEMP v4: zonal annual temperature anomalies
  3. (Optional) PolarWatch ERDDAP: gridded sea ice concentration NetCDF

Run this script first.  It is safe to re-run — existing files are skipped.
"""

import sys
import time
from pathlib import Path

import requests

from config import (
    DATA_DIR,
    GISTEMP_FILE,
    GISTEMP_URL,
    NSIDC_CLIMATOLOGY_FILE,
    NSIDC_DAILY_DATA_URL,
    NSIDC_DAILY_FILE,
    NSIDC_MONTHLY_DATA_URL,
    POLARWATCH_ERDDAP,
    POLARWATCH_DATASET_NH,
)


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """Download a file if it doesn't already exist. Returns True on success."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True

    label = description or dest.name
    print(f"  Downloading {label} ...")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"  [done] {dest.name}  ({size_kb:,.0f} KB)")
        return True
    except requests.RequestException as exc:
        print(f"  [FAIL] {label}: {exc}")
        return False


# ── 1. NSIDC monthly extent CSVs (one per calendar month) ──────────────────

def download_nsidc_monthly():
    """Download the 12 monthly-extent CSV files from NSIDC G02135 v4."""
    print("\n▸ NSIDC Sea Ice Index — monthly extent files")
    ok = 0
    for month in range(1, 13):
        fname = f"N_{month:02d}_extent_v4.0.csv"
        url = f"{NSIDC_MONTHLY_DATA_URL}/{fname}"
        if download_file(url, DATA_DIR / fname):
            ok += 1
        time.sleep(0.3)
    print(f"  {ok}/12 monthly files ready.\n")
    return ok == 12


# ── 2. NSIDC daily extent CSV ──────────────────────────────────────────────

def download_nsidc_daily():
    """Download the daily-extent and climatology CSVs."""
    print("▸ NSIDC Sea Ice Index — daily extent + climatology")
    a = download_file(
        f"{NSIDC_DAILY_DATA_URL}/{NSIDC_DAILY_FILE}",
        DATA_DIR / NSIDC_DAILY_FILE,
    )
    b = download_file(
        f"{NSIDC_DAILY_DATA_URL}/{NSIDC_CLIMATOLOGY_FILE}",
        DATA_DIR / NSIDC_CLIMATOLOGY_FILE,
    )
    print()
    return a and b


# ── 3. NASA GISTEMP v4 zonal-mean temperatures ─────────────────────────────

def download_gistemp():
    """Download the GISTEMP v4 zonal annual-mean temperature anomaly CSV."""
    print("▸ NASA GISTEMP v4 — zonal annual mean temperature anomalies")
    ok = download_file(GISTEMP_URL, DATA_DIR / GISTEMP_FILE)
    print()
    return ok


# ── 4. PolarWatch ERDDAP — gridded sea ice concentration (optional) ────────

def download_polarwatch_sample():
    """
    Download a small NetCDF subset of monthly sea ice concentration
    from PolarWatch ERDDAP for spatial analysis demonstrations.

    This dataset (G10016 v2, NRT) covers ~2021-present.  For the full
    satellite-era record (1979-present), register for a free NASA Earthdata
    account and download the NOAA/NSIDC CDR (G02202) — see the README.
    """
    print("▸ PolarWatch ERDDAP — gridded sea ice concentration (sample)")

    dest = DATA_DIR / "polarwatch_sic_sample.nc"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists\n")
        return True

    # Request September data for a few recent years
    years = [2022, 2023, 2024]
    time_constraints = ",".join(
        f"[('{y}-09-01T00:00:00Z')]" for y in years
    )

    # Build individual URLs per year (ERDDAP subset syntax)
    downloaded = False
    for year in years:
        url = (
            f"{POLARWATCH_ERDDAP}/{POLARWATCH_DATASET_NH}.nc"
            f"?cdr_seaice_conc_monthly[({year}-09-01T00:00:00Z)]"
        )
        yr_dest = DATA_DIR / f"polarwatch_sic_{year}_09.nc"
        if yr_dest.exists():
            print(f"  [skip] {yr_dest.name} already exists")
            downloaded = True
            continue

        try:
            print(f"  Requesting Sep {year} concentration grid ...")
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            yr_dest.write_bytes(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"  [done] {yr_dest.name}  ({size_kb:,.0f} KB)")
            downloaded = True
        except requests.RequestException as exc:
            print(f"  [FAIL] Sep {year}: {exc}")
            print("         Gridded data is optional — spatial analysis")
            print("         will provide manual-download instructions.")

        time.sleep(1)

    print()
    return downloaded


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ARCTIC SEA ICE PROJECT — DATA ACQUISITION")
    print("=" * 64)
    print(f"  Download directory: {DATA_DIR.resolve()}\n")

    results = {
        "NSIDC monthly":  download_nsidc_monthly(),
        "NSIDC daily":    download_nsidc_daily(),
        "GISTEMP":        download_gistemp(),
        "PolarWatch":     download_polarwatch_sample(),
    }

    print("=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    for name, ok in results.items():
        status = "✓ ready" if ok else "✗ needs attention"
        print(f"  {name:20s} {status}")

    if not all(results.values()):
        print("\n  Some downloads failed. Check your internet connection")
        print("  or download manually — see README.md for URLs.")
        sys.exit(1)
    else:
        print("\n  All core datasets downloaded successfully!")
        print("  Proceed to 02_time_series_analysis.py")


if __name__ == "__main__":
    main()

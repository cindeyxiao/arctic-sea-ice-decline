# Arctic Sea Ice Decline — Data-Driven Analysis (1979–Present)

**GEO 307 — Introduction to Ocean Science**
Christopher Chen & Cindey Xiao | Prof. Patrick Heimbach | April 2026

---

## Recommended Databases

| # | Database | What It Provides | Coverage | Format | Auth? | URL |
|---|----------|-----------------|----------|--------|-------|-----|
| **1** | **NSIDC Sea Ice Index v4 (G02135)** | Monthly & daily sea ice extent + area | 1978–present | CSV | **No** | [nsidc.org/data/g02135](https://nsidc.org/data/g02135) |
| **2** | **NASA GISTEMP v4** | Zonal annual-mean surface temperature anomalies | 1880–present | CSV | **No** | [data.giss.nasa.gov/gistemp](https://data.giss.nasa.gov/gistemp/) |
| **3** | **NSIDC CDR (G02202)** | Gridded sea ice concentration (25 km) | 1978–present | NetCDF | Earthdata (free) | [nsidc.org/data/g02202](https://nsidc.org/data/g02202) |
| **4** | **PolarWatch ERDDAP** | Gridded sea ice concentration (NRT) | 2021–present | NetCDF | **No** | [polarwatch.noaa.gov/erddap](https://polarwatch.noaa.gov/erddap/) |
| **5** | **NOAA Arctic Report Card** | Arctic temperature anomalies + context | Annual | Report | **No** | [arctic.noaa.gov/report-card](https://arctic.noaa.gov/report-card/) |
| **6** | **ERA5 Reanalysis** | Gridded temp, wind, SIC (global) | 1940–present | NetCDF | CDS (free) | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/) |

### Primary sources used in this project

- **NSIDC G02135** — the backbone of our analysis. Provides consistently processed monthly CSV files of Arctic sea ice extent and area from passive microwave satellites (SMMR, SSM/I, SSMIS). No login required.
- **NASA GISTEMP v4** — provides the 64N–90N zonal band as a proxy for Arctic (60–90°N) temperature, alongside the global mean. We re-reference from the default 1951–1980 baseline to the 1991–2020 baseline used in NOAA's Arctic Report Card.
- **PolarWatch ERDDAP / NSIDC CDR** — for gridded concentration maps. PolarWatch is programmatic (no login) but only covers ~2021+. For the full 1979+ record, register for a free [NASA Earthdata](https://urs.earthdata.nasa.gov/) account and download from G02202.

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
cd "/Users/cindeyxiao/Documents/GEO Project"
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on Cartopy:** If `pip install cartopy` fails, try
> `conda install -c conda-forge cartopy` (requires Anaconda/Miniconda).
> Cartopy is only needed for polar projection maps in `03_spatial_analysis.py`;
> all other scripts work without it.

---

## How to Run

Execute scripts **in order** — each builds on the previous step's output.

```bash
python 01_download_data.py          # Downloads all datasets to data/
python 02_time_series_analysis.py   # Figures 1 & 4
python 03_spatial_analysis.py       # Polar maps (if gridded data available)
python 04_arctic_amplification.py   # Figure 2
python 05_predictive_model.py       # Figure 3 + model summary
python 06_additional_figures.py     # Figures 5–10
```

---

## All 10 Paper Figures

| Fig # | Script | Description | Section | Output |
|-------|--------|-------------|---------|--------|
| **1** | `02_time_series_analysis.py` | September minimum + OLS trend + blue ocean threshold | 3.1 | `fig01_september_minimum.png/.html` |
| **2** | `04_arctic_amplification.py` | Arctic vs global temperature anomalies | 3.3 | `fig02_arctic_amplification.png/.html` |
| **3** | `05_predictive_model.py` | Linear regression extrapolation → blue ocean event | 3.4 | `fig03_blue_ocean_projection.png/.html` |
| **4** | `02_time_series_analysis.py` | Decadal loss-rate acceleration bar chart | 3.2 | `fig04_decadal_loss_rates.png` |
| **5** | `06_additional_figures.py` | Daily extent spaghetti (all years, record years highlighted) | 3.1 | `fig05_daily_spaghetti.png/.html` |
| **6** | `06_additional_figures.py` | Per-month trend slopes (which months decline fastest) | 3.2 | `fig06_month_trends.png` |
| **7** | `06_additional_figures.py` | All years ranked by September extent, colored by decade | 3.1 | `fig07_ranked_september.png` |
| **8** | `06_additional_figures.py` | Month × year anomaly heatmap | 3 | `fig08_anomaly_heatmap.png` |
| **9** | `06_additional_figures.py` | Arctic temperature vs September extent scatter | 4 | `fig09_temp_vs_extent.png` |
| **10** | `06_additional_figures.py` | Ice–albedo feedback schematic diagram | 4.1 | `fig10_feedback_diagram.png` |

### Supplementary figures (also generated)

| File | Description |
|------|-------------|
| `fig_S1_all_months.png` | All 12 monthly extent curves overlaid |
| `fig_S2_seasonal_cycle.png` | Baseline (1981–2010) vs recent decade seasonal cycle |
| `fig_S3_anomaly_heatmap.png` | Month × year extent anomaly heatmap |
| `fig_S4_amplification_ratio.png` | 10-year rolling Arctic amplification ratio |
| `fig_S5_temp_vs_ice_scatter.png` | Arctic temperature anomaly vs September extent |
| `fig_S6_residual_diagnostics.png` | Linear model residual analysis |
| `fig_S7_quadratic_comparison.png` | Linear vs quadratic extrapolation comparison |

---

## Project Structure

```
GEO Project/
├── README.md                      ← You are here
├── requirements.txt               ← Python dependencies
├── config.py                      ← Paths, URLs, analysis parameters
├── 01_download_data.py            ← Data acquisition
├── 02_time_series_analysis.py     ← Fig 1, Fig 4 + supplementary
├── 03_spatial_analysis.py         ← Polar projection maps
├── 04_arctic_amplification.py     ← Fig 2 + supplementary
├── 05_predictive_model.py         ← Fig 3 + model summary
├── 06_additional_figures.py       ← Figs 5–10
├── data/                          ← Downloaded datasets (auto-created)
└── figures/                       ← Generated plots (auto-created)
```

---

## Key Libraries

| Library | Role in This Project |
|---------|---------------------|
| `pandas` | Tabular time-series handling, CSV I/O |
| `numpy` | Numerical computation, array operations |
| `matplotlib` | Publication-quality static figures |
| `plotly` | Interactive HTML figures |
| `cartopy` | Polar stereographic map projections |
| `xarray` + `netCDF4` | Multidimensional gridded NetCDF fields |
| `scikit-learn` | LinearRegression model for blue ocean projection |
| `scipy` | Statistical tests, confidence intervals, regression |

---

## Manual Data Download (if automated download fails)

### NSIDC Sea Ice Index (G02135)
1. Go to https://noaadata.apps.nsidc.org/NOAA/G02135/north/monthly/data/
2. Download all files named `N_MM_extent_v4.0.csv` (MM = 01 through 12)
3. Also download `N_seaice_extent_daily_v4.0.csv` from the `daily/data/` folder
4. Place all CSV files in the `data/` folder

### NASA GISTEMP v4
1. Go to https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv
2. Save the file to `data/ZonAnn.Ts+dSST.csv`

### Gridded Sea Ice Concentration (for polar maps)
1. Register for a free account at https://urs.earthdata.nasa.gov/
2. Go to https://nsidc.org/data/g02202
3. Download monthly NetCDF files for September, multiple years
4. Place `.nc` files in the `data/` folder

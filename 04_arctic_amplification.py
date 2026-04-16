"""
Step 4 — Arctic Amplification Analysis
========================================
Produces the paper's **Figure 2**: Arctic vs global temperature anomalies.

Additional outputs:
  • fig02_arctic_amplification.png  — Publication-quality Matplotlib figure
  • fig02_arctic_amplification.html — Interactive Plotly version
  • fig_S4_amplification_ratio.png  — Running amplification ratio over time
  • fig_S5_temp_vs_ice_scatter.png  — Scatter of Arctic temp vs Sept extent

Data: NASA GISTEMP v4 zonal annual means (64N–90N as Arctic proxy).
      Anomalies are re-referenced to the 1991–2020 baseline used in the paper.

Requires: 01_download_data.py to have been run first.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from config import (
    DATA_DIR,
    FIGURES_DIR,
    GISTEMP_ARCTIC_COL,
    GISTEMP_FILE,
    GISTEMP_GLOBAL_COL,
    PLOT_STYLE,
    SATELLITE_ERA_START,
    TEMP_BASELINE_PERIOD,
)

plt.rcParams.update(PLOT_STYLE)


# ── Data loading ────────────────────────────────────────────────────────────

def load_gistemp() -> pd.DataFrame:
    """
    Load GISTEMP v4 zonal annual means and re-reference anomalies
    from the default 1951–1980 baseline to the paper's 1991–2020 baseline.
    """
    path = DATA_DIR / GISTEMP_FILE
    if not path.exists():
        sys.exit(f"Missing {path.name}. Run 01_download_data.py first.")

    df = pd.read_csv(path, na_values=["***", "****"])
    df.rename(columns=lambda c: c.strip(), inplace=True)

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df.dropna(subset=["Year"], inplace=True)
    df["Year"] = df["Year"].astype(int)

    for col in [GISTEMP_ARCTIC_COL, GISTEMP_GLOBAL_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Re-reference: subtract 1991–2020 mean so anomalies are relative to that
    ref = df[df["Year"].between(*TEMP_BASELINE_PERIOD)]
    arctic_offset = ref[GISTEMP_ARCTIC_COL].mean()
    global_offset = ref[GISTEMP_GLOBAL_COL].mean()

    df["arctic_anom"] = df[GISTEMP_ARCTIC_COL] - arctic_offset
    df["global_anom"] = df[GISTEMP_GLOBAL_COL] - global_offset

    return df[df["Year"] >= SATELLITE_ERA_START].reset_index(drop=True)


def load_september_extent() -> pd.DataFrame:
    """Load September extent for correlation analysis."""
    path = DATA_DIR / "N_09_extent_v4.0.csv"
    if not path.exists():
        return pd.DataFrame()

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
    df = df[df["extent"] > 0]  # NSIDC uses -9999 as missing-data flag
    df["year"] = df["year"].astype(int)
    return df[df["year"] >= SATELLITE_ERA_START].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Arctic vs Global Temperature Anomalies
# ═══════════════════════════════════════════════════════════════════════════

def figure2_arctic_amplification(temp: pd.DataFrame):
    """Paper Figure 2: diverging Arctic vs global temperature anomalies."""
    years = temp["Year"].values
    arctic = temp["arctic_anom"].values
    globe = temp["global_anom"].values

    # OLS trends
    sa, ia, *_ = stats.linregress(years, arctic)
    sg, ig, *_ = stats.linregress(years, globe)
    ratio = sa / sg if sg != 0 else float("inf")

    # ── Matplotlib ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(years, arctic, "o-", color="#d62728", markersize=3, linewidth=1.2,
            label="Arctic (64–90°N)")
    ax.plot(years, globe, "s-", color="#1f77b4", markersize=3, linewidth=1.2,
            label="Global Mean")

    ax.plot(years, sa * years + ia, "--", color="#d62728", linewidth=1.5,
            alpha=0.6, label=f"Arctic trend: {sa * 10:+.2f} °C/decade")
    ax.plot(years, sg * years + ig, "--", color="#1f77b4", linewidth=1.5,
            alpha=0.6, label=f"Global trend: {sg * 10:+.2f} °C/decade")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.fill_between(years, arctic, globe, alpha=0.08, color="#d62728")

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Temperature Anomaly (°C, rel. to "
                  f"{TEMP_BASELINE_PERIOD[0]}–{TEMP_BASELINE_PERIOD[1]})")
    ax.set_title("Arctic Amplification: Surface Temperature Anomalies (1979–Present)")
    ax.legend(loc="upper left", fontsize=9)

    ax.annotate(f"Amplification ratio ≈ {ratio:.1f}×",
                xy=(0.98, 0.05), xycoords="axes fraction",
                ha="right", fontsize=10, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig02_arctic_amplification.png", dpi=200)
    plt.close(fig)

    # ── Plotly ──────────────────────────────────────────────────────────
    pfig = go.Figure()
    pfig.add_trace(go.Scatter(
        x=years, y=arctic, mode="lines+markers", name="Arctic (64–90°N)",
        marker=dict(size=4, color="#d62728"),
        line=dict(color="#d62728", width=1.5),
    ))
    pfig.add_trace(go.Scatter(
        x=years, y=globe, mode="lines+markers", name="Global Mean",
        marker=dict(size=4, color="#1f77b4"),
        line=dict(color="#1f77b4", width=1.5),
    ))
    pfig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.8)
    pfig.update_layout(
        title="Arctic Amplification: Surface Temperature Anomalies",
        xaxis_title="Year",
        yaxis_title=f"Temperature Anomaly (°C, rel. {TEMP_BASELINE_PERIOD[0]}–{TEMP_BASELINE_PERIOD[1]})",
        template="plotly_white", hovermode="x unified",
    )
    pfig.write_html(FIGURES_DIR / "fig02_arctic_amplification.html")

    print(f"  Fig 02 saved.  Arctic trend = {sa * 10:+.3f} °C/decade, "
          f"Global = {sg * 10:+.3f} °C/decade, ratio ≈ {ratio:.1f}×")

    return ratio


# ═══════════════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY: Running amplification ratio
# ═══════════════════════════════════════════════════════════════════════════

def fig_s4_running_ratio(temp: pd.DataFrame, window: int = 10):
    """Rolling-window amplification ratio."""
    years = temp["Year"].values
    arctic = temp["arctic_anom"].values
    globe = temp["global_anom"].values

    ratios, centres = [], []
    for i in range(len(years) - window + 1):
        sl_a, *_ = stats.linregress(years[i:i + window], arctic[i:i + window])
        sl_g, *_ = stats.linregress(years[i:i + window], globe[i:i + window])
        if sl_g > 0:
            ratios.append(sl_a / sl_g)
            centres.append(years[i] + window // 2)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(centres, ratios, "o-", color="#8c564b", markersize=3)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Centre Year of Window")
    ax.set_ylabel("Amplification Ratio (Arctic / Global trend)")
    ax.set_title(f"{window}-Year Rolling Arctic Amplification Ratio")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_S4_amplification_ratio.png", dpi=200)
    plt.close(fig)
    print("  Fig S4 saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY: Temperature vs September extent scatter
# ═══════════════════════════════════════════════════════════════════════════

def fig_s5_temp_vs_extent(temp: pd.DataFrame):
    """Scatter of Arctic temperature anomaly vs September sea ice extent."""
    ice = load_september_extent()
    if ice.empty:
        print("  Fig S5 skipped (no September extent data).")
        return

    merged = pd.merge(
        temp[["Year", "arctic_anom"]],
        ice[["year", "extent"]],
        left_on="Year", right_on="year", how="inner",
    )

    x = merged["arctic_anom"].values
    y = merged["extent"].values
    slope, intercept, r, p, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(x, y, c=merged["Year"], cmap="viridis", s=40,
                    edgecolors="black", linewidths=0.3, zorder=3)
    ax.plot(x, slope * x + intercept, "--", color="#d62728", linewidth=1.5,
            label=f"R² = {r**2:.3f}, p = {p:.2e}")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Year")
    ax.set_xlabel("Arctic Temperature Anomaly (°C)")
    ax.set_ylabel("September Sea Ice Extent (million km²)")
    ax.set_title("Arctic Warming vs September Sea Ice Extent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_S5_temp_vs_ice_scatter.png", dpi=200)
    plt.close(fig)
    print(f"  Fig S5 saved.  R² = {r**2:.3f}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ARCTIC SEA ICE — ARCTIC AMPLIFICATION ANALYSIS")
    print("=" * 64)

    print("\nLoading GISTEMP v4 temperature data ...")
    temp = load_gistemp()
    print(f"  {len(temp)} years loaded ({temp['Year'].min()}–{temp['Year'].max()})")
    print(f"  Anomalies re-referenced to {TEMP_BASELINE_PERIOD[0]}–"
          f"{TEMP_BASELINE_PERIOD[1]} baseline\n")

    print("Generating figures ...")
    figure2_arctic_amplification(temp)
    fig_s4_running_ratio(temp)
    fig_s5_temp_vs_extent(temp)

    print(f"\nAll figures saved to {FIGURES_DIR.resolve()}")
    print("Proceed to 05_predictive_model.py")


if __name__ == "__main__":
    main()

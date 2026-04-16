"""
Step 2 — Time-Series Analysis of Arctic Sea Ice Extent
=======================================================
Produces the paper's **Figure 1** and **Figure 4**, plus supplementary plots:

  • fig01_september_minimum.png   — Sept minimum extent + OLS trend + blue ocean line
  • fig01_september_minimum.html  — Interactive Plotly version of Figure 1
  • fig04_decadal_loss_rates.png  — Decadal loss-rate bar chart
  • fig_S1_all_months.png         — All 12 months overlaid
  • fig_S2_seasonal_cycle.png     — Baseline vs recent seasonal cycle
  • fig_S3_anomaly_heatmap.png    — Monthly anomalies heatmap

Requires: 01_download_data.py to have been run first.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from config import (
    BASELINE_PERIOD,
    BLUE_OCEAN_THRESHOLD_KM2,
    DATA_DIR,
    FIGURES_DIR,
    PLOT_STYLE,
    SATELLITE_ERA_START,
    SEPTEMBER,
)

plt.rcParams.update(PLOT_STYLE)


# ── Data loading ────────────────────────────────────────────────────────────

def load_monthly_extent(month: int) -> pd.DataFrame:
    """Load one NSIDC monthly-extent CSV and return a tidy DataFrame."""
    path = DATA_DIR / f"N_{month:02d}_extent_v4.0.csv"
    if not path.exists():
        sys.exit(f"Missing {path.name}. Run 01_download_data.py first.")

    df = pd.read_csv(path, skipinitialspace=True, comment="#")
    df.columns = df.columns.str.strip().str.lower()

    rename = {}
    for col in df.columns:
        if "year" in col:
            rename[col] = "year"
        elif "extent" in col:
            rename[col] = "extent"
        elif "area" in col:
            rename[col] = "area"
        elif col in ("mo", "month"):
            rename[col] = "month"
    df.rename(columns=rename, inplace=True)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["extent"] = pd.to_numeric(df["extent"], errors="coerce")
    df.dropna(subset=["year", "extent"], inplace=True)
    df = df[df["extent"] > 0]  # NSIDC uses -9999 as missing-data flag
    df["year"] = df["year"].astype(int)
    if "month" not in df.columns:
        df["month"] = month

    return df[df["year"] >= SATELLITE_ERA_START].reset_index(drop=True)


def load_all_months() -> pd.DataFrame:
    """Concatenate all 12 monthly-extent CSVs into one DataFrame."""
    frames = []
    for m in range(1, 13):
        try:
            frames.append(load_monthly_extent(m))
        except SystemExit:
            print(f"  Warning: month {m:02d} file missing, skipping.")
    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["year", "month"], inplace=True)
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — September Minimum Extent with Trend & Blue Ocean Threshold
# ═══════════════════════════════════════════════════════════════════════════

def figure1_september_minimum(df_all: pd.DataFrame):
    """Paper Figure 1: September extent, OLS trend, blue-ocean threshold."""
    sept = df_all[df_all["month"] == SEPTEMBER].copy().sort_values("year")
    x = sept["year"].values.astype(float)
    y = sept["extent"].values

    slope, intercept, r, p, se = stats.linregress(x, y)
    trend_line = slope * x + intercept

    # ── Matplotlib (publication quality) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(x, y, "o-", color="#1f77b4", markersize=4, linewidth=1.3,
            label="Observed September extent")
    ax.plot(x, trend_line, "--", color="#2ca02c", linewidth=2,
            label=f"Linear trend: {slope * 10:+.2f} M km² decade⁻¹  "
                  f"(R² = {r**2:.3f}, p < 0.001)")
    ax.axhline(BLUE_OCEAN_THRESHOLD_KM2, color="#d62728", linestyle=":",
               linewidth=1.5, label='"Blue ocean" threshold (1 M km²)')

    ax.fill_between(x, y, trend_line, alpha=0.07, color="#1f77b4")

    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Ice Extent (million km²)")
    ax.set_title("Arctic September Minimum Sea Ice Extent (1979–Present)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig01_september_minimum.png", dpi=200)
    plt.close(fig)

    # ── Plotly (interactive) ────────────────────────────────────────────
    pfig = go.Figure()
    pfig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name="Observed",
        marker=dict(size=5, color="#1f77b4"),
        line=dict(color="#1f77b4", width=1.5),
    ))
    pfig.add_trace(go.Scatter(
        x=x, y=trend_line, mode="lines", name="Linear Trend",
        line=dict(color="#2ca02c", width=2, dash="dash"),
    ))
    pfig.add_hline(
        y=BLUE_OCEAN_THRESHOLD_KM2, line_dash="dot", line_color="#d62728",
        annotation_text="Blue Ocean Threshold (1 M km²)",
        annotation_position="bottom right",
    )
    pfig.update_layout(
        title="Arctic September Minimum Sea Ice Extent (1979–Present)",
        xaxis_title="Year", yaxis_title="Sea Ice Extent (million km²)",
        template="plotly_white", hovermode="x unified",
    )
    pfig.write_html(FIGURES_DIR / "fig01_september_minimum.html")

    print(f"  Fig 01 saved.  Slope = {slope:.4f} M km²/yr "
          f"({slope * 10:.3f}/decade), R² = {r**2:.3f}, p = {p:.2e}")

    return sept


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Decadal Loss-Rate Bar Chart
# ═══════════════════════════════════════════════════════════════════════════

def figure4_decadal_loss_rates(df_all: pd.DataFrame):
    """Paper Figure 4: approximate September loss rate by decade."""
    sept = df_all[df_all["month"] == SEPTEMBER].copy().sort_values("year")

    baseline_mean = sept[
        sept["year"].between(*BASELINE_PERIOD)
    ]["extent"].mean()

    decades = [
        ("1980s", 1979, 1989),
        ("1990s", 1990, 1999),
        ("2000s", 2000, 2009),
        ("2010s", 2010, 2019),
        ("2020s", 2020, sept["year"].max()),
    ]

    labels, rates = [], []
    for label, y1, y2 in decades:
        sub = sept[(sept["year"] >= y1) & (sept["year"] <= y2)]
        if len(sub) < 3:
            continue
        x = sub["year"].values.astype(float)
        y = sub["extent"].values
        sl, *_ = stats.linregress(x, y)
        pct_per_decade = (sl * 10 / baseline_mean) * 100
        labels.append(label)
        rates.append(abs(pct_per_decade))

    colors = plt.cm.Reds(np.linspace(0.25, 0.85, len(labels)))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, rates, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_ylabel("Loss Rate (% per decade, relative to 1981–2010 mean)")
    ax.set_title("Acceleration of September Sea Ice Decline by Decade")
    ax.set_ylim(0, max(rates) * 1.25)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig04_decadal_loss_rates.png", dpi=200)
    plt.close(fig)
    print("  Fig 04 saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def fig_s1_all_months(df_all: pd.DataFrame):
    """All 12 monthly extent curves overlaid."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.twilight_shifted
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for m in range(1, 13):
        sub = df_all[df_all["month"] == m].sort_values("year")
        color = cmap(m / 12)
        lw = 2.2 if m == SEPTEMBER else 0.9
        alpha = 1.0 if m == SEPTEMBER else 0.55
        ax.plot(sub["year"], sub["extent"], color=color, linewidth=lw,
                alpha=alpha, label=month_names[m - 1])

    ax.set_xlabel("Year")
    ax.set_ylabel("Extent (million km²)")
    ax.set_title("Monthly Arctic Sea Ice Extent by Calendar Month (1979–Present)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_S1_all_months.png", dpi=200)
    plt.close(fig)
    print("  Fig S1 saved.")


def fig_s2_seasonal_cycle(df_all: pd.DataFrame):
    """Baseline vs. recent-decade seasonal cycle."""
    baseline = df_all[
        df_all["year"].between(*BASELINE_PERIOD)
    ].groupby("month")["extent"].agg(["mean", "std"])

    recent_start = df_all["year"].max() - 9
    recent = df_all[
        df_all["year"] >= recent_start
    ].groupby("month")["extent"].agg(["mean", "std"])

    months = np.arange(1, 13)
    labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(months, baseline["mean"] - baseline["std"],
                     baseline["mean"] + baseline["std"],
                     alpha=0.15, color="#1f77b4")
    ax.plot(months, baseline["mean"], "o-", color="#1f77b4", linewidth=2,
            label=f"Baseline {BASELINE_PERIOD[0]}–{BASELINE_PERIOD[1]}")
    ax.fill_between(months, recent["mean"] - recent["std"],
                     recent["mean"] + recent["std"],
                     alpha=0.15, color="#d62728")
    ax.plot(months, recent["mean"], "s-", color="#d62728", linewidth=2,
            label=f"Recent {recent_start}–{df_all['year'].max()}")

    ax.set_xticks(months)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Extent (million km²)")
    ax.set_title("Seasonal Cycle of Arctic Sea Ice Extent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_S2_seasonal_cycle.png", dpi=200)
    plt.close(fig)
    print("  Fig S2 saved.")


def fig_s3_anomaly_heatmap(df_all: pd.DataFrame):
    """Month × year anomaly heatmap relative to 1981–2010 baseline."""
    baseline_mean = (
        df_all[df_all["year"].between(*BASELINE_PERIOD)]
        .groupby("month")["extent"].mean()
    )

    df = df_all.copy()
    df["anomaly"] = df.apply(
        lambda r: r["extent"] - baseline_mean.get(r["month"], np.nan), axis=1
    )
    pivot = df.pivot_table(index="month", columns="year", values="anomaly")

    fig, ax = plt.subplots(figsize=(14, 5))
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    im = ax.pcolormesh(pivot.columns, pivot.index, pivot.values,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Extent Anomaly (million km²)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Month")
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(["J", "F", "M", "A", "M", "J",
                         "J", "A", "S", "O", "N", "D"])
    ax.set_title(f"Sea Ice Extent Anomalies vs "
                 f"{BASELINE_PERIOD[0]}–{BASELINE_PERIOD[1]} Baseline")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_S3_anomaly_heatmap.png", dpi=200)
    plt.close(fig)
    print("  Fig S3 saved.")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ARCTIC SEA ICE — TIME-SERIES ANALYSIS")
    print("=" * 64)

    print("\nLoading monthly extent data ...")
    df_all = load_all_months()
    n_years = df_all["year"].nunique()
    print(f"  {len(df_all)} records loaded ({n_years} years, "
          f"{SATELLITE_ERA_START}–{df_all['year'].max()})\n")

    print("Generating paper figures ...")
    figure1_september_minimum(df_all)
    figure4_decadal_loss_rates(df_all)

    print("\nGenerating supplementary figures ...")
    fig_s1_all_months(df_all)
    fig_s2_seasonal_cycle(df_all)
    fig_s3_anomaly_heatmap(df_all)

    print(f"\nAll figures saved to {FIGURES_DIR.resolve()}")
    print("Proceed to 03_spatial_analysis.py")


if __name__ == "__main__":
    main()

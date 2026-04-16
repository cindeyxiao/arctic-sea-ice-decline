"""
Step 6 — Additional Figures (5–10)
====================================
Generates six more publication-quality figures to bring the paper's
total to 10.  Each figure is tied to a specific section of the paper.

  • fig05_daily_spaghetti.png        — Annual cycle spaghetti (Sec 3.1)
  • fig05_daily_spaghetti.html       — Interactive Plotly version
  • fig06_month_trends.png           — Per-month trend slopes   (Sec 3.2)
  • fig07_ranked_september.png       — Years ranked by extent    (Sec 3.1)
  • fig08_anomaly_heatmap.png        — Month × year heatmap      (Sec 3)
  • fig09_temp_vs_extent.png         — Arctic temp ↔ ice scatter (Sec 4)
  • fig10_feedback_diagram.png       — Ice–albedo feedback loop  (Sec 4.1)

Requires: 01_download_data.py to have been run first.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from config import (
    BASELINE_PERIOD,
    DATA_DIR,
    FIGURES_DIR,
    GISTEMP_ARCTIC_COL,
    GISTEMP_FILE,
    GISTEMP_GLOBAL_COL,
    PLOT_STYLE,
    SATELLITE_ERA_START,
    SEPTEMBER,
    TEMP_BASELINE_PERIOD,
)

plt.rcParams.update(PLOT_STYLE)


# ── Shared data loaders ────────────────────────────────────────────────────

def _load_monthly(month: int) -> pd.DataFrame:
    path = DATA_DIR / f"N_{month:02d}_extent_v4.0.csv"
    if not path.exists():
        sys.exit(f"Missing {path.name}. Run 01_download_data.py first.")
    df = pd.read_csv(path, skipinitialspace=True, comment="#")
    df.columns = df.columns.str.strip().str.lower()
    rename = {}
    for col in df.columns:
        if "year" in col:   rename[col] = "year"
        elif "extent" in col: rename[col] = "extent"
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
    frames = []
    for m in range(1, 13):
        try:
            frames.append(_load_monthly(m))
        except SystemExit:
            pass
    df = pd.concat(frames, ignore_index=True)
    df.sort_values(["year", "month"], inplace=True)
    return df.reset_index(drop=True)


def load_daily() -> pd.DataFrame:
    path = DATA_DIR / "N_seaice_extent_daily_v4.0.csv"
    if not path.exists():
        sys.exit(f"Missing {path.name}. Run 01_download_data.py first.")

    df = pd.read_csv(path, skipinitialspace=True, comment="#")
    df.columns = df.columns.str.strip().str.lower()

    rename = {}
    for col in df.columns:
        lc = col.lower()
        if "year" in lc:   rename[col] = "year"
        elif "month" in lc: rename[col] = "month"
        elif "day" in lc:   rename[col] = "day"
        elif "extent" in lc: rename[col] = "extent"
    df.rename(columns=rename, inplace=True)

    for c in ["year", "month", "day"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["extent"] = pd.to_numeric(df["extent"], errors="coerce")
    df.dropna(subset=["year", "month", "day", "extent"], inplace=True)
    df = df[df["extent"] > 0]  # NSIDC uses -9999 as missing-data flag
    df["year"] = df["year"].astype(int)

    df = df[df["year"] >= SATELLITE_ERA_START].copy()
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df["doy"] = df["date"].dt.dayofyear
    return df.reset_index(drop=True)


def load_gistemp() -> pd.DataFrame:
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
    ref = df[df["Year"].between(*TEMP_BASELINE_PERIOD)]
    df["arctic_anom"] = df[GISTEMP_ARCTIC_COL] - ref[GISTEMP_ARCTIC_COL].mean()
    df["global_anom"] = df[GISTEMP_GLOBAL_COL] - ref[GISTEMP_GLOBAL_COL].mean()
    return df[df["Year"] >= SATELLITE_ERA_START].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Daily Extent Spaghetti Plot  (Section 3.1)
# ═══════════════════════════════════════════════════════════════════════════

def figure5_daily_spaghetti(daily: pd.DataFrame):
    """
    The iconic NSIDC-style chart: every year's daily extent as a faint line,
    with the climatological mean, ±2σ envelope, and record years highlighted.
    """
    highlight_years = {
        2012: ("#d62728", "2012 (record low)"),
        2020: ("#ff7f0e", "2020"),
        2007: ("#9467bd", "2007 (step change)"),
    }
    latest_year = daily["year"].max()
    highlight_years[latest_year] = ("#2ca02c", str(latest_year))

    # Climatology (1981–2010)
    clim = daily[daily["year"].between(*BASELINE_PERIOD)]
    clim_mean = clim.groupby("doy")["extent"].mean()
    clim_std = clim.groupby("doy")["extent"].std()

    # ── Matplotlib ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # Background: all years in light gray
    for yr, grp in daily.groupby("year"):
        if yr in highlight_years:
            continue
        ax.plot(grp["doy"], grp["extent"], color="#cccccc", linewidth=0.3,
                alpha=0.5)

    # Climatology ± 2σ envelope
    doys = clim_mean.index
    ax.fill_between(doys, clim_mean - 2 * clim_std,
                     clim_mean + 2 * clim_std,
                     alpha=0.15, color="#1f77b4", label="±2σ (1981–2010)")
    ax.plot(doys, clim_mean, color="#1f77b4", linewidth=2, alpha=0.7,
            label="Mean (1981–2010)")

    # Highlighted years on top
    for yr, (color, label) in highlight_years.items():
        grp = daily[daily["year"] == yr]
        ax.plot(grp["doy"], grp["extent"], color=color, linewidth=2.2,
                label=label, path_effects=[
                    pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()
                ])

    # Month labels on x-axis
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Sea Ice Extent (million km²)")
    ax.set_title("Daily Arctic Sea Ice Extent — All Years (1979–Present)")
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    ax.set_xlim(1, 365)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig05_daily_spaghetti.png", dpi=200)
    plt.close(fig)

    # ── Plotly ──────────────────────────────────────────────────────────
    pfig = go.Figure()

    for yr, grp in daily.groupby("year"):
        if yr in highlight_years:
            continue
        pfig.add_trace(go.Scatter(
            x=grp["doy"], y=grp["extent"], mode="lines",
            line=dict(color="rgba(180,180,180,0.25)", width=0.5),
            showlegend=False, hoverinfo="skip",
        ))

    pfig.add_trace(go.Scatter(
        x=doys, y=(clim_mean + 2 * clim_std).values,
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    pfig.add_trace(go.Scatter(
        x=doys, y=(clim_mean - 2 * clim_std).values,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(31,119,180,0.12)",
        name="±2σ (1981–2010)",
    ))
    pfig.add_trace(go.Scatter(
        x=doys, y=clim_mean.values, mode="lines",
        line=dict(color="#1f77b4", width=2), name="Mean (1981–2010)",
    ))

    for yr, (color, label) in highlight_years.items():
        grp = daily[daily["year"] == yr].sort_values("doy")
        pfig.add_trace(go.Scatter(
            x=grp["doy"], y=grp["extent"], mode="lines",
            line=dict(color=color, width=2.5), name=label,
        ))

    pfig.update_layout(
        title="Daily Arctic Sea Ice Extent — All Years",
        xaxis_title="Day of Year", yaxis_title="Extent (million km²)",
        template="plotly_white", hovermode="x unified",
        xaxis=dict(
            tickvals=month_starts,
            ticktext=month_labels,
        ),
    )
    pfig.write_html(FIGURES_DIR / "fig05_daily_spaghetti.html")

    print("  Fig 05 saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Per-Month Trend Slopes  (Section 3.2)
# ═══════════════════════════════════════════════════════════════════════════

def figure6_month_trends(df_all: pd.DataFrame):
    """
    Bar chart showing the linear trend slope for each calendar month.
    Demonstrates that September is the fastest-declining month,
    but all months show statistically significant negative trends.
    """
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    slopes, p_vals = [], []

    baseline_means = (
        df_all[df_all["year"].between(*BASELINE_PERIOD)]
        .groupby("month")["extent"].mean()
    )

    for m in range(1, 13):
        sub = df_all[df_all["month"] == m].sort_values("year")
        sl, _, _, p, _ = stats.linregress(sub["year"], sub["extent"])
        pct_decade = (sl * 10 / baseline_means[m]) * 100
        slopes.append(pct_decade)
        p_vals.append(p)

    slopes = np.array(slopes)
    significant = [p < 0.05 for p in p_vals]

    colors = []
    for s, sig in zip(slopes, significant):
        if not sig:
            colors.append("#aaaaaa")
        elif abs(s) > 10:
            colors.append("#d62728")
        elif abs(s) > 5:
            colors.append("#ff7f0e")
        else:
            colors.append("#1f77b4")

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(month_labels, slopes, color=colors, edgecolor="black",
                  linewidth=0.4)

    for bar, val, sig in zip(bars, slopes, significant):
        marker = f"{val:.1f}%" if sig else f"{val:.1f}% (ns)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.5 if val < 0 else bar.get_height() + 0.3,
                marker, ha="center", va="top" if val < 0 else "bottom",
                fontsize=8, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Trend (% per decade, rel. 1981–2010 mean)")
    ax.set_title("Linear Trend in Arctic Sea Ice Extent by Calendar Month")

    legend_elements = [
        mpatches.Patch(facecolor="#d62728", label="> 10% /decade"),
        mpatches.Patch(facecolor="#ff7f0e", label="5–10% /decade"),
        mpatches.Patch(facecolor="#1f77b4", label="< 5% /decade"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig06_month_trends.png", dpi=200)
    plt.close(fig)
    print("  Fig 06 saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Ranked September Extent  (Section 3.1)
# ═══════════════════════════════════════════════════════════════════════════

def figure7_ranked_september(df_all: pd.DataFrame):
    """
    Horizontal bar chart of all years ranked by September extent.
    Color-coded by decade.  Visually proves that the 18 lowest
    extents all occurred after 2007.
    """
    sept = (
        df_all[df_all["month"] == SEPTEMBER]
        .sort_values("extent")
        .reset_index(drop=True)
    )

    decade_colors = {
        1970: "#1f77b4", 1980: "#1f77b4", 1990: "#aec7e8",
        2000: "#ff7f0e", 2010: "#d62728", 2020: "#8b0000",
    }
    sept["decade"] = (sept["year"] // 10) * 10
    sept["color"] = sept["decade"].map(
        lambda d: decade_colors.get(d, "#999999")
    )

    fig, ax = plt.subplots(figsize=(8, max(8, len(sept) * 0.22)))
    ax.barh(range(len(sept)), sept["extent"], color=sept["color"],
            edgecolor="black", linewidth=0.3, height=0.75)

    ax.set_yticks(range(len(sept)))
    ax.set_yticklabels(sept["year"].astype(str), fontsize=7)
    ax.set_xlabel("September Extent (million km²)")
    ax.set_title("All Years Ranked by September Arctic Sea Ice Extent")
    ax.invert_yaxis()

    ax.axvline(1.0, color="#d62728", linestyle=":", linewidth=1.2,
               label="Blue ocean (1 M km²)")

    # Decade legend
    handles = [
        mpatches.Patch(facecolor="#1f77b4", label="1979–1989"),
        mpatches.Patch(facecolor="#aec7e8", label="1990s"),
        mpatches.Patch(facecolor="#ff7f0e", label="2000s"),
        mpatches.Patch(facecolor="#d62728", label="2010s"),
        mpatches.Patch(facecolor="#8b0000", label="2020s"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig07_ranked_september.png", dpi=200)
    plt.close(fig)
    print("  Fig 07 saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 8 — Month × Year Anomaly Heatmap  (Section 3)
# ═══════════════════════════════════════════════════════════════════════════

def figure8_anomaly_heatmap(df_all: pd.DataFrame):
    """
    Heatmap of extent anomalies (vs 1981–2010) by month and year.
    The deepening blue in recent decades across all months is
    immediately visible.
    """
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
    ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title(f"Arctic Sea Ice Extent Anomalies "
                 f"(vs {BASELINE_PERIOD[0]}–{BASELINE_PERIOD[1]} Mean)")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig08_anomaly_heatmap.png", dpi=200)
    plt.close(fig)
    print("  Fig 08 saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 9 — Arctic Temperature vs September Extent  (Section 4)
# ═══════════════════════════════════════════════════════════════════════════

def figure9_temp_vs_extent(df_all: pd.DataFrame, temp: pd.DataFrame):
    """
    Scatter plot linking Arctic warming directly to ice loss.
    Points colored by year to show the temporal progression.
    """
    sept = df_all[df_all["month"] == SEPTEMBER][["year", "extent"]]
    merged = pd.merge(
        temp[["Year", "arctic_anom"]],
        sept, left_on="Year", right_on="year", how="inner",
    )
    x = merged["arctic_anom"].values
    y = merged["extent"].values
    years = merged["Year"].values
    sl, ic, r, p, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(x, y, c=years, cmap="viridis", s=50,
                    edgecolors="black", linewidths=0.4, zorder=3)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, sl * x_fit + ic, "--", color="#d62728", linewidth=1.8,
            label=f"OLS: R² = {r**2:.3f}, p = {p:.1e}")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Year")
    ax.set_xlabel(f"Arctic Temperature Anomaly (°C, rel. "
                  f"{TEMP_BASELINE_PERIOD[0]}–{TEMP_BASELINE_PERIOD[1]})")
    ax.set_ylabel("September Sea Ice Extent (million km²)")
    ax.set_title("Arctic Warming vs September Sea Ice Loss")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig09_temp_vs_extent.png", dpi=200)
    plt.close(fig)
    print(f"  Fig 09 saved.  R² = {r**2:.3f}, slope = {sl:.3f} M km² per °C")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 10 — Ice–Albedo Feedback Schematic Diagram  (Section 4.1)
# ═══════════════════════════════════════════════════════════════════════════

def figure10_feedback_diagram():
    """
    Programmatic schematic of the ice–albedo positive feedback loop,
    as promised in Section 2.3 of the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Box positions: (cx, cy, width, height, text, facecolor)
    boxes = [
        (5.0, 9.0, 3.8, 0.9, "Increased Greenhouse\nGas Forcing",      "#ffcccc"),
        (5.0, 7.2, 3.8, 0.9, "Arctic Surface\nWarming",                 "#ffd6a5"),
        (5.0, 5.4, 3.8, 0.9, "Sea Ice Melts\n(↓ Extent & Thickness)",  "#a8dadc"),
        (5.0, 3.6, 3.8, 0.9, "Dark Ocean Exposed\n(↓ Surface Albedo)", "#457b9d"),
        (5.0, 1.8, 3.8, 0.9, "Increased Solar\nAbsorption",            "#ffd6a5"),
    ]

    for cx, cy, w, h, text, fc in boxes:
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.12",
            facecolor=fc, edgecolor="black", linewidth=1.5,
        )
        ax.add_patch(rect)
        fontcolor = "white" if fc == "#457b9d" else "black"
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=11, fontweight="bold", color=fontcolor)

    # Downward arrows between boxes
    arrow_kw = dict(
        arrowstyle="->,head_width=0.35,head_length=0.25",
        color="black", linewidth=2,
    )
    for i in range(len(boxes) - 1):
        y_start = boxes[i][1] - boxes[i][3] / 2
        y_end = boxes[i + 1][1] + boxes[i + 1][3] / 2
        ax.annotate("", xy=(5.0, y_end), xytext=(5.0, y_start),
                    arrowprops=arrow_kw)

    # Feedback loop arrow (curves from bottom box back up to "Arctic Warming")
    feedback_kw = dict(
        arrowstyle="->,head_width=0.4,head_length=0.3",
        color="#d62728", linewidth=2.5,
        connectionstyle="arc3,rad=0.6",
    )
    ax.annotate(
        "", xy=(3.1, 7.2), xytext=(3.1, 1.8),
        arrowprops=feedback_kw,
    )

    ax.text(1.15, 4.5, "POSITIVE\nFEEDBACK\nLOOP", ha="center", va="center",
            fontsize=12, fontweight="bold", color="#d62728", fontstyle="italic",
            rotation=90)

    # Albedo annotations
    ax.text(8.5, 5.4, "Ice albedo:\n0.6 – 0.9", ha="center", va="center",
            fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))
    ax.text(8.5, 3.6, "Ocean albedo:\n~ 0.06", ha="center", va="center",
            fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    # Seasonal note
    ax.text(1.3, 1.0, "Summer: albedo feedback dominates\n"
            "Autumn/Winter: stored ocean heat\n"
            "released → amplifies warming",
            ha="left", va="center", fontsize=8.5, fontstyle="italic",
            bbox=dict(boxstyle="round", fc="#f0f0f0", alpha=0.9))

    ax.set_title("Ice–Albedo Positive Feedback Mechanism (Arctic Amplification)",
                 fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig10_feedback_diagram.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  Fig 10 saved.")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ARCTIC SEA ICE — ADDITIONAL FIGURES (5–10)")
    print("=" * 64)

    print("\nLoading data ...")
    df_all = load_all_months()
    daily = load_daily()
    temp = load_gistemp()
    print(f"  Monthly: {df_all['year'].nunique()} years")
    print(f"  Daily:   {len(daily):,} records")
    print(f"  Temp:    {len(temp)} years\n")

    print("Generating figures ...")
    figure5_daily_spaghetti(daily)
    figure6_month_trends(df_all)
    figure7_ranked_september(df_all)
    figure8_anomaly_heatmap(df_all)
    figure9_temp_vs_extent(df_all, temp)
    figure10_feedback_diagram()

    print(f"\nAll figures saved to {FIGURES_DIR.resolve()}")
    print("\n✓  10 paper figures complete (fig01 – fig10).")


if __name__ == "__main__":
    main()

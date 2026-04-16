"""
Step 7 — Advanced Predictive Models
=====================================
Addresses Prof. Heimbach's feedback that acceleration is not uniform
across the full 1979–present period.  Builds two improved models:

  1. Piecewise linear regression (breakpoint at 2007)
  2. Temperature-driven regression (Arctic temp anomaly as predictor)

Outputs:
  • fig_piecewise_projection.png   — Two-regime trend with projection
  • fig_temp_driven_projection.png — Temperature-based projection under scenarios
  • fig_model_comparison.png       — All three models side-by-side
  • advanced_model_summary.txt     — Key statistics

Requires: 01_download_data.py to have been run first.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from config import (
    BLUE_OCEAN_THRESHOLD_KM2,
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

BREAKPOINT_YEAR = 2007


# ── Data loaders ────────────────────────────────────────────────────────────

def load_september() -> pd.DataFrame:
    path = DATA_DIR / "N_09_extent_v4.0.csv"
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
    df = df[df["extent"] > 0]
    df["year"] = df["year"].astype(int)
    return df[df["year"] >= SATELLITE_ERA_START].sort_values("year").reset_index(drop=True)


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
    return df[df["Year"] >= SATELLITE_ERA_START].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 1 — Piecewise Linear Regression (breakpoint at 2007)
# ═══════════════════════════════════════════════════════════════════════════

def fit_piecewise(sept: pd.DataFrame) -> dict:
    """Fit two separate linear regressions: pre-2007 and post-2007."""
    pre = sept[sept["year"] < BREAKPOINT_YEAR]
    post = sept[sept["year"] >= BREAKPOINT_YEAR]

    # Pre-2007 model
    X_pre = pre["year"].values.reshape(-1, 1)
    y_pre = pre["extent"].values
    m_pre = LinearRegression().fit(X_pre, y_pre)
    sl_pre = m_pre.coef_[0]

    # Post-2007 model
    X_post = post["year"].values.reshape(-1, 1)
    y_post = post["extent"].values
    m_post = LinearRegression().fit(X_post, y_post)
    sl_post = m_post.coef_[0]

    # Combined fitted values for R²
    y_pred_all = np.concatenate([m_pre.predict(X_pre), m_post.predict(X_post)])
    y_all = np.concatenate([y_pre, y_post])

    # Blue-ocean year using the post-2007 slope
    bo_year = (BLUE_OCEAN_THRESHOLD_KM2 - m_post.intercept_) / sl_post

    # Confidence interval on the post-2007 slope
    n = len(y_post)
    res = y_post - m_post.predict(X_post)
    se = np.sqrt(np.sum(res ** 2) / (n - 2) / np.sum((X_post.ravel() - X_post.mean()) ** 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)

    return {
        "model_pre": m_pre,
        "model_post": m_post,
        "slope_pre": sl_pre,
        "slope_post": sl_post,
        "slope_post_ci": (sl_post - t_crit * se, sl_post + t_crit * se),
        "r2_combined": r2_score(y_all, y_pred_all),
        "r2_post": r2_score(y_post, m_post.predict(X_post)),
        "rmse_post": np.sqrt(mean_squared_error(y_post, m_post.predict(X_post))),
        "blue_ocean_year": bo_year,
        "n_pre": len(y_pre),
        "n_post": len(y_post),
    }


def plot_piecewise(sept: pd.DataFrame, pw: dict):
    """Piecewise regression with two-regime projection."""
    x_obs = sept["year"].values
    y_obs = sept["extent"].values

    # Pre-2007 trend line
    x_pre = np.arange(1979, BREAKPOINT_YEAR)
    y_pre = pw["model_pre"].predict(x_pre.reshape(-1, 1))

    # Post-2007 trend line extended to blue ocean
    bo_yr = int(np.ceil(pw["blue_ocean_year"])) + 5
    x_post = np.arange(BREAKPOINT_YEAR, bo_yr)
    y_post = pw["model_post"].predict(x_post.reshape(-1, 1))

    # Prediction band for post-2007 extrapolation
    post_data = sept[sept["year"] >= BREAKPOINT_YEAR]
    X_p = post_data["year"].values
    y_p = post_data["extent"].values
    n = len(y_p)
    x_mean = X_p.mean()
    ss_x = np.sum((X_p - x_mean) ** 2)
    s_e = np.sqrt(np.sum((y_p - pw["model_post"].predict(X_p.reshape(-1, 1))) ** 2) / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    pred_se = s_e * np.sqrt(1 + 1 / n + (x_post - x_mean) ** 2 / ss_x)
    upper = y_post.ravel() + t_crit * pred_se
    lower = y_post.ravel() - t_crit * pred_se

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(x_obs, y_obs, "o", color="#1f77b4", markersize=4.5, zorder=4,
            label="Observed September extent")

    ax.plot(x_pre, y_pre, "--", color="#2ca02c", linewidth=2,
            label=f"Pre-2007 trend: {pw['slope_pre'] * 10:+.2f} M km²/decade")
    ax.plot(x_post, y_post, "-", color="#d62728", linewidth=2.2,
            label=f"Post-2007 trend: {pw['slope_post'] * 10:+.2f} M km²/decade")

    ax.fill_between(x_post, lower, upper, alpha=0.12, color="#d62728",
                    label="95% prediction interval (post-2007)")

    ax.axhline(BLUE_OCEAN_THRESHOLD_KM2, color="#d62728", linestyle=":",
               linewidth=1.2)
    ax.axvline(BREAKPOINT_YEAR, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5)
    ax.text(BREAKPOINT_YEAR + 0.5, 7.8, "2007\nbreakpoint", fontsize=8,
            color="gray", va="top")

    bo_yr_round = int(round(pw["blue_ocean_year"]))
    ax.annotate(
        f"Projected ~{bo_yr_round}",
        xy=(pw["blue_ocean_year"], BLUE_OCEAN_THRESHOLD_KM2),
        xytext=(pw["blue_ocean_year"] - 8, BLUE_OCEAN_THRESHOLD_KM2 + 1.5),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        fontsize=10, color="#d62728", fontweight="bold",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Ice Extent (million km²)")
    ax.set_title("Piecewise Linear Regression: Two-Regime Projection")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xlim(1977, bo_yr_round + 5)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_piecewise_projection.png", dpi=200)
    plt.close(fig)

    print(f"  Piecewise fig saved.  Pre-2007: {pw['slope_pre'] * 10:+.3f}/decade, "
          f"Post-2007: {pw['slope_post'] * 10:+.3f}/decade, "
          f"Blue ocean ≈ {bo_yr_round}")


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 2 — Temperature-Driven Regression
# ═══════════════════════════════════════════════════════════════════════════

def fit_temp_model(sept: pd.DataFrame, temp: pd.DataFrame) -> dict:
    """Fit September extent as a function of Arctic temperature anomaly."""
    merged = pd.merge(
        temp[["Year", "arctic_anom"]],
        sept[["year", "extent"]],
        left_on="Year", right_on="year", how="inner",
    )
    X = merged["arctic_anom"].values.reshape(-1, 1)
    y = merged["extent"].values

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_

    # At what Arctic temp anomaly does extent = 1 M km²?
    temp_at_bo = (BLUE_OCEAN_THRESHOLD_KM2 - intercept) / slope

    # Confidence interval on slope
    n = len(y)
    res = y - y_pred
    se = np.sqrt(np.sum(res ** 2) / (n - 2) / np.sum((X.ravel() - X.mean()) ** 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)

    return {
        "model": model,
        "slope": slope,
        "intercept": intercept,
        "slope_ci": (slope - t_crit * se, slope + t_crit * se),
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "temp_at_blue_ocean": temp_at_bo,
        "merged": merged,
        "n": n,
    }


def plot_temp_projection(tm: dict, temp: pd.DataFrame):
    """
    Project ice loss under warming scenarios.
    Uses the observed Arctic warming rate to estimate when each
    temperature threshold is reached.
    """
    merged = tm["merged"]
    x_obs = merged["arctic_anom"].values
    y_obs = merged["extent"].values
    years = merged["Year"].values

    # Observed Arctic warming rate
    sl_temp, int_temp, *_ = stats.linregress(
        temp["Year"].values, temp["arctic_anom"].values
    )

    # Warming scenarios (Arctic temp anomaly per decade, rel. 1991-2020)
    scenarios = {
        "Current rate (+0.68°C/dec)": sl_temp,
        "Moderate acceleration (+0.85°C/dec)": 0.085,
        "High acceleration (+1.0°C/dec)": 0.10,
    }

    # Temperature range for regression line
    t_range = np.linspace(x_obs.min() - 0.3, tm["temp_at_blue_ocean"] + 0.5, 200)
    extent_pred = tm["model"].predict(t_range.reshape(-1, 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                     gridspec_kw={"width_ratios": [1.1, 1]})

    # ── Left panel: temperature-extent relationship ─────────────────────
    sc = ax1.scatter(x_obs, y_obs, c=years, cmap="viridis", s=45,
                     edgecolors="black", linewidths=0.3, zorder=3)
    ax1.plot(t_range, extent_pred, "--", color="#d62728", linewidth=1.8,
             label=f"OLS: {tm['slope']:+.2f} M km² / °C  (R²={tm['r2']:.3f})")
    ax1.axhline(BLUE_OCEAN_THRESHOLD_KM2, color="#d62728", linestyle=":",
                linewidth=1.2)
    ax1.axvline(tm["temp_at_blue_ocean"], color="#d62728", linestyle=":",
                linewidth=0.8, alpha=0.5)

    ax1.annotate(
        f"Blue ocean at\n{tm['temp_at_blue_ocean']:+.1f}°C Arctic anomaly",
        xy=(tm["temp_at_blue_ocean"], BLUE_OCEAN_THRESHOLD_KM2),
        xytext=(tm["temp_at_blue_ocean"] - 1.5, 2.5),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        fontsize=9, color="#d62728", fontweight="bold",
    )

    cbar = fig.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label("Year")
    ax1.set_xlabel("Arctic Temperature Anomaly (°C, rel. 1991–2020)")
    ax1.set_ylabel("September Extent (million km²)")
    ax1.set_title("Temperature–Extent Relationship")
    ax1.legend(fontsize=8.5, loc="upper right")

    # ── Right panel: timeline under warming scenarios ───────────────────
    colors_scen = ["#2ca02c", "#ff7f0e", "#d62728"]
    current_year = years.max()
    current_temp = temp[temp["Year"] == current_year]["arctic_anom"].values
    if len(current_temp) == 0:
        current_temp = x_obs[-1]
    else:
        current_temp = current_temp[0]

    for (label, rate), color in zip(scenarios.items(), colors_scen):
        future_years = np.arange(current_year, 2080)
        future_temps = current_temp + rate * (future_years - current_year)
        future_extent = tm["model"].predict(future_temps.reshape(-1, 1))

        ax2.plot(future_years, future_extent, "-", color=color, linewidth=2,
                 label=label)

        # Find blue-ocean crossing
        cross_idx = np.where(future_extent <= BLUE_OCEAN_THRESHOLD_KM2)[0]
        if len(cross_idx) > 0:
            cross_yr = future_years[cross_idx[0]]
            ax2.plot(cross_yr, BLUE_OCEAN_THRESHOLD_KM2, "v", color=color,
                     markersize=8, zorder=5)
            ax2.annotate(f"~{cross_yr}", xy=(cross_yr, BLUE_OCEAN_THRESHOLD_KM2),
                         xytext=(cross_yr, BLUE_OCEAN_THRESHOLD_KM2 + 0.6),
                         ha="center", fontsize=9, fontweight="bold", color=color)

    # Observed history
    ax2.plot(years, y_obs, "o-", color="#1f77b4", markersize=3, linewidth=1,
             alpha=0.6, label="Observed")

    ax2.axhline(BLUE_OCEAN_THRESHOLD_KM2, color="#d62728", linestyle=":",
                linewidth=1.2)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("September Extent (million km²)")
    ax2.set_title("Projections Under Warming Scenarios")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(1979, 2078)
    ax2.set_ylim(0, None)

    fig.suptitle("Temperature-Driven Model: Arctic Warming as Predictor of Ice Loss",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_temp_driven_projection.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)

    print(f"  Temp-driven fig saved.  Slope = {tm['slope']:+.3f} M km²/°C, "
          f"Blue ocean at {tm['temp_at_blue_ocean']:+.2f}°C Arctic anomaly")


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON — All three approaches side-by-side
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_comparison(sept: pd.DataFrame, pw: dict, tm: dict, temp: pd.DataFrame):
    """Three-panel comparison: full linear, piecewise, temperature-driven."""
    x = sept["year"].values
    y = sept["extent"].values

    # Full-record linear
    m_full = LinearRegression().fit(x.reshape(-1, 1), y)
    bo_full = (BLUE_OCEAN_THRESHOLD_KM2 - m_full.intercept_) / m_full.coef_[0]

    # Piecewise blue ocean
    bo_pw = pw["blue_ocean_year"]

    # Temperature-driven: blue ocean year at current warming rate
    sl_temp, *_ = stats.linregress(temp["Year"].values, temp["arctic_anom"].values)
    current_temp = temp["arctic_anom"].values[-1]
    current_year = temp["Year"].values[-1]
    years_to_bo = (tm["temp_at_blue_ocean"] - current_temp) / sl_temp
    bo_temp = current_year + years_to_bo

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(x, y, "o", color="#1f77b4", markersize=4.5, zorder=4,
            label="Observed")

    # Full-record linear
    x_ext = np.arange(1979, int(bo_full) + 5)
    y_full = m_full.predict(x_ext.reshape(-1, 1))
    ax.plot(x_ext, y_full, "--", color="#2ca02c", linewidth=1.8, alpha=0.7,
            label=f"Full-record linear → ~{int(round(bo_full))}")

    # Piecewise (post-2007 extrapolation)
    x_pw = np.arange(BREAKPOINT_YEAR, int(bo_pw) + 5)
    y_pw = pw["model_post"].predict(x_pw.reshape(-1, 1))
    ax.plot(x_pw, y_pw, "-", color="#d62728", linewidth=2,
            label=f"Post-2007 trend → ~{int(round(bo_pw))}")
    x_pw_pre = np.arange(1979, BREAKPOINT_YEAR)
    y_pw_pre = pw["model_pre"].predict(x_pw_pre.reshape(-1, 1))
    ax.plot(x_pw_pre, y_pw_pre, "-", color="#d62728", linewidth=1.2, alpha=0.4)

    # Temperature-driven (at current warming rate)
    future_yrs = np.arange(current_year, int(bo_temp) + 5)
    future_temps = current_temp + sl_temp * (future_yrs - current_year)
    future_ext = tm["model"].predict(future_temps.reshape(-1, 1))
    # Also compute historical fitted values
    merged = tm["merged"]
    hist_ext = tm["model"].predict(merged["arctic_anom"].values.reshape(-1, 1))
    ax.plot(merged["Year"].values, hist_ext, "-", color="#9467bd",
            linewidth=1, alpha=0.4)
    ax.plot(future_yrs, future_ext, "-", color="#9467bd", linewidth=2,
            label=f"Temp-driven (current rate) → ~{int(round(bo_temp))}")

    ax.axhline(BLUE_OCEAN_THRESHOLD_KM2, color="#d62728", linestyle=":",
               linewidth=1.2, label='"Blue ocean" threshold (1 M km²)')

    ax.set_xlabel("Year")
    ax.set_ylabel("September Extent (million km²)")
    ax.set_title("Model Comparison: Three Approaches to Projecting the Blue Ocean Event")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(1977, max(int(bo_full), int(bo_temp)) + 5)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_model_comparison.png", dpi=200)
    plt.close(fig)

    print(f"  Comparison fig saved.  Full: ~{int(round(bo_full))}, "
          f"Piecewise: ~{int(round(bo_pw))}, Temp-driven: ~{int(round(bo_temp))}")

    return bo_full, bo_pw, bo_temp


# ═══════════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════════

def write_summary(pw: dict, tm: dict, bo_full, bo_pw, bo_temp):
    lines = [
        "=" * 64,
        "  ADVANCED MODEL SUMMARY",
        "=" * 64,
        "",
        "  ── Piecewise Linear Regression (breakpoint: 2007) ──",
        f"  Pre-2007 slope:    {pw['slope_pre']:.5f} M km²/yr "
        f"({pw['slope_pre'] * 10:.3f}/decade)",
        f"  Post-2007 slope:   {pw['slope_post']:.5f} M km²/yr "
        f"({pw['slope_post'] * 10:.3f}/decade)",
        f"  95% CI (post):     [{pw['slope_post_ci'][0]:.5f}, "
        f"{pw['slope_post_ci'][1]:.5f}]",
        f"  Post-2007 R²:      {pw['r2_post']:.4f}",
        f"  Combined R²:       {pw['r2_combined']:.4f}",
        f"  Blue ocean:        ~{int(round(pw['blue_ocean_year']))}",
        f"  N (pre / post):    {pw['n_pre']} / {pw['n_post']}",
        "",
        "  ── Temperature-Driven Model ──",
        f"  Slope:             {tm['slope']:+.4f} M km² per °C Arctic warming",
        f"  95% CI:            [{tm['slope_ci'][0]:+.4f}, {tm['slope_ci'][1]:+.4f}]",
        f"  R²:                {tm['r2']:.4f}",
        f"  RMSE:              {tm['rmse']:.4f} M km²",
        f"  Blue ocean at:     {tm['temp_at_blue_ocean']:+.2f}°C Arctic anomaly",
        f"  → at current rate: ~{int(round(bo_temp))}",
        "",
        "  ── Model Comparison (Blue Ocean Year) ──",
        f"  Full-record linear:      ~{int(round(bo_full))}  (conservative lower bound)",
        f"  Piecewise (post-2007):   ~{int(round(bo_pw))}  (regime-aware)",
        f"  Temperature-driven:      ~{int(round(bo_temp))}  (physically motivated)",
        f"  Literature (CMIP6):      ~2040–2050 (Bonan et al., 2023)",
        "",
        "  The post-2007 slope is steeper than the full-record slope,",
        "  confirming Prof. Heimbach's observation that the decline rate",
        "  is not uniform.  The temperature-driven model provides the most",
        "  physically grounded projection because it is linked to the",
        "  actual driver (Arctic warming) rather than calendar time.",
        "=" * 64,
    ]

    text = "\n".join(lines)
    (FIGURES_DIR / "advanced_model_summary.txt").write_text(text)
    print(f"\n{text}\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ARCTIC SEA ICE — ADVANCED PREDICTIVE MODELS")
    print("=" * 64)

    print("\nLoading data ...")
    sept = load_september()
    temp = load_gistemp()
    print(f"  September: {len(sept)} years ({sept['year'].min()}–{sept['year'].max()})")
    print(f"  Temperature: {len(temp)} years\n")

    print("Fitting piecewise linear model (breakpoint = 2007) ...")
    pw = fit_piecewise(sept)
    plot_piecewise(sept, pw)

    print("\nFitting temperature-driven model ...")
    tm = fit_temp_model(sept, temp)
    plot_temp_projection(tm, temp)

    print("\nGenerating model comparison ...")
    bo_full, bo_pw, bo_temp = plot_model_comparison(sept, pw, tm, temp)

    write_summary(pw, tm, bo_full, bo_pw, bo_temp)

    print(f"All outputs saved to {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()

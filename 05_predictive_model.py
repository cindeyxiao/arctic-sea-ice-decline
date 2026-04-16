"""
Step 5 — Predictive Modeling: Blue Ocean Event Projection
===========================================================
Produces the paper's **Figure 3**: linear regression extrapolation of
September sea ice extent to the 1 million km² "blue ocean" threshold.

Outputs:
  • fig03_blue_ocean_projection.png  — Publication-quality Matplotlib figure
  • fig03_blue_ocean_projection.html — Interactive Plotly version
  • fig_S6_residual_diagnostics.png  — Residual analysis of the linear model
  • fig_S7_quadratic_comparison.png  — Linear vs quadratic extrapolation
  • model_summary.txt               — Key statistics printed to file

Uses scikit-learn LinearRegression as specified in the paper.

Requires: 01_download_data.py to have been run first.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from config import (
    BASELINE_PERIOD,
    BLUE_OCEAN_THRESHOLD_KM2,
    DATA_DIR,
    FIGURES_DIR,
    PLOT_STYLE,
    SATELLITE_ERA_START,
)

plt.rcParams.update(PLOT_STYLE)


# ── Data loading ────────────────────────────────────────────────────────────

def load_september() -> pd.DataFrame:
    path = DATA_DIR / "N_09_extent_v4.0.csv"
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
    df.rename(columns=rename, inplace=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["extent"] = pd.to_numeric(df["extent"], errors="coerce")
    df.dropna(subset=["year", "extent"], inplace=True)
    df = df[df["extent"] > 0]  # NSIDC uses -9999 as missing-data flag
    df["year"] = df["year"].astype(int)
    return df[df["year"] >= SATELLITE_ERA_START].sort_values("year").reset_index(drop=True)


# ── Model fitting ───────────────────────────────────────────────────────────

def fit_linear_model(sept: pd.DataFrame) -> dict:
    """Fit scikit-learn LinearRegression and compute diagnostics."""
    X = sept["year"].values.reshape(-1, 1)
    y = sept["extent"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Blue-ocean year:  slope * year + intercept = threshold
    blue_ocean_year = (BLUE_OCEAN_THRESHOLD_KM2 - intercept) / slope

    # Confidence interval on slope via scipy (equivalent OLS)
    n = len(y)
    residuals = y - y_pred
    se_slope = np.sqrt(
        np.sum(residuals ** 2) / (n - 2)
        / np.sum((X.ravel() - X.mean()) ** 2)
    )
    t_crit = stats.t.ppf(0.975, df=n - 2)
    slope_ci = (slope - t_crit * se_slope, slope + t_crit * se_slope)

    # Blue-ocean year uncertainty from slope CI
    bo_early = (BLUE_OCEAN_THRESHOLD_KM2 - intercept) / slope_ci[0]
    bo_late  = (BLUE_OCEAN_THRESHOLD_KM2 - intercept) / slope_ci[1]
    bo_range = (min(bo_early, bo_late), max(bo_early, bo_late))

    return {
        "model": model,
        "slope": slope,
        "intercept": intercept,
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "slope_ci_95": slope_ci,
        "se_slope": se_slope,
        "blue_ocean_year": blue_ocean_year,
        "blue_ocean_range": bo_range,
        "n": n,
        "residuals": residuals,
        "y_pred": y_pred,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Blue Ocean Event Projection
# ═══════════════════════════════════════════════════════════════════════════

def figure3_projection(sept: pd.DataFrame, result: dict):
    """Paper Figure 3: linear extrapolation with blue-ocean threshold."""
    x_obs = sept["year"].values
    y_obs = sept["extent"].values

    # Extrapolation range
    x_future = np.arange(x_obs.min(), int(result["blue_ocean_year"]) + 10)
    y_future = result["slope"] * x_future + result["intercept"]

    # 95% prediction band for extrapolation
    n = result["n"]
    x_mean = x_obs.mean()
    ss_x = np.sum((x_obs - x_mean) ** 2)
    s_e = np.sqrt(np.sum(result["residuals"] ** 2) / (n - 2))

    pred_se = s_e * np.sqrt(
        1 + 1 / n + (x_future - x_mean) ** 2 / ss_x
    )
    t_crit = stats.t.ppf(0.975, df=n - 2)
    upper = y_future + t_crit * pred_se
    lower = y_future - t_crit * pred_se

    bo_yr = int(round(result["blue_ocean_year"]))

    # ── Matplotlib ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(x_obs, y_obs, "o", color="#1f77b4", markersize=4, zorder=4,
            label="Observed September extent")
    ax.plot(x_future, y_future, "-", color="#2ca02c", linewidth=2,
            label=f"Linear trend ({result['slope']:.4f} M km²/yr)")

    ax.fill_between(x_future, lower, upper, alpha=0.12, color="#2ca02c",
                    label="95% prediction interval")

    ax.axhline(BLUE_OCEAN_THRESHOLD_KM2, color="#d62728", linestyle=":",
               linewidth=1.5, label='"Blue ocean" threshold (1 M km²)')

    ax.axvline(result["blue_ocean_year"], color="#d62728", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.annotate(
        f"Projected ~{bo_yr}",
        xy=(result["blue_ocean_year"], BLUE_OCEAN_THRESHOLD_KM2),
        xytext=(result["blue_ocean_year"] - 15, BLUE_OCEAN_THRESHOLD_KM2 + 1.5),
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        fontsize=10, color="#d62728", fontweight="bold",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Sea Ice Extent (million km²)")
    ax.set_title("Linear Regression Projection of September Sea Ice Extent")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(x_obs.min() - 2, int(result["blue_ocean_year"]) + 8)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig03_blue_ocean_projection.png", dpi=200)
    plt.close(fig)

    # ── Plotly ──────────────────────────────────────────────────────────
    pfig = go.Figure()
    pfig.add_trace(go.Scatter(
        x=x_obs.tolist(), y=y_obs.tolist(),
        mode="markers", name="Observed",
        marker=dict(size=5, color="#1f77b4"),
    ))
    pfig.add_trace(go.Scatter(
        x=x_future.tolist(), y=y_future.tolist(),
        mode="lines", name="Linear Trend",
        line=dict(color="#2ca02c", width=2),
    ))
    pfig.add_trace(go.Scatter(
        x=np.concatenate([x_future, x_future[::-1]]).tolist(),
        y=np.concatenate([upper, lower[::-1]]).tolist(),
        fill="toself", fillcolor="rgba(44,160,44,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% Prediction Interval",
    ))
    pfig.add_hline(
        y=BLUE_OCEAN_THRESHOLD_KM2, line_dash="dot", line_color="#d62728",
        annotation_text=f"Blue Ocean Threshold — projected ~{bo_yr}",
    )
    pfig.update_layout(
        title="Linear Regression Projection: Blue Ocean Event",
        xaxis_title="Year",
        yaxis_title="September Extent (million km²)",
        template="plotly_white", hovermode="x unified",
    )
    pfig.write_html(FIGURES_DIR / "fig03_blue_ocean_projection.html")

    print(f"  Fig 03 saved.  Blue ocean projected ≈ {bo_yr} "
          f"(95% CI: {int(result['blue_ocean_range'][0])}–"
          f"{int(result['blue_ocean_range'][1])})")


# ═══════════════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY: Residual diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def fig_s6_residuals(sept: pd.DataFrame, result: dict):
    """Residual analysis to evaluate linear model assumptions."""
    x = sept["year"].values
    res = result["residuals"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Residuals vs year
    axes[0].scatter(x, res, s=20, color="#1f77b4", edgecolors="black",
                    linewidths=0.3)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Residual (M km²)")
    axes[0].set_title("Residuals vs Year")

    # Histogram
    axes[1].hist(res, bins=15, color="#1f77b4", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residual (M km²)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    # Q-Q plot
    (osm, osr), (sl, ic, _) = stats.probplot(res, dist="norm")
    axes[2].scatter(osm, osr, s=20, color="#1f77b4", edgecolors="black",
                    linewidths=0.3)
    axes[2].plot(osm, sl * np.array(osm) + ic, "r--", linewidth=1.2)
    axes[2].set_xlabel("Theoretical Quantiles")
    axes[2].set_ylabel("Ordered Residuals")
    axes[2].set_title("Normal Q-Q Plot")

    fig.suptitle("Linear Model Residual Diagnostics", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_S6_residual_diagnostics.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("  Fig S6 saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY: Linear vs quadratic comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_s7_quadratic(sept: pd.DataFrame, lin_result: dict):
    """Compare linear vs quadratic extrapolation to illustrate acceleration."""
    x = sept["year"].values.astype(float)
    y = sept["extent"].values

    # Quadratic fit
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)

    x_ext = np.arange(x.min(), 2070)
    y_lin = lin_result["slope"] * x_ext + lin_result["intercept"]
    y_quad = poly(x_ext)

    # Find quadratic blue-ocean crossing
    roots = np.roots([coeffs[0], coeffs[1], coeffs[2] - BLUE_OCEAN_THRESHOLD_KM2])
    real_roots = [r.real for r in roots if np.isreal(r) and r.real > x.max()]
    quad_bo = min(real_roots) if real_roots else None

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, y, "o", color="#1f77b4", markersize=4, label="Observed")
    ax.plot(x_ext, y_lin, "--", color="#2ca02c", linewidth=1.8,
            label=f"Linear → ~{int(round(lin_result['blue_ocean_year']))}")
    ax.plot(x_ext, y_quad, "-.", color="#9467bd", linewidth=1.8,
            label=f"Quadratic → ~{int(round(quad_bo)) if quad_bo else '?'}")

    ax.axhline(BLUE_OCEAN_THRESHOLD_KM2, color="#d62728", linestyle=":",
               linewidth=1.2)
    ax.set_xlabel("Year")
    ax.set_ylabel("September Extent (million km²)")
    ax.set_title("Linear vs Quadratic Extrapolation of September Ice Extent")
    ax.legend(fontsize=9)
    ax.set_xlim(x.min() - 2, 2070)
    ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_S7_quadratic_comparison.png", dpi=200)
    plt.close(fig)

    if quad_bo:
        print(f"  Fig S7 saved.  Quadratic blue ocean ≈ {int(round(quad_bo))}")
    else:
        print("  Fig S7 saved.  Quadratic does not cross threshold in range.")


# ═══════════════════════════════════════════════════════════════════════════
#  Model summary text file
# ═══════════════════════════════════════════════════════════════════════════

def write_summary(sept: pd.DataFrame, result: dict):
    """Write key model statistics to a text file."""
    lines = [
        "=" * 60,
        "  LINEAR REGRESSION MODEL SUMMARY",
        "=" * 60,
        "",
        f"  Training period:  {sept['year'].min()} – {sept['year'].max()}",
        f"  Observations:     {result['n']}",
        f"  Predictor:        Year (sole variable)",
        f"  Response:         September minimum extent (M km²)",
        "",
        f"  Slope:            {result['slope']:.6f} M km²/yr",
        f"                    ({result['slope'] * 10:.4f} M km²/decade)",
        f"                    ({result['slope'] * 1e6:.0f} km²/yr)",
        f"  95% CI on slope:  [{result['slope_ci_95'][0]:.6f}, "
        f"{result['slope_ci_95'][1]:.6f}]",
        f"  Intercept:        {result['intercept']:.4f}",
        "",
        f"  R²:               {result['r2']:.4f}",
        f"  RMSE:             {result['rmse']:.4f} M km²",
        "",
        f"  Blue ocean year:  ≈ {int(round(result['blue_ocean_year']))}",
        f"  95% range:        {int(result['blue_ocean_range'][0])} – "
        f"{int(result['blue_ocean_range'][1])}",
        "",
        "  Note: This baseline linear model is intentionally conservative.",
        "  It does not capture the documented acceleration in loss rates.",
        "  Published observationally-constrained CMIP6 projections place",
        "  the first ice-free September before 2050 (Bonan et al., 2023).",
        "=" * 60,
    ]

    text = "\n".join(lines)
    (FIGURES_DIR / "model_summary.txt").write_text(text)
    print(f"\n{text}\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  ARCTIC SEA ICE — PREDICTIVE MODELING")
    print("=" * 64)

    print("\nLoading September extent data ...")
    sept = load_september()
    print(f"  {len(sept)} years loaded ({sept['year'].min()}–{sept['year'].max()})\n")

    print("Fitting scikit-learn LinearRegression ...")
    result = fit_linear_model(sept)

    print("Generating figures ...")
    figure3_projection(sept, result)
    fig_s6_residuals(sept, result)
    fig_s7_quadratic(sept, result)

    write_summary(sept, result)

    print(f"All outputs saved to {FIGURES_DIR.resolve()}")
    print("\n✓  All analysis scripts complete. See README.md for overview.")


if __name__ == "__main__":
    main()

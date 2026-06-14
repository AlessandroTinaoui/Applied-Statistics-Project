"""
OLS Linear Regression with diagnostics — sklearn Pipeline style.

Implements:
  - Base model: Pipeline(preprocessor, LinearRegression)
  - Diagnostic suite (residual plots, Q-Q, VIF, Shapiro-Wilk)
  - Cook's distance outlier detection & removal
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .data_preparation import TARGET, get_feature_names_after_preprocessing


# ──────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────
def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "OLS",
) -> dict:
    """Evaluate a fitted Pipeline on train and test sets."""
    metrics = {
        "model": model_name,
        "R2_train": r2_score(y_train, pipeline.predict(X_train)),
        "R2_test":  r2_score(y_test,  pipeline.predict(X_test)),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test))),
        "MAE_test":  mean_absolute_error(y_test, pipeline.predict(X_test)),
    }
    print(f"\n[{model_name}] Performance Metrics:")
    print("  " + "─" * 40)
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<20} : {v:.6f}")
    print("  " + "─" * 40)
    return metrics


# ──────────────────────────────────────────────────────────────────────
#  Coefficient table
# ──────────────────────────────────────────────────────────────────────
def print_coefficient_table(
    pipeline: Pipeline,
    feature_names: list[str],
    model_name: str = "OLS",
) -> pd.DataFrame:
    """Print and return a DataFrame of model coefficients sorted by |coef|."""
    model = pipeline.named_steps["model"]
    coef_df = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
        .assign(abs_coef=lambda d: d["Coefficient"].abs())
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n[{model_name}] Coefficients (Intercept: {model.intercept_:.6f}):")
    print("  " + "─" * 60)
    print(f"  {'Feature':<40} {'Coefficient':>15}")
    print("  " + "─" * 60)
    for _, row in coef_df.iterrows():
        print(f"  {row['Feature']:<40} {row['Coefficient']:>15.6f}")
    print("  " + "─" * 60)
    return coef_df


# ──────────────────────────────────────────────────────────────────────
#  VIF
# ──────────────────────────────────────────────────────────────────────
def compute_vif(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    feature_names: list[str],
    model_name: str = "OLS",
) -> pd.DataFrame:
    """Compute VIF from the preprocessed design matrix."""
    X = pd.DataFrame(
        pipeline.named_steps["preprocessor"].transform(X_train),
        columns=feature_names,
    )
    vif_data = (
        pd.DataFrame({
            "Feature": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        })
        .sort_values("VIF", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n[{model_name}] Variance Inflation Factors (VIF):")
    print("  " + "─" * 60)
    print(f"  {'Feature':<40} {'VIF':>15}")
    print("  " + "─" * 60)
    for _, row in vif_data.iterrows():
        flag = " ⚠️" if row["VIF"] > 10 else ""
        print(f"  {row['Feature']:<40} {row['VIF']:>15.2f}{flag}")
    print("  " + "─" * 60)
    high = vif_data[vif_data["VIF"] > 10]
    if not high.empty:
        print(f"  ⚠️  {len(high)} features with VIF > 10 (multicollinearity concern)")
    else:
        print("  ✓ All VIF values ≤ 10 — no multicollinearity concerns")
    return vif_data


# ──────────────────────────────────────────────────────────────────────
#  Diagnostics
# ──────────────────────────────────────────────────────────────────────
def plot_diagnostics(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "OLS",
    save_dir: Path | None = None,
    max_plot_points: int = 10_000,
):
    """
    Generate the 4 standard diagnostic plots:
      1. Residuals vs Fitted   2. Normal Q-Q
      3. Scale-Location        4. Residual histogram
    Also runs the Shapiro-Wilk normality test.
    """
    y_pred = pipeline.predict(X_train)
    residuals = y_train.values - y_pred
    n = len(residuals)

    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=min(n, max_plot_points), replace=False)
    idx.sort()

    y_pred_s = y_pred[idx]
    resid_s  = residuals[idx]
    std_resid_s = (residuals / np.sqrt(np.mean(residuals ** 2)))[idx]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Diagnostic Plots — {model_name} (n={n}, plotted={len(idx)})",
        fontsize=15, fontweight="bold",
    )

    # 1) Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_pred_s, resid_s, alpha=0.15, s=6, color="steelblue")
    ax.axhline(0, color="red", ls="--", lw=1)
    lw = sm.nonparametric.lowess(resid_s, y_pred_s, frac=0.3)
    ax.plot(lw[:, 0], lw[:, 1], color="red", lw=2, alpha=0.8)
    ax.set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs Fitted")

    # 2) Q-Q plot
    ax = axes[0, 1]
    stats.probplot(resid_s, dist="norm", plot=ax)
    ax.set_title("Normal Q-Q Plot")
    ax.get_lines()[0].set(markersize=3, alpha=0.3)

    # 3) Scale-Location
    ax = axes[1, 0]
    sqrt_std = np.sqrt(np.abs(std_resid_s))
    ax.scatter(y_pred_s, sqrt_std, alpha=0.15, s=6, color="steelblue")
    lw2 = sm.nonparametric.lowess(sqrt_std, y_pred_s, frac=0.3)
    ax.plot(lw2[:, 0], lw2[:, 1], color="red", lw=2, alpha=0.8)
    ax.set(xlabel="Fitted values", ylabel="√|Standardised Residuals|", title="Scale-Location")

    # 4) Residual histogram
    ax = axes[1, 1]
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set(xlabel="Residuals", ylabel="Count", title="Residual Distribution")

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / f"diagnostics_{model_name.lower().replace(' ', '_')}.png")
    plt.close(fig)

    # Shapiro-Wilk (subsample to max 5000)
    n_sw = min(5000, n)
    stat, p = stats.shapiro(rng.choice(residuals, size=n_sw, replace=False))
    print(f"\n[{model_name}] Residual Normality (Shapiro-Wilk test):")
    print("  " + "─" * 50)
    print(f"  Sample size (n) : {n_sw}")
    print(f"  Statistic       : {stat:.6f}")
    print(f"  p-value         : {p:.2e}")
    print("  " + "─" * 50)
    verdict = "NOT normally distributed (Normality rejected at α=0.05)" if p < 0.05 \
              else "normally distributed (Cannot reject normality at α=0.05)"
    print(f"  → Residuals are {verdict}")


# ──────────────────────────────────────────────────────────────────────
#  Outlier Detection & Removal
# ──────────────────────────────────────────────────────────────────────
def detect_and_remove_outliers(
    data: dict,
    save_dir: Path | None = None,
    cooks_threshold_multiplier: float = 4.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fit statsmodels OLS, compute Cook's distance, and remove observations
    where D_i > cooks_threshold_multiplier / N.

    Saves influence plots (before/after) and a Q-Q comparison plot.
    Prints a normality summary comparing baseline vs cleaned model.

    Returns
    -------
    X_train_clean, y_train_clean : filtered training data (original feature space)
    """
    X_train = data["X_train"]
    y_train = data["y_train"]

    # Preprocess training data for statsmodels
    prep = clone(data["preprocessor"])
    X_trans = prep.fit_transform(X_train)
    feature_names = get_feature_names_after_preprocessing(prep, data["numeric_features"])
    X_sm = sm.add_constant(pd.DataFrame(X_trans, columns=feature_names, index=X_train.index))

    # Fit baseline OLS
    results = sm.OLS(y_train, X_sm).fit()

    # Shapiro-Wilk on baseline residuals
    rng = np.random.default_rng(42)
    stat_base, p_base = stats.shapiro(rng.choice(results.resid, size=min(5000, len(results.resid)), replace=False))

    # Cook's distance filter
    cooks_d, _ = results.get_influence().cooks_distance
    N = len(y_train)
    clean_mask = cooks_d <= cooks_threshold_multiplier / N
    n_excluded = (~clean_mask).sum()

    X_clean = X_sm[clean_mask]
    y_clean = y_train[clean_mask]
    results_clean = sm.OLS(y_clean, X_clean).fit()
    stat_clean, p_clean = stats.shapiro(rng.choice(results_clean.resid, size=min(5000, len(results_clean.resid)), replace=False))

    # Print summary
    label = f"Cook's D > {cooks_threshold_multiplier}/N"
    print("\n[Outliers] Normality & Outlier Filtering Summary:")
    print("  " + "─" * 75)
    print(f"  {'Metric':<25} {'Baseline Model':<20} {f'Cleaned Model ({label})':<30}")
    print("  " + "─" * 75)
    print(f"  {'Observations':<25} {N:<20} {clean_mask.sum():<30}")
    print(f"  {'Outliers Excluded':<25} {'-':<20} {f'{n_excluded} ({n_excluded/N*100:.2f}%)':<30}")
    print(f"  {'Shapiro-Wilk Stat':<25} {stat_base:<20.6f} {stat_clean:<30.6f}")
    print(f"  {'Shapiro-Wilk p-value':<25} {p_base:<20.2e} {p_clean:<30.2e}")
    print("  " + "─" * 75)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Influence plot (baseline)
        fig, ax = plt.subplots(figsize=(10, 10))
        sm.graphics.influence_plot(results, criterion="cooks", ax=ax)
        plt.tight_layout()
        path = save_dir / "influence_plot.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"\n[Outliers] Influence plot saved to: {path}")

        # Influence plot (cleaned)
        fig, ax = plt.subplots(figsize=(10, 10))
        sm.graphics.influence_plot(results_clean, criterion="cooks", ax=ax)
        plt.tight_layout()
        path = save_dir / "influence_plot_clean.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"[Outliers] Cleaned influence plot saved to: {path}")

        # Q-Q comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, resid, title, color in [
            (axes[0], results.resid,       f"Baseline Q-Q\n(Shapiro p-val: {p_base:.2e})",  None),
            (axes[1], results_clean.resid, f"Clean Q-Q\n(Shapiro p-val: {p_clean:.2e})",    "seagreen"),
        ]:
            stats.probplot(resid, dist="norm", plot=ax)
            ax.set_title(title)
            line_kws = dict(markersize=3, alpha=0.3)
            if color:
                line_kws["color"] = color
            ax.get_lines()[0].set(**line_kws)
        plt.tight_layout()
        path = save_dir / "qq_comparison.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"[Outliers] Q-Q comparison plot saved to: {path}")

    return X_train[clean_mask], y_train[clean_mask]


# ──────────────────────────────────────────────────────────────────────
#  Full pipeline
# ──────────────────────────────────────────────────────────────────────
def run_linear_models(data: dict, save_dir: Path | None = None) -> dict:
    """
    Run the full OLS regression pipeline:
      1. Fit Pipeline(preprocessor, LinearRegression)
      2. Evaluate metrics (train + test)
      3. Print coefficient table
      4. Compute VIF
      5. Generate diagnostic plots + Shapiro-Wilk test

    Parameters
    ----------
    data     : dict from prepare_data()
    save_dir : optional path to save plots

    Returns
    -------
    dict with fitted pipeline, metrics, feature names, and split data
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    pipe = Pipeline([
        ("preprocessor", clone(data["preprocessor"])),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)

    feature_names = get_feature_names_after_preprocessing(
        pipe.named_steps["preprocessor"], data["numeric_features"]
    )

    base_metrics = evaluate_model(pipe, X_train, y_train, X_test, y_test, "Base OLS")
    print_coefficient_table(pipe, feature_names, "Base OLS")
    compute_vif(pipe, X_train, feature_names, "Base OLS")
    plot_diagnostics(pipe, X_train, y_train, "Base OLS", save_dir)

    return {
        "base_pipeline":    pipe,
        "base_metrics":     base_metrics,
        "base_feature_names": feature_names,
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }

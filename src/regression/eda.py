"""
Exploratory Data Analysis and Feature Selection for qubit snapshot regression.

Generates summary statistics, correlation heatmaps, binned profiles,
and plots to understand the target (T1) and its relationship with features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .data_preparation import TARGET, ID_COLUMNS, CATEGORICAL_FEATURES, get_numeric_features

pd.options.display.float_format = "{:.4f}".format


def run_feature_selection(df: pd.DataFrame, save_dir: Path) -> pd.DataFrame:
    """
    Calculate Pearson correlation of all numeric features with the target,
    print the sorted correlation table, and generate bivariate binned profile plots.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    exclude = set(ID_COLUMNS + [TARGET] + CATEGORICAL_FEATURES)
    candidate_features = [c for c in df.columns if c not in exclude]

    corrs = []
    for col in candidate_features:
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        valid = df[[col, TARGET]].dropna()
        if len(valid) < 10:
            continue
        r = valid[col].corr(valid[TARGET])
        corrs.append({
            "feature": col,
            "pearson_r": r,
            "abs_pearson_r": abs(r),
            "missing_pct": df[col].isna().mean() * 100,
        })

    corrs_df = pd.DataFrame(corrs).sort_values("abs_pearson_r", ascending=False)

    print("\n[Feature Selection] Top features correlated with T1 (|Pearson r|):")
    print("  " + "─" * 65)
    print(corrs_df.head(15).to_string(index=False))
    print("  " + "─" * 65)

    # Binned profile plots for key features
    features_to_plot = ["T2_last_obs", "T1_prev", "sx_error_class_last_obs", "readout_error_last_obs"]
    features_to_plot = [f for f in features_to_plot if f in df.columns]

    plot_sample = df.sample(n=min(5000, len(df)), random_state=42)

    print(f"  Generating {len(features_to_plot)} binned profile plots in: {save_dir}...")
    for i, col in enumerate(features_to_plot):
        print(f"    [{i+1}/{len(features_to_plot)}] Plotting T1 vs {col}...")
        fig, ax = plt.subplots(figsize=(6, 5))
        if col == "sx_error_class_last_obs":
            order = [o for o in ["low_error", "high_or_failed"]
                     if o in plot_sample[col].dropna().unique()]
            sns.boxplot(
                data=plot_sample, x=col, y=TARGET, ax=ax, order=order,
                patch_artist=True, boxprops=dict(facecolor="steelblue", alpha=0.6),
            )
            ax.set_title(f"{TARGET} vs {col}", fontsize=13, fontweight="bold")
        else:
            # Binned profile: 20 bins of means + 95% CI + linear fit
            sns.regplot(
                data=plot_sample, x=col, y=TARGET, x_bins=20,
                color="steelblue", ax=ax, line_kws={"color": "red", "lw": 2},
            )
            ax.set_title(f"Binned Profile: {TARGET} vs {col}", fontsize=13, fontweight="bold")
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel(f"{TARGET} (μs)", fontsize=11)
        plt.tight_layout()
        fig.savefig(save_dir / f"t1_vs_{col}.png", dpi=150)
        plt.close(fig)

    return corrs_df


# ──────────────────────────────────────────────────────────────────────
#  EDA
# ──────────────────────────────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame, save_dir: Path | None = None):
    """Plot histogram and boxplot of T1 and log(T1), printing basic statistics."""
    y = df[TARGET].dropna()
    skew = y.skew()
    log_y = np.log(y[y > 0])
    log_skew = log_y.skew()

    print(f"\n[EDA] Target '{TARGET}' statistics:")
    for label, val in [("Mean", y.mean()), ("Median", y.median()), ("Std", y.std()),
                       ("Skewness", skew), ("Kurtosis", y.kurtosis()),
                       ("Min", y.min()), ("Max", y.max())]:
        print(f"  {label:<10}: {val:.4f}")
    if abs(skew) > 1.0:
        print(f"  → T1 is highly skewed ({skew:.2f}). Log transform recommended (log skew = {log_skew:.2f}).")
    else:
        print(f"  → Skewness is moderate ({skew:.2f}). Log transform optional.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Target Distribution: {TARGET}", fontsize=15, fontweight="bold")

    for ax, data, color, title, xlabel in [
        (axes[0, 0], y,     "steelblue", "Histogram (raw)",           TARGET),
        (axes[1, 0], log_y, "seagreen",  f"Histogram log({TARGET})",  f"log({TARGET})"),
    ]:
        ax.hist(data, bins=80, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="red", ls="--", label=f"Mean = {data.mean():.3f}")
        ax.axvline(data.median(), color="orange", ls="--", label=f"Median = {data.median():.3f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=9)

    for ax, data, color, title, ylabel in [
        (axes[0, 1], y,     "steelblue", "Boxplot (raw)",           TARGET),
        (axes[1, 1], log_y, "seagreen",  f"Boxplot log({TARGET})",  f"log({TARGET})"),
    ]:
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.6))
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "eda_target_distribution.png")
    plt.close(fig)


def plot_target_by_backend(df: pd.DataFrame, save_dir: Path | None = None):
    """Boxplot of T1 grouped by backend."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column=TARGET, by="backend", ax=ax, patch_artist=True, showfliers=False)
    ax.set_title(f"{TARGET} by Backend", fontsize=13, fontweight="bold")
    ax.set_xlabel("Backend")
    ax.set_ylabel(TARGET)
    fig.suptitle("")
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "eda_target_by_backend.png")
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, save_dir: Path | None = None):
    """Pearson correlation heatmap (printing of the table is handled in feature selection)."""
    numeric = get_numeric_features(df) + [TARGET]
    corr = df[numeric].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0)
    plt.title("Pearson Correlation Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "eda_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_eda(df: pd.DataFrame, save_dir: Path | None = None):
    """Run Target Distribution, Backend analysis, and Correlation Heatmap steps."""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    run_feature_selection(df, save_dir)

    plot_target_distribution(df, save_dir)
    plot_target_by_backend(df, save_dir)
    plot_correlation_matrix(df, save_dir)

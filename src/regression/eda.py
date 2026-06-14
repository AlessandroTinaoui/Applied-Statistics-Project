"""
Exploratory Data Analysis for qubit snapshot regression.

Generates summary statistics and plots to understand the target (T1)
and its relationship with the selected features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .data_preparation import TARGET, ID_COLUMNS, CATEGORICAL_FEATURES, get_numeric_features

pd.options.display.float_format = "{:.4f}".format


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


def plot_scatter_vs_features(df: pd.DataFrame, save_dir: Path | None = None):
    """Grid of scatter plots: T1 vs each numeric feature (with linear trend line)."""
    numeric = get_numeric_features(df)
    n = min(5_000, len(df))
    sample = df.sample(n=n, random_state=42)

    ncols = 3
    nrows = (len(numeric) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric):
        ax = axes[i]
        ax.scatter(sample[col], sample[TARGET], alpha=0.15, s=8, color="steelblue")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel(TARGET, fontsize=9)
        ax.set_title(col, fontsize=10)
        valid = sample[[col, TARGET]].dropna()
        if len(valid) > 10:
            z = np.polyfit(valid[col], valid[TARGET], 1)
            x_s = np.sort(valid[col])
            ax.plot(x_s, np.polyval(z, x_s), color="red", lw=1.5, alpha=0.7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Scatter: {TARGET} vs numeric features (n={n} sample)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "eda_scatter_features.png")
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, save_dir: Path | None = None):
    """Pearson correlation heatmap and target correlation ranking."""
    numeric = get_numeric_features(df) + [TARGET]
    corr = df[numeric].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0)
    plt.title("Pearson Correlation Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "eda_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[EDA] Pearson Correlations with {TARGET}:")
    print(corr[TARGET].drop(TARGET).sort_values(ascending=False).to_string())


def print_missingness(df: pd.DataFrame):
    """Print missing-value rates for all columns with at least one NaN."""
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    print(f"\n[EDA] Missing values (features with > 0% NaN):")
    if miss.empty:
        print("  None — all features are fully observed.")
    else:
        for col, rate in miss.items():
            print(f"  {col:40s}  {rate:6.2%}")


def run_eda(df: pd.DataFrame, save_dir: Path | None = None):
    """Run all EDA steps."""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    print_missingness(df)
    plot_target_distribution(df, save_dir)
    plot_target_by_backend(df, save_dir)
    plot_scatter_vs_features(df, save_dir)
    plot_correlation_matrix(df, save_dir)

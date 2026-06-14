"""
Model comparison — final summary of all regression models.

Generates a comparison table and plots for:
  - Base OLS (Pipeline)
  - LASSO (GridSearchCV Pipeline)
  - Ridge (GridSearchCV Pipeline)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import r2_score

pd.options.display.float_format = "{:.6f}".format

_STYLE_SET = False

def _setup_style():
    global _STYLE_SET
    if not _STYLE_SET:
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
        plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight"})
        _STYLE_SET = True


def build_comparison_table(ols_results: dict, reg_results: dict) -> pd.DataFrame:
    """Build and print a DataFrame comparing all models on key metrics."""
    rows = [
        ols_results["base_metrics"],
        reg_results["lasso"]["metrics"],
        reg_results["ridge"]["metrics"],
    ]
    df = pd.DataFrame(rows)
    col_order = ["model", "R2_train", "R2_test", "RMSE_test", "MAE_test"]
    extra = [c for c in df.columns if c not in col_order]
    df = df[[c for c in col_order if c in df.columns] + extra]

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON — Summary Table")
    print("=" * 60)
    print(df.to_string(index=False))
    return df


def plot_residual_comparison(ols_results: dict, reg_results: dict, save_dir: Path | None = None):
    """Test-set residual histograms for OLS, LASSO, and Ridge."""
    _setup_style()
    y_test = ols_results["y_test"]
    X_test = ols_results["X_test"]

    models = {
        "Base OLS": y_test.values - ols_results["base_pipeline"].predict(X_test),
        "LASSO":    y_test.values - reg_results["lasso"]["y_pred_test"],
        "Ridge":    y_test.values - reg_results["ridge"]["y_pred_test"],
    }
    colors = ["steelblue", "coral", "mediumpurple"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Test Set Residual Distributions", fontsize=15, fontweight="bold")
    for ax, (name, resid), color in zip(axes, models.items(), colors):
        ax.hist(resid, bins=60, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", ls="--", lw=1)
        ax.set_title(f"{name}  (RMSE = {np.sqrt(np.mean(resid**2)):.6f})", fontsize=11)
        ax.set(xlabel="Residual", ylabel="Count")

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "comparison_residuals.png")
    plt.close(fig)


def plot_predicted_vs_actual(ols_results: dict, reg_results: dict, save_dir: Path | None = None):
    """Scatter plots of predicted vs actual on the test set for all models."""
    _setup_style()
    y_test = ols_results["y_test"]
    X_test = ols_results["X_test"]

    models = {
        "Base OLS": ols_results["base_pipeline"].predict(X_test),
        "LASSO":    reg_results["lasso"]["y_pred_test"],
        "Ridge":    reg_results["ridge"]["y_pred_test"],
    }
    colors = ["steelblue", "coral", "mediumpurple"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Predicted vs Actual (Test Set)", fontsize=15, fontweight="bold")
    for ax, (name, y_pred), color in zip(axes, models.items(), colors):
        ax.scatter(y_test, y_pred, alpha=0.08, s=5, color=color)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", lw=1, alpha=0.7, label="y = x")
        ax.set_title(f"{name}  (R² = {r2_score(y_test, y_pred):.4f})", fontsize=11)
        ax.set(xlabel="Actual", ylabel="Predicted")
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "comparison_predicted_vs_actual.png")
    plt.close(fig)


def plot_metrics_barchart(comparison_df: pd.DataFrame, save_dir: Path | None = None):
    """Bar chart comparing R² (test), RMSE, and MAE across all models."""
    _setup_style()
    metrics_to_plot = [
        ("R2_test",   "R² (Test)",   "steelblue"),
        ("RMSE_test", "RMSE (Test)", "coral"),
        ("MAE_test",  "MAE (Test)",  "seagreen"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Model Comparison — Key Metrics", fontsize=14, fontweight="bold")

    for ax, (metric, label, color) in zip(axes, metrics_to_plot):
        bars = ax.bar(comparison_df["model"], comparison_df[metric],
                      color=color, alpha=0.8, edgecolor="white")
        ax.set_title(label, fontsize=12)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "comparison_metrics_barchart.png")
    plt.close(fig)


def run_comparison(
    ols_results: dict,
    reg_results: dict,
    save_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the full model comparison: print table + generate all plots."""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = build_comparison_table(ols_results, reg_results)
    plot_residual_comparison(ols_results, reg_results, save_dir)
    plot_predicted_vs_actual(ols_results, reg_results, save_dir)
    plot_metrics_barchart(comparison_df, save_dir)
    return comparison_df

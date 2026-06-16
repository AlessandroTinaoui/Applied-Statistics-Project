"""
Feature selection module.
Analyzes correlations with the target and generates bivariate relationship plots.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .data_preparation import TARGET, ID_COLUMNS, CATEGORICAL_FEATURES


def run_feature_selection(df: pd.DataFrame, save_dir: Path) -> pd.DataFrame:
    """
    Calculate Pearson correlation of all numeric features with the target,
    and generate bivariate binned profile plots for the key features.

    Parameters
    ----------
    df       : Training DataFrame (to avoid test leakage).
    save_dir : Directory where plots will be saved.

    Returns
    -------
    corrs_df : DataFrame with Pearson r and missing % per feature.
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
        corrs.append({
            "feature": col,
            "pearson_r": valid[col].corr(valid[TARGET]),
            "abs_pearson_r": abs(valid[col].corr(valid[TARGET])),
            "missing_pct": df[col].isna().mean() * 100,
        })

    corrs_df = pd.DataFrame(corrs).sort_values("abs_pearson_r", ascending=False)

    print("\n[Feature Selection] Top features correlated with T1 (|Pearson r|):")
    print("  " + "─" * 65)
    print(corrs_df.head(15).to_string(index=False))
    print("  " + "─" * 65)


    exclude = set(ID_COLUMNS + [TARGET])
    features_to_plot = ["T2_last_obs", "T1_last_obs", "sx_error_class_last_obs", "readout_error_last_obs"]

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

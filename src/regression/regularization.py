"""
Ridge and LASSO regression with GridSearchCV.

Uses sklearn Pipeline(preprocessor, model) with GridSearchCV for hyperparameter tuning.
The preprocessor is re-fitted within each CV fold, preventing data leakage.
TimeSeriesSplit is used to respect temporal ordering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from .data_preparation import get_feature_names_after_preprocessing


# ──────────────────────────────────────────────────────────────────────
#  Generic regularised model fitter
# ──────────────────────────────────────────────────────────────────────
def _fit_regularized(
    data: dict,
    model,
    alpha_grid: np.ndarray,
    model_name: str,
    n_splits: int = 5,
) -> dict:
    """
    Fit a regularised model (Lasso or Ridge) using GridSearchCV + TimeSeriesSplit.

    Parameters
    ----------
    data       : dict from prepare_data() (must contain X_train, y_train, X_test, y_test,
                 preprocessor, numeric_features)
    model      : unfitted sklearn estimator (Lasso or Ridge instance)
    alpha_grid : array of alpha values to search over
    model_name : label used in printed output
    n_splits   : number of TimeSeriesSplit folds

    Returns
    -------
    dict with pipeline, search object, metrics, test predictions, feature names
    """
    pipe = Pipeline([
        ("preprocessor", clone(data["preprocessor"])),
        ("model", model),
    ])
    search = GridSearchCV(
        pipe,
        param_grid={"model__alpha": alpha_grid},
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="r2",
        n_jobs=1,
        return_train_score=True,
    )
    search.fit(data["X_train"], data["y_train"])

    best_pipe  = search.best_estimator_
    best_alpha = search.best_params_["model__alpha"]
    feature_names = get_feature_names_after_preprocessing(
        best_pipe.named_steps["preprocessor"], data["numeric_features"]
    )

    y_pred_train = best_pipe.predict(data["X_train"])
    y_pred_test  = best_pipe.predict(data["X_test"])
    coef = best_pipe.named_steps["model"].coef_
    n_selected = int((coef != 0).sum())
    n_total    = len(feature_names)

    metrics = {
        "model":    model_name,
        "alpha":    best_alpha,
        "R2_train": r2_score(data["y_train"], y_pred_train),
        "R2_test":  r2_score(data["y_test"],  y_pred_test),
        "RMSE_test": np.sqrt(mean_squared_error(data["y_test"], y_pred_test)),
        "MAE_test":  mean_absolute_error(data["y_test"], y_pred_test),
        "n_features_selected": n_selected,
        "n_features_total":    n_total,
    }

    print(f"\n{'=' * 60}")
    print(f"  {model_name} Regression (GridSearchCV + Pipeline)")
    print(f"{'=' * 60}")
    print(f"  Best α (λ):            {best_alpha:.6f}")
    print(f"  Best CV R²:            {search.best_score_:.6f}")
    if model_name == "LASSO":
        print(f"  Features selected:     {n_selected} / {n_total}")
    for label, key in [("R² (train)", "R2_train"), ("R² (test)", "R2_test"),
                       ("RMSE (test)", "RMSE_test"), ("MAE (test)", "MAE_test")]:
        print(f"  {label + ':':<23} {metrics[key]:.6f}")

    # Top-5 GridSearchCV results
    cv_df = pd.DataFrame(search.cv_results_)
    top5 = cv_df.nsmallest(5, "rank_test_score")[
        ["param_model__alpha", "mean_test_score", "std_test_score", "rank_test_score"]
    ]
    print("\n  GridSearchCV results (top 5 by rank):")
    print(top5.to_string(index=False))

    return {
        "pipeline":     best_pipe,
        "search":       search,
        "metrics":      metrics,
        "y_pred_test":  y_pred_test,
        "feature_names": feature_names,
    }


def fit_lasso(data: dict, n_splits: int = 5) -> dict:
    """Fit LASSO with GridSearchCV (alpha in [1e-2, 1e2], 20 values log-spaced)."""
    return _fit_regularized(
        data,
        Lasso(max_iter=10_000, random_state=42),
        np.logspace(-2, 2, 20),
        "LASSO",
        n_splits,
    )


def fit_ridge(data: dict, n_splits: int = 5) -> dict:
    """Fit Ridge with GridSearchCV (alpha in [1e-4, 1e4], 20 values log-spaced)."""
    return _fit_regularized(
        data,
        Ridge(),
        np.logspace(-4, 4, 20),
        "Ridge",
        n_splits,
    )


# ──────────────────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────────────────
def plot_lasso_coefficients(
    lasso_result: dict,
    feature_names: list[str],
    save_dir: Path | None = None,
):
    """Bar plot of LASSO coefficients; zero-coefficient features shown in gray."""
    coef = lasso_result["pipeline"].named_steps["model"].coef_
    df = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": coef})
        .assign(abs_coef=lambda d: d["Coefficient"].abs())
        .sort_values("abs_coef", ascending=True)
    )
    colors = ["steelblue" if c != 0 else "lightgray" for c in df["Coefficient"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.35)))
    ax.barh(df["Feature"], df["Coefficient"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient value (standardised)")
    m = lasso_result["metrics"]
    ax.set_title(
        f"LASSO Coefficients (α = {m['alpha']:.6f})\n"
        f"{m['n_features_selected']} / {m['n_features_total']} features selected",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "lasso_coefficients.png")
    plt.close(fig)

    selected = df[df["Coefficient"] != 0].sort_values("abs_coef", ascending=False)
    print("\n[LASSO] Selected features (non-zero coefficients):")
    for _, row in selected.iterrows():
        print(f"  {row['Feature']:40s}  β = {row['Coefficient']:+.6f}")


def plot_coefficient_comparison(
    lasso_result: dict,
    ridge_result: dict,
    feature_names: list[str],
    save_dir: Path | None = None,
):
    """Side-by-side horizontal bar chart comparing Ridge and LASSO coefficients."""
    df = pd.DataFrame({
        "Feature": feature_names,
        "LASSO":   lasso_result["pipeline"].named_steps["model"].coef_,
        "Ridge":   ridge_result["pipeline"].named_steps["model"].coef_,
    }).set_index("Feature").sort_values("LASSO", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.35)))
    y_pos  = np.arange(len(df))
    height = 0.35
    ax.barh(y_pos - height / 2, df["Ridge"], height, label="Ridge", color="coral",     alpha=0.7)
    ax.barh(y_pos + height / 2, df["LASSO"], height, label="LASSO", color="steelblue", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Coefficient (standardised)")
    ax.set_title("Ridge vs LASSO Coefficients", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "ridge_vs_lasso_coefficients.png")
    plt.close(fig)


def plot_lasso_path(
    arrays: dict,
    feature_names: list[str],
    n_alphas: int = 80,
    save_dir: Path | None = None,
):
    """
    Plot LASSO regularisation path: how coefficients shrink to zero as α increases.
    Uses a single fit per alpha on the full training set (no CV, no leakage concern).
    """
    alphas = np.logspace(-6, -1, n_alphas)
    coefs  = np.array([
        Lasso(alpha=a, max_iter=10_000).fit(arrays["X_train"], arrays["y_train"]).coef_
        for a in alphas
    ])
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, name in enumerate(feature_names):
        ax.plot(alphas, coefs[:, i], lw=1.2, label=name)
    ax.set(xscale="log",
           xlabel="α (regularisation strength)",
           ylabel="Coefficient value",
           title="LASSO Regularisation Path")
    ax.set_title("LASSO Regularisation Path", fontsize=14, fontweight="bold")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "lasso_regularisation_path.png")
    plt.close(fig)


def plot_gridsearch_curve(
    search: GridSearchCV,
    model_name: str = "LASSO",
    save_dir: Path | None = None,
):
    """Plot mean CV R² vs alpha (log scale), highlighting the best alpha."""
    cv_df = pd.DataFrame(search.cv_results_)
    alphas     = cv_df["param_model__alpha"].astype(float)
    mean_score = cv_df["mean_test_score"]
    std_score  = cv_df["std_test_score"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, mean_score, "o-", color="steelblue", lw=1.5, ms=3)
    ax.fill_between(alphas, mean_score - std_score, mean_score + std_score,
                    alpha=0.15, color="steelblue")
    best_a = search.best_params_["model__alpha"]
    ax.axvline(best_a, color="red", ls="--", lw=1.5, label=f"Best α = {best_a:.6f}")
    ax.set(xscale="log", xlabel="α (regularisation strength)", ylabel="Mean CV R²")
    ax.set_title(f"{model_name} — GridSearchCV Validation Curve", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / f"gridsearch_curve_{model_name.lower().replace(' ', '_')}.png")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
#  Full pipeline
# ──────────────────────────────────────────────────────────────────────
def run_regularization(data: dict, save_dir: Path | None = None) -> dict:
    """
    Fit LASSO and Ridge via GridSearchCV, generate all regularisation plots.

    Parameters
    ----------
    data     : dict from prepare_data()
    save_dir : optional path to save plots

    Returns
    -------
    dict with 'lasso', 'ridge', and preprocessed 'arrays'
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    lasso_result = fit_lasso(data)
    ridge_result = fit_ridge(data)

    feature_names = lasso_result["feature_names"]
    fitted_prep   = lasso_result["pipeline"].named_steps["preprocessor"]
    arrays = {
        "X_train": fitted_prep.transform(data["X_train"]),
        "X_test":  fitted_prep.transform(data["X_test"]),
        "y_train": data["y_train"].values,
        "y_test":  data["y_test"].values,
    }

    plot_lasso_coefficients(lasso_result, feature_names, save_dir)
    plot_lasso_path(arrays, feature_names, save_dir=save_dir)
    plot_coefficient_comparison(lasso_result, ridge_result, feature_names, save_dir)
    plot_gridsearch_curve(lasso_result["search"], "LASSO", save_dir)
    plot_gridsearch_curve(ridge_result["search"], "Ridge", save_dir)

    return {"lasso": lasso_result, "ridge": ridge_result, "arrays": arrays}

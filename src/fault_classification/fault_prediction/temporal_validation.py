from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from . import TARGET_COLUMN, TIME_COLUMN
from .metrics import evaluate_binary_classifier


PreprocessorBuilder = Callable[[pd.DataFrame], Any]
EstimatorBuilder = Callable[..., BaseEstimator]


@dataclass
class TemporalValidationResult:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame


def is_leakage_column(column: str, target: str = TARGET_COLUMN) -> bool:
    if column == target:
        return False
    name = str(column).lower()
    return (
        "_future_" in name
        or name.startswith("future_")
        or name.endswith("_fault_24h")
        or "future_fault" in name
    )


def sort_by_time(df: pd.DataFrame, time_column: str = TIME_COLUMN) -> pd.DataFrame:
    if time_column not in df.columns:
        return df.reset_index(drop=True)

    out = df.copy()
    out[time_column] = pd.to_datetime(out[time_column], utc=True, errors="coerce")
    return out.sort_values(time_column).reset_index(drop=True)


def prepare_modeling_frame(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    time_column: str = TIME_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    if target not in df.columns:
        raise ValueError(f"Target column `{target}` was not found.")

    ordered = sort_by_time(df, time_column=time_column)
    leakage_cols = [c for c in ordered.columns if is_leakage_column(c, target=target)]
    drop_cols = [target] + leakage_cols
    X = ordered.drop(columns=drop_cols, errors="ignore")
    y = ordered[target].astype(int)
    return X, y, ordered, leakage_cols


def get_positive_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        classifier = estimator.named_steps.get("classifier") if isinstance(estimator, Pipeline) else estimator
        classes = getattr(classifier, "classes_", np.array([0, 1]))
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            positive_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
            return probabilities[:, positive_idx]
        if probabilities.ndim == 2 and probabilities.shape[1] == 1:
            only_class = classes[0] if len(classes) else 0
            return np.ones(len(X)) if only_class == 1 else np.zeros(len(X))

    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return np.asarray(scores, dtype=float)

    return np.asarray(estimator.predict(X), dtype=float)


def evaluate_model_temporally(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    estimator_builder: EstimatorBuilder,
    preprocessor_builder: PreprocessorBuilder,
    n_splits: int = 3,
    threshold: float = 0.5,
) -> TemporalValidationResult:
    y_arr = np.asarray(y).astype(int)
    if n_splits < 2:
        raise ValueError("TimeSeriesSplit requires at least 2 splits.")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y_arr[train_idx]
        y_valid = y_arr[valid_idx]
        row: dict[str, Any] = {
            "model": model_name,
            "fold": fold,
            "train_size": int(len(train_idx)),
            "valid_size": int(len(valid_idx)),
            "train_start_index": int(train_idx[0]),
            "train_end_index": int(train_idx[-1]),
            "valid_start_index": int(valid_idx[0]),
            "valid_end_index": int(valid_idx[-1]),
            "train_positive_rate": float(np.mean(y_train == 1)),
            "valid_positive_rate": float(np.mean(y_valid == 1)),
            "error": "",
        }

        try:
            preprocessor = preprocessor_builder(X_train)
            estimator = estimator_builder(y_train=y_train, fold=fold)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", estimator),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_score = get_positive_scores(pipeline, X_valid)
            metrics = evaluate_binary_classifier(y_valid, y_score, threshold=threshold)
            row.update(metrics)

            y_pred = (np.nan_to_num(y_score, nan=-np.inf) >= threshold).astype(int)
            pred = pd.DataFrame(
                {
                    "row_index": X_valid.index.to_numpy(),
                    "model": model_name,
                    "fold": fold,
                    "y_true": y_valid,
                    "y_score": y_score,
                    "y_pred": y_pred,
                }
            )
            if TIME_COLUMN in X_valid.columns:
                pred[TIME_COLUMN] = X_valid[TIME_COLUMN].to_numpy()
            prediction_rows.append(pred)
        except Exception as exc:  # noqa: BLE001 - record and keep remaining folds running.
            row["error"] = f"{type(exc).__name__}: {exc}"
            for metric in [
                "average_precision",
                "pr_auc",
                "roc_auc",
                "f1",
                "precision",
                "recall",
                "balanced_accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
                "support",
                "positive_rate",
            ]:
                row[metric] = np.nan

        metric_rows.append(row)

    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return TemporalValidationResult(pd.DataFrame(metric_rows), predictions)


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "average_precision",
        "pr_auc",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
    ]
    rows = []
    for model, group in fold_metrics.groupby("model", sort=False):
        row: dict[str, Any] = {"model": model, "folds": int(group["fold"].nunique())}
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        row["failed_folds"] = int(group["error"].fillna("").ne("").sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("average_precision_mean", ascending=False)

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from fault_classification.data_preparation import (
    build_preprocessing_pipeline,
    load_and_clean_fault_data,
)

from .model_utils import clean_search_params
from .temporal_validation import (
    evaluate_model_temporally,
    prepare_modeling_frame,
    summarize_fold_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "qiskit_fault_prediction_24h.parquet"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "fault_prediction"

EstimatorFactory = Callable[..., Any]
TuningFunction = Callable[[pd.DataFrame, pd.Series, Any, Any, int], tuple[dict[str, Any], pd.DataFrame]]


@dataclass(frozen=True)
class ModelRunConfig:
    """Configuration needed to run one model end-to-end."""

    model_key: str
    display_name: str
    estimator_factory: EstimatorFactory
    tuning_function: TuningFunction
    default_tune_iter: int = 12


@dataclass
class ModelRunResult:
    """Files and in-memory results produced by one model run."""

    model_key: str
    display_name: str
    output_dir: Path
    best_params: dict[str, Any]
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame
    tuning_results: pd.DataFrame


def build_common_parser(description: str, default_tune_iter: int = 12) -> argparse.ArgumentParser:
    """Create the shared CLI used by all model-specific main files."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--tune-iter", type=int, default=default_tune_iter)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-tuning", action="store_true")
    return parser


def parse_common_args(
    description: str,
    default_tune_iter: int = 12,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the shared command-line arguments."""
    return build_common_parser(description, default_tune_iter).parse_args(argv)


def make_preprocessor_for_frame(X_train: pd.DataFrame) -> Any:
    """Build the shared preprocessing pipeline using only the train fold."""
    preprocessor, _, _ = build_preprocessing_pipeline(X_train.copy())
    return preprocessor


def bounded_n_splits(n_rows: int, requested: int) -> int:
    """Keep TimeSeriesSplit valid for small temporary datasets."""
    if n_rows < 4:
        raise ValueError("Need at least 4 rows for temporal validation.")
    return max(2, min(requested, n_rows - 1))


def estimator_builder(factory: EstimatorFactory, **params: Any) -> Callable[..., Any]:
    """Adapt a model factory to the callback expected by temporal validation."""

    def _build(**_: Any) -> Any:
        return factory(**params)

    return _build


def save_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Save a DataFrame, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Save JSON with stable formatting for reports and inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate non-empty frames and return an empty frame otherwise."""
    valid_frames = [frame for frame in frames if not frame.empty]
    return pd.concat(valid_frames, ignore_index=True) if valid_frames else pd.DataFrame()


def load_modeling_data(
    dataset: Path,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load cleaned data and remove target/leakage columns for modeling."""
    df = load_and_clean_fault_data(dataset)
    X, y, ordered_df, leakage_cols = prepare_modeling_frame(df)
    if max_rows is not None:
        ordered_df = ordered_df.head(max_rows).copy()
        X, y, _, leakage_cols = prepare_modeling_frame(ordered_df)
    return X, y, leakage_cols


def _fit_tuning(
    config: ModelRunConfig,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
    tune_iter: int,
    skip_tuning: bool,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    if skip_tuning:
        return {"note": "Tuning skipped."}, {}, pd.DataFrame()

    preprocessor, _, _ = build_preprocessing_pipeline(X.copy())
    try:
        best_params, tuning_results = config.tuning_function(
            X,
            y,
            preprocessor,
            cv,
            tune_iter,
        )
        clean_params = clean_search_params(best_params)
        return {config.model_key: clean_params}, clean_params, tuning_results
    except Exception as exc:  # noqa: BLE001 - keep the model runnable with defaults.
        error = f"{type(exc).__name__}: {exc}"
        print(f"  {config.display_name} tuning failed, using defaults: {error}")
        return {f"{config.model_key}_error": error}, {}, pd.DataFrame()


def run_prepared_model(
    config: ModelRunConfig,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    threshold: float,
    tune_iter: int,
    skip_tuning: bool,
    results_dir: Path,
) -> ModelRunResult:
    model_dir = results_dir / config.model_key
    cv = TimeSeriesSplit(n_splits=n_splits)

    print(f"\n Tuning {config.display_name}...")
    best_params, estimator_params, tuning_results = _fit_tuning(
        config,
        X,
        y,
        cv,
        tune_iter,
        skip_tuning,
    )

    print(f"\n Evaluating {config.display_name}...")
    result = evaluate_model_temporally(
        config.display_name,
        X,
        y,
        estimator_builder(config.estimator_factory, **estimator_params),
        make_preprocessor_for_frame,
        n_splits=n_splits,
        threshold=threshold,
    )
    summary = summarize_fold_metrics(result.fold_metrics)

    save_json(model_dir / "best_params.json", best_params)
    save_dataframe(model_dir / "tuning_results.csv", tuning_results)
    save_dataframe(model_dir / "metrics_fold_by_fold.csv", result.fold_metrics)
    save_dataframe(model_dir / "metrics_summary.csv", summary)
    save_dataframe(model_dir / "out_of_fold_predictions.csv", result.predictions)

    return ModelRunResult(
        model_key=config.model_key,
        display_name=config.display_name,
        output_dir=model_dir,
        best_params=best_params,
        fold_metrics=result.fold_metrics,
        summary=summary,
        predictions=result.predictions,
        tuning_results=tuning_results,
    )


def run_single_model(
    config: ModelRunConfig,
    dataset: Path = DEFAULT_DATASET,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    n_splits: int = 3,
    tune_iter: int | None = None,
    threshold: float = 0.5,
    max_rows: int | None = None,
    skip_tuning: bool = False,
) -> ModelRunResult:
   
    print(f"\n Loading data for {config.display_name}...")
    X, y, leakage_cols = load_modeling_data(dataset, max_rows=max_rows)
    effective_splits = bounded_n_splits(len(X), n_splits)
    print(f"  Temporal validation: TimeSeriesSplit(n_splits={effective_splits})")
    print(f"  Leakage columns excluded: {len(leakage_cols)}")

    return run_prepared_model(
        config,
        X,
        y,
        n_splits=effective_splits,
        threshold=threshold,
        tune_iter=tune_iter if tune_iter is not None else config.default_tune_iter,
        skip_tuning=skip_tuning,
        results_dir=results_dir,
    )


def run_model_group(
    configs: Sequence[ModelRunConfig],
    dataset: Path = DEFAULT_DATASET,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    n_splits: int = 3,
    tune_iter: int = 12,
    threshold: float = 0.5,
    max_rows: int | None = None,
    skip_tuning: bool = False,
) -> list[ModelRunResult]:
    """Run several separated model configs and save combined comparison files."""
    print("\n Loading shared dataset...")
    X, y, leakage_cols = load_modeling_data(dataset, max_rows=max_rows)
    effective_splits = bounded_n_splits(len(X), n_splits)
    print(f"  Temporal validation: TimeSeriesSplit(n_splits={effective_splits})")
    print(f"  Leakage columns excluded: {len(leakage_cols)}")

    results = [
        run_prepared_model(
            config,
            X,
            y,
            n_splits=effective_splits,
            threshold=threshold,
            tune_iter=tune_iter,
            skip_tuning=skip_tuning,
            results_dir=results_dir,
        )
        for config in configs
    ]

    fold_metrics = concat_frames([item.fold_metrics for item in results])
    predictions = concat_frames([item.predictions for item in results])
    tuning_results = concat_frames([item.tuning_results for item in results])
    summary = summarize_fold_metrics(fold_metrics)
    best_params = {item.model_key: item.best_params for item in results}

    save_json(results_dir / "best_params.json", best_params)
    save_dataframe(results_dir / "tuning_results.csv", tuning_results)
    save_dataframe(results_dir / "metrics_fold_by_fold.csv", fold_metrics)
    save_dataframe(results_dir / "metrics_summary.csv", summary)
    save_dataframe(results_dir / "model_comparison.csv", summary)
    save_dataframe(results_dir / "out_of_fold_predictions.csv", predictions)

    return results

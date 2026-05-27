from __future__ import annotations

from collections.abc import Sequence

from ..runner import ModelRunConfig, parse_common_args, run_single_model
from .estimator import make_random_forest
from .tuning import tune_random_forest


def model_config() -> ModelRunConfig:
    return ModelRunConfig(
        model_key= "random_forest",
        display_name= "Random Forest",
        estimator_factory=make_random_forest,
        tuning_function=tune_random_forest,
        default_tune_iter=12,
    )


def main(argv: Sequence[str] | None = None) -> None:
    config = model_config()
    args = parse_common_args(f"Run {config.display_name}.", config.default_tune_iter, argv)
    run_single_model(
        config,
        dataset=args.dataset,
        results_dir=args.results_dir,
        n_splits=args.n_splits,
        tune_iter=args.tune_iter,
        threshold=args.threshold,
        max_rows=args.max_rows,
        skip_tuning=args.skip_tuning,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

from collections.abc import Sequence

from ..runner import ModelRunConfig, parse_common_args, run_single_model
from .estimator import make_mlp
from .tuning import tune_mlp


def model_config() -> ModelRunConfig:
    """Return the isolated MLP run configuration."""
    return ModelRunConfig(
        model_key="mlp",
        display_name="MLP",
        estimator_factory=make_mlp,
        tuning_function=tune_mlp,
        default_tune_iter=8,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run only the MLP pipeline."""
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

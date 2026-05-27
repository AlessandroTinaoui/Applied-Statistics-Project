from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) in sys.path:
    sys.path.remove(str(PACKAGE_DIR))
sys.path.insert(0, str(SRC_DIR))

from fault_classification.fault_prediction.mlp.main import model_config as mlp_config  # noqa: E402
from fault_classification.fault_prediction.random_forest.main import (  # noqa: E402
    model_config as random_forest_config,
)
from fault_classification.fault_prediction.runner import (  # noqa: E402
    parse_common_args,
    run_model_group,
)
from fault_classification.fault_prediction.xgboost.main import (  # noqa: E402
    model_config as xgboost_config,
)


def main() -> None:
    """Run the separated Random Forest, XGBoost and MLP pipelines together."""
    args = parse_common_args("Run all separated fault prediction models.", default_tune_iter=12)
    run_model_group(
        configs=[
            random_forest_config(),
            xgboost_config(),
            mlp_config(),
        ],
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

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .. import RANDOM_STATE
from ..tuning_utils import bounded_n_iter, cv_results_frame
from .estimator import make_mlp


MLP_PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "classifier__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
    "classifier__alpha": [0.0001, 0.001, 0.01],
    "classifier__learning_rate_init": [0.0005, 0.001],
}


def tune_mlp(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    preprocessor: Any,
    cv: Any,
    n_iter: int = 8,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune the MLP parameters that most directly affect model behavior."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", make_mlp()),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=MLP_PARAM_DISTRIBUTIONS,
        n_iter=bounded_n_iter(MLP_PARAM_DISTRIBUTIONS, n_iter),
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        error_score=np.nan,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)
    return search.best_params_, cv_results_frame(search, "MLP")

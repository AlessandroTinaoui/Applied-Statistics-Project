from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .. import RANDOM_STATE
from ..tuning_utils import bounded_n_iter, cv_results_frame
from .estimator import make_random_forest


RANDOM_FOREST_PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "classifier__n_estimators": [200, 500],
    "classifier__max_depth": [5, 10, 20, None],
    "classifier__min_samples_leaf": [1, 5, 10],
    "classifier__max_features": ["sqrt", "log2"],
}


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    preprocessor: Any,
    cv: Any,
    n_iter: int = 12,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", make_random_forest()),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=RANDOM_FOREST_PARAM_DISTRIBUTIONS,
        n_iter=bounded_n_iter(RANDOM_FOREST_PARAM_DISTRIBUTIONS, n_iter),
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        error_score=np.nan,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)
    return search.best_params_, cv_results_frame(search, "Random Forest")

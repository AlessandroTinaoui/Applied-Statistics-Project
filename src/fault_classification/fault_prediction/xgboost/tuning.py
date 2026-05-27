from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .. import RANDOM_STATE
from ..tuning_utils import bounded_n_iter, cv_results_frame
from .estimator import make_xgboost_classifier


XGBOOST_PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [3, 5],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__subsample": [0.7, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.9, 1.0],
    "classifier__reg_lambda": [1.0, 5.0, 10.0],
    "classifier__reg_alpha": [0.0, 0.1],
}


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    preprocessor: Any,
    cv: Any,
    n_iter: int = 12,
) -> tuple[dict[str, Any], pd.DataFrame]:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", make_xgboost_classifier()),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=XGBOOST_PARAM_DISTRIBUTIONS,
        n_iter=bounded_n_iter(XGBOOST_PARAM_DISTRIBUTIONS, n_iter),
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        error_score=np.nan,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)
    return search.best_params_, cv_results_frame(search, "XGBoost")

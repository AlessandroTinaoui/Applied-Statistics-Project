from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from .. import RANDOM_STATE


def make_random_forest(**params: Any) -> RandomForestClassifier:
    defaults: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    defaults.update(params)
    return RandomForestClassifier(**defaults)

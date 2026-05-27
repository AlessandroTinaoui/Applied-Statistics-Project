from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier


def single_class_dummy(y: np.ndarray) -> DummyClassifier | None:
    """Return a constant classifier when a temporal fold contains one class only.

    Some early temporal folds can contain only failures or only non-failures.
    Tree boosting models cannot be fitted on a single target class, so this
    helper keeps the evaluation loop explicit and prevents a hard crash.
    """
    classes = np.unique(y)
    if len(classes) != 1:
        return None

    dummy = DummyClassifier(strategy="constant", constant=int(classes[0]))
    dummy.fit(np.zeros((len(y), 1)), y)
    return dummy


def clean_search_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove the sklearn Pipeline prefix from tuned classifier parameters."""
    return {key.replace("classifier__", ""): value for key, value in params.items()}

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV


def bounded_n_iter(param_distributions: dict[str, list[Any]], requested: int) -> int:
    """Limit RandomizedSearchCV to the number of available combinations."""
    total = len(list(ParameterGrid(param_distributions)))
    return max(1, min(int(requested), total))


def cv_results_frame(search: RandomizedSearchCV, model_name: str) -> pd.DataFrame:
    """Convert sklearn CV results to a DataFrame with an explicit model name."""
    frame = pd.DataFrame(search.cv_results_)
    frame.insert(0, "model", model_name)
    return frame

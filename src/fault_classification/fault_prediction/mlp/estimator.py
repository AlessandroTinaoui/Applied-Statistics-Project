from __future__ import annotations

from typing import Any

from sklearn.neural_network import MLPClassifier

from .. import RANDOM_STATE


def make_mlp(**params: Any) -> MLPClassifier:
    """Build the Multi-Layer Perceptron used in the comparison.

    The MLP depends heavily on scaled numeric inputs; the shared preprocessing
    pipeline is still responsible for imputation, encoding and scaling before
    this classifier receives the data.

    hidden_layer_sizes, alpha and learning_rate_init are overridable because
    they change model capacity, regularization and optimization behavior.
    class_weight is not available in MLPClassifier, so class imbalance is
    handled through the PR-AUC objective used for validation rather than
    through built-in class weighting.
    """
    defaults: dict[str, Any] = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "learning_rate_init": 0.001,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 10,
        "max_iter": 200,
        "random_state": RANDOM_STATE,
    }
    defaults.update(params)
    return MLPClassifier(**defaults)

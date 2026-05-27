from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

from .. import RANDOM_STATE
from ..model_utils import single_class_dummy


def _prepare_xgboost_matrix(X: Any) -> Any:
    """Convert preprocessed features to a format XGBoost handles reliably."""
    if sparse.issparse(X):
        matrix = X.tocsr(copy=True)
        matrix.data = matrix.data.astype(np.float32, copy=False)
        invalid = ~np.isfinite(matrix.data)
        if invalid.any():
            matrix.data[invalid] = np.nan
        if matrix.shape[1] == 0:
            raise ValueError("XGBoost received zero features after preprocessing.")
        return matrix

    if hasattr(X, "to_numpy"):
        values = X.to_numpy()
    else:
        values = np.asarray(X)

    try:
        matrix = values.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "XGBoost received non-numeric values after preprocessing. "
            "Check that categorical columns are encoded before the classifier."
        ) from exc

    if matrix.ndim != 2:
        raise ValueError(f"XGBoost expected a 2D feature matrix, got shape {matrix.shape}.")
    if matrix.shape[1] == 0:
        raise ValueError("XGBoost received zero features after preprocessing.")

    invalid = ~np.isfinite(matrix)
    if invalid.any():
        matrix = matrix.copy()
        matrix[invalid] = np.nan
    return matrix


def _training_fold_base_score(y: np.ndarray) -> float:
    """Use only the current training fold to initialize the logistic margin."""
    positive_rate = float(np.mean(y == 1))
    return float(np.clip(positive_rate, 1e-6, 1.0 - 1e-6))


class TrainingFoldXGBClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        max_bin: int = 256,
        random_state: int = RANDOM_STATE,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.max_bin = max_bin
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any) -> "TrainingFoldXGBClassifier":
        X_matrix = _prepare_xgboost_matrix(X)
        y_arr = np.asarray(y).astype(int)
        self.classes_ = np.unique(y_arr)

        # XGBoost cannot learn a binary boundary if the fold contains one class.
        # The dummy classifier makes that edge case visible but non-fatal.
        dummy = single_class_dummy(y_arr)
        if dummy is not None:
            self.model_ = dummy
            return self

        from xgboost import XGBClassifier

        positives = max(int(np.sum(y_arr == 1)), 1)
        negatives = int(np.sum(y_arr == 0))

        self.model_ = XGBClassifier(

            # These values define the binary classification task and training
            # backend; they are kept static for reproducibility.
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            missing=np.nan,
            base_score=_training_fold_base_score(y_arr),
            max_bin=self.max_bin,

            # These values are model-capacity or regularization choices. They
            # can be overridden by temporal CV through RandomizedSearchCV.
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            
            # This is computed from y_train only to avoid temporal leakage.
            scale_pos_weight=negatives / positives,
        )
        try:
            self.model_.fit(X_matrix, y_arr)
        except Exception as exc:  # noqa: BLE001 - add useful fold diagnostics.
            shape = getattr(X_matrix, "shape", "unknown")
            raise RuntimeError(
                "XGBoost failed during fit. "
                f"shape={shape}, positives={positives}, negatives={negatives}, "
                f"scale_pos_weight={negatives / positives:.6g}"
            ) from exc
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model_.predict_proba(_prepare_xgboost_matrix(X))

    def predict(self, X: Any) -> np.ndarray:
        return self.model_.predict(_prepare_xgboost_matrix(X))


def make_xgboost_classifier(**params: Any) -> TrainingFoldXGBClassifier:
    """Build the XGBoost classifier with optional CV-selected parameters."""
    return TrainingFoldXGBClassifier(**params)

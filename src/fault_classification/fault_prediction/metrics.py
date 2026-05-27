from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _as_1d(values: Any) -> np.ndarray:
    """Convert metric inputs to one-dimensional numpy arrays."""
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape {arr.shape}.")
    return arr


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC while handling validation folds with one target class."""
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0:
        return 0.0
    if negatives == 0:
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(average_precision_score(y_true, y_score))


def evaluate_binary_classifier(
    y_true: Any,
    y_score: Any,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate an imbalanced binary classifier from probability-like scores.

    The main metric is average precision, also reported as pr_auc. Predictions
    are produced from y_score with the chosen threshold only for threshold-based
    metrics such as F1, precision, recall and the confusion matrix.
    """
    y_true_arr = _as_1d(y_true).astype(int)
    y_score_arr = _as_1d(y_score).astype(float)
    if len(y_true_arr) != len(y_score_arr):
        raise ValueError("`y_true` and `y_score` must have the same length.")
    if len(y_true_arr) == 0:
        raise ValueError("Cannot evaluate an empty validation set.")

    finite_score = np.nan_to_num(y_score_arr, nan=-np.inf)
    y_pred = (finite_score >= threshold).astype(int)
    labels = np.array([0, 1])
    unique_classes = np.unique(y_true_arr)
    one_class_fold = len(unique_classes) < 2

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=labels).ravel()
    average_precision = _safe_average_precision(y_true_arr, y_score_arr)

    if one_class_fold:
        roc_auc = np.nan
        warning = f"Only one class present in y_true: {unique_classes.tolist()}."
    else:
        warning = ""
        roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))

    return {
        "average_precision": average_precision,
        "pr_auc": average_precision,
        "roc_auc": roc_auc,
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support": int(len(y_true_arr)),
        "positive_rate": float(np.mean(y_true_arr == 1)),
        "threshold": float(threshold),
        "one_class_fold": bool(one_class_fold),
        "warning": warning,
    }

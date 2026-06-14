"""
Data preparation for regression analysis on qubit snapshots.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "T1"
ID_COLUMNS = ["backend", "qubit", "model_time"]

SELECTED_FEATURES = [
    # Qubit calibration characteristics
    "T1_prev",
    "T2_last_obs",
    "sx_error_class_last_obs",
    "readout_error_last_obs",

    # Calibration ages & lags
    "calibration_lag_hours",
    "T2_last_obs_age_hours",
    "readout_error_last_obs_age_hours",

    # Temporal & solar position
    "solar_zenith_deg",

    # Environmental rolling statistics (24h averages & fluctuations)
    "temperature_c_mean_prev_24h",
    "temperature_c_std_prev_24h",
    "humidity_pct_mean_prev_24h",
    "humidity_pct_std_prev_24h",
    "pressure_hpa_mean_prev_24h",
    "pressure_hpa_std_prev_24h",
    "neutron_flux_mean_prev_24h",
    "neutron_flux_std_prev_24h",
    "bz_gsm_nt_mean_prev_24h",
    "bz_gsm_nt_std_prev_24h",
]

# Categorical features -> will be encoded as dummies
CATEGORICAL_FEATURES = ["backend", "sx_error_class_last_obs"]

# Temporal split ratio (80% train, 20% test)
TRAIN_RATIO = 0.80


def load_qubit_snapshots(file_path: str | Path) -> pd.DataFrame:
    """Load qubit snapshot dataset, compute T1_prev lag, and sort chronologically."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    df = pd.read_parquet(path)
    df["model_time"] = pd.to_datetime(df["model_time"], utc=True)
    df = df.dropna(subset=[TARGET])

    # Compute T1_prev (autoregressive lag) within each (backend, qubit) group
    df = df.sort_values(["backend", "qubit", "model_time"])
    df["T1_prev"] = df.groupby(["backend", "qubit"])[TARGET].shift(1)
    df = df.dropna(subset=["T1_prev"])

    # Bin sx_error_last_obs into a binary categorical: low_error vs high_or_failed
    if "sx_error_last_obs" in df.columns:
        df["sx_error_class_last_obs"] = np.where(
            df["sx_error_last_obs"].isna(), None,
            np.where(df["sx_error_last_obs"] <= 0.00025, "low_error", "high_or_failed")
        )

    # Keep only available features + IDs + target
    available = [c for c in SELECTED_FEATURES if c in df.columns]
    cols_to_keep = [c for c in ID_COLUMNS + available + [TARGET] if c in df.columns]
    df = df[cols_to_keep].sort_values("model_time").reset_index(drop=True)

    # Convert T1 from seconds to microseconds
    df[TARGET] = df[TARGET] * 1e6
    df["T1_prev"] = df["T1_prev"] * 1e6

    return df


def temporal_train_test_split(
    df: pd.DataFrame, train_ratio: float = TRAIN_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: first train_ratio% for training, rest for testing."""
    df = df.sort_values("model_time").reset_index(drop=True)
    cutoff_time = df.loc[int(len(df) * train_ratio), "model_time"]
    return (
        df[df["model_time"] < cutoff_time].copy(),
        df[df["model_time"] >= cutoff_time].copy(),
    )


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    """Return numeric feature column names (excluding IDs, target, and categoricals)."""
    return [
        c for c in df.columns
        if c not in ID_COLUMNS and c != TARGET and c not in CATEGORICAL_FEATURES
    ]


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X (numeric + categorical) and target y."""
    numeric = get_numeric_features(df)
    return df[numeric + CATEGORICAL_FEATURES].copy(), df[TARGET].copy()


def build_preprocessing_pipeline(
    df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a ColumnTransformer that:
      - Imputes missing numeric values with the median
      - Standardises numeric features (zero-mean, unit-variance)
      - One-hot encodes categorical features (drop='first' to avoid dummy trap)

    Returns: (preprocessor, numeric_feature_names, categorical_feature_names)
    """
    numeric_features = get_numeric_features(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", sparse_output=False)),
            ]), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, CATEGORICAL_FEATURES


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
) -> list[str]:
    """Return feature names after the ColumnTransformer has been fitted."""
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return numeric_features + cat_names


def prepare_data(file_path: str | Path) -> dict:
    """
    Full data preparation pipeline.

    Returns a dict with:
        train_df, test_df    : raw DataFrames (with IDs)
        X_train, X_test      : feature matrices (raw, before sklearn Pipeline)
        y_train, y_test      : target Series (T1 in µs)
        preprocessor         : unfitted ColumnTransformer (to be used inside Pipeline)
        numeric_features     : list of numeric feature column names
    """
    df = load_qubit_snapshots(file_path)
    train_df, test_df = temporal_train_test_split(df)
    X_train, y_train = get_feature_target(train_df)
    X_test, y_test = get_feature_target(test_df)
    preprocessor, numeric_features, _ = build_preprocessing_pipeline(train_df)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numeric_features": numeric_features,
    }

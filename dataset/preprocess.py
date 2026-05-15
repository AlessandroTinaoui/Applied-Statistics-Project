from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "name": "phanerozoic/qiskit-calibration-drift",
        "split": "train",
        "revision": "44118f43caf759b70164506297d35a218976dc5e",
    },
    "columns": {
        "backend": "backend",
        "qubit": "qubit",
        "property": "property",
        "value": "value",
        "observed_time": "observed_time",
        "calibrated_time": "calibrated_time",
        "model_time": "observed_time",
    },
    "qubit_snapshot": {
        "properties": [
            "T1",
            "T2",
            "sx_error",
            "readout_error",
            "prob_meas0_prep1",
            "prob_meas1_prep0",
        ],
        "history_feature_columns": ["T1", "T2", "sx_error", "readout_error"],
        "history_max_age": "24h",
        "include_missing_indicators": True,
    },
    "edge_snapshot": {
        "property_regex": r"^cz_error_(\d+)_(\d+)$",
    },
    "environment": {
        "columns": [
            "latitude",
            "longitude",
            "solar_zenith_deg",
            "temperature_c",
            "pressure_hpa",
            "humidity_pct",
            "kp_index",
            "solar_flux_sfu",
            "dst_nt",
            "bz_gsm_nt",
            "neutron_flux",
        ],
        "causal_fill": True,
        "max_fill_age": "6h",
    },
    "rolling": {
        "source_columns": [
            "temperature_c",
            "pressure_hpa",
            "humidity_pct",
            "kp_index",
            "solar_flux_sfu",
            "dst_nt",
            "bz_gsm_nt",
            "neutron_flux",
        ],
        "windows": ["6h", "12h", "24h"],
        "closed": "left",
        "min_periods": 1,
        "statistics": ["mean", "max", "min", "std", "count"],
        "include_last_previous_value": True,
    },
    "splits": {
        "enabled": False,
        "train_ratio": 0.70,
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "split_column": "split",
    },
    "fault_prediction": {
        "enabled": True,
        "horizon": "24h",
        "target_columns": ["sx_error", "readout_error"],
        "threshold_method": "quantile",
        "threshold_quantile": 0.95,
        "drop_rows_without_future_label": True,
    },
    "outliers": {
        "enabled": False,
    },
    "scaling": {
        "enabled": False,
        "fit_on_split": "train",
        "suffix": "_z",
    },
    "output": {
        "directory": ".",
        "qubit_snapshot_parquet": "qiskit_qubit_snapshots.parquet",
        "qubit_snapshot_csv": "qiskit_qubit_snapshots.csv",
        "edge_snapshot_parquet": "qiskit_edge_snapshots.parquet",
        "edge_snapshot_csv": "qiskit_edge_snapshots.csv",
        "fault_prediction_parquet": "qiskit_fault_prediction_24h.parquet",
        "fault_prediction_csv": "qiskit_fault_prediction_24h.csv",
        "legacy_ml_ready_parquet": "qiskit_calibration_drift_ml_ready.parquet",
        "legacy_ml_ready_csv": "qiskit_calibration_drift_ml_ready.csv",
        "report": "qiskit_calibration_drift_report.json",
    },
}

ID_COLUMNS = {"backend", "qubit", "q1", "q2", "model_time", "split"}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        user_config = tomllib.load(f)
    return deep_update(DEFAULT_CONFIG, user_config)


def resolve_config_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()

    script_relative = SCRIPT_DIR / candidate
    if script_relative.exists():
        return script_relative.resolve()

    return candidate.resolve()


def output_dir(cfg: dict[str, Any], config_path: Path) -> Path:
    out_dir = Path(cfg["output"]["directory"]).expanduser()
    if not out_dir.is_absolute():
        out_dir = config_path.parent / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def sanitize_column_name(value: object) -> str:
    name = str(value).strip()
    name = re.sub(r"[\s/\-]+", "_", name)
    name = re.sub(r"[()]+", "", name)
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_times_and_values(df: pd.DataFrame, cfg: dict[str, Any], report: dict[str, Any]) -> pd.DataFrame:
    cols = cfg["columns"]
    df = df.copy()
    time_candidates = [cols["observed_time"], cols.get("calibrated_time"), cols["model_time"]]

    for col in sorted({c for c in time_candidates if c and c in df.columns}):
        before = df[col].notna().sum()
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        after = df[col].notna().sum()
        report[f"{col}_parse_failures"] = int(before - after)

    model_time = cols["model_time"]
    if model_time not in df.columns:
        raise ValueError(f"Configured model_time column not found: {model_time}")

    df["model_time"] = df[model_time]
    before_numeric = df[cols["value"]].notna().sum()
    df[cols["value"]] = pd.to_numeric(df[cols["value"]], errors="coerce")
    report["value_parse_failures"] = int(before_numeric - df[cols["value"]].notna().sum())
    report["value_missing_rate_after_numeric_parse"] = float(df[cols["value"]].isna().mean())

    observed = cols.get("observed_time")
    calibrated = cols.get("calibrated_time")
    if observed in df.columns and calibrated in df.columns:
        lag = df[observed] - df[calibrated]
        df["calibration_lag_hours"] = lag.dt.total_seconds() / 3600
        report["calibration_lag_hours_missing_rate"] = float(df["calibration_lag_hours"].isna().mean())
        report["negative_calibration_lag_rows"] = int((df["calibration_lag_hours"] < 0).sum())

    required_non_null = [cols["backend"], cols["qubit"], cols["property"], "model_time", cols["value"]]
    before = len(df)
    df = df.dropna(subset=required_non_null).copy()
    report["dropped_structurally_invalid_rows"] = int(before - len(df))
    df[cols["property"]] = df[cols["property"]].astype(str)
    return df


def load_raw_dataset(cfg: dict[str, Any], report: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = cfg["dataset"]
    load_kwargs = {
        "path": dataset_cfg["name"],
        "split": dataset_cfg["split"],
    }
    if dataset_cfg.get("revision"):
        load_kwargs["revision"] = dataset_cfg["revision"]

    ds = load_dataset(**load_kwargs)
    report["dataset_name"] = dataset_cfg["name"]
    report["dataset_split"] = dataset_cfg["split"]
    report["dataset_revision"] = dataset_cfg.get("revision")
    report["dataset_fingerprint"] = getattr(ds, "_fingerprint", None)

    df = ds.to_pandas()
    report["raw_shape"] = list(df.shape)
    report["raw_columns"] = list(df.columns)
    return df


def summarize_raw_dataset(df: pd.DataFrame, cfg: dict[str, Any], report: dict[str, Any]) -> None:
    cols = cfg["columns"]
    report["raw_missingness"] = {
        col: float(rate)
        for col, rate in df.isna().mean().sort_values(ascending=False).items()
        if rate > 0
    }
    report["raw_property_counts_top"] = (
        df[cols["property"]].value_counts(dropna=False).head(20).astype(int).to_dict()
    )
    report["raw_backend_counts"] = df[cols["backend"]].value_counts(dropna=False).astype(int).to_dict()
    report["raw_qubit_range"] = {
        "min": int(df[cols["qubit"]].min()),
        "max": int(df[cols["qubit"]].max()),
        "nunique": int(df[cols["qubit"]].nunique(dropna=True)),
    }


def pivot_properties(
    df: pd.DataFrame,
    id_cols: list[str],
    property_col: str,
    value_col: str,
) -> pd.DataFrame:
    wide = (
        df.pivot_table(
            index=id_cols,
            columns=property_col,
            values=value_col,
            aggfunc="median",
        )
        .reset_index()
    )
    wide.columns = [sanitize_column_name(c) for c in wide.columns]
    return wide


def time_limited_forward_fill(
    df: pd.DataFrame,
    group_cols: list[str],
    time_col: str,
    value_cols: list[str],
    max_age: pd.Timedelta,
) -> pd.DataFrame:
    if not value_cols:
        return df.copy()

    out = []
    for _, g in df.sort_values(group_cols + [time_col]).groupby(group_cols, observed=True, sort=False):
        g = g.sort_values(time_col).copy()
        t = g[time_col]

        for col in value_cols:
            observed = g[col].notna()
            last_value = g[col].ffill()
            last_time = t.where(observed).ffill()
            age = t - last_time
            fill_mask = g[col].isna() & last_value.notna() & (age <= max_age)
            g.loc[fill_mask, col] = last_value.loc[fill_mask]

        out.append(g)

    return pd.concat(out, ignore_index=True)


def add_last_observation_features(
    df: pd.DataFrame,
    group_cols: list[str],
    time_col: str,
    value_cols: list[str],
    max_age: pd.Timedelta,
    include_missing_indicators: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    if not value_cols:
        return df.copy(), []

    out = []
    added: list[str] = []
    for _, g in df.sort_values(group_cols + [time_col]).groupby(group_cols, observed=True, sort=False):
        g = g.sort_values(time_col).copy()
        t = g[time_col]

        for col in value_cols:
            if col not in g.columns:
                continue

            observed = g[col].notna()
            last_value = g[col].ffill()
            last_time = t.where(observed).ffill()
            age = t - last_time
            too_old = age > max_age
            value_name = f"{col}_last_obs"
            age_name = f"{col}_last_obs_age_hours"
            was_missing_name = f"{col}_was_missing"

            g[value_name] = last_value.mask(too_old)
            g[age_name] = (age.dt.total_seconds() / 3600).where(last_value.notna()).mask(too_old)

            new_cols = [value_name, age_name]
            if include_missing_indicators:
                g[was_missing_name] = g[col].isna().astype("int8")
                new_cols.append(was_missing_name)

            for new_col in new_cols:
                if new_col not in added:
                    added.append(new_col)

        out.append(g)

    return pd.concat(out, ignore_index=True), added


def build_environment_features(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    report: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    cols = cfg["columns"]
    backend_col = cols["backend"]
    env_candidates = cfg["environment"]["columns"]
    available = [c for c in env_candidates if c in df.columns]
    numeric = [c for c in available if pd.api.types.is_numeric_dtype(df[c])]

    if not numeric:
        report["environment_available_columns"] = []
        return df[[backend_col, "model_time"]].drop_duplicates(), []

    env = (
        df.groupby([backend_col, "model_time"], as_index=False, observed=True)[numeric]
        .median()
        .sort_values([backend_col, "model_time"])
    )

    missing = env[numeric].isna().mean()
    report["environment_available_columns"] = numeric
    report["environment_all_missing_columns"] = sorted(missing[missing >= 1.0].index.tolist())

    usable_current = [c for c in numeric if missing[c] < 1.0]
    if cfg["environment"].get("causal_fill", True) and usable_current:
        env = time_limited_forward_fill(
            env,
            group_cols=[backend_col],
            time_col="model_time",
            value_cols=usable_current,
            max_age=pd.Timedelta(cfg["environment"]["max_fill_age"]),
        )

    rolling, rolling_cols = add_rolling_features(env, cfg)
    features = env.merge(rolling, on=[backend_col, "model_time"], how="left", validate="one_to_one")
    feature_cols = [c for c in features.columns if c not in [backend_col, "model_time"]]
    report["environment_feature_count_before_low_info_drop"] = len(feature_cols)
    report["rolling_feature_count"] = len(rolling_cols)
    return features, feature_cols


def add_rolling_features(env: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    backend_col = cfg["columns"]["backend"]
    roll_cfg = cfg["rolling"]
    sources = [
        c
        for c in roll_cfg["source_columns"]
        if c in env.columns and env[c].notna().any()
    ]
    windows = roll_cfg["windows"]
    stats = set(roll_cfg["statistics"])
    closed = roll_cfg.get("closed", "left")
    min_periods = int(roll_cfg.get("min_periods", 1))
    include_last = bool(roll_cfg.get("include_last_previous_value", True))

    if not sources:
        return env[[backend_col, "model_time"]].drop_duplicates(), []

    parts = []
    for backend, g in env.groupby(backend_col, observed=True, sort=False):
        g = g.sort_values("model_time").set_index("model_time")
        features = pd.DataFrame(index=g.index)
        features[backend_col] = backend

        for col in sources:
            if include_last:
                features[f"{col}_last_prev"] = g[col].shift(1)

            for window in windows:
                suffix = window.lower().replace(" ", "")
                rolling = g[col].rolling(window=window, closed=closed, min_periods=min_periods)
                if "mean" in stats:
                    features[f"{col}_mean_prev_{suffix}"] = rolling.mean()
                if "max" in stats:
                    features[f"{col}_max_prev_{suffix}"] = rolling.max()
                if "min" in stats:
                    features[f"{col}_min_prev_{suffix}"] = rolling.min()
                if "std" in stats:
                    features[f"{col}_std_prev_{suffix}"] = rolling.std(ddof=0)
                if "median" in stats:
                    features[f"{col}_median_prev_{suffix}"] = rolling.median()
                if "count" in stats:
                    features[f"{col}_count_prev_{suffix}"] = rolling.count()

        parts.append(features.reset_index())

    rolling = pd.concat(parts, ignore_index=True)
    rolling_cols = [c for c in rolling.columns if c not in [backend_col, "model_time"]]
    return rolling, rolling_cols


def compute_split_cutoffs(df: pd.DataFrame, cfg: dict[str, Any], report: dict[str, Any]) -> dict[str, pd.Timestamp]:
    split_cfg = cfg["splits"]
    if not split_cfg.get("enabled", True):
        return {}

    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])
    total = train_ratio + val_ratio + test_ratio
    train_ratio, val_ratio = train_ratio / total, val_ratio / total

    times = np.array(sorted(df["model_time"].dropna().unique()))
    if len(times) < 3:
        return {}

    train_cut = times[max(0, min(len(times) - 1, int(np.floor(len(times) * train_ratio)) - 1))]
    val_cut = times[max(0, min(len(times) - 1, int(np.floor(len(times) * (train_ratio + val_ratio))) - 1))]
    cutoffs = {
        "train_end": pd.Timestamp(train_cut),
        "validation_end": pd.Timestamp(val_cut),
    }
    report["split_cutoffs"] = {k: str(v) for k, v in cutoffs.items()}
    return cutoffs


def assign_temporal_split(df: pd.DataFrame, cfg: dict[str, Any], cutoffs: dict[str, pd.Timestamp]) -> pd.DataFrame:
    split_col = cfg["splits"].get("split_column", "split")
    out = df.copy()
    if not cfg["splits"].get("enabled", True) or not cutoffs:
        return out

    out[split_col] = np.select(
        [out["model_time"] <= cutoffs["train_end"], out["model_time"] <= cutoffs["validation_end"]],
        ["train", "validation"],
        default="test",
    )
    return out


def merge_environment(df: pd.DataFrame, env_features: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    backend_col = cfg["columns"]["backend"]
    return df.merge(env_features, on=[backend_col, "model_time"], how="left", validate="many_to_one")


def build_qubit_snapshots(
    df: pd.DataFrame,
    env_features: pd.DataFrame,
    cfg: dict[str, Any],
    report: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    cols = cfg["columns"]
    property_col = cols["property"]
    value_col = cols["value"]
    edge_pattern = re.compile(cfg["edge_snapshot"]["property_regex"])

    requested = cfg["qubit_snapshot"]["properties"]
    requested_clean = [sanitize_column_name(p) for p in requested]
    qubit_rows = df[
        (df[cols["qubit"]] >= 0)
        & df[property_col].isin(requested)
        & ~df[property_col].str.match(edge_pattern)
    ].copy()

    id_cols = [cols["backend"], cols["qubit"], "model_time"]
    report["qubit_long_rows"] = int(len(qubit_rows))
    report["qubit_property_counts"] = qubit_rows[property_col].value_counts().astype(int).to_dict()
    report["qubit_duplicate_property_keys"] = int(qubit_rows.duplicated(id_cols + [property_col]).sum())

    qubit_rows[property_col] = qubit_rows[property_col].map(sanitize_column_name)
    qubit = pivot_properties(qubit_rows, id_cols, property_col, value_col)
    for col in requested_clean:
        if col not in qubit.columns:
            qubit[col] = np.nan

    metadata = (
        qubit_rows.groupby(id_cols, as_index=False, observed=True)[["calibration_lag_hours"]]
        .median()
    )
    qubit = qubit.merge(metadata, on=id_cols, how="left", validate="one_to_one")
    qubit = merge_environment(qubit, env_features, cfg)

    history_cols = [
        sanitize_column_name(c)
        for c in cfg["qubit_snapshot"]["history_feature_columns"]
        if sanitize_column_name(c) in qubit.columns
    ]
    qubit, history_features = add_last_observation_features(
        qubit,
        group_cols=[cols["backend"], cols["qubit"]],
        time_col="model_time",
        value_cols=history_cols,
        max_age=pd.Timedelta(cfg["qubit_snapshot"]["history_max_age"]),
        include_missing_indicators=bool(cfg["qubit_snapshot"].get("include_missing_indicators", True)),
    )

    report["qubit_snapshot_shape_before_low_info_drop"] = list(qubit.shape)
    report["qubit_history_feature_columns"] = history_features
    return qubit.sort_values(id_cols).reset_index(drop=True), requested_clean


def build_edge_snapshots(
    df: pd.DataFrame,
    env_features: pd.DataFrame,
    cfg: dict[str, Any],
    report: dict[str, Any],
) -> pd.DataFrame:
    cols = cfg["columns"]
    pattern = re.compile(cfg["edge_snapshot"]["property_regex"])
    edge_rows = df[df[cols["property"]].str.match(pattern)].copy()
    report["edge_long_rows"] = int(len(edge_rows))

    if edge_rows.empty:
        return pd.DataFrame(columns=[cols["backend"], "q1", "q2", "model_time", "cz_error"])

    extracted = edge_rows[cols["property"]].str.extract(pattern).astype(int)
    edge_rows["q1"] = extracted[0]
    edge_rows["q2"] = extracted[1]

    edge = (
        edge_rows.groupby([cols["backend"], "q1", "q2", "model_time"], as_index=False, observed=True)
        .agg(
            cz_error=(cols["value"], "median"),
            calibration_lag_hours=("calibration_lag_hours", "median"),
        )
    )
    edge = merge_environment(edge, env_features, cfg)
    report["edge_snapshot_shape_before_low_info_drop"] = list(edge.shape)
    report["edge_unique_directed_edges"] = int(edge[["backend", "q1", "q2"]].drop_duplicates().shape[0])
    return edge.sort_values([cols["backend"], "q1", "q2", "model_time"]).reset_index(drop=True)


def add_future_fault_labels(
    qubit: pd.DataFrame,
    cfg: dict[str, Any],
    report: dict[str, Any],
) -> pd.DataFrame:
    fp_cfg = cfg["fault_prediction"]
    horizon = pd.Timedelta(fp_cfg["horizon"])
    target_cols = [sanitize_column_name(c) for c in fp_cfg["target_columns"] if sanitize_column_name(c) in qubit.columns]

    if not fp_cfg.get("enabled", True) or not target_cols:
        report["fault_prediction_enabled"] = False
        return pd.DataFrame()

    method = fp_cfg.get("threshold_method", "quantile")
    if method not in {"quantile", "train_quantile"}:
        raise ValueError("fault threshold_method must be either 'quantile' or 'train_quantile'.")

    split_col = cfg["splits"].get("split_column", "split")
    if method == "train_quantile":
        if split_col not in qubit.columns:
            raise ValueError("threshold_method='train_quantile' requires an existing split column.")
        threshold_mask = qubit[split_col].eq("train")
    else:
        threshold_mask = pd.Series(True, index=qubit.index)

    quantile = float(fp_cfg["threshold_quantile"])
    thresholds: dict[str, float] = {}
    for col in target_cols:
        threshold = qubit.loc[threshold_mask, col].dropna().quantile(quantile)
        if pd.notna(threshold):
            thresholds[col] = float(threshold)

    report["fault_prediction_enabled"] = True
    report["fault_prediction_horizon"] = str(horizon)
    report["fault_threshold_method"] = fp_cfg["threshold_method"]
    report["fault_threshold_quantile"] = quantile
    report["fault_thresholds"] = thresholds

    out_parts = []
    for _, g in qubit.sort_values(["backend", "qubit", "model_time"]).groupby(["backend", "qubit"], observed=True, sort=False):
        g = g.copy()
        # Pandas 3 may store timezone-aware timestamps at microsecond
        # resolution. Force UTC nanoseconds so the 24h horizon has the same
        # unit as the search array.
        times = (
            pd.to_datetime(g["model_time"], utc=True)
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .astype("datetime64[ns]")
            .astype("int64")
            .to_numpy()
        )
        right_bounds = np.searchsorted(times, times + horizon.value, side="right")

        for col, threshold in thresholds.items():
            values = g[col].to_numpy(dtype=float)
            future_max = np.full(len(g), np.nan, dtype=float)
            future_count = np.zeros(len(g), dtype=int)

            for i, right in enumerate(right_bounds):
                future = values[i + 1:right]
                observed = future[~np.isnan(future)]
                future_count[i] = len(observed)
                if len(observed):
                    future_max[i] = float(np.max(observed))

            g[f"{col}_future_max_24h"] = future_max
            g[f"{col}_future_obs_count_24h"] = future_count
            fault = np.where(np.isnan(future_max), np.nan, (future_max > threshold).astype(float))
            g[f"{col}_fault_24h"] = fault

        out_parts.append(g)

    fault = pd.concat(out_parts, ignore_index=True)
    fault_cols = [f"{col}_fault_24h" for col in thresholds]
    obs_count_cols = [f"{col}_future_obs_count_24h" for col in thresholds]
    fault["future_fault_target_count_24h"] = fault[obs_count_cols].gt(0).sum(axis=1)
    fault["fault_24h"] = fault[fault_cols].max(axis=1, skipna=True)
    fault.loc[fault["future_fault_target_count_24h"].eq(0), "fault_24h"] = np.nan

    before = len(fault)
    if fp_cfg.get("drop_rows_without_future_label", True):
        fault = fault[fault["fault_24h"].notna()].copy()
    report["fault_prediction_rows_dropped_without_future_label"] = int(before - len(fault))
    report["fault_prediction_shape_before_low_info_drop"] = list(fault.shape)

    return fault.sort_values(["backend", "qubit", "model_time"]).reset_index(drop=True)


def drop_low_information_columns(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    dataset_name: str,
    report: dict[str, Any],
    protected_cols: set[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    protected = set(protected_cols or set())
    split_col = cfg["splits"].get("split_column", "split")
    id_like = set(ID_COLUMNS) | {split_col}
    candidate_cols = [c for c in df.columns if c not in id_like and c not in protected]

    all_missing = sorted([c for c in candidate_cols if df[c].isna().all()])
    constant = sorted([
        c for c in candidate_cols
        if c not in all_missing and df[c].nunique(dropna=True) <= 1
    ])
    to_drop = all_missing + constant

    report[f"{dataset_name}_dropped_all_missing_columns"] = all_missing
    report[f"{dataset_name}_dropped_constant_columns"] = constant
    if not to_drop:
        return df
    return df.drop(columns=to_drop)


def numeric_feature_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    split_col = cfg["splits"].get("split_column", "split")
    excluded_suffixes = ("_was_missing", "_fault_24h")
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return [
        c for c in numeric
        if c not in ID_COLUMNS
        and c != split_col
        and not c.endswith(excluded_suffixes)
        and c != "fault_24h"
    ]


def fit_mask(df: pd.DataFrame, cfg: dict[str, Any], section: str) -> pd.Series:
    split_col = cfg["splits"].get("split_column", "split")
    fit_split = cfg[section].get("fit_on_split", "train")
    if cfg["splits"].get("enabled", True) and split_col in df.columns:
        mask = df[split_col] == fit_split
        if mask.any():
            return mask
    return pd.Series(True, index=df.index)


def add_training_standardized_features(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    report: dict[str, Any],
    dataset_name: str,
) -> pd.DataFrame:
    if df.empty or not cfg["scaling"].get("enabled", False):
        return df

    suffix = cfg["scaling"].get("suffix", "_z")
    cols = numeric_feature_columns(df, cfg)
    mask = fit_mask(df, cfg, "scaling")
    out = df.copy()
    added = []

    for col in cols:
        mean = out.loc[mask, col].mean()
        std = out.loc[mask, col].std(ddof=0)
        if pd.notna(mean) and pd.notna(std) and std > 0:
            new_col = f"{col}{suffix}"
            out[new_col] = (out[col] - mean) / std
            added.append(new_col)

    report[f"{dataset_name}_standardized_feature_count"] = len(added)
    return out


def summarize_processed_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    report: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    split_col = cfg["splits"].get("split_column", "split")
    report[f"{dataset_name}_shape"] = list(df.shape)
    if split_col in df.columns:
        report[f"{dataset_name}_split_counts"] = df[split_col].value_counts(dropna=False).astype(int).to_dict()
    report[f"{dataset_name}_missingness"] = {
        col: float(rate)
        for col, rate in df.isna().mean().sort_values(ascending=False).items()
        if rate > 0
    }
    if "fault_24h" in df.columns and split_col in df.columns:
        report[f"{dataset_name}_fault_rate_by_split"] = (
            df.groupby(split_col, observed=True)["fault_24h"].mean().to_dict()
        )


def theoretical_rationale() -> list[str]:
    return [
        "The qubit-level table uses (backend, qubit, observed_time) as the statistical unit because T1, T2, sx_error and readout_error are single-qubit calibration parameters.",
        "CZ errors are kept in a separate edge-level table because each cz_error_i_j belongs to a directed two-qubit coupling, not to a single qubit row.",
        "Observed calibration targets are not overwritten by imputations; causal last-observation features are added separately with an age column so downstream models can account for measurement staleness.",
        "Environmental values are aligned by backend and timestamp and only forward-filled within a bounded causal window, preventing future information from entering features.",
        "Rolling features use closed='left', so summaries at time t use only observations strictly before t.",
        "No split column is emitted by default: the modelling split is intentionally left to the downstream analysis so it can be chosen manually and documented for each research question.",
        "Fault labels are defined from future observed errors within 24h; with the default quantile threshold this is a descriptive critical-error definition, not a fitted model parameter.",
        "Outlier clipping is disabled by default because extreme errors may be genuine degradation events, which are scientifically relevant for fault tolerance.",
    ]


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def write_outputs(
    outputs: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    report: dict[str, Any],
    config_path: Path,
) -> None:
    out_dir = output_dir(cfg, config_path)
    output_cfg = cfg["output"]
    paths: dict[str, str] = {}

    mapping = {
        "qubit_snapshot": ["qubit_snapshot_parquet", "qubit_snapshot_csv", "legacy_ml_ready_parquet", "legacy_ml_ready_csv"],
        "edge_snapshot": ["edge_snapshot_parquet", "edge_snapshot_csv"],
        "fault_prediction": ["fault_prediction_parquet", "fault_prediction_csv"],
    }
    for dataset_name, output_keys in mapping.items():
        df = outputs.get(dataset_name)
        if df is None:
            continue
        for key in output_keys:
            filename = output_cfg.get(key)
            if not filename:
                continue
            path = out_dir / filename
            write_dataframe(df, path)
            paths[key] = str(path)

    report_path = out_dir / output_cfg["report"]
    paths["report"] = str(report_path)
    report["outputs"] = paths
    report["theoretical_rationale"] = theoretical_rationale()

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)


def run(config_path: Path = SCRIPT_DIR / "preprocessing_config.toml") -> dict[str, pd.DataFrame]:
    config_path = resolve_config_path(config_path)
    cfg = load_config(config_path)
    cols = cfg["columns"]
    report: dict[str, Any] = {"config_path": str(config_path)}

    raw = load_raw_dataset(cfg, report)
    summarize_raw_dataset(raw, cfg, report)

    required = [cols["backend"], cols["qubit"], cols["property"], cols["value"], cols["model_time"]]
    validate_required_columns(raw, required)

    df = parse_times_and_values(raw, cfg, report)
    env_features, _ = build_environment_features(df, cfg, report)

    qubit, qubit_properties = build_qubit_snapshots(df, env_features, cfg, report)
    cutoffs = compute_split_cutoffs(qubit, cfg, report)
    qubit = assign_temporal_split(qubit, cfg, cutoffs)

    edge = build_edge_snapshots(df, env_features, cfg, report)
    edge = assign_temporal_split(edge, cfg, cutoffs)

    fault = add_future_fault_labels(qubit, cfg, report)

    qubit = drop_low_information_columns(
        qubit,
        cfg,
        "qubit_snapshot",
        report,
        protected_cols=set(qubit_properties) | {"calibration_lag_hours"},
    )
    edge = drop_low_information_columns(
        edge,
        cfg,
        "edge_snapshot",
        report,
        protected_cols={"cz_error", "calibration_lag_hours"},
    )
    fault = drop_low_information_columns(
        fault,
        cfg,
        "fault_prediction",
        report,
        protected_cols=set(qubit_properties) | {"fault_24h"},
    )

    qubit = add_training_standardized_features(qubit, cfg, report, "qubit_snapshot")
    edge = add_training_standardized_features(edge, cfg, report, "edge_snapshot")
    fault = add_training_standardized_features(fault, cfg, report, "fault_prediction")

    outputs = {
        "qubit_snapshot": qubit,
        "edge_snapshot": edge,
        "fault_prediction": fault,
    }
    for name, frame in outputs.items():
        summarize_processed_dataset(frame, name, report, cfg)

    write_outputs(outputs, cfg, report, config_path)
    return outputs


def main(config_path: Path = SCRIPT_DIR / "preprocessing_config.toml") -> pd.DataFrame:
    outputs = run(config_path)
    return outputs["fault_prediction"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "preprocessing_config.toml")
    args = parser.parse_args()
    result = run(args.config)
    for name, df in result.items():
        print(f"{name}: {df.shape}")

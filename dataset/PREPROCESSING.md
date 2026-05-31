# Preprocessing

This document summarizes, in bullet points, what
[`dataset/preprocess.py`](preprocess.py) does.

## Purpose

- Loads the Hugging Face dataset `phanerozoic/qiskit-calibration-drift`.
- Uses a pinned dataset revision from `dataset/preprocessing_config.toml`, so the downloaded data are fixed in time and reproducible.
- Converts the source dataset from long format to wider tables that are easier to use for statistical analysis.
- Produces separate datasets for:
  - qubit-level snapshots;
  - edge-level snapshots for two-qubit gate errors;
  - 24-hour fault prediction.
- Keeps identifiers and timestamps such as `backend`, `qubit`, `q1`, `q2`, and `model_time`.
- Does not force the output into a fully numeric sklearn matrix: categorical encoding, imputation, and scaling should happen in the modelling stage, after the temporal split.

## Reproducibility

- The dataset is pinned through the `revision` field in `dataset/preprocessing_config.toml`.
- The current pinned revision is:
  - `44118f43caf759b70164506297d35a218976dc5e`.
- This snapshot covers roughly:
  - `2026-04-07` to `2026-05-01`.
- This gives more than one week of data while avoiding moving-target results from the Hugging Face `main` branch.
- If the revision is changed, all generated CSV/parquet files and the report should be regenerated.
- The report records:
  - dataset name;
  - split;
  - revision;
  - Hugging Face fingerprint;
  - raw shape;
  - output paths.

## Configuration

- Reads configuration from `dataset/preprocessing_config.toml`.
- Falls back to `DEFAULT_CONFIG` in `preprocess.py` when a config key is missing.
- Configurable sections include:
  - Hugging Face dataset name, split, and pinned revision;
  - raw column names;
  - qubit-level properties to extract;
  - regex for edge-level errors;
  - environmental columns;
  - rolling windows;
  - fault prediction labels;
  - CSV/parquet/report output names.

## Raw Dataset Loading

- Uses `datasets.load_dataset`.
- Converts the Hugging Face dataset to a `pandas.DataFrame`.
- Stores raw dataset metadata in the JSON report:
  - dataset name;
  - split;
  - revision;
  - fingerprint;
  - shape;
  - column list.

## Raw Dataset Summary

- Computes raw missingness by column.
- Counts the most frequent calibration properties, for example:
  - `sx_error`;
  - `readout_error`;
  - `T1`;
  - `T2`;
  - `cz_error_i_j`.
- Counts rows by backend.
- Records the qubit range and number of unique qubits.

## Required Column Validation

- Checks that the raw dataset contains the required configured columns:
  - `backend`;
  - `qubit`;
  - `property`;
  - `value`;
  - the timestamp column used as `model_time`.
- Raises an error if a required column is missing.

## Time and Value Parsing

- Parses the following columns as UTC datetimes:
  - `observed_time`;
  - `calibrated_time`;
  - the configured `model_time` source column.
- Converts `value` to numeric.
- Counts parsing failures in the JSON report.
- Creates a normalized `model_time` column.
- Computes `calibration_lag_hours` as:
  - `observed_time - calibrated_time`.
- Drops structurally invalid rows without:
  - backend;
  - qubit;
  - property;
  - model time;
  - numeric value.

## Column Name Sanitization

- Uses `sanitize_column_name` to make property names safe as dataframe columns.
- Replaces spaces, slashes, and hyphens with `_`.
- Removes unnecessary non-alphanumeric characters.
- Collapses repeated underscores.

## Environmental Features

- Selects configured environmental columns when present in the raw dataset:
  - `latitude`;
  - `longitude`;
  - `solar_zenith_deg`;
  - `temperature_c`;
  - `pressure_hpa`;
  - `humidity_pct`;
  - `kp_index`;
  - `solar_flux_sfu`;
  - `dst_nt`;
  - `bz_gsm_nt`;
  - `neutron_flux`.
- Keeps numeric environmental columns only.
- Aggregates environmental values by:
  - `backend`;
  - `model_time`.
- Uses the median when repeated values exist for the same backend and timestamp.
- Records available environmental columns in the report.
- Records fully missing environmental columns in the report.

## Causal Environmental Forward Fill

- If `causal_fill = true`, forward-fills missing environmental values.
- Performs the forward fill separately for each backend.
- Uses only past values, never future values.
- Limits forward fill to `max_fill_age`, default `6h`.
- Does not fill values when the last observation is too old.

## Environmental Rolling Features

- Builds historical features for environmental columns that are not fully missing.
- Default rolling windows:
  - `6h`;
  - `12h`;
  - `24h`.
- Uses `closed = "left"`, so features at time `t` use only observations strictly before `t`.
- For each window, computes the configured statistics:
  - mean;
  - maximum;
  - minimum;
  - standard deviation;
  - observation count.
- Also adds the previous observed value with suffix `_last_prev`.
- Example generated columns:
  - `temperature_c_mean_prev_24h`;
  - `humidity_pct_std_prev_12h`;
  - `neutron_flux_max_prev_6h`;
  - `bz_gsm_nt_last_prev`.

## Qubit-Level Dataset

- Keeps only rows with `qubit >= 0`.
- Keeps configured single-qubit properties:
  - `T1`;
  - `T2`;
  - `sx_error`;
  - `readout_error`;
  - `prob_meas0_prep1`;
  - `prob_meas1_prep0`.
- Excludes edge-level properties such as `cz_error_i_j`.
- Pivots from long to wide format using this key:
  - `backend`;
  - `qubit`;
  - `model_time`.
- Uses the median if duplicate property measurements exist for the same key.
- Adds `calibration_lag_hours`.
- Merges current and rolling environmental features.
- Sorts by backend, qubit, and timestamp.

## Qubit Calibration History Features

- Builds last-observation features for configured calibration columns:
  - `T1`;
  - `T2`;
  - `sx_error`;
  - `readout_error`.
- Adds:
  - `<col>_last_obs`;
  - `<col>_last_obs_age_hours`;
  - `<col>_was_missing`.
- Uses only previous observations from the same backend/qubit group.
- Masks last observations older than `history_max_age`, default `24h`.
- Does not overwrite the originally observed values.
- Keeps missing values explicit so downstream models can distinguish real observations from missing measurements.

## Edge-Level Dataset

- Detects edge-level properties with this regex:
  - `^cz_error_(\d+)_(\d+)$`.
- Extracts the directed qubit pair:
  - `q1`;
  - `q2`.
- Builds a table keyed by:
  - `backend`;
  - `q1`;
  - `q2`;
  - `model_time`.
- Computes `cz_error` as the median if duplicate rows exist.
- Adds `calibration_lag_hours`.
- Merges current and rolling environmental features.
- Keeps edge-level data separate from qubit-level data to avoid mixing node features and edge features.


## 24-Hour Fault Prediction Labels

- If `fault_prediction.enabled = true`, creates a dedicated fault prediction dataset.
- Uses configured target columns:
  - `sx_error`;
  - `readout_error`.
- For each qubit-level row, looks at future observations from the same backend/qubit within `24h`.
- Computes:
  - `sx_error_future_max_24h`;
  - `sx_error_future_obs_count_24h`;
  - `sx_error_fault_24h`;
  - `readout_error_future_max_24h`;
  - `readout_error_future_obs_count_24h`;
  - `readout_error_fault_24h`.
- Creates `fault_24h` as the maximum of the individual fault labels.
- Leaves the label missing if no future target is observed within the horizon.
- Drops rows without a future label when configured to do so.

## Fault Threshold

- Default method:
  - `threshold_method = "quantile"`.
- Default threshold:
  - 95th percentile of the observed target distribution.
- Saves the thresholds in the JSON report.
- Methodological note:
  - thresholds are computed from past observations only;
  - stricter modelling can still define thresholds inside downstream temporal validation.

## Low-Information Column Removal

- Drops fully missing columns.
- Drops constant columns.
- Preserves protected columns, including:
  - requested qubit-level properties;
  - `calibration_lag_hours`;
  - `cz_error`;
  - `fault_24h`.
- Records dropped all-missing and constant columns in the report.

## Optional Scaling

- Scaling is disabled by default.
- If enabled, creates standardized columns with configurable suffix, default `_z`.
- Uses numeric non-ID features.
- Excludes:
  - ID columns;
  - missing indicators;
  - fault labels.
- With the current configuration, scaling is intentionally left to the modelling pipeline.

## Outputs

- Writes datasets in both parquet and CSV format.
- Qubit-level outputs:
  - `qiskit_qubit_snapshots.parquet`;
  - `qiskit_qubit_snapshots.csv`.
- Edge-level outputs:
  - `qiskit_edge_snapshots.parquet`;
  - `qiskit_edge_snapshots.csv`.
- Fault prediction outputs:
  - `qiskit_fault_prediction_24h.parquet`;
  - `qiskit_fault_prediction_24h.csv`.
- JSON report:
  - `qiskit_calibration_drift_report.json`.

## JSON Report

- Stores raw dataset metadata.
- Stores raw and processed missingness.
- Stores output dataset shapes.
- Stores backend and property counts.
- Stores low-information columns that were removed.
- Stores fault thresholds.
- Stores a theoretical rationale for preprocessing choices.

## What the Script Intentionally Does Not Do

- Does not encode `backend` as numeric.
- Does not convert `model_time` into final numeric time features.
- Does not globally impute every missing value.
- Does not scale all features by default.
- Does not perform random splitting.
- Does not clip outliers by default.
- Does not use future columns as model features.

## Why Backend, Datetime, and Missing Values Remain in the CSVs

- `backend` is a categorical variable useful for EDA, grouping, fixed effects, and later one-hot encoding.
- `model_time` is needed for chronological ordering, temporal splitting, and time-series cross-validation.
- Missing values represent real absence of measurement, especially for sparse calibration metrics such as T1/T2 and readout.
- Imputation and scaling should happen inside modelling pipelines after the temporal split to avoid leakage.
- The generated CSVs are observational and analysis-ready, not final numeric design matrices for every algorithm.

## Recommended Use in Modelling

- For PCA, UMAP, and clustering:
  - select numeric features;
  - remove or impute missing values;
  - standardize features.
- For regression:
  - choose an observable target such as `sx_error`, `readout_error_last_obs`, or `cz_error`;
  - use Lasso/Ridge with imputation and scaling inside an sklearn pipeline.
- For fault prediction:
  - use `fault_24h` as the target;
  - exclude every `*_future_*` column;
  - exclude auxiliary `*_fault_24h` labels when predicting `fault_24h`;
  - use temporal splitting, not random splitting.
- For graph analysis:
  - use the edge-level dataset;
  - build the graph from `q1` and `q2`;
  - use `cz_error` as an edge weight or edge metric.

## Rationale del preprocessing

- **Unità statistica qubit**: la tabella qubit usa `(backend, qubit, observed_time)` come unità perché T1, T2, `sx_error` e `readout_error` sono parametri di calibrazione a singolo qubit.
- **Tabella separata per gli edge**: gli errori CZ sono tenuti in una tabella distinta perché ogni `cz_error_i_j` appartiene a un coupling a due qubit, non a una riga di singolo qubit.
- **Nessuna sovrascrittura dei target**: i valori calibrati osservati non vengono sovrascritti da imputazioni; le feature di ultima osservazione vengono aggiunte separatamente con una colonna di età, così i modelli possono tenere conto della *staleness* della misura.
- **Fill causale dell'ambiente**: i valori ambientali sono allineati per backend e timestamp e forward-filled solo entro una finestra causale limitata, impedendo che informazioni future entrino nelle feature.
- **Rolling con `closed='left'`**: i riepiloghi al tempo *t* usano solo osservazioni strettamente precedenti a *t*.
- **Fault label future**: le etichette di guasto sono definite dagli errori osservati nelle 24h successive; con la soglia quantile predefinita si tratta di una definizione descrittiva di errore critico, non di un parametro di modello fittato.
- **Outlier clipping disabilitato di default**: gli errori estremi possono essere eventi di degradazione genuini, scientificamente rilevanti per la fault tolerance.

import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_and_clean_fault_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load fault prediction dataset view, setup temporal indices and remove 
    all features that refer to the future to avoid Data Leakage.
    """

    df = pd.read_parquet(file_path)
    print(f"Loading dataset view: {file_path}")

    # Remove all features that contain '_future_' excluding the target itself
    future_cols = [col for col in df.columns if '_future_' in col]
    specific_fault_cols = [col for col in df.columns if '_fault_24h' in col and col != 'fault_24h']
    leakage_columns = future_cols + specific_fault_cols

    # Drop leakage columns
    df_cleaned = df.drop(columns=leakage_columns)
    
    # Drop redundant/sparse columns 
    # T1, T2, prob_meas... are 95% NaNs. Useful values are stored in T1_last_obs and T1_last_obs_age_hours.
    nan_ratios = df_cleaned.isna().mean()
    sparse_cols = nan_ratios[nan_ratios > 0.50].index.tolist()
    df_cleaned = df_cleaned.drop(columns=sparse_cols)
    
    # Drop collinear features discovered by EDA (VIF = inf)
    collinear_cols = [c for c in df_cleaned.columns if '_count_' in c]

    # Drop instantaneous error targets (we must use _last_obs instead to avoid leakage)
    instant_targets = ['sx_error', 'readout_error', 'prob_meas0_prep1', 'prob_meas1_prep0']
    collinear_cols += [c for c in instant_targets if c in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(columns=collinear_cols)

    # Further VIF reduction: for each environmental variable, the rolling windows
    # 6h/12h/24h produce highly correlated features (VIF > 500 for pressure/temperature).
    # Strategy: keep only {var}, {var}_mean_prev_24h, {var}_std_prev_24h per variable.
    # This preserves the current level, the recent trend, and the recent variability.
    env_vars = ['temperature_c', 'pressure_hpa', 'humidity_pct', 'bz_gsm_nt', 'neutron_flux']
    keep_suffixes = {'_mean_prev_24h', '_std_prev_24h'}
    redundant_rolling = []
    for var in env_vars:
        for col in df_cleaned.columns:
            if col.startswith(var + '_') and col != var:
                suffix = col[len(var):]
                if suffix not in keep_suffixes:
                    redundant_rolling.append(col)
    df_cleaned = df_cleaned.drop(columns=redundant_rolling)

    # Cast model_time to pdatetime and sort chronologically
    df_cleaned['model_time'] = pd.to_datetime(df_cleaned['model_time'], utc=True)
    df_cleaned = df_cleaned.sort_values(by='model_time').reset_index(drop=True)
    
    print(f"Covered period: from {df_cleaned['model_time'].min()} to {df_cleaned['model_time'].max()}")
    print(f"Dataset shape: {df_cleaned.shape}")

    # Verify imbalance
    if 'fault_24h' in df_cleaned.columns:
        faults = df_cleaned['fault_24h'].sum()
        total = len(df_cleaned)
        fault_rate = (faults / total) * 100

        print(f"No faults: {total - faults}")
        print(f"Faults: {faults} ({fault_rate:.2f}%)")
        if fault_rate < 10:
            print("The dataset is heavily imbalanced")
            
    return df_cleaned



def build_preprocessing_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """
    Data cleaning pipeline
    Return a ColumnTransformer that manage imputation, categorical variables encoding and standardization
    """
    
    target = 'fault_24h'
    id_columns = ['backend', 'qubit', 'model_time'] # removed as predictors
    
    categorical_features = ['backend']

    numeric_features = [col for col in df.columns 
                       if col not in id_columns and col != target]
    
    # Tranformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' # Drop other features
    )

    return preprocessor, numeric_features, categorical_features


def main() -> None:
    """Run a lightweight data preparation check on the fault prediction view."""
    dataset_path = Path("dataset/qiskit_fault_prediction_24h.parquet")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_path} does not exist. Run `python src/main.py` first."
        )

    df = load_and_clean_fault_data(dataset_path)
    _, numeric_features, categorical_features = build_preprocessing_pipeline(df)

    print("\nData preparation check completed.")
    print(f"Rows: {len(df)}")
    print(f"Columns after leakage/sparse cleanup: {df.shape[1]}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)} {categorical_features}")


if __name__ == "__main__":
    main()

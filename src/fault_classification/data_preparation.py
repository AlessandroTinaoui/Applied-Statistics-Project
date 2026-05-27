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
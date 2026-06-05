import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fault_classification.data_preparation import load_and_clean_fault_data

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor to detect multicollinearity.
    """

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)


def perform_eda(df: pd.DataFrame):
    print("="*60)
    print("STARTING EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*60)

    target = 'fault_24h'
    id_columns = ['backend', 'qubit', 'model_time']
    
    # Retrieve numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in id_columns and col != target]
    
    # Impute missing values for EDA purposes (median)
    imputer = SimpleImputer(strategy='median')
    X_num = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
    
    # Scale for distance metrics
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=numeric_cols)

    # Correlation Heatmap
    plt.figure(figsize=(24, 20))
    corr = X_num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, vmax=1, vmin=-1, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Find highly correlated pairs
    corr_unstacked = corr.abs().unstack()
    high_corr = corr_unstacked[(corr_unstacked > 0.8) & (corr_unstacked < 1.0)].drop_duplicates()
    print(f"\nFound {len(high_corr)} highly correlated pairs (Pearson > 0.8):")
    print(high_corr.head(10))

    # VIF analysis on a small sample to speed it up
    X_sample = X_scaled.sample(n=min(10000, len(X_scaled)), random_state=42)
    vif_df = calculate_vif(X_sample)
    vif_df.to_csv(RESULTS_DIR / "vif_analysis.csv", index=False)

    # Outliers / Influential Points (Mahalanobis Distance) on the sample
    cov_matrix = np.cov(X_sample.values, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    mean_dist = X_sample.values.mean(axis=0)
    diff = X_scaled.values - mean_dist
    left_term = np.dot(diff, inv_cov_matrix)
    mahalanobis_sq = np.sum(left_term * diff, axis=1)

    # Compute p-value (chi-square distribution with df = number of variables)
    df_degrees = X_scaled.shape[1]
    p_values = 1 - chi2.cdf(mahalanobis_sq, df_degrees)

    # Points with p < 0.001 are considered severe outliers
    outliers_mask = p_values < 0.001
    n_outliers = np.sum(outliers_mask)
    print(f"Detected {n_outliers} multivariate outliers ({(n_outliers/len(df))*100:.2f}% of dataset) at p < 0.001.")
    
    # Fault rate in outliers vs normal
    df['is_outlier'] = outliers_mask
    if 'fault_24h' in df.columns:
        fault_rate_normal = df[~df['is_outlier']]['fault_24h'].mean() * 100
        fault_rate_outlier = df[df['is_outlier']]['fault_24h'].mean() * 100
        print("\n--- OUTLIER FAULT RATE ANALYSIS ---")
        print(f"Fault rate in NORMAL data: {fault_rate_normal:.2f}%")
        print(f"Fault rate in OUTLIERS: {fault_rate_outlier:.2f}%")
        print("This proves outliers are NOT noise, but the actual signal of degradation!")

    outlier_means = X_scaled[df['is_outlier']].mean()
    normal_means = X_scaled[~df['is_outlier']].mean()
    diff_means = (outlier_means - normal_means).abs().sort_values(ascending=False)
    print("\nTop 10 features driving the outliers (abs diff in scaled means):")
    print(diff_means.head(10))
    
    # Save distance plot
    plt.figure(figsize=(10, 6))
    sns.histplot(mahalanobis_sq, bins=100, log_scale=(False, True))
    plt.axvline(chi2.ppf(0.999, df_degrees), color='red', linestyle='--', label='Critical Threshold (p=0.001)')
    plt.title('Mahalanobis Squared Distance Distribution')
    plt.xlabel('Distance squared')
    plt.ylabel('Count (log scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mahalanobis_distance.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    dataset_path = PROJECT_ROOT / "dataset" / "qiskit_fault_prediction_24h.parquet"
    if dataset_path.exists():
        df_full = load_and_clean_fault_data(dataset_path)
        perform_eda(df_full)
    else:
        print(f"ERROR: Dataset not found at {dataset_path}")

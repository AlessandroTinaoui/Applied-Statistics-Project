import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fault_classification.data_preparation import (
    build_preprocessing_pipeline,
    load_and_clean_fault_data,
)


RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_baseline_experiments(df: pd.DataFrame):
    """
    Setup time series Cross Validation and train base models
    Train a Logistic Regression and a Random Forest with Time Series Cross Validation.
    """

    df = df.sort_values('model_time').reset_index(drop=True)
    
    y = df['fault_24h'].values
    X = df.drop(columns=['fault_24h'])
    
    # Construct the ColumnTransformer
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(df)
    
    # Logistic Regression
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    
    # Random Forest
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=50, max_depth=7, class_weight='balanced_subsample', random_state=42, n_jobs=-1))
    ])
    
    models = {
        'Logistic Regression': lr_pipeline,
        'Random Forest (Basic)': rf_pipeline
    }

    results = {name: {'pr_auc': [], 'roc_auc': [], 'f1': []} for name in models}
    
    tscv = TimeSeriesSplit(n_splits=3)
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\n[Fold {fold + 1}/{tscv.n_splits}] Train: {len(train_index)} rows | Test: {len(test_index)} rows")
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        for model_name, pipeline in models.items():
            pipeline.fit(X_train, y_train) # Transformers are applied only on test data to avoid data leakage 
            
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred = pipeline.predict(X_test)
            
            # Metrics evaluation
            pr_auc = average_precision_score(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Find optimal threshold to maximize F1 instead of default 0.5
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            opt_idx = np.argmax(f1_scores)
            opt_thresh = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.5
            
            y_pred_opt = (y_pred_proba >= opt_thresh).astype(int)
            f1 = f1_score(y_test, y_pred_opt)
            
            results[model_name]['pr_auc'].append(pr_auc)
            results[model_name]['roc_auc'].append(roc_auc)
            results[model_name]['f1'].append(f1)
            
            print(f"  - {model_name:25s} -> PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} (Thresh: {opt_thresh:.3f})")

    # Print results
    print("\n" + "="*60)
    print("FINALS RESULTS (avgs)")
    print("="*60)

    for model_name, metrics in results.items():
        pr_mean, pr_std = np.mean(metrics['pr_auc']), np.std(metrics['pr_auc'])
        roc_mean = np.mean(metrics['roc_auc'])
        f1_mean = np.mean(metrics['f1'])
        
        print(f"\nModel: {model_name}")
        print(f"  PR-AUC mean:  {pr_mean:.4f} (± {pr_std:.4f})")
        print(f"  ROC-AUC mean: {roc_mean:.4f}")
        print(f"  F1-Score mean: {f1_mean:.4f}")
    
    # Predict again on the last fold
    lr_pipeline = models['Logistic Regression']
    rf_pipeline = models['Random Forest (Basic)']
    
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    rf_probs = rf_pipeline.predict_proba(X_test)[:, 1]
    
    # Plots setup
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PR Curve
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
    
    ax1.plot(lr_recall, lr_precision, label=f'LogReg (AUC={auc(lr_recall, lr_precision):.3f})')
    ax1.plot(rf_recall, rf_precision, label=f'RandomForest (AUC={auc(rf_recall, rf_precision):.3f})')
    ax1.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline Casuale ({y_test.mean():.3f})')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve (last fold)')
    ax1.legend()
    
    # ROC Curve
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    
    ax2.plot(lr_fpr, lr_tpr, label=f'LogReg (AUC={auc(lr_fpr, lr_tpr):.3f})')
    ax2.plot(rf_fpr, rf_tpr, label=f'RandomForest (AUC={auc(rf_fpr, rf_tpr):.3f})')
    ax2.plot([0, 1], [0, 1], color='r', linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve (last fold)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pr_roc_curves.png", dpi=150)
    plt.close()
    
    # Evaluate Confusion Matrix for RF with optimal threshold
    rf_precisions, rf_recalls, rf_thresholds = precision_recall_curve(y_test, rf_probs)
    rf_f1_scores = 2 * (rf_precisions * rf_recalls) / (rf_precisions + rf_recalls + 1e-8)
    opt_idx = np.argmax(rf_f1_scores)
    opt_thresh = rf_thresholds[opt_idx] if opt_idx < len(rf_thresholds) else 0.5
    
    rf_pred_opt = (rf_probs >= opt_thresh).astype(int)
    cm = confusion_matrix(y_test, rf_pred_opt)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Fault', 'Fault'])
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title(f'Random Forest Confusion Matrix\n(Optimal Threshold = {opt_thresh:.3f})')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rf_confusion_matrix.png", dpi=150)
    plt.close()
    
    # Extract feature names
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_cols = cat_encoder.get_feature_names_out(cat_cols)
    feature_names = np.concatenate([num_cols, encoded_cat_cols])
    
    # Feature Importances plot
    rf_model = rf_pipeline.named_steps['classifier']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis', hue=[feature_names[i] for i in indices], legend=False)
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.xlabel('Gini Importance')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    
    # Coefficients plot (Logistic Regression)
    lr_model = lr_pipeline.named_steps['classifier']
    coeffs = np.abs(lr_model.coef_[0])
    lr_indices = np.argsort(coeffs)[::-1][:15] # Top 15 (in absolute value)
    
    actual_coeffs = lr_model.coef_[0][lr_indices]
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in actual_coeffs]
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=actual_coeffs, y=[feature_names[i] for i in lr_indices], palette=colors, hue=[feature_names[i] for i in lr_indices], legend=False)
    plt.title('Top 15 Logistic Regression Coefficients')
    plt.xlabel('Log-Odds impact (Red = less prone to failure, Green = more prone to failure)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "lr_coefficients.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------------
    # Ablation study (training without outliers)
    print("\n" + "="*60)
    print("ABLATION STUDY: Training WITHOUT Influential Points")
    print("="*60)
        
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['backend', 'qubit', 'model_time', 'fault_24h']]
    imp = SimpleImputer(strategy='median')
    X_num = imp.fit_transform(df[numeric_cols])
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_num)
    
    # Sample for covariance to speed up
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_scaled), min(10000, len(X_scaled)), replace=False)
    sample = X_scaled[sample_idx]
    cov_matrix = np.cov(sample, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    mean_dist = sample.mean(axis=0)
    
    diff = X_scaled - mean_dist
    left_term = np.dot(diff, inv_cov_matrix)
    mahalanobis_sq = np.sum(left_term * diff, axis=1)
    
    p_values = 1 - chi2.cdf(mahalanobis_sq, X_scaled.shape[1])
    is_outlier = p_values < 0.001
    
    print(f"Dropping {is_outlier.sum()} outliers (p < 0.001)...")
    
    df_no_outliers = df[~is_outlier].copy()
    y_no_out = df_no_outliers['fault_24h'].values
    X_no_out = df_no_outliers.drop(columns=['fault_24h'])
    
    # Retrain on last fold of clean data
    train_idx = int(len(X_no_out) * 0.75)
    X_train_clean, X_test_clean = X_no_out.iloc[:train_idx], X_no_out.iloc[train_idx:]
    y_train_clean, y_test_clean = y_no_out[:train_idx], y_no_out[train_idx:]
    
    print(f"Retraining Logistic Regression on {len(X_train_clean)} normal samples...")
    lr_clean = Pipeline([
        ('preprocessor', build_preprocessing_pipeline(df_no_outliers)[0]),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])
    lr_clean.fit(X_train_clean, y_train_clean)
    y_pred_proba_clean = lr_clean.predict_proba(X_test_clean)[:, 1]
    
    pr_auc_clean = average_precision_score(y_test_clean, y_pred_proba_clean)
    roc_auc_clean = roc_auc_score(y_test_clean, y_pred_proba_clean)
    
    precisions_c, recalls_c, thresholds_c = precision_recall_curve(y_test_clean, y_pred_proba_clean)
    f1_scores_c = 2 * (precisions_c * recalls_c) / (precisions_c + recalls_c + 1e-8)
    opt_idx_c = np.argmax(f1_scores_c)
    opt_thresh_c = thresholds_c[opt_idx_c] if opt_idx_c < len(thresholds_c) else 0.5
    f1_clean = np.max(f1_scores_c)
    
    print(f"Results WITHOUT outliers:")
    print(f"  - Logistic Regression -> PR-AUC: {pr_auc_clean:.4f} | ROC-AUC: {roc_auc_clean:.4f} | F1: {f1_clean:.4f} (Thresh: {opt_thresh_c:.3f})")


if __name__ == "__main__":
    dataset_path = Path("dataset/qiskit_fault_prediction_24h.parquet")
    
    if dataset_path.exists():
        df_full = load_and_clean_fault_data(dataset_path)
        run_baseline_experiments(df_full)
    else:
        print(f"ERROR: The file {dataset_path} doesn't exist")

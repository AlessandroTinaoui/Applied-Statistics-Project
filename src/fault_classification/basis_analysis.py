import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fault_classification.data_preparation import (  # noqa: E402
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
            f1 = f1_score(y_test, y_pred)
            
            results[model_name]['pr_auc'].append(pr_auc)
            results[model_name]['roc_auc'].append(roc_auc)
            results[model_name]['f1'].append(f1)
            
            print(f"  - {model_name:25s} -> PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")

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
    

if __name__ == "__main__":
    dataset_path = Path("dataset/qiskit_fault_prediction_24h.parquet")
    
    if dataset_path.exists():
        df_full = load_and_clean_fault_data(dataset_path)
        run_baseline_experiments(df_full)
    else:
        print(f"ERROR: The file {dataset_path} doesn't exist")

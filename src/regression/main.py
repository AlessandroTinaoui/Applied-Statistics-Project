"""
main entry point for the regression analysis
"""

from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

pd.options.display.float_format = "{:.6f}".format

import matplotlib
matplotlib.use("Agg")

from src.regression.data_preparation  import prepare_data
from src.regression.eda               import run_eda, run_feature_selection
from src.regression.linear_model      import run_linear_models, detect_and_remove_outliers
from src.regression.regularization    import run_regularization
from src.regression.comparison        import run_comparison

DATASET_PATH = ROOT / "dataset" / "qiskit_qubit_snapshots.parquet"
OUTPUT_DIR   = ROOT / "output" / "regression"


def _banner(step: int, title: str):
    print("\n" + "═" * 80)
    print(f" {f'{step}. {title}':^78}")
    print("═" * 80)


def main():
    print("\n╔" + "═" * 78 + "╗")
    print(f"║ {'QUBITsnap REGRESSION: T1 DECOHERENCE TIME PREDICTION':^76} ║")
    print("╚" + "═" * 78 + "╝")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _banner(1, "DATA PREPARATION & CHRONOLOGICAL SPLIT")
    data = prepare_data(DATASET_PATH)

    _banner(3, "EXPLORATORY DATA ANALYSIS (EDA)")
    run_eda(data["train_df"], save_dir=OUTPUT_DIR / "eda")

    _banner(4, "OUTLIER DIAGNOSTICS & FILTERING (COOK'S DISTANCE)")
    X_clean, y_clean = detect_and_remove_outliers(data, save_dir=OUTPUT_DIR / "ols")
    data["X_train"] = X_clean
    data["y_train"] = y_clean

    _banner(5, "ORDINARY LEAST SQUARES (OLS) REGRESSION (ON CLEAN DATA)")
    ols_results = run_linear_models(data, save_dir=OUTPUT_DIR / "ols")

    _banner(6, "REGULARIZED MODELS (LASSO & RIDGE VIA GRID SEARCH)")
    reg_results = run_regularization(data, save_dir=OUTPUT_DIR / "regularization")

    _banner(7, "FINAL MODEL COMPARISON & METRICS SUMMARY")
    comparison_df = run_comparison(ols_results, reg_results, save_dir=OUTPUT_DIR / "comparison")
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\n" + "═" * 80)
    print(f" {'ANALYSIS COMPLETE':^78}")
    print("═" * 80)
    print(f"  ✓ Output directory : {OUTPUT_DIR}")
    print(f"  ✓ Summary table    : {OUTPUT_DIR / 'model_comparison.csv'}")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    main()

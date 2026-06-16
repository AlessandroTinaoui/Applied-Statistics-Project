# Report: Regression for $T_1$ Prediction

This document summarizes the rationale, methodological choices, and key results of the regression module built to predict the longitudinal coherence time ($T_1$) of superconducting qubits.

---

## 1. Workflow

The main script in src/regression orchestrates the analysis in these phases:

1.  **Data Preparation & Chronological Split**: Loads the qubit_snapshot view, computes the autoregressive lag `T1_prev`, applies the dynamic missingness filter, and performs a chronological split (80% train, 20% test) to prevent future data leakage.
2.  **Exploratory Data Analysis (EDA)**: Computes Pearson correlations on the training set and generates _binned profile_ plots (boxplots for categorical variables and binned means with 95% confidence intervals for numeric variables) to visualize bivariate relationships with $T_1$. Then analyzes the distribution of $T_1$ (mean, median, skewness, kurtosis) and groups the target by backend.
3.  **Outlier Diagnostics & Filtering**: Fits a preliminary OLS model to calculate Cook's distance and removes highly influential observations ($D_i > 4/N$).
4.  **Ordinary Least Squares (OLS) on Clean Data**: Fits the final OLS model, extracts coefficients, and calculates Variance Inflation Factors (VIF) to assess multicollinearity.
5.  **Regularized Models (LASSO & Ridge)**: Optimizes the regularization strength $\alpha$ using `GridSearchCV` with 5-fold temporal cross-validation.
6.  **Final Model Comparison**: Evaluates and compares performance metrics ($R^2$, RMSE, MAE) on the test set, saving the summary table.

---

## 2. Decisions & Critical Insights

### A. Dynamic Missingness Filter (30% Threshold)

- **The Decision**: Feature columns with a missingness rate exceeding $30\%$ are automatically discarded before any imputation or preprocessing.
- **The Rationale**: Imputing columns with high missingness using the median (via the pipeline's `SimpleImputer`) introduces massive artificial noise. It flattens the variable's true variance and distorts standard scaling, leading to an underestimation of standard deviation.
- **Effect**: Columns like `prob_meas0_prep1` (~82.3% missingness) and `prob_meas1_prep0` (~81.8% missingness) are dynamically detected and dropped at the start of the pipeline, safeguarding the scaling process.

### B. Binarization of `sx_error`

- **The Decision**: The single-qubit gate error (`sx_error_last_obs`) is binarized into two classes in [data_preparation.py](file:///Users/simo/Desktop/uni/corsi/applied_stat/Applied-Statistics-Project/src/regression/data_preparation.py):
  - `"low_error"` ($\le 0.00025$, representing excellent gate calibration).
  - `"high_or_failed"` ($> 0.00025$, grouping standard-to-high gate errors and failed calibrations).
- **The Rationale**: The raw gate error has a highly bimodal distribution: 99% of values are extremely small ($[0.0001, 0.001]$), representing normal operation, while 1% of values are exactly `1.0` (failed calibrations).
- **Effect**: Binarization increased the test $R^2$ and provided a direct physical interpretation: a degraded or failed calibration (`high_or_failed`) inflicts a net penalty of approximately **$-8.11\,\mu\text{s}$** on the $T_1$ time.

---

## 3. Outlier Diagnostics via Cook's Distance ($4/N$)

To ensure the validity of hypothesis testing (t-tests, F-tests, and coefficient confidence intervals), we verified the normality of residuals by diagnosing influential outliers.

### A. Outlier Removal Strategies

We evaluated the impact of filtering outliers based on Cook's Distance ($D_i > 4/N$) on the OLS model:

| Strategy                            | Training Obs | Train $R^2$ | Test $R^2$ | Shapiro-Wilk (Stat) |   Shapiro-Wilk (p-value)   |
| :---------------------------------- | :----------: | :---------: | :--------: | :-----------------: | :------------------------: |
| **Baseline** (No Filter)            |     8184     |   0.6053    | **0.6231** |       0.9832        |   $6.45 \times 10^{-24}$   |
| **Cook's Distance $> 4/N$ removed** |     7779     | **0.6909**  |   0.6160   |     **0.9954**      | **$1.75 \times 10^{-11}$** |

- **The Optimal Compromise**: The Cook's D filter ($4/N \approx 0.00051$) removes **4.95%** of the most influential training points. This increases the training $R^2$ from **0.605** to **0.691** and significantly improves residual distribution without causing truncation bias (test $R^2$ remains stable at **0.616**).

### B. Shapiro-Wilk Normality Analysis: Statistical vs. Practical Significance

1.  **The $W$ Statistic**: Measures how closely the sample residuals align with a theoretical normal distribution (where $W = 1.0$ is perfect normality). A value of **0.9954** indicates that our residuals are **99.54% aligned with a perfect Gaussian curve**.
2.  **The Tiny p-value ($1.75 \times 10^{-11}$)**: For large sample sizes ($N = 5000$ for the test), the Shapiro-Wilk test has **extremely high statistical power**. Consequently, even microscopic, non-critical deviations from perfect normality lead to a formal rejection of the null hypothesis ($p < 0.05$).
3.  **Practical Verdict**: For all practical purposes (e.g., standard errors, confidence intervals), the residuals are **normal enough**, as visually confirmed by their tight alignment along the diagonal of the Q-Q Plot.

---

## 4. Regression Results & Comparison

The performance metrics of the models trained on the clean dataset (20 features total after dummy encoding) are summarized below:

| Model        | $R^2$ (Train) | $R^2$ (Test) | RMSE (Test) | MAE (Test)  | Optimal $\alpha$ |
| :----------- | :-----------: | :----------: | :---------: | :---------: | :--------------: |
| **Base OLS** |  **0.6909**   |  **0.6160**  | **54.5345** | **41.2596** |        -         |
| **LASSO**    |  **0.6907**   |  **0.6172**  | **54.4491** | **41.1739** |      0.1833      |
| **Ridge**    |  **0.6909**   |  **0.6154**  | **54.5732** | **41.3184** |     29.7635      |

### LASSO Coefficients (Physical Interpretation)

The coefficients selected by LASSO (optimized at $\alpha \approx 0.183$) align closely with quantum hardware principles:

- **`T1_prev`** ($\beta = +45.12$): The previous calibration value is the dominant predictor (strong autoregressive component).
- **`backend_ibm_kingston`** ($\beta = +37.43$) / **`backend_ibm_marrakesh`** ($\beta = +15.19$): Capture systematic baseline shifts in chip fabrication quality relative to the reference backend (`ibm_fez`).
- **`T2_last_obs`** ($\beta = +19.49$): Strong positive relation. Qubits with higher dephasing times ($T_2$) consistently display higher relaxation times ($T_1$).
- **`sx_error_class_last_obs_low_error`** ($\beta = +8.11$): A well-calibrated, low-error single-qubit gate is associated with longer $T_1$ times.
- **`temperature_c_std_prev_24h`** ($\beta = +2.19$): Weak positive correlation, likely reflecting active cryostat cooling cycle responses.
- **`solar_zenith_deg`** ($\beta = +1.99$): Identifies a subtle diurnal trend.
- **`readout_error_last_obs`** ($\beta = -1.57$): Higher readout errors correlate with shorter $T_1$ times.
- **`neutron_flux_mean_prev_24h`** ($\beta = +1.25$): Minor environmental correlation.
- **`calibration_lag_hours`** ($\beta = -0.89$): Indicates a slight degradation (drift) in qubit performance as time passes since the last physical calibration.

---

## 5. Why Ridge and LASSO do not Bring Significant Benefits Over OLS?

All three models yield virtually identical test $R^2$ values (within $0.1\%$). This equivalence is statistically expected and is explained by three main factors:

### 1. Absence of Multicollinear

Ridge regularization (L2 penalty) stabilizes coefficient estimation in the presence of strong multicollinearity, which inflates the variance of OLS estimates.
We calculated the **Variance Inflation Factors (VIF)** for our final model:

- The maximum VIF is only **5.33** (for `pressure_hpa_mean_prev_24h`).
- All other VIFs are well below the standard threshold of **10.0** (most are under 3.0).
  Because the feature set lacks severe multicollinearity, OLS estimates are already stable, leaving no collinearity for Ridge to correct.

### 2. High Sample-to-Feature Ratio ($N/p$)

Regularization is crucial in high-dimensional settings where the number of parameters $p$ is close to or exceeds the number of observations $N$ ($N \approx p$ or $N < p$).
In our setup:

- The training set size post-outliers is $N \approx 7800$.
- The total number of features (including dummy variables) is $p = 20$.
  The ratio $\frac{N}{p} \approx 390$ is very high. With such an abundance of degrees of freedom, the variance of the OLS estimator is already close to zero. Introducing bias (via Ridge or LASSO) does not yield any reduction in generalization error (test $R^2$ or RMSE).

### 3. Prior Manual Feature Selection

LASSO (L1 penalty) performs feature selection by shrinking coefficients of uninformative variables to zero. However, our manual feature selection already removed the noisy meteorological variables and redundant rolling windows.
LASSO only shrank **4 out of 20 features** to zero (e.g., standard deviations of humidity and geomagnetic z-field). This confirms that our pre-selected feature set was already highly optimized, leaving very little pruning for LASSO to perform.

**Conclusion**: On a clean, well-specified, and large dataset without multicollinearity, OLS is the minimum-variance unbiased estimator (Gauss-Markov theorem). Regularization does not outperform OLS because there is no overfitting, collinearity, or noise to correct.

---

## 6. Key Takeaways

1.  **Physics vs. External Environment**: External weather conditions (temperature, pressure, humidity) do not directly affect the qubit, which operates at $15\text{ mK}$ inside a dilution refrigerator. Minor correlations (`solar_zenith_deg`) capture indirect lab-level temperature or line instabilities.
2.  **Hardware Coherence Memory**: Qubit coherence is highly conservative. The best predictor is the qubit's previous calibration state (`T1_prev`, $\beta = +45.12$).
3.  **OLS is the Optimal Choice**: OLS is robust and optimal. Ridge and LASSO validate the stability of our OLS coefficients.
4.  **Linear Regression vs. Time-Series Autoregression**: While the inclusion of the lag-1 target variable (`T1_prev`) introduces a dynamic autoregressive effect, we restricted our modeling to the standard static multivariate linear regression framework (OLS, LASSO, Ridge). A formal time-series autoregressive analysis falls outside the curriculum of this applied statistics course. Treating `T1_prev` as a static predictor allowed us to leverage the strong historical coherence memory while remaining strictly within the course's framework.

"""
Seaborn Regression Plots — Beginner to Advanced
===============================================

File:
03-Python-for-Machine-Learning/04-Seaborn/regression-plots.py

Purpose:
A practical guide to visualizing relationships between numerical
variables and understanding regression behavior using Seaborn.

Topics Covered:
1. Regression fundamentals
2. Scatter plot with regression line
3. sns.regplot()
4. Confidence intervals
5. Scatter-point customization
6. Regression without confidence interval
7. Linear regression
8. Polynomial regression
9. Robust regression
10. Logistic regression visualization
11. Residual plots
12. Residual analysis
13. Residual distribution
14. Actual vs predicted values
15. Regression by category
16. Hue-based regression
17. Faceted regression
18. sns.lmplot()
19. sns.residplot()
20. Log-transformed regression
21. Nonlinear relationships
22. Outlier impact
23. Train/test regression visualization
24. Feature-target regression analysis
25. Multiple feature analysis
26. Regression diagnostics
27. R² visualization
28. MAE / RMSE visualization
29. Reusable regression functions
30. Saving regression plots

Important:
- A regression line summarizes an estimated relationship.
- Correlation does not establish causation.
- Confidence intervals describe uncertainty around the estimated
regression relationship; they are not prediction intervals.
- Visualization alone does not validate all regression assumptions.
- ML metrics in this file are calculated when demonstrated.
- Synthetic examples are clearly labeled.

Requirements:
pip install numpy pandas matplotlib seaborn scikit-learn

Python:
Recommended Python 3.10+
"""

# ============================================================

# 1. IMPORT LIBRARIES

# ============================================================

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.linear_model import (
LinearRegression,
LogisticRegression,
)
from sklearn.metrics import (
mean_absolute_error,
mean_squared_error,
r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

sns.set_theme(
style="whitegrid",
context="notebook",
)

RANDOM_STATE = 42

# ============================================================

# 2. HELPER FUNCTION

# ============================================================

def section(title: str) -> None:
"""Print a readable section separator."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

# ============================================================

# 3. CREATE SYNTHETIC REGRESSION DATA

# ============================================================

def create_regression_data(
n_samples: int = 250,
random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
"""
Create a synthetic regression dataset.

```
Variables:
    study_hours:
        Number of hours studied.

    exam_score:
        Numerical outcome related to study hours.

    department:
        Categorical grouping variable.

    experience:
        Additional numerical feature.

    income:
        Numerical target-like variable.

The dataset is synthetic and intended only for learning.
"""
rng = np.random.default_rng(random_state)

study_hours = rng.uniform(
    1,
    12,
    n_samples,
)

experience = rng.uniform(
    0,
    15,
    n_samples,
)

department = rng.choice(
    [
        "Engineering",
        "Data Science",
        "Marketing",
    ],
    size=n_samples,
    p=[0.45, 0.30, 0.25],
)

department_effect = np.select(
    [
        department == "Engineering",
        department == "Data Science",
        department == "Marketing",
    ],
    [
        4,
        7,
        1,
    ],
    default=0,
)

exam_score = (
    45
    + 3.8 * study_hours
    + 0.8 * experience
    + department_effect
    + rng.normal(0, 5, n_samples)
)

income = (
    30000
    + 2500 * experience
    + 900 * study_hours
    + department_effect * 1000
    + rng.normal(0, 8000, n_samples)
)

return pd.DataFrame(
    {
        "study_hours": study_hours.round(2),
        "experience": experience.round(2),
        "department": department,
        "exam_score": exam_score.round(2),
        "income": income.round(2),
    }
)
```

# ============================================================

# 4. BASIC DATASET INSPECTION

# ============================================================

def inspect_dataset(
df: pd.DataFrame,
) -> None:
"""Inspect the regression dataset."""
section("4. BASIC DATASET INSPECTION")

```
print(df.head())

print("\nShape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nNumerical summary:")
print(df.describe())
```

# ============================================================

# 5. BASIC SCATTER PLOT

# ============================================================

def basic_scatter_plot(
df: pd.DataFrame,
) -> None:
"""Visualize the relationship before fitting regression."""
section("5. BASIC SCATTER PLOT")

```
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="study_hours",
    y="exam_score",
)

plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 6. BASIC REGRESSION PLOT

# ============================================================

def basic_regression_plot(
df: pd.DataFrame,
) -> None:
"""
sns.regplot() adds a fitted regression line to a scatter plot.
"""
section("6. BASIC REGRESSION PLOT")

```
plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="study_hours",
    y="exam_score",
)

plt.title("Study Hours vs Exam Score with Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 7. REGRESSION WITHOUT CONFIDENCE INTERVAL

# ============================================================

def regression_without_ci(
df: pd.DataFrame,
) -> None:
"""Disable the confidence interval for a cleaner regression line."""
section("7. REGRESSION WITHOUT CONFIDENCE INTERVAL")

```
plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="study_hours",
    y="exam_score",
    ci=None,
)

plt.title(
    "Study Hours vs Exam Score — Regression Line Only"
)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 8. CONFIDENCE INTERVAL

# ============================================================

def regression_confidence_interval(
df: pd.DataFrame,
) -> None:
"""
Demonstrate a regression line with an uncertainty band.

```
ci:
    The confidence interval level used by Seaborn for the
    estimated regression relationship.
"""
section("8. REGRESSION CONFIDENCE INTERVAL")

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="study_hours",
    y="exam_score",
    ci=95,
)

plt.title(
    "Regression with 95% Confidence Interval"
)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 9. CUSTOM SCATTER POINTS

# ============================================================

def customized_regression_plot(
df: pd.DataFrame,
) -> None:
"""Customize scatter-point appearance."""
section("9. CUSTOMIZED REGRESSION PLOT")

```
plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="study_hours",
    y="exam_score",
    scatter_kws={
        "alpha": 0.55,
        "s": 45,
    },
    line_kws={
        "linewidth": 2.5,
    },
)

plt.title("Customized Regression Visualization")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 10. REGRESSION BY CATEGORY

# ============================================================

def regression_by_category(
df: pd.DataFrame,
) -> None:
"""
Compare category-specific regression relationships.

```
Each category receives its own regression line.
"""
section("10. REGRESSION BY CATEGORY")

plt.figure(figsize=(11, 7))

sns.lmplot(
    data=df,
    x="study_hours",
    y="exam_score",
    hue="department",
    height=6,
    aspect=1.5,
    ci=95,
)

plt.title("Regression Relationship by Department")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 11. FACETED REGRESSION

# ============================================================

def faceted_regression(
df: pd.DataFrame,
) -> None:
"""Create separate regression panels for each category."""
section("11. FACETED REGRESSION")

```
grid = sns.lmplot(
    data=df,
    x="study_hours",
    y="exam_score",
    col="department",
    height=5,
    aspect=0.9,
    ci=95,
)

grid.set_axis_labels(
    "Study Hours",
    "Exam Score",
)

grid.set_titles(
    "{col_name}"
)

plt.tight_layout()
plt.show()
```

# ============================================================

# 12. HUE + STYLE SCATTER

# ============================================================

def hue_style_relationship(
df: pd.DataFrame,
) -> None:
"""Use category encoding to make relationships easier to compare."""
section("12. HUE + STYLE RELATIONSHIP")

```
plt.figure(figsize=(11, 7))

sns.scatterplot(
    data=df,
    x="study_hours",
    y="exam_score",
    hue="department",
    style="department",
    alpha=0.75,
)

plt.title(
    "Study Hours vs Exam Score by Department"
)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 13. LINEAR REGRESSION WITH SCIKIT-LEARN

# ============================================================

def sklearn_linear_regression(
df: pd.DataFrame,
) -> tuple[LinearRegression, np.ndarray, np.ndarray]:
"""
Fit a linear regression model using scikit-learn.

```
Returns:
    model
    predictions
    residuals
"""
section("13. SCIKIT-LEARN LINEAR REGRESSION")

X = df[["study_hours"]]
y = df["exam_score"]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
residuals = y.to_numpy() - predictions

print("Intercept:")
print(model.intercept_)

print("\nCoefficient:")
print(model.coef_[0])

print("\nR²:")
print(model.score(X, y))

return (
    model,
    predictions,
    residuals,
)
```

# ============================================================

# 14. PLOT SCIKIT-LEARN REGRESSION

# ============================================================

def plot_sklearn_regression(
df: pd.DataFrame,
model: LinearRegression,
) -> None:
"""Plot observations and the fitted scikit-learn model."""
section("14. PLOT SCIKIT-LEARN REGRESSION")

```
X = df[["study_hours"]]
y = df["exam_score"]

x_sorted = np.linspace(
    X["study_hours"].min(),
    X["study_hours"].max(),
    200,
)

predictions = model.predict(
    x_sorted.reshape(-1, 1)
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="study_hours",
    y="exam_score",
    alpha=0.6,
)

plt.plot(
    x_sorted,
    predictions,
    linewidth=2.5,
    label="Linear regression",
)

plt.title(
    "Scikit-Learn Linear Regression"
)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.tight_layout()
plt.show()
```

# ============================================================

# 15. POLYNOMIAL REGRESSION

# ============================================================

def polynomial_regression(
df: pd.DataFrame,
) -> None:
"""
Demonstrate polynomial regression.

```
Polynomial regression can model nonlinear relationships while
remaining linear in its learned coefficients.
"""
section("15. POLYNOMIAL REGRESSION")

X = df[["study_hours"]]
y = df["exam_score"]

polynomial_model = make_pipeline(
    PolynomialFeatures(
        degree=2,
        include_bias=False,
    ),
    LinearRegression(),
)

polynomial_model.fit(
    X,
    y,
)

x_grid = np.linspace(
    X["study_hours"].min(),
    X["study_hours"].max(),
    300,
)

predictions = polynomial_model.predict(
    x_grid.reshape(-1, 1)
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="study_hours",
    y="exam_score",
    alpha=0.55,
)

plt.plot(
    x_grid,
    predictions,
    linewidth=2.5,
    label="Degree-2 polynomial",
)

plt.title("Polynomial Regression")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.tight_layout()
plt.show()
```

# ============================================================

# 16. POLYNOMIAL REGRESSION WITH SEABORN

# ============================================================

def seaborn_polynomial_regression(
df: pd.DataFrame,
) -> None:
"""
Use order=2 in sns.regplot() for polynomial regression.

```
This is useful for visualization, but model validation should
still be performed separately using an appropriate ML workflow.
"""
section("16. SEABORN POLYNOMIAL REGRESSION")

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="study_hours",
    y="exam_score",
    order=2,
    ci=None,
)

plt.title(
    "Polynomial Regression — Degree 2"
)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 17. ROBUST REGRESSION

# ============================================================

def robust_regression(
df: pd.DataFrame,
) -> None:
"""
Demonstrate robust regression.

```
Robust regression can reduce the influence of observations that
strongly affect ordinary least-squares regression.
"""
section("17. ROBUST REGRESSION")

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="study_hours",
    y="exam_score",
    robust=True,
    ci=None,
)

plt.title(
    "Robust Regression"
)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 18. OUTLIER IMPACT

# ============================================================

def regression_outlier_impact(
df: pd.DataFrame,
) -> None:
"""
Demonstrate how extreme observations can affect a fitted line.
"""
section("18. REGRESSION OUTLIER IMPACT")

```
clean = df[
    [
        "study_hours",
        "exam_score",
    ]
].copy()

outlier = pd.DataFrame(
    {
        "study_hours": [11.5],
        "exam_score": [20],
    }
)

with_outlier = pd.concat(
    [
        clean,
        outlier,
    ],
    ignore_index=True,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 6),
)

sns.regplot(
    data=clean,
    x="study_hours",
    y="exam_score",
    ax=axes[0],
    ci=None,
)

axes[0].set_title(
    "Regression Without Extreme Observation"
)

sns.regplot(
    data=with_outlier,
    x="study_hours",
    y="exam_score",
    ax=axes[1],
    ci=None,
)

axes[1].set_title(
    "Regression With Extreme Observation"
)

fig.tight_layout()
plt.show()
```

# ============================================================

# 19. RESIDUAL PLOT

# ============================================================

def residual_plot(
df: pd.DataFrame,
) -> None:
"""
Residual:
residual = observed value - predicted value

```
A residual plot helps inspect whether systematic patterns remain
after fitting a regression relationship.
"""
section("19. RESIDUAL PLOT")

plt.figure(figsize=(10, 6))

sns.residplot(
    data=df,
    x="study_hours",
    y="exam_score",
    lowess=True,
)

plt.axhline(
    0,
    linewidth=1.5,
    linestyle="--",
)

plt.title(
    "Residual Plot — Study Hours vs Exam Score"
)
plt.xlabel("Study Hours")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()
```

# ============================================================

# 20. RESIDUAL DISTRIBUTION

# ============================================================

def residual_distribution(
residuals: np.ndarray,
) -> None:
"""Visualize the distribution of regression residuals."""
section("20. RESIDUAL DISTRIBUTION")

```
residual_series = pd.Series(
    residuals,
    name="residual",
)

plt.figure(figsize=(10, 6))

sns.histplot(
    residual_series,
    bins=25,
    kde=True,
)

plt.axvline(
    0,
    linewidth=1.5,
    linestyle="--",
)

plt.title(
    "Residual Distribution"
)
plt.xlabel("Residual")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

# ============================================================

# 21. ACTUAL VS PREDICTED

# ============================================================

def actual_vs_predicted(
y_true: pd.Series,
predictions: np.ndarray,
) -> None:
"""
Compare actual and predicted values.

```
Points closer to the diagonal generally indicate smaller
prediction errors.
"""
section("21. ACTUAL VS PREDICTED")

plt.figure(figsize=(9, 7))

sns.scatterplot(
    x=y_true,
    y=predictions,
    alpha=0.7,
)

minimum = min(
    y_true.min(),
    predictions.min(),
)

maximum = max(
    y_true.max(),
    predictions.max(),
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2,
    label="Perfect prediction",
)

plt.title(
    "Actual vs Predicted Values"
)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.legend()
plt.tight_layout()
plt.show()
```

# ============================================================

# 22. REGRESSION METRICS

# ============================================================

def regression_metrics(
y_true: pd.Series,
predictions: np.ndarray,
) -> dict[str, float]:
"""Calculate common regression evaluation metrics."""
section("22. REGRESSION METRICS")

```
mae = mean_absolute_error(
    y_true,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        predictions,
    )
)

r2 = r2_score(
    y_true,
    predictions,
)

metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
}

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

return metrics
```

# ============================================================

# 23. METRIC VISUALIZATION

# ============================================================

def metric_visualization(
metrics: dict[str, float],
) -> None:
"""
Visualize regression metrics.

```
Note:
    MAE/RMSE and R² have different units/scales, so this chart
    is primarily illustrative rather than a direct comparison
    of metric magnitude.
"""
section("23. METRIC VISUALIZATION")

metric_names = list(metrics.keys())
metric_values = list(metrics.values())

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)

for ax, name, value in zip(
    axes,
    metric_names,
    metric_values,
):
    sns.barplot(
        x=[name],
        y=[value],
        ax=ax,
    )

    ax.set_title(name)
    ax.set_xlabel("")
    ax.set_ylabel("Value")

fig.suptitle(
    "Regression Evaluation Metrics"
)

fig.tight_layout()
plt.show()
```

# ============================================================

# 24. DIABETES REGRESSION DATASET

# ============================================================

def diabetes_regression_plot() -> None:
"""
Use scikit-learn's diabetes dataset.

```
The dataset is used here for visualization and educational
regression analysis.
"""
section("24. DIABETES REGRESSION DATASET")

diabetes = load_diabetes(
    as_frame=True
)

X = diabetes.data
y = diabetes.target

plot_data = X.copy()
plot_data["target"] = y

plt.figure(figsize=(10, 6))

sns.regplot(
    data=plot_data,
    x="bmi",
    y="target",
    scatter_kws={
        "alpha": 0.6,
    },
)

plt.title(
    "Diabetes Dataset: BMI vs Target"
)
plt.xlabel("BMI")
plt.ylabel("Disease Progression Target")
plt.tight_layout()
plt.show()
```

# ============================================================

# 25. MULTIPLE FEATURE REGRESSION ANALYSIS

# ============================================================

def multiple_feature_analysis(
df: pd.DataFrame,
) -> None:
"""
Visualize several feature-target relationships.

```
These plots show pairwise relationships. They do not represent
a complete multivariable regression model.
"""
section("25. MULTIPLE FEATURE REGRESSION ANALYSIS")

features = [
    "study_hours",
    "experience",
    "income",
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5),
)

for ax, feature in zip(
    axes,
    features,
):
    sns.regplot(
        data=df,
        x=feature,
        y="exam_score",
        ci=None,
        ax=ax,
        scatter_kws={
            "alpha": 0.5,
        },
    )

    ax.set_title(
        f"{feature.replace('_', ' ').title()} vs Exam Score"
    )

fig.suptitle(
    "Feature-Target Regression Relationships"
)

fig.tight_layout()
plt.show()
```

# ============================================================

# 26. CORRELATION + REGRESSION

# ============================================================

def correlation_with_regression(
df: pd.DataFrame,
) -> None:
"""
Show Pearson correlation alongside a regression visualization.

```
Correlation summarizes linear association.
Regression estimates a conditional relationship under a model.
"""
section("26. CORRELATION + REGRESSION")

x = df["study_hours"]
y = df["exam_score"]

correlation = x.corr(y)

print(
    f"Pearson correlation: {correlation:.4f}"
)

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x="study_hours",
    y="exam_score",
    ci=95,
)

plt.title(
    f"Study Hours vs Exam Score "
    f"(Pearson r = {correlation:.3f})"
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 27. LOG-TRANSFORMED REGRESSION

# ============================================================

def log_transformed_regression(
df: pd.DataFrame,
) -> None:
"""
Demonstrate regression using log-transformed income.

```
log1p(x) = log(1 + x)

This transformation is useful for positive right-skewed
variables, depending on the modeling objective.
"""
section("27. LOG-TRANSFORMED REGRESSION")

plot_data = df.copy()

plot_data["log_income"] = np.log1p(
    plot_data["income"]
)

plt.figure(figsize=(10, 6))

sns.regplot(
    data=plot_data,
    x="experience",
    y="log_income",
    ci=95,
)

plt.title(
    "Experience vs log(1 + Income)"
)
plt.xlabel("Experience")
plt.ylabel("log(1 + Income)")
plt.tight_layout()
plt.show()
```

# ============================================================

# 28. LOGISTIC REGRESSION VISUALIZATION

# ============================================================

def logistic_regression_visualization(
df: pd.DataFrame,
) -> None:
"""
Visualize a binary classification relationship using logistic
regression.

```
This section is included because Seaborn's regression tools can
also visualize logistic relationships.

Target:
    Artificially generated from whether exam_score exceeds
    a threshold.

Note:
    The threshold is only for demonstration.
"""
section("28. LOGISTIC REGRESSION VISUALIZATION")

classification_df = df[
    [
        "study_hours",
        "exam_score",
    ]
].copy()

classification_df["passed"] = (
    classification_df["exam_score"] >= 70
).astype(int)

plt.figure(figsize=(10, 6))

sns.regplot(
    data=classification_df,
    x="study_hours",
    y="passed",
    logistic=True,
    y_jitter=0.03,
    scatter_kws={
        "alpha": 0.4,
    },
)

plt.title(
    "Logistic Regression Visualization"
)
plt.xlabel("Study Hours")
plt.ylabel("Probability / Class")
plt.tight_layout()
plt.show()

X = classification_df[
    ["study_hours"]
]

y = classification_df["passed"]

model = LogisticRegression()
model.fit(X, y)

x_grid = np.linspace(
    X["study_hours"].min(),
    X["study_hours"].max(),
    300,
).reshape(-1, 1)

probability = model.predict_proba(
    x_grid
)[:, 1]

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=classification_df,
    x="study_hours",
    y="passed",
    alpha=0.4,
)

plt.plot(
    x_grid.ravel(),
    probability,
    linewidth=2.5,
    label="Predicted probability",
)

plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1.5,
    label="0.5 probability",
)

plt.title(
    "Logistic Regression Probability Curve"
)
plt.xlabel("Study Hours")
plt.ylabel("Probability of Class 1")
plt.legend()
plt.tight_layout()
plt.show()
```

# ============================================================

# 29. TRAIN / TEST REGRESSION

# ============================================================

def train_test_regression(
df: pd.DataFrame,
) -> None:
"""
Fit a simple regression model using training data and evaluate
it on held-out test data.
"""
section("29. TRAIN / TEST REGRESSION")

```
X = df[["study_hours"]]
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

model = LinearRegression()
model.fit(
    X_train,
    y_train,
)

train_predictions = model.predict(
    X_train
)

test_predictions = model.predict(
    X_test
)

train_r2 = r2_score(
    y_train,
    train_predictions,
)

test_r2 = r2_score(
    y_test,
    test_predictions,
)

print(
    f"Train R²: {train_r2:.4f}"
)
print(
    f"Test R²:  {test_r2:.4f}"
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=y_test,
    y=test_predictions,
    alpha=0.75,
)

minimum = min(
    y_test.min(),
    test_predictions.min(),
)

maximum = max(
    y_test.max(),
    test_predictions.max(),
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2,
)

plt.title(
    "Held-Out Test Set: Actual vs Predicted"
)
plt.xlabel("Actual Test Values")
plt.ylabel("Predicted Test Values")
plt.tight_layout()
plt.show()
```

# ============================================================

# 30. RESIDUALS VS PREDICTED

# ============================================================

def residuals_vs_predicted(
y_true: pd.Series,
predictions: np.ndarray,
) -> None:
"""Plot residuals against predicted values."""
section("30. RESIDUALS VS PREDICTED")

```
residuals = (
    y_true.to_numpy()
    - predictions
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=predictions,
    y=residuals,
    alpha=0.7,
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1.5,
)

plt.title(
    "Residuals vs Predicted Values"
)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()
```

# ============================================================

# 31. REGRESSION DIAGNOSTIC DASHBOARD

# ============================================================

def regression_diagnostic_dashboard(
df: pd.DataFrame,
model: LinearRegression,
) -> None:
"""
Create a compact regression diagnostic dashboard.

```
Panels:
    1. Actual vs predicted
    2. Residuals vs predicted
    3. Residual distribution
    4. Feature vs target
"""
section("31. REGRESSION DIAGNOSTIC DASHBOARD")

X = df[["study_hours"]]
y = df["exam_score"]

predictions = model.predict(X)

residuals = (
    y.to_numpy()
    - predictions
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 11),
)

# --------------------------------------------------------
# Actual vs predicted
# --------------------------------------------------------
sns.scatterplot(
    x=y,
    y=predictions,
    alpha=0.6,
    ax=axes[0, 0],
)

minimum = min(
    y.min(),
    predictions.min(),
)

maximum = max(
    y.max(),
    predictions.max(),
)

axes[0, 0].plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
)

axes[0, 0].set_title(
    "Actual vs Predicted"
)
axes[0, 0].set_xlabel("Actual")
axes[0, 0].set_ylabel("Predicted")

# --------------------------------------------------------
# Residuals vs predicted
# --------------------------------------------------------
sns.scatterplot(
    x=predictions,
    y=residuals,
    alpha=0.6,
    ax=axes[0, 1],
)

axes[0, 1].axhline(
    0,
    linestyle="--",
)

axes[0, 1].set_title(
    "Residuals vs Predicted"
)
axes[0, 1].set_xlabel("Predicted")
axes[0, 1].set_ylabel("Residual")

# --------------------------------------------------------
# Residual distribution
# --------------------------------------------------------
sns.histplot(
    residuals,
    bins=25,
    kde=True,
    ax=axes[1, 0],
)

axes[1, 0].axvline(
    0,
    linestyle="--",
)

axes[1, 0].set_title(
    "Residual Distribution"
)
axes[1, 0].set_xlabel("Residual")

# --------------------------------------------------------
# Feature vs target
# --------------------------------------------------------
sns.regplot(
    data=df,
    x="study_hours",
    y="exam_score",
    ci=None,
    ax=axes[1, 1],
)

axes[1, 1].set_title(
    "Feature vs Target"
)

fig.suptitle(
    "Regression Diagnostic Dashboard",
    fontsize=16,
)

fig.tight_layout()
plt.show()
```

# ============================================================

# 32. SAVE REGRESSION PLOT

# ============================================================

def save_regression_plot(
df: pd.DataFrame,
output_path: str = "regression_plot.png",
) -> None:
"""Save a high-resolution regression visualization."""
section("32. SAVE REGRESSION PLOT")

```
fig, ax = plt.subplots(
    figsize=(10, 6)
)

sns.regplot(
    data=df,
    x="study_hours",
    y="exam_score",
    ci=95,
    ax=ax,
)

ax.set_title(
    "Study Hours vs Exam Score"
)
ax.set_xlabel("Study Hours")
ax.set_ylabel("Exam Score")

fig.tight_layout()

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"Regression plot saved to: {output_path}"
)

plt.show()
```

# ============================================================

# 33. REUSABLE REGRESSION FUNCTION

# ============================================================

def plot_regression(
data: pd.DataFrame,
x: str,
y: str,
*,
order: int = 1,
ci: int | None = 95,
robust: bool = False,
) -> None:
"""
Reusable Seaborn regression plotting function.

```
Parameters:
    data:
        Input DataFrame.

    x:
        Predictor column.

    y:
        Target column.

    order:
        Polynomial order.
        1 = linear regression.
        2 = quadratic regression.

    ci:
        Confidence interval.
        None disables the interval.

    robust:
        Use robust regression.
"""
required = {
    x,
    y,
}

missing = required - set(
    data.columns
)

if missing:
    raise KeyError(
        f"Missing columns: {sorted(missing)}"
    )

if not pd.api.types.is_numeric_dtype(
    data[x]
):
    raise TypeError(
        f"'{x}' must be numerical."
    )

if not pd.api.types.is_numeric_dtype(
    data[y]
):
    raise TypeError(
        f"'{y}' must be numerical."
    )

plt.figure(figsize=(10, 6))

sns.regplot(
    data=data,
    x=x,
    y=y,
    order=order,
    ci=ci,
    robust=robust,
    scatter_kws={
        "alpha": 0.6,
    },
)

plt.title(
    f"{y.replace('_', ' ').title()} "
    f"vs "
    f"{x.replace('_', ' ').title()}"
)

plt.xlabel(
    x.replace("_", " ").title()
)
plt.ylabel(
    y.replace("_", " ").title()
)

plt.tight_layout()
plt.show()
```

# ============================================================

# 34. COMPLETE REGRESSION WORKFLOW

# ============================================================

def complete_regression_workflow(
df: pd.DataFrame,
) -> None:
"""
Print a practical regression visualization workflow.
"""
section("34. COMPLETE REGRESSION WORKFLOW")

```
workflow = [
    "1. Inspect the feature and target data.",
    "2. Plot the raw scatter relationship.",
    "3. Add a regression line when appropriate.",
    "4. Inspect the confidence interval.",
    "5. Check for nonlinear patterns.",
    "6. Investigate possible outliers.",
    "7. Inspect residuals.",
    "8. Compare actual and predicted values.",
    "9. Evaluate on held-out validation/test data.",
    "10. Compare relevant regression metrics.",
    "11. Check whether transformations are justified.",
    "12. Avoid interpreting correlation as causation.",
]

for step in workflow:
    print(step)
```

# ============================================================

# 35. COMMON MISTAKES

# ============================================================

def common_mistakes() -> None:
"""Print common regression-visualization mistakes."""
section("35. COMMON MISTAKES")

```
mistakes = [
    (
        "Treating a regression line as proof of causation",
        "Regression visualizes an association under a fitted model; it does not establish causality.",
    ),
    (
        "Ignoring nonlinear patterns",
        "A straight line can miss important structure in the data.",
    ),
    (
        "Using a high polynomial degree automatically",
        "Higher-order models can overfit and should be validated.",
    ),
    (
        "Ignoring outliers",
        "Extreme observations can substantially affect ordinary least-squares fits.",
    ),
    (
        "Reading the confidence band as a prediction interval",
        "A confidence interval around the estimated mean relationship is not the same as an interval for individual future observations.",
    ),
    (
        "Evaluating only on training data",
        "Training performance can be optimistic; use appropriate validation/test data.",
    ),
    (
        "Comparing metrics with different units directly",
        "MAE and RMSE have target units while R² is unitless.",
    ),
    (
        "Ignoring residual patterns",
        "Systematic residual patterns can indicate model misspecification or other issues.",
    ),
    (
        "Assuming a good-looking plot means a good model",
        "Model quality requires quantitative validation and context.",
    ),
    (
        "Using visualization as the only diagnostic",
        "Regression assumptions and model behavior should be assessed with multiple diagnostics.",
    ),
]

for mistake, explanation in mistakes:
    print(f"\n{mistake}:")
    print(f"  → {explanation}")
```

# ============================================================

# 36. PROFESSIONAL CHECKLIST

# ============================================================

def professional_checklist() -> None:
"""Print a professional regression-analysis checklist."""
section("36. PROFESSIONAL CHECKLIST")

```
checklist = [
    "Inspect raw scatter relationships.",
    "Use regression lines only when they answer a meaningful question.",
    "Choose linear vs nonlinear forms based on evidence and validation.",
    "Inspect confidence intervals.",
    "Check outliers and influential observations.",
    "Analyze residuals.",
    "Compare actual and predicted values.",
    "Separate training and evaluation data.",
    "Report appropriate regression metrics.",
    "Use domain knowledge when interpreting relationships.",
    "Avoid causal claims from observational plots alone.",
    "Document transformations and preprocessing.",
    "Save final visualizations at sufficient resolution.",
]

for index, item in enumerate(
    checklist,
    start=1,
):
    print(
        f"{index:02d}. {item}"
    )
```

# ============================================================

# 37. QUICK REFERENCE

# ============================================================

def quick_reference() -> None:
"""Print a compact Seaborn regression reference."""
section("37. QUICK REFERENCE")

```
reference = {
    "Scatter": "sns.scatterplot(data=df, x='x', y='y')",
    "Linear regression": "sns.regplot(data=df, x='x', y='y')",
    "No CI": "sns.regplot(data=df, x='x', y='y', ci=None)",
    "Polynomial": "sns.regplot(data=df, x='x', y='y', order=2)",
    "Robust": "sns.regplot(data=df, x='x', y='y', robust=True)",
    "Logistic": "sns.regplot(data=df, x='x', y='y', logistic=True)",
    "Residuals": "sns.residplot(data=df, x='x', y='y')",
    "Grouped regression": "sns.lmplot(data=df, x='x', y='y', hue='group')",
    "Faceted regression": "sns.lmplot(data=df, x='x', y='y', col='group')",
    "Save": "fig.savefig('regression.png', dpi=300, bbox_inches='tight')",
}

for name, syntax in reference.items():
    print(
        f"{name:22} → {syntax}"
    )
```

# ============================================================

# 38. MAIN

# ============================================================

def main() -> None:
"""Run the complete Seaborn regression tutorial."""
section(
"SEABORN REGRESSION PLOTS — BEGINNER TO ADVANCED"
)

```
df = create_regression_data()

# --------------------------------------------------------
# Dataset inspection
# --------------------------------------------------------
inspect_dataset(df)

# --------------------------------------------------------
# Basic regression visualization
# --------------------------------------------------------
basic_scatter_plot(df)
basic_regression_plot(df)
regression_without_ci(df)
regression_confidence_interval(df)
customized_regression_plot(df)

# --------------------------------------------------------
# Category-based regression
# --------------------------------------------------------
regression_by_category(df)
faceted_regression(df)
hue_style_relationship(df)

# --------------------------------------------------------
# Machine learning regression
# --------------------------------------------------------
model, predictions, residuals = (
    sklearn_linear_regression(df)
)

plot_sklearn_regression(
    df,
    model,
)

polynomial_regression(df)
seaborn_polynomial_regression(df)
robust_regression(df)
regression_outlier_impact(df)

# --------------------------------------------------------
# Residual analysis
# --------------------------------------------------------
residual_plot(df)
residual_distribution(residuals)
actual_vs_predicted(
    df["exam_score"],
    predictions,
)

metrics = regression_metrics(
    df["exam_score"],
    predictions,
)

metric_visualization(metrics)

# --------------------------------------------------------
# Real dataset
# --------------------------------------------------------
diabetes_regression_plot()

# --------------------------------------------------------
# Feature analysis
# --------------------------------------------------------
multiple_feature_analysis(df)
correlation_with_regression(df)
log_transformed_regression(df)

# --------------------------------------------------------
# Logistic regression visualization
# --------------------------------------------------------
logistic_regression_visualization(df)

# --------------------------------------------------------
# Train/test evaluation
# --------------------------------------------------------
train_test_regression(df)

# --------------------------------------------------------
# Additional diagnostics
# --------------------------------------------------------
residuals_vs_predicted(
    df["exam_score"],
    predictions,
)

regression_diagnostic_dashboard(
    df,
    model,
)

# --------------------------------------------------------
# Reusable regression function
# --------------------------------------------------------
plot_regression(
    df,
    x="study_hours",
    y="exam_score",
    order=1,
    ci=95,
)

# Example polynomial visualization:
#
# plot_regression(
#     df,
#     x="study_hours",
#     y="exam_score",
#     order=2,
#     ci=None,
# )

# --------------------------------------------------------
# Saving
# --------------------------------------------------------
# Uncomment to save:
#
# save_regression_plot(
#     df,
#     "regression_plot.png",
# )

# --------------------------------------------------------
# Final guidance
# --------------------------------------------------------
complete_regression_workflow(df)
common_mistakes()
professional_checklist()
quick_reference()

section("TUTORIAL COMPLETE")

print(
    "Regression visualization tutorial completed successfully."
)
```

# ============================================================

# 39. ENTRY POINT

# ============================================================

if **name** == "**main**":
main()

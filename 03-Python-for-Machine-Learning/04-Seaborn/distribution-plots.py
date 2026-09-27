"""
Seaborn Distribution Plots — Beginner to Advanced
=================================================

File:
03-Python-for-Machine-Learning/04-Seaborn/distribution-plots.py

Purpose:
A practical guide to visualizing numerical distributions with Seaborn.

Topics Covered:
1. Importing libraries and configuring plots
2. Understanding numerical distributions
3. Histogram
4. Histogram with custom bins
5. Histogram with KDE
6. KDE plot
7. KDE bandwidth
8. Cumulative distribution
9. ECDF plot
10. Rug plot
11. Distribution by category
12. Hue-based distributions
13. Multiple distributions
14. Log-scale distributions
15. Distribution of transformed data
16. Outlier-aware distributions
17. Boxplot + distribution analysis
18. Violin + distribution analysis
19. Distribution comparison
20. ML feature distributions
21. Target-variable distributions
22. Class-wise feature distributions
23. Feature skewness
24. Correlation vs distribution analysis
25. Missing-value impact
26. Train/test distribution comparison
27. Data leakage awareness
28. Distribution quality checks
29. Reusable distribution functions
30. Saving publication-quality plots

Notes:
- Most datasets in this tutorial are synthetic or built-in datasets.
- ML metrics/results are not claimed unless explicitly calculated.
- Distribution plots help understand data; they do not prove causation.
- KDE is an estimate of a probability density and depends on bandwidth.
- Always inspect distributions before deciding on transformations.

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

from sklearn.datasets import load_diabetes, load_wine
from sklearn.model_selection import train_test_split

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
"""Print a readable section separator in the terminal."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

# ============================================================

# 3. CREATE SAMPLE DATA

# ============================================================

def create_sample_data(
n_samples: int = 1000,
random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
"""
Create a synthetic dataset containing several distributions.

```
Returns:
    DataFrame with:
        - age
        - income
        - exam_score
        - study_hours
        - department
        - experience
        - target
"""
rng = np.random.default_rng(random_state)

age = rng.normal(
    loc=35,
    scale=10,
    size=n_samples,
).clip(18, 70)

income = rng.lognormal(
    mean=10.7,
    sigma=0.55,
    size=n_samples,
)

exam_score = rng.normal(
    loc=72,
    scale=12,
    size=n_samples,
).clip(0, 100)

study_hours = rng.gamma(
    shape=2.2,
    scale=2.0,
    size=n_samples,
)

experience = np.maximum(
    age - 22 + rng.normal(0, 3, n_samples),
    0,
)

department = rng.choice(
    ["Engineering", "Data Science", "Marketing"],
    size=n_samples,
    p=[0.45, 0.30, 0.25],
)

target_probability = (
    0.25
    \+ 0.05 * (study_hours > 6)
    \+ 0.10 * (exam_score > 80)
)

target = (
    rng.random(n_samples) < target_probability
).astype(int)

return pd.DataFrame(
    {
        "age": age.round(1),
        "income": income.round(2),
        "exam_score": exam_score.round(2),
        "study_hours": study_hours.round(2),
        "experience": experience.round(2),
        "department": department,
        "target": target,
    }
)
```

# ============================================================

# 4. BASIC DATASET

# ============================================================

def basic_dataset_demo(df: pd.DataFrame) -> None:
"""Inspect the sample data before visualization."""
section("4. BASIC DATASET")

```
print(df.head())
print("\nShape:", df.shape)
print("\nData types:")
print(df.dtypes)

print("\nDescriptive statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isna().sum())
```

# ============================================================

# 5. BASIC HISTOGRAM

# ============================================================

def basic_histogram(df: pd.DataFrame) -> None:
"""
Histogram:
A histogram divides numerical values into bins and displays
the number of observations falling into each bin.
"""
section("5. BASIC HISTOGRAM")

```
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="age",
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

# ============================================================

# 6. CUSTOM HISTOGRAM BINS

# ============================================================

def custom_histogram_bins(df: pd.DataFrame) -> None:
"""Demonstrate how the number of histogram bins changes detail."""
section("6. CUSTOM HISTOGRAM BINS")

```
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5),
)

sns.histplot(
    data=df,
    x="exam_score",
    bins=5,
    ax=axes[0],
)
axes[0].set_title("5 Bins")

sns.histplot(
    data=df,
    x="exam_score",
    bins=15,
    ax=axes[1],
)
axes[1].set_title("15 Bins")

sns.histplot(
    data=df,
    x="exam_score",
    bins=30,
    ax=axes[2],
)
axes[2].set_title("30 Bins")

fig.suptitle("Effect of Histogram Bin Count")
fig.tight_layout()
plt.show()
```

# ============================================================

# 7. HISTOGRAM WITH KDE

# ============================================================

def histogram_with_kde(df: pd.DataFrame) -> None:
"""
Combine a histogram with KDE.

```
KDE:
    Kernel Density Estimation provides a smooth estimate of
    the underlying probability density.
"""
section("7. HISTOGRAM WITH KDE")

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="exam_score",
    bins=25,
    kde=True,
)

plt.title("Exam Score Distribution with KDE")
plt.xlabel("Exam Score")
plt.ylabel("Frequency / Density")
plt.tight_layout()
plt.show()
```

# ============================================================

# 8. DENSITY NORMALIZATION

# ============================================================

def density_histogram(df: pd.DataFrame) -> None:
"""
Use stat='density' so the histogram represents density
rather than raw observation counts.
"""
section("8. DENSITY HISTOGRAM")

```
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="exam_score",
    bins=25,
    stat="density",
    kde=True,
)

plt.title("Exam Score Density")
plt.xlabel("Exam Score")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
```

# ============================================================

# 9. BASIC KDE

# ============================================================

def basic_kde(df: pd.DataFrame) -> None:
"""Display a smooth kernel density estimate."""
section("9. BASIC KDE")

```
plt.figure(figsize=(10, 6))

sns.kdeplot(
    data=df,
    x="exam_score",
    fill=True,
)

plt.title("Kernel Density Estimate of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
```

# ============================================================

# 10. KDE BANDWIDTH

# ============================================================

def kde_bandwidth_demo(df: pd.DataFrame) -> None:
"""
Demonstrate KDE bandwidth.

```
Smaller bandwidth:
    More detail and potentially more noise.

Larger bandwidth:
    Smoother distribution but potentially less detail.
"""
section("10. KDE BANDWIDTH")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5),
)

bandwidths = [0.2, 0.5, 1.0]

for ax, bandwidth in zip(axes, bandwidths):
    sns.kdeplot(
        data=df,
        x="exam_score",
        bw_adjust=bandwidth,
        fill=True,
        ax=ax,
    )

    ax.set_title(
        f"KDE bw_adjust={bandwidth}"
    )

fig.suptitle("Effect of KDE Bandwidth")
fig.tight_layout()
plt.show()
```

# ============================================================

# 11. ECDF PLOT

# ============================================================

def ecdf_demo(df: pd.DataFrame) -> None:
"""
ECDF = Empirical Cumulative Distribution Function.

```
Interpretation:
    The y-axis represents the proportion of observations
    less than or equal to a given x value.
"""
section("11. ECDF PLOT")

plt.figure(figsize=(10, 6))

sns.ecdfplot(
    data=df,
    x="exam_score",
)

plt.title("Empirical Cumulative Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Cumulative Proportion")
plt.tight_layout()
plt.show()
```

# ============================================================

# 12. RUG PLOT

# ============================================================

def rug_plot_demo(df: pd.DataFrame) -> None:
"""
Rug plot:
Displays individual observations as small marks along
an axis.
"""
section("12. RUG PLOT")

```
plt.figure(figsize=(10, 5))

sns.kdeplot(
    data=df,
    x="exam_score",
    fill=True,
)

sns.rugplot(
    data=df,
    x="exam_score",
)

plt.title("KDE with Individual Observations")
plt.xlabel("Exam Score")
plt.tight_layout()
plt.show()
```

# ============================================================

# 13. DISTRIBUTION BY CATEGORY

# ============================================================

def distribution_by_category(df: pd.DataFrame) -> None:
"""Compare numerical distributions across categories."""
section("13. DISTRIBUTION BY CATEGORY")

```
plt.figure(figsize=(11, 6))

sns.histplot(
    data=df,
    x="exam_score",
    hue="department",
    kde=True,
    element="step",
    stat="density",
    common_norm=False,
)

plt.title("Exam Score Distribution by Department")
plt.xlabel("Exam Score")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
```

# ============================================================

# 14. KDE BY CATEGORY

# ============================================================

def kde_by_category(df: pd.DataFrame) -> None:
"""Compare category-specific density curves."""
section("14. KDE BY CATEGORY")

```
plt.figure(figsize=(11, 6))

sns.kdeplot(
    data=df,
    x="exam_score",
    hue="department",
    fill=True,
    common_norm=False,
    alpha=0.25,
)

plt.title("Exam Score KDE by Department")
plt.xlabel("Exam Score")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
```

# ============================================================

# 15. MULTIPLE NUMERICAL DISTRIBUTIONS

# ============================================================

def multiple_distributions(df: pd.DataFrame) -> None:
"""Visualize several numerical variables separately."""
section("15. MULTIPLE NUMERICAL DISTRIBUTIONS")

```
numeric_columns = [
    "age",
    "exam_score",
    "study_hours",
    "experience",
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
)

axes = axes.flatten()

for ax, column in zip(axes, numeric_columns):
    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax,
    )

    ax.set_title(
        f"{column.replace('_', ' ').title()} Distribution"
    )

fig.suptitle("Numerical Feature Distributions")
fig.tight_layout()
plt.show()
```

# ============================================================

# 16. LOG-NORMAL DISTRIBUTION

# ============================================================

def skewed_distribution(df: pd.DataFrame) -> None:
"""
Income is intentionally generated using a log-normal
distribution to demonstrate positive/right skew.
"""
section("16. SKEWED DISTRIBUTION")

```
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="income",
    bins=30,
    kde=True,
)

plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

# ============================================================

# 17. LOG-SCALE DISTRIBUTION

# ============================================================

def log_scale_distribution(df: pd.DataFrame) -> None:
"""Visualize a highly skewed variable using a logarithmic x-axis."""
section("17. LOG-SCALE DISTRIBUTION")

```
plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="income",
    bins=30,
    kde=True,
)

plt.xscale("log")

plt.title("Income Distribution on a Logarithmic Scale")
plt.xlabel("Income — Log Scale")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

# ============================================================

# 18. LOG TRANSFORMATION

# ============================================================

def log_transformation_demo(df: pd.DataFrame) -> None:
"""
Demonstrate a common transformation for positive,
right-skewed variables.

```
log1p(x) = log(1 + x)

log1p is useful when values may include zero.
"""
section("18. LOG TRANSFORMATION")

transformed = np.log1p(df["income"])

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 5),
)

sns.histplot(
    df["income"],
    bins=30,
    kde=True,
    ax=axes[0],
)
axes[0].set_title("Original Income")

sns.histplot(
    transformed,
    bins=30,
    kde=True,
    ax=axes[1],
)
axes[1].set_title("log1p(Income)")

fig.suptitle("Distribution Before and After Log Transformation")
fig.tight_layout()
plt.show()
```

# ============================================================

# 19. OUTLIER-AWARE DISTRIBUTION

# ============================================================

def outlier_distribution_demo(df: pd.DataFrame) -> None:
"""Add artificial outliers and visualize their effect."""
section("19. OUTLIER-AWARE DISTRIBUTION")

```
values = df["exam_score"].copy()

outliers = pd.Series(
    [150, 160, 170],
    index=[0, 1, 2],
)

values_with_outliers = pd.concat(
    [
        values.iloc[3:],
        outliers,
    ],
    ignore_index=True,
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 5),
)

sns.histplot(
    values,
    bins=25,
    kde=True,
    ax=axes[0],
)
axes[0].set_title("Original Distribution")

sns.histplot(
    values_with_outliers,
    bins=25,
    kde=True,
    ax=axes[1],
)
axes[1].set_title("Distribution with Extreme Values")

fig.suptitle("Effect of Outliers on a Distribution")
fig.tight_layout()
plt.show()
```

# ============================================================

# 20. HISTOGRAM + BOX PLOT

# ============================================================

def histogram_and_boxplot(df: pd.DataFrame) -> None:
"""Combine distribution shape and outlier information."""
section("20. HISTOGRAM + BOX PLOT")

```
fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    gridspec_kw={"height_ratios": [3, 1]},
)

sns.histplot(
    data=df,
    x="exam_score",
    kde=True,
    ax=axes[0],
)

axes[0].set_title("Exam Score Distribution")

sns.boxplot(
    data=df,
    x="exam_score",
    ax=axes[1],
)

axes[1].set_title("Exam Score Box Plot")

fig.tight_layout()
plt.show()
```

# ============================================================

# 21. VIOLIN DISTRIBUTION

# ============================================================

def violin_distribution(df: pd.DataFrame) -> None:
"""
Violin plots combine:
- distribution shape
- density
- summary statistics
"""
section("21. VIOLIN DISTRIBUTION")

```
plt.figure(figsize=(11, 6))

sns.violinplot(
    data=df,
    x="department",
    y="exam_score",
    inner="quartile",
)

plt.title("Exam Score Distribution by Department")
plt.xlabel("Department")
plt.ylabel("Exam Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```

# ============================================================

# 22. DISTRIBUTION OF TARGET VARIABLE

# ============================================================

def target_distribution(df: pd.DataFrame) -> None:
"""Visualize a binary ML target distribution."""
section("22. TARGET DISTRIBUTION")

```
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="target",
    discrete=True,
)

plt.title("Target Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.xticks([0, 1])
plt.tight_layout()
plt.show()

print("\nTarget distribution:")
print(df["target"].value_counts())

print("\nTarget proportions:")
print(df["target"].value_counts(normalize=True))
```

# ============================================================

# 23. FEATURE DISTRIBUTION BY TARGET

# ============================================================

def feature_by_target_distribution(df: pd.DataFrame) -> None:
"""Compare a feature distribution across target classes."""
section("23. FEATURE DISTRIBUTION BY TARGET")

```
plt.figure(figsize=(11, 6))

sns.histplot(
    data=df,
    x="exam_score",
    hue="target",
    kde=True,
    stat="density",
    common_norm=False,
    element="step",
)

plt.title("Exam Score Distribution by Target Class")
plt.xlabel("Exam Score")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
```

# ============================================================

# 24. KDE FEATURE VS TARGET

# ============================================================

def kde_feature_by_target(df: pd.DataFrame) -> None:
"""Use KDE to compare feature distributions between classes."""
section("24. KDE FEATURE VS TARGET")

```
plt.figure(figsize=(11, 6))

sns.kdeplot(
    data=df,
    x="study_hours",
    hue="target",
    fill=True,
    common_norm=False,
    alpha=0.25,
)

plt.title("Study Hours Distribution by Target Class")
plt.xlabel("Study Hours")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
```

# ============================================================

# 25. WINE DATASET DISTRIBUTIONS

# ============================================================

def wine_distribution_demo() -> None:
"""Explore feature distributions from scikit-learn's wine dataset."""
section("25. WINE DATASET DISTRIBUTIONS")

```
wine = load_wine(as_frame=True)

X = wine.data.copy()

selected_features = [
    "alcohol",
    "malic_acid",
    "color_intensity",
    "proline",
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10),
)

axes = axes.flatten()

for ax, feature in zip(axes, selected_features):
    sns.histplot(
        data=X,
        x=feature,
        kde=True,
        ax=ax,
    )

    ax.set_title(
        f"{feature.replace('_', ' ').title()} Distribution"
    )

fig.suptitle("Wine Dataset Feature Distributions")
fig.tight_layout()
plt.show()
```

# ============================================================

# 26. DIABETES DATASET

# ============================================================

def diabetes_distribution_demo() -> None:
"""Explore distributions from the diabetes regression dataset."""
section("26. DIABETES DATASET DISTRIBUTIONS")

```
diabetes = load_diabetes(as_frame=True)

X = diabetes.data

selected_features = [
    "bmi",
    "bp",
    "s5",
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(17, 5),
)

for ax, feature in zip(axes, selected_features):
    sns.histplot(
        data=X,
        x=feature,
        kde=True,
        ax=ax,
    )

    ax.set_title(
        f"{feature.upper()} Distribution"
    )

fig.suptitle("Diabetes Dataset Feature Distributions")
fig.tight_layout()
plt.show()
```

# ============================================================

# 27. DISTRIBUTION SKEWNESS

# ============================================================

def skewness_analysis(df: pd.DataFrame) -> None:
"""
Calculate skewness.

```
Interpretation:
    skewness ≈ 0:
        approximately symmetric

    positive skew:
        longer right tail

    negative skew:
        longer left tail

Note:
    The threshold used here is only a practical heuristic,
    not a universal statistical rule.
"""
section("27. SKEWNESS ANALYSIS")

numeric_columns = df.select_dtypes(
    include=np.number
).columns

skewness = df[numeric_columns].skew()

result = (
    skewness
    .sort_values(
        key=np.abs,
        ascending=False,
    )
    .to_frame("skewness")
)

print(result)
```

# ============================================================

# 28. DISTRIBUTION SUMMARY

# ============================================================

def distribution_summary(df: pd.DataFrame) -> None:
"""Create a compact numerical summary of distributions."""
section("28. DISTRIBUTION SUMMARY")

```
numeric_columns = df.select_dtypes(
    include=np.number
).columns

summary = pd.DataFrame(
    {
        "mean": df[numeric_columns].mean(),
        "median": df[numeric_columns].median(),
        "std": df[numeric_columns].std(),
        "min": df[numeric_columns].min(),
        "max": df[numeric_columns].max(),
        "skewness": df[numeric_columns].skew(),
        "missing": df[numeric_columns].isna().sum(),
    }
)

print(summary)
```

# ============================================================

# 29. MISSING VALUES AND DISTRIBUTIONS

# ============================================================

def missing_value_distribution_demo(
df: pd.DataFrame,
) -> None:
"""
Demonstrate why missing values should be inspected before
distribution analysis.
"""
section("29. MISSING VALUES AND DISTRIBUTIONS")

```
data = df["exam_score"].copy()

rng = np.random.default_rng(RANDOM_STATE)

missing_indices = rng.choice(
    len(data),
    size=50,
    replace=False,
)

data.iloc[missing_indices] = np.nan

print(
    "Missing values introduced:",
    data.isna().sum(),
)

plt.figure(figsize=(10, 6))

sns.histplot(
    data=data,
    bins=25,
    kde=True,
)

plt.title(
    "Distribution with Missing Values Ignored by Visualization"
)
plt.xlabel("Exam Score")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

# ============================================================

# 30. TRAIN / TEST DISTRIBUTION COMPARISON

# ============================================================

def train_test_distribution_demo(df: pd.DataFrame) -> None:
"""
Compare a feature's distribution in training and test sets.

```
Purpose:
    Identify obvious distribution differences before modeling.

Important:
    This is diagnostic analysis. It does not prove that a model
    will fail or succeed.
"""
section("30. TRAIN / TEST DISTRIBUTION COMPARISON")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["target"],
)

train_df = train_df.copy()
test_df = test_df.copy()

train_df["dataset"] = "Train"
test_df["dataset"] = "Test"

combined = pd.concat(
    [
        train_df,
        test_df,
    ],
    ignore_index=True,
)

plt.figure(figsize=(11, 6))

sns.kdeplot(
    data=combined,
    x="exam_score",
    hue="dataset",
    fill=True,
    common_norm=False,
    alpha=0.25,
)

plt.title("Train vs Test Feature Distribution")
plt.xlabel("Exam Score")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
```

# ============================================================

# 31. DISTRIBUTION BY TRAIN/TEST USING HISTOGRAM

# ============================================================

def train_test_histogram_demo(df: pd.DataFrame) -> None:
"""Compare train/test distributions with step histograms."""
section("31. TRAIN / TEST HISTOGRAM")

```
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["target"],
)

plt.figure(figsize=(11, 6))

sns.histplot(
    train_df["study_hours"],
    bins=25,
    stat="density",
    element="step",
    fill=False,
    label="Train",
)

sns.histplot(
    test_df["study_hours"],
    bins=25,
    stat="density",
    element="step",
    fill=False,
    label="Test",
)

plt.title("Study Hours: Train vs Test")
plt.xlabel("Study Hours")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
```

# ============================================================

# 32. DISTRIBUTION QUALITY CHECKS

# ============================================================

def distribution_quality_checks(
df: pd.DataFrame,
) -> None:
"""
Perform basic data-quality checks before interpreting plots.
"""
section("32. DISTRIBUTION QUALITY CHECKS")

```
numeric = df.select_dtypes(
    include=np.number
)

quality_report = pd.DataFrame(
    {
        "dtype": numeric.dtypes.astype(str),
        "missing": numeric.isna().sum(),
        "unique": numeric.nunique(),
        "mean": numeric.mean(),
        "median": numeric.median(),
        "std": numeric.std(),
        "skewness": numeric.skew(),
    }
)

print(quality_report)

print("\nPotential constant columns:")
constant_columns = [
    column
    for column in numeric.columns
    if numeric[column].nunique(dropna=True) <= 1
]

print(
    constant_columns
    if constant_columns
    else "None"
)
```

# ============================================================

# 33. REUSABLE DISTRIBUTION FUNCTION

# ============================================================

def plot_distribution(
data: pd.DataFrame,
column: str,
*,
bins: int = 25,
kde: bool = True,
title: str | None = None,
) -> None:
"""
Reusable histogram + KDE function.

```
Parameters:
    data:
        Input DataFrame.

    column:
        Numerical column to visualize.

    bins:
        Number of histogram bins.

    kde:
        Whether to display KDE.

    title:
        Optional custom title.
"""
if column not in data.columns:
    raise KeyError(
        f"Column '{column}' does not exist."
    )

if not pd.api.types.is_numeric_dtype(data[column]):
    raise TypeError(
        f"Column '{column}' must be numerical."
    )

plt.figure(figsize=(10, 6))

sns.histplot(
    data=data,
    x=column,
    bins=bins,
    kde=kde,
)

plt.title(
    title
    or f"Distribution of {column.replace('_', ' ').title()}"
)

plt.xlabel(
    column.replace("_", " ").title()
)
plt.ylabel("Count")
plt.tight_layout()
plt.show()
```

# ============================================================

# 34. REUSABLE CATEGORY DISTRIBUTION FUNCTION

# ============================================================

def plot_distribution_by_category(
data: pd.DataFrame,
numerical_column: str,
category_column: str,
) -> None:
"""Plot a numerical distribution across categories."""
required_columns = {
numerical_column,
category_column,
}

```
missing_columns = (
    required_columns - set(data.columns)
)

if missing_columns:
    raise KeyError(
        f"Missing columns: {sorted(missing_columns)}"
    )

if not pd.api.types.is_numeric_dtype(
    data[numerical_column]
):
    raise TypeError(
        f"'{numerical_column}' must be numerical."
    )

plt.figure(figsize=(11, 6))

sns.kdeplot(
    data=data,
    x=numerical_column,
    hue=category_column,
    fill=True,
    common_norm=False,
    alpha=0.25,
)

plt.title(
    f"{numerical_column.replace('_', ' ').title()} "
    f"Distribution by "
    f"{category_column.replace('_', ' ').title()}"
)

plt.xlabel(
    numerical_column.replace("_", " ").title()
)
plt.ylabel("Density")
plt.tight_layout()
plt.show()
```

# ============================================================

# 35. REUSABLE EDA DISTRIBUTION DASHBOARD

# ============================================================

def distribution_dashboard(
df: pd.DataFrame,
columns: list[str],
) -> None:
"""
Create a reusable distribution dashboard.

```
Each numerical feature receives:
    - histogram
    - KDE curve
"""
valid_columns = [
    column
    for column in columns
    if column in df.columns
    and pd.api.types.is_numeric_dtype(df[column])
]

if not valid_columns:
    raise ValueError(
        "No valid numerical columns were provided."
    )

n_columns = 2
n_rows = int(
    np.ceil(len(valid_columns) / n_columns)
)

fig, axes = plt.subplots(
    n_rows,
    n_columns,
    figsize=(15, 5 * n_rows),
)

axes = np.atleast_1d(axes).flatten()

for ax, column in zip(
    axes,
    valid_columns,
):
    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax,
    )

    ax.set_title(
        f"{column.replace('_', ' ').title()}"
    )

for ax in axes[len(valid_columns):]:
    ax.remove()

fig.suptitle(
    "Machine Learning Feature Distribution Dashboard",
    fontsize=16,
)

fig.tight_layout()
plt.show()
```

# ============================================================

# 36. SAVE DISTRIBUTION PLOT

# ============================================================

def save_distribution_plot(
df: pd.DataFrame,
output_path: str = "exam_score_distribution.png",
) -> None:
"""
Save a high-resolution distribution plot.

```
DPI:
    300 is commonly suitable for high-quality reports.
"""
section("36. SAVE DISTRIBUTION PLOT")

fig, ax = plt.subplots(
    figsize=(10, 6)
)

sns.histplot(
    data=df,
    x="exam_score",
    bins=25,
    kde=True,
    ax=ax,
)

ax.set_title("Exam Score Distribution")
ax.set_xlabel("Exam Score")
ax.set_ylabel("Count")

fig.tight_layout()

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

print(
    f"Distribution plot saved to: {output_path}"
)

plt.show()
```

# ============================================================

# 37. COMPLETE DISTRIBUTION ANALYSIS WORKFLOW

# ============================================================

def complete_distribution_workflow(
df: pd.DataFrame,
) -> None:
"""
Demonstrate a practical ML distribution-analysis workflow.

```
Workflow:
    1. Identify numerical features
    2. Check missing values
    3. Inspect summary statistics
    4. Inspect skewness
    5. Plot distributions
    6. Inspect target distribution
    7. Compare feature distributions by target
    8. Compare train/test distributions
    9. Decide whether transformations need investigation

Important:
    A transformation should be selected based on the modeling
    objective, feature meaning, validation strategy, and model
    assumptions—not solely because a histogram looks skewed.
"""
section("37. COMPLETE DISTRIBUTION ANALYSIS WORKFLOW")

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("Numerical columns:")
print(numeric_columns)

print("\nMissing values:")
print(df[numeric_columns].isna().sum())

print("\nSkewness:")
print(df[numeric_columns].skew())

print("\nSuggested workflow:")
workflow = [
    "1. Inspect the raw numerical distributions.",
    "2. Check missing values and invalid observations.",
    "3. Check extreme values and possible outliers.",
    "4. Compare train and test distributions.",
    "5. Analyze feature distributions by target.",
    "6. Identify strongly skewed positive variables.",
    "7. Consider transformations where appropriate.",
    "8. Validate preprocessing using the training data.",
    "9. Re-check distributions after preprocessing.",
    "10. Evaluate model performance using validation data.",
]

for step in workflow:
    print(step)
```

# ============================================================

# 38. COMMON MISTAKES

# ============================================================

def common_mistakes() -> None:
"""Print common mistakes when interpreting distributions."""
section("38. COMMON MISTAKES")

```
mistakes = [
    (
        "Using too few bins",
        "Important distribution structure can disappear.",
    ),
    (
        "Using too many bins",
        "The histogram can become noisy and difficult to interpret.",
    ),
    (
        "Treating KDE as exact",
        "KDE is an estimate and depends on bandwidth.",
    ),
    (
        "Ignoring outliers",
        "Extreme values can strongly affect the visual shape.",
    ),
    (
        "Ignoring missing values",
        "The plotted observations may not represent the full dataset.",
    ),
    (
        "Assuming normality from appearance",
        "A visual inspection alone does not establish a statistical distribution.",
    ),
    (
        "Transforming every skewed feature",
        "Transformation decisions should consider the model and feature meaning.",
    ),
    (
        "Comparing train/test after leakage",
        "Preprocessing should be learned from training data when appropriate.",
    ),
    (
        "Confusing density with probability",
        "Density height is not itself the probability of an observation.",
    ),
    (
        "Assuming correlation from similar distributions",
        "Distribution shape and variable correlation answer different questions.",
    ),
]

for mistake, explanation in mistakes:
    print(f"\n{mistake}:")
    print(f"  → {explanation}")
```

# ============================================================

# 39. PROFESSIONAL DISTRIBUTION CHECKLIST

# ============================================================

def professional_checklist() -> None:
"""Print a practical distribution-analysis checklist."""
section("39. PROFESSIONAL CHECKLIST")

```
checklist = [
    "Inspect all important numerical features.",
    "Check missing values before interpretation.",
    "Check data types and invalid numerical values.",
    "Review histogram shape.",
    "Compare histogram and KDE when useful.",
    "Inspect skewness.",
    "Investigate potential outliers.",
    "Compare distributions across meaningful categories.",
    "Inspect the target distribution.",
    "Compare important features across target classes.",
    "Compare train/test distributions.",
    "Consider transformations for strongly skewed features.",
    "Avoid making distribution-based causal claims.",
    "Validate preprocessing using appropriate data splits.",
    "Save important plots with sufficient resolution.",
]

for index, item in enumerate(
    checklist,
    start=1,
):
    print(f"{index:02d}. {item}")
```

# ============================================================

# 40. QUICK REFERENCE

# ============================================================

def quick_reference() -> None:
"""Print a compact Seaborn distribution reference."""
section("40. QUICK REFERENCE")

```
reference = {
    "Histogram": "sns.histplot(data=df, x='feature')",
    "Histogram + KDE": "sns.histplot(data=df, x='feature', kde=True)",
    "Density histogram": "sns.histplot(data=df, x='feature', stat='density')",
    "KDE": "sns.kdeplot(data=df, x='feature', fill=True)",
    "ECDF": "sns.ecdfplot(data=df, x='feature')",
    "Rug": "sns.rugplot(data=df, x='feature')",
    "Hue": "sns.histplot(data=df, x='feature', hue='category')",
    "Category KDE": "sns.kdeplot(data=df, x='feature', hue='category')",
    "Bins": "sns.histplot(data=df, x='feature', bins=30)",
    "Log axis": "plt.xscale('log')",
    "Save": "fig.savefig('plot.png', dpi=300, bbox_inches='tight')",
}

for name, syntax in reference.items():
    print(f"{name:20} → {syntax}")
```

# ============================================================

# 41. MAIN

# ============================================================

def main() -> None:
"""Run the complete distribution-plot tutorial."""
section("SEABORN DISTRIBUTION PLOTS — BEGINNER TO ADVANCED")

```
df = create_sample_data()

basic_dataset_demo(df)

# --------------------------------------------------------
# Core distribution plots
# --------------------------------------------------------
basic_histogram(df)
custom_histogram_bins(df)
histogram_with_kde(df)
density_histogram(df)
basic_kde(df)
kde_bandwidth_demo(df)
ecdf_demo(df)
rug_plot_demo(df)

# --------------------------------------------------------
# Category-based distributions
# --------------------------------------------------------
distribution_by_category(df)
kde_by_category(df)

# --------------------------------------------------------
# Feature distributions
# --------------------------------------------------------
multiple_distributions(df)
skewed_distribution(df)
log_scale_distribution(df)
log_transformation_demo(df)

# --------------------------------------------------------
# Outliers and summary visualizations
# --------------------------------------------------------
outlier_distribution_demo(df)
histogram_and_boxplot(df)
violin_distribution(df)

# --------------------------------------------------------
# Machine Learning distributions
# --------------------------------------------------------
target_distribution(df)
feature_by_target_distribution(df)
kde_feature_by_target(df)

# --------------------------------------------------------
# Real datasets
# --------------------------------------------------------
wine_distribution_demo()
diabetes_distribution_demo()

# --------------------------------------------------------
# Numerical analysis
# --------------------------------------------------------
skewness_analysis(df)
distribution_summary(df)
missing_value_distribution_demo(df)

# --------------------------------------------------------
# Train/test analysis
# --------------------------------------------------------
train_test_distribution_demo(df)
train_test_histogram_demo(df)

# --------------------------------------------------------
# Quality checks
# --------------------------------------------------------
distribution_quality_checks(df)

# --------------------------------------------------------
# Reusable functions
# --------------------------------------------------------
plot_distribution(
    df,
    "exam_score",
    bins=25,
    kde=True,
)

plot_distribution_by_category(
    df,
    numerical_column="exam_score",
    category_column="department",
)

distribution_dashboard(
    df,
    columns=[
        "age",
        "income",
        "exam_score",
        "study_hours",
    ],
)

# --------------------------------------------------------
# Saving
# --------------------------------------------------------
# Uncomment to save:
#
# save_distribution_plot(
#     df,
#     "exam_score_distribution.png",
# )

# --------------------------------------------------------
# Final workflow
# --------------------------------------------------------
complete_distribution_workflow(df)
common_mistakes()
professional_checklist()
quick_reference()

section("TUTORIAL COMPLETE")

print(
    "Distribution analysis completed successfully."
)
```

# ============================================================

# 42. ENTRY POINT

# ============================================================

if **name** == "**main**":
main()

"""
HISTOGRAMS WITH MATPLOTLIB

File:
03-Python-for-Machine-Learning/03-Matplotlib/histogram.py

Description:
A complete beginner-to-advanced tutorial on creating and customizing
histograms with Matplotlib.

Histograms are especially useful in Machine Learning for understanding:

    - Feature distributions
    - Data spread
    - Central tendency
    - Skewness
    - Outliers
    - Class distributions
    - Model prediction distributions
    - Residual distributions
    - Feature transformations
    - Probability density
    - Data preprocessing

Requirements:
pip install numpy pandas matplotlib

Run:
python histogram.py

"""

=============================================================================
1. IMPORT LIBRARIES
=============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

=============================================================================
2. MATPLOTLIB CONFIGURATION
=============================================================================

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

def section(title: str) -> None:
"""Print a formatted section heading in the terminal."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

=============================================================================
3. BASIC HISTOGRAM
=============================================================================

def basic_histogram() -> None:
"""
Create a basic histogram.

A histogram divides numerical data into intervals called bins and counts
how many observations fall into each bin.
"""

section("3. BASIC HISTOGRAM")

data = np.array([
    12, 15, 18, 20, 22,
    22, 24, 25, 26, 28,
    30, 30, 31, 32, 34,
    35, 36, 38, 40, 42
])

fig, ax = plt.subplots()

ax.hist(data)

ax.set_title("Basic Histogram")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
4. HISTOGRAM WITH CUSTOM BINS
=============================================================================

def custom_bins() -> None:
"""
Control the number of histogram bins.

More bins:
    More detail, but potentially more noise.

Fewer bins:
    Simpler distribution, but potentially less detail.
"""

section("4. CUSTOM BINS")

np.random.seed(42)

data = np.random.normal(
    loc=50,
    scale=10,
    size=1000
)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(data, bins=5)
axes[0].set_title("5 Bins")

axes[1].hist(data, bins=15)
axes[1].set_title("15 Bins")

axes[2].hist(data, bins=40)
axes[2].set_title("40 Bins")

for ax in axes:
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

fig.suptitle("Effect of Bin Count")

plt.tight_layout()
plt.show()
=============================================================================
5. HISTOGRAM WITH EXPLICIT BIN EDGES
=============================================================================

def explicit_bin_edges() -> None:
"""Define exact intervals using custom bin edges."""

section("5. EXPLICIT BIN EDGES")

data = np.array([
    5, 8, 12, 14, 17,
    21, 24, 25, 29,
    31, 34, 38, 42
])

bins = [0, 10, 20, 30, 40, 50]

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=bins,
    edgecolor="black"
)

ax.set_title("Histogram with Custom Bin Edges")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
6. HISTOGRAM WITH EDGE STYLING
=============================================================================

def styled_histogram() -> None:
"""Customize histogram transparency and bar edges."""

section("6. STYLED HISTOGRAM")

np.random.seed(42)

data = np.random.normal(100, 15, 1000)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=25,
    alpha=0.75,
    edgecolor="black",
    linewidth=1
)

ax.set_title("Styled Histogram")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
7. HISTOGRAM ORIENTATION
=============================================================================

def horizontal_histogram() -> None:
"""Create a horizontal histogram."""

section("7. HORIZONTAL HISTOGRAM")

np.random.seed(42)

data = np.random.normal(0, 1, 1000)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=30,
    orientation="horizontal"
)

ax.set_title("Horizontal Histogram")
ax.set_xlabel("Frequency")
ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
=============================================================================
8. DENSITY HISTOGRAM
=============================================================================

def density_histogram() -> None:
"""
Create a probability-density histogram.

density=True normalizes the histogram so that the total area is
approximately 1.
"""

section("8. DENSITY HISTOGRAM")

np.random.seed(42)

data = np.random.normal(50, 10, 2000)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=30,
    density=True,
    alpha=0.7,
    edgecolor="black"
)

ax.set_title("Probability Density Histogram")
ax.set_xlabel("Value")
ax.set_ylabel("Density")

plt.tight_layout()
plt.show()
=============================================================================
9. FREQUENCY VS DENSITY
=============================================================================

def frequency_vs_density() -> None:
"""Compare frequency and density representations."""

section("9. FREQUENCY VS DENSITY")

np.random.seed(42)

data = np.random.normal(0, 1, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(
    data,
    bins=30,
    edgecolor="black"
)

axes[0].set_title("Frequency")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Count")

axes[1].hist(
    data,
    bins=30,
    density=True,
    edgecolor="black"
)

axes[1].set_title("Density")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Density")

plt.tight_layout()
plt.show()
=============================================================================
10. CUMULATIVE HISTOGRAM
=============================================================================

def cumulative_histogram() -> None:
"""
Create a cumulative histogram.

cumulative=True shows accumulated counts from left to right.
"""

section("10. CUMULATIVE HISTOGRAM")

np.random.seed(42)

data = np.random.normal(50, 10, 1000)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=30,
    cumulative=True,
    edgecolor="black"
)

ax.set_title("Cumulative Histogram")
ax.set_xlabel("Value")
ax.set_ylabel("Cumulative Count")

plt.tight_layout()
plt.show()
=============================================================================
11. CUMULATIVE DENSITY
=============================================================================

def cumulative_density() -> None:
"""Display cumulative probability distribution."""

section("11. CUMULATIVE DENSITY")

np.random.seed(42)

data = np.random.normal(0, 1, 2000)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=40,
    density=True,
    cumulative=True,
    alpha=0.75,
    edgecolor="black"
)

ax.set_title("Cumulative Probability Distribution")
ax.set_xlabel("Value")
ax.set_ylabel("Cumulative Probability")

plt.tight_layout()
plt.show()
=============================================================================
12. TWO DISTRIBUTIONS
=============================================================================

def compare_distributions() -> None:
"""Compare two numerical distributions using overlapping histograms."""

section("12. COMPARING TWO DISTRIBUTIONS")

np.random.seed(42)

group_a = np.random.normal(50, 8, 1000)
group_b = np.random.normal(60, 10, 1000)

bins = np.linspace(
    min(group_a.min(), group_b.min()),
    max(group_a.max(), group_b.max()),
    30
)

fig, ax = plt.subplots()

ax.hist(
    group_a,
    bins=bins,
    alpha=0.5,
    label="Group A",
    edgecolor="black"
)

ax.hist(
    group_b,
    bins=bins,
    alpha=0.5,
    label="Group B",
    edgecolor="black"
)

ax.set_title("Distribution Comparison")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
13. MULTIPLE DISTRIBUTIONS
=============================================================================

def multiple_distributions() -> None:
"""Compare multiple distributions with a common bin structure."""

section("13. MULTIPLE DISTRIBUTIONS")

np.random.seed(42)

data_a = np.random.normal(50, 8, 800)
data_b = np.random.normal(60, 10, 800)
data_c = np.random.normal(70, 7, 800)

combined = np.concatenate([
    data_a,
    data_b,
    data_c
])

bins = np.linspace(
    combined.min(),
    combined.max(),
    35
)

fig, ax = plt.subplots()

ax.hist(
    data_a,
    bins=bins,
    alpha=0.4,
    label="Distribution A"
)

ax.hist(
    data_b,
    bins=bins,
    alpha=0.4,
    label="Distribution B"
)

ax.hist(
    data_c,
    bins=bins,
    alpha=0.4,
    label="Distribution C"
)

ax.set_title("Multiple Distribution Comparison")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
14. OVERLAY WITH DENSITY
=============================================================================

def density_comparison() -> None:
"""Compare distributions using normalized density."""

section("14. DENSITY COMPARISON")

np.random.seed(42)

group_a = np.random.normal(0, 1, 2000)
group_b = np.random.normal(1, 1.2, 2000)

bins = np.linspace(-5, 6, 45)

fig, ax = plt.subplots()

ax.hist(
    group_a,
    bins=bins,
    density=True,
    alpha=0.45,
    label="Group A"
)

ax.hist(
    group_b,
    bins=bins,
    density=True,
    alpha=0.45,
    label="Group B"
)

ax.set_title("Density Distribution Comparison")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
15. LOG-NORMAL DISTRIBUTION
=============================================================================

def skewed_distribution() -> None:
"""
Visualize a right-skewed distribution.

Skewed features are common in real-world ML datasets, especially:
    - Income
    - Transaction amounts
    - Population
    - House prices
"""

section("15. SKEWED DISTRIBUTION")

np.random.seed(42)

data = np.random.lognormal(
    mean=2,
    sigma=0.7,
    size=2000
)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=40,
    edgecolor="black"
)

ax.set_title("Right-Skewed Distribution")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
16. OUTLIER DETECTION
=============================================================================

def visualize_outliers() -> None:
"""
Demonstrate how extreme observations appear in a histogram.

A histogram is useful for spotting suspicious tails, although it should
not be the only method used for outlier detection.
"""

section("16. OUTLIER VISUALIZATION")

np.random.seed(42)

normal_data = np.random.normal(
    50,
    8,
    1000
)

outliers = np.array([
    100,
    110,
    120,
    130
])

data = np.concatenate([
    normal_data,
    outliers
])

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=40,
    edgecolor="black"
)

ax.set_title("Distribution with Extreme Values")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
17. HISTOGRAM WITH MEAN AND MEDIAN
=============================================================================

def mean_median_histogram() -> None:
"""Display mean and median on a distribution."""

section("17. MEAN AND MEDIAN")

np.random.seed(42)

data = np.random.normal(
    100,
    15,
    1000
)

mean_value = np.mean(data)
median_value = np.median(data)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=30,
    edgecolor="black",
    alpha=0.75
)

ax.axvline(
    mean_value,
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean_value:.2f}"
)

ax.axvline(
    median_value,
    linestyle="-.",
    linewidth=2,
    label=f"Median = {median_value:.2f}"
)

ax.set_title("Mean and Median")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
18. STANDARD DEVIATION VISUALIZATION
=============================================================================

def standard_deviation_histogram() -> None:
"""Visualize one and two standard deviations from the mean."""

section("18. STANDARD DEVIATION")

np.random.seed(42)

data = np.random.normal(
    100,
    15,
    2000
)

mean = np.mean(data)
std = np.std(data)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=35,
    density=True,
    alpha=0.75,
    edgecolor="black"
)

ax.axvline(
    mean,
    linestyle="--",
    linewidth=2,
    label=f"Mean = {mean:.1f}"
)

ax.axvline(
    mean - std,
    linestyle=":",
    linewidth=2,
    label="Mean - 1 Std"
)

ax.axvline(
    mean + std,
    linestyle=":",
    linewidth=2,
    label="Mean + 1 Std"
)

ax.axvline(
    mean - 2 * std,
    linestyle="-.",
    linewidth=1.5,
    label="Mean - 2 Std"
)

ax.axvline(
    mean + 2 * std,
    linestyle="-.",
    linewidth=1.5,
    label="Mean + 2 Std"
)

ax.set_title("Mean and Standard Deviation")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
19. NORMAL DISTRIBUTION
=============================================================================

def normal_distribution() -> None:
"""
Visualize a normal distribution.

The normal distribution is characterized by:
    - Mean
    - Standard deviation
    - Symmetric bell-shaped curve
"""

section("19. NORMAL DISTRIBUTION")

np.random.seed(42)

data = np.random.normal(
    loc=0,
    scale=1,
    size=5000
)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=50,
    density=True,
    alpha=0.75,
    edgecolor="black"
)

ax.set_title("Approximately Normal Distribution")
ax.set_xlabel("Value")
ax.set_ylabel("Density")

plt.tight_layout()
plt.show()
=============================================================================
20. EXPONENTIAL DISTRIBUTION
=============================================================================

def exponential_distribution() -> None:
"""Visualize an exponentially distributed dataset."""

section("20. EXPONENTIAL DISTRIBUTION")

np.random.seed(42)

data = np.random.exponential(
    scale=2,
    size=3000
)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=40,
    density=True,
    alpha=0.75,
    edgecolor="black"
)

ax.set_title("Exponential Distribution")
ax.set_xlabel("Value")
ax.set_ylabel("Density")

plt.tight_layout()
plt.show()
=============================================================================
21. HISTOGRAM WITH PERCENTAGE
=============================================================================

def percentage_histogram() -> None:
"""Show histogram bins as percentages rather than raw counts."""

section("21. PERCENTAGE HISTOGRAM")

np.random.seed(42)

data = np.random.normal(
    70,
    10,
    1000
)

weights = np.ones_like(data) / len(data) * 100

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=25,
    weights=weights,
    edgecolor="black"
)

ax.set_title("Histogram as Percentage")
ax.set_xlabel("Value")
ax.set_ylabel("Percentage (%)")

plt.tight_layout()
plt.show()
=============================================================================
22. ML FEATURE DISTRIBUTION
=============================================================================

def ml_feature_distribution() -> None:
"""
Visualize the distribution of a Machine Learning feature.

Understanding feature distributions can help decide whether a feature
requires transformation or scaling.
"""

section("22. MACHINE LEARNING FEATURE DISTRIBUTION")

np.random.seed(42)

age = np.random.normal(
    35,
    10,
    1000
)

age = np.clip(
    age,
    18,
    80
)

fig, ax = plt.subplots()

ax.hist(
    age,
    bins=20,
    edgecolor="black"
)

ax.set_title("Age Feature Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Number of Samples")

plt.tight_layout()
plt.show()
=============================================================================
23. CLASS-CONDITIONED FEATURE DISTRIBUTION
=============================================================================

def class_conditioned_distribution() -> None:
"""
Compare a feature distribution across two ML classes.

This can help visually investigate whether a feature separates classes.
"""

section("23. CLASS-CONDITIONED DISTRIBUTION")

np.random.seed(42)

class_0 = np.random.normal(
    40,
    7,
    500
)

class_1 = np.random.normal(
    55,
    8,
    500
)

bins = np.linspace(
    15,
    85,
    30
)

fig, ax = plt.subplots()

ax.hist(
    class_0,
    bins=bins,
    alpha=0.5,
    label="Class 0",
    edgecolor="black"
)

ax.hist(
    class_1,
    bins=bins,
    alpha=0.5,
    label="Class 1",
    edgecolor="black"
)

ax.set_title("Feature Distribution by Class")
ax.set_xlabel("Feature Value")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
24. CLASS IMBALANCE
=============================================================================

def class_imbalance_histogram() -> None:
"""
Visualize class labels using a histogram.

For discrete class labels, a bar chart is generally more appropriate.
This example demonstrates the histogram representation for learning
purposes.
"""

section("24. CLASS DISTRIBUTION")

np.random.seed(42)

labels = np.concatenate([
    np.zeros(900),
    np.ones(100)
])

fig, ax = plt.subplots()

ax.hist(
    labels,
    bins=[-0.5, 0.5, 1.5],
    edgecolor="black",
    rwidth=0.8
)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Class 0", "Class 1"])

ax.set_title("Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Number of Samples")

plt.tight_layout()
plt.show()
=============================================================================
25. MODEL PREDICTION DISTRIBUTION
=============================================================================

def prediction_distribution() -> None:
"""
Visualize predicted probabilities from a classification model.

Values closer to:
    0 -> lower probability of positive class
    1 -> higher probability of positive class
"""

section("25. MODEL PREDICTION DISTRIBUTION")

np.random.seed(42)

probabilities = np.random.beta(
    a=2,
    b=2,
    size=1000
)

fig, ax = plt.subplots()

ax.hist(
    probabilities,
    bins=20,
    edgecolor="black"
)

ax.set_title("Predicted Probability Distribution")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
26. RESIDUAL DISTRIBUTION
=============================================================================

def residual_distribution() -> None:
"""
Visualize regression residuals.

Residual:

    residual = actual - predicted

A residual histogram centered near zero can be useful during regression
model diagnostics, although it should be interpreted alongside other
diagnostic plots and metrics.
"""

section("26. RESIDUAL DISTRIBUTION")

np.random.seed(42)

residuals = np.random.normal(
    0,
    5,
    1000
)

fig, ax = plt.subplots()

ax.hist(
    residuals,
    bins=30,
    edgecolor="black",
    alpha=0.75
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Zero Error"
)

ax.set_title("Regression Residual Distribution")
ax.set_xlabel("Residual")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
27. TRAINING VS VALIDATION ERROR DISTRIBUTION
=============================================================================

def error_distribution_comparison() -> None:
"""
Compare example training and validation error distributions.

The values here are illustrative rather than results from a trained model.
"""

section("27. TRAINING VS VALIDATION ERROR DISTRIBUTION")

np.random.seed(42)

training_errors = np.random.normal(
    0,
    2,
    1000
)

validation_errors = np.random.normal(
    0,
    3,
    1000
)

bins = np.linspace(
    -12,
    12,
    35
)

fig, ax = plt.subplots()

ax.hist(
    training_errors,
    bins=bins,
    alpha=0.5,
    label="Training Error"
)

ax.hist(
    validation_errors,
    bins=bins,
    alpha=0.5,
    label="Validation Error"
)

ax.set_title("Example Training vs Validation Error")
ax.set_xlabel("Error")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
28. PANDAS SERIES HISTOGRAM
=============================================================================

def pandas_series_histogram() -> None:
"""Create a histogram directly from a Pandas Series."""

section("28. PANDAS SERIES HISTOGRAM")

np.random.seed(42)

series = pd.Series(
    np.random.normal(
        50,
        10,
        1000
    ),
    name="age"
)

ax = series.plot.hist(
    bins=25,
    edgecolor="black"
)

ax.set_title("Pandas Series Histogram")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
29. PANDAS DATAFRAME HISTOGRAM
=============================================================================

def pandas_dataframe_histogram() -> None:
"""Create histograms for multiple DataFrame columns."""

section("29. PANDAS DATAFRAME HISTOGRAM")

np.random.seed(42)

df = pd.DataFrame({
    "age": np.random.normal(35, 8, 500),
    "income": np.random.normal(60000, 12000, 500),
    "score": np.random.normal(75, 10, 500)
})

axes = df.hist(
    bins=20,
    figsize=(12, 8),
    edgecolor="black"
)

plt.suptitle("Multiple Feature Distributions")

for row in axes:
    for ax in row:
        ax.set_xlabel(ax.get_xlabel())
        ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
30. HISTOGRAM OF MULTIPLE ML FEATURES
=============================================================================

def ml_feature_histograms() -> None:
"""Inspect several ML features at the same time."""

section("30. MULTIPLE ML FEATURE HISTOGRAMS")

np.random.seed(42)

df = pd.DataFrame({
    "age": np.random.normal(35, 8, 1000),
    "income": np.random.lognormal(
        mean=10.5,
        sigma=0.4,
        size=1000
    ),
    "credit_score": np.random.normal(
        700,
        50,
        1000
    ),
    "transactions": np.random.poisson(
        20,
        1000
    )
})

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9)
)

for ax, column in zip(
    axes.ravel(),
    df.columns
):
    ax.hist(
        df[column],
        bins=25,
        edgecolor="black"
    )

    ax.set_title(f"{column.title()} Distribution")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

fig.suptitle("Machine Learning Feature Distributions")

plt.tight_layout()
plt.show()
=============================================================================
31. BEFORE AND AFTER LOG TRANSFORMATION
=============================================================================

def log_transformation_comparison() -> None:
"""
Compare a skewed feature before and after log transformation.

Log transformations can sometimes reduce strong right skew in positive
numerical features.
"""

section("31. BEFORE AND AFTER LOG TRANSFORMATION")

np.random.seed(42)

original = np.random.lognormal(
    mean=3,
    sigma=1,
    size=2000
)

transformed = np.log1p(original)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

axes[0].hist(
    original,
    bins=40,
    edgecolor="black"
)

axes[0].set_title("Original Feature")
axes[0].set_xlabel("Original Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(
    transformed,
    bins=40,
    edgecolor="black"
)

axes[1].set_title("After log1p Transformation")
axes[1].set_xlabel("log1p(Value)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
32. STANDARDIZATION VISUALIZATION
=============================================================================

def standardization_comparison() -> None:
"""
Visualize a feature before and after standardization.

Standardization:

    z = (x - mean) / standard_deviation
"""

section("32. STANDARDIZATION")

np.random.seed(42)

feature = np.random.normal(
    500,
    100,
    2000
)

mean = np.mean(feature)
std = np.std(feature)

standardized = (
    feature - mean
) / std

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

axes[0].hist(
    feature,
    bins=35,
    edgecolor="black"
)

axes[0].set_title("Original Feature")
axes[0].set_xlabel("Original Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(
    standardized,
    bins=35,
    edgecolor="black"
)

axes[1].set_title("Standardized Feature")
axes[1].set_xlabel("Z-Score")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
33. HISTOGRAM WITH QUANTILE LINES
=============================================================================

def quantile_histogram() -> None:
"""Display quartiles on a numerical distribution."""

section("33. QUANTILES")

np.random.seed(42)

data = np.random.normal(
    100,
    15,
    1500
)

q1 = np.percentile(data, 25)
q2 = np.percentile(data, 50)
q3 = np.percentile(data, 75)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=30,
    edgecolor="black",
    alpha=0.75
)

ax.axvline(
    q1,
    linestyle="--",
    linewidth=2,
    label=f"Q1 = {q1:.1f}"
)

ax.axvline(
    q2,
    linestyle="-",
    linewidth=2,
    label=f"Median = {q2:.1f}"
)

ax.axvline(
    q3,
    linestyle="--",
    linewidth=2,
    label=f"Q3 = {q3:.1f}"
)

ax.set_title("Distribution with Quartiles")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
34. REUSABLE HISTOGRAM FUNCTION
=============================================================================

def plot_histogram(
data,
bins=30,
title="Histogram",
xlabel="Value",
ylabel="Frequency",
density=False,
alpha=0.75,
edgecolor="black",
show_mean=False,
show_median=False
):
"""
Reusable histogram plotting function.

Parameters
----------
data : array-like
    Numerical observations.

bins : int or sequence
    Number of bins or explicit bin edges.

title : str
    Plot title.

xlabel : str
    X-axis label.

ylabel : str
    Y-axis label.

density : bool
    If True, display probability density instead of raw frequency.

alpha : float
    Transparency of histogram bars.

edgecolor : str
    Histogram bar edge color.

show_mean : bool
    Display the mean.

show_median : bool
    Display the median.

Returns
-------
matplotlib.axes.Axes
    The created Axes object.
"""

data = np.asarray(data)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=bins,
    density=density,
    alpha=alpha,
    edgecolor=edgecolor
)

if show_mean:
    ax.axvline(
        np.mean(data),
        linestyle="--",
        linewidth=2,
        label=f"Mean = {np.mean(data):.2f}"
    )

if show_median:
    ax.axvline(
        np.median(data),
        linestyle="-.",
        linewidth=2,
        label=f"Median = {np.median(data):.2f}"
    )

ax.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

if show_mean or show_median:
    ax.legend()

fig.tight_layout()

return ax
=============================================================================
35. REUSABLE FUNCTION DEMONSTRATION
=============================================================================

def reusable_function_demo() -> None:
"""Demonstrate the reusable histogram function."""

section("35. REUSABLE HISTOGRAM FUNCTION")

np.random.seed(42)

data = np.random.normal(
    100,
    15,
    1000
)

plot_histogram(
    data,
    bins=30,
    title="Reusable Histogram Function",
    xlabel="Value",
    ylabel="Frequency",
    show_mean=True,
    show_median=True
)

plt.show()
=============================================================================
36. SAVE HISTOGRAM
=============================================================================

def save_histogram() -> None:
"""Save a histogram as a high-resolution image."""

section("36. SAVE HISTOGRAM")

np.random.seed(42)

data = np.random.normal(
    50,
    10,
    1000
)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=30,
    edgecolor="black"
)

ax.set_title("Saved Histogram")
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")

fig.tight_layout()

output_path = "histogram.png"

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Histogram saved to: {output_path}")

plt.close(fig)
=============================================================================
37. COMPLETE MACHINE LEARNING EXAMPLE
=============================================================================

def complete_ml_histogram_example() -> None:
"""
Complete example showing how histograms support ML data exploration.

Workflow:
    1. Create example dataset
    2. Inspect feature distributions
    3. Identify skewness
    4. Compare classes
    5. Inspect residuals
"""

section("37. COMPLETE MACHINE LEARNING EXAMPLE")

np.random.seed(42)

n_samples = 1200

df = pd.DataFrame({
    "age": np.random.normal(
        35,
        9,
        n_samples
    ),

    "income": np.random.lognormal(
        mean=10.5,
        sigma=0.45,
        size=n_samples
    ),

    "credit_score": np.random.normal(
        700,
        55,
        n_samples
    ),

    "prediction_probability": np.random.beta(
        2,
        2,
        n_samples
    )
})

df["age"] = np.clip(
    df["age"],
    18,
    80
)

df["credit_score"] = np.clip(
    df["credit_score"],
    300,
    850
)

print("\nDataset preview:")
print(df.head())

print("\nDescriptive statistics:")
print(df.describe())

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 9)
)

for ax, column in zip(
    axes.ravel(),
    df.columns
):
    ax.hist(
        df[column],
        bins=30,
        edgecolor="black",
        alpha=0.75
    )

    ax.set_title(
        f"{column.replace('_', ' ').title()} Distribution"
    )

    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

fig.suptitle(
    "Machine Learning Dataset Distribution Analysis"
)

plt.tight_layout()
plt.show()
=============================================================================
38. DATA DISTRIBUTION CHECK
=============================================================================

def distribution_summary() -> None:
"""Print simple distribution statistics."""

section("38. DISTRIBUTION SUMMARY")

np.random.seed(42)

data = np.random.normal(
    100,
    20,
    1000
)

mean = np.mean(data)
median = np.median(data)
std = np.std(data)
minimum = np.min(data)
maximum = np.max(data)

print(f"Mean:   {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Std:    {std:.2f}")
print(f"Min:    {minimum:.2f}")
print(f"Max:    {maximum:.2f}")
=============================================================================
39. HISTOGRAM INTERPRETATION
=============================================================================

def histogram_interpretation() -> None:
"""
Print important histogram interpretation concepts.

Useful patterns:
    - Symmetric      -> balanced distribution
    - Right-skewed   -> long right tail
    - Left-skewed    -> long left tail
    - Bimodal        -> two prominent peaks
    - Multimodal     -> multiple peaks
    - Uniform        -> approximately equal frequencies
    - Heavy-tailed   -> many extreme observations
"""

section("39. HISTOGRAM INTERPRETATION")

concepts = {
    "Symmetric": "Left and right sides are approximately balanced.",
    "Right-skewed": "Long tail extends toward larger values.",
    "Left-skewed": "Long tail extends toward smaller values.",
    "Bimodal": "Two prominent peaks may indicate subgroups.",
    "Multimodal": "Multiple peaks may indicate several subpopulations.",
    "Uniform": "Values occur at roughly similar frequencies.",
    "Heavy-tailed": "Extreme values occur more frequently than expected."
}

for pattern, description in concepts.items():
    print(f"{pattern:15} -> {description}")
=============================================================================
40. PROFESSIONAL HISTOGRAM CHECKLIST
=============================================================================

def professional_checklist() -> None:
"""Print a practical checklist for professional visualization."""

section("40. PROFESSIONAL HISTOGRAM CHECKLIST")

checklist = [
    "Choose a meaningful numerical feature.",
    "Check missing values before plotting.",
    "Choose a reasonable bin count.",
    "Use consistent bins when comparing distributions.",
    "Label both axes clearly.",
    "Use density=True when comparing differently sized samples.",
    "Use transparency when overlaying distributions.",
    "Inspect skewness and long tails.",
    "Look for possible outliers.",
    "Compare distributions across classes when useful.",
    "Use transformations when appropriate.",
    "Avoid over-interpreting histogram shape.",
    "Use domain knowledge alongside visual analysis.",
    "Save important figures at sufficient resolution."
]

for index, item in enumerate(
    checklist,
    start=1
):
    print(f"{index:02}. {item}")
=============================================================================
41. COMMON MISTAKES
=============================================================================

def common_mistakes() -> None:
"""Print common histogram mistakes."""

section("41. COMMON MISTAKES")

mistakes = [
    "Using too few bins and hiding important structure.",
    "Using too many bins and creating excessive noise.",
    "Comparing distributions with different bin boundaries.",
    "Ignoring missing values.",
    "Ignoring extreme outliers.",
    "Using frequency when density comparison is required.",
    "Forgetting axis labels.",
    "Using histograms for purely categorical data.",
    "Treating histogram appearance as proof of a statistical assumption.",
    "Applying transformations without understanding the feature."
]

for index, mistake in enumerate(
    mistakes,
    start=1
):
    print(f"{index:02}. {mistake}")
=============================================================================
42. QUICK REFERENCE
=============================================================================

def quick_reference() -> None:
"""Print a compact Matplotlib histogram reference."""

section("42. QUICK REFERENCE")

print(
    """

Basic:
ax.hist(data)

Bins:
ax.hist(data, bins=20)

Density:
ax.hist(data, bins=20, density=True)

Transparency:
ax.hist(data, alpha=0.6)

Edges:
ax.hist(data, edgecolor="black")

Cumulative:
ax.hist(data, cumulative=True)

Horizontal:
ax.hist(data, orientation="horizontal")

Multiple:
ax.hist(data_a, alpha=0.5)
ax.hist(data_b, alpha=0.5)

Save:
fig.savefig("histogram.png", dpi=300)

Mean:
ax.axvline(np.mean(data))

Median:
ax.axvline(np.median(data))
"""
)

=============================================================================
43. MAIN FUNCTION
=============================================================================

def main() -> None:
"""
Run the histogram tutorial.

Most plotting functions call plt.show(), so running the complete tutorial
will display many figures sequentially.

Comment or uncomment individual functions below when studying a specific
concept.
"""

# -------------------------------------------------------------------------
# Basic histogram concepts
# -------------------------------------------------------------------------

basic_histogram()
custom_bins()
explicit_bin_edges()
styled_histogram()
horizontal_histogram()

# -------------------------------------------------------------------------
# Distribution analysis
# -------------------------------------------------------------------------

density_histogram()
frequency_vs_density()
cumulative_histogram()
cumulative_density()

compare_distributions()
multiple_distributions()
density_comparison()

skewed_distribution()
visualize_outliers()

mean_median_histogram()
standard_deviation_histogram()

normal_distribution()
exponential_distribution()

percentage_histogram()

# -------------------------------------------------------------------------
# Machine Learning applications
# -------------------------------------------------------------------------

ml_feature_distribution()
class_conditioned_distribution()
class_imbalance_histogram()

prediction_distribution()
residual_distribution()
error_distribution_comparison()

# -------------------------------------------------------------------------
# Pandas integration
# -------------------------------------------------------------------------

pandas_series_histogram()
pandas_dataframe_histogram()
ml_feature_histograms()

# -------------------------------------------------------------------------
# Feature engineering / preprocessing
# -------------------------------------------------------------------------

log_transformation_comparison()
standardization_comparison()
quantile_histogram()

# -------------------------------------------------------------------------
# Reusable utilities
# -------------------------------------------------------------------------

reusable_function_demo()

# -------------------------------------------------------------------------
# File output
# -------------------------------------------------------------------------

save_histogram()

# -------------------------------------------------------------------------
# Complete ML workflow
# -------------------------------------------------------------------------

complete_ml_histogram_example()

# -------------------------------------------------------------------------
# Terminal-based learning references
# -------------------------------------------------------------------------

distribution_summary()
histogram_interpretation()
professional_checklist()
common_mistakes()
quick_reference()
=============================================================================
44. SCRIPT ENTRY POINT
=============================================================================

if name == "main":
main()

=============================================================================
END OF FILE
=============================================================================
A histogram shows the distribution of numerical data.
Bins determine how observations are grouped.
The choice of bins strongly affects the visual interpretation.
density=True converts the histogram from frequency to probability density.
Histograms are useful for identifying:
- Skewness
- Outliers
- Spread
- Concentration
- Multiple peaks
- Approximate distribution shape
Overlaying histograms can help compare groups, but the same bin edges
should normally be used for a fair visual comparison.
In Machine Learning, histograms are useful during:
- Exploratory Data Analysis
- Feature engineering
- Data preprocessing
- Distribution analysis
- Model diagnostics
- Residual analysis
- Prediction analysis
A histogram is a visualization tool, not a statistical proof.
Always combine visualization with numerical summaries and domain knowledge.
A professional ML workflow should inspect feature distributions before
blindly applying preprocessing transformations.
"""

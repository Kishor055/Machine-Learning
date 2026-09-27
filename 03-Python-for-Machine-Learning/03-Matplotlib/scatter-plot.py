"""
SCATTER PLOTS WITH MATPLOTLIB

File:
03-Python-for-Machine-Learning/03-Matplotlib/scatter-plot.py

Description:
A complete beginner-to-advanced tutorial on creating and customizing
scatter plots with Matplotlib.

Scatter plots are especially useful in Machine Learning for exploring:

    - Relationships between numerical features
    - Correlation
    - Clusters
    - Outliers
    - Regression relationships
    - Feature engineering
    - Classification boundaries
    - Actual vs predicted values
    - Residuals
    - Model errors
    - Multivariate relationships

Requirements:
pip install numpy pandas matplotlib

Run:
python scatter-plot.py

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
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

=============================================================================
3. BASIC SCATTER PLOT
=============================================================================

def basic_scatter() -> None:
"""
Create a basic scatter plot.

Scatter plots represent individual observations as points.
"""

section("3. BASIC SCATTER PLOT")

x = [1, 2, 3, 4, 5, 6]
y = [2, 4, 3, 7, 8, 10]

fig, ax = plt.subplots()

ax.scatter(
    x,
    y
)

ax.set_title("Basic Scatter Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
4. NUMPY SCATTER PLOT
=============================================================================

def numpy_scatter() -> None:
"""Create a scatter plot using NumPy arrays."""

section("4. NUMPY SCATTER PLOT")

np.random.seed(42)

x = np.random.uniform(
    0,
    10,
    100
)

y = (
    2 * x
    + np.random.normal(
        0,
        3,
        100
    )
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y
)

ax.set_title("NumPy Scatter Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
5. MARKER SIZE
=============================================================================

def marker_size() -> None:
"""Customize scatter-point size."""

section("5. MARKER SIZE")

x = np.arange(1, 8)
y = np.array([
    10,
    15,
    12,
    20,
    25,
    22,
    30
])

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    s=120
)

ax.set_title("Scatter Plot with Custom Marker Size")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
6. TRANSPARENCY
=============================================================================

def transparency() -> None:
"""
Use transparency to make overlapping observations easier to inspect.
"""

section("6. TRANSPARENCY")

np.random.seed(42)

x = np.random.normal(
    50,
    10,
    1000
)

y = (
    x * 0.8
    + np.random.normal(
        0,
        8,
        1000
    )
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    alpha=0.35
)

ax.set_title("Scatter Plot with Transparency")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
7. EDGE STYLING
=============================================================================

def edge_styling() -> None:
"""Customize marker edges."""

section("7. EDGE STYLING")

x = np.arange(1, 11)
y = x + np.random.default_rng(42).normal(
    0,
    2,
    10
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    s=100,
    alpha=0.75,
    edgecolors="black",
    linewidths=1
)

ax.set_title("Scatter Plot with Marker Edges")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
8. CUSTOM MARKERS
=============================================================================

def custom_markers() -> None:
"""Demonstrate different marker shapes."""

section("8. CUSTOM MARKERS")

x = np.arange(1, 7)
y = np.array([
    10,
    15,
    12,
    20,
    25,
    30
])

markers = [
    "o",
    "s",
    "^",
    "D",
    "*",
    "P"
]

fig, ax = plt.subplots()

for index, marker in enumerate(markers):
    ax.scatter(
        x[index],
        y[index],
        s=150,
        marker=marker
    )

ax.set_title("Different Scatter Markers")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
9. MULTIPLE GROUPS
=============================================================================

def multiple_groups() -> None:
"""Compare two groups in a scatter plot."""

section("9. MULTIPLE GROUPS")

np.random.seed(42)

group_a_x = np.random.normal(
    30,
    5,
    100
)

group_a_y = (
    group_a_x
    + np.random.normal(
        0,
        4,
        100
    )
)

group_b_x = np.random.normal(
    60,
    5,
    100
)

group_b_y = (
    group_b_x
    + np.random.normal(
        0,
        4,
        100
    )
)

fig, ax = plt.subplots()

ax.scatter(
    group_a_x,
    group_a_y,
    alpha=0.7,
    label="Group A"
)

ax.scatter(
    group_b_x,
    group_b_y,
    alpha=0.7,
    label="Group B"
)

ax.set_title("Scatter Plot by Group")
ax.set_xlabel("Feature X")
ax.set_ylabel("Feature Y")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
10. COLOR BY CLASS
=============================================================================

def color_by_class() -> None:
"""
Color observations according to class labels.

This is a common visualization technique for classification datasets.
"""

section("10. COLOR BY CLASS")

np.random.seed(42)

class_0_x = np.random.normal(
    40,
    7,
    150
)

class_0_y = np.random.normal(
    40,
    7,
    150
)

class_1_x = np.random.normal(
    60,
    7,
    150
)

class_1_y = np.random.normal(
    60,
    7,
    150
)

fig, ax = plt.subplots()

ax.scatter(
    class_0_x,
    class_0_y,
    alpha=0.6,
    label="Class 0"
)

ax.scatter(
    class_1_x,
    class_1_y,
    alpha=0.6,
    label="Class 1"
)

ax.set_title("Scatter Plot by Class")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
11. SIZE REPRESENTS A THIRD VARIABLE
=============================================================================

def size_represents_variable() -> None:
"""
Encode a third numerical variable using marker size.

This creates a simple form of multivariate visualization.
"""

section("11. SIZE AS THIRD VARIABLE")

np.random.seed(42)

x = np.random.uniform(
    10,
    100,
    80
)

y = (
    0.7 * x
    + np.random.normal(
        0,
        10,
        80
    )
)

sizes = np.random.uniform(
    30,
    500,
    80
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    s=sizes,
    alpha=0.5
)

ax.set_title("Marker Size Represents a Third Variable")
ax.set_xlabel("Feature X")
ax.set_ylabel("Feature Y")

plt.tight_layout()
plt.show()
=============================================================================
12. COLOR REPRESENTS A THIRD VARIABLE
=============================================================================

def color_represents_variable() -> None:
"""Use color intensity to represent another numerical variable."""

section("12. COLOR AS THIRD VARIABLE")

np.random.seed(42)

x = np.random.uniform(
    0,
    10,
    150
)

y = (
    2 * x
    + np.random.normal(
        0,
        3,
        150
    )
)

third_variable = (
    x
    + y
)

fig, ax = plt.subplots()

scatter = ax.scatter(
    x,
    y,
    c=third_variable,
    alpha=0.75
)

fig.colorbar(
    scatter,
    ax=ax,
    label="Third Variable"
)

ax.set_title("Color Represents a Third Variable")
ax.set_xlabel("Feature X")
ax.set_ylabel("Feature Y")

plt.tight_layout()
plt.show()
=============================================================================
13. CORRELATION
=============================================================================

def correlation_visualization() -> None:
"""
Visualize a positive correlation.

Correlation measures the strength and direction of a linear relationship.
A scatter plot should be inspected alongside the numerical correlation.
"""

section("13. CORRELATION")

np.random.seed(42)

x = np.random.normal(
    50,
    10,
    300
)

y = (
    0.75 * x
    + np.random.normal(
        0,
        5,
        300
    )
)

correlation = np.corrcoef(
    x,
    y
)[0, 1]

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    alpha=0.5
)

ax.set_title(
    f"Scatter Plot - Correlation = {correlation:.2f}"
)

ax.set_xlabel("Feature X")
ax.set_ylabel("Feature Y")

plt.tight_layout()
plt.show()
=============================================================================
14. POSITIVE, NEGATIVE AND WEAK CORRELATION
=============================================================================

def correlation_types() -> None:
"""Compare different correlation patterns."""

section("14. CORRELATION TYPES")

np.random.seed(42)

x = np.linspace(
    0,
    10,
    100
)

positive = (
    2 * x
    + np.random.normal(
        0,
        2,
        100
    )
)

negative = (
    -2 * x
    + np.random.normal(
        0,
        2,
        100
    )
)

weak = np.random.normal(
    0,
    5,
    100
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

axes[0].scatter(
    x,
    positive,
    alpha=0.6
)

axes[0].set_title("Positive Correlation")

axes[1].scatter(
    x,
    negative,
    alpha=0.6
)

axes[1].set_title("Negative Correlation")

axes[2].scatter(
    x,
    weak,
    alpha=0.6
)

axes[2].set_title("Weak / No Linear Pattern")

for ax in axes:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
15. NON-LINEAR RELATIONSHIP
=============================================================================

def nonlinear_relationship() -> None:
"""
Demonstrate a relationship that is clearly non-linear.

A near-zero Pearson correlation does not necessarily imply that two
variables have no relationship.
"""

section("15. NON-LINEAR RELATIONSHIP")

np.random.seed(42)

x = np.linspace(
    -5,
    5,
    300
)

y = (
    x ** 2
    + np.random.normal(
        0,
        2,
        len(x)
    )
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    alpha=0.5
)

ax.set_title("Non-Linear Relationship")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
16. OUTLIERS
=============================================================================

def visualize_outliers() -> None:
"""Visualize extreme observations in a scatter plot."""

section("16. OUTLIERS")

np.random.seed(42)

x = np.random.normal(
    50,
    8,
    200
)

y = (
    1.5 * x
    + np.random.normal(
        0,
        5,
        200
    )
)

outlier_x = np.array([
    90,
    95,
    100
])

outlier_y = np.array([
    10,
    5,
    15
])

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    alpha=0.5,
    label="Main Data"
)

ax.scatter(
    outlier_x,
    outlier_y,
    s=100,
    marker="x",
    label="Potential Outliers"
)

ax.set_title("Scatter Plot with Potential Outliers")
ax.set_xlabel("Feature X")
ax.set_ylabel("Feature Y")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
17. REGRESSION LINE
=============================================================================

def regression_line() -> None:
"""
Add a simple least-squares regression line.

For a model:

    y = mx + b

the slope and intercept are estimated from the observations.
"""

section("17. REGRESSION LINE")

np.random.seed(42)

x = np.linspace(
    0,
    10,
    100
)

y = (
    3 * x
    + 5
    + np.random.normal(
        0,
        4,
        100
    )
)

slope, intercept = np.polyfit(
    x,
    y,
    1
)

predicted = (
    slope * x
    + intercept
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    alpha=0.5,
    label="Observations"
)

ax.plot(
    x,
    predicted,
    linewidth=2,
    label=(
        f"Regression: "
        f"y = {slope:.2f}x + {intercept:.2f}"
    )
)

ax.set_title("Scatter Plot with Regression Line")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
18. RESIDUALS
=============================================================================

def residual_visualization() -> None:
"""
Visualize regression residuals.

Residual:

    residual = actual - predicted
"""

section("18. RESIDUAL VISUALIZATION")

np.random.seed(42)

x = np.linspace(
    0,
    10,
    150
)

actual = (
    4 * x
    + 3
    + np.random.normal(
        0,
        4,
        len(x)
    )
)

slope, intercept = np.polyfit(
    x,
    actual,
    1
)

predicted = (
    slope * x
    + intercept
)

residuals = (
    actual - predicted
)

fig, ax = plt.subplots()

ax.scatter(
    predicted,
    residuals,
    alpha=0.6
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=2
)

ax.set_title("Residuals vs Predicted Values")
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Residual")

plt.tight_layout()
plt.show()
=============================================================================
19. ACTUAL VS PREDICTED
=============================================================================

def actual_vs_predicted() -> None:
"""Visualize actual and predicted regression values."""

section("19. ACTUAL VS PREDICTED")

np.random.seed(42)

actual = np.linspace(
    20,
    100,
    150
)

predicted = (
    actual
    + np.random.normal(
        0,
        7,
        len(actual)
    )
)

fig, ax = plt.subplots()

ax.scatter(
    actual,
    predicted,
    alpha=0.6
)

minimum = min(
    actual.min(),
    predicted.min()
)

maximum = max(
    actual.max(),
    predicted.max()
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

ax.set_title("Actual vs Predicted Values")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
20. CLUSTERING
=============================================================================

def clustering_visualization() -> None:
"""
Visualize synthetic clusters.

Scatter plots are commonly used to inspect clustering results when the
data can be meaningfully represented in two dimensions.
"""

section("20. CLUSTERING")

np.random.seed(42)

cluster_a = np.random.normal(
    loc=[20, 20],
    scale=3,
    size=(100, 2)
)

cluster_b = np.random.normal(
    loc=[50, 50],
    scale=3,
    size=(100, 2)
)

cluster_c = np.random.normal(
    loc=[80, 25],
    scale=3,
    size=(100, 2)
)

fig, ax = plt.subplots()

ax.scatter(
    cluster_a[:, 0],
    cluster_a[:, 1],
    alpha=0.7,
    label="Cluster A"
)

ax.scatter(
    cluster_b[:, 0],
    cluster_b[:, 1],
    alpha=0.7,
    label="Cluster B"
)

ax.scatter(
    cluster_c[:, 0],
    cluster_c[:, 1],
    alpha=0.7,
    label="Cluster C"
)

ax.set_title("Synthetic Clustering Visualization")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
21. CLUSTER CENTROIDS
=============================================================================

def cluster_centroids() -> None:
"""Display cluster observations and their synthetic centroids."""

section("21. CLUSTER CENTROIDS")

np.random.seed(42)

centers = np.array([
    [20, 20],
    [50, 50],
    [80, 25]
])

clusters = []

for center in centers:
    points = np.random.normal(
        center,
        4,
        size=(80, 2)
    )

    clusters.append(points)

fig, ax = plt.subplots()

for index, points in enumerate(clusters):
    ax.scatter(
        points[:, 0],
        points[:, 1],
        alpha=0.5,
        label=f"Cluster {index + 1}"
    )

ax.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

ax.set_title("Clusters and Centroids")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
22. CLASSIFICATION DATA
=============================================================================

def classification_data() -> None:
"""
Visualize a simple two-class dataset.

The plot can help inspect whether two classes appear visually separable
in the selected two-dimensional feature space.
"""

section("22. CLASSIFICATION DATA")

np.random.seed(42)

class_0 = np.random.normal(
    loc=[30, 30],
    scale=5,
    size=(150, 2)
)

class_1 = np.random.normal(
    loc=[60, 60],
    scale=5,
    size=(150, 2)
)

fig, ax = plt.subplots()

ax.scatter(
    class_0[:, 0],
    class_0[:, 1],
    alpha=0.6,
    label="Class 0"
)

ax.scatter(
    class_1[:, 0],
    class_1[:, 1],
    alpha=0.6,
    label="Class 1"
)

ax.set_title("Two-Class Feature Visualization")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
23. DECISION BOUNDARY
=============================================================================

def decision_boundary() -> None:
"""
Demonstrate a simple linear decision boundary.

This is an educational visualization rather than the output of a trained
classification model.
"""

section("23. DECISION BOUNDARY")

np.random.seed(42)

class_0 = np.random.normal(
    loc=[30, 30],
    scale=4,
    size=(100, 2)
)

class_1 = np.random.normal(
    loc=[70, 70],
    scale=4,
    size=(100, 2)
)

x_boundary = np.array([
    10,
    90
])

y_boundary = (
    x_boundary
    + 5
)

fig, ax = plt.subplots()

ax.scatter(
    class_0[:, 0],
    class_0[:, 1],
    alpha=0.6,
    label="Class 0"
)

ax.scatter(
    class_1[:, 0],
    class_1[:, 1],
    alpha=0.6,
    label="Class 1"
)

ax.plot(
    x_boundary,
    y_boundary,
    linestyle="--",
    linewidth=2,
    label="Illustrative Boundary"
)

ax.set_title("Illustrative Classification Boundary")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
24. BUBBLE SCATTER PLOT
=============================================================================

def bubble_scatter() -> None:
"""
Create a bubble-style scatter plot.

Bubble size represents an additional numerical variable.
"""

section("24. BUBBLE SCATTER PLOT")

np.random.seed(42)

x = np.random.uniform(
    10,
    100,
    50
)

y = np.random.uniform(
    20,
    200,
    50
)

size = np.random.uniform(
    50,
    800,
    50
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    s=size,
    alpha=0.4
)

ax.set_title("Bubble Scatter Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
25. SCATTER MATRIX WITH PANDAS
=============================================================================

def pandas_scatter_matrix() -> None:
"""
Create a scatter matrix using Pandas.

Scatter matrices are useful for quickly inspecting pairwise relationships
among several numerical features.
"""

section("25. PANDAS SCATTER MATRIX")

np.random.seed(42)

df = pd.DataFrame({
    "age": np.random.normal(
        35,
        8,
        120
    ),

    "income": np.random.normal(
        60000,
        12000,
        120
    ),

    "score": np.random.normal(
        75,
        10,
        120
    )
})

axes = pd.plotting.scatter_matrix(
    df,
    figsize=(10, 10),
    diagonal="hist",
    alpha=0.6
)

fig = axes[0, 0].get_figure()

fig.suptitle(
    "Pandas Scatter Matrix",
    y=1.02
)

plt.tight_layout()
plt.show()
=============================================================================
26. FEATURE RELATIONSHIP
=============================================================================

def feature_relationship() -> None:
"""
Inspect relationships between two ML features.

This is an exploratory analysis example.
"""

section("26. MACHINE LEARNING FEATURE RELATIONSHIP")

np.random.seed(42)

age = np.random.normal(
    35,
    10,
    300
)

income = (
    25000
    + age * 1200
    + np.random.normal(
        0,
        12000,
        300
    )
)

fig, ax = plt.subplots()

ax.scatter(
    age,
    income,
    alpha=0.5
)

ax.set_title("Age vs Income")
ax.set_xlabel("Age")
ax.set_ylabel("Income")

plt.tight_layout()
plt.show()
=============================================================================
27. FEATURE VS TARGET
=============================================================================

def feature_vs_target() -> None:
"""
Visualize a numerical feature against a numerical target.

This can help identify:
    - Linear relationships
    - Non-linear relationships
    - Outliers
    - Heteroscedasticity
    - Potential feature usefulness
"""

section("27. FEATURE VS TARGET")

np.random.seed(42)

feature = np.random.uniform(
    0,
    100,
    250
)

target = (
    2.5 * feature
    + np.random.normal(
        0,
        25,
        250
    )
)

fig, ax = plt.subplots()

ax.scatter(
    feature,
    target,
    alpha=0.5
)

ax.set_title("Feature vs Target")
ax.set_xlabel("Feature")
ax.set_ylabel("Target")

plt.tight_layout()
plt.show()
=============================================================================
28. HETEROSCEDASTICITY
=============================================================================

def heteroscedasticity() -> None:
"""
Demonstrate changing variance across the x-axis.

Heteroscedasticity means the spread of errors or observations changes
across levels of a predictor.
"""

section("28. HETEROSCEDASTICITY")

np.random.seed(42)

x = np.linspace(
    1,
    10,
    300
)

noise = np.random.normal(
    0,
    x * 1.5,
    len(x)
)

y = (
    3 * x
    + noise
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    alpha=0.5
)

ax.set_title("Heteroscedasticity Example")
ax.set_xlabel("Feature")
ax.set_ylabel("Target")

plt.tight_layout()
plt.show()
=============================================================================
29. LOG SCALE SCATTER
=============================================================================

def logarithmic_scatter() -> None:
"""Use logarithmic axes for wide-ranging numerical data."""

section("29. LOGARITHMIC SCATTER PLOT")

np.random.seed(42)

x = np.logspace(
    0,
    4,
    200
)

y = (
    2 * x ** 1.5
    * np.exp(
        np.random.normal(
            0,
            0.4,
            len(x)
        )
    )
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    alpha=0.5
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_title("Scatter Plot with Logarithmic Axes")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
30. DENSITY-LIKE VISUALIZATION
=============================================================================

def dense_scatter() -> None:
"""
Use transparency to inspect a high-density scatter dataset.

For extremely large datasets, specialized density plots or hexbin plots
may be more informative.
"""

section("30. DENSE SCATTER DATA")

np.random.seed(42)

x = np.random.normal(
    50,
    12,
    10000
)

y = (
    1.2 * x
    + np.random.normal(
        0,
        10,
        10000
    )
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    alpha=0.08,
    s=15
)

ax.set_title("Dense Scatter Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
31. HEXBIN
=============================================================================

def hexbin_plot() -> None:
"""
Use hexbin for large scatter datasets.

Hexbin groups observations into hexagonal regions and is often useful
when millions of overlapping points make a regular scatter plot difficult
to interpret.
"""

section("31. HEXBIN PLOT")

np.random.seed(42)

x = np.random.normal(
    50,
    12,
    10000
)

y = (
    1.2 * x
    + np.random.normal(
        0,
        10,
        10000
    )
)

fig, ax = plt.subplots()

image = ax.hexbin(
    x,
    y,
    gridsize=35
)

fig.colorbar(
    image,
    ax=ax,
    label="Count"
)

ax.set_title("Hexbin Visualization")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
32. ERROR BARS
=============================================================================

def scatter_error_bars() -> None:
"""Add horizontal and vertical uncertainty to observations."""

section("32. SCATTER WITH ERROR BARS")

x = np.arange(1, 8)

y = np.array([
    20,
    24,
    22,
    30,
    35,
    33,
    40
])

x_error = np.array([
    0.2,
    0.3,
    0.2,
    0.3,
    0.2,
    0.3,
    0.2
])

y_error = np.array([
    2,
    3,
    2,
    3,
    2,
    3,
    2
])

fig, ax = plt.subplots()

ax.errorbar(
    x,
    y,
    xerr=x_error,
    yerr=y_error,
    fmt="o",
    capsize=4
)

ax.set_title("Scatter Plot with Error Bars")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
33. PANDAS DATAFRAME SCATTER
=============================================================================

def pandas_dataframe_scatter() -> None:
"""Create a scatter plot from DataFrame columns."""

section("33. PANDAS DATAFRAME SCATTER")

np.random.seed(42)

df = pd.DataFrame({
    "hours_studied": np.random.uniform(
        1,
        10,
        100
    )
})

df["exam_score"] = (
    45
    + 5 * df["hours_studied"]
    + np.random.normal(
        0,
        5,
        100
    )
)

ax = df.plot.scatter(
    x="hours_studied",
    y="exam_score",
    alpha=0.7
)

ax.set_title("Study Hours vs Exam Score")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Exam Score")

plt.tight_layout()
plt.show()
=============================================================================
34. SCATTER WITH TREND LINE
=============================================================================

def scatter_with_trend_line() -> None:
"""Create a reusable trend-line visualization."""

section("34. SCATTER WITH TREND LINE")

np.random.seed(42)

x = np.random.uniform(
    0,
    100,
    200
)

y = (
    1.8 * x
    + 20
    + np.random.normal(
        0,
        15,
        200
    )
)

slope, intercept = np.polyfit(
    x,
    y,
    1
)

order = np.argsort(x)

x_sorted = x[order]

y_trend = (
    slope * x_sorted
    + intercept
)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    alpha=0.45,
    label="Observations"
)

ax.plot(
    x_sorted,
    y_trend,
    linewidth=2,
    label="Linear Trend"
)

ax.set_title("Scatter Plot with Trend Line")
ax.set_xlabel("Feature")
ax.set_ylabel("Target")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
35. REUSABLE SCATTER FUNCTION
=============================================================================

def plot_scatter(
x,
y,
title="Scatter Plot",
xlabel="X",
ylabel="Y",
label=None,
size=60,
alpha=0.7,
marker="o",
show_grid=True,
show_trend=False
):
"""
Create a reusable scatter plot.

Parameters
----------
x : array-like
    X-axis observations.

y : array-like
    Y-axis observations.

title : str
    Plot title.

xlabel : str
    X-axis label.

ylabel : str
    Y-axis label.

label : str or None
    Optional legend label.

size : float
    Marker size.

alpha : float
    Marker transparency.

marker : str
    Marker style.

show_grid : bool
    Whether to show gridlines.

show_trend : bool
    Whether to add a simple linear trend line.

Returns
-------
matplotlib.axes.Axes
    Created Axes object.
"""

x = np.asarray(x)
y = np.asarray(y)

fig, ax = plt.subplots()

ax.scatter(
    x,
    y,
    s=size,
    alpha=alpha,
    marker=marker,
    label=label
)

if show_trend:
    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    order = np.argsort(x)

    x_sorted = x[order]

    trend = (
        slope * x_sorted
        + intercept
    )

    ax.plot(
        x_sorted,
        trend,
        linewidth=2,
        label="Linear Trend"
    )

ax.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

if show_grid:
    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

if label is not None or show_trend:
    ax.legend()

fig.tight_layout()

return ax
=============================================================================
36. REUSABLE FUNCTION DEMONSTRATION
=============================================================================

def reusable_function_demo() -> None:
"""Demonstrate the reusable scatter plotting function."""

section("36. REUSABLE SCATTER FUNCTION")

np.random.seed(42)

x = np.random.uniform(
    0,
    20,
    150
)

y = (
    2.5 * x
    + np.random.normal(
        0,
        6,
        150
    )
)

plot_scatter(
    x,
    y,
    title="Reusable Scatter Plot",
    xlabel="Feature X",
    ylabel="Target Y",
    label="Observations",
    show_trend=True
)

plt.show()
=============================================================================
37. COMPLETE MACHINE LEARNING EXAMPLE
=============================================================================

def complete_ml_scatter_example() -> None:
"""
Complete ML-oriented scatter plot example.

The example demonstrates:
    - Feature vs target
    - Class information
    - Correlation
    - Trend line
    - Outlier inspection

The data is synthetic for educational purposes.
"""

section("37. COMPLETE MACHINE LEARNING SCATTER EXAMPLE")

np.random.seed(42)

n_samples = 500

age = np.random.uniform(
    18,
    70,
    n_samples
)

income = (
    20000
    + age * 1500
    + np.random.normal(
        0,
        15000,
        n_samples
    )
)

income = np.maximum(
    income,
    5000
)

target = (
    0.5 * age
    + 0.00002 * income
    + np.random.normal(
        0,
        5,
        n_samples
    )
)

correlation = np.corrcoef(
    age,
    income
)[0, 1]

slope, intercept = np.polyfit(
    age,
    income,
    1
)

order = np.argsort(age)

age_sorted = age[order]

trend = (
    slope * age_sorted
    + intercept
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 5)
)

# -------------------------------------------------------------------------
# Age vs Income
# -------------------------------------------------------------------------

axes[0].scatter(
    age,
    income,
    alpha=0.35
)

axes[0].plot(
    age_sorted,
    trend,
    linewidth=2
)

axes[0].set_title(
    f"Age vs Income (r = {correlation:.2f})"
)

axes[0].set_xlabel("Age")
axes[0].set_ylabel("Income")

# -------------------------------------------------------------------------
# Age vs Target
# -------------------------------------------------------------------------

axes[1].scatter(
    age,
    target,
    alpha=0.35
)

axes[1].set_title(
    "Age vs Target"
)

axes[1].set_xlabel("Age")
axes[1].set_ylabel("Target")

fig.suptitle(
    "Machine Learning Feature Relationship Analysis"
)

plt.tight_layout()
plt.show()
=============================================================================
38. PROFESSIONAL SCATTER PLOT CHECKLIST
=============================================================================

def professional_checklist() -> None:
"""Print a practical scatter-plot checklist."""

section("38. PROFESSIONAL SCATTER PLOT CHECKLIST")

checklist = [
    "Use scatter plots for relationships between numerical variables.",
    "Clearly label both axes.",
    "Use transparency for dense datasets.",
    "Use consistent scales when comparison requires it.",
    "Use color or marker shape to represent meaningful groups.",
    "Use marker size for a meaningful third numerical variable.",
    "Avoid encoding too many variables in one chart.",
    "Inspect possible outliers.",
    "Inspect non-linear relationships.",
    "Do not assume correlation proves causation.",
    "Use a trend line when it improves interpretation.",
    "Use the same x/y scale when geometric comparison matters.",
    "Use hexbin for very dense datasets.",
    "Use log scales for variables spanning several orders of magnitude.",
    "For ML, inspect feature-feature and feature-target relationships."
]

for index, item in enumerate(
    checklist,
    start=1
):
    print(f"{index:02}. {item}")
=============================================================================
39. COMMON MISTAKES
=============================================================================

def common_mistakes() -> None:
"""Print common scatter-plot mistakes."""

section("39. COMMON MISTAKES")

mistakes = [
    "Assuming correlation means causation.",
    "Ignoring non-linear relationships.",
    "Ignoring outliers.",
    "Using opaque markers on dense data.",
    "Encoding too many variables simultaneously.",
    "Using misleading axis limits.",
    "Forgetting units.",
    "Using a line plot when observations are unordered.",
    "Adding a regression line without explaining what it represents.",
    "Interpreting synthetic educational data as real-world evidence."
]

for index, mistake in enumerate(
    mistakes,
    start=1
):
    print(f"{index:02}. {mistake}")
=============================================================================
40. QUICK REFERENCE
=============================================================================

def quick_reference() -> None:
"""Print a compact Matplotlib scatter-plot reference."""

section("40. QUICK REFERENCE")

print(
    """

Basic:
ax.scatter(x, y)

Marker size:
ax.scatter(x, y, s=100)

Transparency:
ax.scatter(x, y, alpha=0.5)

Color:
ax.scatter(x, y, c=value)

Multiple groups:
ax.scatter(x1, y1, label="Group A")
ax.scatter(x2, y2, label="Group B")
ax.legend()

Third variable:
ax.scatter(x, y, s=sizes, c=values)

Colorbar:
scatter = ax.scatter(x, y, c=values)
fig.colorbar(scatter, ax=ax)

Regression:
slope, intercept = np.polyfit(x, y, 1)

Trend:
y_pred = slope * x + intercept

Horizontal reference:
ax.axhline(0)

Vertical reference:
ax.axvline(0)

Log scale:
ax.set_xscale("log")
ax.set_yscale("log")

Dense data:
ax.hexbin(x, y)

Pandas:
df.plot.scatter(
x="feature",
y="target"
)

Save:
fig.savefig(
"scatter-plot.png",
dpi=300
)
"""
)

=============================================================================
41. MAIN FUNCTION
=============================================================================

def main() -> None:
"""
Run the complete scatter-plot tutorial.

The module contains many examples. During study, individual functions
can be called instead of running every visualization.
"""

# -------------------------------------------------------------------------
# Fundamentals
# -------------------------------------------------------------------------

basic_scatter()
numpy_scatter()
marker_size()
transparency()
edge_styling()
custom_markers()

# -------------------------------------------------------------------------
# Groups and multivariate visualization
# -------------------------------------------------------------------------

multiple_groups()
color_by_class()
size_represents_variable()
color_represents_variable()

# -------------------------------------------------------------------------
# Relationship analysis
# -------------------------------------------------------------------------

correlation_visualization()
correlation_types()
nonlinear_relationship()
visualize_outliers()
regression_line()
residual_visualization()
actual_vs_predicted()

# -------------------------------------------------------------------------
# Machine Learning concepts
# -------------------------------------------------------------------------

clustering_visualization()
cluster_centroids()
classification_data()
decision_boundary()
bubble_scatter()

# -------------------------------------------------------------------------
# Pandas and feature analysis
# -------------------------------------------------------------------------

pandas_scatter_matrix()
feature_relationship()
feature_vs_target()
heteroscedasticity()

# -------------------------------------------------------------------------
# Advanced visualization
# -------------------------------------------------------------------------

logarithmic_scatter()
dense_scatter()
hexbin_plot()
scatter_error_bars()
pandas_dataframe_scatter()
scatter_with_trend_line()

# -------------------------------------------------------------------------
# Reusable utility
# -------------------------------------------------------------------------

reusable_function_demo()

# -------------------------------------------------------------------------
# Complete ML example
# -------------------------------------------------------------------------

complete_ml_scatter_example()

# -------------------------------------------------------------------------
# Learning references
# -------------------------------------------------------------------------

professional_checklist()
common_mistakes()
quick_reference()
=============================================================================
42. SCRIPT ENTRY POINT
=============================================================================

if name == "main":
main()

=============================================================================
END OF FILE
=============================================================================
Scatter plots show relationships between two numerical variables.
Each point represents one observation.
Scatter plots are useful for identifying:
- Positive relationships
- Negative relationships
- Weak relationships
- Non-linear relationships
- Clusters
- Outliers
- Changing variance
Color can represent a class or another numerical variable.
Marker size can represent a third numerical variable.
Correlation should be interpreted together with the actual scatter pattern.
Correlation does not prove causation.
A regression line summarizes a linear relationship but does not prove that
the relationship is causal.
In Machine Learning, scatter plots are particularly useful for:
- Feature exploration
- Feature-target analysis
- Classification analysis
- Regression analysis
- Clustering
- Residual analysis
- Model diagnostics
For very dense datasets, consider:
Transparency
Smaller markers
Hexbin plots
Density-based visualization
Always label axes and document what each visual encoding represents.
Choose the visualization based on the data:
Numerical relationship -> scatter plot
Ordered trend -> line plot
Distribution -> histogram
Categorical comparison -> bar chart
"""

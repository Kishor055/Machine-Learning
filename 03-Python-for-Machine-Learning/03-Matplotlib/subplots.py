"""
SUBPLOTS WITH MATPLOTLIB

File:
03-Python-for-Machine-Learning/03-Matplotlib/subplots.py

Description:
A complete beginner-to-advanced tutorial on creating multiple plots in
a single Matplotlib figure.

Subplots are useful for:

    - Comparing multiple visualizations
    - Building ML analysis dashboards
    - Feature exploration
    - Model evaluation
    - Distribution analysis
    - Comparing training and validation metrics
    - Residual analysis
    - Presenting related charts together

Requirements:
pip install numpy pandas matplotlib

Run:
python subplots.py

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
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

def section(title: str) -> None:
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

=============================================================================
3. BASIC SUBPLOT
=============================================================================

def basic_subplot() -> None:
"""Create two plots inside one figure."""

section("3. BASIC SUBPLOT")

x = np.arange(1, 11)

y1 = x ** 2
y2 = x * 3

fig, axes = plt.subplots(
    1,
    2
)

axes[0].plot(
    x,
    y1
)

axes[0].set_title("Quadratic")
axes[0].set_xlabel("X")
axes[0].set_ylabel("X²")

axes[1].plot(
    x,
    y2
)

axes[1].set_title("Linear")
axes[1].set_xlabel("X")
axes[1].set_ylabel("3X")

fig.suptitle("Basic Subplots")

plt.tight_layout()
plt.show()
=============================================================================
4. TWO ROWS
=============================================================================

def two_rows() -> None:
"""Create two vertically stacked plots."""

section("4. TWO ROWS")

x = np.linspace(
    0,
    10,
    100
)

fig, axes = plt.subplots(
    2,
    1
)

axes[0].plot(
    x,
    np.sin(x)
)

axes[0].set_title("Sine Wave")

axes[1].plot(
    x,
    np.cos(x)
)

axes[1].set_title("Cosine Wave")

fig.suptitle("Vertical Subplots")

plt.tight_layout()
plt.show()
=============================================================================
5. TWO COLUMNS
=============================================================================

def two_columns() -> None:
"""Create two horizontally arranged plots."""

section("5. TWO COLUMNS")

x = np.linspace(
    0,
    10,
    100
)

fig, axes = plt.subplots(
    1,
    2
)

axes[0].plot(
    x,
    np.sin(x)
)

axes[0].set_title("Sine")

axes[1].plot(
    x,
    np.cos(x)
)

axes[1].set_title("Cosine")

plt.tight_layout()
plt.show()
=============================================================================
6. 2x2 SUBPLOT GRID
=============================================================================

def grid_2x2() -> None:
"""Create a four-panel subplot grid."""

section("6. 2x2 SUBPLOT GRID")

x = np.linspace(
    0,
    10,
    100
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

axes[0, 0].plot(
    x,
    x
)

axes[0, 0].set_title("Linear")

axes[0, 1].plot(
    x,
    x ** 2
)

axes[0, 1].set_title("Quadratic")

axes[1, 0].plot(
    x,
    np.sin(x)
)

axes[1, 0].set_title("Sine")

axes[1, 1].plot(
    x,
    np.cos(x)
)

axes[1, 1].set_title("Cosine")

fig.suptitle(
    "2 × 2 Subplot Grid",
    fontsize=16
)

plt.tight_layout()
plt.show()
=============================================================================
7. 3x3 SUBPLOT GRID
=============================================================================

def grid_3x3() -> None:
"""Create a nine-panel subplot grid."""

section("7. 3x3 SUBPLOT GRID")

x = np.linspace(
    0,
    10,
    100
)

fig, axes = plt.subplots(
    3,
    3,
    figsize=(12, 10)
)

functions = [
    ("Linear", x),
    ("Quadratic", x ** 2),
    ("Cubic", x ** 3),
    ("Sine", np.sin(x)),
    ("Cosine", np.cos(x)),
    ("Tangent", np.tan(x)),
    ("Square Root", np.sqrt(x)),
    ("Logarithm", np.log1p(x)),
    ("Exponential", np.exp(x / 3))
]

for ax, (title, values) in zip(
    axes.flat,
    functions
):
    ax.plot(
        x,
        values
    )

    ax.set_title(title)
    ax.grid(
        True,
        alpha=0.3
    )

fig.suptitle(
    "3 × 3 Subplot Grid",
    fontsize=16
)

plt.tight_layout()
plt.show()
=============================================================================
8. ITERATING OVER AXES
=============================================================================

def iterate_axes() -> None:
"""Demonstrate efficient iteration over subplot axes."""

section("8. ITERATING OVER AXES")

x = np.linspace(
    0,
    10,
    100
)

values = [
    x,
    x ** 2,
    np.sqrt(x),
    np.sin(x)
]

titles = [
    "Linear",
    "Quadratic",
    "Square Root",
    "Sine"
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8)
)

for ax, values, title in zip(
    axes.flat,
    values,
    titles
):
    ax.plot(
        x,
        values
    )

    ax.set_title(title)

plt.tight_layout()
plt.show()
=============================================================================
9. SHARED X-AXIS
=============================================================================

def shared_x_axis() -> None:
"""Create subplots sharing the same X-axis."""

section("9. SHARED X-AXIS")

x = np.arange(
    1,
    13
)

sales = np.array([
    120,
    135,
    142,
    150,
    160,
    175,
    180,
    190,
    210,
    220,
    235,
    250
])

profit = np.array([
    20,
    22,
    25,
    27,
    30,
    34,
    35,
    38,
    42,
    45,
    48,
    52
])

fig, axes = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(10, 7)
)

axes[0].plot(
    x,
    sales,
    marker="o"
)

axes[0].set_title("Monthly Sales")
axes[0].set_ylabel("Sales")

axes[1].plot(
    x,
    profit,
    marker="o"
)

axes[1].set_title("Monthly Profit")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Profit")

plt.tight_layout()
plt.show()
=============================================================================
10. SHARED Y-AXIS
=============================================================================

def shared_y_axis() -> None:
"""Create subplots sharing the same Y-axis."""

section("10. SHARED Y-AXIS")

categories = [
    "A",
    "B",
    "C",
    "D",
    "E"
]

values_a = [
    10,
    20,
    15,
    25,
    30
]

values_b = [
    12,
    18,
    20,
    28,
    26
]

fig, axes = plt.subplots(
    1,
    2,
    sharey=True,
    figsize=(10, 5)
)

axes[0].bar(
    categories,
    values_a
)

axes[0].set_title("Dataset A")

axes[1].bar(
    categories,
    values_b
)

axes[1].set_title("Dataset B")

plt.tight_layout()
plt.show()
=============================================================================
11. SHARED X AND Y AXES
=============================================================================

def shared_both_axes() -> None:
"""Create a grid with both axes shared."""

section("11. SHARED X AND Y AXES")

np.random.seed(42)

fig, axes = plt.subplots(
    2,
    2,
    sharex=True,
    sharey=True,
    figsize=(10, 8)
)

for index, ax in enumerate(
    axes.flat,
    start=1
):
    x = np.random.normal(
        index * 10,
        2,
        100
    )

    y = (
        x
        + np.random.normal(
            0,
            2,
            100
        )
    )

    ax.scatter(
        x,
        y,
        alpha=0.6
    )

    ax.set_title(
        f"Dataset {index}"
    )

fig.supxlabel("Feature X")
fig.supylabel("Feature Y")

plt.tight_layout()
plt.show()
=============================================================================
12. FIGURE SIZE
=============================================================================

def custom_figure_size() -> None:
"""Control the overall figure dimensions."""

section("12. CUSTOM FIGURE SIZE")

x = np.arange(
    1,
    11
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 8)
)

for index, ax in enumerate(
    axes.flat,
    start=1
):
    ax.plot(
        x,
        x * index
    )

    ax.set_title(
        f"Plot {index}"
    )

fig.suptitle(
    "Large Figure with Custom Dimensions",
    fontsize=16
)

plt.tight_layout()
plt.show()
=============================================================================
13. FIGURE TITLE
=============================================================================

def figure_title() -> None:
"""Add a title for the complete figure."""

section("13. FIGURE TITLE")

x = np.linspace(
    0,
    10,
    100
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(14, 4)
)

axes[0].plot(
    x,
    x
)

axes[0].set_title("Linear")

axes[1].plot(
    x,
    x ** 2
)

axes[1].set_title("Quadratic")

axes[2].plot(
    x,
    np.sqrt(x)
)

axes[2].set_title("Square Root")

fig.suptitle(
    "Comparison of Mathematical Functions",
    fontsize=16
)

plt.tight_layout()
plt.show()
=============================================================================
14. SUPTITLE WITH RECTANGULAR GRID
=============================================================================

def professional_dashboard() -> None:
"""Create a simple professional multi-chart dashboard."""

section("14. PROFESSIONAL DASHBOARD")

np.random.seed(42)

months = np.arange(
    1,
    13
)

sales = (
    100
    + months * 12
    + np.random.normal(
        0,
        8,
        12
    )
)

profit = (
    sales * 0.2
    + np.random.normal(
        0,
        3,
        12
    )
)

customers = (
    500
    + months * 35
    + np.random.normal(
        0,
        20,
        12
    )
)

conversion = (
    5
    + months * 0.2
    + np.random.normal(
        0,
        0.2,
        12
    )
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9)
)

axes[0, 0].plot(
    months,
    sales,
    marker="o"
)

axes[0, 0].set_title("Sales")

axes[0, 1].plot(
    months,
    profit,
    marker="o"
)

axes[0, 1].set_title("Profit")

axes[1, 0].plot(
    months,
    customers,
    marker="o"
)

axes[1, 0].set_title("Customers")

axes[1, 1].plot(
    months,
    conversion,
    marker="o"
)

axes[1, 1].set_title("Conversion Rate")

for ax in axes.flat:
    ax.set_xlabel("Month")
    ax.grid(
        True,
        alpha=0.3
    )

fig.suptitle(
    "Business Performance Dashboard",
    fontsize=17
)

plt.tight_layout()
plt.show()
=============================================================================
15. BAR + LINE
=============================================================================

def bar_and_line() -> None:
"""Combine different chart types inside one figure."""

section("15. BAR + LINE")

months = np.arange(
    1,
    7
)

sales = np.array([
    100,
    120,
    135,
    150,
    165,
    190
])

growth = np.array([
    5,
    8,
    7,
    10,
    9,
    12
])

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axes[0].bar(
    months,
    sales
)

axes[0].set_title("Sales by Month")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Sales")

axes[1].plot(
    months,
    growth,
    marker="o"
)

axes[1].set_title("Growth Rate")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Growth")

plt.tight_layout()
plt.show()
=============================================================================
16. BAR + SCATTER + HISTOGRAM
=============================================================================

def mixed_chart_types() -> None:
"""Create a figure containing different chart types."""

section("16. MIXED CHART TYPES")

np.random.seed(42)

categories = [
    "A",
    "B",
    "C",
    "D",
    "E"
]

values = [
    25,
    40,
    30,
    50,
    35
]

x = np.random.normal(
    50,
    10,
    200
)

y = (
    2 * x
    + np.random.normal(
        0,
        15,
        200
    )
)

data = np.random.normal(
    50,
    10,
    500
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

axes[0].bar(
    categories,
    values
)

axes[0].set_title("Bar Chart")

axes[1].scatter(
    x,
    y,
    alpha=0.5
)

axes[1].set_title("Scatter Plot")

axes[2].hist(
    data,
    bins=20,
    edgecolor="black"
)

axes[2].set_title("Histogram")

plt.tight_layout()
plt.show()
=============================================================================
17. HISTOGRAM SUBPLOTS
=============================================================================

def histogram_comparison() -> None:
"""Compare distributions using multiple histograms."""

section("17. HISTOGRAM COMPARISON")

np.random.seed(42)

dataset_a = np.random.normal(
    50,
    10,
    1000
)

dataset_b = np.random.normal(
    60,
    15,
    1000
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axes[0].hist(
    dataset_a,
    bins=30,
    alpha=0.7,
    edgecolor="black"
)

axes[0].set_title("Dataset A")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")

axes[1].hist(
    dataset_b,
    bins=30,
    alpha=0.7,
    edgecolor="black"
)

axes[1].set_title("Dataset B")
axes[1].set_xlabel("Value")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
=============================================================================
18. BOX PLOTS IN SUBPLOTS
=============================================================================

def boxplot_subplots() -> None:
"""Compare multiple distributions using box plots."""

section("18. BOXPLOT SUBPLOTS")

np.random.seed(42)

data_a = np.random.normal(
    50,
    10,
    300
)

data_b = np.random.normal(
    60,
    15,
    300
)

data_c = np.random.normal(
    70,
    8,
    300
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 5)
)

axes[0].boxplot(data_a)
axes[0].set_title("Dataset A")

axes[1].boxplot(data_b)
axes[1].set_title("Dataset B")

axes[2].boxplot(data_c)
axes[2].set_title("Dataset C")

for ax in axes:
    ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
=============================================================================
19. PIE CHART SUBPLOTS
=============================================================================

def pie_subplots() -> None:
"""Compare two categorical distributions."""

section("19. PIE CHART SUBPLOTS")

labels_a = [
    "Python",
    "Java",
    "C++",
    "JavaScript"
]

values_a = [
    40,
    25,
    20,
    15
]

labels_b = [
    "ML",
    "Web",
    "Data",
    "Cloud"
]

values_b = [
    35,
    25,
    25,
    15
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 5)
)

axes[0].pie(
    values_a,
    labels=labels_a,
    autopct="%1.1f%%"
)

axes[0].set_title("Programming Languages")

axes[1].pie(
    values_b,
    labels=labels_b,
    autopct="%1.1f%%"
)

axes[1].set_title("Project Areas")

plt.tight_layout()
plt.show()
=============================================================================
20. ANNOTATIONS
=============================================================================

def annotated_subplots() -> None:
"""Add annotations to individual subplots."""

section("20. ANNOTATED SUBPLOTS")

x = np.arange(
    1,
    11
)

y = np.array([
    10,
    14,
    13,
    18,
    25,
    22,
    30,
    28,
    35,
    40
])

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axes[0].plot(
    x,
    y,
    marker="o"
)

maximum_index = np.argmax(y)

axes[0].annotate(
    "Maximum",
    xy=(
        x[maximum_index],
        y[maximum_index]
    ),
    xytext=(
        x[maximum_index] - 2,
        y[maximum_index] - 8
    ),
    arrowprops={
        "arrowstyle": "->"
    }
)

axes[0].set_title("Annotated Line Plot")

axes[1].scatter(
    x,
    y,
    s=80
)

axes[1].annotate(
    "Important Point",
    xy=(5, 25),
    xytext=(6, 15),
    arrowprops={
        "arrowstyle": "->"
    }
)

axes[1].set_title("Annotated Scatter Plot")

plt.tight_layout()
plt.show()
=============================================================================
21. SUBPLOT SPACING
=============================================================================

def subplot_spacing() -> None:
"""Control spacing between subplot panels."""

section("21. SUBPLOT SPACING")

x = np.linspace(
    0,
    10,
    100
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8)
)

for index, ax in enumerate(
    axes.flat,
    start=1
):
    ax.plot(
        x,
        x * index
    )

    ax.set_title(
        f"Plot {index}"
    )

fig.subplots_adjust(
    hspace=0.4,
    wspace=0.3
)

plt.show()
=============================================================================
22. CONSTRAINED LAYOUT
=============================================================================

def constrained_layout() -> None:
"""
Use constrained_layout for automatic spacing.

This is often convenient for figures containing titles, labels and
colorbars.
"""

section("22. CONSTRAINED LAYOUT")

x = np.linspace(
    0,
    10,
    100
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
    constrained_layout=True
)

axes[0, 0].plot(
    x,
    x
)

axes[0, 0].set_title("Linear")

axes[0, 1].plot(
    x,
    x ** 2
)

axes[0, 1].set_title("Quadratic")

axes[1, 0].plot(
    x,
    np.sin(x)
)

axes[1, 0].set_title("Sine")

axes[1, 1].plot(
    x,
    np.cos(x)
)

axes[1, 1].set_title("Cosine")

fig.suptitle(
    "Constrained Layout Example"
)

plt.show()
=============================================================================
23. DIFFERENT Y LIMITS
=============================================================================

def different_axis_limits() -> None:
"""Demonstrate independent axis limits."""

section("23. DIFFERENT AXIS LIMITS")

x = np.linspace(
    0,
    10,
    100
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axes[0].plot(
    x,
    x ** 2
)

axes[0].set_ylim(
    0,
    120
)

axes[0].set_title("Restricted Y Range")

axes[1].plot(
    x,
    x ** 2
)

axes[1].set_ylim(
    0,
    1200
)

axes[1].set_title("Expanded Y Range")

plt.tight_layout()
plt.show()
=============================================================================
24. GRIDSPEC
=============================================================================

def gridspec_layout() -> None:
"""
Create an asymmetric subplot layout using GridSpec.

GridSpec is useful when all subplot panels do not need identical sizes.
"""

section("24. GRIDSPEC LAYOUT")

x = np.linspace(
    0,
    10,
    200
)

fig = plt.figure(
    figsize=(12, 8)
)

grid = fig.add_gridspec(
    2,
    3
)

ax1 = fig.add_subplot(
    grid[0, :]
)

ax2 = fig.add_subplot(
    grid[1, 0]
)

ax3 = fig.add_subplot(
    grid[1, 1]
)

ax4 = fig.add_subplot(
    grid[1, 2]
)

ax1.plot(
    x,
    np.sin(x)
)

ax1.set_title("Large Overview")

ax2.plot(
    x,
    x
)

ax2.set_title("Linear")

ax3.plot(
    x,
    x ** 2
)

ax3.set_title("Quadratic")

ax4.plot(
    x,
    np.sqrt(x)
)

ax4.set_title("Square Root")

fig.suptitle(
    "GridSpec Asymmetric Layout",
    fontsize=16
)

fig.tight_layout()
plt.show()
=============================================================================
25. SUBPLOT MOSAIC
=============================================================================

def subplot_mosaic() -> None:
"""
Use subplot_mosaic for named subplot layouts.

subplot_mosaic is especially readable for dashboards with asymmetric
layouts.
"""

section("25. SUBPLOT MOSAIC")

x = np.linspace(
    0,
    10,
    100
)

layout = [
    ["main", "main", "right"],
    ["bottom_left", "bottom_middle", "right"]
]

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(12, 8),
    constrained_layout=True
)

axes["main"].plot(
    x,
    np.sin(x)
)

axes["main"].set_title(
    "Main Analysis"
)

axes["right"].plot(
    x,
    np.cos(x)
)

axes["right"].set_title(
    "Secondary Analysis"
)

axes["bottom_left"].plot(
    x,
    x
)

axes["bottom_left"].set_title(
    "Linear"
)

axes["bottom_middle"].plot(
    x,
    x ** 2
)

axes["bottom_middle"].set_title(
    "Quadratic"
)

fig.suptitle(
    "Named Subplot Mosaic"
)

plt.show()
=============================================================================
26. COLORBAR WITH SUBPLOTS
=============================================================================

def colorbar_with_subplots() -> None:
"""Add a shared colorbar to a subplot figure."""

section("26. COLORBAR WITH SUBPLOTS")

np.random.seed(42)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

x = np.random.normal(
    0,
    1,
    500
)

y = np.random.normal(
    0,
    1,
    500
)

values = x ** 2 + y ** 2

image = axes[0].scatter(
    x,
    y,
    c=values,
    alpha=0.7
)

axes[0].set_title(
    "Scatter with Color"
)

axes[1].scatter(
    x,
    y,
    c=values,
    alpha=0.7
)

axes[1].set_title(
    "Same Color Encoding"
)

fig.colorbar(
    image,
    ax=axes,
    label="Magnitude"
)

plt.tight_layout()
plt.show()
=============================================================================
27. MACHINE LEARNING EDA DASHBOARD
=============================================================================

def ml_eda_dashboard() -> None:
"""
Create a Machine Learning exploratory-data-analysis dashboard.

Includes:
    - Target distribution
    - Feature distribution
    - Feature-target relationship
    - Feature correlation
"""

section("27. MACHINE LEARNING EDA DASHBOARD")

np.random.seed(42)

n_samples = 500

age = np.random.normal(
    35,
    10,
    n_samples
)

income = (
    25000
    + age * 1200
    + np.random.normal(
        0,
        12000,
        n_samples
    )
)

target = (
    0.5 * age
    \+ 0.00002 * income
    + np.random.normal(
        0,
        4,
        n_samples
    )
)

data = pd.DataFrame({
    "age": age,
    "income": income,
    "target": target
})

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9)
)

# -------------------------------------------------------------------------
# Target distribution
# -------------------------------------------------------------------------

axes[0, 0].hist(
    data["target"],
    bins=25,
    edgecolor="black"
)

axes[0, 0].set_title(
    "Target Distribution"
)

axes[0, 0].set_xlabel(
    "Target"
)

axes[0, 0].set_ylabel(
    "Frequency"
)

# -------------------------------------------------------------------------
# Feature distribution
# -------------------------------------------------------------------------

axes[0, 1].hist(
    data["age"],
    bins=25,
    edgecolor="black"
)

axes[0, 1].set_title(
    "Age Distribution"
)

axes[0, 1].set_xlabel(
    "Age"
)

axes[0, 1].set_ylabel(
    "Frequency"
)

# -------------------------------------------------------------------------
# Feature-target relationship
# -------------------------------------------------------------------------

axes[1, 0].scatter(
    data["age"],
    data["target"],
    alpha=0.5
)

axes[1, 0].set_title(
    "Age vs Target"
)

axes[1, 0].set_xlabel(
    "Age"
)

axes[1, 0].set_ylabel(
    "Target"
)

# -------------------------------------------------------------------------
# Correlation matrix
# -------------------------------------------------------------------------

correlation = data.corr(
    numeric_only=True
)

image = axes[1, 1].imshow(
    correlation,
    interpolation="nearest"
)

axes[1, 1].set_title(
    "Correlation Matrix"
)

axes[1, 1].set_xticks(
    range(len(correlation.columns))
)

axes[1, 1].set_yticks(
    range(len(correlation.columns))
)

axes[1, 1].set_xticklabels(
    correlation.columns,
    rotation=45,
    ha="right"
)

axes[1, 1].set_yticklabels(
    correlation.columns
)

fig.colorbar(
    image,
    ax=axes[1, 1],
    fraction=0.046,
    pad=0.04
)

fig.suptitle(
    "Machine Learning EDA Dashboard",
    fontsize=17
)

plt.tight_layout()
plt.show()
=============================================================================
28. MODEL TRAINING DASHBOARD
=============================================================================

def model_training_dashboard() -> None:
"""
Visualize model training and validation metrics.

The values are illustrative and are not results from a trained model.
"""

section("28. MODEL TRAINING DASHBOARD")

epochs = np.arange(
    1,
    21
)

training_loss = (
    1.2 * np.exp(
        -epochs / 7
    )
    \+ 0.08
)

validation_loss = (
    1.4 * np.exp(
        -epochs / 6
    )
    \+ 0.15
    \+ 0.005 * np.maximum(
        epochs - 12,
        0
    ) ** 2
)

training_accuracy = (
    0.55
    \+ 0.4 * (
        1 - np.exp(
            -epochs / 6
        )
    )
)

validation_accuracy = (
    0.52
    \+ 0.36 * (
        1 - np.exp(
            -epochs / 7
        )
    )
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

axes[0].plot(
    epochs,
    training_loss,
    marker="o",
    label="Training Loss"
)

axes[0].plot(
    epochs,
    validation_loss,
    marker="o",
    label="Validation Loss"
)

axes[0].set_title(
    "Loss During Training"
)

axes[0].set_xlabel(
    "Epoch"
)

axes[0].set_ylabel(
    "Loss"
)

axes[0].legend()

axes[1].plot(
    epochs,
    training_accuracy,
    marker="o",
    label="Training Accuracy"
)

axes[1].plot(
    epochs,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy"
)

axes[1].set_title(
    "Accuracy During Training"
)

axes[1].set_xlabel(
    "Epoch"
)

axes[1].set_ylabel(
    "Accuracy"
)

axes[1].legend()

fig.suptitle(
    "Model Training Dashboard"
)

plt.tight_layout()
plt.show()
=============================================================================
29. MODEL EVALUATION DASHBOARD
=============================================================================

def model_evaluation_dashboard() -> None:
"""
Create a model-evaluation dashboard.

Metrics are illustrative and should not be interpreted as actual model
performance.
"""

section("29. MODEL EVALUATION DASHBOARD")

models = [
    "Model A",
    "Model B",
    "Model C",
    "Model D"
]

accuracy = [
    0.84,
    0.89,
    0.87,
    0.91
]

precision = [
    0.82,
    0.88,
    0.86,
    0.90
]

recall = [
    0.80,
    0.87,
    0.84,
    0.89
]

f1 = [
    0.81,
    0.875,
    0.85,
    0.895
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9)
)

axes[0, 0].bar(
    models,
    accuracy
)

axes[0, 0].set_title(
    "Accuracy"
)

axes[0, 1].bar(
    models,
    precision
)

axes[0, 1].set_title(
    "Precision"
)

axes[1, 0].bar(
    models,
    recall
)

axes[1, 0].set_title(
    "Recall"
)

axes[1, 1].bar(
    models,
    f1
)

axes[1, 1].set_title(
    "F1 Score"
)

for ax in axes.flat:
    ax.set_ylim(
        0,
        1
    )
    ax.tick_params(
        axis="x",
        rotation=20
    )

fig.suptitle(
    "Illustrative Model Evaluation Dashboard",
    fontsize=17
)

plt.tight_layout()
plt.show()
=============================================================================
30. REGRESSION MODEL DASHBOARD
=============================================================================

def regression_model_dashboard() -> None:
"""
Visualize regression-model diagnostics.

Data and metrics are synthetic for educational purposes.
"""

section("30. REGRESSION MODEL DASHBOARD")

np.random.seed(42)

actual = np.linspace(
    10,
    100,
    200
)

predicted = (
    actual
    + np.random.normal(
        0,
        8,
        200
    )
)

residuals = (
    actual
    - predicted
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

# -------------------------------------------------------------------------
# Actual vs predicted
# -------------------------------------------------------------------------

axes[0].scatter(
    actual,
    predicted,
    alpha=0.5
)

axes[0].plot(
    [actual.min(), actual.max()],
    [actual.min(), actual.max()],
    linestyle="--"
)

axes[0].set_title(
    "Actual vs Predicted"
)

axes[0].set_xlabel(
    "Actual"
)

axes[0].set_ylabel(
    "Predicted"
)

# -------------------------------------------------------------------------
# Residuals
# -------------------------------------------------------------------------

axes[1].scatter(
    predicted,
    residuals,
    alpha=0.5
)

axes[1].axhline(
    0,
    linestyle="--"
)

axes[1].set_title(
    "Residuals vs Predicted"
)

axes[1].set_xlabel(
    "Predicted"
)

axes[1].set_ylabel(
    "Residual"
)

# -------------------------------------------------------------------------
# Residual distribution
# -------------------------------------------------------------------------

axes[2].hist(
    residuals,
    bins=25,
    edgecolor="black"
)

axes[2].set_title(
    "Residual Distribution"
)

axes[2].set_xlabel(
    "Residual"
)

axes[2].set_ylabel(
    "Frequency"
)

fig.suptitle(
    "Regression Model Diagnostic Dashboard"
)

plt.tight_layout()
plt.show()
=============================================================================
31. CLASSIFICATION DASHBOARD
=============================================================================

def classification_dashboard() -> None:
"""
Create an educational classification visualization dashboard.

The confusion matrix values are illustrative.
"""

section("31. CLASSIFICATION DASHBOARD")

np.random.seed(42)

class_0 = np.random.normal(
    loc=[30, 30],
    scale=4,
    size=(150, 2)
)

class_1 = np.random.normal(
    loc=[60, 60],
    scale=4,
    size=(150, 2)
)

confusion_matrix = np.array([
    [85, 10],
    [7, 98]
])

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axes[0].scatter(
    class_0[:, 0],
    class_0[:, 1],
    alpha=0.5,
    label="Class 0"
)

axes[0].scatter(
    class_1[:, 0],
    class_1[:, 1],
    alpha=0.5,
    label="Class 1"
)

axes[0].set_title(
    "Feature Space"
)

axes[0].set_xlabel(
    "Feature 1"
)

axes[0].set_ylabel(
    "Feature 2"
)

axes[0].legend()

image = axes[1].imshow(
    confusion_matrix,
    interpolation="nearest"
)

axes[1].set_title(
    "Illustrative Confusion Matrix"
)

axes[1].set_xlabel(
    "Predicted Class"
)

axes[1].set_ylabel(
    "Actual Class"
)

axes[1].set_xticks([
    0,
    1
])

axes[1].set_yticks([
    0,
    1
])

axes[1].set_xticklabels([
    "Class 0",
    "Class 1"
])

axes[1].set_yticklabels([
    "Class 0",
    "Class 1"
])

for row in range(2):
    for column in range(2):
        axes[1].text(
            column,
            row,
            confusion_matrix[row, column],
            ha="center",
            va="center"
        )

fig.colorbar(
    image,
    ax=axes[1],
    fraction=0.046,
    pad=0.04
)

fig.suptitle(
    "Classification Analysis Dashboard"
)

plt.tight_layout()
plt.show()
=============================================================================
32. FEATURE IMPORTANCE DASHBOARD
=============================================================================

def feature_importance_dashboard() -> None:
"""
Visualize illustrative feature importance values.

These values are examples and are not calculated from a trained model.
"""

section("32. FEATURE IMPORTANCE DASHBOARD")

features = [
    "Income",
    "Age",
    "Experience",
    "Education",
    "Credit Score",
    "Location"
]

importance = np.array([
    0.32,
    0.24,
    0.18,
    0.12,
    0.09,
    0.05
])

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 6)
)

# -------------------------------------------------------------------------
# Vertical
# -------------------------------------------------------------------------

axes[0].bar(
    features,
    importance
)

axes[0].set_title(
    "Feature Importance"
)

axes[0].set_ylabel(
    "Importance"
)

axes[0].tick_params(
    axis="x",
    rotation=45
)

# -------------------------------------------------------------------------
# Horizontal
# -------------------------------------------------------------------------

axes[1].barh(
    features,
    importance
)

axes[1].set_title(
    "Horizontal Feature Importance"
)

axes[1].set_xlabel(
    "Importance"
)

plt.tight_layout()
plt.show()
=============================================================================
33. SAVE SUBPLOT FIGURE
=============================================================================

def save_subplot_figure(
filename: str = "subplots-example.png"
) -> None:
"""Create and save a multi-panel figure."""

section("33. SAVE SUBPLOT FIGURE")

x = np.linspace(
    0,
    10,
    100
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

axes[0, 0].plot(
    x,
    x
)

axes[0, 0].set_title(
    "Linear"
)

axes[0, 1].plot(
    x,
    x ** 2
)

axes[0, 1].set_title(
    "Quadratic"
)

axes[1, 0].plot(
    x,
    np.sin(x)
)

axes[1, 0].set_title(
    "Sine"
)

axes[1, 1].plot(
    x,
    np.cos(x)
)

axes[1, 1].set_title(
    "Cosine"
)

fig.suptitle(
    "Saved Subplot Figure"
)

plt.tight_layout()

fig.savefig(
    filename,
    dpi=300,
    bbox_inches="tight"
)

print(f"Figure saved to: {filename}")

plt.close(fig)
=============================================================================
34. REUSABLE SUBPLOT HELPER
=============================================================================

def create_comparison_figure(
x,
datasets,
titles,
figure_title="Comparison"
):
"""
Create a row of line-plot comparisons.

Parameters
----------
x : array-like
    Shared X-axis values.

datasets : list
    List of Y-axis datasets.

titles : list
    Titles for each subplot.

figure_title : str
    Overall figure title.

Returns
-------
matplotlib.figure.Figure
    Created figure.

numpy.ndarray
    Array of subplot axes.
"""

if len(datasets) != len(titles):
    raise ValueError(
        "datasets and titles must have the same length."
    )

figure_count = len(datasets)

fig, axes = plt.subplots(
    1,
    figure_count,
    figsize=(
        5 * figure_count,
        5
    ),
    squeeze=False
)

axes = axes.ravel()

for ax, data, title in zip(
    axes,
    datasets,
    titles
):
    ax.plot(
        x,
        data
    )

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(
        True,
        alpha=0.3
    )

fig.suptitle(
    figure_title,
    fontsize=16
)

fig.tight_layout()

return fig, axes
=============================================================================
35. REUSABLE HELPER DEMONSTRATION
=============================================================================

def reusable_helper_demo() -> None:
"""Demonstrate the reusable comparison-figure helper."""

section("35. REUSABLE SUBPLOT HELPER")

x = np.linspace(
    0,
    10,
    100
)

datasets = [
    x,
    x ** 2,
    np.sqrt(x)
]

titles = [
    "Linear",
    "Quadratic",
    "Square Root"
]

fig, _ = create_comparison_figure(
    x=x,
    datasets=datasets,
    titles=titles,
    figure_title="Reusable Comparison Figure"
)

plt.show()
=============================================================================
36. PROFESSIONAL SUBPLOT CHECKLIST
=============================================================================

def professional_checklist() -> None:
"""Print a practical subplot checklist."""

section("36. PROFESSIONAL SUBPLOT CHECKLIST")

checklist = [
    "Use plt.subplots() for standard subplot grids.",
    "Use sharex/sharey when axes should use the same scale.",
    "Use figsize to control figure dimensions.",
    "Use fig.suptitle() for an overall figure title.",
    "Use ax.set_title() for individual subplot titles.",
    "Use constrained_layout=True for complex layouts.",
    "Use GridSpec for asymmetric layouts.",
    "Use subplot_mosaic() when named layouts improve readability.",
    "Keep related charts together in one figure.",
    "Avoid putting too many unrelated charts in one figure.",
    "Use consistent axis scales when comparison requires it.",
    "Label axes clearly.",
    "Use legends only when necessary.",
    "Use annotations for important observations.",
    "Use colorbars when color encodes numerical information.",
    "Use high DPI when saving figures for reports.",
    "Close figures when generating many charts programmatically."
]

for index, item in enumerate(
    checklist,
    start=1
):
    print(
        f"{index:02}. {item}"
    )
=============================================================================
37. COMMON MISTAKES
=============================================================================

def common_mistakes() -> None:
"""Print common subplot mistakes."""

section("37. COMMON MISTAKES")

mistakes = [
    "Confusing Figure and Axes objects.",
    "Using plt.subplot() repeatedly when plt.subplots() is clearer.",
    "Forgetting that axes is an array for multi-panel figures.",
    "Incorrectly indexing a 2D axes array.",
    "Using inconsistent scales when comparing plots.",
    "Creating overcrowded dashboards.",
    "Forgetting tight_layout() or constrained_layout.",
    "Overlapping titles and axis labels.",
    "Using too many legends.",
    "Using tiny figure dimensions.",
    "Forgetting to close figures in batch plotting code."
]

for index, mistake in enumerate(
    mistakes,
    start=1
):
    print(
        f"{index:02}. {mistake}"
    )
=============================================================================
38. QUICK REFERENCE
=============================================================================

def quick_reference() -> None:
"""Print a compact subplot reference."""

section("38. QUICK REFERENCE")

print(
    """

Basic:
fig, ax = plt.subplots()

1 row, 2 columns:
fig, axes = plt.subplots(1, 2)

2 rows, 1 column:
fig, axes = plt.subplots(2, 1)

2 × 2:
fig, axes = plt.subplots(2, 2)

Shared X:
fig, axes = plt.subplots(
2,
1,
sharex=True
)

Shared Y:
fig, axes = plt.subplots(
1,
2,
sharey=True
)

Shared X and Y:
fig, axes = plt.subplots(
2,
2,
sharex=True,
sharey=True
)

Figure size:
fig, axes = plt.subplots(
2,
2,
figsize=(12, 8)
)

Overall title:
fig.suptitle("Dashboard")

Individual title:
ax.set_title("Chart")

Automatic spacing:
plt.tight_layout()

Modern automatic spacing:
fig, axes = plt.subplots(
2,
2,
constrained_layout=True
)

Asymmetric layout:
fig.add_gridspec(...)

Named layout:
fig, axes = plt.subplot_mosaic(...)

Save:
fig.savefig(
"figure.png",
dpi=300,
bbox_inches="tight"
)

Close:
plt.close(fig)

Important:
Figure = complete canvas
Axes = individual plotting area
"""
)

=============================================================================
39. MAIN FUNCTION
=============================================================================

def main() -> None:
"""
Run the complete subplot tutorial.

The file intentionally contains many examples. During normal study,
individual functions can be executed instead of running every example.
"""

# -------------------------------------------------------------------------
# Fundamentals
# -------------------------------------------------------------------------

basic_subplot()
two_rows()
two_columns()
grid_2x2()
grid_3x3()
iterate_axes()

# -------------------------------------------------------------------------
# Shared axes and figure configuration
# -------------------------------------------------------------------------

shared_x_axis()
shared_y_axis()
shared_both_axes()
custom_figure_size()
figure_title()

# -------------------------------------------------------------------------
# Dashboards and mixed charts
# -------------------------------------------------------------------------

professional_dashboard()
bar_and_line()
mixed_chart_types()
histogram_comparison()
boxplot_subplots()
pie_subplots()
annotated_subplots()

# -------------------------------------------------------------------------
# Layout control
# -------------------------------------------------------------------------

subplot_spacing()
constrained_layout()
different_axis_limits()
gridspec_layout()
subplot_mosaic()
colorbar_with_subplots()

# -------------------------------------------------------------------------
# Machine Learning visualizations
# -------------------------------------------------------------------------

ml_eda_dashboard()
model_training_dashboard()
model_evaluation_dashboard()
regression_model_dashboard()
classification_dashboard()
feature_importance_dashboard()

# -------------------------------------------------------------------------
# Saving and reusable helpers
# -------------------------------------------------------------------------

save_subplot_figure()
reusable_helper_demo()

# -------------------------------------------------------------------------
# References
# -------------------------------------------------------------------------

professional_checklist()
common_mistakes()
quick_reference()
=============================================================================
40. SCRIPT ENTRY POINT
=============================================================================

if name == "main":
main()

=============================================================================
END OF FILE
=============================================================================
A Figure represents the complete Matplotlib canvas.
An Axes represents an individual plotting area inside a Figure.

The recommended modern pattern is:

fig, axes = plt.subplots(...)
Subplots make it possible to compare related visualizations within a
single figure.
sharex=True and sharey=True are useful when comparing charts with common
axes.
tight_layout() and constrained_layout=True help prevent overlapping labels.
GridSpec provides flexible asymmetric layouts.
subplot_mosaic() provides readable named layouts.

Subplots are especially useful in Machine Learning for:

- Exploratory Data Analysis
- Feature distributions
- Feature-target relationships
- Correlation analysis
- Model training curves
- Regression diagnostics
- Classification analysis
- Feature importance
- Model comparison
A professional ML visualization dashboard should communicate related
information together without overcrowding the figure.
Always distinguish illustrative/example metrics from measurements produced
by an actual trained model.
When generating many figures programmatically, close figures after saving
them to avoid unnecessary memory usage.
"""

"""
LINE PLOTS WITH MATPLOTLIB

File:
03-Python-for-Machine-Learning/03-Matplotlib/line-plot.py

Description:
A complete beginner-to-advanced tutorial on creating and customizing
line plots with Matplotlib.

Line plots are especially useful in Machine Learning for visualizing:

    - Trends over time
    - Training and validation loss
    - Model performance
    - Learning curves
    - Time-series data
    - Cumulative metrics
    - Prediction trends
    - Feature behavior
    - Experiment comparisons
    - Hyperparameter experiments

Requirements:
pip install numpy pandas matplotlib

Run:
python line-plot.py

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
3. BASIC LINE PLOT
=============================================================================

def basic_line_plot() -> None:
"""Create the simplest possible line plot."""

section("3. BASIC LINE PLOT")

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

fig, ax = plt.subplots()

ax.plot(x, y)

ax.set_title("Basic Line Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
4. NUMPY LINE PLOT
=============================================================================

def numpy_line_plot() -> None:
"""Create a line plot using NumPy arrays."""

section("4. NUMPY LINE PLOT")

x = np.arange(1, 11)
y = x ** 2

fig, ax = plt.subplots()

ax.plot(x, y)

ax.set_title("NumPy Line Plot")
ax.set_xlabel("X")
ax.set_ylabel("X²")

plt.tight_layout()
plt.show()
=============================================================================
5. MARKERS
=============================================================================

def line_with_markers() -> None:
"""Add markers to individual observations."""

section("5. LINE WITH MARKERS")

x = np.arange(1, 8)
y = [12, 18, 15, 25, 30, 28, 35]

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="o"
)

ax.set_title("Line Plot with Markers")
ax.set_xlabel("Day")
ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
=============================================================================
6. LINE STYLE
=============================================================================

def line_styles() -> None:
"""Demonstrate common Matplotlib line styles."""

section("6. LINE STYLES")

x = np.arange(1, 8)

fig, ax = plt.subplots()

ax.plot(
    x,
    x,
    linestyle="-",
    label="Solid"
)

ax.plot(
    x,
    x + 2,
    linestyle="--",
    label="Dashed"
)

ax.plot(
    x,
    x + 4,
    linestyle=":",
    label="Dotted"
)

ax.plot(
    x,
    x + 6,
    linestyle="-.",
    label="Dash-dot"
)

ax.set_title("Line Styles")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
7. LINE WIDTH
=============================================================================

def line_width() -> None:
"""Compare different line widths."""

section("7. LINE WIDTH")

x = np.arange(1, 8)

fig, ax = plt.subplots()

ax.plot(
    x,
    x,
    linewidth=1,
    label="Width 1"
)

ax.plot(
    x,
    x + 2,
    linewidth=2,
    label="Width 2"
)

ax.plot(
    x,
    x + 4,
    linewidth=4,
    label="Width 4"
)

ax.set_title("Line Width")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
8. MARKER CUSTOMIZATION
=============================================================================

def marker_customization() -> None:
"""Customize marker size and style."""

section("8. MARKER CUSTOMIZATION")

x = np.arange(1, 8)
y = [10, 15, 13, 20, 24, 22, 30]

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="s",
    markersize=8,
    markeredgewidth=1.5,
    linewidth=2
)

ax.set_title("Customized Markers")
ax.set_xlabel("Observation")
ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
=============================================================================
9. MULTIPLE LINES
=============================================================================

def multiple_lines() -> None:
"""Plot multiple series on the same axes."""

section("9. MULTIPLE LINES")

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun"
]

product_a = [100, 120, 135, 150, 170, 190]
product_b = [80, 95, 110, 125, 140, 160]

fig, ax = plt.subplots()

ax.plot(
    months,
    product_a,
    marker="o",
    label="Product A"
)

ax.plot(
    months,
    product_b,
    marker="s",
    label="Product B"
)

ax.set_title("Multiple Line Plot")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
10. LEGEND CUSTOMIZATION
=============================================================================

def legend_customization() -> None:
"""Demonstrate legend positioning and styling."""

section("10. LEGEND CUSTOMIZATION")

x = np.arange(1, 8)

fig, ax = plt.subplots()

ax.plot(
    x,
    x ** 2,
    marker="o",
    label="Model A"
)

ax.plot(
    x,
    x ** 2 + 10,
    marker="s",
    label="Model B"
)

ax.set_title("Legend Customization")
ax.set_xlabel("Epoch")
ax.set_ylabel("Metric")

ax.legend(
    loc="upper left",
    frameon=True
)

plt.tight_layout()
plt.show()
=============================================================================
11. GRIDLINES
=============================================================================

def gridlines() -> None:
"""Add gridlines to improve readability."""

section("11. GRIDLINES")

x = np.arange(1, 11)
y = x ** 2

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="o"
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.5
)

ax.set_title("Line Plot with Gridlines")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
12. AXIS LIMITS
=============================================================================

def axis_limits() -> None:
"""Control the visible x-axis and y-axis ranges."""

section("12. AXIS LIMITS")

x = np.arange(1, 11)
y = x ** 2

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="o"
)

ax.set_xlim(1, 10)
ax.set_ylim(0, 110)

ax.set_title("Custom Axis Limits")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
13. CUSTOM TICKS
=============================================================================

def custom_ticks() -> None:
"""Customize tick locations and labels."""

x = np.arange(1, 13)
y = [20, 25, 23, 30, 35, 40, 38, 45, 50, 55, 53, 60]

section("13. CUSTOM TICKS")

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="o"
)

ax.set_xticks(x)
ax.set_xticklabels([
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
])

ax.set_title("Custom Tick Labels")
ax.set_xlabel("Month")
ax.set_ylabel("Value")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
=============================================================================
14. ROTATED LABELS
=============================================================================

def rotated_labels() -> None:
"""Rotate categorical x-axis labels."""

section("14. ROTATED LABELS")

categories = [
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "NLP",
    "Generative AI"
]

values = [82, 90, 76, 88, 95]

fig, ax = plt.subplots()

ax.plot(
    categories,
    values,
    marker="o"
)

ax.tick_params(
    axis="x",
    rotation=30
)

ax.set_title("Rotated X-Axis Labels")
ax.set_ylabel("Score")

plt.tight_layout()
plt.show()
=============================================================================
15. ANNOTATIONS
=============================================================================

def annotations() -> None:
"""Annotate important points on a line plot."""

section("15. ANNOTATIONS")

x = np.arange(1, 8)
y = [10, 15, 18, 25, 22, 30, 28]

max_index = np.argmax(y)

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="o"
)

ax.annotate(
    "Maximum",
    xy=(
        x[max_index],
        y[max_index]
    ),
    xytext=(
        x[max_index] - 1,
        y[max_index] + 5
    ),
    arrowprops={
        "arrowstyle": "->"
    }
)

ax.set_title("Annotated Line Plot")
ax.set_xlabel("Observation")
ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
=============================================================================
16. HIGHLIGHT MINIMUM AND MAXIMUM
=============================================================================

def highlight_extremes() -> None:
"""Identify and annotate minimum and maximum observations."""

section("16. HIGHLIGHT EXTREMES")

x = np.arange(1, 11)
y = np.array([
    20,
    25,
    18,
    32,
    40,
    35,
    45,
    42,
    50,
    48
])

min_index = np.argmin(y)
max_index = np.argmax(y)

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="o",
    linewidth=2
)

ax.annotate(
    f"Minimum: {y[min_index]}",
    xy=(
        x[min_index],
        y[min_index]
    ),
    xytext=(
        x[min_index] + 0.5,
        y[min_index] - 8
    ),
    arrowprops={
        "arrowstyle": "->"
    }
)

ax.annotate(
    f"Maximum: {y[max_index]}",
    xy=(
        x[max_index],
        y[max_index]
    ),
    xytext=(
        x[max_index] - 2,
        y[max_index] + 6
    ),
    arrowprops={
        "arrowstyle": "->"
    }
)

ax.set_title("Minimum and Maximum")
ax.set_xlabel("Observation")
ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
=============================================================================
17. FILL BETWEEN CURVES
=============================================================================

def fill_between() -> None:
"""Fill the area between a line and a baseline."""

section("17. FILL BETWEEN")

x = np.arange(1, 11)
y = np.array([
    10,
    15,
    20,
    18,
    25,
    30,
    28,
    35,
    40,
    38
])

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="o"
)

ax.fill_between(
    x,
    y,
    alpha=0.2
)

ax.set_title("Line Plot with Filled Area")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
18. CONFIDENCE INTERVAL
=============================================================================

def confidence_interval() -> None:
"""
Visualize a mean curve with an illustrative uncertainty interval.

The values are generated for demonstration and are not statistical
estimates from an actual experiment.
"""

section("18. CONFIDENCE / UNCERTAINTY BAND")

x = np.arange(1, 11)

mean = np.array([
    50,
    54,
    58,
    61,
    65,
    68,
    72,
    75,
    78,
    82
])

lower = mean - 5
upper = mean + 5

fig, ax = plt.subplots()

ax.plot(
    x,
    mean,
    marker="o",
    label="Mean"
)

ax.fill_between(
    x,
    lower,
    upper,
    alpha=0.2,
    label="Uncertainty Band"
)

ax.set_title("Mean with Uncertainty Band")
ax.set_xlabel("Observation")
ax.set_ylabel("Metric")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
19. TIME SERIES
=============================================================================

def time_series() -> None:
"""Create a simple time-series line plot."""

section("19. TIME SERIES")

dates = pd.date_range(
    "2026-01-01",
    periods=30,
    freq="D"
)

np.random.seed(42)

values = (
    100
    + np.cumsum(
        np.random.normal(
            0,
            2,
            len(dates)
        )
    )
)

fig, ax = plt.subplots()

ax.plot(
    dates,
    values,
    marker="o",
    markersize=3
)

ax.set_title("Time-Series Line Plot")
ax.set_xlabel("Date")
ax.set_ylabel("Value")

fig.autofmt_xdate()

plt.tight_layout()
plt.show()
=============================================================================
20. TIME SERIES WITH MOVING AVERAGE
=============================================================================

def moving_average() -> None:
"""Plot a time series together with its moving average."""

section("20. MOVING AVERAGE")

dates = pd.date_range(
    "2026-01-01",
    periods=60,
    freq="D"
)

np.random.seed(42)

values = (
    100
    + np.cumsum(
        np.random.normal(
            0,
            3,
            len(dates)
        )
    )
)

series = pd.Series(
    values,
    index=dates
)

moving_avg = series.rolling(
    window=7
).mean()

fig, ax = plt.subplots()

ax.plot(
    dates,
    values,
    alpha=0.5,
    label="Original"
)

ax.plot(
    dates,
    moving_avg,
    linewidth=2,
    label="7-Day Moving Average"
)

ax.set_title("Time Series with Moving Average")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()

fig.autofmt_xdate()

plt.tight_layout()
plt.show()
=============================================================================
21. CUMULATIVE SUM
=============================================================================

def cumulative_sum() -> None:
"""Visualize cumulative progress over time."""

section("21. CUMULATIVE SUM")

x = np.arange(1, 11)

daily_values = np.array([
    10,
    15,
    12,
    20,
    18,
    25,
    22,
    30,
    28,
    35
])

cumulative = np.cumsum(
    daily_values
)

fig, ax = plt.subplots()

ax.plot(
    x,
    cumulative,
    marker="o"
)

ax.set_title("Cumulative Progress")
ax.set_xlabel("Day")
ax.set_ylabel("Cumulative Value")

plt.tight_layout()
plt.show()
=============================================================================
22. MACHINE LEARNING TRAINING LOSS
=============================================================================

def training_loss() -> None:
"""
Plot an illustrative training loss curve.

A decreasing training loss often indicates that the optimization process
is reducing the objective on the training data. Actual model behavior
depends on the algorithm, objective, data, and optimization procedure.
"""

section("22. MACHINE LEARNING TRAINING LOSS")

epochs = np.arange(1, 21)

loss = (
    1.5 * np.exp(-epochs / 6)
    + 0.08
)

fig, ax = plt.subplots()

ax.plot(
    epochs,
    loss,
    marker="o"
)

ax.set_title("Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(
    True,
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()
plt.show()
=============================================================================
23. TRAINING VS VALIDATION LOSS
=============================================================================

def training_validation_loss() -> None:
"""
Compare illustrative training and validation loss curves.

These are synthetic educational values, not measured model results.
"""

section("23. TRAINING VS VALIDATION LOSS")

epochs = np.arange(1, 21)

training_loss_values = (
    1.4 * np.exp(-epochs / 6)
    + 0.05
)

validation_loss_values = (
    1.2 * np.exp(-epochs / 7)
    + 0.18
    + 0.005 * epochs
)

fig, ax = plt.subplots()

ax.plot(
    epochs,
    training_loss_values,
    marker="o",
    label="Training Loss"
)

ax.plot(
    epochs,
    validation_loss_values,
    marker="s",
    label="Validation Loss"
)

ax.set_title("Training vs Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

ax.grid(
    True,
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()
plt.show()
=============================================================================
24. TRAINING VS VALIDATION ACCURACY
=============================================================================

def training_validation_accuracy() -> None:
"""Plot illustrative training and validation accuracy."""

section("24. TRAINING VS VALIDATION ACCURACY")

epochs = np.arange(1, 21)

training_accuracy = (
    0.55
    + 0.42 * (
        1 - np.exp(-epochs / 6)
    )
)

validation_accuracy = (
    0.53
    + 0.36 * (
        1 - np.exp(-epochs / 7)
    )
)

fig, ax = plt.subplots()

ax.plot(
    epochs,
    training_accuracy,
    marker="o",
    label="Training Accuracy"
)

ax.plot(
    epochs,
    validation_accuracy,
    marker="s",
    label="Validation Accuracy"
)

ax.set_title("Training vs Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.legend()

ax.grid(
    True,
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()
plt.show()
=============================================================================
25. LEARNING RATE EXPERIMENT
=============================================================================

def learning_rate_experiment() -> None:
"""
Compare illustrative validation loss values for different learning rates.

The values are synthetic and demonstrate visualization technique only.
"""

section("25. LEARNING RATE EXPERIMENT")

epochs = np.arange(1, 16)

learning_rates = {
    "0.001": 1.0 * np.exp(-epochs / 7) + 0.18,
    "0.010": 1.0 * np.exp(-epochs / 5) + 0.10,
    "0.100": 1.0 * np.exp(-epochs / 2.5) + 0.28
}

fig, ax = plt.subplots()

for rate, values in learning_rates.items():
    ax.plot(
        epochs,
        values,
        marker="o",
        label=f"LR = {rate}"
    )

ax.set_title("Illustrative Learning Rate Experiment")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Loss")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
26. MODEL PERFORMANCE OVER TIME
=============================================================================

def model_performance() -> None:
"""Plot an illustrative model metric across experiments."""

section("26. MODEL PERFORMANCE")

experiments = np.arange(1, 11)

accuracy = np.array([
    0.72,
    0.75,
    0.77,
    0.78,
    0.80,
    0.81,
    0.82,
    0.83,
    0.84,
    0.85
])

fig, ax = plt.subplots()

ax.plot(
    experiments,
    accuracy,
    marker="o",
    linewidth=2
)

ax.set_title("Model Accuracy Across Experiments")
ax.set_xlabel("Experiment")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()
=============================================================================
27. REGRESSION ACTUAL VS PREDICTED TREND
=============================================================================

def actual_vs_predicted_trend() -> None:
"""
Compare actual and predicted values in ordered observations.

For arbitrary independent observations, a scatter plot may be more
appropriate. A line plot is useful when observation order itself matters.
"""

section("27. ACTUAL VS PREDICTED")

x = np.arange(1, 11)

actual = np.array([
    100,
    110,
    105,
    120,
    130,
    125,
    140,
    145,
    150,
    160
])

predicted = np.array([
    98,
    112,
    108,
    117,
    127,
    129,
    138,
    143,
    153,
    157
])

fig, ax = plt.subplots()

ax.plot(
    x,
    actual,
    marker="o",
    label="Actual"
)

ax.plot(
    x,
    predicted,
    marker="s",
    linestyle="--",
    label="Predicted"
)

ax.set_title("Actual vs Predicted Values")
ax.set_xlabel("Observation")
ax.set_ylabel("Target")
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
28. RESIDUAL TREND
=============================================================================

def residual_trend() -> None:
"""
Plot residuals across ordered observations.

Residual:
    actual - predicted
"""

section("28. RESIDUAL TREND")

x = np.arange(1, 21)

np.random.seed(42)

residuals = np.random.normal(
    0,
    3,
    len(x)
)

fig, ax = plt.subplots()

ax.plot(
    x,
    residuals,
    marker="o"
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=2
)

ax.set_title("Residual Trend")
ax.set_xlabel("Observation")
ax.set_ylabel("Residual")

plt.tight_layout()
plt.show()
=============================================================================
29. DATAFRAME LINE PLOT
=============================================================================

def pandas_line_plot() -> None:
"""Create line plots using a Pandas DataFrame."""

section("29. PANDAS LINE PLOT")

data = pd.DataFrame({
    "Month": [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun"
    ],
    "Sales": [
        120,
        135,
        128,
        150,
        165,
        180
    ],
    "Profit": [
        30,
        35,
        32,
        42,
        48,
        55
    ]
})

ax = data.plot(
    x="Month",
    y=["Sales", "Profit"],
    marker="o"
)

ax.set_title("Pandas DataFrame Line Plot")
ax.set_xlabel("Month")
ax.set_ylabel("Amount")

plt.tight_layout()
plt.show()
=============================================================================
30. MULTI-SERIES DATAFRAME
=============================================================================

def multi_series_dataframe() -> None:
"""Plot multiple DataFrame columns using Matplotlib."""

section("30. MULTI-SERIES DATAFRAME")

dates = pd.date_range(
    "2026-01-01",
    periods=12,
    freq="MS"
)

df = pd.DataFrame({
    "date": dates,
    "model_a": [
        0.70,
        0.72,
        0.74,
        0.75,
        0.77,
        0.78,
        0.80,
        0.81,
        0.82,
        0.83,
        0.84,
        0.85
    ],
    "model_b": [
        0.68,
        0.71,
        0.73,
        0.76,
        0.78,
        0.79,
        0.81,
        0.82,
        0.83,
        0.84,
        0.85,
        0.86
    ]
})

fig, ax = plt.subplots()

ax.plot(
    df["date"],
    df["model_a"],
    marker="o",
    label="Model A"
)

ax.plot(
    df["date"],
    df["model_b"],
    marker="s",
    label="Model B"
)

ax.set_title("Model Performance Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Accuracy")
ax.legend()

fig.autofmt_xdate()

plt.tight_layout()
plt.show()
=============================================================================
31. SUBPLOTS
=============================================================================

def line_subplots() -> None:
"""Create multiple line plots in a single figure."""

section("31. LINE SUBPLOTS")

x = np.arange(1, 11)

y1 = x
y2 = x ** 2
y3 = np.sqrt(x)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

axes[0].plot(x, y1)
axes[0].set_title("Linear")

axes[1].plot(x, y2)
axes[1].set_title("Quadratic")

axes[2].plot(x, y3)
axes[2].set_title("Square Root")

for ax in axes:
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(
        True,
        linestyle="--",
        alpha=0.3
    )

fig.suptitle("Multiple Line Plot Subplots")

plt.tight_layout()
plt.show()
=============================================================================
32. SECONDARY AXIS
=============================================================================

def secondary_axis() -> None:
"""
Demonstrate a secondary y-axis.

Use secondary axes carefully because different scales can make visual
comparisons misleading if the relationship is not clearly explained.
"""

section("32. SECONDARY Y-AXIS")

months = np.arange(1, 13)

sales = np.array([
    100,
    120,
    130,
    145,
    160,
    175,
    190,
    205,
    220,
    235,
    250,
    270
])

customers = np.array([
    20,
    24,
    26,
    29,
    32,
    35,
    38,
    42,
    45,
    48,
    52,
    57
])

fig, ax1 = plt.subplots()

ax1.plot(
    months,
    sales,
    marker="o",
    label="Sales"
)

ax1.set_xlabel("Month")
ax1.set_ylabel("Sales")

ax2 = ax1.twinx()

ax2.plot(
    months,
    customers,
    marker="s",
    linestyle="--",
    label="Customers"
)

ax2.set_ylabel("Customers")

ax1.set_title("Sales and Customers")

fig.tight_layout()
plt.show()
=============================================================================
33. STEP PLOT
=============================================================================

def step_plot() -> None:
"""Create a step-style line plot."""

section("33. STEP PLOT")

x = np.arange(1, 9)
y = [10, 10, 20, 20, 30, 30, 40, 40]

fig, ax = plt.subplots()

ax.step(
    x,
    y,
    where="mid"
)

ax.set_title("Step Plot")
ax.set_xlabel("X")
ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
=============================================================================
34. ERROR BARS
=============================================================================

def error_bars() -> None:
"""Add uncertainty/error bars to observations."""

section("34. ERROR BARS")

x = np.arange(1, 7)

values = np.array([
    70,
    73,
    76,
    78,
    81,
    84
])

errors = np.array([
    3,
    2,
    4,
    3,
    2,
    3
])

fig, ax = plt.subplots()

ax.errorbar(
    x,
    values,
    yerr=errors,
    marker="o",
    capsize=5,
    linestyle="-"
)

ax.set_title("Line Plot with Error Bars")
ax.set_xlabel("Experiment")
ax.set_ylabel("Score")

plt.tight_layout()
plt.show()
=============================================================================
35. LOGARITHMIC Y-AXIS
=============================================================================

def logarithmic_axis() -> None:
"""Use a logarithmic y-axis for rapidly changing values."""

section("35. LOGARITHMIC Y-AXIS")

x = np.arange(1, 11)

y = 10 ** (x / 2)

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="o"
)

ax.set_yscale("log")

ax.set_title("Line Plot with Logarithmic Y-Axis")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.tight_layout()
plt.show()
=============================================================================
36. SAVE LINE PLOT
=============================================================================

def save_line_plot() -> None:
"""Save a line plot as a high-resolution image."""

section("36. SAVE LINE PLOT")

x = np.arange(1, 11)
y = x ** 2

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker="o"
)

ax.set_title("Saved Line Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")

fig.tight_layout()

output_path = "line-plot.png"

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Line plot saved to: {output_path}")

plt.close(fig)
=============================================================================
37. REUSABLE LINE PLOT FUNCTION
=============================================================================

def plot_line(
x,
y,
title="Line Plot",
xlabel="X",
ylabel="Y",
label=None,
marker=None,
linestyle="-",
linewidth=2,
show_grid=True
):
"""
Create a reusable line plot.

Parameters
----------
x : array-like
    X-axis values.

y : array-like
    Y-axis values.

title : str
    Plot title.

xlabel : str
    X-axis label.

ylabel : str
    Y-axis label.

label : str or None
    Optional legend label.

marker : str or None
    Optional marker style.

linestyle : str
    Line style.

linewidth : float
    Width of the line.

show_grid : bool
    Whether to show gridlines.

Returns
-------
matplotlib.axes.Axes
    Created Axes object.
"""

fig, ax = plt.subplots()

ax.plot(
    x,
    y,
    marker=marker,
    linestyle=linestyle,
    linewidth=linewidth,
    label=label
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

if label is not None:
    ax.legend()

fig.tight_layout()

return ax
=============================================================================
38. REUSABLE FUNCTION DEMONSTRATION
=============================================================================

def reusable_function_demo() -> None:
"""Demonstrate the reusable line plotting function."""

section("38. REUSABLE LINE PLOT FUNCTION")

x = np.arange(1, 11)

y = np.array([
    10,
    15,
    14,
    20,
    25,
    23,
    30,
    35,
    33,
    40
])

plot_line(
    x,
    y,
    title="Reusable Line Plot",
    xlabel="Observation",
    ylabel="Value",
    label="Series",
    marker="o"
)

plt.show()
=============================================================================
39. COMPLETE MACHINE LEARNING VISUALIZATION
=============================================================================

def complete_ml_line_plot() -> None:
"""
Complete Machine Learning visualization example.

Demonstrates:
    - Training loss
    - Validation loss
    - Training accuracy
    - Validation accuracy

The values are synthetic educational examples.
"""

section("39. COMPLETE MACHINE LEARNING LINE PLOT")

epochs = np.arange(1, 21)

training_loss_values = (
    1.5 * np.exp(-epochs / 6)
    + 0.05
)

validation_loss_values = (
    1.35 * np.exp(-epochs / 7)
    + 0.15
    + 0.003 * epochs
)

training_accuracy_values = (
    0.50
    + 0.45 * (
        1 - np.exp(-epochs / 6)
    )
)

validation_accuracy_values = (
    0.50
    + 0.38 * (
        1 - np.exp(-epochs / 7)
    )
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(15, 5)
)

# -------------------------------------------------------------------------
# Loss
# -------------------------------------------------------------------------

axes[0].plot(
    epochs,
    training_loss_values,
    marker="o",
    label="Training Loss"
)

axes[0].plot(
    epochs,
    validation_loss_values,
    marker="s",
    label="Validation Loss"
)

axes[0].set_title("Loss Curves")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(
    True,
    linestyle="--",
    alpha=0.3
)

# -------------------------------------------------------------------------
# Accuracy
# -------------------------------------------------------------------------

axes[1].plot(
    epochs,
    training_accuracy_values,
    marker="o",
    label="Training Accuracy"
)

axes[1].plot(
    epochs,
    validation_accuracy_values,
    marker="s",
    label="Validation Accuracy"
)

axes[1].set_title("Accuracy Curves")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(
    True,
    linestyle="--",
    alpha=0.3
)

fig.suptitle(
    "Machine Learning Training Curves"
)

plt.tight_layout()
plt.show()
=============================================================================
40. MODEL COMPARISON
=============================================================================

def model_comparison() -> None:
"""
Compare illustrative model performance across experiments.

These values are synthetic and demonstrate visualization technique only.
"""

section("40. MODEL COMPARISON")

experiments = np.arange(1, 11)

model_a = np.array([
    0.70,
    0.72,
    0.74,
    0.76,
    0.77,
    0.79,
    0.80,
    0.81,
    0.82,
    0.83
])

model_b = np.array([
    0.68,
    0.71,
    0.73,
    0.75,
    0.78,
    0.80,
    0.81,
    0.83,
    0.84,
    0.85
])

model_c = np.array([
    0.65,
    0.69,
    0.72,
    0.74,
    0.75,
    0.77,
    0.78,
    0.80,
    0.81,
    0.82
])

fig, ax = plt.subplots()

ax.plot(
    experiments,
    model_a,
    marker="o",
    label="Model A"
)

ax.plot(
    experiments,
    model_b,
    marker="s",
    label="Model B"
)

ax.plot(
    experiments,
    model_c,
    marker="^",
    label="Model C"
)

ax.set_title("Illustrative Model Comparison")
ax.set_xlabel("Experiment")
ax.set_ylabel("Validation Accuracy")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
41. PROFESSIONAL LINE PLOT CHECKLIST
=============================================================================

def professional_checklist() -> None:
"""Print a professional visualization checklist."""

section("41. PROFESSIONAL LINE PLOT CHECKLIST")

checklist = [
    "Use line plots for ordered or continuous x-axis data.",
    "Use meaningful axis labels.",
    "Add a descriptive title.",
    "Use markers when individual observations matter.",
    "Use legends for multiple series.",
    "Avoid excessive lines in one chart.",
    "Use consistent x-axis ordering.",
    "Use gridlines only when they improve readability.",
    "Highlight important observations with annotations.",
    "Use uncertainty bands when uncertainty is meaningful.",
    "Use the same scale when comparing compatible metrics.",
    "Use logarithmic axes only when appropriate.",
    "Avoid misleading axis limits.",
    "Save important figures at sufficient resolution.",
    "For ML curves, clearly distinguish training and validation data."
]

for index, item in enumerate(
    checklist,
    start=1
):
    print(f"{index:02}. {item}")
=============================================================================
42. COMMON MISTAKES
=============================================================================

def common_mistakes() -> None:
"""Print common mistakes when creating line plots."""

section("42. COMMON MISTAKES")

mistakes = [
    "Using a line plot for unordered categorical data.",
    "Connecting observations when their order has no meaning.",
    "Using too many series in one chart.",
    "Forgetting axis labels.",
    "Missing a legend when multiple series are present.",
    "Using misleading axis limits.",
    "Overusing markers and annotations.",
    "Using dual axes without clearly explaining the scales.",
    "Comparing metrics with incompatible units.",
    "Interpreting synthetic demonstration values as actual model results."
]

for index, mistake in enumerate(
    mistakes,
    start=1
):
    print(f"{index:02}. {mistake}")
=============================================================================
43. QUICK REFERENCE
=============================================================================

def quick_reference() -> None:
"""Print a compact Matplotlib line-plot reference."""

section("43. QUICK REFERENCE")

print(
    """

Basic:
ax.plot(x, y)

Markers:
ax.plot(x, y, marker="o")

Line style:
ax.plot(x, y, linestyle="--")

Line width:
ax.plot(x, y, linewidth=2)

Multiple lines:
ax.plot(x, y1, label="Series A")
ax.plot(x, y2, label="Series B")
ax.legend()

Grid:
ax.grid(True)

Axis limits:
ax.set_xlim(...)
ax.set_ylim(...)

Annotation:
ax.annotate(...)

Fill:
ax.fill_between(x, lower, upper)

Horizontal reference:
ax.axhline(y=0)

Vertical reference:
ax.axvline(x=0)

Log scale:
ax.set_yscale("log")

Save:
fig.savefig("line-plot.png", dpi=300)

Pandas:
df.plot(x="date", y="value")

Moving average:
df["value"].rolling(7).mean()
"""
)

=============================================================================
44. MAIN FUNCTION
=============================================================================

def main() -> None:
"""
Run the complete line-plot tutorial.

The tutorial contains many visual examples. During normal study,
individual functions can be called instead of running every example.
"""

# -------------------------------------------------------------------------
# Fundamentals
# -------------------------------------------------------------------------

basic_line_plot()
numpy_line_plot()
line_with_markers()
line_styles()
line_width()
marker_customization()

# -------------------------------------------------------------------------
# Multiple series and styling
# -------------------------------------------------------------------------

multiple_lines()
legend_customization()
gridlines()
axis_limits()
custom_ticks()
rotated_labels()
annotations()
highlight_extremes()
fill_between()
confidence_interval()

# -------------------------------------------------------------------------
# Time-series concepts
# -------------------------------------------------------------------------

time_series()
moving_average()
cumulative_sum()

# -------------------------------------------------------------------------
# Machine Learning visualization
# -------------------------------------------------------------------------

training_loss()
training_validation_loss()
training_validation_accuracy()
learning_rate_experiment()
model_performance()
actual_vs_predicted_trend()
residual_trend()

# -------------------------------------------------------------------------
# Pandas integration
# -------------------------------------------------------------------------

pandas_line_plot()
multi_series_dataframe()

# -------------------------------------------------------------------------
# Advanced Matplotlib
# -------------------------------------------------------------------------

line_subplots()
secondary_axis()
step_plot()
error_bars()
logarithmic_axis()

# -------------------------------------------------------------------------
# Saving and reusable functions
# -------------------------------------------------------------------------

save_line_plot()
reusable_function_demo()

# -------------------------------------------------------------------------
# Complete ML example
# -------------------------------------------------------------------------

complete_ml_line_plot()
model_comparison()

# -------------------------------------------------------------------------
# Learning references
# -------------------------------------------------------------------------

professional_checklist()
common_mistakes()
quick_reference()
=============================================================================
45. SCRIPT ENTRY POINT
=============================================================================

if name == "main":
main()

=============================================================================
END OF FILE
=============================================================================
A line plot connects ordered observations.
Line plots are particularly useful for:
- Time series
- Trends
- Training curves
- Validation curves
- Experiment progression
- Cumulative metrics
Markers make individual observations easier to inspect.
Multiple lines allow related series to be compared.
Moving averages can help visualize trends in noisy time-series data.
fill_between() is useful for displaying uncertainty intervals.
axhline() and axvline() are useful for reference thresholds.
In Machine Learning, line plots are commonly used for:
- Training loss
- Validation loss
- Accuracy
- Learning-rate experiments
- Model performance
- Residual trends
Training and validation curves should be interpreted together rather than
from a single metric in isolation.
Always distinguish synthetic educational examples from actual model
measurements.
Choose the chart type according to the structure of the data:
Ordered/time-based data -> line plot
Categories -> bar chart
Distribution -> histogram
Numeric relationship -> scatter plot
"""

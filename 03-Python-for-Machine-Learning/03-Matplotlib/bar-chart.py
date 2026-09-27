Path:
03-Python-for-Machine-Learning/03-Matplotlib/bar-chart.py

This module demonstrates how to create and customize bar charts
using Matplotlib.

Topics covered:
1. Basic bar chart
2. Vertical bar chart
3. Horizontal bar chart
4. Custom labels
5. Colors
6. Bar width
7. Edge styling
8. Value labels
9. Multiple datasets
10. Grouped bar charts
11. Stacked bar charts
12. Percentage stacked bars
13. Sorting bars
14. Horizontal ranking charts
15. Error bars
16. Negative values
17. Custom ticks
18. Grid customization
19. Annotations
20. Subplots
21. Pandas integration
22. ML class distribution
23. Feature importance
24. Model comparison
25. Training vs validation metrics
26. Reusable plotting functions
27. Saving figures
28. Professional visualization practices

Requirements:
pip install matplotlib pandas numpy

Run:
python bar-chart.py
"""

=============================================================================
1. IMPORTS
=============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

=============================================================================
2. DISPLAY CONFIGURATION
=============================================================================

plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

def section(title: str) -> None:
"""Print a formatted section heading."""

print("\n" + "=" * 80)
print(title)
print("=" * 80)
=============================================================================
3. BASIC BAR CHART
=============================================================================

def basic_bar_chart() -> None:
"""
Create a simple vertical bar chart.

Bar charts are useful when comparing discrete categories.
"""

section("3. BASIC BAR CHART")

categories = ["Python", "Java", "C++", "JavaScript"]
values = [90, 75, 60, 70]

fig, ax = plt.subplots()

ax.bar(categories, values)

ax.set_title("Programming Language Popularity")
ax.set_xlabel("Language")
ax.set_ylabel("Popularity")

plt.tight_layout()
plt.show()
=============================================================================
4. BAR CHART FROM NUMERICAL POSITIONS
=============================================================================

def numerical_position_bar_chart() -> None:
"""Create bars using numerical x positions."""

section("4. NUMERICAL BAR POSITIONS")

positions = np.arange(4)
values = [25, 40, 30, 50]

fig, ax = plt.subplots()

ax.bar(positions, values)

ax.set_xticks(positions)
ax.set_xticklabels(
    ["A", "B", "C", "D"]
)

ax.set_title("Category Values")
ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
=============================================================================
5. HORIZONTAL BAR CHART
=============================================================================

def horizontal_bar_chart() -> None:
"""
Create a horizontal bar chart.

Horizontal bars are particularly useful when category names
are long.
"""

section("5. HORIZONTAL BAR CHART")

categories = [
    "Machine Learning",
    "Data Science",
    "Artificial Intelligence",
    "Cloud Computing",
    "Cyber Security",
]

values = [95, 88, 92, 80, 75]

fig, ax = plt.subplots()

ax.barh(categories, values)

ax.set_title("Technology Interest")
ax.set_xlabel("Interest Score")

plt.tight_layout()
plt.show()
=============================================================================
6. CUSTOM BAR WIDTH
=============================================================================

def custom_bar_width() -> None:
"""Control the width of bars."""

section("6. CUSTOM BAR WIDTH")

categories = ["A", "B", "C", "D"]
values = [20, 35, 30, 45]

fig, ax = plt.subplots()

ax.bar(
    categories,
    values,
    width=0.5,
)

ax.set_title("Custom Bar Width")

plt.tight_layout()
plt.show()
=============================================================================
7. BAR COLORS
=============================================================================

def custom_bar_colors() -> None:
"""Assign individual colors to bars."""

section("7. CUSTOM BAR COLORS")

categories = ["Python", "Java", "C++", "JavaScript"]
values = [90, 75, 60, 70]

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
]

fig, ax = plt.subplots()

ax.bar(
    categories,
    values,
    color=colors,
)

ax.set_title("Language Comparison")

plt.tight_layout()
plt.show()
=============================================================================
8. EDGE STYLING
=============================================================================

def edge_styling() -> None:
"""Customize bar borders."""

section("8. BAR EDGE STYLING")

categories = ["A", "B", "C", "D"]
values = [30, 45, 25, 55]

fig, ax = plt.subplots()

ax.bar(
    categories,
    values,
    edgecolor="black",
    linewidth=1.2,
)

ax.set_title("Bar Edge Styling")

plt.tight_layout()
plt.show()
=============================================================================
9. TRANSPARENCY
=============================================================================

def transparency() -> None:
"""Use alpha to control transparency."""

section("9. BAR TRANSPARENCY")

categories = ["A", "B", "C", "D"]
values = [25, 40, 35, 50]

fig, ax = plt.subplots()

ax.bar(
    categories,
    values,
    alpha=0.7,
)

ax.set_title("Transparent Bars")

plt.tight_layout()
plt.show()
=============================================================================
10. ADD VALUE LABELS
=============================================================================

def add_value_labels() -> None:
"""
Display numerical values above bars.

This is useful when exact values matter.
"""

section("10. VALUE LABELS")

categories = ["A", "B", "C", "D"]
values = [25, 40, 35, 50]

fig, ax = plt.subplots()

bars = ax.bar(
    categories,
    values,
)

ax.bar_label(
    bars,
    padding=3,
)

ax.set_title("Values on Bars")
ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
=============================================================================
11. BAR LABEL FORMAT
=============================================================================

def formatted_bar_labels() -> None:
"""Format bar labels using a custom formatter."""

section("11. FORMATTED BAR LABELS")

products = ["Product A", "Product B", "Product C"]
revenue = [125000, 185000, 95000]

fig, ax = plt.subplots()

bars = ax.bar(
    products,
    revenue,
)

labels = [
    f"₹{value / 1000:.0f}K"
    for value in revenue
]

ax.bar_label(
    bars,
    labels=labels,
    padding=3,
)

ax.set_title("Product Revenue")
ax.set_ylabel("Revenue")

plt.tight_layout()
plt.show()
=============================================================================
12. GROUPED BAR CHART
=============================================================================

def grouped_bar_chart() -> None:
"""
Compare two or more values for every category.

Grouped bars are useful when categories contain multiple
comparable measurements.
"""

section("12. GROUPED BAR CHART")

subjects = [
    "Python",
    "Statistics",
    "ML",
    "Deep Learning",
]

student_a = [85, 78, 90, 82]
student_b = [90, 85, 88, 91]

x = np.arange(len(subjects))
width = 0.35

fig, ax = plt.subplots()

bars_a = ax.bar(
    x - width / 2,
    student_a,
    width,
    label="Student A",
)

bars_b = ax.bar(
    x + width / 2,
    student_b,
    width,
    label="Student B",
)

ax.set_xticks(x)
ax.set_xticklabels(subjects)

ax.set_ylabel("Score")
ax.set_title("Student Performance Comparison")
ax.legend()

ax.bar_label(
    bars_a,
    padding=2,
)

ax.bar_label(
    bars_b,
    padding=2,
)

plt.tight_layout()
plt.show()
=============================================================================
13. THREE-GROUP BAR CHART
=============================================================================

def three_group_bar_chart() -> None:
"""Compare three datasets using grouped bars."""

section("13. THREE-GROUP BAR CHART")

categories = ["Q1", "Q2", "Q3", "Q4"]

sales_a = [100, 120, 140, 160]
sales_b = [90, 130, 150, 170]
sales_c = [80, 110, 145, 180]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots()

ax.bar(
    x - width,
    sales_a,
    width,
    label="Region A",
)

ax.bar(
    x,
    sales_b,
    width,
    label="Region B",
)

ax.bar(
    x + width,
    sales_c,
    width,
    label="Region C",
)

ax.set_xticks(x)
ax.set_xticklabels(categories)

ax.set_xlabel("Quarter")
ax.set_ylabel("Sales")
ax.set_title("Regional Sales Comparison")

ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
14. STACKED BAR CHART
=============================================================================

def stacked_bar_chart() -> None:
"""
Stacked bars show how multiple components contribute to
a total category value.
"""

section("14. STACKED BAR CHART")

months = ["Jan", "Feb", "Mar", "Apr"]

product_a = np.array([30, 35, 40, 45])
product_b = np.array([20, 25, 30, 35])
product_c = np.array([10, 15, 20, 25])

fig, ax = plt.subplots()

ax.bar(
    months,
    product_a,
    label="Product A",
)

ax.bar(
    months,
    product_b,
    bottom=product_a,
    label="Product B",
)

ax.bar(
    months,
    product_c,
    bottom=product_a + product_b,
    label="Product C",
)

ax.set_title("Stacked Product Sales")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")

ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
15. PERCENTAGE STACKED BAR
=============================================================================

def percentage_stacked_bar() -> None:
"""
Normalize each category to 100%.

Useful for comparing composition rather than absolute totals.
"""

section("15. PERCENTAGE STACKED BAR")

categories = ["Class A", "Class B", "Class C"]

passed = np.array([80, 60, 90])
failed = np.array([20, 40, 10])

totals = passed + failed

passed_pct = passed / totals * 100
failed_pct = failed / totals * 100

fig, ax = plt.subplots()

ax.bar(
    categories,
    passed_pct,
    label="Passed",
)

ax.bar(
    categories,
    failed_pct,
    bottom=passed_pct,
    label="Failed",
)

ax.set_ylim(0, 100)

ax.set_ylabel("Percentage")
ax.set_title("Pass/Fail Composition")

ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
16. SORTED BAR CHART
=============================================================================

def sorted_bar_chart() -> None:
"""Sort categories by value before plotting."""

section("16. SORTED BAR CHART")

data = pd.Series(
    {
        "Python": 95,
        "Machine Learning": 88,
        "Statistics": 72,
        "SQL": 80,
        "Deep Learning": 91,
    }
)

sorted_data = data.sort_values(
    ascending=False
)

fig, ax = plt.subplots()

bars = ax.bar(
    sorted_data.index,
    sorted_data.values,
)

ax.bar_label(
    bars,
    padding=3,
)

ax.set_title("Skills Ranked by Score")
ax.set_ylabel("Score")

plt.xticks(rotation=20)

plt.tight_layout()
plt.show()
=============================================================================
17. HORIZONTAL RANKING
=============================================================================

def horizontal_ranking() -> None:
"""
Horizontal bar charts are excellent for rankings and
long category names.
"""

section("17. HORIZONTAL RANKING")

skills = pd.Series(
    {
        "Machine Learning": 95,
        "Python": 92,
        "Data Analysis": 88,
        "Statistics": 82,
        "SQL": 78,
        "Visualization": 75,
    }
)

skills = skills.sort_values()

fig, ax = plt.subplots()

bars = ax.barh(
    skills.index,
    skills.values,
)

ax.bar_label(
    bars,
    padding=3,
)

ax.set_xlabel("Score")
ax.set_title("Technical Skills Ranking")

plt.tight_layout()
plt.show()
=============================================================================
18. NEGATIVE VALUES
=============================================================================

def negative_values() -> None:
"""Display positive and negative values."""

section("18. NEGATIVE VALUES")

categories = ["A", "B", "C", "D"]

values = [30, -20, 40, -15]

fig, ax = plt.subplots()

ax.bar(
    categories,
    values,
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_title("Positive and Negative Values")
ax.set_ylabel("Change")

plt.tight_layout()
plt.show()
=============================================================================
19. ERROR BARS
=============================================================================

def error_bars() -> None:
"""
Error bars communicate uncertainty or variability.

The exact interpretation depends on how the error values
were calculated.
"""

section("19. ERROR BARS")

models = [
    "Model A",
    "Model B",
    "Model C",
    "Model D",
]

scores = [0.82, 0.87, 0.91, 0.89]
errors = [0.03, 0.02, 0.015, 0.025]

fig, ax = plt.subplots()

ax.bar(
    models,
    scores,
    yerr=errors,
    capsize=5,
)

ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy with Error Bars")

plt.tight_layout()
plt.show()
=============================================================================
20. CUSTOM TICKS
=============================================================================

def custom_ticks() -> None:
"""Customize tick positions and labels."""

section("20. CUSTOM TICKS")

categories = ["A", "B", "C", "D"]
values = [1000, 2500, 5000, 7500]

fig, ax = plt.subplots()

ax.bar(
    categories,
    values,
)

ax.set_yticks(
    [0, 2500, 5000, 7500]
)

ax.set_yticklabels(
    ["0", "2.5K", "5K", "7.5K"]
)

ax.set_title("Custom Axis Formatting")

plt.tight_layout()
plt.show()
=============================================================================
21. GRIDLINES
=============================================================================

def gridlines() -> None:
"""Add horizontal gridlines for easier value comparison."""

section("21. GRIDLINES")

categories = ["A", "B", "C", "D"]
values = [30, 50, 40, 70]

fig, ax = plt.subplots()

ax.bar(
    categories,
    values,
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.5,
)

ax.set_axisbelow(True)

ax.set_title("Bar Chart with Gridlines")

plt.tight_layout()
plt.show()
=============================================================================
22. ANNOTATIONS
=============================================================================

def annotations() -> None:
"""Highlight the maximum bar with an annotation."""

section("22. ANNOTATIONS")

categories = ["A", "B", "C", "D", "E"]
values = [30, 45, 80, 55, 40]

fig, ax = plt.subplots()

bars = ax.bar(
    categories,
    values,
)

max_index = int(np.argmax(values))
max_value = values[max_index]

ax.annotate(
    "Highest",
    xy=(
        max_index,
        max_value,
    ),
    xytext=(
        max_index,
        max_value + 10,
    ),
    ha="center",
    arrowprops={
        "arrowstyle": "->",
    },
)

ax.bar_label(
    bars,
    padding=3,
)

ax.set_title("Annotated Bar Chart")

plt.tight_layout()
plt.show()
=============================================================================
23. ROTATING LONG LABELS
=============================================================================

def rotate_labels() -> None:
"""Rotate long category labels for readability."""

section("23. ROTATING LABELS")

categories = [
    "Machine Learning",
    "Deep Learning",
    "Natural Language Processing",
    "Computer Vision",
    "Reinforcement Learning",
]

values = [90, 85, 80, 75, 65]

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.bar(
    categories,
    values,
)

ax.tick_params(
    axis="x",
    rotation=25,
)

ax.set_title("ML Specialization Scores")

plt.tight_layout()
plt.show()
=============================================================================
24. BAR CHART WITH SUBPLOTS
=============================================================================

def bar_subplots() -> None:
"""Create multiple bar charts in one figure."""

section("24. BAR CHART SUBPLOTS")

categories = ["A", "B", "C", "D"]

values_1 = [20, 35, 30, 45]
values_2 = [25, 30, 40, 35]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5),
)

axes[0].bar(
    categories,
    values_1,
)

axes[0].set_title("Dataset A")

axes[1].bar(
    categories,
    values_2,
)

axes[1].set_title("Dataset B")

fig.suptitle(
    "Category Comparison"
)

plt.tight_layout()
plt.show()
=============================================================================
25. PANDAS SERIES BAR CHART
=============================================================================

def pandas_series_bar_chart() -> None:
"""Plot a Pandas Series using Matplotlib."""

section("25. PANDAS SERIES BAR CHART")

scores = pd.Series(
    {
        "Python": 95,
        "Pandas": 90,
        "NumPy": 88,
        "Matplotlib": 85,
    }
)

fig, ax = plt.subplots()

ax.bar(
    scores.index,
    scores.values,
)

ax.set_title("Python Ecosystem Skills")
ax.set_ylabel("Score")

plt.tight_layout()
plt.show()
=============================================================================
26. PANDAS DATAFRAME BAR CHART
=============================================================================

def pandas_dataframe_bar_chart() -> None:
"""Create grouped bars from a DataFrame."""

section("26. PANDAS DATAFRAME BAR CHART")

data = pd.DataFrame(
    {
        "Training": [0.82, 0.87, 0.91],
        "Validation": [0.78, 0.84, 0.88],
    },
    index=[
        "Model A",
        "Model B",
        "Model C",
    ],
)

fig, ax = plt.subplots()

data.plot(
    kind="bar",
    ax=ax,
)

ax.set_title("Model Performance")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
=============================================================================
27. ML CLASS DISTRIBUTION
=============================================================================

def ml_class_distribution() -> None:
"""
Visualize the distribution of target classes.

This is useful during classification dataset exploration.
"""

section("27. ML CLASS DISTRIBUTION")

target = pd.Series(
    [
        "Class 0",
        "Class 0",
        "Class 1",
        "Class 0",
        "Class 2",
        "Class 1",
        "Class 0",
        "Class 2",
        "Class 1",
        "Class 0",
    ],
    name="target",
)

counts = target.value_counts().sort_index()

fig, ax = plt.subplots()

bars = ax.bar(
    counts.index,
    counts.values,
)

ax.bar_label(
    bars,
    padding=3,
)

ax.set_title("Target Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Count")

plt.tight_layout()
plt.show()
=============================================================================
28. CLASS IMBALANCE VISUALIZATION
=============================================================================

def class_imbalance_visualization() -> None:
"""
Visualize an imbalanced classification target.

A class imbalance plot is descriptive. Whether imbalance
requires special handling depends on the modeling task,
evaluation metric, and dataset.
"""

section("28. CLASS IMBALANCE")

classes = ["Negative", "Positive"]
counts = [950, 50]

fig, ax = plt.subplots()

bars = ax.bar(
    classes,
    counts,
)

ax.bar_label(
    bars,
    padding=3,
)

ax.set_title("Class Distribution")
ax.set_ylabel("Samples")

plt.tight_layout()
plt.show()
=============================================================================
29. FEATURE IMPORTANCE
=============================================================================

def feature_importance() -> None:
"""
Visualize example model feature importance values.

The values below are illustrative. In a real project, use
importance values produced by the fitted model.
"""

section("29. FEATURE IMPORTANCE")

features = [
    "Age",
    "Income",
    "Experience",
    "Education",
    "Credit Score",
]

importance = [
    0.12,
    0.35,
    0.20,
    0.10,
    0.23,
]

order = np.argsort(importance)

sorted_features = [
    features[index]
    for index in order
]

sorted_importance = [
    importance[index]
    for index in order
]

fig, ax = plt.subplots()

bars = ax.barh(
    sorted_features,
    sorted_importance,
)

ax.bar_label(
    bars,
    fmt="%.2f",
    padding=3,
)

ax.set_xlabel("Importance")
ax.set_title("Feature Importance")

plt.tight_layout()
plt.show()
=============================================================================
30. MODEL COMPARISON
=============================================================================

def model_comparison() -> None:
"""Compare example model metrics."""

section("30. MODEL COMPARISON")

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "SVM",
]

accuracy = [
    0.82,
    0.85,
    0.91,
    0.88,
]

fig, ax = plt.subplots(
    figsize=(10, 5)
)

bars = ax.bar(
    models,
    accuracy,
)

ax.bar_label(
    bars,
    fmt="%.2f",
    padding=3,
)

ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")

ax.tick_params(
    axis="x",
    rotation=15,
)

plt.tight_layout()
plt.show()
=============================================================================
31. TRAINING VS VALIDATION
=============================================================================

def training_vs_validation() -> None:
"""
Compare training and validation metrics.

The values are illustrative and are intended to demonstrate
visualization technique rather than report measured model results.
"""

section("31. TRAINING VS VALIDATION")

models = [
    "Model A",
    "Model B",
    "Model C",
    "Model D",
]

training = [
    0.95,
    0.92,
    0.98,
    0.90,
]

validation = [
    0.86,
    0.88,
    0.80,
    0.87,
]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()

ax.bar(
    x - width / 2,
    training,
    width,
    label="Training",
)

ax.bar(
    x + width / 2,
    validation,
    width,
    label="Validation",
)

ax.set_xticks(x)
ax.set_xticklabels(models)

ax.set_ylim(0, 1)

ax.set_ylabel("Score")
ax.set_title("Training vs Validation Performance")

ax.legend()

plt.tight_layout()
plt.show()
=============================================================================
32. REGRESSION METRIC COMPARISON
=============================================================================

def regression_metric_comparison() -> None:
"""
Compare regression metrics.

Do not directly compare metrics with different units on the
same axis unless the visualization is designed appropriately.
"""

section("32. REGRESSION METRIC COMPARISON")

models = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
]

mae = [
    12.5,
    10.2,
    8.7,
]

fig, ax = plt.subplots()

bars = ax.bar(
    models,
    mae,
)

ax.bar_label(
    bars,
    fmt="%.1f",
    padding=3,
)

ax.set_ylabel("MAE")
ax.set_title("Mean Absolute Error Comparison")

ax.tick_params(
    axis="x",
    rotation=15,
)

plt.tight_layout()
plt.show()
=============================================================================
33. TOP-N FEATURE IMPORTANCE
=============================================================================

def top_n_feature_importance() -> None:
"""Display only the highest-ranked features."""

section("33. TOP-N FEATURE IMPORTANCE")

data = pd.Series(
    {
        "Age": 0.10,
        "Income": 0.30,
        "Experience": 0.22,
        "Education": 0.08,
        "Credit Score": 0.18,
        "Debt": 0.07,
        "Savings": 0.05,
    }
)

top_n = (
    data
    .sort_values(ascending=False)
    .head(5)
    .sort_values()
)

fig, ax = plt.subplots()

bars = ax.barh(
    top_n.index,
    top_n.values,
)

ax.bar_label(
    bars,
    fmt="%.2f",
    padding=3,
)

ax.set_xlabel("Importance")
ax.set_title("Top 5 Features")

plt.tight_layout()
plt.show()
=============================================================================
34. REUSABLE VERTICAL BAR FUNCTION
=============================================================================

def plot_bar_chart(
categories,
values,
title="Bar Chart",
xlabel="Category",
ylabel="Value",
show_values=True,
):
"""
Create a reusable professional bar chart.

Parameters
----------
categories:
    Category labels.

values:
    Numerical values corresponding to categories.

title:
    Plot title.

xlabel:
    X-axis label.

ylabel:
    Y-axis label.

show_values:
    Whether to display values above bars.

Returns
-------
tuple
    (fig, ax)
"""

fig, ax = plt.subplots()

bars = ax.bar(
    categories,
    values,
)

ax.set_title(title)
ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)

if show_values:
    ax.bar_label(
        bars,
        padding=3,
    )

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
)

ax.set_axisbelow(True)

fig.tight_layout()

return fig, ax
=============================================================================
35. REUSABLE HORIZONTAL BAR FUNCTION
=============================================================================

def plot_horizontal_bar_chart(
categories,
values,
title="Horizontal Bar Chart",
xlabel="Value",
):
"""Create a reusable horizontal bar chart."""

fig, ax = plt.subplots()

bars = ax.barh(
    categories,
    values,
)

ax.set_title(title)
ax.set_xlabel(xlabel)

ax.bar_label(
    bars,
    padding=3,
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.4,
)

ax.set_axisbelow(True)

fig.tight_layout()

return fig, ax
=============================================================================
36. REUSABLE FUNCTION DEMONSTRATION
=============================================================================

def reusable_function_demo() -> None:
"""Demonstrate reusable plotting functions."""

section("36. REUSABLE PLOTTING FUNCTION")

categories = [
    "Python",
    "Pandas",
    "NumPy",
    "Matplotlib",
]

values = [
    95,
    90,
    88,
    85,
]

fig, ax = plot_bar_chart(
    categories=categories,
    values=values,
    title="Python Data Stack Skills",
    xlabel="Technology",
    ylabel="Score",
)

plt.show()
=============================================================================
37. SAVE BAR CHART
=============================================================================

def save_bar_chart() -> None:
"""Save a publication-quality bar chart."""

section("37. SAVING A BAR CHART")

categories = [
    "A",
    "B",
    "C",
    "D",
]

values = [
    30,
    50,
    40,
    70,
]

fig, ax = plt.subplots(
    figsize=(9, 5)
)

bars = ax.bar(
    categories,
    values,
)

ax.bar_label(
    bars,
    padding=3,
)

ax.set_title("Saved Bar Chart")
ax.set_xlabel("Category")
ax.set_ylabel("Value")

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
)

ax.set_axisbelow(True)

fig.tight_layout()

output_path = "bar-chart.png"

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"Chart saved to: {output_path}")

plt.close(fig)
=============================================================================
38. COMPLETE ML VISUALIZATION EXAMPLE
=============================================================================

def complete_ml_example() -> None:
"""
Complete ML-oriented example.

Workflow:
    Model metrics
        ↓
    Sort models
        ↓
    Visualize comparison
        ↓
    Label exact values
"""

section("38. COMPLETE ML EXAMPLE")

model_results = pd.DataFrame(
    {
        "model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM",
        ],
        "accuracy": [
            0.82,
            0.85,
            0.91,
            0.88,
        ],
    }
)

model_results = model_results.sort_values(
    "accuracy"
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

bars = ax.barh(
    model_results["model"],
    model_results["accuracy"],
)

ax.bar_label(
    bars,
    fmt="%.2f",
    padding=3,
)

ax.set_xlim(0, 1)

ax.set_xlabel("Accuracy")
ax.set_title("Machine Learning Model Accuracy")

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.4,
)

ax.set_axisbelow(True)

fig.tight_layout()

plt.show()
=============================================================================
39. PROFESSIONAL BAR CHART CHECKLIST
=============================================================================

def professional_checklist() -> None:
"""Print a practical visualization checklist."""

section("39. PROFESSIONAL BAR CHART CHECKLIST")

checklist = [
    "Use bars for categorical comparisons.",
    "Start the quantitative axis at zero when appropriate.",
    "Use meaningful category labels.",
    "Label the axis and include units.",
    "Use a descriptive title.",
    "Avoid unnecessary visual decoration.",
    "Use horizontal bars for long category names.",
    "Sort ranking data when ranking is the main purpose.",
    "Add exact value labels when they improve readability.",
    "Use grouped bars for comparable subcategories.",
    "Use stacked bars for additive composition.",
    "Do not imply precision beyond the underlying data.",
    "Use error bars only when their meaning is clearly defined.",
    "Avoid excessive colors.",
    "Save figures at an appropriate resolution.",
]

for number, item in enumerate(
    checklist,
    start=1,
):
    print(f"{number:02d}. {item}")
=============================================================================
40. COMMON MISTAKES
=============================================================================

def common_mistakes() -> None:
"""Print common bar-chart mistakes."""

section("40. COMMON MISTAKES")

mistakes = [
    "Using a bar chart for continuous time-series data.",
    "Using a truncated axis that exaggerates differences.",
    "Using too many categories in one chart.",
    "Using too many colors without meaning.",
    "Leaving axes unlabeled.",
    "Using unreadable category names.",
    "Comparing quantities with incompatible units.",
    "Using 3D bars when a 2D chart communicates the data better.",
    "Hiding uncertainty in model metrics.",
    "Presenting illustrative ML metrics as measured results.",
]

for number, mistake in enumerate(
    mistakes,
    start=1,
):
    print(f"{number:02d}. {mistake}")
=============================================================================
41. MAIN
=============================================================================

def main() -> None:
"""Run the complete bar-chart tutorial."""

basic_bar_chart()
numerical_position_bar_chart()
horizontal_bar_chart()
custom_bar_width()
custom_bar_colors()
edge_styling()
transparency()

add_value_labels()
formatted_bar_labels()

grouped_bar_chart()
three_group_bar_chart()
stacked_bar_chart()
percentage_stacked_bar()

sorted_bar_chart()
horizontal_ranking()

negative_values()
error_bars()
custom_ticks()
gridlines()
annotations()
rotate_labels()

bar_subplots()

pandas_series_bar_chart()
pandas_dataframe_bar_chart()

ml_class_distribution()
class_imbalance_visualization()
feature_importance()
model_comparison()
training_vs_validation()
regression_metric_comparison()
top_n_feature_importance()

reusable_function_demo()

# Saving is demonstrated separately because it writes a file.
# Uncomment if you want to generate bar-chart.png.
#
# save_bar_chart()

complete_ml_example()

professional_checklist()
common_mistakes()
=============================================================================
42. SCRIPT ENTRY POINT
=============================================================================

if name == "main":
main()

=============================================================================
QUICK REFERENCE
=============================================================================


Basic:
ax.bar(categories, values)


Horizontal:
ax.barh(categories, values)


Width:
ax.bar(categories, values, width=0.5)


Labels:
bars = ax.bar(categories, values)
ax.bar_label(bars)


Grouped:
ax.bar(x - width / 2, values_a, width)
ax.bar(x + width / 2, values_b, width)


Stacked:
ax.bar(categories, values_a)
ax.bar(categories, values_b, bottom=values_a)


Error bars:
ax.bar(categories, values, yerr=errors, capsize=5)


Grid:
ax.grid(axis="y", linestyle="--", alpha=0.4)


Save:
fig.savefig(
"plot.png",
dpi=300,
bbox_inches="tight",
)


ML applications:
- Class distribution
- Feature importance
- Model comparison
- Training vs validation metrics
- Category frequency
- Business/feature analysis


=============================================================================

"""

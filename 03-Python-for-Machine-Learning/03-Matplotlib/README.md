# 📊 Matplotlib — Data Visualization for Machine Learning

> A complete, practical guide to **Matplotlib** for creating professional visualizations, exploring datasets, understanding machine-learning behavior, and communicating insights effectively.

---

## 📌 Overview

**Matplotlib** is one of Python's most widely used data visualization libraries.

In Machine Learning, visualization is essential for:

* Understanding datasets
* Detecting outliers
* Finding patterns and trends
* Understanding feature distributions
* Comparing variables
* Analyzing correlations
* Evaluating model predictions
* Visualizing classification results
* Understanding regression performance
* Communicating ML results

This section teaches Matplotlib from **beginner fundamentals to advanced machine-learning visualization techniques**.

---

## 🎯 Learning Objectives

By completing this section, you will learn how to:

* Understand Matplotlib architecture
* Create basic plots
* Customize figures and axes
* Create line, bar, scatter, histogram, and box plots
* Create multiple plots using subplots
* Customize colors, markers, and line styles
* Add titles, labels, legends, and annotations
* Control axes and ticks
* Create statistical visualizations
* Visualize distributions
* Detect outliers
* Visualize correlations
* Visualize time-series data
* Create ML-specific visualizations
* Visualize regression predictions
* Visualize classification results
* Plot confusion matrices
* Visualize feature importance
* Visualize learning curves
* Save high-quality figures
* Build reusable visualization functions
* Follow professional visualization practices

---

> The exact filenames can be expanded as the repository grows. The important goal is to maintain a progression from **visualization fundamentals → data analysis → machine-learning visualization**.

---

# 🚀 1. Installation

Install Matplotlib using pip:

```bash
pip install matplotlib
```

For a typical Machine Learning environment:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Verify the installation:

```python
import matplotlib

print(matplotlib.__version__)
```

---

# 📦 2. Importing Matplotlib

The most common import is:

```python
import matplotlib.pyplot as plt
```

`pyplot` provides the high-level plotting interface used throughout most beginner and intermediate workflows.

Example:

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.show()
```

---

# 🧠 3. Understanding Matplotlib

A typical Matplotlib visualization has the following structure:

```text
Figure
│
├── Axes
│   ├── X-axis
│   ├── Y-axis
│   ├── Title
│   ├── Labels
│   ├── Legend
│   └── Data
│
└── Other Axes
```

### Important terminology

| Concept    | Meaning                                        |
| ---------- | ---------------------------------------------- |
| Figure     | Entire visualization canvas                    |
| Axes       | Individual plotting area                       |
| Axis       | X or Y coordinate system                       |
| Title      | Description of the plot                        |
| Label      | Description of an axis                         |
| Legend     | Identifies plotted data                        |
| Tick       | Individual axis marker                         |
| Annotation | Text or marker pointing to a specific location |

---

# 📈 4. Basic Line Plot

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 30, 25]

plt.plot(x, y)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Basic Line Plot")

plt.show()
```

Line plots are useful for:

* Trends
* Time series
* Continuous measurements
* Training progress
* Validation metrics

---

# 📊 5. Bar Chart

Bar charts are useful for comparing discrete categories.

```python
import matplotlib.pyplot as plt

categories = ["A", "B", "C", "D"]
values = [20, 35, 30, 40]

plt.bar(categories, values)

plt.xlabel("Category")
plt.ylabel("Value")
plt.title("Category Comparison")

plt.show()
```

Common ML applications:

* Class distribution
* Feature importance
* Model comparison
* Category frequency

---

# 📉 6. Horizontal Bar Chart

```python
plt.barh(categories, values)

plt.xlabel("Value")
plt.ylabel("Category")

plt.show()
```

Horizontal bars are especially useful when category names are long.

---

# 🔵 7. Scatter Plot

Scatter plots show relationships between two numerical variables.

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 3, 8, 10]

plt.scatter(x, y)

plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Feature Relationship")

plt.show()
```

Useful for:

* Correlation analysis
* Cluster visualization
* Feature relationships
* Regression analysis
* Outlier detection

---

# 📊 8. Histogram

Histograms show the distribution of numerical data.

```python
import matplotlib.pyplot as plt

values = [10, 12, 13, 15, 15, 16, 18, 20, 21, 22]

plt.hist(values, bins=5)

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Value Distribution")

plt.show()
```

Histograms help identify:

* Skewness
* Spread
* Concentration
* Possible outliers
* Approximate distribution shape

---

# 🥧 9. Pie Chart

```python
import matplotlib.pyplot as plt

labels = ["Python", "Java", "C++", "JavaScript"]
values = [40, 25, 20, 15]

plt.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
)

plt.title("Programming Language Distribution")

plt.show()
```

Pie charts should generally be reserved for a small number of clearly distinct proportions.

---

# 📦 10. Box Plot

Box plots are useful for understanding distributions and detecting potential outliers.

```python
import matplotlib.pyplot as plt

values = [10, 12, 13, 15, 15, 16, 18, 20, 50]

plt.boxplot(values)

plt.ylabel("Value")
plt.title("Box Plot")

plt.show()
```

A box plot summarizes:

```text
Minimum
   │
Q1 ├──────────────┐
   │              │
Median           │
   │              │
Q3 ├──────────────┘
   │
Maximum
```

Potential outliers are displayed separately from the main distribution.

---

# 🧩 11. Subplots

Subplots allow multiple visualizations inside one figure.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2)

axes[0, 0].plot([1, 2, 3], [1, 4, 9])
axes[0, 1].bar(["A", "B"], [10, 20])
axes[1, 0].scatter([1, 2, 3], [3, 2, 5])
axes[1, 1].hist([1, 2, 2, 3, 3, 3])

plt.tight_layout()
plt.show()
```

Useful for comparing:

* Multiple features
* Multiple models
* Training vs validation
* Different distributions

---

# 🎨 12. Plot Customization

Matplotlib allows detailed control over visualization appearance.

```python
plt.plot(
    x,
    y,
    linestyle="--",
    marker="o",
    linewidth=2,
)
```

Common properties:

| Property     | Purpose           |
| ------------ | ----------------- |
| `color`      | Line/marker color |
| `linestyle`  | Line pattern      |
| `linewidth`  | Line thickness    |
| `marker`     | Point style       |
| `markersize` | Marker size       |
| `alpha`      | Transparency      |

---

# 🏷️ 13. Titles and Labels

A professional plot should clearly communicate what it represents.

```python
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
```

Good labels should include units where appropriate:

```python
plt.ylabel("Revenue (₹)")
```

---

# 🧾 14. Legends

When multiple datasets are plotted, use a legend.

```python
plt.plot(
    x,
    y1,
    label="Training",
)

plt.plot(
    x,
    y2,
    label="Validation",
)

plt.legend()
```

---

# 📐 15. Axis Limits

Control the visible range using:

```python
plt.xlim(0, 100)
plt.ylim(0, 500)
```

Or with the object-oriented API:

```python
ax.set_xlim(0, 100)
ax.set_ylim(0, 500)
```

---

# 🔲 16. Grid

A grid can make numerical plots easier to read.

```python
plt.grid(True)
```

For more control:

```python
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5,
)
```

Use grids selectively rather than adding visual noise.

---

# 📝 17. Annotations

Annotations highlight important observations.

```python
plt.annotate(
    "Peak",
    xy=(5, 100),
    xytext=(3, 120),
    arrowprops={"arrowstyle": "->"},
)
```

Useful for:

* Maximum values
* Outliers
* Important events
* Model performance changes

---

# 📏 18. Figure Size

Control figure dimensions:

```python
plt.figure(figsize=(10, 6))
```

For high-quality saved figures:

```python
plt.figure(
    figsize=(10, 6),
    dpi=150,
)
```

---

# 💾 19. Saving Figures

Save a plot using:

```python
plt.savefig("plot.png")
```

For higher-quality output:

```python
plt.savefig(
    "plot.png",
    dpi=300,
    bbox_inches="tight",
)
```

Common formats:

```text
.png
.jpg
.jpeg
.svg
.pdf
```

---

# 🧱 20. Object-Oriented Matplotlib

For larger projects, the object-oriented API provides better control.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(
    [1, 2, 3, 4],
    [10, 20, 15, 30],
)

ax.set_title("Sales Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")

plt.show()
```

### Recommended mental model

```text
fig
 │
 └── ax
      ├── plot()
      ├── set_title()
      ├── set_xlabel()
      ├── set_ylabel()
      └── legend()
```

For reusable and production-oriented visualization code, prefer the object-oriented API.

---

# 📊 21. Multiple Data Series

```python
import matplotlib.pyplot as plt

months = [1, 2, 3, 4, 5]

product_a = [100, 120, 140, 160, 180]
product_b = [90, 110, 130, 150, 170]

fig, ax = plt.subplots()

ax.plot(
    months,
    product_a,
    label="Product A",
)

ax.plot(
    months,
    product_b,
    label="Product B",
)

ax.set_title("Product Sales")
ax.set_xlabel("Month")
ax.set_ylabel("Sales")

ax.legend()

plt.show()
```

---

# 📈 22. Distribution Visualization

Understanding feature distributions is a fundamental ML task.

Common visualizations:

```text
Histogram
    ↓
Box Plot
    ↓
Density/Distribution Visualization
    ↓
Statistical Summary
```

Questions to ask:

* Is the feature skewed?
* Are there extreme values?
* Is the distribution approximately symmetric?
* Are there multiple groups?
* Does the feature require transformation?

---

# 🔗 23. Correlation Visualization

Scatter plots can help investigate relationships between features.

```python
import matplotlib.pyplot as plt

age = [20, 25, 30, 35, 40]
income = [25000, 30000, 42000, 55000, 70000]

fig, ax = plt.subplots()

ax.scatter(age, income)

ax.set_xlabel("Age")
ax.set_ylabel("Income")
ax.set_title("Age vs Income")

plt.show()
```

Visualization does not prove causation.

---

# 🕒 24. Time-Series Visualization

Line plots are commonly used for time-dependent data.

```python
import pandas as pd
import matplotlib.pyplot as plt

dates = pd.date_range(
    "2026-01-01",
    periods=6,
    freq="MS",
)

sales = [100, 120, 115, 140, 160, 180]

fig, ax = plt.subplots()

ax.plot(dates, sales)

ax.set_title("Monthly Sales")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")

fig.autofmt_xdate()

plt.show()
```

---

# 🤖 25. Matplotlib for Machine Learning

Visualization is used throughout the ML lifecycle:

```text
Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ├── Feature Distribution
     ├── Correlation
     ├── Outliers
     └── Class Distribution
     │
     ▼
Feature Engineering
     │
     ▼
Model Training
     │
     ├── Learning Curves
     ├── Training Metrics
     └── Validation Metrics
     │
     ▼
Model Evaluation
     │
     ├── Predictions
     ├── Residuals
     ├── Confusion Matrix
     └── Feature Importance
```

---

# 📊 26. Feature Distribution

Before training a model, inspect numerical features.

```python
import matplotlib.pyplot as plt

age = [18, 20, 21, 22, 25, 27, 30, 35, 40, 50]

fig, ax = plt.subplots()

ax.hist(
    age,
    bins=5,
)

ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")

plt.show()
```

---

# 🔍 27. Outlier Visualization

Box plots provide a simple way to inspect potential outliers.

```python
fig, ax = plt.subplots()

ax.boxplot(
    income,
    vert=False,
)

ax.set_xlabel("Income")
ax.set_title("Income Outlier Analysis")

plt.show()
```

Remember:

> A statistical outlier is not automatically an incorrect observation.

---

# 📈 28. Regression Visualization

For regression models, compare actual and predicted values.

```python
import matplotlib.pyplot as plt

actual = [100, 120, 150, 180, 200]
predicted = [105, 118, 145, 175, 210]

fig, ax = plt.subplots()

ax.scatter(
    actual,
    predicted,
)

ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Actual vs Predicted")

plt.show()
```

A useful reference line is:

```text
Predicted
   │
   │       /
   │      /
   │     /
   │    /
   │   /
   └──────────── Actual
```

The closer the observations are to the ideal `y = x` relationship, the closer predictions are to actual values.

---

# 📉 29. Regression Residuals

Residual:

```text
Residual = Actual - Predicted
```

Visualization:

```python
residuals = [
    actual_value - predicted_value
    for actual_value, predicted_value
    in zip(actual, predicted)
]
```

Plot:

```python
fig, ax = plt.subplots()

ax.scatter(
    predicted,
    residuals,
)

ax.axhline(
    0,
    linestyle="--",
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Residual")
ax.set_title("Residual Analysis")

plt.show()
```

Residual plots can help identify:

* Non-linearity
* Heteroscedasticity
* Systematic errors
* Potential outliers

---

# 🧮 30. Classification Visualization

Classification models can be visualized using:

* Class distributions
* Decision boundaries
* Confusion matrices
* ROC curves
* Precision-recall curves
* Prediction probabilities

---

# 🎯 31. Confusion Matrix

A confusion matrix contains:

```text
                 Predicted
               0         1

Actual 0      TN        FP

Actual 1      FN        TP
```

Matplotlib can visualize a confusion matrix manually:

```python
import numpy as np
import matplotlib.pyplot as plt

matrix = np.array([
    [90, 10],
    [5, 95],
])

fig, ax = plt.subplots()

image = ax.imshow(matrix)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Class 0", "Class 1"])
ax.set_yticklabels(["Class 0", "Class 1"])

for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
        ax.text(
            column,
            row,
            matrix[row, column],
            ha="center",
            va="center",
        )

fig.colorbar(image)

plt.show()
```

---

# 🧠 32. Feature Importance

Tree-based models often provide feature importance values.

Example:

```python
features = [
    "age",
    "income",
    "experience",
    "education",
]

importance = [
    0.20,
    0.45,
    0.25,
    0.10,
]
```

Visualize:

```python
fig, ax = plt.subplots()

ax.barh(
    features,
    importance,
)

ax.set_xlabel("Importance")
ax.set_title("Feature Importance")

plt.show()
```

Important:

> Feature importance is model-specific and should not automatically be interpreted as causal importance.

---

# 📚 33. Learning Curves

Learning curves show how model performance changes as training progresses.

Typical metrics:

```text
Epoch
  │
  ├── Training Loss
  ├── Validation Loss
  ├── Training Accuracy
  └── Validation Accuracy
```

Example:

```python
epochs = [1, 2, 3, 4, 5]

training_loss = [
    0.80,
    0.60,
    0.45,
    0.35,
    0.28,
]

validation_loss = [
    0.85,
    0.65,
    0.52,
    0.55,
    0.62,
]
```

Plot:

```python
fig, ax = plt.subplots()

ax.plot(
    epochs,
    training_loss,
    label="Training Loss",
)

ax.plot(
    epochs,
    validation_loss,
    label="Validation Loss",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Learning Curve")

ax.legend()

plt.show()
```

A widening gap between training and validation performance can be a signal of overfitting, but interpretation should consider the model, metric, dataset, and training setup.

---

# 🧪 34. Model Comparison

Suppose multiple models produce:

```python
models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
]

accuracy = [
    0.82,
    0.85,
    0.91,
]
```

Visualize:

```python
fig, ax = plt.subplots()

ax.bar(
    models,
    accuracy,
)

ax.set_ylabel("Accuracy")
ax.set_title("Model Performance Comparison")

plt.xticks(rotation=15)

plt.show()
```

When comparing models, use the same:

* Dataset split
* Evaluation metric
* Evaluation protocol
* Preprocessing strategy

---

# 🧮 35. NumPy + Matplotlib

Matplotlib works naturally with NumPy.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(
    0,
    10,
    100,
)

y = np.sin(x)

fig, ax = plt.subplots()

ax.plot(x, y)

ax.set_title("Sine Wave")

plt.show()
```

---

# 🐼 36. Pandas + Matplotlib

Pandas integrates directly with Matplotlib.

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(
    {
        "month": ["Jan", "Feb", "Mar", "Apr"],
        "sales": [100, 120, 150, 180],
    }
)

data.plot(
    x="month",
    y="sales",
)

plt.title("Monthly Sales")

plt.show()
```

For larger projects, it is useful to understand both:

```text
Pandas plotting
      ↓
Quick exploration

Matplotlib
      ↓
Fine-grained customization
```

---

# 🧱 37. Reusable Visualization Functions

Avoid duplicating visualization logic.

```python
import matplotlib.pyplot as plt


def plot_feature_distribution(
    values,
    title,
    xlabel,
):
    fig, ax = plt.subplots()

    ax.hist(values)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")

    return fig, ax
```

Usage:

```python
fig, ax = plot_feature_distribution(
    values=[10, 20, 20, 30, 40],
    title="Age Distribution",
    xlabel="Age",
)

plt.show()
```

Reusable functions make ML projects easier to maintain.

---

# 🧹 38. Visualization Best Practices

## Use meaningful titles

Bad:

```text
Plot 1
```

Better:

```text
Monthly Revenue Trend — 2026
```

---

## Label your axes

Bad:

```python
plt.xlabel("X")
```

Better:

```python
plt.xlabel("Age (years)")
```

---

## Avoid unnecessary decoration

A visualization should communicate information rather than distract from it.

---

## Choose the correct chart

| Goal                       | Recommended Chart |
| -------------------------- | ----------------- |
| Trend                      | Line              |
| Category comparison        | Bar               |
| Distribution               | Histogram         |
| Relationship               | Scatter           |
| Outliers                   | Box plot          |
| Composition                | Pie / stacked bar |
| Correlation                | Scatter / matrix  |
| Model error                | Residual plot     |
| Classification performance | Confusion matrix  |
| Feature importance         | Horizontal bar    |

---

# ⚠️ 39. Common Visualization Mistakes

### 1. Misleading axis ranges

An inappropriate axis range can exaggerate differences.

### 2. Too many colors

Excessive colors reduce readability.

### 3. Missing labels

Readers should not have to guess what the axes represent.

### 4. Overlapping points

Large datasets can make scatter plots difficult to interpret.

Possible solutions:

* Smaller markers
* Transparency
* Sampling
* Aggregation
* Hexbin plots

### 5. Too many subplots

Only include plots that answer meaningful questions.

### 6. Ignoring units

Always include units where applicable.

### 7. Treating correlation as causation

A visual relationship does not establish a causal relationship.

### 8. Hiding uncertainty

Where appropriate, communicate uncertainty rather than presenting estimates as exact.

---

# 🧠 40. Matplotlib in an ML Workflow

A practical ML visualization workflow:

```text
             Dataset
                │
                ▼
       ┌─────────────────┐
       │ Data Inspection │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Distributions   │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Relationships   │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Outlier Analysis│
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Feature         │
       │ Engineering     │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Model Training  │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Model Evaluation│
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Visualization   │
       │ of Results      │
       └─────────────────┘
```

---

# 🧪 41. Recommended Learning Sequence

Follow the files in this order:

### 🟢 Beginner

```text
01-basic-plot.py
02-line-plot.py
03-bar-chart.py
04-histogram.py
05-scatter-plot.py
06-pie-chart.py
07-box-plot.py
```

### 🟡 Intermediate

```text
08-subplots.py
09-customization.py
10-legends-labels.py
11-colors-markers-styles.py
12-axis-customization.py
13-annotations.py
14-grid.py
15-figure-size.py
16-saving-plots.py
```

### 🟠 Data Analysis

```text
17-multiple-plots.py
18-statistical-plots.py
19-distribution.py
20-correlation.py
21-time-series.py
```

### 🔴 Machine Learning

```text
22-ml-data-visualization.py
23-feature-distribution.py
24-outlier-visualization.py
25-regression-plot.py
26-classification-plot.py
27-confusion-matrix.py
28-feature-importance.py
29-learning-curve.py
30-model-comparison.py
```

---

# 🛠️ 42. Useful Matplotlib Functions

## Figure

```python
plt.figure()
plt.subplots()
plt.savefig()
plt.show()
```

## Line

```python
plt.plot()
```

## Scatter

```python
plt.scatter()
```

## Bar

```python
plt.bar()
plt.barh()
```

## Distribution

```python
plt.hist()
plt.boxplot()
```

## Labels

```python
plt.title()
plt.xlabel()
plt.ylabel()
plt.legend()
```

## Axis

```python
plt.xlim()
plt.ylim()
plt.xticks()
plt.yticks()
```

## Grid

```python
plt.grid()
```

## Annotation

```python
plt.annotate()
plt.text()
```

---

# ⚡ 43. Quick Reference

```python
import matplotlib.pyplot as plt

# Figure
fig, ax = plt.subplots(figsize=(10, 6))

# Line
ax.plot(x, y)

# Scatter
ax.scatter(x, y)

# Bar
ax.bar(categories, values)

# Horizontal bar
ax.barh(categories, values)

# Histogram
ax.hist(values, bins=10)

# Box plot
ax.boxplot(values)

# Labels
ax.set_title("Title")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Legend
ax.legend()

# Grid
ax.grid(True)

# Limits
ax.set_xlim(...)
ax.set_ylim(...)

# Save
fig.savefig(
    "plot.png",
    dpi=300,
    bbox_inches="tight",
)

# Display
plt.show()
```

---

# 📚 44. Matplotlib + ML Ecosystem

Matplotlib is commonly used together with:

```text
NumPy
  │
  ├── Numerical computation
  │
  ▼
Pandas
  │
  ├── Data manipulation
  │
  ▼
Matplotlib
  │
  ├── Visualization
  │
  ▼
Scikit-learn
  │
  ├── Machine Learning
  │
  ▼
Model Evaluation
```

Typical stack:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

---

# 🧪 45. Practice Problems

## Beginner

1. Create a line plot for weekly temperatures.
2. Create a bar chart of programming languages.
3. Create a histogram of exam scores.
4. Create a scatter plot of study hours vs marks.
5. Create a box plot of salaries.

## Intermediate

6. Create a 2×2 subplot dashboard.
7. Customize titles and axis labels.
8. Add legends to multiple lines.
9. Annotate the maximum value.
10. Save a plot as a high-resolution PNG.
11. Create a time-series sales chart.
12. Compare two products.
13. Visualize class distributions.
14. Detect potential outliers visually.

## Machine Learning

15. Plot feature distributions.
16. Plot actual vs predicted regression values.
17. Plot regression residuals.
18. Visualize a confusion matrix.
19. Plot feature importance.
20. Plot training and validation loss.
21. Compare multiple ML models.
22. Visualize class separation.
23. Create an ML evaluation dashboard.

---

# 🚀 46. Mini Projects

### 📊 Project 1 — Student Performance Visualization

Analyze:

* Study hours
* Attendance
* Marks
* Subject performance
* Grade distribution

Create:

```text
Histogram
Bar Chart
Scatter Plot
Box Plot
```

---

### 💰 Project 2 — Sales Analytics

Analyze:

* Monthly sales
* Product performance
* Regional performance
* Revenue trends

Create:

```text
Line Chart
Bar Chart
Histogram
Subplots
```

---

### 🤖 Project 3 — ML Model Visualization

Visualize:

* Training data
* Predictions
* Residuals
* Confusion matrix
* Feature importance
* Learning curves

---

# 📖 47. Recommended Project Structure

A professional ML project can organize visualization utilities like:

```text
project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
│       ├── distributions.py
│       ├── correlations.py
│       ├── model_metrics.py
│       └── plots.py
│
├── reports/
│   └── figures/
│
└── README.md
```

This keeps visualization code reusable instead of embedding every plot directly inside notebooks.

---

# 🔐 48. Reproducibility

When visualizations depend on random sampling:

```python
import numpy as np

rng = np.random.default_rng(42)

values = rng.normal(
    loc=0,
    scale=1,
    size=1000,
)
```

Using a fixed random seed can make exploratory visualizations reproducible.

---

# 🏆 49. Professional Visualization Checklist

Before committing a visualization:

* [ ] Is the chart type appropriate?
* [ ] Is the title meaningful?
* [ ] Are axes labeled?
* [ ] Are units included?
* [ ] Is the legend necessary?
* [ ] Are values readable?
* [ ] Is the figure large enough?
* [ ] Are outliers handled transparently?
* [ ] Is the visualization free from unnecessary clutter?
* [ ] Does the visualization answer a specific question?
* [ ] Is the interpretation supported by the underlying data?
* [ ] Is uncertainty communicated where appropriate?
* [ ] Is the figure reproducible?
* [ ] Is the output saved at an appropriate resolution?

---

# 🔑 50. Key Takeaways

After completing this section, you should understand that:

1. **Matplotlib is a foundational Python visualization library.**
2. **Figures contain one or more Axes objects.**
3. **Line plots are useful for trends and time series.**
4. **Bar charts compare categories.**
5. **Scatter plots investigate relationships.**
6. **Histograms visualize numerical distributions.**
7. **Box plots help summarize distributions and identify potential outliers.**
8. **Subplots allow multiple visualizations in one figure.**
9. **The object-oriented API is useful for reusable and complex visualizations.**
10. **Visualization is an important part of exploratory data analysis.**
11. **ML models can be visualized through predictions, residuals, confusion matrices, learning curves, and feature importance.**
12. **A visualization should communicate evidence clearly rather than exaggerate conclusions.**

---

# 🗺️ 51. Machine Learning Learning Roadmap

This section fits into the broader repository roadmap:

```text
Python
  │
  ▼
NumPy
  │
  ▼
Pandas
  │
  ▼
Matplotlib
  │
  ▼
Seaborn / Advanced Visualization
  │
  ▼
Statistics
  │
  ▼
Data Preprocessing
  │
  ▼
Feature Engineering
  │
  ▼
Machine Learning
  │
  ├── Regression
  ├── Classification
  ├── Clustering
  └── Dimensionality Reduction
  │
  ▼
Model Evaluation
  │
  ▼
Hyperparameter Tuning
  │
  ▼
Deployment
```

---

# 📚 52. Useful Resources

* **Matplotlib Documentation:** https://matplotlib.org/stable/
* **Matplotlib Pyplot API:** https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
* **Matplotlib Examples:** https://matplotlib.org/stable/gallery/
* **NumPy Documentation:** https://numpy.org/doc/
* **Pandas Documentation:** https://pandas.pydata.org/docs/
* **Scikit-learn Documentation:** https://scikit-learn.org/stable/

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning • Python • Data Science • Artificial Intelligence

GitHub:
https://github.com/Kishor055

---

# 🤝 Contributing

Contributions are welcome.

If you find an issue or want to improve the examples:

1. Fork the repository.
2. Create a feature branch.
3. Add or improve the visualization example.
4. Test the code.
5. Commit your changes.
6. Open a Pull Request.

Example:

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/matplotlib-improvements
```

---

# 📄 License

This project is intended for educational and learning purposes.

See the repository license for complete terms.

---

## ⭐ Keep Learning

> **Visualize the data. Understand the pattern. Question the assumption. Build the model.**

```text
Learn → Explore → Visualize → Analyze → Model → Evaluate → Deploy
```

⭐ If this repository helps you learn Machine Learning, consider giving it a star and following the project as it evolves.

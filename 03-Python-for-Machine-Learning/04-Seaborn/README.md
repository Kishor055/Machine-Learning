# 📊 Seaborn — Statistical Data Visualization for Machine Learning

> A professional, practical guide to **Seaborn** for statistical data visualization, exploratory data analysis (EDA), feature analysis, correlation analysis, and machine learning workflows.

---

## 📌 Overview

**Seaborn** is a Python data visualization library built on top of **Matplotlib**. It provides a high-level interface for creating attractive and statistically meaningful visualizations with significantly less code.

In Machine Learning, visualization is an essential part of understanding data before training a model.

Seaborn is particularly useful for:

* 📊 Exploratory Data Analysis (EDA)
* 🔍 Finding relationships between variables
* 📈 Understanding distributions
* 🔗 Correlation analysis
* 🎯 Comparing target classes
* 🧹 Detecting outliers
* 🧩 Understanding categorical features
* 🤖 Analyzing machine learning datasets
* 📉 Evaluating model-related patterns
* 🎨 Creating professional statistical plots

---

## 🎯 Learning Objectives

After completing this module, you should be able to:

* Understand Seaborn fundamentals
* Install and import Seaborn
* Work with Pandas DataFrames
* Create statistical visualizations
* Visualize numerical and categorical data
* Analyze distributions
* Create relationship plots
* Create categorical plots
* Build correlation heatmaps
* Detect potential outliers
* Create pair plots
* Customize Seaborn charts
* Use Matplotlib together with Seaborn
* Visualize machine learning datasets
* Create professional EDA dashboards
* Save high-quality visualizations
* Follow visualization best practices

---

# 🧠 Why Seaborn for Machine Learning?

Machine Learning datasets often contain many variables.

For example:

```text
Age
Salary
Experience
Education
Department
City
Performance
Purchased
```

Looking at raw data alone may not reveal important patterns.

Visualization can help answer questions such as:

```text
Does experience increase salary?
Are some classes imbalanced?
Which variables are correlated?
Are there extreme outliers?
Does one category behave differently?
Are features normally distributed?
```

A good ML workflow therefore includes:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Visualization
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
```

---

# 🛠️ Installation

Install Seaborn using pip:

```bash
pip install seaborn
```

Recommended complete environment:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Verify installation:

```python
import seaborn as sns

print(sns.__version__)
```

---

# 📦 Importing Seaborn

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

Common convention:

```python
import seaborn as sns
```

---

# 🎨 Seaborn Themes

Seaborn provides built-in styles.

```python
sns.set_theme()
```

Other styles can be selected with:

```python
sns.set_style("whitegrid")
```

Available styles include:

```python
sns.set_style("darkgrid")
sns.set_style("whitegrid")
sns.set_style("dark")
sns.set_style("white")
sns.set_style("ticks")
```

Example:

```python
sns.set_theme(style="whitegrid")
```

---

# 📊 Seaborn Dataset

Seaborn includes example datasets that are useful for learning.

```python
import seaborn as sns

df = sns.load_dataset("tips")

print(df.head())
```

Inspect the data:

```python
print(df.shape)
print(df.info())
print(df.describe())
```

---

# 📋 Common Example Datasets

Some commonly available Seaborn datasets include:

```text
tips
iris
titanic
penguins
flights
diamonds
mpg
exercise
car_crashes
planets
```

Example:

```python
df = sns.load_dataset("iris")
```

---

# 📈 1. Line Plot

Line plots are useful for understanding trends.

```python
sns.lineplot(
    data=df,
    x="day",
    y="total_bill"
)

plt.show()
```

Multiple groups:

```python
sns.lineplot(
    data=df,
    x="day",
    y="total_bill",
    hue="sex"
)

plt.show()
```

Useful for:

* Time-series analysis
* Trends
* Model learning curves
* Performance changes

---

# 📊 2. Bar Plot

Bar plots compare categorical groups.

```python
sns.barplot(
    data=df,
    x="day",
    y="total_bill"
)

plt.show()
```

Using an additional category:

```python
sns.barplot(
    data=df,
    x="day",
    y="total_bill",
    hue="sex"
)

plt.show()
```

---

# 📉 3. Count Plot

Count plots show the frequency of categories.

```python
sns.countplot(
    data=df,
    x="day"
)

plt.show()
```

Useful for identifying:

* Class imbalance
* Category frequency
* Dataset composition

Example:

```python
sns.countplot(
    data=df,
    x="sex"
)

plt.show()
```

---

# 📊 4. Histogram

Histograms show the distribution of numerical data.

```python
sns.histplot(
    data=df,
    x="total_bill"
)

plt.show()
```

Specify bins:

```python
sns.histplot(
    data=df,
    x="total_bill",
    bins=20
)

plt.show()
```

Add KDE:

```python
sns.histplot(
    data=df,
    x="total_bill",
    bins=20,
    kde=True
)

plt.show()
```

---

# 📈 5. KDE Plot

KDE represents the estimated probability density of a variable.

```python
sns.kdeplot(
    data=df,
    x="total_bill"
)

plt.show()
```

Multiple distributions:

```python
sns.kdeplot(
    data=df,
    x="total_bill",
    hue="sex",
    fill=True
)

plt.show()
```

---

# 📦 6. Box Plot

Box plots are useful for detecting potential outliers.

```python
sns.boxplot(
    data=df,
    x="day",
    y="total_bill"
)

plt.show()
```

A box plot summarizes:

```text
Minimum
Q1
Median
Q3
Maximum
Potential Outliers
```

---

# 🎻 7. Violin Plot

Violin plots combine distribution information with box-plot-like summaries.

```python
sns.violinplot(
    data=df,
    x="day",
    y="total_bill"
)

plt.show()
```

With groups:

```python
sns.violinplot(
    data=df,
    x="day",
    y="total_bill",
    hue="sex"
)

plt.show()
```

---

# 🔵 8. Scatter Plot

Scatter plots visualize relationships between two numerical variables.

```python
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip"
)

plt.show()
```

Add category information:

```python
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="sex"
)

plt.show()
```

Change point size:

```python
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    size="size",
    hue="sex"
)

plt.show()
```

---

# 🔗 9. Regression Plot

Seaborn can visualize a regression relationship.

```python
sns.regplot(
    data=df,
    x="total_bill",
    y="tip"
)

plt.show()
```

The plot contains:

```text
Data Points
+
Regression Line
+
Confidence Interval
```

---

# 📈 10. Linear Model Plot

`lmplot()` provides a figure-level interface for regression visualization.

```python
sns.lmplot(
    data=df,
    x="total_bill",
    y="tip"
)

plt.show()
```

Group by category:

```python
sns.lmplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="sex"
)

plt.show()
```

---

# 🔥 11. Heatmap

Heatmaps are extremely useful in Machine Learning.

They can visualize correlation matrices.

Example:

```python
numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

A correlation matrix helps identify relationships between numerical features.

---

# 🔢 Understanding Correlation

Correlation values generally range from:

```text
-1 → Strong negative relationship
 0 → Little or no linear relationship
+1 → Strong positive relationship
```

Example:

```text
          Age  Salary
Age      1.00   0.72
Salary   0.72   1.00
```

The `0.72` indicates a positive linear association between the two variables.

> Correlation does not automatically imply causation.

---

# 🔍 12. Pair Plot

Pair plots visualize relationships between multiple numerical variables.

```python
iris = sns.load_dataset("iris")

sns.pairplot(iris)

plt.show()
```

Color by class:

```python
sns.pairplot(
    iris,
    hue="species"
)

plt.show()
```

Pair plots are especially useful during EDA.

---

# 🧩 13. Joint Plot

Joint plots visualize the relationship between two variables along with their individual distributions.

```python
sns.jointplot(
    data=df,
    x="total_bill",
    y="tip"
)

plt.show()
```

Scatter-based joint plot:

```python
sns.jointplot(
    data=df,
    x="total_bill",
    y="tip",
    kind="scatter"
)

plt.show()
```

Regression:

```python
sns.jointplot(
    data=df,
    x="total_bill",
    y="tip",
    kind="reg"
)

plt.show()
```

---

# 📊 14. Strip Plot

Strip plots display individual observations across categories.

```python
sns.stripplot(
    data=df,
    x="day",
    y="total_bill"
)

plt.show()
```

Add jitter:

```python
sns.stripplot(
    data=df,
    x="day",
    y="total_bill",
    jitter=True
)

plt.show()
```

---

# 📌 15. Swarm Plot

Swarm plots show individual observations while attempting to prevent overlap.

```python
sns.swarmplot(
    data=df,
    x="day",
    y="total_bill"
)

plt.show()
```

Useful for:

* Small datasets
* Distribution comparison
* Individual observations

---

# 🧱 16. Categorical Plot

Seaborn provides `catplot()` as a figure-level interface for categorical visualizations.

```python
sns.catplot(
    data=df,
    x="day",
    y="total_bill",
    kind="box"
)

plt.show()
```

Other options include:

```text
strip
swarm
box
violin
boxen
point
bar
count
```

---

# 📦 17. Boxen Plot

Boxen plots are useful for larger datasets.

```python
sns.boxenplot(
    data=df,
    x="day",
    y="total_bill"
)

plt.show()
```

---

# 📍 18. Point Plot

Point plots compare estimated values across categories.

```python
sns.pointplot(
    data=df,
    x="day",
    y="total_bill"
)

plt.show()
```

---

# 🧭 19. Hue

`hue` is one of the most useful Seaborn parameters.

Example:

```python
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="sex"
)

plt.show()
```

Hue allows another categorical variable to be represented visually.

---

# 📐 20. Size

Point size can represent another variable.

```python
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    size="size"
)

plt.show()
```

---

# 🔄 21. Style

Different marker styles can represent categories.

```python
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    style="sex"
)

plt.show()
```

Multiple encodings:

```python
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip",
    hue="sex",
    style="time",
    size="size"
)

plt.show()
```

---

# 🎨 22. Color Palettes

Seaborn provides many color palettes.

```python
sns.color_palette()
```

Examples:

```python
sns.color_palette("deep")
sns.color_palette("muted")
sns.color_palette("bright")
sns.color_palette("pastel")
sns.color_palette("dark")
sns.color_palette("colorblind")
```

Use a palette:

```python
sns.set_palette("deep")
```

Or directly:

```python
sns.barplot(
    data=df,
    x="day",
    y="total_bill",
    palette="viridis"
)

plt.show()
```

---

# 🌈 23. Sequential Palettes

Useful when values progress from low to high.

Examples:

```python
viridis
mako
rocket
flare
crest
```

Example:

```python
sns.heatmap(
    correlation,
    cmap="viridis"
)

plt.show()
```

---

# 🔀 24. Diverging Palettes

Useful when values have a meaningful center such as zero.

```python
sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0
)

plt.show()
```

---

# 🏷️ 25. Titles and Labels

Seaborn works naturally with Matplotlib.

```python
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip"
)

plt.title("Total Bill vs Tip")
plt.xlabel("Total Bill")
plt.ylabel("Tip")

plt.show()
```

---

# 📐 26. Figure Size

```python
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip"
)

plt.show()
```

---

# 🧹 27. Removing Spines

```python
sns.despine()
```

Example:

```python
sns.scatterplot(
    data=df,
    x="total_bill",
    y="tip"
)

sns.despine()

plt.show()
```

---

# 🧩 28. FacetGrid

FacetGrid allows the same visualization to be split across categories.

```python
g = sns.FacetGrid(
    df,
    col="sex"
)

g.map_dataframe(
    sns.scatterplot,
    x="total_bill",
    y="tip"
)

plt.show()
```

This is useful for comparing groups independently.

---

# 📊 29. FacetGrid with Multiple Dimensions

```python
g = sns.FacetGrid(
    df,
    row="time",
    col="sex"
)

g.map_dataframe(
    sns.scatterplot,
    x="total_bill",
    y="tip"
)

plt.show()
```

This creates a multi-dimensional EDA visualization.

---

# 🤖 Seaborn for Machine Learning

Seaborn becomes especially valuable during the **Exploratory Data Analysis** stage of an ML project.

A typical workflow:

```text
Load Dataset
     ↓
Inspect Data
     ↓
Clean Data
     ↓
Analyze Missing Values
     ↓
Analyze Distributions
     ↓
Analyze Relationships
     ↓
Analyze Correlations
     ↓
Detect Outliers
     ↓
Study Target Variable
     ↓
Feature Engineering
     ↓
Model Training
```

---

# 🔍 EDA Workflow

## Step 1 — Load Data

```python
import pandas as pd

df = pd.read_csv("dataset.csv")
```

## Step 2 — Inspect

```python
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
```

## Step 3 — Missing Values

```python
sns.heatmap(
    df.isnull(),
    cbar=False
)

plt.show()
```

## Step 4 — Distributions

```python
sns.histplot(
    data=df,
    x="age",
    kde=True
)

plt.show()
```

## Step 5 — Relationships

```python
sns.scatterplot(
    data=df,
    x="experience",
    y="salary"
)

plt.show()
```

## Step 6 — Correlation

```python
correlation = df.select_dtypes(
    include="number"
).corr()

sns.heatmap(
    correlation,
    annot=True
)

plt.show()
```

---

# 🎯 Target Variable Analysis

Suppose a classification dataset contains:

```text
Purchased
0
1
```

Analyze class distribution:

```python
sns.countplot(
    data=df,
    x="Purchased"
)

plt.show()
```

This can help identify potential class imbalance.

Example:

```text
Class 0 → 900 samples
Class 1 → 100 samples
```

This dataset may require additional investigation before model training.

---

# 📊 Feature vs Target

For numerical features:

```python
sns.boxplot(
    data=df,
    x="Purchased",
    y="Age"
)

plt.show()
```

For another feature:

```python
sns.boxplot(
    data=df,
    x="Purchased",
    y="Salary"
)

plt.show()
```

These plots can help identify differences in feature distributions across target classes.

---

# 🔗 Feature Correlation

```python
numeric_features = df.select_dtypes(
    include="number"
)

correlation = numeric_features.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Matrix")

plt.show()
```

---

# 🚨 Outlier Detection

Box plots can help identify potential outliers.

```python
sns.boxplot(
    data=df,
    x="salary"
)

plt.show()
```

Potential outliers should be investigated rather than automatically removed.

Possible causes include:

```text
Data entry errors
Legitimate extreme values
Rare observations
Measurement errors
Different populations
```

---

# 🧠 Pair Plot for ML EDA

```python
sns.pairplot(
    df,
    hue="target"
)

plt.show()
```

This can help investigate:

* Feature relationships
* Class separation
* Potential clusters
* Strong correlations
* Distribution differences

---

# 📈 Model Training Visualization

Seaborn can also visualize model-training metrics.

Example:

```python
epochs = [1, 2, 3, 4, 5]

train_loss = [0.80, 0.60, 0.45, 0.35, 0.28]
val_loss = [0.85, 0.68, 0.52, 0.48, 0.50]

results = pd.DataFrame({
    "epoch": epochs,
    "train_loss": train_loss,
    "validation_loss": val_loss
})

sns.lineplot(
    data=results,
    x="epoch",
    y="train_loss",
    marker="o",
    label="Training Loss"
)

sns.lineplot(
    data=results,
    x="epoch",
    y="validation_loss",
    marker="o",
    label="Validation Loss"
)

plt.title("Training vs Validation Loss")
plt.show()
```

> The values above are illustrative examples, not results from a trained model.

---

# 📊 Model Evaluation Visualization

Example metrics:

```python
metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Score": [
        0.91,
        0.89,
        0.87,
        0.88
    ]
})
```

Visualize:

```python
sns.barplot(
    data=metrics,
    x="Metric",
    y="Score"
)

plt.ylim(0, 1)
plt.title("Model Evaluation Metrics")

plt.show()
```

Again, these are illustrative values.

---

# 🔥 Feature Importance Visualization

Suppose a model provides feature importance values:

```python
importance = pd.DataFrame({
    "Feature": [
        "Age",
        "Experience",
        "Salary",
        "Education"
    ],
    "Importance": [
        0.18,
        0.31,
        0.36,
        0.15
    ]
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)
```

Visualize:

```python
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.show()
```

> These values are illustrative. Actual feature importance must come from the trained model.

---

# 🧪 Seaborn + Pandas

Seaborn integrates naturally with Pandas.

```python
import pandas as pd
import seaborn as sns

df = pd.DataFrame({
    "Age": [21, 25, 30, 35, 40],
    "Salary": [25000, 35000, 45000, 55000, 70000]
})

sns.scatterplot(
    data=df,
    x="Age",
    y="Salary"
)
```

---

# 📐 Seaborn + Matplotlib

Seaborn handles statistical visualization while Matplotlib provides low-level figure control.

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Age",
    y="Salary",
    ax=ax
)

ax.set_title("Age vs Salary")
ax.set_xlabel("Age")
ax.set_ylabel("Salary")

plt.show()
```

This approach is recommended for reusable and professional visualization code.

---

# 💾 Saving Seaborn Figures

Because Seaborn uses Matplotlib underneath:

```python
plt.savefig(
    "plot.png",
    dpi=300,
    bbox_inches="tight"
)
```

For PDF:

```python
plt.savefig(
    "plot.pdf",
    bbox_inches="tight"
)
```

Professional recommendation:

```python
plt.savefig(
    "figure.png",
    dpi=300,
    bbox_inches="tight"
)
```

---

# 📁 Recommended Directory Structure

```text
04-Seaborn/
│
├── README.md
│
├── basic-plots.py
├── distribution-plots.py
├── categorical-plots.py
├── relational-plots.py
├── heatmap.py
├── pairplot.py
├── jointplot.py
├── styling.py
├── faceting.py
├── ml-visualization.py
└── eda-dashboard.py
```

---

# 🗺️ Seaborn Learning Roadmap

```text
                    Seaborn
                       │
          ┌────────────┴────────────┐
          │                         │
      Fundamentals               EDA
          │                         │
    ┌─────┼─────┐           ┌──────┼──────┐
    │     │     │           │      │      │
  Style  Data  Figure    Distributions Relationships
    │     │     │           │      │      │
    └─────┼─────┘           └──────┼──────┘
          │                        │
          └───────────┬────────────┘
                      │
                 Statistics
                      │
            ┌─────────┼─────────┐
            │         │         │
         Heatmap   Pairplot   Boxplot
            │         │         │
            └─────────┼─────────┘
                      │
                 ML Analysis
                      │
            ┌─────────┼─────────┐
            │         │         │
          Target   Features  Correlation
            │         │         │
            └─────────┼─────────┘
                      │
                ML Visualization
```

---

# 📚 Core Seaborn Functions

| Function            | Purpose                      |
| ------------------- | ---------------------------- |
| `sns.lineplot()`    | Line visualization           |
| `sns.scatterplot()` | Relationship visualization   |
| `sns.barplot()`     | Category comparison          |
| `sns.countplot()`   | Category frequency           |
| `sns.histplot()`    | Distribution                 |
| `sns.kdeplot()`     | Density estimation           |
| `sns.boxplot()`     | Distribution + outliers      |
| `sns.violinplot()`  | Distribution comparison      |
| `sns.stripplot()`   | Individual observations      |
| `sns.swarmplot()`   | Non-overlapping observations |
| `sns.regplot()`     | Regression relationship      |
| `sns.lmplot()`      | Regression visualization     |
| `sns.heatmap()`     | Matrix visualization         |
| `sns.pairplot()`    | Pairwise relationships       |
| `sns.jointplot()`   | Joint distributions          |
| `sns.catplot()`     | Categorical plots            |
| `sns.FacetGrid()`   | Multi-panel visualization    |
| `sns.boxenplot()`   | Distribution visualization   |
| `sns.pointplot()`   | Category estimates           |

---

# 🧾 Common Parameters

| Parameter | Purpose                |
| --------- | ---------------------- |
| `data`    | DataFrame or dataset   |
| `x`       | X-axis variable        |
| `y`       | Y-axis variable        |
| `hue`     | Color grouping         |
| `style`   | Marker/style grouping  |
| `size`    | Size encoding          |
| `palette` | Color palette          |
| `ax`      | Matplotlib Axes        |
| `kind`    | Plot type              |
| `bins`    | Histogram bins         |
| `kde`     | Add density curve      |
| `annot`   | Display heatmap values |
| `fmt`     | Annotation formatting  |
| `col`     | Facet columns          |
| `row`     | Facet rows             |

---

# ⚠️ Common Mistakes

## 1. Plotting without understanding the data

Bad workflow:

```text
Load Dataset
↓
Immediately Plot Everything
```

Better:

```text
Load
↓
Inspect
↓
Clean
↓
Understand
↓
Visualize
```

---

## 2. Ignoring missing values

Always inspect:

```python
print(df.isnull().sum())
```

---

## 3. Using inappropriate plots

Examples:

```text
Numerical distribution → Histogram/KDE
Two numerical variables → Scatter
Categorical frequency → Countplot
Categorical vs numerical → Boxplot/Violinplot
Correlation → Heatmap
Multiple numerical relationships → Pairplot
```

---

## 4. Overusing pair plots

Pair plots can become difficult to read with many features.

For example:

```text
5 features → manageable
20 features → potentially cluttered
100 features → inappropriate
```

Select relevant features when necessary.

---

## 5. Treating correlation as causation

A high correlation does not prove that one variable causes another.

---

## 6. Misinterpreting outliers

An outlier is not automatically an error.

Investigate the underlying data first.

---

## 7. Data Leakage

EDA can involve the target variable, but preprocessing decisions used for model training must be designed carefully.

For example, when building a production ML pipeline:

```text
Train Data
    ↓
Fit preprocessing
    ↓
Transform Train
    ↓
Transform Validation/Test
```

Do not use information from the test set to fit preprocessing transformations.

---

# 🧠 Professional Visualization Principles

### 1. Choose the right chart

```text
Trend          → Line Plot
Comparison     → Bar Plot
Distribution   → Histogram/KDE
Relationship   → Scatter Plot
Outliers       → Box Plot
Correlation    → Heatmap
Many Features  → Pair Plot
Categories     → Count/Box/Violin
```

### 2. Label everything

Always consider:

```text
Title
X-axis
Y-axis
Legend
Units
```

### 3. Avoid unnecessary decoration

Visualization should communicate information clearly.

### 4. Use consistent styling

Maintain consistent:

```text
Font sizes
Figure sizes
Palettes
Labels
Titles
```

### 5. Consider accessibility

Prefer palettes that remain distinguishable for users with color-vision deficiencies.

```python
sns.set_palette("colorblind")
```

---

# 🚀 Professional EDA Checklist

Before training an ML model, investigate:

```text
☐ Dataset dimensions
☐ Data types
☐ Missing values
☐ Duplicate records
☐ Numerical distributions
☐ Categorical distributions
☐ Target distribution
☐ Class imbalance
☐ Feature relationships
☐ Correlations
☐ Potential outliers
☐ Feature scales
☐ Suspicious values
☐ Potential leakage
```

---

# 🔬 Mini EDA Template

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("dataset.csv")


# Inspect
print(df.head())
print(df.info())
print(df.describe())


# Missing values
print(df.isnull().sum())


# Numerical features
numeric_df = df.select_dtypes(include="number")


# Correlation
correlation = numeric_df.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# Distribution
for column in numeric_df.columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()
```

---

# 🔄 Seaborn vs Matplotlib

| Feature               | Seaborn    | Matplotlib |
| --------------------- | ---------- | ---------- |
| Abstraction           | High-level | Low-level  |
| Statistical plots     | Excellent  | Manual     |
| Default aesthetics    | Strong     | Basic      |
| DataFrame integration | Excellent  | Good       |
| Customization         | High       | Very high  |
| EDA                   | Excellent  | Good       |
| Fine-grained control  | Good       | Excellent  |
| ML visualization      | Excellent  | Excellent  |

### Recommended approach

Use both:

```text
Seaborn
   ↓
Statistical Visualization
   ↓
Matplotlib
   ↓
Fine Customization
```

They are complementary libraries rather than strict alternatives.

---

# 🧰 Recommended Learning Order

```text
01. Seaborn Fundamentals
        ↓
02. Themes & Styling
        ↓
03. Relational Plots
        ↓
04. Distribution Plots
        ↓
05. Categorical Plots
        ↓
06. Regression Plots
        ↓
07. Heatmaps
        ↓
08. Pair Plots
        ↓
09. Joint Plots
        ↓
10. FacetGrid
        ↓
11. Advanced Customization
        ↓
12. Exploratory Data Analysis
        ↓
13. Machine Learning Visualization
        ↓
14. Professional Dashboards
```

---

# 🧪 Practice Projects

## Beginner

### Project 1 — Iris EDA

Analyze:

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
Species
```

Required visualizations:

* Histogram
* Scatter plot
* Box plot
* Pair plot
* Heatmap

---

## Intermediate

### Project 2 — Titanic EDA

Analyze:

```text
Age
Sex
Class
Fare
Survival
```

Create:

* Count plots
* Bar plots
* Box plots
* Violin plots
* Heatmap
* Group comparisons

---

## Advanced

### Project 3 — Customer Analytics

Analyze:

```text
Age
Income
Spending Score
Membership
Purchase Frequency
```

Build:

* Distribution dashboard
* Correlation heatmap
* Feature relationships
* Customer segmentation visualizations

---

## ML Project

### Project 4 — Model Analysis Dashboard

Build a visualization dashboard containing:

```text
Class Distribution
Feature Distribution
Correlation Matrix
Feature Importance
Prediction Distribution
Model Metrics
```

Use:

```text
Pandas
NumPy
Seaborn
Matplotlib
Scikit-learn
```

---

# 📖 Recommended Repository Structure

```text
03-Python-for-Machine-Learning/
│
├── 01-NumPy/
│
├── 02-Pandas/
│
├── 03-Matplotlib/
│
└── 04-Seaborn/
    │
    ├── README.md
    ├── basic-plots.py
    ├── distribution-plots.py
    ├── categorical-plots.py
    ├── relational-plots.py
    ├── heatmap.py
    ├── pairplot.py
    ├── jointplot.py
    ├── styling.py
    ├── faceting.py
    ├── ml-visualization.py
    └── eda-dashboard.py
```

---

# 🔗 Seaborn Ecosystem

Seaborn works particularly well with:

```text
NumPy
   ↓
Numerical Computing

Pandas
   ↓
Data Manipulation

Matplotlib
   ↓
Visualization Foundation

Seaborn
   ↓
Statistical Visualization

Scikit-learn
   ↓
Machine Learning
```

Together, these libraries form a major part of the Python Machine Learning ecosystem.

---

# ⚡ Quick Reference

```python
# Import
import seaborn as sns
import matplotlib.pyplot as plt

# Theme
sns.set_theme()

# Line
sns.lineplot(data=df, x="x", y="y")

# Scatter
sns.scatterplot(data=df, x="x", y="y")

# Bar
sns.barplot(data=df, x="category", y="value")

# Count
sns.countplot(data=df, x="category")

# Histogram
sns.histplot(data=df, x="value")

# KDE
sns.kdeplot(data=df, x="value")

# Box
sns.boxplot(data=df, x="category", y="value")

# Violin
sns.violinplot(data=df, x="category", y="value")

# Regression
sns.regplot(data=df, x="x", y="y")

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True)

# Pair plot
sns.pairplot(df)

# Joint plot
sns.jointplot(data=df, x="x", y="y")

# Save
plt.savefig("figure.png", dpi=300, bbox_inches="tight")

# Display
plt.show()
```

---

# 🎯 Key Takeaways

By completing this Seaborn module, you should understand how to:

* Create statistical visualizations
* Work with Pandas DataFrames
* Analyze numerical distributions
* Compare categorical variables
* Visualize feature relationships
* Detect potential outliers
* Analyze correlations
* Understand class distributions
* Perform visual EDA
* Visualize ML features and targets
* Analyze model metrics
* Create professional visualizations
* Combine Seaborn with Matplotlib
* Build reusable visualization workflows

The most important principle is:

> **Visualization is not decoration — it is a tool for understanding data and validating assumptions.**

---

## 👨‍💻 Author

**Kishor Patil**

Machine Learning • Python • Data Science • AI

GitHub:
`https://github.com/Kishor055`

---

## ⭐ Contributing

Contributions are welcome.

If you find an issue or have an improvement:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your examples
5. Commit your changes
6. Open a Pull Request

---

## 📄 License

This project is intended for educational and learning purposes.

See the repository license for complete terms.

---

## 🚀 Next Step

After completing Seaborn, continue with:

```text
Python for Machine Learning
        ↓
NumPy
        ↓
Pandas
        ↓
Matplotlib
        ↓
Seaborn
        ↓
Scikit-learn
        ↓
Machine Learning Algorithms
        ↓
Projects
        ↓
Deep Learning
```

**Keep learning. Keep building. Keep experimenting. 🚀**

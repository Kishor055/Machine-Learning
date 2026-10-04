
# 📈 Correlation Analysis

> **Correlation Analysis = Measure Relationships → Identify Patterns → Compare Variables → Detect Multicollinearity → Improve Features → Build Better Models**

![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-blue)
![EDA](https://img.shields.io/badge/Topic-Exploratory%20Data%20Analysis-orange)
![Python](https://img.shields.io/badge/Python-3.x-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple)
![Status](https://img.shields.io/badge/Status-Learning%20Module-success)

---

# 🌱 GROW → EXPLORE → MEASURE → INTERPRET → SELECT → BUILD 🚀

```text
                         📊 DATASET
                            │
                            ▼
                  🔍 EXPLORE VARIABLES
                            │
                            ▼
                  📈 CORRELATION ANALYSIS
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        COVARIANCE       PEARSON        SPEARMAN
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                  🔥 CORRELATION MATRIX
                            │
                            ▼
                     📊 HEATMAP
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
       STRONG RELATION   WEAK RELATION   NO RELATION
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                  🧠 FEATURE ANALYSIS
                            │
                            ▼
               ⚠️ MULTICOLLINEARITY
                            │
                            ▼
                    🤖 MACHINE LEARNING
```

---

## 📚 Table of Contents

1. [What Is Correlation Analysis?](#-what-is-correlation-analysis)
2. [Why Correlation Matters](#-why-correlation-matters)
3. [Correlation vs Causation](#-correlation-vs-causation)
4. [Types of Relationships](#-types-of-relationships)
5. [Correlation Coefficient](#-correlation-coefficient)
6. [Correlation Range](#-correlation-range)
7. [Positive Correlation](#-positive-correlation)
8. [Negative Correlation](#-negative-correlation)
9. [Zero or Weak Correlation](#-zero-or-weak-correlation)
10. [Covariance](#-covariance)
11. [Pearson Correlation](#-pearson-correlation)
12. [Spearman Rank Correlation](#-spearman-rank-correlation)
13. [Kendall Rank Correlation](#-kendall-rank-correlation)
14. [Pearson vs Spearman vs Kendall](#-pearson-vs-spearman-vs-kendall)
15. [Manual Correlation Calculation](#-manual-correlation-calculation)
16. [Correlation with Python](#-correlation-with-python)
17. [Correlation with NumPy](#-correlation-with-numpy)
18. [Correlation with Pandas](#-correlation-with-pandas)
19. [Correlation Matrix](#-correlation-matrix)
20. [Correlation Heatmap](#-correlation-heatmap)
21. [Scatter Plots](#-scatter-plots)
22. [Interpreting Correlation](#-interpreting-correlation)
23. [Correlation Strength](#-correlation-strength)
24. [Outliers and Correlation](#-outliers-and-correlation)
25. [Non-Linear Relationships](#-non-linear-relationships)
26. [Missing Values](#-missing-values)
27. [Categorical Variables](#-categorical-variables)
28. [Correlation with Target](#-correlation-with-target)
29. [Feature Selection](#-feature-selection)
30. [Multicollinearity](#-multicollinearity)
31. [Variance Inflation Factor](#-variance-inflation-factor)
32. [Correlation and Machine Learning](#-correlation-and-machine-learning)
33. [Correlation Does Not Mean Causation](#-correlation-does-not-mean-causation)
34. [Common Mistakes](#-common-mistakes)
35. [Best Practices](#-best-practices)
36. [Reusable Python Functions](#-reusable-python-functions)
37. [Automated Correlation Report](#-automated-correlation-report)
38. [End-to-End Example](#-end-to-end-example)
39. [Mini Projects](#-mini-projects)
40. [Exercises](#-exercises)
41. [Project Structure](#-project-structure)
42. [Correlation Analysis Checklist](#-correlation-analysis-checklist)
43. [EDA Roadmap](#-eda-roadmap)
44. [Key Takeaways](#-key-takeaways)
45. [Next Step](#-next-step)

---

# 🔍 What Is Correlation Analysis?

**Correlation analysis** is a statistical technique used to measure the **strength and direction of the relationship between variables**.

For example:

```text
Study Hours  ───────────────► Exam Score
```

If students who study more hours generally achieve higher scores, the variables may have a **positive correlation**.

Another example:

```text
Price  ───────────────► Demand
          decreases
```

If demand generally decreases as price increases, the variables may have a **negative correlation**.

Correlation is one of the most useful techniques in Exploratory Data Analysis because it helps us understand relationships before building a machine learning model.

---

# 🎯 Why Correlation Matters

Correlation analysis helps answer questions such as:

* Which variables move together?
* Which features are strongly related?
* Which features are related to the target?
* Are some features redundant?
* Are two variables approximately linearly related?
* Are there possible multicollinearity problems?
* Which features deserve further investigation?
* Are there suspicious relationships in the dataset?

A typical EDA workflow is:

```text
Dataset
   ↓
Understand Variables
   ↓
Clean Data
   ↓
Univariate Analysis
   ↓
Bivariate Analysis
   ↓
Correlation Analysis
   ↓
Feature Relationships
   ↓
Feature Selection
   ↓
Machine Learning
```

---

# ⚠️ Correlation vs Causation

This is one of the most important concepts in data science.

> **Correlation does not imply causation.**

Suppose:

```text
Ice Cream Sales ↑
      │
      │ correlation
      ▼
Swimming Accidents ↑
```

The two variables may be correlated, but ice cream sales do not necessarily cause swimming accidents.

A possible third variable is:

```text
             ☀️ Temperature
               /       \
              /         \
             ▼           ▼
      Ice Cream Sales   Swimming
                         Accidents
```

Temperature may influence both variables.

Therefore:

```text
Correlation ≠ Causation
```

Correlation identifies relationships. Establishing causation generally requires stronger evidence, domain knowledge, experiments, or causal inference methods.

---

# 📊 Types of Relationships

Two numerical variables can have several types of relationships.

## 1. Positive Relationship

```text
Y
│          •
│        •
│      •
│    •
│  •
└──────────────── X
```

As `X` increases, `Y` tends to increase.

---

## 2. Negative Relationship

```text
Y
│  •
│    •
│      •
│        •
│          •
└──────────────── X
```

As `X` increases, `Y` tends to decrease.

---

## 3. No Linear Relationship

```text
Y
│   •     •
│      •
│ •       •
│    •
│       •
└──────────────── X
```

There is little or no linear relationship.

---

## 4. Non-Linear Relationship

```text
Y
│      •••
│    •     •
│   •       •
│    •     •
│      •••
└──────────────── X
```

Two variables can have a strong relationship even when Pearson correlation is close to zero.

This is why correlation should be combined with visualization.

---

# 📐 Correlation Coefficient

A **correlation coefficient** summarizes the direction and strength of a relationship.

The most common coefficient is **Pearson's correlation coefficient**, represented by:

```text
r
```

Its value lies between:

```text
-1 ≤ r ≤ +1
```

Interpretation:

|      Correlation | General Interpretation        |
| ---------------: | ----------------------------- |
|             `+1` | Perfect positive relationship |
| `+0.8` to `+1.0` | Very strong positive          |
| `+0.6` to `+0.8` | Strong positive               |
| `+0.3` to `+0.6` | Moderate positive             |
|    `0` to `+0.3` | Weak positive                 |
|              `0` | No linear correlation         |
|    `-0.3` to `0` | Weak negative                 |
| `-0.6` to `-0.3` | Moderate negative             |
| `-0.8` to `-0.6` | Strong negative               |
| `-1.0` to `-0.8` | Very strong negative          |
|             `-1` | Perfect negative              |

> These ranges are practical guidelines, not universal scientific thresholds. Interpretation depends on the domain.

---

# ➕ Positive Correlation

Positive correlation means that two variables generally move in the same direction.

Example:

```text
Experience ↑
     ↓
Salary ↑
```

Possible correlation:

```text
r = +0.82
```

This indicates a strong positive linear relationship.

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "experience": [1, 2, 3, 4, 5],
    "salary": [30, 35, 42, 50, 60]
})

correlation = df["experience"].corr(df["salary"])

print(correlation)
```

---

# ➖ Negative Correlation

Negative correlation means that variables generally move in opposite directions.

Example:

```text
Price ↑
  ↓
Demand ↓
```

Possible correlation:

```text
r = -0.78
```

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "price": [10, 20, 30, 40, 50],
    "demand": [100, 85, 70, 50, 30]
})

print(df["price"].corr(df["demand"]))
```

---

# ⚪ Zero or Weak Correlation

A correlation close to zero means there is little evidence of a **linear** relationship.

```text
r ≈ 0
```

However:

> `r ≈ 0` does not necessarily mean that no relationship exists.

The relationship may be non-linear.

---

# 🔢 Covariance

Covariance measures how two variables change together.

For a sample:

$$
Cov(X,Y)=\frac{\sum_{i=1}^{n}(X_i-\bar X)(Y_i-\bar Y)}{n-1}
$$

Interpretation:

```text
Cov(X,Y) > 0
    → Variables tend to increase together

Cov(X,Y) < 0
    → One tends to increase when the other decreases

Cov(X,Y) ≈ 0
    → Little linear co-movement
```

The major limitation of covariance is that its magnitude depends on the units of the variables.

For example:

```text
Income measured in ₹
Income measured in lakh ₹
```

will produce different covariance magnitudes.

Correlation solves much of this problem by standardizing covariance.

---

# 📏 Pearson Correlation

Pearson correlation measures the strength and direction of a **linear relationship** between two numerical variables.

The formula is:

$$
r =
\frac{
\sum (x_i-\bar{x})(y_i-\bar{y})
}{
\sqrt{
\sum(x_i-\bar{x})^2
\sum(y_i-\bar{y})^2
}
}
$$

An equivalent relationship is:

$$
r =
\frac{Cov(X,Y)}{\sigma_X\sigma_Y}
$$

where:

* `Cov(X,Y)` = covariance
* `σX` = standard deviation of X
* `σY` = standard deviation of Y

---

# 🧮 Manual Correlation Calculation

Consider:

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]
```

A simple implementation:

```python
import math


def pearson_correlation(x, y):
    if len(x) != len(y):
        raise ValueError("Both variables must have the same length.")

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)

    numerator = sum(
        (xi - x_mean) * (yi - y_mean)
        for xi, yi in zip(x, y)
    )

    x_squared = sum(
        (xi - x_mean) ** 2
        for xi in x
    )

    y_squared = sum(
        (yi - y_mean) ** 2
        for yi in y
    )

    denominator = math.sqrt(x_squared * y_squared)

    if denominator == 0:
        raise ValueError("Correlation is undefined for a constant variable.")

    return numerator / denominator


x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]

print(pearson_correlation(x, y))
```

This implementation is useful for understanding the mathematics behind correlation.

For production analysis, use tested statistical libraries.

---

# 🐍 Correlation with Python

## Using Pandas

```python
import pandas as pd

df = pd.DataFrame({
    "hours": [1, 2, 3, 4, 5],
    "score": [45, 50, 62, 72, 85]
})

correlation = df["hours"].corr(df["score"])

print(f"Correlation: {correlation:.3f}")
```

---

# 🔢 Correlation with NumPy

```python
import numpy as np

x = [1, 2, 3, 4, 5]
y = [45, 50, 62, 72, 85]

matrix = np.corrcoef(x, y)

print(matrix)
```

The resulting matrix contains:

```text
          X       Y
X       1.0       r
Y         r     1.0
```

---

# 🐼 Correlation with Pandas

Pandas provides a convenient way to calculate correlations between numerical columns.

```python
correlation_matrix = df.corr(numeric_only=True)

print(correlation_matrix)
```

Example:

```text
          age  income  score
age      1.00    0.72   0.31
income   0.72    1.00   0.65
score    0.31    0.65   1.00
```

The diagonal is always:

```text
1.0
```

because every variable is perfectly correlated with itself.

---

# 🧮 Spearman Rank Correlation

Spearman correlation measures the relationship between the **ranks** of variables.

It is especially useful when:

* The relationship is monotonic rather than linear.
* Variables contain outliers.
* Data is ordinal.
* The assumptions of Pearson correlation are not appropriate.

Example:

```python
from scipy.stats import spearmanr

x = [1, 2, 3, 4, 5]
y = [10, 20, 30, 80, 100]

correlation, p_value = spearmanr(x, y)

print("Correlation:", correlation)
print("p-value:", p_value)
```

Pandas:

```python
correlation = df.corr(
    method="spearman",
    numeric_only=True
)
```

---

# 🏆 Kendall Rank Correlation

Kendall's tau is another rank-based correlation measure.

It evaluates the agreement between pairs of observations.

```python
from scipy.stats import kendalltau

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 4, 8]

tau, p_value = kendalltau(x, y)

print("Kendall tau:", tau)
print("p-value:", p_value)
```

Kendall correlation can be particularly useful for:

* Ordinal data
* Ranking problems
* Smaller datasets
* Situations with many ties

---

# ⚖️ Pearson vs Spearman vs Kendall

| Method     | Measures                    | Best For                       |
| ---------- | --------------------------- | ------------------------------ |
| Pearson    | Linear relationship         | Continuous numerical variables |
| Spearman   | Monotonic rank relationship | Ordinal/non-normal data        |
| Kendall    | Rank concordance            | Ordinal/ranking data           |
| Covariance | Joint variation             | Statistical foundations        |

A practical decision process:

```text
Are variables numerical?
        │
        ├── No → Consider other association measures
        │
        ▼
Is the relationship approximately linear?
        │
        ├── Yes → Pearson
        │
        └── No
             │
             ▼
      Is it monotonic?
             │
             ├── Yes → Spearman
             │
             └── Ranking/ordinal → Kendall
```

---

# 🔥 Correlation Matrix

A correlation matrix contains correlations between multiple variables.

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "age": [20, 22, 25, 28, 30],
    "experience": [0, 1, 3, 5, 7],
    "salary": [25, 30, 40, 55, 70],
    "score": [60, 65, 72, 80, 88]
})

matrix = df.corr(numeric_only=True)

print(matrix)
```

Example output:

```text
              age  experience  salary  score
age          1.00        0.98    0.97   0.95
experience   0.98        1.00    0.99   0.96
salary       0.97        0.99    1.00   0.94
score        0.95        0.96    0.94   1.00
```

This immediately reveals groups of strongly related variables.

---

# 🌡️ Correlation Heatmap

A heatmap provides a visual representation of the correlation matrix.

```python
import matplotlib.pyplot as plt
import seaborn as sns

correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
```

A heatmap makes it easier to identify:

```text
Strong positive  → values close to +1
Strong negative  → values close to -1
Weak relationship → values close to 0
```

---

# 📊 Scatter Plots

Correlation should not be interpreted without visual inspection.

Use a scatter plot to examine the actual relationship.

```python
import matplotlib.pyplot as plt

plt.scatter(df["experience"], df["salary"])

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")

plt.show()
```

A scatter plot can reveal:

* Linear patterns
* Non-linear patterns
* Clusters
* Outliers
* Heteroscedasticity
* Data-entry errors

---

# 🧠 Interactive Correlation Visualization

Explore how the direction and strength of a correlation change as data points move:

genui{"learning_viz":{"type_id":"CORRELATION","initial_values":{"pattern":"positive"}}}

This is useful for developing intuition before relying on numerical correlation coefficients.

---

# 📈 Interpreting Correlation

Suppose:

```text
Study Hours ↔ Exam Score

r = 0.91
```

This indicates:

```text
Strong positive linear association
```

But it does **not** prove:

```text
Studying causes higher scores.
```

A responsible interpretation is:

> Students with more study hours tend to have higher exam scores in this dataset.

Avoid saying:

> Studying more causes students to score higher.

unless causal evidence supports that conclusion.

---

# 💪 Correlation Strength

A practical interpretation:

```text
        -1                     0                     +1
         │                     │                      │
         │                     │                      │
    Strong Negative       No Linear             Strong Positive
         │                 Relationship               │
         ▼                     ▼                      ▼

       -0.9                  0.0                   +0.9
```

Example:

|     `r` | Interpretation         |
| ------: | ---------------------- |
|  `0.95` | Very strong positive   |
|  `0.75` | Strong positive        |
|  `0.45` | Moderate positive      |
|  `0.15` | Weak positive          |
|  `0.00` | No linear relationship |
| `-0.20` | Weak negative          |
| `-0.55` | Moderate negative      |
| `-0.80` | Strong negative        |
| `-0.97` | Very strong negative   |

Always interpret correlation in context.

---

# 🚨 Outliers and Correlation

Outliers can significantly change correlation.

Consider:

```text
Normal Data

     •
   • •
  • •
 •
```

Now introduce one extreme point:

```text
                     •
     •
   • •
  • •
 •
```

The correlation can change substantially.

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 100],
    "y": [2, 4, 6, 8, 10, 5]
})

print(df["x"].corr(df["y"]))
```

Therefore:

```text
Correlation
     ↓
Check Scatter Plot
     ↓
Check Outliers
     ↓
Interpret Carefully
```

Do not automatically remove an outlier merely because it changes correlation. First determine whether it is a valid observation.

---

# 🔄 Non-Linear Relationships

Pearson correlation primarily captures linear relationships.

Consider:

$$
Y=X^2
$$

Example:

```python
import numpy as np
import pandas as pd

x = np.arange(-10, 11)
y = x ** 2

df = pd.DataFrame({
    "x": x,
    "y": y
})

print(df["x"].corr(df["y"]))
```

The Pearson correlation may be close to zero even though the variables have a very clear mathematical relationship.

Therefore:

```text
Low Pearson correlation
        ≠
No relationship
```

Always visualize important relationships.

---

# 🕳️ Missing Values

Missing observations affect correlation calculations.

Pandas generally performs pairwise calculations while handling missing values.

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "x": [1, 2, 3, None, 5],
    "y": [2, 4, 6, 8, 10]
})

print(df["x"].corr(df["y"]))
```

Before interpreting the result, inspect:

```python
print(df.isna().sum())
```

Important questions:

* How many observations remain?
* Is missingness random?
* Could missingness introduce bias?
* Are enough observations available?

---

# 🏷️ Categorical Variables

Standard Pearson correlation is designed for numerical variables.

Do not blindly convert categories to numbers:

```text
Red   → 1
Blue  → 2
Green → 3
```

and then calculate Pearson correlation.

Those numerical labels may not represent meaningful distances.

Depending on the data, consider:

* Point-biserial correlation
* Spearman correlation for ordinal variables
* Chi-square tests
* Cramér's V
* Mutual information
* ANOVA
* Domain-specific association measures

The appropriate method depends on variable types and the question being asked.

---

# 🎯 Correlation with Target

Correlation can help identify numerical features related to a numerical target.

Example:

```python
target_correlation = (
    df.corr(numeric_only=True)["salary"]
      .sort_values(ascending=False)
)

print(target_correlation)
```

For example:

```text
experience    0.91
education     0.72
age           0.64
hours         0.35
```

This can help prioritize further analysis.

However:

> Do not use correlation alone as a complete feature-selection strategy.

A feature with low linear correlation can still be useful to a machine learning model.

---

# 🎯 Selecting Correlations with a Target

```python
correlations = (
    df.corr(numeric_only=True)["target"]
      .drop("target")
      .sort_values(key=abs, ascending=False)
)

print(correlations)
```

This ranks features according to the absolute magnitude of their correlation with the target.

Example:

```text
feature_3     0.91
feature_7    -0.84
feature_1     0.65
feature_4     0.22
feature_2     0.03
```

Both:

```text
+0.91
-0.84
```

represent strong relationships.

---

# 🧹 Feature Selection

Correlation can help identify redundant features.

Suppose:

```text
age             ↔ experience     0.91
experience      ↔ salary         0.88
age             ↔ salary         0.82
```

Several features contain overlapping information.

A possible strategy is:

```text
Calculate Correlation Matrix
          ↓
Find Highly Correlated Features
          ↓
Understand Their Meaning
          ↓
Check Model Requirements
          ↓
Remove / Combine / Keep
```

Do not automatically remove every highly correlated feature.

Tree-based models, regularized models, domain requirements, and interpretability goals can change the decision.

---

# ⚠️ Multicollinearity

**Multicollinearity** occurs when predictor variables are highly related to one another.

Example:

```text
Feature A ─────────┐
                   ├──► Model
Feature B ─────────┘
     ↑
Highly correlated
```

For linear models, multicollinearity can cause:

* Unstable coefficients
* Large coefficient uncertainty
* Difficult interpretation
* Redundant information
* Numerical instability in some settings

Correlation analysis is often an early warning signal.

---

# 📊 Detecting Highly Correlated Features

```python
import numpy as np

corr_matrix = df.corr(numeric_only=True).abs()

upper_triangle = corr_matrix.where(
    np.triu(
        np.ones(corr_matrix.shape),
        k=1
    ).astype(bool)
)

highly_correlated = [
    column
    for column in upper_triangle.columns
    if any(upper_triangle[column] > 0.90)
]

print(highly_correlated)
```

This identifies columns with absolute correlation greater than `0.90`.

The threshold should be chosen based on the problem rather than treated as a universal rule.

---

# 📐 Variance Inflation Factor

Correlation between pairs of variables is useful, but multicollinearity can involve combinations of several variables.

**Variance Inflation Factor (VIF)** is another diagnostic.

Conceptually:

$$
VIF_j = \frac{1}{1-R_j^2}
$$

where `R²` comes from predicting feature `j` using the other predictor variables.

Typical interpretation:

|    VIF | Possible Interpretation  |
| -----: | ------------------------ |
|  `≈ 1` | Little multicollinearity |
|  `1–5` | Usually manageable       |
|  `> 5` | Potential concern        |
| `> 10` | Often considered serious |

These are rules of thumb, not strict laws.

Example:

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = df[["age", "experience", "education_years"]].dropna()

vif = []

for i in range(X.shape[1]):
    vif.append(
        variance_inflation_factor(X.values, i)
    )

print(vif)
```

---

# 🤖 Correlation and Machine Learning

Correlation analysis can support machine learning workflows.

## Regression

Useful for:

* Understanding numerical predictors
* Detecting redundancy
* Exploring relationships with the target
* Investigating multicollinearity

## Classification

Correlation can help with numerical features, but target relationships may require other techniques.

Consider:

* Mutual information
* ANOVA
* Chi-square
* Tree-based feature importance
* Permutation importance

## Unsupervised Learning

Correlation can help identify redundant features before:

* PCA
* Clustering
* Dimensionality reduction

---

# 🔐 Correlation and Data Leakage

Be careful when calculating target correlations during model development.

For example:

```text
Full Dataset
     ↓
Correlation Analysis
     ↓
Feature Selection
     ↓
Train/Test Split
```

This can leak information from the validation/test set into the modeling process.

A safer workflow is:

```text
Dataset
   ↓
Train/Test Split
   ↓
Analyze Training Data
   ↓
Select Features
   ↓
Fit Preprocessing
   ↓
Evaluate on Test Data
```

For final model evaluation, the test set should remain untouched until the appropriate evaluation stage.

---

# 🧪 Reusable Python Functions

## Calculate Pairwise Correlation

```python
def calculate_correlation(
    df,
    column_x,
    column_y,
    method="pearson"
):
    return df[column_x].corr(
        df[column_y],
        method=method
    )
```

Usage:

```python
correlation = calculate_correlation(
    df,
    "experience",
    "salary"
)

print(f"Correlation: {correlation:.3f}")
```

---

## Create Correlation Matrix

```python
def correlation_matrix(df, method="pearson"):
    return df.corr(
        method=method,
        numeric_only=True
    )
```

---

## Find Highly Correlated Features

```python
import numpy as np


def find_high_correlations(
    df,
    threshold=0.90
):
    matrix = df.corr(
        numeric_only=True
    ).abs()

    upper = matrix.where(
        np.triu(
            np.ones(matrix.shape),
            k=1
        ).astype(bool)
    )

    pairs = []

    for column in upper.columns:
        for row in upper.index:
            value = upper.loc[row, column]

            if pd.notna(value) and value >= threshold:
                pairs.append(
                    (row, column, value)
                )

    return sorted(
        pairs,
        key=lambda item: item[2],
        reverse=True
    )
```

Remember to import Pandas:

```python
import pandas as pd
```

---

# 📋 Automated Correlation Report

A reusable report can summarize relationships quickly.

```python
import pandas as pd


def correlation_report(df):
    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return pd.DataFrame()

    matrix = numeric_df.corr()

    rows = []

    for column in matrix.columns:
        for other in matrix.columns:
            if column >= other:
                continue

            value = matrix.loc[column, other]

            rows.append({
                "feature_1": column,
                "feature_2": other,
                "correlation": value,
                "absolute_correlation": abs(value)
            })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
        .reset_index(drop=True)
    )
```

Usage:

```python
report = correlation_report(df)

print(report.head(10))
```

This produces a table that is easier to inspect than a large correlation matrix.

---

# 📊 Complete Visualization Workflow

```python
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_analysis(df):
    numeric_df = df.select_dtypes(
        include="number"
    )

    correlation = numeric_df.corr()

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="coolwarm"
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
```

Usage:

```python
plot_correlation_analysis(df)
```

---

# 🚀 End-to-End Example

The following example demonstrates a complete correlation-analysis workflow.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_dataset():
    rng = np.random.default_rng(42)

    experience = np.arange(1, 11)

    salary = (
        30000
        + experience * 6000
        + rng.normal(0, 3500, 10)
    )

    age = (
        21
        + experience
        + rng.normal(0, 1.2, 10)
    )

    performance = (
        55
        + experience * 3
        + rng.normal(0, 4, 10)
    )

    return pd.DataFrame({
        "experience": experience,
        "age": age,
        "salary": salary,
        "performance": performance
    })


def show_correlation_matrix(df):
    correlation = df.corr(
        numeric_only=True
    )

    print("\nCorrelation Matrix:")
    print(correlation.round(2))

    return correlation


def show_target_correlations(df, target):
    correlations = (
        df.corr(numeric_only=True)[target]
        .drop(target)
        .sort_values(
            key=abs,
            ascending=False
        )
    )

    print(f"\nCorrelation with '{target}':")
    print(correlations.round(3))

    return correlations


def plot_heatmap(df):
    correlation = df.corr(
        numeric_only=True
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_scatter(df):
    sns.scatterplot(
        data=df,
        x="experience",
        y="salary"
    )

    plt.title("Experience vs Salary")
    plt.tight_layout()
    plt.show()


def main():
    df = create_dataset()

    print("Dataset:")
    print(df)

    show_correlation_matrix = show_correlation_matrix
    show_correlation_matrix(df)

    show_target_correlations(
        df,
        target="salary"
    )

    plot_heatmap(df)
    plot_scatter(df)


if __name__ == "__main__":
    main()
```

### Workflow

```text
Create Dataset
      ↓
Inspect Variables
      ↓
Select Numerical Columns
      ↓
Calculate Correlation Matrix
      ↓
Analyze Target Correlations
      ↓
Visualize Heatmap
      ↓
Inspect Scatter Plots
      ↓
Investigate Strong Relationships
      ↓
Check Multicollinearity
      ↓
Make Modeling Decisions
```

> **Implementation note:** In the example above, `show_correlation_matrix = show_correlation_matrix` is unnecessary and can be removed. A cleaner `main()` is shown below.

```python
def main():
    df = create_dataset()

    print("Dataset:")
    print(df)

    show_correlation_matrix(df)

    show_target_correlations(
        df,
        target="salary"
    )

    plot_heatmap(df)
    plot_scatter(df)


if __name__ == "__main__":
    main()
```

---

# 🧩 Correlation Analysis Decision Guide

```text
                 START
                   │
                   ▼
          What are the variables?
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
    Numerical             Categorical
        │                     │
        ▼                     ▼
 Is relationship          Use suitable
 approximately linear?    association method
        │
   ┌────┴────┐
   ▼         ▼
  YES        NO
   │          │
   ▼          ▼
Pearson    Is it monotonic?
              │
         ┌────┴────┐
         ▼         ▼
        YES        NO
         │          │
         ▼          ▼
     Spearman   Visualize +
                other methods
```

---

# ❌ Common Mistakes

## 1. Assuming Correlation Means Causation

Wrong:

```text
A correlates with B
        ↓
A causes B
```

Correct:

```text
A correlates with B
        ↓
Investigate further
```

---

## 2. Looking Only at the Correlation Number

A coefficient does not tell the entire story.

Always combine:

```text
Correlation
+
Scatter Plot
+
Domain Knowledge
```

---

## 3. Ignoring Outliers

One extreme observation can substantially change Pearson correlation.

---

## 4. Assuming Zero Correlation Means No Relationship

Non-linear relationships may have weak Pearson correlation.

---

## 5. Treating Correlation Thresholds as Universal

A correlation of `0.5` can mean different things in different domains.

---

## 6. Encoding Nominal Categories as Numbers

Do not assume:

```text
Apple = 1
Banana = 2
Orange = 3
```

creates meaningful numerical distances.

---

## 7. Removing Highly Correlated Features Automatically

High feature-feature correlation can be a warning sign, but the correct action depends on:

* Model type
* Interpretability
* Domain knowledge
* Feature meaning
* Regularization
* Dataset size

---

## 8. Ignoring Missing Values

Always inspect the number of observations used to calculate the relationship.

---

## 9. Performing Feature Selection Before Splitting Data

This can introduce data leakage.

---

## 10. Using Pearson for Every Problem

Different relationships require different statistical methods.

---

# ✅ Best Practices

### 1. Understand the Variables First

Know:

* Data types
* Units
* Measurement scale
* Domain meaning

### 2. Visualize Before Interpreting

Use:

```text
Scatter Plot
+
Correlation Coefficient
```

### 3. Check Multiple Correlation Methods

For example:

```python
df.corr(method="pearson")
df.corr(method="spearman")
```

### 4. Investigate Outliers

Do not remove valid observations just to increase correlation.

### 5. Consider Non-Linear Relationships

Use visualizations and appropriate modeling techniques.

### 6. Separate Feature-Feature and Feature-Target Analysis

They answer different questions.

### 7. Avoid Leakage

Perform modeling-related feature selection using training data only.

### 8. Use Domain Knowledge

Statistics should support—not replace—domain understanding.

### 9. Document Decisions

Record:

```text
Feature
Correlation
Method
Reason
Decision
```

### 10. Correlation Is a Starting Point

Correlation should lead to further investigation rather than automatically determine modeling decisions.

---

# 🧪 Mini Projects

## 🏠 Project 1 — House Price Correlation

Analyze relationships between:

* Area
* Bedrooms
* Bathrooms
* Age
* Location score
* Price

Questions:

* Which feature has the strongest correlation with price?
* Which features are highly correlated with each other?
* Are there outliers?
* Does a heatmap reveal redundant features?

---

## 👨‍💼 Project 2 — Employee Salary Analysis

Variables:

```text
Age
Experience
Education
Performance
Salary
```

Analyze:

* Experience vs salary
* Age vs salary
* Performance vs salary
* Feature-feature correlations
* Potential multicollinearity

---

## 🛒 Project 3 — E-Commerce Analysis

Analyze:

```text
Price
Discount
Quantity
Revenue
Customer Rating
```

Questions:

* Does discount correlate with quantity?
* Does rating correlate with revenue?
* Which numerical features are redundant?

---

## 🎓 Project 4 — Student Performance

Variables:

```text
Study Hours
Attendance
Assignments
Sleep Hours
Previous Score
Final Score
```

Investigate:

* Strongest target relationships
* Negative relationships
* Outliers
* Non-linear patterns
* Feature redundancy

---

# 📝 Exercises

## 🌱 Beginner

1. Create two numerical lists and calculate Pearson correlation manually.
2. Calculate correlation using Pandas.
3. Create a correlation matrix.
4. Interpret `r = 0.82`.
5. Interpret `r = -0.73`.
6. Create a scatter plot.
7. Identify positive and negative relationships.
8. Explain correlation vs causation.

---

## 🚀 Intermediate

1. Compare Pearson and Spearman correlation.
2. Build a correlation heatmap.
3. Find the top five correlations with a target.
4. Identify highly correlated feature pairs.
5. Investigate the effect of an outlier.
6. Analyze missing values before calculating correlations.
7. Find correlations in a real-world dataset.
8. Compare correlation before and after removing an invalid data point.

---

## 🧠 Advanced

1. Calculate VIF for a regression dataset.
2. Investigate multicollinearity.
3. Compare Pearson, Spearman, and Kendall.
4. Find non-linear relationships that Pearson misses.
5. Build an automated correlation report.
6. Create a feature-selection workflow.
7. Study correlation stability across train/test splits.
8. Investigate how correlation changes across subgroups.
9. Compare correlation-based selection with mutual information.
10. Study correlation drift over time.

---

# 📁 Project Structure

```text
06-Exploratory-Data-Analysis/
│
├── README.md
│
├── 01-Univariate-Analysis/
│   └── README.md
│
├── 02-Bivariate-Analysis/
│   └── README.md
│
├── 03-Multivariate-Analysis/
│   └── README.md
│
├── 04-Descriptive-Statistics/
│   └── README.md
│
├── 05-Correlation-Analysis/
│   ├── README.md
│   ├── pearson-correlation.py
│   ├── spearman-correlation.py
│   ├── correlation-matrix.py
│   ├── correlation-heatmap.py
│   ├── target-correlation.py
│   ├── multicollinearity.py
│   └── correlation-report.py
│
└── 06-Distribution-Analysis/
    └── README.md
```

---

# 🔎 Correlation Analysis Checklist

Before finalizing your analysis:

```text
☐ Understand variable types
☐ Inspect missing values
☐ Inspect outliers
☐ Select appropriate variables
☐ Choose an appropriate correlation method
☐ Calculate pairwise correlations
☐ Generate correlation matrix
☐ Visualize important relationships
☐ Investigate strong correlations
☐ Investigate unexpected correlations
☐ Check for non-linear patterns
☐ Check feature multicollinearity
☐ Analyze target relationships
☐ Avoid data leakage
☐ Use domain knowledge
☐ Document decisions
```

---

# 🗺️ EDA Roadmap

```text
06-Exploratory-Data-Analysis/
│
├── 01-Univariate-Analysis
│       ↓
├── 02-Bivariate-Analysis
│       ↓
├── 03-Multivariate-Analysis
│       ↓
├── 04-Descriptive-Statistics
│       ↓
├── 05-Correlation-Analysis
│       ↓
├── 06-Distribution-Analysis
│       ↓
├── 07-Outlier-Analysis
│       ↓
├── 08-Data-Visualization
│       ↓
└── 09-EDA-Projects
```

---

# 🧠 Key Takeaways

```text
📌 Correlation measures association between variables.

📌 Pearson measures linear correlation.

📌 Spearman measures monotonic rank correlation.

📌 Kendall measures rank concordance.

📌 Correlation ranges from -1 to +1.

📌 Positive correlation means variables tend to move together.

📌 Negative correlation means variables tend to move in opposite directions.

📌 Correlation does not prove causation.

📌 Outliers can strongly affect correlation.

📌 Zero Pearson correlation does not rule out non-linear relationships.

📌 Correlation matrices summarize many relationships.

📌 Heatmaps make correlation patterns easier to inspect.

📌 Highly correlated predictors can indicate multicollinearity.

📌 Correlation can support feature selection but should not be the only method.

📌 Always combine statistical results with visualization and domain knowledge.
```

---

# 🚀 Next Step

After understanding relationships between variables, the next important question is:

> **How are individual variables distributed?**

Move to:

```text
06-Exploratory-Data-Analysis/
└── 06-Distribution-Analysis/
    └── README.md
```

There you will learn:

```text
Distribution
   ↓
Shape
   ↓
Center
   ↓
Spread
   ↓
Skewness
   ↓
Kurtosis
   ↓
Normal Distribution
   ↓
Histograms
   ↓
KDE
   ↓
Q-Q Plots
   ↓
Outlier Analysis
   ↓
Machine Learning
```

---

# 🌟 Final Thought

> **Correlation helps you discover how variables move together—but good data science begins when you ask why.**

A strong correlation analysis combines:

```text
📊 Statistics
      +
📈 Visualization
      +
🧠 Domain Knowledge
      +
🧪 Statistical Testing
      +
🤖 Machine Learning
```

That combination transforms a simple correlation matrix into meaningful data understanding.

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

🔗 GitHub: [Kishor055](https://github.com/Kishor055)

---

# 🤝 Contributing

Contributions are welcome!

If you find:

* A mistake
* An unclear explanation
* A better implementation
* A missing example
* A useful visualization
* An additional exercise

feel free to open an issue or submit a pull request.

### Contribution Workflow

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/correlation-analysis

git add .

git commit -m "Improve correlation analysis module"

git push origin feature/correlation-analysis
```

Then open a Pull Request.

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples
* 📢 Share it with other learners

---

# 📄 License

This project is intended for **educational and learning purposes**.

See the repository license for complete terms.

---

<div align="center">

### 🌱 Learn → Analyze → Experiment → Build → Improve 🚀

**Keep learning. Keep experimenting. Keep building.**

</div>

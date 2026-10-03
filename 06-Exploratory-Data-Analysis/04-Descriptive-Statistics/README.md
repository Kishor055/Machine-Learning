# 📊 Descriptive Statistics

> **Descriptive Statistics = Summarize → Measure → Understand → Communicate Data**

Descriptive statistics is a fundamental part of **Exploratory Data Analysis (EDA)**.

It provides mathematical techniques for summarizing and describing the important characteristics of a dataset.

Instead of examining every observation individually, descriptive statistics helps answer questions such as:

* What is the typical value?
* How spread out is the data?
* What is the minimum and maximum?
* Where is the middle of the distribution?
* How variable is the dataset?
* Is the distribution skewed?
* Are there potential outliers?
* How do different groups compare?

# 🌱 GROW → SUMMARIZE → MEASURE → COMPARE → INTERPRET → DISCOVER → BUILD 🚀

```text
                         📊 DATA
                            │
                            ↓
                  ┌───────────────────┐
                  │ DESCRIPTIVE       │
                  │    STATISTICS     │
                  └─────────┬─────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ↓                 ↓                 ↓
       CENTER            SPREAD           SHAPE
          │                 │                 │
          ↓                 ↓                 ↓
       Mean              Range           Skewness
       Median            Variance        Kurtosis
       Mode              Std Dev
          │                 │
          └─────────────────┼─────────────────┘
                            ↓
                       DISTRIBUTION
                            ↓
                       INTERPRET
                            ↓
                      EDA → ML → BUILD
```

---

# 📚 Table of Contents

1. [What Is Descriptive Statistics?](#-what-is-descriptive-statistics)
2. [Why Descriptive Statistics Matters](#-why-descriptive-statistics-matters)
3. [Descriptive vs Inferential Statistics](#-descriptive-vs-inferential-statistics)
4. [Population vs Sample](#-population-vs-sample)
5. [Types of Data](#-types-of-data)
6. [Measures of Central Tendency](#-measures-of-central-tendency)
7. [Mean](#-mean)
8. [Weighted Mean](#-weighted-mean)
9. [Median](#-median)
10. [Mode](#-mode)
11. [Mean vs Median vs Mode](#-mean-vs-median-vs-mode)
12. [Measures of Dispersion](#-measures-of-dispersion)
13. [Range](#-range)
14. [Variance](#-variance)
15. [Standard Deviation](#-standard-deviation)
16. [Quartiles](#-quartiles)
17. [Percentiles](#-percentiles)
18. [Interquartile Range](#-interquartile-range)
19. [Five-Number Summary](#-five-number-summary)
20. [Measures of Position](#-measures-of-position)
21. [Z-Score](#-z-score)
22. [Distribution Shape](#-distribution-shape)
23. [Skewness](#-skewness)
24. [Kurtosis](#-kurtosis)
25. [Normal Distribution](#-normal-distribution)
26. [Empirical Rule](#-empirical-rule)
27. [Outliers](#-outliers)
28. [Descriptive Statistics With Python](#-descriptive-statistics-with-python)
29. [Python Built-in Statistics](#-python-built-in-statistics)
30. [NumPy Statistics](#-numpy-statistics)
31. [Pandas Statistics](#-pandas-statistics)
32. [DataFrame describe()](#-dataframe-describe)
33. [Grouped Descriptive Statistics](#-grouped-descriptive-statistics)
34. [Categorical Descriptive Statistics](#-categorical-descriptive-statistics)
35. [Missing Values](#-missing-values)
36. [Descriptive Statistics and Outliers](#-descriptive-statistics-and-outliers)
37. [Visualization](#-visualization)
38. [Summary Tables](#-summary-tables)
39. [Reusable Functions](#-reusable-functions)
40. [Automated Statistical Summary](#-automated-statistical-summary)
41. [Descriptive Statistics for Machine Learning](#-descriptive-statistics-for-machine-learning)
42. [Common Mistakes](#-common-mistakes)
43. [Best Practices](#-best-practices)
44. [Mini Projects](#-mini-projects)
45. [Exercises](#-exercises)
46. [Project Structure](#-project-structure)
47. [End-to-End Example](#-end-to-end-example)
48. [Descriptive Statistics Checklist](#-descriptive-statistics-checklist)
49. [Decision Guide](#-decision-guide)
50. [EDA Roadmap](#-eda-roadmap)
51. [Key Takeaways](#-key-takeaways)
52. [Next Step](#-next-step)
53. [Author](#-author)
54. [Contributing](#-contributing)
55. [Support](#-support)
56. [License](#-license)

---

# 🔎 What Is Descriptive Statistics?

**Descriptive statistics** is the process of summarizing and describing observed data using numerical measures and visual representations.

For example, suppose exam scores are:

```text
65, 72, 78, 80, 85, 90, 92
```

Instead of examining every value, we can summarize the dataset using:

```text
Mean       → 80.3
Median     → 80
Minimum    → 65
Maximum    → 92
Range      → 27
Std Dev    → Measure of spread
```

The goal is:

```text
Raw Data
   ↓
Summary
   ↓
Understanding
   ↓
Interpretation
```

---

# 🎯 Why Descriptive Statistics Matters

Descriptive statistics provides a compact overview of a dataset.

It helps us understand:

### Center

```text
What is a typical observation?
```

### Spread

```text
How much do observations vary?
```

### Position

```text
Where does an observation lie relative to others?
```

### Shape

```text
Is the distribution symmetric or skewed?
```

### Extremes

```text
Are there unusually large or small values?
```

These insights form the foundation of EDA.

---

# 📚 Descriptive vs Inferential Statistics

| Descriptive Statistics   | Inferential Statistics                 |
| ------------------------ | -------------------------------------- |
| Summarizes observed data | Draws conclusions beyond observed data |
| Mean                     | Confidence interval                    |
| Median                   | Hypothesis test                        |
| Standard deviation       | p-value                                |
| Percentiles              | Regression inference                   |
| Frequency                | Population estimation                  |

Example:

### Descriptive

> The average salary in this dataset is ₹65,000.

### Inferential

> Based on a sample, we estimate the population's average salary.

---

# 🌍 Population vs Sample

A **population** is the complete group we want to study.

A **sample** is a subset of that population.

```text
Population
┌──────────────────────────────┐
│ • • • • • • • • • • • • • • │
│ • • • • • • • • • • • • • • │
└──────────────────────────────┘
              ↓
            Sample
        ┌─────────────┐
        │ • • • • • • │
        └─────────────┘
```

Example:

```text
Population → All students in a university
Sample     → 500 selected students
```

This distinction matters when calculating variance and standard deviation.

---

# 🧩 Types of Data

## Numerical Data

### Discrete

Countable values:

```text
Number of customers
Number of products
Number of calls
```

### Continuous

Measurements:

```text
Height
Weight
Temperature
Income
```

---

## Categorical Data

### Nominal

No inherent order:

```text
City
Department
Color
Product Category
```

### Ordinal

Meaningful order:

```text
Low
Medium
High
```

Descriptive techniques should match the type of variable.

---

# 🎯 Measures of Central Tendency

Central tendency describes the **center or typical location** of a distribution.

The main measures are:

```text
Mean
Median
Mode
```

---

# ➗ Mean

The arithmetic mean is:

$$
\bar{x} =
\frac{\sum_{i=1}^{n}x_i}{n}
$$

Example:

```text
10, 20, 30, 40, 50
```

$$
Mean = \frac{10+20+30+40+50}{5}=30
$$

Python:

```python
values = [10, 20, 30, 40, 50]

mean = sum(values) / len(values)

print(mean)
```

Using Python's `statistics` module:

```python
from statistics import mean

print(mean(values))
```

---

# ⚖️ Weighted Mean

Sometimes observations have different importance.

Formula:

$$
\bar{x}_w =
\frac{\sum w_i x_i}
{\sum w_i}
$$

Example:

```text
Assignment → 20%
Midterm    → 30%
Final      → 50%
```

```python
values = [80, 70, 90]
weights = [0.2, 0.3, 0.5]

weighted_mean = sum(
    value * weight
    for value, weight in zip(values, weights)
)

print(weighted_mean)
```

Weighted averages are common in:

* Academic grading
* Finance
* Survey analysis
* Business metrics
* ML evaluation

---

# 📍 Median

The median is the middle value after sorting.

Example:

```text
10, 20, 30, 40, 50

Median = 30
```

For an even number of observations:

```text
10, 20, 30, 40
```

$$
Median = \frac{20+30}{2}=25
$$

Python:

```python
from statistics import median

values = [10, 20, 30, 40, 50]

print(median(values))
```

Pandas:

```python
df["salary"].median()
```

---

# 🔁 Mode

The mode is the most frequently occurring value.

Example:

```text
10, 20, 20, 30, 40

Mode = 20
```

Python:

```python
from statistics import mode

values = [10, 20, 20, 30, 40]

print(mode(values))
```

Pandas:

```python
df["department"].mode()
```

Important:

A dataset can have:

* No unique mode
* One mode
* Multiple modes

Pandas may return multiple values when the distribution is multimodal.

---

# ⚖️ Mean vs Median vs Mode

| Measure | Best Use                   |
| ------- | -------------------------- |
| Mean    | Symmetric numerical data   |
| Median  | Skewed numerical data      |
| Mode    | Most common category/value |

Example:

```text
10, 20, 20, 30, 1000
```

The mean is heavily influenced by `1000`.

The median is more resistant to this extreme value.

---

# 📏 Measures of Dispersion

Central tendency alone is not enough.

Consider:

```text
Dataset A:
48, 49, 50, 51, 52

Dataset B:
10, 30, 50, 70, 90
```

Both have:

```text
Mean = 50
```

But their spreads are very different.

Measures of dispersion include:

* Range
* Variance
* Standard deviation
* Quartiles
* IQR

---

# 📐 Range

Range is:

$$
Range = Maximum - Minimum
$$

Example:

```text
10, 20, 30, 40, 50
```

```text
Range = 50 - 10 = 40
```

Python:

```python
values = [10, 20, 30, 40, 50]

data_range = max(values) - min(values)

print(data_range)
```

Range is simple but depends only on two observations.

---

# 📊 Variance

Variance measures average squared deviation from the mean.

### Population variance

$$
\sigma^2 =
\frac{\sum (x_i-\mu)^2}{N}
$$

### Sample variance

$$
s^2 =
\frac{\sum (x_i-\bar{x})^2}{n-1}
$$

The sample formula uses `n - 1` to provide an unbiased estimator of population variance under the usual assumptions.

Python:

```python
from statistics import variance

sample_variance = variance(values)

print(sample_variance)
```

NumPy population variance:

```python
import numpy as np

population_variance = np.var(values)

print(population_variance)
```

---

# 📏 Standard Deviation

Standard deviation is the square root of variance.

$$
\sigma = \sqrt{\sigma^2}
$$

It is expressed in the same units as the original variable.

Example:

```text
Salary
```

Variance:

```text
Salary²
```

Standard deviation:

```text
Salary
```

Python:

```python
from statistics import stdev

sample_std = stdev(values)

print(sample_std)
```

NumPy:

```python
np.std(values)
```

Remember that NumPy's default `ddof=0` computes population standard deviation, while sample standard deviation uses `ddof=1`.

---

# 📦 Quartiles

Quartiles divide ordered data into four parts.

```text
Minimum
   │
   Q1
   │
   Q2 = Median
   │
   Q3
   │
Maximum
```

### Q1

25th percentile.

### Q2

50th percentile = Median.

### Q3

75th percentile.

Pandas:

```python
df["salary"].quantile(0.25)
df["salary"].quantile(0.50)
df["salary"].quantile(0.75)
```

---

# 📈 Percentiles

A percentile indicates the position below which a specified percentage of observations falls.

Examples:

```text
25th percentile → Q1
50th percentile → Median
75th percentile → Q3
90th percentile → P90
95th percentile → P95
99th percentile → P99
```

Python:

```python
df["salary"].quantile(0.90)
```

Percentiles are widely used in:

* Performance analysis
* Income analysis
* Latency monitoring
* Risk analysis
* Business analytics

---

# 📦 Interquartile Range

The interquartile range is:

$$
IQR = Q3-Q1
$$

It describes the spread of the middle 50% of observations.

```python
q1 = df["salary"].quantile(0.25)
q3 = df["salary"].quantile(0.75)

iqr = q3 - q1

print(iqr)
```

IQR is more resistant to extreme values than the range.

---

# 📋 Five-Number Summary

The five-number summary contains:

```text
Minimum
Q1
Median
Q3
Maximum
```

Pandas:

```python
summary = df["salary"].describe()

print(summary)
```

Or manually:

```python
summary = {
    "min": df["salary"].min(),
    "q1": df["salary"].quantile(0.25),
    "median": df["salary"].median(),
    "q3": df["salary"].quantile(0.75),
    "max": df["salary"].max()
}

print(summary)
```

---

# 📍 Measures of Position

Measures of position describe where an observation lies within a distribution.

Important concepts:

* Percentiles
* Quartiles
* Quantiles
* Z-scores

These are especially useful when comparing observations from different distributions.

---

# 🎯 Z-Score

A z-score indicates how many standard deviations an observation is from the mean.

$$
z = \frac{x-\mu}{\sigma}
$$

Example:

```text
Mean = 70
Std Dev = 10
Score = 90
```

$$
z = \frac{90-70}{10}=2
$$

The score is two standard deviations above the mean.

Python:

```python
from scipy.stats import zscore

df["salary_z"] = zscore(
    df["salary"],
    nan_policy="omit"
)
```

---

# 📈 Distribution Shape

Descriptive statistics also help describe distribution shape.

Important characteristics:

```text
Center
Spread
Skewness
Kurtosis
```

---

# ↗️ Skewness

Skewness measures asymmetry in a distribution.

### Symmetric

```text
        █
      ███
    █████
  ███████
█████████
```

### Right-skewed

```text
████████
██████
████
██
█
      →
        Long right tail
```

### Left-skewed

```text
        ████████
          ██████
            ████
              ██
                █
← Long left tail
```

Pandas:

```python
df["salary"].skew()
```

A positive value generally indicates right skew, while a negative value generally indicates left skew.

---

# 📊 Kurtosis

Kurtosis describes aspects of tail heaviness and distribution shape relative to a reference distribution.

Pandas:

```python
df["salary"].kurt()
```

Be careful when interpreting kurtosis because different libraries and definitions may use different conventions, such as **excess kurtosis** versus raw kurtosis.

In Pandas, `Series.kurt()` returns Fisher's definition, where a normal distribution has kurtosis approximately equal to zero.

---

# 🔔 Normal Distribution

A normal distribution is symmetric and bell-shaped.

```text
                 █
              █████
            █████████
          █████████████
       ███████████████████
─────────────────────────────
```

Important properties:

```text
Mean ≈ Median ≈ Mode
```

For an ideal normal distribution:

```text
Approximately 68% → within ±1 SD
Approximately 95% → within ±2 SD
Approximately 99.7% → within ±3 SD
```

These percentages are known as the **empirical rule**.

---

# 📐 Empirical Rule

For approximately normal data:

```text
            68%
       ┌───────────┐
       │           │
───────┼───────────┼───────
      -1σ         +1σ
```

Approximately:

```text
68%  → μ ± 1σ
95%  → μ ± 2σ
99.7% → μ ± 3σ
```

This rule should not be applied blindly to strongly skewed or non-normal distributions.

---

# 🚨 Outliers

Descriptive statistics can help identify unusual observations.

The IQR rule is:

$$
Lower\ Bound = Q1 - 1.5(IQR)
$$

$$
Upper\ Bound = Q3 + 1.5(IQR)
$$

Python:

```python
q1 = df["salary"].quantile(0.25)
q3 = df["salary"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[
    (df["salary"] < lower) |
    (df["salary"] > upper)
]

print(outliers)
```

An observation outside these bounds is a potential outlier, not automatically an error.

---

# 🐍 Descriptive Statistics With Python

Python provides several useful libraries:

```text
Python statistics
      ↓
NumPy
      ↓
Pandas
      ↓
SciPy
```

Each provides different levels of functionality.

---

# 🐍 Python Built-in Statistics

```python
from statistics import (
    mean,
    median,
    mode,
    variance,
    stdev
)

values = [10, 20, 20, 30, 40]

print("Mean:", mean(values))
print("Median:", median(values))
print("Mode:", mode(values))
print("Variance:", variance(values))
print("Std Dev:", stdev(values))
```

For more general multimodal data, `statistics.multimode()` can be useful:

```python
from statistics import multimode

print(multimode(values))
```

---

# 🔢 NumPy Statistics

```python
import numpy as np

values = np.array([
    10,
    20,
    20,
    30,
    40
])

print("Mean:", np.mean(values))
print("Median:", np.median(values))
print("Minimum:", np.min(values))
print("Maximum:", np.max(values))
print("Variance:", np.var(values))
print("Std Dev:", np.std(values))
```

Sample standard deviation:

```python
np.std(values, ddof=1)
```

Sample variance:

```python
np.var(values, ddof=1)
```

---

# 🐼 Pandas Statistics

Pandas makes descriptive statistics easy for DataFrames.

```python
print(df["salary"].mean())
print(df["salary"].median())
print(df["salary"].min())
print(df["salary"].max())
print(df["salary"].std())
print(df["salary"].var())
```

Multiple statistics:

```python
df["salary"].agg(
    [
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ]
)
```

---

# 📋 DataFrame `describe()`

One of the most useful Pandas methods:

```python
print(df.describe())
```

Typical output:

```text
       salary
count   100.0
mean     65.2
std      12.4
min      35.0
25%      55.0
50%      64.0
75%      74.0
max      110.0
```

For categorical columns:

```python
print(
    df.describe(
        include="object"
    )
)
```

For all columns:

```python
print(
    df.describe(
        include="all"
    )
)
```

Note that output depends on the column types present.

---

# 👥 Grouped Descriptive Statistics

Grouped statistics are essential for comparing populations or categories.

Example:

```python
summary = (
    df.groupby("department")["salary"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )
)

print(summary)
```

For multiple numerical columns:

```python
summary = (
    df.groupby("department")
    [["salary", "experience"]]
    .agg(
        ["mean", "median", "std"]
    )
)

print(summary)
```

---

# 🏷️ Categorical Descriptive Statistics

For categorical variables, numerical measures such as mean may not be meaningful.

Use:

```python
print(df["department"].value_counts())
```

Percentages:

```python
print(
    df["department"]
    .value_counts(normalize=True)
    .mul(100)
)
```

Unique values:

```python
print(df["department"].nunique())
```

Mode:

```python
print(df["department"].mode())
```

---

# 🕳️ Missing Values

Missing values can affect descriptive statistics.

Check first:

```python
print(df.isna().sum())
```

Pandas aggregation methods commonly skip missing values by default.

For example:

```python
df["salary"].mean()
```

is generally calculated using available non-missing observations.

You can explicitly control this behavior:

```python
df["salary"].mean(skipna=False)
```

This returns `NaN` if any missing value is present.

Always understand the missingness pattern before interpreting a summary.

---

# 🚨 Descriptive Statistics and Outliers

Outliers can strongly affect:

```text
Mean
Variance
Standard Deviation
Correlation
Regression
```

But they have less influence on:

```text
Median
Quartiles
IQR
```

Example:

```text
10, 20, 30, 40, 1000
```

The mean changes dramatically because of `1000`.

The median remains:

```text
30
```

Therefore, compare both mean and median when distributions may be skewed.

---

# 📊 Visualization

Descriptive statistics become more useful when combined with visualizations.

## Histogram

```python
import matplotlib.pyplot as plt

df["salary"].hist()

plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.title("Salary Distribution")

plt.show()
```

## Box Plot

```python
df["salary"].plot(
    kind="box"
)

plt.title("Salary Box Plot")
plt.show()
```

## KDE

```python
df["salary"].plot(
    kind="density"
)

plt.title("Salary Density")
plt.show()
```

A numerical summary and visualization should support each other.

---

# 📋 Summary Tables

A reusable summary table can combine important statistics.

```python
summary = df["salary"].agg(
    [
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ]
)

print(summary)
```

More detailed:

```python
summary = {
    "count": df["salary"].count(),
    "mean": df["salary"].mean(),
    "median": df["salary"].median(),
    "std": df["salary"].std(),
    "variance": df["salary"].var(),
    "min": df["salary"].min(),
    "q1": df["salary"].quantile(0.25),
    "q3": df["salary"].quantile(0.75),
    "max": df["salary"].max(),
    "skewness": df["salary"].skew(),
    "kurtosis": df["salary"].kurt()
}

print(summary)
```

---

# 🧰 Reusable Functions

## Numerical Summary

```python
def numerical_summary(series):
    return {
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "skewness": series.skew(),
        "kurtosis": series.kurt()
    }
```

Usage:

```python
summary = numerical_summary(
    df["salary"]
)

for key, value in summary.items():
    print(f"{key}: {value}")
```

---

# 🤖 Automated Statistical Summary

A dataset-level function:

```python
def descriptive_report(df):
    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns
    )

    report = {}

    for column in numeric_columns:
        report[column] = numerical_summary(
            df[column]
        )

    return report
```

Usage:

```python
report = descriptive_report(df)

for column, summary in report.items():
    print(f"\n{column}")
    print("-" * 30)

    for key, value in summary.items():
        print(f"{key}: {value}")
```

---

# 🤖 Descriptive Statistics for Machine Learning

Descriptive statistics is essential before model training.

It helps identify:

### Scale Differences

```text
Age        → 18–80
Income     → 20,000–500,000
```

### Missing Values

```text
Feature A → 2%
Feature B → 45%
```

### Outliers

```text
Income → Extreme observations
```

### Skewed Features

```text
Income
Sales
Transaction Amount
```

### Low-Variance Features

A feature with almost no variation may provide little information.

### Potential Data Leakage

An unexpectedly strong summary or relationship can prompt investigation into whether information from the target or future has entered the features.

---

# 🧠 Descriptive Statistics in an ML Workflow

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Descriptive Statistics
     ↓
Missing Values
     ↓
Distribution Analysis
     ↓
Outlier Analysis
     ↓
Bivariate Analysis
     ↓
Multivariate Analysis
     ↓
Feature Engineering
     ↓
Modeling
```

---

# ⚠️ Common Mistakes

## ❌ Mistake 1: Using Mean for Every Dataset

Mean can be misleading for strongly skewed data.

Consider median as well.

---

## ❌ Mistake 2: Ignoring Standard Deviation

Two datasets can have the same mean but very different variability.

---

## ❌ Mistake 3: Confusing Variance and Standard Deviation

Variance is in squared units.

Standard deviation is in the original units.

---

## ❌ Mistake 4: Ignoring Missing Values

A summary based on a small subset of available observations may not represent the full dataset.

---

## ❌ Mistake 5: Automatically Removing Outliers

An outlier may represent:

* Data entry error
* Measurement error
* Genuine rare event
* Important business case

Investigate before removing.

---

## ❌ Mistake 6: Assuming Normality

Not every dataset follows a normal distribution.

Always inspect:

* Histogram
* Box plot
* Quantiles
* Skewness
* Domain context

---

## ❌ Mistake 7: Confusing Sample and Population Statistics

For example:

```python
np.std(values)
```

uses population-style normalization by default.

For sample standard deviation:

```python
np.std(
    values,
    ddof=1
)
```

---

## ❌ Mistake 8: Treating Summary Statistics as the Entire EDA

Statistics compress information.

Two datasets can have similar means and standard deviations while having very different distributions.

Always combine statistics with visualization.

---

# ✅ Best Practices

### 1. Start With Data Types

```text
Numerical?
Categorical?
Ordinal?
Date-Time?
```

### 2. Report Center and Spread Together

Prefer:

```text
Mean + Standard Deviation
```

or:

```text
Median + IQR
```

depending on the distribution and purpose.

### 3. Inspect Distribution Shape

Use:

* Histogram
* Density plot
* Box plot

### 4. Check Missingness

Always know how many observations support each statistic.

### 5. Investigate Outliers

Do not automatically delete unusual values.

### 6. Compare Groups Carefully

Use grouped descriptive statistics where relevant.

### 7. Use Appropriate Precision

Avoid reporting unnecessary decimal places.

### 8. Combine Numerical and Visual Analysis

```text
Statistics + Visualization + Context
```

provides a stronger understanding than any one source alone.

---

# 🚀 Mini Projects

## 🏠 Project 1 — House Price Statistics

Variables:

```text
price
size
bedrooms
bathrooms
age
```

Calculate:

* Mean
* Median
* Mode where meaningful
* Range
* Variance
* Standard deviation
* Quartiles
* IQR
* Skewness
* Outliers

---

## 💼 Project 2 — Employee Salary Analysis

Variables:

```text
age
experience
salary
department
performance
```

Tasks:

* Overall statistics
* Department-wise statistics
* Salary distribution
* Outlier detection
* Mean vs median comparison

---

## 🛒 Project 3 — E-Commerce Transactions

Variables:

```text
customer_age
purchase_amount
session_time
items
category
```

Calculate:

* Average purchase
* Median purchase
* P90 purchase
* IQR
* Standard deviation
* Category-wise statistics

---

## 📚 Project 4 — Student Performance

Variables:

```text
study_hours
attendance
previous_score
final_score
```

Analyze:

* Mean scores
* Median scores
* Score distribution
* Standard deviation
* Percentiles
* Outliers
* Group comparisons

---

# 🧠 Exercises

## 🟢 Beginner

1. Calculate the mean of a list.
2. Calculate the median.
3. Find the mode.
4. Calculate the range.
5. Calculate variance.
6. Calculate standard deviation.
7. Find Q1 and Q3.
8. Calculate IQR.
9. Calculate P90.
10. Generate a Pandas `describe()` report.

---

## 🟡 Intermediate

1. Compare mean and median for skewed data.
2. Calculate grouped descriptive statistics.
3. Detect IQR-based outliers.
4. Calculate z-scores.
5. Compare population and sample standard deviation.
6. Analyze skewness.
7. Analyze kurtosis.
8. Create a five-number summary.
9. Create a statistical report for every numerical column.
10. Combine descriptive statistics with visualizations.

---

## 🔴 Advanced

1. Build an automated EDA statistics report.
2. Compare several distributions using robust statistics.
3. Investigate how outliers affect mean and standard deviation.
4. Compare mean/IQR and median/IQR reporting strategies.
5. Build a statistical profiling utility.
6. Analyze grouped distributions.
7. Implement descriptive statistics manually using NumPy.
8. Compare your implementation with Pandas.
9. Create a production-ready data-quality summary.
10. Integrate descriptive statistics into an ML preprocessing workflow.

---

# 📁 Project Structure

```text
06-Exploratory-Data-Analysis/
│
├── README.md
│
├── 01-Univariate-Analysis/
│   ├── README.md
│   ├── numerical-analysis.py
│   ├── categorical-analysis.py
│   ├── distribution-analysis.py
│   ├── outlier-analysis.py
│   └── univariate-report.py
│
├── 02-Bivariate-Analysis/
│   ├── README.md
│   ├── numerical-relationships.py
│   ├── correlation-analysis.py
│   ├── categorical-relationships.py
│   ├── grouped-analysis.py
│   ├── contingency-analysis.py
│   └── bivariate-report.py
│
├── 03-Multivariate-Analysis/
│   ├── README.md
│   ├── correlation-matrix.py
│   ├── pair-plot.py
│   ├── multivariate-visualization.py
│   ├── multicollinearity.py
│   ├── pca-analysis.py
│   └── multivariate-report.py
│
├── 04-Descriptive-Statistics/
│   ├── README.md
│   ├── central-tendency.py
│   ├── dispersion.py
│   ├── percentiles.py
│   ├── z-score.py
│   ├── skewness-kurtosis.py
│   └── statistical-report.py
│
├── 05-Data-Visualization/
│
└── 06-EDA-Projects/
```

---

# 💻 End-to-End Example

```python
"""
Descriptive Statistics
----------------------
A practical demonstration of descriptive statistics
using Python, NumPy, Pandas, and Matplotlib.

Requirements:
    pip install pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_dataset():
    """Create a demonstration dataset."""

    return pd.DataFrame({
        "name": [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H"
        ],
        "department": [
            "IT",
            "IT",
            "HR",
            "HR",
            "Sales",
            "Sales",
            "IT",
            "Sales"
        ],
        "salary": [
            45000,
            52000,
            48000,
            50000,
            55000,
            62000,
            70000,
            80000
        ],
        "experience": [
            1,
            3,
            2,
            4,
            5,
            7,
            10,
            12
        ]
    })


def numerical_summary(series):
    """Return descriptive statistics for a numerical Series."""

    return {
        "count": series.count(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "skewness": series.skew(),
        "kurtosis": series.kurt()
    }


def display_summary(summary):
    """Print a formatted statistical summary."""

    print("\nDescriptive Statistics")
    print("-" * 40)

    for key, value in summary.items():
        print(f"{key:12}: {value:.4f}")


def grouped_statistics(df):
    """Display statistics grouped by department."""

    summary = (
        df.groupby("department")["salary"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
    )

    print("\nDepartment-Wise Statistics")
    print("-" * 40)
    print(summary)


def detect_outliers(df, column):
    """Detect potential outliers using the IQR method."""

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower)
        | (df[column] > upper)
    ]

    print("\nPotential Outliers")
    print("-" * 40)
    print(outliers)

    return outliers


def visualize_salary(df):
    """Display salary distribution."""

    plt.hist(
        df["salary"],
        bins=5
    )

    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    plt.title("Salary Distribution")

    plt.tight_layout()
    plt.show()


def main():
    """Run the complete descriptive statistics analysis."""

    df = create_dataset()

    print("Dataset")
    print("-" * 40)
    print(df)

    summary = numerical_summary(
        df["salary"]
    )

    display_summary(summary)

    grouped_statistics(df)

    detect_outliers(
        df,
        "salary"
    )

    print("\nPandas describe()")
    print("-" * 40)
    print(df.describe())

    visualize_salary(df)


if __name__ == "__main__":
    main()
```

---

# 🔬 Descriptive Statistics Workflow

A practical workflow:

```text
                    DATASET
                       │
                       ↓
                Identify Variables
                       │
                       ↓
                 Check Data Types
                       │
                       ↓
              Handle/Understand Missing
                       │
                       ↓
              ┌────────┴────────┐
              ↓                 ↓
          CENTRALITY          SPREAD
              │                 │
              ↓                 ↓
           Mean              Range
           Median            Variance
           Mode              Std Dev
              │                 │
              └────────┬────────┘
                       ↓
                    POSITION
                       │
                       ↓
               Quartiles / Percentiles
                       │
                       ↓
                     SHAPE
                       │
                       ↓
             Skewness / Kurtosis
                       │
                       ↓
                   OUTLIERS
                       │
                       ↓
                VISUALIZATION
                       │
                       ↓
                  INTERPRET
                       │
                       ↓
                 EDA → ML 🚀
```

---

# 📋 Descriptive Statistics Checklist

## 🔍 Data

* [ ] Identify numerical columns.
* [ ] Identify categorical columns.
* [ ] Check data types.
* [ ] Check missing values.
* [ ] Check sample size.

## 📍 Central Tendency

* [ ] Mean
* [ ] Median
* [ ] Mode where meaningful

## 📏 Dispersion

* [ ] Range
* [ ] Variance
* [ ] Standard deviation
* [ ] IQR

## 📈 Position

* [ ] Q1
* [ ] Median
* [ ] Q3
* [ ] P90
* [ ] P95
* [ ] P99 when relevant

## 📊 Distribution

* [ ] Histogram
* [ ] Box plot
* [ ] Skewness
* [ ] Kurtosis

## 🚨 Quality

* [ ] Missing values
* [ ] Outliers
* [ ] Impossible values
* [ ] Unexpected ranges

## 🧠 Interpretation

* [ ] Choose appropriate summary measures.
* [ ] Consider distribution shape.
* [ ] Compare groups where relevant.
* [ ] Consider sample size.
* [ ] Combine statistics with visualization.

---

# 🧭 Decision Guide

```text
                 What type of data?
                        │
          ┌─────────────┴─────────────┐
          ↓                           ↓
       Numerical                  Categorical
          │                           │
          ↓                           ↓
     Centrality                  Frequency
          │                           │
     ┌────┼────┐                      ↓
     ↓    ↓    ↓                 Percentages
   Mean Median Mode                    │
     │    │    │                      ↓
     └────┼────┘                 Mode / Counts
          ↓
        Spread
          │
     ┌────┼──────────┐
     ↓    ↓          ↓
   Range Variance   IQR
          │
          ↓
    Standard Deviation
          │
          ↓
        Shape
          │
    ┌─────┴─────┐
    ↓           ↓
 Skewness    Kurtosis
```

### Choosing a center

```text
Symmetric distribution
        ↓
      Mean

Skewed distribution
        ↓
     Median

Categorical data
        ↓
       Mode
```

---

# 🗺️ EDA Roadmap

The EDA curriculum now connects:

```text
06-Exploratory-Data-Analysis/
│
├── 01-Univariate-Analysis/
│       ↓
│   One Variable
│
├── 02-Bivariate-Analysis/
│       ↓
│   Two Variables
│
├── 03-Multivariate-Analysis/
│       ↓
│   Multiple Variables
│
├── 04-Descriptive-Statistics/
│       ↓
│   Summarize Data
│
├── 05-Data-Visualization/
│       ↓
│   Communicate Insights
│
└── 06-EDA-Projects/
        ↓
    Real-World Analysis
```

# 🌱 GROW WITH STATISTICS

```text
🌱 GROW
  ↓
📥 COLLECT
  ↓
🔍 INSPECT
  ↓
📊 SUMMARIZE
  ↓
📐 MEASURE
  ↓
📈 VISUALIZE
  ↓
🧠 INTERPRET
  ↓
🔗 CONNECT
  ↓
🤖 MODEL
  ↓
🚀 BUILD
```

---

# 🎯 Key Takeaways

> **Descriptive statistics turns raw observations into meaningful numerical summaries.**

Remember:

1. **Mean measures arithmetic center.**
2. **Median represents the middle observation after ordering.**
3. **Mode identifies the most frequent value(s).**
4. **Range measures the distance between minimum and maximum.**
5. **Variance measures squared dispersion.**
6. **Standard deviation expresses spread in the original units.**
7. **Quartiles divide ordered data into four parts.**
8. **IQR measures the spread of the middle 50%.**
9. **Percentiles describe relative position.**
10. **Z-scores measure distance from the mean in standard deviation units.**
11. **Skewness describes distribution asymmetry.**
12. **Kurtosis describes tail/shape characteristics under a specified convention.**
13. **Outliers can strongly affect mean and standard deviation.**
14. **Median and IQR are often more robust for skewed data.**
15. **Statistics should be combined with visualization.**
16. **Sample and population formulas are not always the same.**
17. **Descriptive statistics is a foundation for EDA and Machine Learning.**

---

# 🚀 Next Step

After understanding how to numerically summarize data, the next step is learning how to **communicate those summaries visually**.

Continue with:

```text
06-Exploratory-Data-Analysis/
└── 05-Data-Visualization/
```

You will learn:

* Matplotlib fundamentals
* Seaborn
* Histograms
* KDE plots
* Box plots
* Violin plots
* Bar charts
* Scatter plots
* Line charts
* Heatmaps
* Pair plots
* Statistical visualization
* Advanced visualization
* EDA storytelling
* Visualization best practices

The complete journey:

```text
🌱 DATA
  ↓
🔍 INSPECT
  ↓
📊 UNIVARIATE
  ↓
🔗 BIVARIATE
  ↓
🧠 MULTIVARIATE
  ↓
📐 DESCRIPTIVE STATISTICS
  ↓
🎨 VISUALIZE
  ↓
💡 DISCOVER
  ↓
🤖 MODEL
  ↓
🚀 BUILD
```

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

🔗 GitHub:

[https://github.com/Kishor055](https://github.com/Kishor055)

---

# 🤝 Contributing

Contributions are welcome!

You can contribute by:

* Improving explanations
* Fixing technical errors
* Adding statistical examples
* Adding datasets
* Improving visualizations
* Adding exercises
* Adding real-world EDA projects
* Improving documentation

### Contribution Workflow

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/improve-descriptive-statistics
```

Make your changes, test the examples, and submit a pull request.

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples

Your support helps improve this learning resource.

---

# 📄 License

This project is intended for educational and learning purposes.

Please check the repository's license file for the applicable terms.

---

<div align="center">

### 🌱 GROW → SUMMARIZE → MEASURE → UNDERSTAND → BUILD 🚀

**Turn raw data into measurable insights.**

</div>

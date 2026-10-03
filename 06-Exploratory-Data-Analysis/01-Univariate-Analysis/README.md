# 📊 Univariate Analysis

> **Univariate Analysis** is the first major step in Exploratory Data Analysis (EDA). It focuses on understanding **one variable at a time** using statistics, distributions, frequency tables, and visualizations.

Univariate analysis helps answer questions such as:

* What type of variable is this?
* What values does it contain?
* How frequently do values occur?
* What is the center of the distribution?
* How much does the variable vary?
* Are there missing values?
* Are there outliers?
* Is the distribution skewed?
* Is the variable suitable for Machine Learning?

---

# 🌱 GROW → LEARN → ANALYZE → VISUALIZE → UNDERSTAND → BUILD 🚀

```text
              📊 UNIVARIATE ANALYSIS
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       🔢 Numeric    🏷️ Categorical  📅 Date/Time
          │            │            │
          ↓            ↓            ↓
     Statistics     Frequencies    Trends
          │            │            │
          └────────────┼────────────┘
                       ↓
                 📈 Visualize
                       ↓
                 🔍 Identify
              Patterns & Outliers
                       ↓
                 🧠 Understand
                       ↓
                  🤖 Prepare
                    for ML
```

---

# 📚 Table of Contents

1. [What Is Univariate Analysis?](#-what-is-univariate-analysis)
2. [Why Univariate Analysis Matters](#-why-univariate-analysis-matters)
3. [Univariate vs Bivariate vs Multivariate](#-univariate-vs-bivariate-vs-multivariate)
4. [Types of Variables](#-types-of-variables)
5. [Numerical Variables](#-numerical-variables)
6. [Categorical Variables](#-categorical-variables)
7. [Date-Time Variables](#-date-time-variables)
8. [Data Inspection](#-data-inspection)
9. [Missing Values](#-missing-values)
10. [Unique Values](#-unique-values)
11. [Frequency Analysis](#-frequency-analysis)
12. [Central Tendency](#-central-tendency)
13. [Mean](#-mean)
14. [Median](#-median)
15. [Mode](#-mode)
16. [Measures of Dispersion](#-measures-of-dispersion)
17. [Range](#-range)
18. [Variance](#-variance)
19. [Standard Deviation](#-standard-deviation)
20. [Quartiles](#-quartiles)
21. [Percentiles](#-percentiles)
22. [IQR](#-iqr)
23. [Skewness](#-skewness)
24. [Kurtosis](#-kurtosis)
25. [Distribution Analysis](#-distribution-analysis)
26. [Histogram](#-histogram)
27. [Density Plot](#-density-plot)
28. [Box Plot](#-box-plot)
29. [Violin Plot](#-violin-plot)
30. [Bar Chart](#-bar-chart)
31. [Count Plot](#-count-plot)
32. [Pie Chart](#-pie-chart)
33. [ECDF](#-ecdf)
34. [Categorical Analysis](#-categorical-analysis)
35. [Numerical Analysis](#-numerical-analysis)
36. [Outlier Detection](#-outlier-detection)
37. [Z-Score Analysis](#-z-score-analysis)
38. [Skewed Data](#-skewed-data)
39. [Normal Distribution](#-normal-distribution)
40. [Log Transformation](#-log-transformation)
41. [Univariate Analysis with Pandas](#-univariate-analysis-with-pandas)
42. [Univariate Analysis with Matplotlib](#-univariate-analysis-with-matplotlib)
43. [Univariate Analysis with Seaborn](#-univariate-analysis-with-seaborn)
44. [Reusable Analysis Functions](#-reusable-analysis-functions)
45. [Automated Univariate Report](#-automated-univariate-report)
46. [End-to-End Example](#-end-to-end-example)
47. [Common Mistakes](#-common-mistakes)
48. [Best Practices](#-best-practices)
49. [Mini Projects](#-mini-projects)
50. [Exercises](#-exercises)
51. [Project Structure](#-project-structure)
52. [Univariate Analysis Checklist](#-univariate-analysis-checklist)
53. [EDA Roadmap](#-eda-roadmap)
54. [Key Takeaways](#-key-takeaways)
55. [Next Step](#-next-step)

---

# 🔍 What Is Univariate Analysis?

**Univariate analysis** means analyzing a single variable independently.

The word comes from:

```text
Uni   → One
Variate → Variable
```

Example dataset:

```text
Age    Salary    City       Gender
21     25000     Pune       Male
25     40000     Mumbai     Female
30     55000     Delhi      Male
35     70000     Pune       Female
```

Univariate analysis considers one column at a time.

For example:

```text
Age
 ↓
Distribution
 ↓
Mean
 ↓
Median
 ↓
Standard Deviation
 ↓
Outliers
```

It does **not** study relationships between variables.

For example:

```text
Age ↔ Salary
```

would be **bivariate analysis**.

---

# 🎯 Why Univariate Analysis Matters

Before building Machine Learning models, you need to understand your data.

Univariate analysis can reveal:

* Missing values
* Incorrect values
* Outliers
* Skewness
* Class imbalance
* Rare categories
* Constant features
* Unexpected ranges
* Data-entry errors
* Distribution shape

Example:

```text
Age
10
21
24
27
31
35
250
```

A simple summary may reveal:

```text
Mean   → 57.4
Median → 27
```

The large difference suggests that the distribution may contain an extreme value.

---

# 🔬 Univariate vs Bivariate vs Multivariate

| Analysis     | Variables | Example                   |
| ------------ | --------: | ------------------------- |
| Univariate   |         1 | Age                       |
| Bivariate    |         2 | Age vs Salary             |
| Multivariate |        3+ | Age + Salary + Experience |

### Univariate

```text
Age
 ↓
Distribution
```

### Bivariate

```text
Age ───── Salary
```

### Multivariate

```text
Age
 │
Salary ─── Target
 │
Experience
```

---

# 🧩 Types of Variables

Before analyzing a variable, identify its type.

```text
Variable
│
├── Numerical
│   ├── Discrete
│   └── Continuous
│
├── Categorical
│   ├── Nominal
│   └── Ordinal
│
└── Date-Time
```

---

# 🔢 Numerical Variables

Numerical variables contain measurable quantities.

Examples:

```text
Age
Salary
Height
Weight
Temperature
Experience
Sales
```

They can be analyzed using:

* Mean
* Median
* Variance
* Standard deviation
* Quartiles
* Percentiles
* Histograms
* Box plots
* Density plots
* Skewness

---

# 🔢 Discrete Variables

Discrete variables generally represent countable values.

Examples:

```text
Number of children
Number of purchases
Number of products
Number of defects
```

Example:

```text
0
1
2
3
4
```

---

# 📏 Continuous Variables

Continuous variables can take values across a range.

Examples:

```text
Height
Weight
Temperature
Time
Distance
```

Example:

```text
172.5 cm
172.51 cm
172.512 cm
```

---

# 🏷️ Categorical Variables

Categorical variables represent groups or labels.

Examples:

```text
Gender
City
Department
Product Category
Payment Method
Education Level
```

Categorical variables are usually analyzed using:

* Frequency
* Percentage
* Mode
* Bar charts
* Count plots

---

# 🏷️ Nominal Variables

Nominal categories have no inherent ordering.

Example:

```text
City:
Pune
Mumbai
Delhi
Nashik
```

There is no meaningful order.

---

# 📊 Ordinal Variables

Ordinal categories have an order.

Example:

```text
Education:
School
Bachelor
Master
PhD
```

or:

```text
Satisfaction:
Poor
Average
Good
Excellent
```

The order matters.

---

# 📅 Date-Time Variables

Date-time variables contain temporal information.

Examples:

```text
2026-01-01
2026-01-15
2026-02-10
```

Univariate date-time analysis can examine:

* Year
* Month
* Day
* Day of week
* Hour
* Seasonal patterns
* Frequency over time

---

# 🔎 Data Inspection

Start by understanding the dataset.

```python
import pandas as pd


df = pd.read_csv("data.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
```

A good EDA workflow begins with:

```python
print(df.info())
```

---

# 🕳️ Missing Values

Check missing values before calculating statistics.

```python
print(df.isna().sum())
```

Missing percentage:

```python
missing_percentage = (
    df.isna().mean() * 100
)

print(missing_percentage)
```

For a single column:

```python
print(df["age"].isna().sum())
```

---

# 🔢 Unique Values

For categorical variables:

```python
print(df["city"].unique())
```

Count unique values:

```python
print(df["city"].nunique())
```

For a complete dataset:

```python
print(df.nunique())
```

---

# 🔢 Frequency Analysis

Frequency analysis tells us how often each value occurs.

```python
print(df["city"].value_counts())
```

Example:

```text
Pune       250
Mumbai     180
Delhi      120
Nashik      75
```

---

# 📊 Relative Frequency

Frequency can be converted into percentages.

```python
percentage = (
    df["city"]
    .value_counts(normalize=True)
    .mul(100)
)

print(percentage)
```

Example:

```text
Pune       40.0
Mumbai     28.8
Delhi      19.2
Nashik     12.0
```

This is useful for understanding category proportions.

---

# 🎯 Central Tendency

Central tendency describes the center of a distribution.

The three major measures are:

```text
Mean
Median
Mode
```

---

# ➗ Mean

The mean is the arithmetic average.

$$
\bar{x} =
\frac{1}{n}
\sum_{i=1}^{n}x_i
$$

Example:

```text
10, 20, 30
```

$$
Mean = \frac{10+20+30}{3}=20
$$

Python:

```python
mean = df["salary"].mean()

print(mean)
```

---

# 🧮 Median

The median is the middle value after sorting.

Example:

```text
10, 20, 30
```

Median:

```text
20
```

For an even number of values:

```text
10, 20, 30, 40
```

$$
Median =
\frac{20+30}{2}=25
$$

Python:

```python
median = df["salary"].median()

print(median)
```

### When is median useful?

Median is generally less sensitive to extreme values than the mean.

---

# 🔁 Mode

Mode is the most frequently occurring value.

Example:

```text
10, 20, 20, 30
```

Mode:

```text
20
```

Python:

```python
mode = df["city"].mode()

print(mode)
```

For a single result:

```python
mode_value = df["city"].mode().iloc[0]
```

Be careful: a dataset can have multiple modes.

---

# 📏 Measures of Dispersion

Central tendency alone is not enough.

Consider:

```text
Dataset A:
10, 20, 30

Dataset B:
1, 20, 49
```

Both have mean:

```text
20
```

But their spread is different.

Important dispersion measures include:

* Range
* Variance
* Standard deviation
* IQR

---

# ↔️ Range

Range is:

$$
Range = Maximum-Minimum
$$

Python:

```python
data_range = (
    df["salary"].max()
    - df["salary"].min()
)

print(data_range)
```

---

# 📐 Variance

Variance measures average squared deviation from the mean.

For a population:

$$
\sigma^2 =
\frac{1}{N}
\sum_{i=1}^{N}
(x_i-\mu)^2
$$

Pandas:

```python
variance = df["salary"].var()

print(variance)
```

Pandas uses sample variance by default.

For population variance:

```python
population_variance = (
    df["salary"].var(ddof=0)
)
```

---

# 📏 Standard Deviation

Standard deviation is the square root of variance.

$$
\sigma = \sqrt{\sigma^2}
$$

Python:

```python
std = df["salary"].std()

print(std)
```

A larger standard deviation generally indicates greater dispersion around the mean.

---

# 🔢 Quartiles

Quartiles divide ordered data into four parts.

```text
Minimum
   │
   ├── Q1 → 25%
   │
   ├── Q2 → 50% → Median
   │
   ├── Q3 → 75%
   │
Maximum
```

Calculate:

```python
print(df["salary"].quantile([0.25, 0.50, 0.75]))
```

---

# 📊 Percentiles

A percentile indicates the value below which a specified percentage of observations falls.

Example:

```python
p90 = df["salary"].quantile(0.90)

print(p90)
```

This represents the 90th percentile.

Common percentiles:

```text
25th
50th
75th
90th
95th
99th
```

---

# 📦 IQR

The Interquartile Range measures the middle 50% of the data.

$$
IQR = Q3-Q1
$$

Python:

```python
q1 = df["salary"].quantile(0.25)
q3 = df["salary"].quantile(0.75)

iqr = q3 - q1

print(iqr)
```

IQR is useful for detecting potential outliers.

---

# 🚨 Outlier Detection Using IQR

A common rule defines potential outliers as values below:

$$
Q1-1.5(IQR)
$$

or above:

$$
Q3+1.5(IQR)
$$

Python:

```python
q1 = df["salary"].quantile(0.25)
q3 = df["salary"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[
    (df["salary"] < lower_bound)
    | (df["salary"] > upper_bound)
]

print(outliers)
```

Important:

> An outlier is not automatically an error.

It may represent a legitimate observation.

---

# 📈 Skewness

Skewness describes asymmetry in a distribution.

```text
Negative Skew
      █
     ██
    ███
   ████
████████████
        ↓
     Tail

Positive Skew
████████████
     ████
      ███
       ██
        █
        ↓
      Tail
```

A rough interpretation:

```text
Skewness ≈ 0
    ↓
Approximately symmetric

Positive
    ↓
Right-skewed

Negative
    ↓
Left-skewed
```

Calculate:

```python
skewness = df["salary"].skew()

print(skewness)
```

Do not treat arbitrary thresholds as universal rules; interpretation depends on the data and context.

---

# 🏔️ Kurtosis

Kurtosis describes aspects of the tails and extremity of a distribution.

Pandas:

```python
kurtosis = df["salary"].kurt()

print(kurtosis)
```

Be aware that software packages may use different conventions for reporting kurtosis.

---

# 📊 Distribution Analysis

A distribution describes how values are spread.

For numerical variables, inspect:

* Center
* Spread
* Shape
* Skewness
* Tails
* Outliers
* Gaps
* Multiple peaks

---

# 📊 Histogram

A histogram groups numerical values into bins.

```python
import matplotlib.pyplot as plt


plt.hist(df["age"].dropna(), bins=20)

plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")

plt.show()
```

Histograms are useful for identifying:

* Shape
* Skewness
* Concentration
* Possible outliers
* Multiple modes

---

# 📈 Choosing Histogram Bins

The number of bins affects interpretation.

Too few bins:

```text
Important structure may disappear.
```

Too many bins:

```text
The distribution may appear noisy.
```

Start with a reasonable value and experiment.

```python
plt.hist(
    df["age"].dropna(),
    bins=20
)
```

---

# 🌊 Density Plot

A density plot provides a smoothed estimate of a numerical distribution.

Using Seaborn:

```python
import seaborn as sns
import matplotlib.pyplot as plt


sns.kdeplot(
    data=df,
    x="salary",
    fill=True
)

plt.title("Salary Density")
plt.show()
```

KDE plots are useful for comparing distribution shapes, but the smoothing parameter can affect the appearance.

---

# 📦 Box Plot

A box plot summarizes a numerical distribution using:

```text
Minimum / lower whisker
Q1
Median
Q3
Maximum / upper whisker
Potential outliers
```

Example:

```python
sns.boxplot(
    data=df,
    x="salary"
)

plt.title("Salary Box Plot")
plt.show()
```

---

# 🎻 Violin Plot

A violin plot combines ideas from:

* Box plot
* Density plot

Example:

```python
sns.violinplot(
    data=df,
    x="salary"
)

plt.title("Salary Distribution")
plt.show()
```

It can reveal the shape of the distribution more clearly than a simple box plot.

---

# 📊 Bar Chart

Bar charts are useful for categorical variables.

```python
city_counts = df["city"].value_counts()

city_counts.plot(
    kind="bar"
)

plt.xlabel("City")
plt.ylabel("Count")
plt.title("Customers by City")

plt.show()
```

---

# 📊 Count Plot

Seaborn provides a convenient count plot:

```python
sns.countplot(
    data=df,
    x="city"
)

plt.title("Customer Count by City")
plt.xticks(rotation=45)

plt.show()
```

---

# 🥧 Pie Chart

Pie charts can show category proportions when there are only a few categories.

```python
counts = df["city"].value_counts()

plt.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%"
)

plt.title("Customer Distribution by City")

plt.show()
```

For many categories, a bar chart is generally easier to read.

---

# 📈 ECDF

An Empirical Cumulative Distribution Function shows the proportion of observations less than or equal to each value.

Example:

```python
import numpy as np
import matplotlib.pyplot as plt


values = df["age"].dropna().sort_values()

y = np.arange(1, len(values) + 1) / len(values)

plt.plot(values, y)

plt.xlabel("Age")
plt.ylabel("Cumulative Proportion")
plt.title("ECDF of Age")

plt.show()
```

ECDFs are useful because they show the entire distribution without requiring histogram bins.

---

# 🏷️ Categorical Analysis

For categorical variables, analyze:

```text
Number of categories
       ↓
Frequency
       ↓
Percentage
       ↓
Most common category
       ↓
Rare categories
       ↓
Missing values
```

Example:

```python
category_summary = (
    df["department"]
    .value_counts(dropna=False)
)

print(category_summary)
```

---

# 📊 Categorical Percentage Analysis

```python
category_percentage = (
    df["department"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(2)
)

print(category_percentage)
```

---

# 🔢 Numerical Analysis

For numerical columns, calculate:

```python
numeric_summary = df["salary"].describe()

print(numeric_summary)
```

Typical output contains:

```text
count
mean
std
min
25%
50%
75%
max
```

---

# 📋 Complete Numerical Summary

```python
numeric_columns = df.select_dtypes(
    include="number"
).columns

print(
    df[numeric_columns].describe().T
)
```

The transpose makes it easier to inspect variables row by row.

---

# 🏷️ Complete Categorical Summary

```python
categorical_columns = df.select_dtypes(
    include=["object", "category", "string"]
).columns

print(
    df[categorical_columns].describe().T
)
```

This can show:

* Count
* Unique values
* Most frequent value
* Frequency of most frequent value

---

# 🔍 Outlier Detection

Outliers should be investigated using both statistics and visualizations.

Useful techniques:

```text
IQR
Z-Score
Box Plot
Histogram
Domain Rules
```

---

# 📐 Z-Score Analysis

A z-score measures how far an observation is from the mean in standard deviation units.

$$
z =
\frac{x-\mu}{\sigma}
$$

Python:

```python
mean = df["salary"].mean()
std = df["salary"].std()

df["salary_zscore"] = (
    (df["salary"] - mean) / std
)

print(df[["salary", "salary_zscore"]])
```

A common heuristic is to investigate observations with large absolute z-scores, but the appropriate threshold depends on the distribution and domain.

---

# 📈 Skewed Data

Consider income:

```text
30000
35000
40000
45000
50000
60000
100000
500000
```

The extreme high values may produce right skew.

Check:

```python
print(df["income"].skew())
```

Visualize:

```python
sns.histplot(
    data=df,
    x="income",
    kde=True
)

plt.show()
```

---

# 🔄 Log Transformation

For strictly positive right-skewed numerical variables, a log transformation can sometimes make the distribution easier to model.

```python
import numpy as np


df["income_log"] = np.log1p(
    df["income"]
)
```

`log1p(x)` computes:

$$
\log(1+x)
$$

which is useful when values can include zero.

Important:

> Transformation should be based on the modeling goal and data characteristics. Do not transform a variable simply because it is skewed.

---

# 🔔 Normal Distribution

A normal distribution is approximately symmetric and bell-shaped.

Characteristics:

```text
Mean ≈ Median ≈ Mode
```

Approximately:

```text
68% → within 1 standard deviation
95% → within 2 standard deviations
99.7% → within 3 standard deviations
```

These percentages apply to an ideal normal distribution, not arbitrary datasets.

---

# 🧪 Univariate Analysis with Pandas

A basic numerical workflow:

```python
import pandas as pd


df = pd.read_csv("data.csv")

column = df["age"]

print("Count:", column.count())
print("Missing:", column.isna().sum())
print("Unique:", column.nunique())
print("Mean:", column.mean())
print("Median:", column.median())
print("Std:", column.std())
print("Min:", column.min())
print("Max:", column.max())
print("Q1:", column.quantile(0.25))
print("Q3:", column.quantile(0.75))
print("Skewness:", column.skew())
```

---

# 📊 Univariate Analysis with Matplotlib

```python
import matplotlib.pyplot as plt


def plot_numeric_distribution(df, column):
    values = df[column].dropna()

    plt.figure(figsize=(8, 5))

    plt.hist(
        values,
        bins=20
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(
        f"Distribution of {column}"
    )

    plt.tight_layout()
    plt.show()


plot_numeric_distribution(
    df,
    "age"
)
```

---

# 📊 Univariate Analysis with Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns


def plot_numeric_eda(df, column):
    values = df[column].dropna()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8)
    )

    sns.histplot(
        values,
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title(
        f"{column} Distribution"
    )

    sns.boxplot(
        x=values,
        ax=axes[1]
    )

    axes[1].set_title(
        f"{column} Box Plot"
    )

    plt.tight_layout()
    plt.show()


plot_numeric_eda(
    df,
    "salary"
)
```

---

# 🧰 Reusable Analysis Functions

## Numerical Summary Function

```python
def numerical_summary(series):
    """Return descriptive statistics for a numeric Series."""

    return {
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique()),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "skewness": series.skew(),
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

# 🏷️ Categorical Summary Function

```python
def categorical_summary(series):
    """Return a summary for a categorical Series."""

    counts = series.value_counts(
        dropna=False
    )

    return {
        "count": int(series.count()),
        "missing": int(series.isna().sum()),
        "unique": int(series.nunique(dropna=True)),
        "most_common": (
            counts.index[0]
            if len(counts) > 0
            else None
        ),
        "frequency": (
            int(counts.iloc[0])
            if len(counts) > 0
            else 0
        ),
    }
```

---

# 🤖 Automated Univariate Report

A reusable report can classify columns automatically.

```python
import pandas as pd


def univariate_report(df):
    """Generate a basic univariate report."""

    rows = []

    for column in df.columns:
        series = df[column]

        row = {
            "column": column,
            "dtype": str(series.dtype),
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "missing_pct": (
                series.isna().mean() * 100
            ),
            "unique": int(
                series.nunique(dropna=True)
            ),
        }

        if pd.api.types.is_numeric_dtype(series):
            row.update({
                "mean": series.mean(),
                "median": series.median(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "skewness": series.skew(),
            })

        else:
            mode = series.mode()

            row.update({
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "skewness": None,
                "mode": (
                    mode.iloc[0]
                    if not mode.empty
                    else None
                ),
            })

        rows.append(row)

    return pd.DataFrame(rows)
```

Usage:

```python
report = univariate_report(df)

print(report)
```

---

# 📊 Automated Numerical Visualization

```python
import matplotlib.pyplot as plt


def plot_all_numeric_columns(df):
    """Create one histogram per numerical column."""

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        values = df[column].dropna()

        if values.empty:
            continue

        plt.figure(figsize=(8, 5))

        plt.hist(
            values,
            bins=20
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()
```

---

# 🏷️ Automated Categorical Visualization

```python
import matplotlib.pyplot as plt


def plot_categorical_columns(df):
    """Create bar charts for categorical columns."""

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "string"
        ]
    ).columns

    for column in categorical_columns:
        counts = df[column].value_counts(
            dropna=False
        )

        plt.figure(figsize=(8, 5))

        counts.plot(
            kind="bar"
        )

        plt.title(
            f"{column} Frequency"
        )

        plt.xlabel(column)
        plt.ylabel("Count")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()
        plt.show()
```

---

# 🚀 End-to-End Example

The following example creates a dataset and performs a complete univariate analysis.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_dataset():
    """Create a sample customer dataset."""

    return pd.DataFrame({
        "age": [
            19, 21, 22, 24, 25,
            27, 29, 31, 34, 38,
            42, 45, 52, 60, 72
        ],
        "salary": [
            22000, 25000, 28000, 30000,
            35000, 38000, 42000, 45000,
            50000, 55000, 60000, 70000,
            85000, 100000, 300000
        ],
        "city": [
            "Pune",
            "Mumbai",
            "Pune",
            "Delhi",
            "Mumbai",
            "Pune",
            "Delhi",
            "Pune",
            "Mumbai",
            "Pune",
            "Delhi",
            "Mumbai",
            "Pune",
            "Delhi",
            "Pune",
        ],
    })


def analyze_numeric_column(df, column):
    """Print numerical univariate statistics."""

    series = df[column]

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    print(f"\n{'=' * 50}")
    print(f"Numerical Analysis: {column}")
    print(f"{'=' * 50}")

    print("Count:", series.count())
    print("Missing:", series.isna().sum())
    print("Unique:", series.nunique())
    print("Mean:", series.mean())
    print("Median:", series.median())
    print("Standard Deviation:", series.std())
    print("Minimum:", series.min())
    print("Q1:", q1)
    print("Q3:", q3)
    print("Maximum:", series.max())
    print("IQR:", iqr)
    print("Skewness:", series.skew())

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[
        (series < lower)
        | (series > upper)
    ]

    print("Potential outliers:", len(outliers))


def analyze_categorical_column(df, column):
    """Print categorical univariate statistics."""

    series = df[column]

    print(f"\n{'=' * 50}")
    print(f"Categorical Analysis: {column}")
    print(f"{'=' * 50}")

    print("Count:", series.count())
    print("Missing:", series.isna().sum())
    print("Unique:", series.nunique())

    print("\nFrequency:")
    print(
        series.value_counts(
            dropna=False
        )
    )

    print("\nPercentage:")
    print(
        series.value_counts(
            normalize=True,
            dropna=False
        ).mul(100).round(2)
    )


def plot_numeric(df, column):
    """Plot histogram and box plot."""

    values = df[column].dropna()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 8)
    )

    sns.histplot(
        values,
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title(
        f"{column} Distribution"
    )

    sns.boxplot(
        x=values,
        ax=axes[1]
    )

    axes[1].set_title(
        f"{column} Box Plot"
    )

    plt.tight_layout()
    plt.show()


def plot_categorical(df, column):
    """Plot category frequencies."""

    counts = (
        df[column]
        .value_counts()
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        x=counts.index,
        y=counts.values
    )

    plt.title(
        f"{column} Frequency"
    )

    plt.xlabel(column)
    plt.ylabel("Count")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()
    plt.show()


def main():
    df = create_dataset()

    print("Dataset:")
    print(df)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isna().sum())

    analyze_numeric_column(
        df,
        "age"
    )

    analyze_numeric_column(
        df,
        "salary"
    )

    analyze_categorical_column(
        df,
        "city"
    )

    plot_numeric(
        df,
        "age"
    )

    plot_numeric(
        df,
        "salary"
    )

    plot_categorical(
        df,
        "city"
    )


if __name__ == "__main__":
    main()
```

---

# 🧠 How to Interpret the Results

Suppose the salary analysis gives:

```text
Mean       → 71,333
Median     → 42,000
Std        → Large
Skewness   → Positive
```

A reasonable interpretation is:

> The salary distribution appears right-skewed, with relatively high values pulling the mean above the median. The distribution should be inspected visually and in domain context before deciding whether transformation or outlier treatment is appropriate.

This is stronger than simply saying:

> "There are outliers."

---

# 📋 Example Interpretation Template

For each numerical variable, document:

```text
Variable:
    salary

Type:
    Numerical

Missing:
    0%

Unique:
    150

Central Tendency:
    Mean = ...
    Median = ...

Spread:
    Std = ...
    IQR = ...

Shape:
    Right-skewed

Outliers:
    Potential high-value observations detected

Visualization:
    Histogram + Box Plot

Action:
    Investigate extreme values before modeling
```

For categorical variables:

```text
Variable:
    city

Type:
    Categorical

Missing:
    2%

Unique:
    8

Most Common:
    Pune

Distribution:
    Uneven

Rare Categories:
    2

Action:
    Review rare categories and missing values
```

---

# ⚠️ Common Mistakes

## 1. Looking only at the mean

The mean can hide skewness and extreme values.

Use:

```text
Mean
Median
Std
Quartiles
Histogram
Box Plot
```

together.

---

## 2. Treating every outlier as an error

An extreme observation may be:

* A legitimate customer
* A high-value transaction
* A rare medical measurement
* A real sensor event

Investigate before removing.

---

## 3. Ignoring missing values

Statistics calculated on incomplete data can be misleading.

Always inspect:

```python
df.isna().sum()
```

---

## 4. Using the wrong visualization

Use:

```text
Numerical → Histogram / Box Plot / KDE
Categorical → Bar Chart / Count Plot
```

---

## 5. Creating too many histogram bins

Excessive bins can make the distribution appear noisy.

---

## 6. Treating skewness as automatically bad

A skewed variable is not inherently problematic.

Some models and transformations are sensitive to distribution shape; others are less so.

---

## 7. Using pie charts for many categories

If there are many categories, a bar chart is generally easier to interpret.

---

## 8. Ignoring rare categories

Rare categories can affect:

* Model stability
* Encoding
* Feature dimensionality
* Generalization

---

## 9. Automatically transforming every skewed feature

Transformation should be justified by the data, model, and objective.

---

## 10. Confusing univariate analysis with feature selection

Univariate analysis describes individual variables.

A variable may appear useful individually but have limited predictive value after considering other variables.

---

# ✅ Best Practices

### 1. Identify variable type first

```text
Numerical
Categorical
Ordinal
Date-Time
Text
```

---

### 2. Combine statistics and visualization

Do not rely exclusively on one.

```text
Statistics
    +
Visualization
    =
Better Understanding
```

---

### 3. Always inspect missing values

```python
df.isna().sum()
```

---

### 4. Check distributions

For numerical variables:

```python
df["column"].describe()
```

and:

```python
sns.histplot(
    data=df,
    x="column",
    kde=True
)
```

---

### 5. Check outliers

Use:

* IQR
* Box plots
* Domain knowledge
* Robust statistics

---

### 6. Analyze categorical frequency

```python
df["category"].value_counts(
    normalize=True
)
```

---

### 7. Document findings

Good EDA should produce conclusions, not just charts.

---

### 8. Use domain knowledge

Statistical anomalies may be perfectly valid real-world observations.

---

### 9. Keep EDA reproducible

Put analysis into:

```text
.py
.ipynb
README.md
```

rather than relying only on manually created charts.

---

### 10. Separate exploration from production preprocessing

EDA helps you understand the data.

Final ML transformations should be implemented carefully, especially when fitting transformations from training data to avoid leakage.

---

# 🧪 Mini Projects

## 🏠 Project 1 — House Prices

Analyze:

```text
price
area
bedrooms
bathrooms
city
```

Tasks:

* Calculate mean price
* Calculate median price
* Analyze price distribution
* Detect potential outliers
* Analyze bedroom frequency
* Analyze city distribution

---

## 👨‍💼 Project 2 — Employee Dataset

Analyze:

```text
age
salary
department
experience
education
```

Tasks:

* Analyze salary distribution
* Find median experience
* Identify rare departments
* Detect potential age outliers
* Analyze education categories

---

## 🛒 Project 3 — E-Commerce Dataset

Analyze:

```text
order_amount
product_category
customer_age
payment_method
```

Tasks:

* Analyze order amount
* Find most common category
* Analyze payment methods
* Detect extreme transactions
* Study customer age distribution

---

## 🌡️ Project 4 — Weather Dataset

Analyze:

```text
temperature
humidity
wind_speed
weather_condition
```

Tasks:

* Analyze temperature distribution
* Find median humidity
* Detect extreme wind speeds
* Analyze weather-condition frequency
* Identify missing values

---

# 📝 Exercises

## 🟢 Beginner

1. Define univariate analysis.
2. What is the difference between numerical and categorical variables?
3. Calculate the mean of a list.
4. Calculate the median.
5. Find the mode.
6. Count missing values.
7. Find unique categories.
8. Create a frequency table.

---

## 🟡 Intermediate

9. Calculate standard deviation.
10. Calculate IQR.
11. Detect potential outliers using IQR.
12. Create a histogram.
13. Create a box plot.
14. Calculate skewness.
15. Calculate category percentages.
16. Create a reusable numerical-summary function.

---

## 🟠 Advanced

17. Build an automated univariate report.
18. Compare mean and median for skewed data.
19. Analyze a highly imbalanced categorical variable.
20. Detect rare categories.
21. Compare histogram and ECDF interpretations.
22. Investigate potential outliers using domain knowledge.
23. Create an automated visualization pipeline.
24. Write an EDA findings report.

---

# 📁 Project Structure

Recommended structure:

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
│   └── README.md
│
├── 03-Multivariate-Analysis/
│   └── README.md
│
├── 04-Data-Visualization/
│   └── README.md
│
└── 05-EDA-Projects/
    └── README.md
```

---

# 🧾 Univariate Analysis Checklist

Before moving to the next EDA stage:

### Dataset

* [ ] Dataset loaded successfully
* [ ] Shape inspected
* [ ] Data types checked
* [ ] Missing values checked
* [ ] Duplicate records checked

### Numerical Variables

* [ ] Mean calculated
* [ ] Median calculated
* [ ] Standard deviation calculated
* [ ] Minimum checked
* [ ] Maximum checked
* [ ] Quartiles calculated
* [ ] IQR calculated
* [ ] Skewness checked
* [ ] Distribution visualized
* [ ] Potential outliers investigated

### Categorical Variables

* [ ] Unique categories checked
* [ ] Frequency calculated
* [ ] Percentage calculated
* [ ] Most common category identified
* [ ] Rare categories investigated
* [ ] Missing categories checked
* [ ] Distribution visualized

### Findings

* [ ] Important patterns documented
* [ ] Anomalies investigated
* [ ] Data-quality issues recorded
* [ ] Potential preprocessing requirements identified

---

# 🔄 EDA Roadmap

This repository follows a structured EDA progression:

```text
06-Exploratory-Data-Analysis/
│
├── 01-Univariate-Analysis/
│       ↓
│   One variable
│       ↓
├── 02-Bivariate-Analysis/
│       ↓
│   Two variables
│       ↓
├── 03-Multivariate-Analysis/
│       ↓
│   Multiple variables
│       ↓
├── 04-Data-Visualization/
│       ↓
│   Communicate insights
│       ↓
└── 05-EDA-Projects/
        ↓
    Apply everything
```

---

# 🧠 Univariate Analysis Decision Guide

```text
                 Variable
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
      Numerical           Categorical
          │                   │
     ┌────┴────┐         ┌────┴────┐
     ↓         ↓         ↓         ↓
 Continuous  Discrete  Nominal   Ordinal
     │         │         │         │
     └────┬────┘         └────┬────┘
          ↓                   ↓
      Statistics          Frequencies
          ↓                   ↓
   Histogram / KDE       Bar / Count Plot
          ↓                   ↓
       Box Plot          Rare Categories
          │                   │
          └─────────┬─────────┘
                    ↓
              🧠 Insights
```

---

# 🎯 Key Takeaways

1. **Univariate analysis studies one variable at a time.**
2. Always identify the variable type before choosing an analysis method.
3. Numerical variables can be analyzed using statistics and distributions.
4. Categorical variables are commonly analyzed using frequency and percentage.
5. Mean, median, and mode describe central tendency.
6. Range, variance, standard deviation, and IQR describe dispersion.
7. Histograms reveal numerical distribution shapes.
8. Box plots are useful for identifying potential outliers.
9. Skewness describes distribution asymmetry.
10. Outliers should be investigated rather than automatically deleted.
11. Missing values must be considered when interpreting statistics.
12. ECDFs provide a bin-free view of cumulative distributions.
13. Visualization and statistical summaries should complement each other.
14. Domain knowledge is essential when interpreting unusual observations.
15. Good EDA produces **insights and decisions**, not just charts.

---

# 🚀 Next Step

After understanding one variable at a time, move to:

```text
06-Exploratory-Data-Analysis/02-Bivariate-Analysis/
```

There you will study **relationships between two variables**, including:

* Numerical vs Numerical
* Numerical vs Categorical
* Categorical vs Categorical
* Correlation
* Covariance
* Scatter plots
* Grouped statistics
* Cross-tabulation
* Relationship analysis

---

# 🌱 The EDA Mindset

```text
          📂 RAW DATA
               ↓
          🔍 INSPECT
               ↓
          🧹 CHECK QUALITY
               ↓
       📊 UNIVARIATE ANALYSIS
               ↓
       🔗 BIVARIATE ANALYSIS
               ↓
       🧩 MULTIVARIATE ANALYSIS
               ↓
          📈 VISUALIZE
               ↓
          🧠 FIND INSIGHTS
               ↓
        🤖 PREPARE FOR ML
               ↓
             🚀 BUILD
```

> **First understand each variable. Then understand how variables interact. Finally, use those insights to build better Machine Learning solutions.**

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

GitHub:
`https://github.com/Kishor055`

---

# 🤝 Contributing

Contributions are welcome.

If you find an error, missing concept, incorrect example, or improvement opportunity:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test the examples.
5. Commit your changes.
6. Open a Pull Request.

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 📚 Share it with other learners

---

# 📄 License

This project is intended for educational purposes.

See the repository root for the applicable license.

---

**🌱 GROW → 📚 LEARN → 📊 ANALYZE → 🧠 UNDERSTAND → 💻 BUILD → 🚀 DEPLOY**

# 🕳️ Missing Values in Machine Learning

Missing values are one of the most common problems encountered when working with real-world datasets.

In practical Machine Learning projects, data is rarely perfect. Some observations may have missing:

* Age
* Salary
* Income
* Address
* Category
* Measurements
* Dates
* Labels
* Sensor readings
* Transaction information

If missing values are not handled correctly, they can cause:

* Incorrect statistical analysis
* Data loss
* Model training errors
* Biased predictions
* Incorrect feature relationships
* Reduced model performance
* Data leakage
* Production failures

This chapter explains how to **detect, understand, analyze, visualize, and handle missing values** using Python and Pandas, from beginner concepts to production-oriented strategies.

---

# 📚 Table of Contents

1. [What Are Missing Values?](#-what-are-missing-values)
2. [Why Missing Values Occur](#-why-missing-values-occur)
3. [Types of Missingness](#-types-of-missingness)
4. [Missing Value Representations](#-missing-value-representations)
5. [Creating Missing Values](#-creating-missing-values)
6. [Detecting Missing Values](#-detecting-missing-values)
7. [Counting Missing Values](#-counting-missing-values)
8. [Missing Value Percentage](#-missing-value-percentage)
9. [Dataset-Level Missingness](#-dataset-level-missingness)
10. [Column-Level Analysis](#-column-level-analysis)
11. [Row-Level Analysis](#-row-level-analysis)
12. [Missingness Patterns](#-missingness-patterns)
13. [Understanding Why Values Are Missing](#-understanding-why-values-are-missing)
14. [Deleting Missing Values](#-deleting-missing-values)
15. [Dropping Rows](#-dropping-rows)
16. [Dropping Columns](#-dropping-columns)
17. [Mean Imputation](#-mean-imputation)
18. [Median Imputation](#-median-imputation)
19. [Mode Imputation](#-mode-imputation)
20. [Constant Imputation](#-constant-imputation)
21. [Forward Fill](#-forward-fill)
22. [Backward Fill](#-backward-fill)
23. [Interpolation](#-interpolation)
24. [Group-Based Imputation](#-group-based-imputation)
25. [KNN Imputation](#-knn-imputation)
26. [Iterative Imputation](#-iterative-imputation)
27. [Missing Indicator Features](#-missing-indicator-features)
28. [Numerical vs Categorical Missing Values](#-numerical-vs-categorical-missing-values)
29. [Time-Series Missing Values](#-time-series-missing-values)
30. [Target Missing Values](#-target-missing-values)
31. [Train/Test Leakage](#-traintest-leakage)
32. [Missing Values and Outliers](#-missing-values-and-outliers)
33. [Missing Values and Data Quality](#-missing-values-and-data-quality)
34. [Pandas Implementation](#-pandas-implementation)
35. [Reusable Functions](#-reusable-functions)
36. [Scikit-Learn Pipelines](#-scikit-learn-pipelines)
37. [Production Considerations](#-production-considerations)
38. [Common Mistakes](#-common-mistakes)
39. [Best Practices](#-best-practices)
40. [Mini Projects](#-mini-projects)
41. [Exercises](#-exercises)
42. [Project Structure](#-project-structure)
43. [End-to-End Example](#-end-to-end-example)
44. [Missing Value Checklist](#-missing-value-checklist)
45. [Decision Framework](#-decision-framework)
46. [Roadmap](#-roadmap)
47. [Key Takeaways](#-key-takeaways)
48. [Next Step](#-next-step)

---

# 🎯 What Are Missing Values?

A missing value means that a data field does not contain a usable observation.

Example:

```text
Name       Age    Salary
-------------------------
Amit       25     50000
Priya      NaN    60000
Rahul      31     NaN
Sneha      28     72000
```

Here:

```text
Age → missing for Priya
Salary → missing for Rahul
```

In Pandas, missing data may be represented using:

```python
NaN
None
pd.NA
NaT
```

---

# 🤔 Why Missing Values Occur

Missing values can occur for many reasons.

## 1. User did not provide information

Example:

```text
Age = missing
```

## 2. Sensor failure

```text
Temperature = missing
```

## 3. Database issue

```text
Column not populated
```

## 4. API failure

```text
Field absent from response
```

## 5. Data integration

One dataset contains:

```text
customer_id
name
```

Another contains:

```text
customer_id
income
```

Some records may not match.

## 6. Feature not applicable

For example:

```text
Number of children
```

may not apply to certain records depending on the data definition.

## 7. Data collection errors

A form or process may fail to record a value.

---

# 🧠 Types of Missingness

A major concept in statistics and Machine Learning is the **mechanism behind missingness**.

There are three commonly discussed categories:

```text
MCAR
MAR
MNAR
```

---

# 1️⃣ MCAR — Missing Completely At Random

The probability of a value being missing is unrelated to observed or unobserved variables.

Conceptually:

```text
Missingness
    │
    ├── Not related to observed data
    └── Not related to missing value itself
```

Example:

A small number of records are randomly corrupted during file transfer.

MCAR is often the simplest missingness mechanism to reason about.

---

# 2️⃣ MAR — Missing At Random

Missingness depends on other observed variables.

Example:

```text
Income missing
      ↑
      │
      └── Education level
```

Suppose people with a particular education category are more likely to omit income.

The missingness depends on information already observed in the dataset.

---

# 3️⃣ MNAR — Missing Not At Random

Missingness depends on the missing value itself or an unobserved factor.

Example:

```text
High income
    ↓
More likely to refuse reporting income
```

The probability of missingness is related to the value that is missing.

MNAR can be difficult to handle because the reason for missingness is not fully observable.

---

# 🔍 Why Missingness Mechanisms Matter

Consider:

```text
Age
Income
Education
```

If income is missing randomly, simple imputation may be reasonable.

But if:

```text
High-income users
        ↓
more likely to omit income
```

then replacing missing income with the overall median may distort the distribution.

Therefore:

> **Do not choose an imputation strategy before understanding the missingness mechanism and business context.**

---

# 🧾 Missing Value Representations

Missing values may appear as:

```text
NaN
None
NaT
pd.NA
""
" "
"NA"
"N/A"
"null"
"NULL"
"unknown"
"-"
"?"
```

Some are actual missing values.

Others are strings that **represent** missingness.

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 30, "NA", 40]
})

print(df)
```

Here `"NA"` is a string, not necessarily a Pandas missing value.

Normalize representations carefully:

```python
df["age"] = df["age"].replace(
    ["NA", "N/A", "null", "?"],
    pd.NA
)
```

---

# 🧪 Creating Missing Values

Example:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name": ["A", "B", "C", "D"],
    "age": [25, np.nan, 30, 40],
    "salary": [50000, 60000, np.nan, 80000]
})

print(df)
```

Output:

```text
  name   age   salary
0    A  25.0  50000.0
1    B   NaN  60000.0
2    C  30.0      NaN
3    D  40.0  80000.0
```

---

# 🔎 Detecting Missing Values

The most common Pandas method is:

```python
df.isna()
```

Example:

```python
print(df.isna())
```

Result:

```text
    name    age  salary
0  False  False   False
1  False   True   False
2  False  False    True
3  False  False   False
```

---

# 🔍 `isna()` vs `isnull()`

These are equivalent in Pandas:

```python
df.isna()
```

and:

```python
df.isnull()
```

For modern code, `.isna()` is clear and commonly preferred.

---

# 🔢 Counting Missing Values

Count missing values per column:

```python
missing_count = df.isna().sum()

print(missing_count)
```

Example:

```text
name      0
age       1
salary    1
```

---

# 📊 Missing Value Percentage

Count alone is not enough.

For example:

```text
Dataset A:
10 missing values / 10,000 rows = 0.1%

Dataset B:
10 missing values / 20 rows = 50%
```

Calculate the percentage:

```python
missing_rate = df.isna().mean()

print(missing_rate)
```

Convert to percentage:

```python
missing_percentage = (
    df.isna().mean() * 100
)

print(missing_percentage)
```

---

# 📋 Missing Value Report

A useful report:

```python
missing_report = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_rate": df.isna().mean(),
    "present_count": df.notna().sum(),
})

print(missing_report)
```

Sort by missing rate:

```python
missing_report = missing_report.sort_values(
    "missing_rate",
    ascending=False
)

print(missing_report)
```

---

# 📦 Dataset-Level Missingness

Calculate the total number of missing cells:

```python
total_missing = df.isna().sum().sum()

print("Total missing values:", total_missing)
```

Total cells:

```python
total_cells = df.size

print("Total cells:", total_cells)
```

Overall missing rate:

```python
overall_missing_rate = (
    df.isna().sum().sum() / df.size
)

print(
    f"Overall missing rate: "
    f"{overall_missing_rate:.2%}"
)
```

---

# 📌 Column-Level Analysis

Find columns containing missing values:

```python
columns_with_missing = df.columns[
    df.isna().any()
]

print(columns_with_missing.tolist())
```

Find columns with no missing values:

```python
complete_columns = df.columns[
    df.notna().all()
]

print(complete_columns.tolist())
```

---

# 📌 Row-Level Analysis

Find rows containing at least one missing value:

```python
rows_with_missing = df[
    df.isna().any(axis=1)
]

print(rows_with_missing)
```

Find rows with all values missing:

```python
empty_rows = df[
    df.isna().all(axis=1)
]

print(empty_rows)
```

---

# 🔬 Missingness Patterns

Sometimes missing values occur together.

Example:

```text
Age     Income    Education
✔       ✔         ✔
✔       ❌        ✔
❌       ❌        ✔
✔       ❌        ❌
```

Check missingness correlation:

```python
missing_matrix = df.isna().astype(int)

print(
    missing_matrix.corr()
)
```

This can help identify whether missingness in one feature is associated with missingness in another.

---

# 📈 Visualizing Missingness

A simple Pandas-based visualization:

```python
import matplotlib.pyplot as plt

missing_rate = df.isna().mean()

missing_rate.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.ylabel("Missing Rate")
plt.title("Missing Values by Column")
plt.tight_layout()
plt.show()
```

For larger projects, dedicated data-profiling and visualization libraries can provide richer missingness analysis.

---

# 🧠 Understanding Why Values Are Missing

Before choosing a strategy, ask:

### Question 1

Why is the value missing?

### Question 2

Is missingness random?

### Question 3

Is missingness related to another feature?

### Question 4

Is the missing value meaningful?

### Question 5

Would replacing it change the meaning of the dataset?

### Question 6

Was the value unavailable at prediction time?

These questions are often more important than the imputation technique itself.

---

# 🗑️ Deleting Missing Values

There are two broad approaches:

```text
Missing Values
      │
      ├── Remove
      │
      └── Impute
```

Removal can be appropriate when:

* Very few values are missing.
* Records are not important.
* Missingness is approximately random.
* Removing observations does not introduce bias.
* The dataset remains sufficiently large.

---

# 🧹 Dropping Rows

Remove rows containing any missing value:

```python
clean_df = df.dropna()
```

Example:

```text
Before:
4 rows

After:
2 rows
```

Be careful: this can remove a large amount of information.

---

# 🎯 Dropping Rows Based on Specific Columns

```python
clean_df = df.dropna(
    subset=["age", "salary"]
)
```

This means rows are removed only when `age` or `salary` is missing.

---

# 🧹 Dropping Columns

If a feature contains an extremely high proportion of missing values and provides limited value, removing the column may be considered.

Example:

```python
clean_df = df.drop(
    columns=["rare_feature"]
)
```

But do not automatically drop columns simply because they contain missing values.

Consider:

* Feature importance
* Business meaning
* Missingness mechanism
* Cost of collecting the feature
* Alternative imputation strategies

---

# 📊 Mean Imputation

For numerical data:

```python
mean_value = df["age"].mean()

df["age"] = df["age"].fillna(
    mean_value
)
```

Example:

```text
Age:
20
25
NaN
30

Mean = 25

After:
20
25
25
30
```

### Advantages

* Simple
* Fast
* Easy to explain

### Limitations

* Sensitive to outliers
* Reduces variance
* Can distort distributions

---

# 📊 Median Imputation

```python
median_value = df["age"].median()

df["age"] = df["age"].fillna(
    median_value
)
```

Median is often more robust than mean when data is skewed.

Example:

```text
Income:
30000
35000
40000
50000
1000000
```

The extreme value can heavily influence the mean.

The median is less sensitive.

---

# 🏷️ Mode Imputation

For categorical variables:

```python
mode_value = df["city"].mode()[0]

df["city"] = df["city"].fillna(
    mode_value
)
```

Example:

```text
City:
Pune
Mumbai
Pune
NaN
Pune
```

Mode:

```text
Pune
```

Missing value becomes:

```text
Pune
```

### Limitation

If the missingness is meaningful, replacing everything with the most common category can hide information.

---

# 🏷️ Constant Imputation

Categorical example:

```python
df["city"] = df["city"].fillna(
    "Unknown"
)
```

Numerical example:

```python
df["income"] = df["income"].fillna(
    0
)
```

However:

> **Zero should only be used when zero is semantically correct.**

Missing income does not automatically mean:

```text
income = 0
```

---

# ➡️ Forward Fill

Forward fill uses the previous valid observation.

```python
df["temperature"] = df["temperature"].ffill()
```

Example:

```text
10
12
NaN
NaN
15
```

After forward fill:

```text
10
12
12
12
15
```

Useful for certain ordered/time-series datasets.

Not appropriate for every dataset.

---

# ⬅️ Backward Fill

Backward fill uses the next valid observation.

```python
df["temperature"] = df["temperature"].bfill()
```

Example:

```text
10
NaN
NaN
15
```

Becomes:

```text
10
15
15
15
```

Again, this should be justified by the data-generating process.

---

# 📈 Interpolation

Interpolation estimates values between known observations.

```python
df["temperature"] = df["temperature"].interpolate()
```

Example:

```text
10
12
NaN
16
```

Linear interpolation may produce:

```text
10
12
14
16
```

Interpolation is particularly useful for some continuous time-series measurements.

---

# 👥 Group-Based Imputation

Sometimes global statistics are inappropriate.

Example:

```text
Department
Salary
```

Different departments may have different salary distributions.

Instead of:

```python
df["salary"] = df["salary"].fillna(
    df["salary"].median()
)
```

use group-specific values:

```python
df["salary"] = df["salary"].fillna(
    df.groupby("department")["salary"]
      .transform("median")
)
```

Conceptually:

```text
Department
    │
    ├── Engineering → Engineering median
    ├── Marketing   → Marketing median
    └── Finance     → Finance median
```

This can preserve more structure.

---

# 🧠 KNN Imputation

K-Nearest Neighbors can estimate missing values based on similar observations.

Scikit-Learn:

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(
    n_neighbors=5
)

X_imputed = imputer.fit_transform(X)
```

Conceptually:

```text
Missing Record
      │
      ▼
Find Similar Records
      │
      ▼
Use Neighbor Information
      │
      ▼
Estimate Missing Value
```

### Advantages

* Uses relationships between features.
* Can capture local structure.

### Limitations

* Computationally more expensive.
* Sensitive to feature scales.
* Requires careful preprocessing.
* Can perform poorly when similarity is not meaningful.

---

# 🔁 Iterative Imputation

Iterative imputation estimates one feature using other features.

Conceptually:

```text
Feature A
   ↓
Feature B
   ↓
Feature C
   ↓
Estimate Missing Values
   ↓
Repeat
```

Scikit-Learn:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    random_state=42
)

X_imputed = imputer.fit_transform(X)
```

This can be more sophisticated than simple mean/median imputation, but it requires careful validation.

---

# 🚩 Missing Indicator Features

Sometimes the fact that a value is missing is itself informative.

Example:

```text
income = NaN
income_missing = 1
```

Create an indicator:

```python
df["income_missing"] = (
    df["income"].isna().astype(int)
)
```

Then impute:

```python
df["income"] = df["income"].fillna(
    df["income"].median()
)
```

Now the model receives:

```text
income
income_missing
```

This allows the model to distinguish:

```text
Actual median income
```

from:

```text
Originally missing income
```

---

# 🔢 Numerical vs Categorical Missing Values

## Numerical

Common strategies:

```text
Mean
Median
Group Median
KNN
Iterative
Model-Based
```

## Categorical

Common strategies:

```text
Mode
"Unknown"
"Missing"
Group-Based
Model-Based
```

---

# 📅 Time-Series Missing Values

Time-series data requires special attention.

Example:

```text
Timestamp     Temperature
09:00         20
10:00         21
11:00         NaN
12:00         24
```

Possible strategies:

```text
Forward Fill
Backward Fill
Interpolation
Rolling Statistics
Domain-Based Imputation
```

Before using a method, understand whether the process changes smoothly over time.

---

# 🎯 Target Missing Values

Missing target values are different from missing features.

Example:

```text
Age     Income     Churn
25      50000      Yes
30      60000      No
35      70000      NaN
```

If `Churn` is the supervised-learning target, the third row does not have a known label.

Usually, you should not casually impute a classification target using:

```python
df["churn"] = df["churn"].fillna(
    "No"
)
```

That creates artificial labels.

Depending on the project, target-missing rows may need to be:

* Removed from supervised training.
* Reserved for future labeling.
* Investigated separately.
* Handled through a domain-specific process.

---

# ⚠️ Train/Test Leakage

One of the most important rules in Machine Learning:

> **Never use information from validation or test data to calculate training-time imputation statistics.**

Incorrect:

```python
median = df["age"].median()

df["age"] = df["age"].fillna(median)

train = df.iloc[:800]
test = df.iloc[800:]
```

The median was calculated using the entire dataset.

This can leak information from the test set.

---

# ✅ Correct Approach

Split first:

```python
train = df.iloc[:800].copy()
test = df.iloc[800:].copy()

median = train["age"].median()

train["age"] = train["age"].fillna(median)
test["age"] = test["age"].fillna(median)
```

The test set does not influence the imputation statistic.

For real ML pipelines, prefer Scikit-Learn transformers.

---

# 🛡️ Missing Values with Scikit-Learn Pipelines

Example:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

numeric_features = [
    "age",
    "income"
]

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    )
])

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression()
    )
])
```

Training:

```python
model.fit(X_train, y_train)
```

Prediction:

```python
predictions = model.predict(X_test)
```

The imputer learns its statistics from the training data during `.fit()`.

---

# 🔬 Missing Values and Outliers

Consider:

```text
Salary:
30000
35000
40000
45000
5000000
NaN
```

If you calculate the mean before understanding the outlier:

```python
mean = df["salary"].mean()
```

the extreme value may heavily influence the imputation.

A better approach may involve:

1. Investigating outliers.
2. Understanding the domain.
3. Choosing an appropriate statistic.
4. Validating the result.

---

# 📊 Missing Values and Data Quality

Missingness should be included in your data-quality report.

Example:

```python
quality_report = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "missing_rate": df.isna().mean(),
    "unique_count": df.nunique(dropna=False)
})

print(quality_report)
```

---

# 🐼 Pandas Implementation

Complete example:

```python
import pandas as pd
import numpy as np


def main():
    df = pd.DataFrame({
        "name": ["A", "B", "C", "D", "E"],
        "age": [25, np.nan, 30, 35, np.nan],
        "salary": [
            50000,
            60000,
            np.nan,
            80000,
            90000
        ],
        "city": [
            "Pune",
            "Mumbai",
            np.nan,
            "Pune",
            "Delhi"
        ]
    })

    print("Original Dataset:")
    print(df)

    print("\nMissing Values:")
    print(df.isna().sum())

    print("\nMissing Percentage:")
    print(df.isna().mean() * 100)

    # Numerical imputation
    df["age"] = df["age"].fillna(
        df["age"].median()
    )

    df["salary"] = df["salary"].fillna(
        df["salary"].median()
    )

    # Categorical imputation
    df["city"] = df["city"].fillna(
        "Unknown"
    )

    print("\nAfter Imputation:")
    print(df)


if __name__ == "__main__":
    main()
```

---

# 🔧 Reusable Functions

## Missing-value report

```python
def missing_value_report(df):
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean(),
        "non_missing_count": df.notna().sum(),
    })

    return report.sort_values(
        "missing_rate",
        ascending=False
    )
```

Usage:

```python
report = missing_value_report(df)

print(report)
```

---

## Remove high-missing columns

```python
def drop_high_missing_columns(
    df,
    threshold=0.8
):
    missing_rate = df.isna().mean()

    columns_to_drop = missing_rate[
        missing_rate > threshold
    ].index

    return df.drop(
        columns=columns_to_drop
    )
```

Usage:

```python
df = drop_high_missing_columns(
    df,
    threshold=0.8
)
```

---

## Median imputation

```python
def median_impute(
    df,
    columns
):
    df = df.copy()

    for column in columns:
        median = df[column].median()

        df[column] = df[column].fillna(
            median
        )

    return df
```

Usage:

```python
df = median_impute(
    df,
    ["age", "salary"]
)
```

---

# 🧪 Testing Imputation

After imputation, always check whether missing values remain.

```python
remaining_missing = df.isna().sum()

print(remaining_missing)
```

Or:

```python
print(
    df.isna().sum().sum()
)
```

A result of:

```text
0
```

means there are no Pandas-detectable missing values left.

But this does **not automatically mean the data is correct**.

You should also verify:

* Distributions
* Statistics
* Categories
* Relationships
* Business rules

---

# 🏭 Production Considerations

In production systems, missing-value handling should be deterministic and reproducible.

Store:

```text
Imputation Method
Training Statistic
Feature Name
Version
Timestamp
Validation Result
```

Example:

```yaml
imputation:
  age:
    method: median
    value: 31.0

  income:
    method: median
    value: 65000
```

This helps reproduce the same preprocessing behavior.

---

# 🚨 Production Missing-Data Monitoring

Suppose training data has:

```text
income missing rate = 2%
```

Production suddenly has:

```text
income missing rate = 35%
```

The model may receive a very different input distribution.

Monitor:

```text
Missing Rate
     ↓
Training vs Production
     ↓
Compare
     ↓
Threshold
     ↓
Alert
```

Example:

```python
training_rate = 0.02
production_rate = 0.35

if production_rate > training_rate + 0.10:
    print("ALERT: Missingness increased significantly.")
```

Real production systems should use domain-specific thresholds rather than arbitrary constants.

---

# ❌ Common Mistakes

## 1. Replacing all missing values with zero

```python
df = df.fillna(0)
```

This can completely change the meaning of the dataset.

---

## 2. Using mean everywhere

Mean imputation is not universally appropriate.

---

## 3. Removing all rows with missing values

```python
df = df.dropna()
```

This can cause:

* Large data loss
* Selection bias
* Reduced training data
* Distorted distributions

---

## 4. Ignoring missingness patterns

Missingness itself can contain useful information.

---

## 5. Calculating imputation statistics on the entire dataset

This can introduce leakage.

---

## 6. Imputing the target blindly

Target labels require special handling.

---

## 7. Ignoring categorical missing values

Categorical columns need their own strategy.

---

## 8. Using forward fill on unordered data

Forward fill assumes an ordering relationship.

It should generally be used only when that ordering is meaningful.

---

## 9. Assuming zero means missing

```text
0 ≠ Missing
```

unless the domain explicitly defines them as equivalent.

---

# ✅ Best Practices

1. Detect missing values before modifying them.
2. Measure missing rates.
3. Understand why values are missing.
4. Analyze missingness patterns.
5. Distinguish numerical and categorical features.
6. Do not assume missing means zero.
7. Do not automatically delete missing rows.
8. Use median for skewed numerical data when appropriate.
9. Use mode or explicit categories for categorical data when appropriate.
10. Consider missing indicators.
11. Use group-based imputation when domain structure supports it.
12. Fit imputers only on training data.
13. Use Scikit-Learn pipelines for production ML.
14. Validate the dataset after imputation.
15. Monitor missingness in production.
16. Document imputation rules.
17. Preserve the original dataset.
18. Reassess imputation when the data distribution changes.

---

# 🧪 Mini Projects

## Project 1 — Student Missing Data Analyzer

Dataset:

```text
student_id
age
gender
attendance
math_score
science_score
```

Tasks:

* Detect missing values.
* Calculate missing percentages.
* Visualize missingness.
* Impute numerical features.
* Handle categorical features.
* Compare before/after statistics.

---

## Project 2 — Customer Dataset

Columns:

```text
customer_id
age
income
city
subscription
```

Implement:

```text
Missing Report
     ↓
Column Classification
     ↓
Imputation Strategy
     ↓
Validation
     ↓
Final Dataset
```

---

## Project 3 — Time-Series Imputation

Create a temperature dataset containing missing observations.

Compare:

```text
Forward Fill
Backward Fill
Linear Interpolation
Rolling Mean
```

Analyze how each method affects the time series.

---

## Project 4 — ML Pipeline

Build a classification model using:

```text
Numerical Features
      ↓
Median Imputation
      ↓
Scaling
      ↓
Categorical Features
      ↓
Most-Frequent Imputation
      ↓
One-Hot Encoding
      ↓
Classifier
```

Use a Scikit-Learn `Pipeline` and `ColumnTransformer`.

---

# 📝 Exercises

### Beginner

1. Detect missing values in a DataFrame.
2. Count missing values by column.
3. Calculate missing percentages.
4. Find rows containing missing values.
5. Find columns containing missing values.
6. Replace a custom missing marker with `pd.NA`.

### Intermediate

7. Implement mean imputation.
8. Implement median imputation.
9. Implement mode imputation.
10. Compare mean and median imputation.
11. Implement group-based imputation.
12. Use forward fill on a time series.
13. Use interpolation.
14. Build a missing-value report.

### Advanced

15. Implement a missing indicator.
16. Build a KNN imputation pipeline.
17. Implement iterative imputation.
18. Compare multiple imputation strategies.
19. Evaluate imputation impact on model performance.
20. Build automated missingness monitoring.
21. Detect changes in production missing rates.
22. Create a reusable missing-value preprocessing module.

---

# 📁 Project Structure

Recommended structure:

```text
05-Data-Cleaning/
│
├── README.md
│
└── 01-Missing-Values/
    │
    ├── README.md
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── examples/
    │   ├── missing_detection.py
    │   ├── mean_imputation.py
    │   ├── median_imputation.py
    │   ├── categorical_imputation.py
    │   ├── group_imputation.py
    │   └── interpolation.py
    │
    ├── notebooks/
    │   └── missing_values.ipynb
    │
    └── tests/
        └── test_missing_values.py
```

---

# 🚀 End-to-End Example

```python
import pandas as pd
import numpy as np


def create_dataset():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [25, np.nan, 31, 45, np.nan],
        "income": [
            50000,
            60000,
            np.nan,
            90000,
            70000
        ],
        "city": [
            "Pune",
            "Mumbai",
            None,
            "Pune",
            "Delhi"
        ],
        "target": [0, 1, 0, 1, 0]
    })


def missing_report(df):
    return pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean(),
        "non_missing_count": df.notna().sum(),
    }).sort_values(
        "missing_rate",
        ascending=False
    )


def clean_data(df):
    df = df.copy()

    # Numerical features
    numerical_columns = [
        "age",
        "income"
    ]

    for column in numerical_columns:
        median = df[column].median()

        df[column] = df[column].fillna(
            median
        )

    # Categorical feature
    df["city"] = df["city"].fillna(
        "Unknown"
    )

    return df


def main():
    df = create_dataset()

    print("=" * 60)
    print("ORIGINAL DATASET")
    print("=" * 60)
    print(df)

    print("\n" + "=" * 60)
    print("MISSING VALUE REPORT")
    print("=" * 60)
    print(missing_report(df))

    cleaned_df = clean_data(df)

    print("\n" + "=" * 60)
    print("CLEANED DATASET")
    print("=" * 60)
    print(cleaned_df)

    print("\nRemaining missing values:")
    print(cleaned_df.isna().sum())


if __name__ == "__main__":
    main()
```

---

# 🧭 Decision Framework

There is no universal missing-value strategy.

Use the following reasoning process:

```text
             Missing Value
                   │
                   ▼
          Why is it missing?
                   │
          ┌────────┴────────┐
          │                 │
       Random?          Systematic?
          │                 │
          ▼                 ▼
    Analyze impact      Investigate cause
          │                 │
          └────────┬────────┘
                   ▼
           How much missing?
                   │
          ┌────────┴────────┐
          │                 │
       Very low          Very high
          │                 │
          ▼                 ▼
       Impute /          Consider
       investigate       dropping /
                        redesigning
                   │
                   ▼
           Choose strategy
                   │
                   ▼
              Validate
                   │
                   ▼
              Monitor
```

---

# 📊 Strategy Comparison

| Strategy      | Best For                       | Advantages           | Limitations                       |
| ------------- | ------------------------------ | -------------------- | --------------------------------- |
| Drop rows     | Very low missingness           | Simple               | Data loss                         |
| Drop columns  | Extremely high missingness     | Simple               | Feature loss                      |
| Mean          | Roughly symmetric numeric data | Easy                 | Sensitive to outliers             |
| Median        | Skewed numeric data            | Robust               | Can reduce variance               |
| Mode          | Categorical data               | Simple               | Can dominate minority classes     |
| Constant      | Meaningful unknown category    | Explicit             | May introduce artificial patterns |
| Forward fill  | Ordered/time-series data       | Simple               | Can propagate stale values        |
| Backward fill | Ordered/time-series data       | Simple               | Uses future observation           |
| Interpolation | Continuous time-series         | Smooth               | Requires meaningful ordering      |
| Group-based   | Heterogeneous populations      | Preserves groups     | More complexity                   |
| KNN           | Similar observations exist     | Uses local structure | More computationally expensive    |
| Iterative     | Relationships between features | More sophisticated   | More assumptions/complexity       |

---

# 🔬 Evaluating an Imputation Strategy

Do not evaluate imputation only by asking:

```text
"Did all NaN values disappear?"
```

Also evaluate:

### Distribution

```python
before.describe()
after.describe()
```

### Relationships

```python
before.corr(numeric_only=True)
after.corr(numeric_only=True)
```

### Model Performance

Compare:

```text
No Imputation Strategy
        ↓
Strategy A
        ↓
Strategy B
        ↓
Strategy C
```

using appropriate cross-validation.

### Data Quality

Check:

* Missing values
* Invalid values
* New outliers
* Distribution changes
* Category changes

---

# 🧠 Important Principle

The goal of missing-value handling is **not simply to produce a dataset with zero missing values**.

The real goal is:

> **Preserve as much useful information as possible while minimizing bias, leakage, distortion, and unnecessary data loss.**

---

# 📋 Missing Value Checklist

Before finalizing a dataset:

### Detection

* [ ] Missing values detected
* [ ] Missing representations standardized
* [ ] Missing count calculated
* [ ] Missing rate calculated

### Understanding

* [ ] Missingness mechanism considered
* [ ] Missingness patterns investigated
* [ ] Domain meaning understood
* [ ] Target missingness checked

### Strategy

* [ ] Row deletion evaluated
* [ ] Column deletion evaluated
* [ ] Numerical strategy selected
* [ ] Categorical strategy selected
* [ ] Time-series strategy selected where applicable

### ML Safety

* [ ] Train/test split considered
* [ ] Imputation fitted only on training data
* [ ] Leakage avoided
* [ ] Pipeline used where appropriate

### Validation

* [ ] Remaining missing values checked
* [ ] Distributions compared
* [ ] Relationships checked
* [ ] Model impact evaluated

### Production

* [ ] Strategy documented
* [ ] Imputation statistics versioned
* [ ] Missingness monitored
* [ ] Alerts defined

---

# 🗺️ Roadmap

This chapter is the first stage of the **Data Cleaning** section:

```text
05-Data-Cleaning/
│
├── 01-Missing-Values/
│       ↓
├── 02-Duplicate-Data/
│       ↓
├── 03-Data-Type-Conversion/
│       ↓
├── 04-Inconsistent-Data/
│       ↓
├── 05-Outlier-Handling/
│       ↓
├── 06-String-Cleaning/
│       ↓
├── 07-Date-Time-Cleaning/
│       ↓
├── 08-Categorical-Cleaning/
│       ↓
├── 09-Numerical-Transformation/
│       ↓
└── 10-Data-Cleaning-Pipeline/
```

The broader Machine Learning flow becomes:

```text
Data Collection
      ↓
Data Inspection
      ↓
Data Quality
      ↓
Missing Values
      ↓
Data Cleaning
      ↓
EDA
      ↓
Feature Engineering
      ↓
Feature Selection
      ↓
Modeling
```

---

# 🎯 Key Takeaways

1. Missing values are normal in real-world datasets.
2. Always detect missing values before cleaning.
3. Measure both missing counts and missing percentages.
4. Understand why data is missing.
5. MCAR, MAR, and MNAR describe different missingness mechanisms.
6. Missingness itself can sometimes contain useful information.
7. Dropping rows is not always the best solution.
8. Mean imputation is simple but can be sensitive to outliers.
9. Median imputation is often more robust for skewed numerical features.
10. Mode imputation can work for categorical features.
11. `"Unknown"` can be useful when missingness has semantic meaning.
12. Forward fill and interpolation require meaningful ordering.
13. Group-based imputation can preserve population structure.
14. KNN and iterative methods can capture feature relationships.
15. Missing target values require special treatment.
16. Imputation statistics must not be learned from test data.
17. Scikit-Learn pipelines help prevent preprocessing leakage.
18. Always validate the dataset after imputation.
19. Monitor missingness in production.
20. The best imputation strategy depends on the data, domain, and ML task.

---

# 🚀 Next Step

After Missing Values, continue with:

```text
05-Data-Cleaning/
        ↓
02-Duplicate-Data/
```

There you will learn how to identify and handle:

* Exact duplicate rows
* Duplicate identifiers
* Business-key duplicates
* Near duplicates
* Duplicate transactions
* Conflicting records
* Deduplication strategies
* Data integrity
* Duplicate detection pipelines

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

GitHub:

`https://github.com/Kishor055`

---

# 🤝 Contributing

Contributions are welcome.

To contribute:

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b improve-missing-values

git add .

git commit -m "Improve missing values documentation"

git push origin improve-missing-values
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
* 📚 Continue through the complete ML roadmap

---

## 📄 License

This project is intended for educational purposes and is available under the repository's license.

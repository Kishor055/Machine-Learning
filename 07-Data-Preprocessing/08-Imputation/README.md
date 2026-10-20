# 🧩 Missing Value Imputation in Python

🌱 **INSPECT → UNDERSTAND → SELECT → IMPUTE → VALIDATE → BUILD**

Welcome to **Imputation**, part of the Data Preprocessing module in the Machine Learning learning path. This guide explains how to handle missing values and prepare reliable datasets using Python, Pandas, NumPy, and Scikit-learn.

Missing values are a common challenge in real-world datasets. They can arise from incomplete forms, sensor failures, unavailable measurements, data integration errors, or information that was never collected. Choosing an appropriate imputation strategy helps preserve useful observations while preparing data for statistical analysis and machine learning.

This tutorial covers missing-data analysis, mean and median imputation, mode and constant imputation, missing indicators, KNN imputation, iterative imputation, time-series techniques, preprocessing pipelines, and leakage prevention.

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [What Is Imputation?](#-what-is-imputation)
3. [Understanding Missing Data](#-understanding-missing-data)
4. [Inspecting Missing Values](#-inspecting-missing-values)
5. [Reasons for Missing Data](#-reasons-for-missing-data)
6. [Deleting Missing Values](#-deleting-missing-values)
7. [Mean Imputation](#-mean-imputation)
8. [Median Imputation](#-median-imputation)
9. [Mode Imputation](#-mode-imputation)
10. [Constant Value Imputation](#-constant-value-imputation)
11. [Group-Based Imputation](#-group-based-imputation)
12. [Missing Value Indicators](#-missing-value-indicators)
13. [KNN Imputation](#-knn-imputation)
14. [Iterative Imputation](#-iterative-imputation)
15. [Time-Series Imputation](#-time-series-imputation)
16. [Imputation with Pandas](#-imputation-with-pandas)
17. [Imputation with Scikit-learn](#-imputation-with-scikit-learn)
18. [Handling Different Feature Types](#-handling-different-feature-types)
19. [Complete Machine Learning Example](#-complete-machine-learning-example)
20. [Validate Imputed Data](#-validate-imputed-data)
21. [Common Mistakes](#-common-mistakes)
22. [Best Practices](#-best-practices)
23. [Mini Projects](#-mini-projects)
24. [Practice Exercises](#-practice-exercises)
25. [Quick Revision](#-quick-revision)
26. [Project Structure](#-project-structure)
27. [Learning Roadmap](#-learning-roadmap)
28. [References](#-references)

---

## 🎯 Introduction

Real-world datasets frequently contain incomplete observations.

Consider the following student dataset:

| Student | Age | Marks | Attendance |
|---|---:|---:|---:|
| Amit | 20 | 85 | 90 |
| Priya | 21 | Missing | 95 |
| Rahul | Missing | 78 | 80 |
| Neha | 19 | 92 | Missing |
| Kiran | 22 | 88 | 85 |

Missing values may cause problems when calculating statistics, plotting distributions, or training machine learning models.

One solution is **imputation**, which replaces missing values using an appropriate estimate or a predefined value.

For example, if the nonmissing marks are `85`, `78`, `92`, and `88`, the median is `86.5`. That value could be used to fill the missing marks, provided median imputation is appropriate for the problem.

However, imputation is not always the correct response. Some records should be rejected, some fields should remain missing, and some missing values should be investigated at their source.

### Learning objectives

By completing this guide, you will learn to:

- Identify and summarize missing values.
- Understand common missing-data patterns.
- Compare deletion and imputation strategies.
- Apply mean, median, mode, and constant imputation.
- Add missingness indicators when useful.
- Use KNN and iterative imputation.
- Handle missing values in time-series data.
- Build Scikit-learn preprocessing pipelines.
- Prevent data leakage during model development.
- Validate the dataset after imputation.

## 🧠 What Is Imputation?

**Imputation** is the process of filling in missing values using a defined rule, statistical estimate, model, or domain-specific value.

For example:

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [20, 21, None, 23, 24]
})

print(df)
```

Output:

```text
    Age
0  20.0
1  21.0
2   NaN
3  23.0
4  24.0
```

Impute the missing age using the median:

```python
median_age = df["Age"].median()

df["Age_Imputed"] = df["Age"].fillna(median_age)

print(df)
```

The original `Age` column remains unchanged, while `Age_Imputed` contains the filled values.

### Why is imputation important?

Imputation can help:

- Preserve useful observations.
- Reduce information loss from deleting incomplete rows.
- Prepare numerical and categorical features for machine learning.
- Keep training and prediction inputs compatible.
- Maintain a consistent dataset schema.

However, imputation also introduces assumptions. For example, replacing every missing age with the median assumes that a single central estimate is an acceptable substitute for the missing observations.

No imputation strategy is universally best.

## 🔬 Understanding Missing Data

Before choosing an imputation strategy, understand why values are missing and whether that missingness contains useful information.

Three commonly discussed missing-data mechanisms are MCAR, MAR, and MNAR.

### 1. MCAR — Missing Completely at Random

A value is missing for reasons unrelated to the observed and unobserved data.

Example: a random technical failure prevents a small number of measurements from being recorded.

If missingness is genuinely MCAR, complete-case analysis may be less biased than under other mechanisms, although it can still reduce statistical precision.

### 2. MAR — Missing at Random

Missingness depends on observed variables.

Example: participants in one recorded age group are less likely to provide an optional income value.

Age is observed, and the probability of missing income is related to that observed information.

### 3. MNAR — Missing Not at Random

Missingness depends on the missing value itself or on information that is not observed.

Example: people with very high incomes may be less likely to report their income.

This mechanism can create bias that simple imputation does not resolve.

| Mechanism | Missingness depends on | General implication |
|---|---|---|
| MCAR | Neither observed nor unobserved values | Complete-case analysis may be reasonable in some cases |
| MAR | Other observed information | Conditional or model-based approaches may help |
| MNAR | Missing values themselves or unobserved information | Additional assumptions or sensitivity analysis may be necessary |

These mechanisms cannot generally be identified conclusively from a missing-value count alone. Domain knowledge and study design are important.

## 🔍 Inspecting Missing Values

Always examine the missingness pattern before modifying the dataset.

### 1. Detect missing values

Pandas uses values such as `NaN`, `None`, and `pd.NA` to represent missing information, depending on the data type.

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [20, 21, None, 23],
    "Marks": [85, None, 78, 92],
    "City": ["Mumbai", "Pune", None, "Delhi"]
})

print(df.isna())
```

### 2. Count missing values per column

```python
print(df.isna().sum())
```

### 3. Calculate missing percentages

```python
missing_counts = df.isna().sum()

missing_percentage = (
    df.isna().mean().mul(100).round(2)
)

missing_report = pd.DataFrame({
    "Missing_Count": missing_counts,
    "Missing_Percentage": missing_percentage
})

print(missing_report)
```

### 4. Count missing values per row

```python
df["Missing_Count"] = df.isna().sum(axis=1)

print(df)
```

If you add diagnostic columns to your DataFrame, keep track of whether they should become model features. A missing-count column is not automatically suitable as an input feature.

### 5. Find rows containing missing values

```python
missing_rows = df[df.isna().any(axis=1)]

print(missing_rows)
```

### 6. Find columns without missing values

```python
complete_columns = df.columns[df.notna().all()]

print(complete_columns.tolist())
```

### 7. Inspect data types and summary statistics

```python
df.info()

print(df.describe(include="all"))
```

Inspection helps identify which columns need numeric imputation, categorical imputation, date handling, or no imputation at all.

### 8. Visualize missingness

A simple missingness chart can help identify columns with many missing values.

```python
import matplotlib.pyplot as plt

missing_counts = df.isna().sum()
missing_counts = missing_counts[missing_counts > 0]

missing_counts.sort_values().plot(
    kind="barh",
    figsize=(8, 4)
)

plt.title("Missing Values by Column")
plt.xlabel("Number of Missing Values")
plt.ylabel("Column")
plt.tight_layout()
plt.show()
```

Visualization libraries can provide more advanced missingness plots, but a simple count chart is often a useful starting point.

## 🧹 Reasons for Missing Data

Missing values can result from several different problems:

- Users leave form fields blank.
- Sensors or devices fail to record measurements.
- Historical records lack information required by a newer system.
- Different datasets contain different columns.
- Data is suppressed for privacy or confidentiality.
- Measurements are not applicable to certain observations.
- Data ingestion or parsing fails.
- Invalid placeholders are mistakenly interpreted as real values.

### Distinguish missingness from invalid values

For example:

```python
df = pd.DataFrame({
    "Age": [20, None, -5, 25],
    "Marks": [85, 90, 150, None]
})
```

Here:

- `None` is missing information.
- Age `-5` may violate the age rules.
- Marks `150` may violate an examination's scoring range.

Replacing invalid values with `NaN` may be appropriate after validation, but do not treat every unusual value as missing without checking the domain.

For example, if `-999` is a documented missing-value sentinel in a sensor dataset, convert it to missing. If it is a legitimate measurement in the relevant domain, retain it.

## 🗑️ Deleting Missing Values

Before imputing, consider whether deletion is appropriate.

### 1. Remove rows with any missing values

```python
df_complete = df.dropna()
```

This may be reasonable when the dataset is small, the missingness pattern supports it, and losing those observations is acceptable.

### 2. Remove rows only when selected fields are missing

```python
df_complete = df.dropna(
    subset=["Age", "Marks"]
)
```

Only rows with missing `Age` or `Marks` are removed.

### 3. Remove columns with excessive missingness

```python
threshold = 0.60

columns_to_drop = df.columns[
    df.isna().mean() > threshold
]

df_reduced = df.drop(columns=columns_to_drop)
```

A 60% threshold is illustrative, not universal. A column with many missing values may still be important, while one with few missing values may be unsuitable if the missingness makes its meaning unreliable.

### Advantages of deletion

- Simple to implement.
- Does not invent replacement values.
- Can be appropriate for clearly unusable records.

### Disadvantages of deletion

- Reduces the amount of available data.
- May introduce bias if missingness is systematic.
- Can remove rare but important observations.
- May produce an unrepresentative training dataset.

Deletion and imputation are alternative strategies. Select the method based on the dataset, missingness mechanism, and intended application.

## 📊 Mean Imputation

Mean imputation replaces missing numerical values with the average of the observed values.

### Formula

For observed values \(x_1, x_2, \ldots, x_n\):

\[
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i
\]

Every missing value is replaced with \(\bar{x}\).

### Example

```python
import pandas as pd

df = pd.DataFrame({
    "Marks": [80, 90, None, 70, 100]
})

mean_marks = df["Marks"].mean()

df["Marks_Imputed"] = df["Marks"].fillna(mean_marks)

print("Mean:", mean_marks)
print(df)
```

The mean is `85`, so the missing value is replaced by `85`.

### Using Scikit-learn

```python
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "Marks": [80, 90, None, 70, 100]
})

imputer = SimpleImputer(strategy="mean")

df["Marks_Imputed"] = imputer.fit_transform(
    df[["Marks"]]
).ravel()

print(df)
```

This direct example fits the imputer to the provided dataset. In an evaluated machine learning workflow, fit the imputer only on the training partition.

### Advantages

- Easy to understand.
- Fast to compute.
- Useful for numerical features when the mean is a reasonable substitute.

### Limitations

- Sensitive to extreme values.
- Can reduce the observed variance.
- Does not preserve all relationships between variables.
- May be biased under certain missingness mechanisms.

Mean imputation is most defensible when the feature's distribution and missingness mechanism make a single mean value a reasonable approximation.

## 📐 Median Imputation

Median imputation replaces missing numerical values with the median of observed values.

The median is the central value after sorting the observations.

### Example

```python
import pandas as pd

df = pd.DataFrame({
    "Salary": [30000, 35000, None, 40000, 500000]
})

median_salary = df["Salary"].median()

df["Salary_Imputed"] = df["Salary"].fillna(median_salary)

print("Median salary:", median_salary)
print(df)
```

The extreme salary influences the mean much more than it influences the median.

### Using Scikit-learn

```python
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "Age": [20, 22, None, 25, 30]
})

imputer = SimpleImputer(strategy="median")

df["Age_Imputed"] = imputer.fit_transform(
    df[["Age"]]
).ravel()

print(df)
```

### Advantages

- Less sensitive to extreme observations than the mean.
- Useful for skewed numerical distributions.
- Simple and computationally efficient.

### Limitations

- Can distort feature relationships.
- Reduces variation in the imputed values.
- Does not account for other features that may explain the missing value.

**General guidance:** Median imputation is often a practical starting point for numerical features, particularly when the data is skewed or contains outliers. It should still be evaluated against alternatives.

## 🏷️ Mode Imputation

Mode imputation replaces missing values with the most frequently occurring value.

It is commonly used for categorical features.

### Example

```python
import pandas as pd

df = pd.DataFrame({
    "Department": [
        "IT", "HR", "IT", None, "Finance", "IT"
    ]
})

mode_department = df["Department"].mode().iloc[0]

df["Department_Imputed"] = df["Department"].fillna(
    mode_department
)

print(df)
```

The most frequent department is `IT`.

### Using Scikit-learn

```python
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "Department": [
        "IT", "HR", None, "Finance", "IT"
    ]
})

imputer = SimpleImputer(strategy="most_frequent")

df["Department_Imputed"] = imputer.fit_transform(
    df[["Department"]]
).ravel()

print(df)
```

### Advantages

- Supports categorical features.
- Simple and computationally efficient.
- Uses an existing category rather than inventing a new one.

### Limitations

- Can overrepresent the most frequent category.
- May weaken relationships between features.
- May be misleading when many categories are equally common.
- Can hide a meaningful missingness pattern.

When missingness itself has meaning, consider representing it explicitly rather than replacing every missing value with the mode.

## 🧱 Constant Value Imputation

Constant imputation fills missing values with a predefined value.

For numerical data, the constant may be zero or another domain-specific value. For categorical features, it may be an explicitly defined label such as `"Missing"`.

### Numerical example

```python
import pandas as pd

df = pd.DataFrame({
    "Discount": [10, None, 5, 0, None]
})

df["Discount_Imputed"] = df["Discount"].fillna(0)

print(df)
```

This is suitable only if zero is a valid and meaningful replacement for the missing discount in the application.

### Categorical example

```python
import pandas as pd

df = pd.DataFrame({
    "City": ["Mumbai", None, "Pune", "Delhi", None]
})

df["City_Imputed"] = df["City"].fillna("Missing")

print(df)
```

The new label distinguishes an absent city from an observed city.

However, a missing label may need to be included in the encoding strategy, and it should not be mistaken for a real location.

### Using Scikit-learn

```python
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "City": ["Mumbai", None, "Pune", "Delhi"]
})

imputer = SimpleImputer(
    strategy="constant",
    fill_value="Missing"
)

df["City_Imputed"] = imputer.fit_transform(
    df[["City"]]
).ravel()

print(df)
```

### When to consider constant imputation

- Zero has a defined interpretation in the application.
- A missing category must remain identifiable.
- A system requires an explicit default value.
- A sentinel is defined by the data contract.

Never use zero automatically just because it is convenient. For many numerical variables, zero is a real measurement and is not equivalent to missing.

## 👥 Group-Based Imputation

Sometimes a single overall statistic is inappropriate because different groups have different distributions.

For example, students from different departments may have different typical study hours.

### Example: Median study hours by department

```python
import pandas as pd

df = pd.DataFrame({
    "Department": [
        "IT", "IT", "IT",
        "HR", "HR", "HR"
    ],
    "Study_Hours": [
        4, 6, None,
        2, 3, None
    ]
})

df["Study_Hours_Imputed"] = df["Study_Hours"].fillna(
    df.groupby("Department")["Study_Hours"].transform("median")
)

print(df)
```

Each missing value is replaced by the median for its department.

### What if a group has no observed values?

Group-based imputation can leave missing values when an entire group lacks observations for a column.

A fallback strategy may be necessary:

```python
group_medians = df.groupby("Department")["Study_Hours"].transform("median")
overall_median = df["Study_Hours"].median()

df["Study_Hours_Imputed"] = (
    df["Study_Hours"]
    .fillna(group_medians)
    .fillna(overall_median)
)
```

This example uses the overall observed median when a group median is unavailable.

### Important limitation: group-based leakage

This Pandas example computes group statistics from the DataFrame being processed. That is useful for demonstrating the technique, but it can introduce leakage if the DataFrame includes validation or test observations.

For machine learning, use training-only statistics or implement the group logic in a transformer that is fitted on training data and reused consistently. Also consider whether the group information will exist at prediction time.

Group-based imputation is not automatically better than global imputation. Evaluate whether it improves the suitability of the data for the target task.

## 🚩 Missing Value Indicators

Sometimes the fact that a value is missing contains useful information.

For example, a customer who chooses not to report income may have a different pattern of behavior from a customer whose income is recorded.

A **missingness indicator** adds a separate feature that records whether the original value was missing.

### Manual indicator

```python
import pandas as pd

df = pd.DataFrame({
    "Age": [20, None, 25, 30, None]
})

df["Age_Was_Missing"] = df["Age"].isna().astype("int8")

median_age = df["Age"].median()

df["Age_Imputed"] = df["Age"].fillna(median_age)

print(df)
```

The indicator uses:

- `1` when the original age was missing.
- `0` when the original age was observed.

### Using Scikit-learn

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="median",
    add_indicator=True
)
```

When this imputer is fitted, it records the features that contained missing values in the training data and adds corresponding indicator features.

A missingness indicator may improve a model when missingness carries predictive information, but it may also encode operational or demographic patterns that require careful interpretation.

Evaluate it using appropriate validation data.

## 🧮 KNN Imputation

**K-Nearest Neighbors (KNN) imputation** estimates a missing value using observations that are similar to the incomplete observation.

Instead of using one overall mean or median, KNN looks at nearby observations according to a distance measure.

### Basic idea

Suppose an observation is missing its income value. Other observations with similar known feature values may provide information that can help estimate the missing income.

The algorithm considers available features and uses the nearest eligible observations to estimate the missing value.

### Example

```python
import pandas as pd
from sklearn.impute import KNNImputer

df = pd.DataFrame({
    "Age": [20, 21, None, 24, 25],
    "Marks": [80, None, 90, 85, 95],
    "Study_Hours": [3, 4, 5, None, 6]
})

imputer = KNNImputer(n_neighbors=2)

imputed_array = imputer.fit_transform(df)

imputed_df = pd.DataFrame(
    imputed_array,
    columns=df.columns,
    index=df.index
)

print(imputed_df)
```

### Important considerations

- KNN imputation can use relationships between numerical features.
- It can be computationally expensive on large datasets.
- Results depend on the choice of `n_neighbors`.
- It relies on a suitable distance measure.
- Variables with large numerical scales can dominate distances.
- It is not a universal replacement for categorical or time-series imputation.

### Scaling and KNN imputation

Distance-based methods are sensitive to feature scale. A feature such as annual income may dominate a feature such as age if their numerical ranges differ considerably.

However, a standard scaler cannot directly process raw missing values in a typical pipeline.

Possible solutions include:

- Applying domain-appropriate preprocessing before calculating distances.
- Using a carefully designed imputation workflow.
- Comparing KNN with simpler methods on representative validation data.
- Using algorithms that explicitly support missing values when appropriate.

Do not assume KNN is more accurate merely because it is more complex.

## 🔁 Iterative Imputation

**Iterative imputation** estimates missing values by repeatedly modeling each incomplete feature using other features.

For example, age, education, and experience may contain information related to salary. An iterative procedure can use relationships among the features to estimate missing values.

Scikit-learn provides `IterativeImputer` for this approach.

### Example

```python
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

df = pd.DataFrame({
    "Age": [20, 21, None, 24, 25],
    "Marks": [80, None, 90, 85, 95],
    "Study_Hours": [3, 4, 5, None, 6]
})

imputer = IterativeImputer(
    max_iter=10,
    random_state=42
)

imputed_array = imputer.fit_transform(df)

imputed_df = pd.DataFrame(
    imputed_array,
    columns=df.columns,
    index=df.index
)

print(imputed_df)
```

The `enable_iterative_imputer` import is required because this estimator remains experimental in Scikit-learn.

### Advantages

- Uses relationships between features.
- Can be useful when variables are correlated.
- Offers a flexible alternative to single-feature imputation.

### Limitations

- More computationally expensive than simple imputation.
- Depends on the models and assumptions used during estimation.
- Can produce implausible values if the underlying relationships are inappropriate.
- Does not automatically account for uncertainty in a way that is equivalent to every multiple-imputation methodology.

Always validate the plausibility of imputed values and compare performance with simpler approaches.

## 📈 Time-Series Imputation

Time-series datasets often contain missing values in chronological measurements.

For example:

```python
import pandas as pd

df = pd.DataFrame({
    "Date": pd.date_range("2026-01-01", periods=6, freq="D"),
    "Temperature": [25, None, 27, None, 29, 30]
})

print(df)
```

Different methods may be appropriate depending on whether the task is retrospective analysis or forecasting.

### 1. Forward fill

Forward fill uses the most recent preceding observed value.

```python
df["Temperature_FFILL"] = df["Temperature"].ffill()

print(df)
```

This is often useful for state-like measurements that remain valid until the next update, such as a recorded status or account balance.

It is not necessarily appropriate for rapidly changing measurements.

### 2. Backward fill

Backward fill uses the next observed value.

```python
df["Temperature_BFILL"] = df["Temperature"].bfill()
```

Backward fill may be useful for certain retrospective data-cleaning tasks, but it uses information from a later observation.

**Warning:** In forecasting or online prediction, backward fill can introduce future-information leakage when the later observation would not be available at prediction time.

### 3. Linear interpolation

```python
df["Temperature_Interpolated"] = df["Temperature"].interpolate(
    method="linear"
)

print(df)
```

Linear interpolation estimates missing values between known observations.

It can be useful for retrospective analysis when the values are reasonably modeled by linear change between adjacent observations.

However, standard linear interpolation can use future observations to estimate gaps. It is not automatically suitable for real-time forecasting.

### 4. Group-based forward fill

For data containing multiple entities, sort by entity and timestamp before filling values.

```python
import pandas as pd

df = pd.DataFrame({
    "Sensor": ["A", "A", "A", "B", "B", "B"],
    "Time": pd.to_datetime([
        "2026-01-01", "2026-01-02", "2026-01-03",
        "2026-01-01", "2026-01-02", "2026-01-03"
    ]),
    "Reading": [10, None, 12, 20, None, 22]
})

df = df.sort_values(["Sensor", "Time"]).copy()

df["Reading_Filled"] = (
    df.groupby("Sensor")["Reading"]
    .ffill()
)

print(df)
```

This avoids carrying a value from one sensor into another sensor's records.

### Time-series best practices

- Sort by timestamp before applying temporal methods.
- Keep independent entities separate.
- Understand the time interval and meaning of each measurement.
- Avoid future-information leakage in forecasting tasks.
- Investigate long gaps rather than filling them automatically.
- Compare imputation methods using a realistic chronological validation strategy.

## 🐼 Imputation with Pandas

Pandas provides convenient methods for filling missing values.

### 1. Use a fixed value

```python
df["Age"] = df["Age"].fillna(0)
```

Only use zero when it has a valid interpretation for the feature.

### 2. Fill with the mean

```python
df["Age"] = df["Age"].fillna(
    df["Age"].mean()
)
```

### 3. Fill with the median

```python
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)
```

### 4. Fill categorical data using the mode

```python
mode_city = df["City"].mode()

if not mode_city.empty:
    df["City"] = df["City"].fillna(mode_city.iloc[0])
```

### 5. Forward fill

```python
df["Reading"] = df["Reading"].ffill()
```

### 6. Interpolate numerical values

```python
df["Measurement"] = df["Measurement"].interpolate()
```

### 7. Inspect the result

```python
print(df.isna().sum())
```

These operations are convenient for exploration and deterministic data workflows. For machine learning evaluation, calculate learned statistics using training data only, or incorporate them into a pipeline.

## 🤖 Imputation with Scikit-learn

Scikit-learn's imputers provide a consistent interface for preprocessing machine learning features.

### `SimpleImputer`

Common strategies include:

- `"mean"` for numerical features.
- `"median"` for numerical features.
- `"most_frequent"` for numerical or categorical features.
- `"constant"` for predefined replacement values.

Example:

```python
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "Age": [20, None, 25, 30],
    "Salary": [30000, 40000, None, 50000]
})

imputer = SimpleImputer(strategy="median")

imputed = imputer.fit_transform(df)

imputed_df = pd.DataFrame(
    imputed,
    columns=df.columns,
    index=df.index
)

print(imputed_df)
```

This demonstrates the imputer on a standalone dataset. For a model evaluation workflow, fit it on training data only.

### `MissingIndicator`

Use a dedicated missingness indicator when you need binary features that identify original missing values.

```python
from sklearn.impute import MissingIndicator

indicator = MissingIndicator(features="missing-only")
```

A missing indicator is often easier to manage as part of a complete pipeline.

### `KNNImputer`

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3)
```

Suitable for some numerical datasets where similarity among observations is meaningful.

### `IterativeImputer`

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
```

Useful for exploring model-based estimation of incomplete numerical features.

**Important:** Choose an imputer according to the data structure, feature meaning, and evaluation results. More sophisticated imputation is not automatically more accurate.

## 🧱 Handling Different Feature Types

A dataset may contain numerical, categorical, ordinal, date, identifier, and target columns. These fields should not necessarily be imputed in the same way.

| Feature type | Example | Potential strategy |
|---|---|---|
| Numerical | Age, salary | Median or mean imputation |
| Categorical nominal | City, department | Most frequent category or an explicit missing label |
| Ordinal | Satisfaction rating | Most frequent category or an appropriate ordinal strategy |
| Boolean | Passed/failed flag | Mode or a deliberate missing-state representation |
| Time-series | Daily temperature | Forward fill or interpolation, when justified |
| Identifier | Student ID | Usually investigate missing IDs instead of statistically imputing them |
| Target | Pass/fail label | Usually exclude rows without a reliable target in supervised training, or resolve through a separate process |

### Why identifier imputation is usually inappropriate

A missing student ID cannot be meaningfully repaired by filling it with the average or median of other IDs. Identifiers represent identity, not magnitude.

Recover identifiers from a trusted source or use the application's missing-record policy.

### Why target imputation is risky

In supervised learning, the target is the answer that the model is trained to predict.

Filling unknown target values with the most frequent class invents labels that may be wrong. For ordinary supervised training, it is generally safer to exclude rows without a reliable target while retaining their other information separately if useful for another purpose.

### Date features

When a date is missing, do not replace it automatically with the mean timestamp without considering its meaning. Depending on the application, it may be better to recover the date, add a missingness indicator, exclude the observation, or derive valid features from other recorded information.

## 🔐 Complete Machine Learning Example

The following example demonstrates a leakage-aware pipeline for a classification task with missing numeric and categorical features.

Save this code as `imputation.py`.

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_dataset() -> pd.DataFrame:
    """Create a small synthetic student classification dataset."""

    return pd.DataFrame({
        "Age": [
            18, 19, 20, 21, 22, 23, 24, 20, 21, 22,
            18, 19, 20, 21, 22, 23, 24, 20, 21, 22
        ],
        "Attendance": [
            95, 85, 65, 90, 75, 55, 80, 70, 92, 60,
            50, 68, 88, 72, 96, 62, 78, 84, 58, 91
        ],
        "Study_Hours": [
            6, 5, 2, 5, 3, 1, 4, 3, 5, 2,
            1, 2, 5, 3, 6, 2, 4, 5, 1, 6
        ],
        "City": [
            "Mumbai", "Pune", "Delhi", "Mumbai", "Pune",
            "Delhi", "Mumbai", "Pune", "Delhi", "Mumbai",
            "Pune", "Delhi", "Mumbai", "Pune", "Delhi",
            "Mumbai", "Pune", "Delhi", "Mumbai", "Pune"
        ],
        "Passed": [
            1, 1, 0, 1, 0, 0, 1, 0, 1, 0,
            0, 0, 1, 0, 1, 0, 1, 1, 0, 1
        ]
    })


def build_model() -> Pipeline:
    """Create a pipeline containing imputation, encoding, scaling, and classification."""

    numeric_features = [
        "Age",
        "Attendance",
        "Study_Hours"
    ]

    categorical_features = ["City"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])


def main() -> None:
    df = create_dataset()

    # Introduce missing feature values for demonstration.
    df.loc[[2, 8, 14], "Age"] = float("nan")
    df.loc[[3, 11, 17], "Attendance"] = float("nan")
    df.loc[[4, 10], "City"] = None

    print("DATASET SHAPE")
    print(df.shape)

    print("\nMISSING VALUES BEFORE IMPUTATION")
    print(df.isna().sum())

    X = df.drop(columns=["Passed"])
    y = df["Passed"]

    # Split before fitting learned preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = build_model()

    # The imputer, scaler, encoder, and classifier are fitted on training data.
    model.fit(X_train, y_train)

    # The test data is processed with the fitted training pipeline.
    predictions = model.predict(X_test)

    print("\nTRAINING ACCURACY")
    print(accuracy_score(y_train, model.predict(X_train)))

    print("\nTEST ACCURACY")
    print(accuracy_score(y_test, predictions))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("\nMISSING VALUES IN ORIGINAL FEATURE DATA")
    print(X.isna().sum())


if __name__ == "__main__":
    main()
```

### What this project demonstrates

- Creating a reproducible student dataset.
- Introducing missing values in numeric and categorical features.
- Splitting data before fitting learned preprocessing.
- Using median imputation for numerical features.
- Using most-frequent imputation for categorical features.
- Scaling numerical features.
- Applying One-Hot Encoding to nominal features.
- Training and evaluating a classifier in a single pipeline.

The synthetic dataset is small and intended for learning. Its model scores do not represent expected performance on a real educational dataset.

**Why is this workflow safer?** The imputer, scaler, and encoder are fitted using the training partition. The test partition is processed with those fitted transformations without contributing to their learned statistics.

For a real project, select a split strategy suited to the data, especially when observations share users, groups, or time dependencies.

## ✅ Validate Imputed Data

Imputation is not complete until the resulting dataset has been checked.

### 1. Count remaining missing values

```python
print(df.isna().sum())
```

Some columns may legitimately remain missing if no suitable imputation strategy was selected.

### 2. Compare before and after

```python
before = df.isna().sum()

df_filled = df.copy()
df_filled["Age"] = df_filled["Age"].fillna(
    df_filled["Age"].median()
)

after = df_filled.isna().sum()

comparison = pd.DataFrame({
    "Missing_Before": before,
    "Missing_After": after
})

print(comparison)
```

### 3. Check numerical ranges

```python
invalid_age = ~df_filled["Age"].between(0, 120)

print(df_filled.loc[invalid_age])
```

This assumes the intended age range is 0–120.

### 4. Compare descriptive statistics

```python
print("Before imputation:")
print(df["Age"].describe())

print("\nAfter imputation:")
print(df_filled["Age"].describe())
```

Imputation can change the mean, median, variance, and distribution of a feature. These changes should be understood rather than treated as automatically beneficial.

### 5. Confirm that identifiers and targets remain unchanged

Imputation should be limited to the intended feature columns. Check that identifiers, target labels, and other protected fields have not been accidentally modified.

### 6. Validate outputs during inference

For production models, ensure that:

- Required columns are present.
- Numeric fields have acceptable types.
- Imputed values remain within plausible ranges.
- Previously unseen categories are handled intentionally.
- The fitted preprocessing pipeline is reused.
- Missingness rates are monitored over time.

## ❌ Common Mistakes

### 1. Applying mean imputation to every numerical column

The mean can be unsuitable for highly skewed distributions or extreme values.

**Solution:** Examine the distribution and compare reasonable alternatives.

### 2. Using zero for every missing value

Zero may be a genuine measurement, not an indication of missingness.

**Solution:** Use a constant only when it has a justified meaning.

### 3. Using mode imputation without checking the category distribution

Repeatedly filling a category with the mode can make it artificially dominant.

**Solution:** Consider whether missingness is informative and evaluate alternative approaches.

### 4. Imputing before splitting training and test data

Global statistics may incorporate information from held-out observations.

**Solution:** Split first and fit learned imputers on training data only.

### 5. Imputing target values without justification

Unknown labels may become incorrect training targets.

**Solution:** Obtain reliable labels or exclude unresolved targets from ordinary supervised training.

### 6. Imputing identifiers statistically

Taking the mean of student IDs, phone numbers, or customer IDs has no meaningful interpretation.

**Solution:** Recover the identifier from a trusted source or apply a defined missing-record policy.

### 7. Ignoring the reason values are missing

A technical failure and a deliberately withheld value may represent very different situations.

**Solution:** Investigate missingness mechanisms and consider missingness indicators or specialized methods when appropriate.

### 8. Using backward fill in forecasting without checking future-information leakage

A later observation may not be available when making an earlier prediction.

**Solution:** Choose time-series methods that reflect the information available at the prediction timestamp.

### 9. Assuming KNN or iterative imputation always performs better

Complex methods introduce assumptions and computational costs.

**Solution:** Compare methods using realistic validation and domain checks.

### 10. Overwriting the original values

Permanent replacement makes later auditing difficult.

**Solution:** Preserve raw data where possible and record the transformation rules.

## 🌟 Best Practices

- Profile missing values before choosing a strategy.
- Investigate why each important field is missing.
- Separate missing data from invalid data and documented missing-value sentinels.
- Preserve raw source data and maintain a cleaning report.
- Use median imputation as a baseline for numerical data when appropriate.
- Use mode or an explicit missing category for categorical features when justified.
- Consider missingness indicators when absence may be informative.
- Evaluate advanced methods such as KNN and iterative imputation rather than assuming they are superior.
- Fit learned imputation statistics on training data only.
- Use Scikit-learn pipelines for reproducible preprocessing.
- Keep target labels and identifiers out of inappropriate imputation steps.
- Account for groups and temporal dependencies.
- Validate ranges, categories, and remaining missingness after transformation.
- Monitor missingness patterns in deployed systems.
- Document assumptions, thresholds, and limitations.
- Test imputation functions with all-missing columns, unexpected categories, and incomplete inputs.

## 💡 Mini Projects

### Project 1: Student Dataset Imputation

Create a student dataset containing Age, Marks, Attendance, and Department.

**Tasks:**

1. Introduce missing values into numeric and categorical columns.
2. Report missing counts and percentages.
3. Apply mean and median imputation to numerical features.
4. Apply mode and constant imputation to categorical features.
5. Compare descriptive statistics before and after imputation.
6. Validate the final dataset.

### Project 2: Employee Salary Imputation

Create an employee dataset containing salary, experience, department, and education level.

**Tasks:**

1. Examine salary distribution and missingness.
2. Compare mean and median imputation.
3. Evaluate group-based imputation by department.
4. Track which salary values were originally missing.
5. Explain the limitations of each strategy.

### Project 3: Time-Series Missing Value Handling

Create a daily temperature or sales dataset.

**Tasks:**

1. Introduce missing periods.
2. Visualize the gaps.
3. Compare forward fill and interpolation.
4. Avoid using future values for a simulated real-time forecasting task.
5. Evaluate the impact of each strategy.

### Project 4: Compare Imputation Algorithms

Create a numerical dataset with correlated features and missing values.

**Tasks:**

1. Establish a reproducible training and validation split.
2. Compare median, KNN, and iterative imputation.
3. Evaluate data quality and model performance.
4. Inspect implausible imputed values.
5. Explain why the best method depends on the dataset.

### Project 5: Build a Leakage-Safe Preprocessing Pipeline

Create a dataset with numeric and categorical features.

**Tasks:**

1. Split the dataset before fitting transformations.
2. Impute numeric values using the median.
3. Impute categorical values using the most frequent category.
4. Scale numerical features.
5. Encode nominal features.
6. Train a model through a Scikit-learn pipeline.
7. Evaluate it on held-out data.

## 📝 Practice Exercises

1. Create a DataFrame containing missing numerical values.
2. Count missing observations in every column.
3. Calculate missing percentages.
4. Impute an age column using the mean.
5. Impute salary using the median.
6. Impute department using the mode.
7. Replace missing city values with a documented constant.
8. Create a missingness indicator.
9. Apply `KNNImputer` to a numerical dataset.
10. Apply `IterativeImputer` to correlated numerical features.
11. Demonstrate forward fill and linear interpolation on time-series data.
12. Explain the difference between MCAR, MAR, and MNAR.
13. Show how fitting an imputer before splitting can contaminate model evaluation.
14. Build a pipeline containing an imputer, scaler, encoder, and classifier.
15. Explain why target labels should not usually be imputed using the most frequent class.
16. Compare model metrics after testing two different imputation strategies.

## ⚡ Quick Revision

| Task | Method |
|---|---|
| Detect missing values | `df.isna()` |
| Count missing values | `df.isna().sum()` |
| Calculate missing percentages | `df.isna().mean() * 100` |
| Drop incomplete rows | `df.dropna()` |
| Fill with the mean | `series.fillna(series.mean())` |
| Fill with the median | `series.fillna(series.median())` |
| Fill with the mode | `series.fillna(series.mode().iloc[0])` |
| Fill with a constant | `series.fillna(value)` |
| Group-based imputation | `groupby().transform()` |
| Create missing indicators | `series.isna().astype("int8")` |
| Simple imputation | `SimpleImputer()` |
| KNN imputation | `KNNImputer()` |
| Iterative imputation | `IterativeImputer()` |
| Forward fill | `series.ffill()` |
| Backward fill | `series.bfill()` |
| Interpolation | `series.interpolate()` |
| Prevent preprocessing leakage | `Pipeline` |
| Handle mixed feature types | `ColumnTransformer` |

## 📂 Project Structure

Recommended repository structure:

```text
07-Data-Preprocessing/
│
├── README.md
│
├── 01-Train-Test-Split/
│   └── README.md
│
├── 02-Feature-Scaling/
│   └── README.md
│
├── 03-Standardization/
│   └── README.md
│
├── 04-Normalization/
│   └── README.md
│
├── 05-Label-Encoding/
│   └── README.md
│
├── 06-One-Hot-Encoding/
│   └── README.md
│
├── 07-Ordinal-Encoding/
│   └── README.md
│
└── 08-Imputation/
    ├── README.md
    ├── imputation.py
    ├── mean_imputation.py
    ├── median_imputation.py
    ├── mode_imputation.py
    ├── constant_imputation.py
    ├── knn_imputation.py
    ├── iterative_imputation.py
    ├── time_series_imputation.py
    ├── missing_indicators.py
    └── imputation_pipeline.py
```

| File | Purpose |
|---|---|
| `README.md` | Concepts, techniques, and complete learning guide |
| `imputation.py` | End-to-end ML example |
| `mean_imputation.py` | Mean-based imputation |
| `median_imputation.py` | Median-based imputation |
| `mode_imputation.py` | Categorical mode imputation |
| `constant_imputation.py` | Constant-value replacement |
| `knn_imputation.py` | Nearest-neighbor imputation |
| `iterative_imputation.py` | Iterative model-based imputation |
| `time_series_imputation.py` | Temporal missing-value handling |
| `missing_indicators.py` | Missingness indicator features |
| `imputation_pipeline.py` | Leakage-safe preprocessing workflow |

Create individual scripts as you progress through the topic. The examples in this README can serve as a starting point for each file.

## 🗺️ Learning Roadmap

Follow this progression to build practical imputation skills:

1. Understand missing values and missing-data mechanisms.
2. Inspect missingness counts, percentages, and patterns.
3. Learn when deletion is appropriate.
4. Practice mean, median, mode, and constant imputation.
5. Add missingness indicators when justified.
6. Explore KNN and iterative imputation.
7. Learn time-series imputation and future-information leakage.
8. Handle numerical and categorical columns separately.
9. Build Scikit-learn preprocessing pipelines.
10. Compare imputation strategies with appropriate validation.
11. Test the resulting data and monitor missingness over time.

## 🎯 Key Takeaways

- Imputation fills missing values using a defined rule or estimation method.
- Mean, median, mode, and constant imputation are common starting points.
- KNN and iterative imputation use relationships among observations or features.
- Missing-data mechanisms and domain context matter when selecting a strategy.
- Imputation can change distributions and relationships among variables.
- Missingness indicators can preserve information about the absence of a value.
- Identifiers and target labels require special care.
- Learned imputers should be fitted on training data only.
- Scikit-learn pipelines support consistent, leakage-aware preprocessing.
- Every imputation strategy should be validated against the intended application.

## 👨‍💻 Author

**Kishor Patil**

Python | Data Science | Machine Learning | Artificial Intelligence

GitHub: [Kishor055](https://github.com/Kishor055)

Repository: [Machine-Learning](https://github.com/Kishor055/Machine-Learning)

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Add examples, improvements, or corrections.
4. Test your code.
5. Submit a pull request with a clear description.

## ⭐ Support

If this guide helps you learn data preprocessing and machine learning, consider starring the repository on GitHub.

**Keep learning, keep building, and keep growing!** 🌱

## 📖 References

- [Pandas: Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Scikit-learn: Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)
- [Scikit-learn: SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [Scikit-learn: KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)
- [Scikit-learn: IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html)
- [Scikit-learn: Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

---

*Part of the Machine Learning learning path: Data Collection → Data Cleaning → Exploratory Data Analysis → Data Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment.*

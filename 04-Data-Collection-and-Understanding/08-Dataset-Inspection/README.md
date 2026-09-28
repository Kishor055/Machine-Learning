
# 🔍 Dataset Inspection for Machine Learning

Dataset inspection is the process of **systematically examining a dataset before cleaning, visualization, feature engineering, or Machine Learning**.

A Machine Learning model is only as reliable as the data pipeline behind it.

Before training a model, you should understand:

* 📦 How large the dataset is
* 🧱 What columns exist
* 🔢 Which columns are numeric
* 🏷️ Which columns are categorical
* 📅 Which columns contain dates
* 🎯 What the target variable is
* ❓ How much data is missing
* 🔁 Whether duplicate records exist
* 📊 How values are distributed
* 🚨 Whether suspicious values or outliers exist
* 🔗 How features relate to the target
* ⚠️ Whether data leakage may exist
* 🧪 Whether the dataset is suitable for ML

The general workflow is:

```text
Raw Dataset
     ↓
Load Dataset
     ↓
Basic Inspection
     ↓
Schema Inspection
     ↓
Data Type Inspection
     ↓
Missing Value Inspection
     ↓
Duplicate Inspection
     ↓
Statistical Inspection
     ↓
Target Inspection
     ↓
Relationship Inspection
     ↓
Data Quality Checks
     ↓
Leakage Checks
     ↓
Data Understanding
     ↓
Cleaning / EDA
     ↓
Machine Learning
```

---

# 📚 Table of Contents

1. [What Is Dataset Inspection?](#-what-is-dataset-inspection)
2. [Why Dataset Inspection Matters](#-why-dataset-inspection-matters)
3. [Inspection Workflow](#-inspection-workflow)
4. [Loading a Dataset](#-loading-a-dataset)
5. [First Look at the Data](#-first-look-at-the-data)
6. [Dataset Shape](#-dataset-shape)
7. [Rows and Columns](#-rows-and-columns)
8. [Column Names](#-column-names)
9. [Data Types](#-data-types)
10. [Dataset Information](#-dataset-information)
11. [Sample Records](#-sample-records)
12. [Unique Values](#-unique-values)
13. [Value Counts](#-value-counts)
14. [Missing Values](#-missing-values)
15. [Missing Value Percentage](#-missing-value-percentage)
16. [Duplicate Records](#-duplicate-records)
17. [Constant Columns](#-constant-columns)
18. [Near-Constant Columns](#-near-constant-columns)
19. [Numeric Columns](#-numeric-columns)
20. [Categorical Columns](#-categorical-columns)
21. [Date and Time Columns](#-date-and-time-columns)
22. [Text Columns](#-text-columns)
23. [Statistical Summary](#-statistical-summary)
24. [Mean](#-mean)
25. [Median](#-median)
26. [Minimum and Maximum](#-minimum-and-maximum)
27. [Standard Deviation](#-standard-deviation)
28. [Quantiles](#-quantiles)
29. [Distribution Inspection](#-distribution-inspection)
30. [Outlier Inspection](#-outlier-inspection)
31. [Target Variable Inspection](#-target-variable-inspection)
32. [Class Distribution](#-class-distribution)
33. [Feature Cardinality](#-feature-cardinality)
34. [Correlation Inspection](#-correlation-inspection)
35. [Data Range Validation](#-data-range-validation)
36. [Schema Validation](#-schema-validation)
37. [Identifier Columns](#-identifier-columns)
38. [Potential Leakage](#-potential-leakage)
39. [Train/Test Inspection](#-traintest-inspection)
40. [Data Quality Report](#-data-quality-report)
41. [Automated Inspection](#-automated-inspection)
42. [Pandas Inspection Toolkit](#-pandas-inspection-toolkit)
43. [Reusable Inspection Functions](#-reusable-inspection-functions)
44. [Production Dataset Inspection](#-production-dataset-inspection)
45. [Common Mistakes](#-common-mistakes)
46. [Mini Projects](#-mini-projects)
47. [Exercises](#-exercises)
48. [Project Structure](#-project-structure)
49. [End-to-End Inspection Example](#-end-to-end-inspection-example)
50. [Dataset Inspection Checklist](#-dataset-inspection-checklist)
51. [ML Readiness Checklist](#-ml-readiness-checklist)
52. [Learning Roadmap](#-learning-roadmap)
53. [Key Takeaways](#-key-takeaways)
54. [Next Step](#-next-step)

---

# 🔹 What Is Dataset Inspection?

Dataset inspection means examining a dataset **before making assumptions about it**.

Suppose we have:

```text
customer_id
age
city
income
purchase_count
membership
churn
```

Inspection should answer:

```text
How many rows?
How many columns?
Which columns are numeric?
Which are categorical?
Are values missing?
Are IDs unique?
What is the target?
Is the target balanced?
Are there impossible values?
Are any features suspicious?
```

Dataset inspection is not the same as data cleaning.

### Inspection

Answers:

> "What is in the dataset?"

### Cleaning

Answers:

> "How should invalid or problematic data be handled?"

### EDA

Answers:

> "What patterns and relationships exist in the data?"

---

# 🔹 Why Dataset Inspection Matters

A dataset may appear correct but contain serious problems.

For example:

```text
age
---
21
24
25
27
-400
```

Or:

```text
gender
------
Male
male
M
MALE
Female
female
```

Or:

```text
income
------
50000
60000
NaN
70000
```

Or even:

```text
target
------
0
0
0
0
0
1
```

These issues can significantly affect ML performance.

Inspection helps identify them before modeling.

---

# 🔹 Inspection Workflow

A professional inspection process:

```text
1. Load
   ↓
2. Shape
   ↓
3. Schema
   ↓
4. Data Types
   ↓
5. Samples
   ↓
6. Missing Values
   ↓
7. Duplicates
   ↓
8. Unique Values
   ↓
9. Statistics
   ↓
10. Distribution
   ↓
11. Target
   ↓
12. Relationships
   ↓
13. Quality Rules
   ↓
14. Leakage
   ↓
15. ML Readiness
```

---

# 🔹 Loading a Dataset

Using Pandas:

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

For Excel:

```python
df = pd.read_excel("data.xlsx")
```

For JSON:

```python
df = pd.read_json("data.json")
```

For Parquet:

```python
df = pd.read_parquet("data.parquet")
```

Always inspect the result immediately.

---

# 🔹 First Look at the Data

The first step:

```python
print(df.head())
```

Example:

```text
   age      city  income  churn
0   21     Pune   35000      0
1   28    Mumbai   52000      0
2   35     Delhi   78000      1
3   42     Pune   91000      0
```

Also inspect the last records:

```python
print(df.tail())
```

---

# 🔹 Dataset Shape

Use:

```python
rows, columns = df.shape

print("Rows:", rows)
print("Columns:", columns)
```

Example:

```text
Rows: 10000
Columns: 15
```

This immediately tells you the dataset's dimensionality.

---

# 🔹 Rows and Columns

Rows represent observations.

Columns represent variables.

Example:

```text
Customer Dataset

Rows:
1 customer
2 customer
3 customer
...

Columns:
age
income
city
membership
churn
```

Machine Learning generally represents data as:

```text
Rows → samples
Columns → features / target / metadata
```

---

# 🔹 Column Names

Inspect:

```python
print(df.columns)
```

Convert to a list:

```python
columns = df.columns.tolist()

print(columns)
```

Check for duplicate column names:

```python
print(df.columns.duplicated().any())
```

Clean column names carefully:

```python
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
```

Be careful not to rename columns in a way that loses important semantic meaning.

---

# 🔹 Data Types

Use:

```python
print(df.dtypes)
```

Example:

```text
age          int64
income     float64
city        object
churn        int64
```

Common Pandas types include:

```text
int64
float64
bool
object
string
category
datetime64
```

Data types matter because ML algorithms expect appropriate numerical representations.

---

# 🔹 Dataset Information

Use:

```python
df.info()
```

Example:

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries
Data columns:
age       1000 non-null int64
income     990 non-null float64
city       998 non-null object
churn     1000 non-null int64
```

`info()` provides:

* Number of rows
* Number of columns
* Non-null counts
* Data types
* Memory usage

It is one of the most useful first inspection commands.

---

# 🔹 Sample Records

Use:

```python
df.head(10)
```

Random sample:

```python
df.sample(
    n=10,
    random_state=42,
)
```

Why random samples?

Because the first rows may not represent the entire dataset.

---

# 🔹 Unique Values

For a column:

```python
print(df["city"].unique())
```

Count unique values:

```python
print(df["city"].nunique())
```

This helps identify:

```text
Low-cardinality categorical variables
High-cardinality variables
Unexpected categories
Potential IDs
```

---

# 🔹 Value Counts

For categorical columns:

```python
print(df["city"].value_counts())
```

Include missing values:

```python
print(
    df["city"]
    .value_counts(dropna=False)
)
```

This is useful for identifying:

* Dominant categories
* Rare categories
* Unexpected values
* Missing categories

---

# 🔹 Missing Values

Check total missing values:

```python
print(df.isna().sum())
```

Example:

```text
age          0
income      10
city         2
churn        0
```

Check whether any missing values exist:

```python
print(df.isna().any().any())
```

---

# 🔹 Missing Value Percentage

Counts alone are not enough.

Calculate percentages:

```python
missing_percentage = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(
        ascending=False
    )
)

print(missing_percentage)
```

Example:

```text
income    12.4%
city       2.1%
age        0.0%
```

This helps prioritize data-quality problems.

---

# 🔹 Duplicate Records

Count duplicate rows:

```python
duplicate_count = df.duplicated().sum()

print(
    "Duplicates:",
    duplicate_count,
)
```

Inspect them:

```python
duplicates = df[
    df.duplicated(
        keep=False
    )
]

print(duplicates)
```

Do not automatically delete duplicates.

First determine whether repeated rows are:

* Genuine repeated observations
* Data-entry errors
* Multiple events
* Legitimate transactions

---

# 🔹 Constant Columns

A constant column has only one unique value.

```python
constant_columns = [
    column
    for column in df.columns
    if df[column].nunique(
        dropna=False
    ) <= 1
]

print(constant_columns)
```

Example:

```text
source_system = "CRM"
```

Such columns often provide little predictive information.

However, metadata columns may still be useful for auditing, so inspect before removing them.

---

# 🔹 Near-Constant Columns

A column may technically contain multiple values but still be dominated by one value.

Example:

```text
active
------
1
1
1
1
1
0
```

Inspect the frequency:

```python
print(
    df["active"]
    .value_counts(
        normalize=True
    )
)
```

A highly dominant value may indicate:

* Near-constant feature
* Class imbalance
* Data collection issue
* Rare-event behavior

The interpretation depends on the domain.

---

# 🔹 Numeric Columns

Find numeric columns:

```python
numeric_columns = (
    df.select_dtypes(
        include="number"
    )
    .columns
    .tolist()
)

print(numeric_columns)
```

Example:

```text
age
income
purchase_count
credit_score
```

Inspect:

```python
print(
    df[numeric_columns]
    .head()
)
```

---

# 🔹 Categorical Columns

Find categorical-like columns:

```python
categorical_columns = (
    df.select_dtypes(
        include=[
            "object",
            "string",
            "category",
        ]
    )
    .columns
    .tolist()
)

print(categorical_columns)
```

Typical examples:

```text
city
gender
membership
department
product_category
```

Remember that some categorical variables may be stored as numbers.

For example:

```text
1 = Bronze
2 = Silver
3 = Gold
```

Numeric storage does not automatically make a feature numerically continuous.

---

# 🔹 Date and Time Columns

Inspect possible date columns:

```python
print(df.dtypes)
```

Convert explicitly when appropriate:

```python
df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce",
)
```

Then inspect:

```python
print(df["created_at"].min())
print(df["created_at"].max())
```

Check missing parsed dates:

```python
print(
    df["created_at"].isna().sum()
)
```

Date ranges are especially important for:

* Time series
* Forecasting
* Fraud detection
* Event prediction
* Temporal ML datasets

---

# 🔹 Text Columns

Text columns may include:

```text
reviews
descriptions
comments
titles
articles
messages
```

Inspect:

```python
text_lengths = (
    df["review"]
    .fillna("")
    .str.len()
)

print(text_lengths.describe())
```

Check empty values:

```python
empty_reviews = (
    df["review"]
    .fillna("")
    .str.strip()
    .eq("")
)

print(empty_reviews.sum())
```

---

# 🔹 Statistical Summary

For numerical columns:

```python
print(
    df.describe()
)
```

This typically includes:

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

For all columns:

```python
print(
    df.describe(
        include="all"
    )
)
```

For categorical columns:

```python
print(
    df.describe(
        include=["object", "category"]
    )
)
```

---

# 🔹 Mean

The mean is:

$$
\text{Mean} =
\frac{\sum x_i}{n}
$$

Python:

```python
mean_age = df["age"].mean()

print(mean_age)
```

The mean can be strongly affected by extreme values.

---

# 🔹 Median

The median is the middle value after sorting.

```python
median_income = (
    df["income"]
    .median()
)

print(median_income)
```

Median is often useful for skewed data.

Example:

```text
Income:
30k
35k
40k
45k
500k
```

The mean is pulled upward by the extreme value.

---

# 🔹 Minimum and Maximum

```python
print(
    df["age"].min()
)

print(
    df["age"].max()
)
```

This can reveal impossible values.

Example:

```text
Minimum age: -5
Maximum age: 900
```

These values require investigation.

---

# 🔹 Standard Deviation

Standard deviation measures the spread of numerical values around the mean.

$$
\sigma =
\sqrt{
\frac{
\sum (x_i-\mu)^2
}{
n
}
}
$$

In Pandas:

```python
std = df["income"].std()

print(std)
```

A high standard deviation indicates greater dispersion.

---

# 🔹 Quantiles

Quantiles divide values into portions.

```python
print(
    df["income"].quantile(
        [0.25, 0.50, 0.75]
    )
)
```

Important values:

```text
25% → Q1
50% → Median
75% → Q3
```

Quantiles are useful for understanding skewness and identifying potential outliers.

---

# 🔹 Distribution Inspection

A simple numerical inspection:

```python
print(
    df["income"]
    .describe()
)
```

A visual inspection can use histograms:

```python
import matplotlib.pyplot as plt

df["income"].plot(
    kind="hist",
    bins=30,
)

plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Income Distribution")
plt.show()
```

Look for:

```text
Symmetry
Skewness
Multiple peaks
Long tails
Extreme values
```

---

# 🔹 Outlier Inspection

One common method uses the Interquartile Range (IQR).

$$
IQR = Q_3 - Q_1
$$

Lower boundary:

$$
Q_1 - 1.5(IQR)
$$

Upper boundary:

$$
Q_3 + 1.5(IQR)
$$

Python:

```python
q1 = df["income"].quantile(0.25)
q3 = df["income"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[
    (df["income"] < lower)
    | (df["income"] > upper)
]

print(outliers)
```

Important:

> An outlier is not automatically an error.

It may represent a legitimate rare observation.

---

# 🔹 Target Variable Inspection

The target variable is what the ML model attempts to predict.

Examples:

```text
Classification:
churn
fraud
disease
spam

Regression:
price
salary
sales
temperature
```

Always inspect the target before modeling.

```python
target = "churn"

print(df[target].describe())
```

For classification:

```python
print(
    df[target]
    .value_counts(
        dropna=False
    )
)
```

---

# 🔹 Class Distribution

For classification:

```python
class_distribution = (
    df["churn"]
    .value_counts(
        normalize=True
    )
    .mul(100)
)

print(class_distribution)
```

Example:

```text
0 → 95%
1 → 5%
```

This indicates an imbalanced target.

Do not automatically "fix" imbalance during inspection.

First understand:

* Business meaning
* Sampling process
* Evaluation metric
* Cost of errors
* Dataset construction

---

# 🔹 Feature Cardinality

Cardinality means the number of unique values.

```python
cardinality = (
    df.nunique()
    .sort_values(
        ascending=False
    )
)

print(cardinality)
```

Example:

```text
customer_id       100000
city                  50
gender                 2
membership             3
```

A column with nearly one unique value per row may be:

* Identifier
* Transaction ID
* Timestamp
* High-cardinality feature

High cardinality is not automatically bad, but it requires investigation.

---

# 🔹 Correlation Inspection

Correlation measures relationships between numerical variables.

A common measure is Pearson correlation:

$$
r =
\frac{
\operatorname{cov}(X,Y)
}{
\sigma_X\sigma_Y
}
$$

Pandas:

```python
correlation = (
    df
    .select_dtypes(
        include="number"
    )
    .corr()
)

print(correlation)
```

Correlation values are approximately:

```text
+1 → strong positive linear relationship
 0 → little linear relationship
-1 → strong negative linear relationship
```

Important:

> Correlation does not prove causation.

Also, correlation only captures particular forms of relationship and can miss nonlinear patterns.

---

# 🔹 Data Range Validation

Domain knowledge is extremely important.

Example rules:

```python
assert (
    df["age"]
    .dropna()
    .between(0, 120)
    .all()
)
```

For percentages:

```python
assert (
    df["conversion_rate"]
    .dropna()
    .between(0, 100)
    .all()
)
```

For positive income:

```python
invalid_income = df[
    df["income"].notna()
    & (df["income"] < 0)
]

print(invalid_income)
```

Do not blindly use assertions in production pipelines when a recoverable validation error should instead be logged and handled explicitly.

---

# 🔹 Schema Validation

A dataset schema describes expected structure.

Example:

```python
expected_schema = {
    "age": "int64",
    "income": "float64",
    "city": "object",
    "churn": "int64",
}
```

A simple column check:

```python
expected_columns = set(
    expected_schema
)

actual_columns = set(
    df.columns
)

missing = (
    expected_columns
    - actual_columns
)

unexpected = (
    actual_columns
    - expected_columns
)

print("Missing:", missing)
print("Unexpected:", unexpected)
```

Schema validation becomes especially important in automated ML pipelines.

---

# 🔹 Identifier Columns

Common identifiers:

```text
customer_id
user_id
transaction_id
order_id
product_id
```

Identifiers may be useful for:

* Tracking
* Joining datasets
* Auditing
* Deduplication

But they often should **not** be treated as ordinary ML features.

Example:

```text
customer_id = 100001
customer_id = 100002
customer_id = 100003
```

The numerical values do not necessarily have meaningful mathematical relationships.

---

# 🔹 Potential Leakage

Dataset inspection should include an initial leakage check.

Potential leakage sources:

```text
Post-outcome variables
Future timestamps
Target-derived features
Duplicate records across splits
Aggregated information using future data
Human-entered fields created after the outcome
```

Example:

```text
Predict:
Loan Default

Feature:
"collection_status"
```

If `collection_status` is assigned after default occurs, using it for prediction may leak the outcome.

A useful question:

> Was this feature genuinely available at the moment the prediction would have been made?

---

# 🔹 Train/Test Inspection

After splitting a dataset, inspect both partitions.

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
)
```

Check:

```python
print(train_df.shape)
print(test_df.shape)
```

For classification:

```python
print(
    train_df["churn"]
    .value_counts(
        normalize=True
    )
)

print(
    test_df["churn"]
    .value_counts(
        normalize=True
    )
)
```

For time-dependent datasets, random splitting may be inappropriate. A chronological split may better represent real deployment conditions.

---

# 🔹 Data Quality Report

A useful inspection report can contain:

```text
Dataset name
Rows
Columns
Memory usage
Data types
Missing values
Duplicate rows
Unique values
Numeric summary
Categorical summary
Target distribution
Potential outliers
Potential leakage
Schema violations
```

Example:

```text
Dataset: customer_churn.csv

Rows: 10,000
Columns: 15

Missing Values:
income → 1.2%
city → 0.4%

Duplicates:
12

Target:
churn

Class Distribution:
0 → 84%
1 → 16%

Potential Issues:
- 1 constant column
- 2 suspicious ranges
- 1 high-cardinality ID
```

---

# 🔹 Automated Inspection

For repeated projects, automate common checks.

Example:

```python
def inspect_dataset(df):
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "numeric_columns": len(
            df.select_dtypes(
                include="number"
            ).columns
        ),
        "categorical_columns": len(
            df.select_dtypes(
                include=[
                    "object",
                    "string",
                    "category",
                ]
            ).columns
        ),
    }

    return report
```

Usage:

```python
report = inspect_dataset(df)

for key, value in report.items():
    print(f"{key}: {value}")
```

---

# 🔹 Pandas Inspection Toolkit

The following commands form a practical inspection toolkit:

| Task                | Pandas                               |
| ------------------- | ------------------------------------ |
| First rows          | `df.head()`                          |
| Last rows           | `df.tail()`                          |
| Random rows         | `df.sample()`                        |
| Shape               | `df.shape`                           |
| Columns             | `df.columns`                         |
| Types               | `df.dtypes`                          |
| Overview            | `df.info()`                          |
| Statistics          | `df.describe()`                      |
| Missing values      | `df.isna().sum()`                    |
| Duplicates          | `df.duplicated().sum()`              |
| Unique values       | `df.nunique()`                       |
| Categories          | `df["col"].value_counts()`           |
| Correlation         | `df.corr(numeric_only=True)`         |
| Numeric columns     | `df.select_dtypes(include="number")` |
| Categorical columns | `df.select_dtypes(include="object")` |

---

# 🔹 Reusable Inspection Functions

A professional inspection module can contain reusable functions.

```python
from __future__ import annotations

import pandas as pd


def basic_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset dimensions."""

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "memory_bytes": int(
            df.memory_usage(
                deep=True
            ).sum()
        ),
    }


def missing_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return missing-value counts and percentages."""

    result = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percentage": (
            df.isna()
            .mean()
            .mul(100)
        ),
    })

    return (
        result
        .sort_values(
            "missing_percentage",
            ascending=False,
        )
    )


def cardinality_summary(
    df: pd.DataFrame,
) -> pd.Series:
    """Return number of unique values per column."""

    return (
        df.nunique(
            dropna=False
        )
        .sort_values(
            ascending=False
        )
    )


def duplicate_summary(
    df: pd.DataFrame,
) -> dict:
    """Return duplicate-row information."""

    duplicate_mask = df.duplicated()

    return {
        "duplicate_rows": int(
            duplicate_mask.sum()
        ),
        "duplicate_percentage": (
            duplicate_mask.mean() * 100
        ),
    }
```

---

# 🔹 Production Dataset Inspection

For production ML systems, inspection should become part of the pipeline.

Example:

```text
Incoming Dataset
       ↓
Schema Check
       ↓
Row Count Check
       ↓
Missing Value Check
       ↓
Range Check
       ↓
Duplicate Check
       ↓
Distribution Check
       ↓
Data Drift Check
       ↓
Accept / Reject
```

For example:

```python
if len(df) == 0:
    raise ValueError(
        "Dataset is empty."
    )
```

Check expected columns:

```python
required = {
    "age",
    "income",
    "churn",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )
```

Production validation should be explicit, observable, and version-controlled.

---

# 🔹 Common Mistakes

## ❌ Training immediately

```text
Load CSV
 ↓
model.fit()
```

### Better

```text
Load
 ↓
Inspect
 ↓
Validate
 ↓
Understand
 ↓
Clean
 ↓
Split
 ↓
Train
```

---

## ❌ Assuming column types

A column containing:

```text
"100"
"200"
"300"
```

may be loaded as text.

Check:

```python
print(df.dtypes)
```

---

## ❌ Ignoring missing values

Missing values can affect:

* Statistics
* Training
* Feature engineering
* Model compatibility

---

## ❌ Removing every duplicate

Duplicates can be legitimate observations.

Investigate first.

---

## ❌ Removing every outlier

An outlier may be:

* A real customer
* A rare event
* A legitimate transaction
* A data-entry error

Context matters.

---

## ❌ Treating IDs as normal numerical features

Identifiers often represent labels rather than meaningful quantities.

---

## ❌ Ignoring target imbalance

A 95/5 target split has different modeling implications than a 50/50 split.

---

## ❌ Looking only at the first rows

The first rows may not represent the entire dataset.

Use:

```python
df.sample(
    10,
    random_state=42,
)
```

---

## ❌ Assuming correlation means causation

Correlation is evidence of association, not proof of causal relationships.

---

## ❌ Inspecting the full dataset after preprocessing only

You should understand the raw dataset before transformations change its structure.

---

# 🔹 Mini Projects

## 🟢 Project 1 — Dataset Profiler

Build a Python script that accepts:

```text
CSV file
```

and generates:

```text
Dataset size
Column names
Data types
Missing values
Duplicates
Unique values
Statistics
```

---

## 🟢 Project 2 — Data Quality Report

Generate a report containing:

```text
Dataset overview
Missing percentage
Duplicate percentage
Cardinality
Numeric statistics
Categorical statistics
```

Save it as:

```text
data_quality_report.txt
```

---

## 🟡 Project 3 — ML Dataset Inspector

Create a tool that accepts:

```text
dataset path
target column
```

and automatically reports:

```text
Feature count
Target type
Class distribution
Missing values
Numeric features
Categorical features
High-cardinality columns
Potential identifiers
```

---

## 🟡 Project 4 — Dataset Validation Framework

Create reusable validation rules:

```text
Required columns
Allowed data types
Minimum rows
Maximum missing percentage
Allowed ranges
Unique ID requirements
Target requirements
```

---

## 🔴 Project 5 — Production Data Quality Monitor

Build:

```text
Incoming Dataset
      ↓
Schema Validation
      ↓
Data Quality Checks
      ↓
Distribution Checks
      ↓
Drift Detection
      ↓
Validation Report
      ↓
Accept / Reject
```

---

# 🔹 Exercises

### Beginner

1. What is dataset inspection?
2. Why should data be inspected before ML?
3. What does `df.shape` return?
4. What does `df.info()` show?
5. How do you find missing values?
6. How do you count duplicate rows?
7. How do you find unique values?
8. How do you inspect the first five rows?

### Intermediate

9. Calculate missing-value percentages.
10. Find numeric columns.
11. Find categorical columns.
12. Calculate descriptive statistics.
13. Calculate feature cardinality.
14. Identify constant columns.
15. Identify potential outliers using IQR.
16. Inspect class distribution.
17. Parse a date column.
18. Generate a dataset quality report.

### Advanced

19. Build a reusable dataset profiler.
20. Create schema validation.
21. Add range validation.
22. Build automated target inspection.
23. Detect high-cardinality features.
24. Detect suspicious identifier columns.
25. Build train/test comparison reports.
26. Create automated leakage checks.
27. Add dataset validation to an ML pipeline.
28. Design production data-quality monitoring.

---

# 🔹 Project Structure

A professional dataset inspection module:

```text
08-Dataset-Inspection/
│
├── README.md
│
├── inspect.py
├── profiler.py
├── validators.py
├── statistics.py
├── report.py
│
├── data/
│   └── sample.csv
│
├── reports/
│   └── data-quality-report.txt
│
├── tests/
│   ├── test_inspect.py
│   ├── test_validators.py
│   └── test_profiler.py
│
├── requirements.txt
└── pyproject.toml
```

---

# 🔹 End-to-End Inspection Example

The following example combines the most important inspection steps into one reusable program.

```python
"""
Dataset inspection example.

Run:
    python inspect_dataset.py data.csv

The script reports:
- Dataset shape
- Column names
- Data types
- Missing values
- Duplicates
- Cardinality
- Numeric statistics
- Categorical distributions
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def inspect_dataset(
    df: pd.DataFrame,
) -> None:
    """Print a structured dataset inspection report."""

    print("=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Shape
    # ---------------------------------------------------------

    rows, columns = df.shape

    print("\n[1] Shape")
    print(f"Rows: {rows}")
    print(f"Columns: {columns}")

    # ---------------------------------------------------------
    # 2. Columns
    # ---------------------------------------------------------

    print("\n[2] Columns")

    for column in df.columns:
        print(f"- {column}")

    # ---------------------------------------------------------
    # 3. Data types
    # ---------------------------------------------------------

    print("\n[3] Data Types")
    print(df.dtypes)

    # ---------------------------------------------------------
    # 4. Missing values
    # ---------------------------------------------------------

    print("\n[4] Missing Values")

    missing = pd.DataFrame({
        "count": df.isna().sum(),
        "percentage": (
            df.isna()
            .mean()
            .mul(100)
        ),
    })

    print(
        missing
        .sort_values(
            "percentage",
            ascending=False,
        )
    )

    # ---------------------------------------------------------
    # 5. Duplicates
    # ---------------------------------------------------------

    print("\n[5] Duplicate Rows")

    duplicate_count = (
        df.duplicated().sum()
    )

    print(
        f"Count: {duplicate_count}"
    )

    # ---------------------------------------------------------
    # 6. Cardinality
    # ---------------------------------------------------------

    print("\n[6] Cardinality")

    print(
        df.nunique(
            dropna=False
        )
        .sort_values(
            ascending=False
        )
    )

    # ---------------------------------------------------------
    # 7. Numeric statistics
    # ---------------------------------------------------------

    print("\n[7] Numeric Statistics")

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        print("No numeric columns.")
    else:
        print(
            numeric_df.describe()
        )

    # ---------------------------------------------------------
    # 8. Categorical columns
    # ---------------------------------------------------------

    print("\n[8] Categorical Columns")

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
    )

    if len(categorical_columns) == 0:
        print("No categorical columns.")
    else:
        for column in categorical_columns:
            print(f"\n{column}")

            print(
                df[column]
                .value_counts(
                    dropna=False
                )
                .head(10)
            )

    # ---------------------------------------------------------
    # 9. Sample
    # ---------------------------------------------------------

    print("\n[9] Sample Records")

    print(
        df.sample(
            min(len(df), 5),
            random_state=42,
        )
    )

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python inspect_dataset.py <dataset.csv>"
        )
        raise SystemExit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        print(
            f"Dataset not found: {path}"
        )
        raise SystemExit(1)

    df = pd.read_csv(path)

    inspect_dataset(df)


if __name__ == "__main__":
    main()
```

Run:

```bash
python inspect_dataset.py data.csv
```

---

# 🔹 Dataset Inspection Checklist

Before moving to data cleaning, verify:

### Structure

* [ ] Dataset loaded successfully
* [ ] Row count known
* [ ] Column count known
* [ ] Column names reviewed
* [ ] Duplicate column names checked

### Data Types

* [ ] Numeric columns identified
* [ ] Categorical columns identified
* [ ] Date columns identified
* [ ] Text columns identified
* [ ] IDs identified

### Quality

* [ ] Missing values inspected
* [ ] Duplicate rows inspected
* [ ] Constant columns identified
* [ ] Cardinality inspected
* [ ] Invalid ranges investigated
* [ ] Potential outliers investigated

### Target

* [ ] Target identified
* [ ] Target data type checked
* [ ] Target missing values checked
* [ ] Class distribution checked
* [ ] Regression target distribution checked

### ML

* [ ] Potential leakage investigated
* [ ] Identifier features reviewed
* [ ] Time-dependent features reviewed
* [ ] Dataset size considered
* [ ] Train/test strategy considered

---

# 🔹 ML Readiness Checklist

A dataset is not automatically ML-ready just because it loads successfully.

Before modeling, ask:

```text
Is the target clearly defined?
        ↓
Are the features meaningful?
        ↓
Are data types correct?
        ↓
Are missing values understood?
        ↓
Are duplicates understood?
        ↓
Are outliers understood?
        ↓
Are categorical variables identified?
        ↓
Are IDs separated from features?
        ↓
Is target imbalance understood?
        ↓
Is leakage ruled out?
        ↓
Is the split strategy appropriate?
        ↓
Is the dataset representative?
```

Only after these questions should you move toward:

```text
Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Modeling
```

---

# 🔹 Learning Roadmap

Follow this progression:

```text
Load Dataset
     ↓
Shape
     ↓
Columns
     ↓
Data Types
     ↓
Samples
     ↓
Missing Values
     ↓
Duplicates
     ↓
Unique Values
     ↓
Cardinality
     ↓
Statistics
     ↓
Distribution
     ↓
Outliers
     ↓
Target
     ↓
Correlation
     ↓
Schema Validation
     ↓
Leakage Inspection
     ↓
Data Quality Report
     ↓
Data Cleaning
     ↓
EDA
     ↓
Feature Engineering
     ↓
Machine Learning
```

---

# 🔹 Key Takeaways

> Dataset inspection is the first serious quality-control stage of a Machine Learning workflow.

Remember:

1. **Always inspect data before modeling.**
2. **Understand rows, columns, and schema.**
3. **Check data types explicitly.**
4. **Inspect missing values and their percentages.**
5. **Investigate duplicates instead of automatically deleting them.**
6. **Check unique values and feature cardinality.**
7. **Use descriptive statistics to understand numerical features.**
8. **Inspect distributions, not just averages.**
9. **Investigate outliers rather than automatically removing them.**
10. **Understand categorical variables separately from numerical variables.**
11. **Identify date/time columns and their temporal ranges.**
12. **Inspect the target variable carefully.**
13. **Check class imbalance for classification problems.**
14. **Identify identifier columns.**
15. **Validate domain-specific ranges.**
16. **Check for potential data leakage.**
17. **Inspect train/test datasets appropriately.**
18. **Automate repeatable data-quality checks.**
19. **Treat dataset inspection as part of the ML pipeline.**
20. **Do not move to modeling until you understand what the dataset represents.**

---

# 🔹 Data Collection & Understanding Roadmap

The repository workflow now becomes:

```text
04-Data-Collection-and-Understanding
│
├── 01-Data-Sources
│
├── 02-CSV-Data
│
├── 03-Excel-Data
│
├── 04-JSON-Data
│
├── 05-SQL-Data
│
├── 06-API-Data
│
├── 07-Web-Scraping
│
└── 08-Dataset-Inspection
        │
        ▼
    Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
 Feature Engineering
        │
        ▼
 Machine Learning
```

The progression is intentional:

```text
Collect Data
     ↓
Understand Data
     ↓
Inspect Data
     ↓
Clean Data
     ↓
Explore Data
     ↓
Engineer Features
     ↓
Train Models
```

---

# 🔹 Next Step

After Dataset Inspection, the natural next stage is **Data Cleaning**.

The next chapter should focus on transforming identified data-quality problems into a reliable dataset:

```text
Dataset Inspection
       ↓
Missing Values
       ↓
Duplicates
       ↓
Invalid Values
       ↓
Incorrect Data Types
       ↓
Outliers
       ↓
Inconsistent Categories
       ↓
Data Cleaning
       ↓
Validated Dataset
       ↓
EDA
       ↓
Feature Engineering
       ↓
Machine Learning
```

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

GitHub:
[https://github.com/Kishor055](https://github.com/Kishor055)

---

# 🤝 Contributing

Contributions are welcome!

If you find an issue or want to improve this learning material:

1. Fork the repository.
2. Create a feature branch.
3. Add your changes.
4. Test all code examples.
5. Commit your changes.
6. Open a Pull Request.

Please keep contributions:

* Beginner-friendly
* Technically accurate
* Reproducible
* Well documented
* Consistent with the repository structure

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples

**Learn → Inspect → Clean → Explore → Engineer → Model → Deploy 🚀**

Happy Learning! 🐍📊🤖

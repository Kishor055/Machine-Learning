# 🧹 Data Quality for Machine Learning

Data quality is one of the most important foundations of a reliable Machine Learning system.

A Machine Learning model can only be as reliable as the data used to train and evaluate it.

Poor-quality data can introduce:

* Missing values
* Duplicate records
* Incorrect data types
* Invalid values
* Outliers
* Inconsistent categories
* Impossible dates
* Broken relationships
* Data leakage
* Sampling bias
* Measurement errors
* Incorrect labels
* Distribution shifts
* Schema violations

These problems can silently reduce model performance and make predictions unreliable.

This chapter explains how to **measure, diagnose, validate, improve, and monitor data quality** before data reaches a Machine Learning pipeline.

---

## 📚 Table of Contents

1. [What is Data Quality?](#-what-is-data-quality)
2. [Why Data Quality Matters](#-why-data-quality-matters)
3. [Data Quality Dimensions](#-data-quality-dimensions)
4. [Data Quality Lifecycle](#-data-quality-lifecycle)
5. [Data Quality vs Data Cleaning](#-data-quality-vs-data-cleaning)
6. [Completeness](#-1-completeness)
7. [Uniqueness](#-2-uniqueness)
8. [Validity](#-3-validity)
9. [Accuracy](#-4-accuracy)
10. [Consistency](#-5-consistency)
11. [Timeliness](#-6-timeliness)
12. [Integrity](#-7-integrity)
13. [Missing Data Quality](#-missing-data-quality)
14. [Duplicate Data](#-duplicate-data)
15. [Data Type Quality](#-data-type-quality)
16. [Range Validation](#-range-validation)
17. [Categorical Validation](#-categorical-validation)
18. [Date and Time Validation](#-date-and-time-validation)
19. [String Quality](#-string-quality)
20. [Numerical Quality](#-numerical-quality)
21. [Outlier Quality Checks](#-outlier-quality-checks)
22. [Target Quality](#-target-quality)
23. [Relationship Validation](#-relationship-validation)
24. [Business Rules](#-business-rules)
25. [Schema Validation](#-schema-validation)
26. [Data Leakage](#-data-leakage)
27. [Class Imbalance](#-class-imbalance)
28. [Distribution Quality](#-distribution-quality)
29. [Data Drift](#-data-drift)
30. [Quality Metrics](#-quality-metrics)
31. [Data Quality Score](#-data-quality-score)
32. [Python Implementation](#-python-implementation)
33. [Reusable Quality Functions](#-reusable-quality-functions)
34. [Automated Data Quality Report](#-automated-data-quality-report)
35. [Production Data Validation](#-production-data-validation)
36. [Data Quality Pipeline](#-data-quality-pipeline)
37. [Common Data Quality Problems](#-common-data-quality-problems)
38. [Common Mistakes](#-common-mistakes)
39. [Best Practices](#-best-practices)
40. [Mini Projects](#-mini-projects)
41. [Exercises](#-exercises)
42. [Project Structure](#-project-structure)
43. [End-to-End Example](#-end-to-end-example)
44. [Data Quality Checklist](#-data-quality-checklist)
45. [Roadmap](#-roadmap)
46. [Key Takeaways](#-key-takeaways)
47. [Next Step](#-next-step)

---

# 🎯 What is Data Quality?

**Data quality** describes how suitable a dataset is for its intended purpose.

For Machine Learning, high-quality data should generally be:

* Complete
* Valid
* Accurate
* Consistent
* Unique
* Relevant
* Timely
* Correctly labeled
* Representative
* Free from unintended leakage

A simple conceptual model is:

```text
Raw Data
   │
   ▼
Data Quality Checks
   │
   ├── Missing Values
   ├── Duplicates
   ├── Invalid Values
   ├── Wrong Types
   ├── Outliers
   ├── Inconsistencies
   ├── Schema Violations
   ├── Leakage
   └── Distribution Problems
   │
   ▼
Clean / Validated Data
   │
   ▼
EDA
   │
   ▼
Feature Engineering
   │
   ▼
Machine Learning
```

---

# 🚨 Why Data Quality Matters

Consider a house-price dataset:

```text
Area      Bedrooms    Price
1000      2           5000000
1200      3           6500000
-500      4           7000000
1500      NaN         8000000
1500      3           8000000
1500      3           8000000
```

Potential problems:

* `Area = -500` is invalid.
* `Bedrooms = NaN` is missing.
* The final two rows may be duplicates.
* Price may use inconsistent units.
* The dataset may contain additional hidden problems.

A model trained on this data may learn relationships that do not represent reality.

---

# 📐 Data Quality Dimensions

Data quality is multidimensional.

| Dimension          | Question                                          |
| ------------------ | ------------------------------------------------- |
| Completeness       | Is required information present?                  |
| Uniqueness         | Are duplicate records avoided?                    |
| Validity           | Are values allowed by the schema?                 |
| Accuracy           | Does the data represent reality?                  |
| Consistency        | Do related values agree?                          |
| Timeliness         | Is the data current enough?                       |
| Integrity          | Are relationships preserved?                      |
| Relevance          | Is the data appropriate for the task?             |
| Representativeness | Does the dataset reflect the intended population? |

---

# 🔄 Data Quality Lifecycle

A practical workflow:

```text
1. Define Data Requirements
          ↓
2. Collect Data
          ↓
3. Inspect Data
          ↓
4. Validate Schema
          ↓
5. Measure Quality
          ↓
6. Identify Problems
          ↓
7. Investigate Root Causes
          ↓
8. Correct / Remove / Impute
          ↓
9. Revalidate
          ↓
10. Monitor Continuously
```

The important principle is:

> **Do not clean blindly. Understand why a quality problem exists before fixing it.**

---

# 🧹 Data Quality vs Data Cleaning

These concepts are related but different.

### Data Quality

Measures whether the data is suitable.

```text
How good is the data?
```

### Data Cleaning

Changes the data to address identified problems.

```text
How do we improve the data?
```

Example:

```python
missing_rate = df["age"].isna().mean()

print(f"Missing rate: {missing_rate:.2%}")
```

This measures quality.

Cleaning could then be:

```python
df["age"] = df["age"].fillna(df["age"].median())
```

---

# 1️⃣ Completeness

Completeness measures how much required information is present.

## Missing-value rate

```python
missing_rate = df.isna().mean()

print(missing_rate)
```

Example:

```text
age       0.02
salary    0.05
city      0.00
```

Meaning:

* 2% of `age` values are missing.
* 5% of `salary` values are missing.
* `city` has no missing values.

---

## Dataset completeness

```python
total_cells = df.size
missing_cells = df.isna().sum().sum()

completeness = 1 - (missing_cells / total_cells)

print(f"Completeness: {completeness:.2%}")
```

---

## Column completeness

```python
completeness = df.notna().mean()

print(completeness)
```

---

# 2️⃣ Uniqueness

Uniqueness checks whether records or identifiers are duplicated.

## Duplicate rows

```python
duplicate_count = df.duplicated().sum()

print("Duplicates:", duplicate_count)
```

## Duplicate percentage

```python
duplicate_rate = df.duplicated().mean()

print(f"Duplicate rate: {duplicate_rate:.2%}")
```

---

## Unique identifier validation

```python
is_unique = df["customer_id"].is_unique

print("Unique ID:", is_unique)
```

If `customer_id` is supposed to uniquely identify customers, duplicates indicate a quality issue.

---

# 3️⃣ Validity

Validity checks whether values follow predefined rules.

Example:

```text
Age:
0 <= age <= 120
```

Validation:

```python
invalid_age = ~df["age"].between(0, 120)

print(df.loc[invalid_age])
```

---

# 4️⃣ Accuracy

Accuracy asks:

> Does the value represent reality?

This is often harder to measure automatically.

Example:

```text
customer_age = 25
```

The value may be syntactically valid but factually incorrect.

A database may contain:

```text
Age = 250
```

This is both invalid and likely inaccurate.

But:

```text
Age = 35
```

could be valid while still being wrong if the person's actual age is 40.

Accuracy often requires:

* Trusted reference data
* Human verification
* External systems
* Domain knowledge
* Measurement validation

---

# 5️⃣ Consistency

Consistency means related values do not contradict each other.

Example:

```text
country = "India"
currency = "USD"
```

Potential inconsistency depending on the business context.

Another example:

```text
start_date = 2026-05-01
end_date   = 2026-04-01
```

This violates the expected temporal relationship.

```python
invalid_dates = df["end_date"] < df["start_date"]

print(df.loc[invalid_dates])
```

---

# 6️⃣ Timeliness

Timeliness measures whether data is available and updated when needed.

For real-time systems:

```text
Event Time
    ↓
Data Arrival
    ↓
Processing
    ↓
Prediction
```

A dataset can be accurate but too old to be useful.

Example:

```python
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

age = pd.Timestamp.now() - df["timestamp"]

print(age.describe())
```

For production systems, define an explicit freshness threshold.

---

# 7️⃣ Integrity

Integrity checks whether relationships between datasets remain valid.

Consider:

```text
customers
    customer_id

orders
    customer_id
```

Every order should normally reference a valid customer.

Conceptually:

```text
orders.customer_id
        │
        ▼
customers.customer_id
```

This is called **referential integrity**.

---

# 🕳️ Missing Data Quality

Missing values can be represented as:

```text
NaN
None
NaT
pd.NA
""
"unknown"
"N/A"
"?"
```

Not every missing representation is automatically recognized by Pandas.

Example:

```python
df = pd.DataFrame({
    "name": ["A", "B", "C"],
    "age": [20, None, 30]
})

print(df.isna())
```

---

## Missing-value summary

```python
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_rate": df.isna().mean()
})

print(missing)
```

---

## Missingness patterns

Missingness itself can contain information.

For example:

```text
income missing → self-employed customer
```

Do not automatically assume:

```text
missing = random
```

Investigate the reason for missingness.

---

# ♻️ Duplicate Data

Detect duplicates:

```python
duplicates = df[df.duplicated(keep=False)]

print(duplicates)
```

Remove exact duplicates:

```python
df = df.drop_duplicates()
```

Duplicate detection may also need business keys.

Example:

```python
duplicate_customers = df.duplicated(
    subset=["email"],
    keep=False
)

print(df.loc[duplicate_customers])
```

---

# 🔢 Data Type Quality

Inspect data types:

```python
print(df.dtypes)
```

Example:

```text
age           int64
salary      float64
name         object
created_at   object
```

A date stored as `object` may require conversion:

```python
df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce"
)
```

---

# 📏 Range Validation

Range checks are one of the simplest quality rules.

### Age

```python
valid_age = df["age"].between(0, 120)
```

### Percentage

```python
valid_percentage = df["conversion_rate"].between(0, 100)
```

### Probability

```python
valid_probability = df["probability"].between(0, 1)
```

### Positive amount

```python
valid_amount = df["amount"] >= 0
```

Find invalid records:

```python
invalid = df.loc[~valid_age]

print(invalid)
```

---

# 🏷️ Categorical Validation

Categorical columns should contain expected values.

Example:

```python
allowed = {
    "Male",
    "Female",
    "Other"
}

invalid = ~df["gender"].isin(allowed)

print(df.loc[invalid])
```

For production systems, keep allowed categories centralized:

```python
ALLOWED_GENDERS = {
    "Male",
    "Female",
    "Other"
}
```

---

## Category normalization

These values may represent the same category:

```text
India
india
INDIA
 India
```

Normalize carefully:

```python
df["country"] = (
    df["country"]
    .astype("string")
    .str.strip()
    .str.lower()
)
```

However, normalization rules should be defined based on domain requirements rather than applied blindly.

---

# 📅 Date and Time Validation

Convert dates:

```python
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)
```

Find invalid dates:

```python
invalid_dates = df["date"].isna()

print(df.loc[invalid_dates])
```

Check future dates:

```python
future_dates = df["date"] > pd.Timestamp.now()

print(df.loc[future_dates])
```

Check chronological consistency:

```python
invalid_periods = df["end_date"] < df["start_date"]

print(df.loc[invalid_periods])
```

---

# 🔤 String Quality

Common string problems include:

* Leading whitespace
* Trailing whitespace
* Inconsistent casing
* Empty strings
* Unexpected characters
* Different spellings
* Incorrect encoding

Example:

```python
df["name"] = df["name"].astype("string")

empty_names = df["name"].str.strip().eq("")

print(df.loc[empty_names])
```

Normalize:

```python
df["city"] = (
    df["city"]
    .astype("string")
    .str.strip()
)
```

---

# 🔢 Numerical Quality

Numerical validation should check:

```text
Missing values
↓
Infinite values
↓
Invalid ranges
↓
Unexpected precision
↓
Extreme values
↓
Unit consistency
```

Check infinite values:

```python
import numpy as np

numeric_columns = df.select_dtypes(
    include=np.number
).columns

infinite_counts = np.isinf(
    df[numeric_columns]
).sum()

print(infinite_counts)
```

---

# 📊 Outlier Quality Checks

Outliers are not automatically errors.

An outlier may represent:

* A genuine rare event
* A measurement error
* A data-entry mistake
* Fraud
* A special population
* A distribution problem

Therefore:

```text
Outlier ≠ Automatically Bad Data
```

---

## IQR Method

The Interquartile Range is:

```text
IQR = Q3 - Q1
```

Lower boundary:

```text
Q1 - 1.5 × IQR
```

Upper boundary:

```text
Q3 + 1.5 × IQR
```

Python:

```python
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df.loc[
    (df["salary"] < lower) |
    (df["salary"] > upper)
]

print(outliers)
```

---

# 🎯 Target Quality

The target variable requires special attention.

For classification:

```python
print(df["target"].value_counts(dropna=False))
```

Check missing target values:

```python
print(df["target"].isna().sum())
```

For regression:

```python
print(df["target"].describe())
```

Investigate:

* Missing labels
* Invalid labels
* Label imbalance
* Unexpected classes
* Incorrect units
* Label leakage
* Label noise

---

# 🔗 Relationship Validation

Data quality is not only about individual columns.

Sometimes multiple columns must satisfy a relationship.

Example:

```text
total = quantity × unit_price
```

Validation:

```python
expected_total = df["quantity"] * df["unit_price"]

difference = (
    df["total"] - expected_total
).abs()

invalid = difference > 0.01

print(df.loc[invalid])
```

---

# 🧠 Business Rules

Business rules encode domain knowledge.

Examples:

```text
age >= 18 for adult accounts

delivery_date >= order_date

quantity > 0

discount between 0 and 100

salary >= 0

end_date >= start_date
```

Python:

```python
rules = {
    "age_valid": df["age"].between(0, 120),
    "salary_valid": df["salary"] >= 0,
    "quantity_valid": df["quantity"] > 0,
}
```

Calculate violations:

```python
for name, condition in rules.items():
    violations = (~condition).sum()

    print(
        f"{name}: {violations} violations"
    )
```

---

# 🧱 Schema Validation

A schema defines expected structure.

Example:

```python
expected_schema = {
    "customer_id": "int64",
    "age": "int64",
    "salary": "float64",
    "city": "object",
}
```

Check columns:

```python
expected_columns = set(expected_schema)

actual_columns = set(df.columns)

missing_columns = expected_columns - actual_columns
unexpected_columns = actual_columns - expected_columns

print("Missing:", missing_columns)
print("Unexpected:", unexpected_columns)
```

Schema validation should happen before downstream processing.

---

# ⚠️ Data Leakage

Data leakage occurs when information unavailable at prediction time influences model training.

Example:

```text
Customer Churn Prediction
```

Suppose:

```text
customer_id
age
subscription
monthly_usage
cancellation_date
```

If predicting whether a customer will cancel, using `cancellation_date` may reveal the outcome directly.

The model can appear highly accurate while failing in real-world usage.

---

## Leakage questions

For every feature ask:

1. Was this value available at prediction time?
2. Was it generated after the target event?
3. Does it directly encode the target?
4. Was preprocessing performed before train/test splitting?
5. Did information flow from validation/test data into training?

---

# ⚖️ Class Imbalance

For classification:

```python
distribution = df["target"].value_counts(
    normalize=True
)

print(distribution)
```

Example:

```text
No     0.95
Yes    0.05
```

This does not automatically mean the dataset is invalid.

It means the class distribution must be understood.

Potential considerations:

* Stratified splitting
* Appropriate evaluation metrics
* Resampling
* Class weights
* Threshold selection
* Domain requirements

---

# 📈 Distribution Quality

Inspect numerical distributions:

```python
print(df.describe())
```

Compare:

```text
Training Distribution
        ↓
Validation Distribution
        ↓
Test Distribution
```

Large unexplained differences may indicate:

* Sampling problems
* Collection differences
* Data leakage
* Distribution shift
* Time effects

---

# 🌊 Data Drift

Data drift occurs when incoming data changes relative to a reference dataset.

Conceptually:

```text
Training Data
     │
     ▼
Reference Distribution
     │
     │ compare
     ▼
Production Data
```

Example:

```text
Training age:
Mean = 35

Production age:
Mean = 52
```

This does not automatically prove a problem, but it should trigger investigation.

Common monitoring signals:

* Mean
* Median
* Standard deviation
* Missing rate
* Category frequencies
* Quantiles
* Distribution statistics

---

# 📏 Quality Metrics

Useful metrics include:

### Completeness

```text
1 - missing_rate
```

### Duplicate rate

```text
duplicate_rows / total_rows
```

### Validity rate

```text
valid_records / total_records
```

### Error rate

```text
invalid_records / total_records
```

### Null rate

```text
null_values / total_values
```

### Coverage

```text
records_received / records_expected
```

---

# 🧮 Data Quality Score

A simple educational quality score can combine multiple dimensions.

For example:

```text
Quality Score =
    0.25 × Completeness
  + 0.20 × Validity
  + 0.20 × Uniqueness
  + 0.15 × Consistency
  + 0.20 × Integrity
```

Python:

```python
score = (
    0.25 * completeness +
    0.20 * validity +
    0.20 * uniqueness +
    0.15 * consistency +
    0.20 * integrity
)

print(f"Quality Score: {score:.2%}")
```

### Important

This is an **example framework**, not a universal industry standard.

Weights should depend on:

* Business requirements
* ML task
* Risk level
* Data source
* Domain
* Regulatory requirements

A single score should never replace detailed quality diagnostics.

---

# 🐍 Python Implementation

Example dataset:

```python
import pandas as pd


def create_sample_data():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 4],
        "age": [25, 31, None, 200, 40],
        "salary": [50000, 65000, 72000, -1000, 90000],
        "city": [
            "Pune",
            "Mumbai",
            "Pune",
            "Delhi",
            " Mumbai "
        ],
        "target": [
            0,
            1,
            0,
            1,
            None
        ],
    })
```

---

## Basic Inspection

```python
def inspect_basic(df):
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isna().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())
```

---

# 🔍 Reusable Quality Functions

## Missing-value report

```python
def missing_report(df):
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_rate": df.isna().mean(),
    })

    return report.sort_values(
        "missing_rate",
        ascending=False
    )
```

---

## Duplicate report

```python
def duplicate_report(df):
    return pd.DataFrame({
        "duplicate_rows": [df.duplicated().sum()],
        "duplicate_rate": [df.duplicated().mean()],
    })
```

---

## Numeric report

```python
def numeric_quality_report(df):
    numeric = df.select_dtypes(
        include="number"
    )

    return numeric.describe().T
```

---

## Cardinality report

```python
def cardinality_report(df):
    return pd.DataFrame({
        "unique_values": df.nunique(dropna=False),
        "unique_rate": (
            df.nunique(dropna=False) / len(df)
        ),
    }).sort_values(
        "unique_values",
        ascending=False
    )
```

---

# 🤖 Automated Data Quality Report

A reusable quality-report function:

```python
import pandas as pd


def quality_report(df):
    report = pd.DataFrame(index=df.columns)

    report["dtype"] = df.dtypes.astype(str)

    report["missing_count"] = (
        df.isna().sum()
    )

    report["missing_rate"] = (
        df.isna().mean()
    )

    report["unique_count"] = (
        df.nunique(dropna=False)
    )

    report["unique_rate"] = (
        df.nunique(dropna=False) / len(df)
    )

    report["constant"] = (
        report["unique_count"] <= 1
    )

    return report
```

Usage:

```python
report = quality_report(df)

print(report)
```

---

# 🏭 Production Data Validation

Production systems should validate data **before** model inference.

Example:

```text
Incoming Data
     │
     ▼
Schema Validation
     │
     ▼
Type Validation
     │
     ▼
Missing-Value Checks
     │
     ▼
Range Checks
     │
     ▼
Category Checks
     │
     ▼
Business Rules
     │
     ▼
Drift Checks
     │
     ▼
Model
```

Invalid data should not silently pass through the pipeline.

Depending on system requirements, invalid records may be:

* Rejected
* Quarantined
* Logged
* Corrected
* Sent for manual review
* Processed with controlled fallback logic

---

# 🔄 Data Quality Pipeline

A robust ML data pipeline might look like:

```text
             Data Source
                  │
                  ▼
          Data Ingestion
                  │
                  ▼
          Schema Validation
                  │
                  ▼
         Quality Validation
                  │
       ┌──────────┴──────────┐
       │                     │
     Valid                 Invalid
       │                     │
       ▼                     ▼
 Data Processing       Error / Quarantine
       │
       ▼
 Feature Engineering
       │
       ▼
 Train / Validation / Test
       │
       ▼
       Model
       │
       ▼
 Production Monitoring
```

---

# 🧪 Common Data Quality Problems

| Problem          | Example              | Detection              |
| ---------------- | -------------------- | ---------------------- |
| Missing values   | `age = NaN`          | `isna()`               |
| Duplicate rows   | Same record twice    | `duplicated()`         |
| Invalid range    | `age = -5`           | `between()`            |
| Wrong type       | Date as string       | `dtypes`               |
| Invalid category | `country = "XYZ"`    | `isin()`               |
| Empty strings    | `name = ""`          | `str.strip()`          |
| Infinite value   | `salary = inf`       | `np.isinf()`           |
| Impossible date  | End before start     | Comparison             |
| Wrong unit       | Dollars vs rupees    | Domain validation      |
| Leakage          | Future information   | Temporal analysis      |
| Class imbalance  | 98/2 split           | `value_counts()`       |
| Data drift       | Distribution changes | Statistical monitoring |

---

# ❌ Common Mistakes

## 1. Cleaning without investigating

Bad:

```python
df = df.dropna()
```

without understanding why values are missing.

Better:

```text
Detect
  ↓
Understand
  ↓
Choose strategy
  ↓
Clean
  ↓
Validate again
```

---

## 2. Removing all outliers

Not every outlier is incorrect.

An unusual transaction could be:

* Fraud
* A legitimate high-value transaction
* A rare customer
* A real-world event

---

## 3. Treating missing values as zero

This:

```python
df["income"] = df["income"].fillna(0)
```

may change the meaning of the data.

Missing income does not necessarily mean:

```text
income = 0
```

---

## 4. Ignoring categorical inconsistencies

These may represent the same category:

```text
Pune
pune
PUNE
 Pune
```

---

## 5. Ignoring schema changes

A production data source may suddenly change:

```text
age → customer_age
```

or:

```text
price → price_in_usd
```

Such changes can break downstream ML systems.

---

## 6. Measuring quality only once

Data quality is not a one-time activity.

Production data changes.

Therefore:

```text
Quality Checks
      +
Monitoring
      +
Alerts
```

are required for reliable systems.

---

# ✅ Best Practices

### 1. Define quality requirements before modeling

Know what valid data means.

### 2. Validate early

Catch problems before expensive processing.

### 3. Keep validation rules explicit

```python
MIN_AGE = 0
MAX_AGE = 120
```

### 4. Separate detection from correction

First identify the problem.

Then decide how to fix it.

### 5. Preserve raw data

Keep an immutable copy of original data when appropriate.

```text
raw/
processed/
validated/
```

### 6. Track data provenance

Record:

* Source
* Timestamp
* Version
* Transformation
* Owner

### 7. Log quality failures

Quality problems should be observable.

### 8. Test validation rules

Treat data validation as production code.

### 9. Monitor production data

Training-time validation is not enough.

### 10. Avoid leakage

Validate the information timeline.

### 11. Use domain knowledge

Statistical checks alone cannot detect every quality issue.

### 12. Revalidate after cleaning

Cleaning itself can introduce errors.

---

# 🧪 Mini Projects

## Project 1 — Student Dataset Quality Analyzer

Create a quality report for:

```text
student_id
name
age
gender
attendance
marks
department
```

Detect:

* Missing values
* Duplicate IDs
* Invalid ages
* Invalid marks
* Invalid attendance
* Category inconsistencies

---

## Project 2 — Customer Dataset Validator

Validate:

```text
customer_id
name
age
email
city
income
```

Requirements:

```text
age: 0–120
income: >= 0
customer_id: unique
email: non-empty
city: valid category
```

---

## Project 3 — E-Commerce Data Quality

Validate:

```text
order_id
customer_id
quantity
unit_price
discount
order_date
delivery_date
```

Rules:

```text
quantity > 0
unit_price >= 0
discount between 0 and 100
delivery_date >= order_date
```

---

## Project 4 — ML Dataset Quality Dashboard

Build a report containing:

```text
Dataset Size
Missing Rate
Duplicate Rate
Invalid Rate
Column Types
Cardinality
Outliers
Target Distribution
```

---

# 📝 Exercises

### Beginner

1. Calculate missing-value percentages.
2. Count duplicate rows.
3. Find unique values for every categorical column.
4. Detect negative values in a numeric column.
5. Check whether an ID column is unique.

### Intermediate

6. Build a reusable quality-report function.
7. Validate date relationships.
8. Detect invalid categories.
9. Detect infinite numerical values.
10. Calculate completeness.

### Advanced

11. Create a configurable validation framework.
12. Implement business-rule validation.
13. Build a data-quality score.
14. Compare training and production distributions.
15. Detect data drift.
16. Build automated quality tests.
17. Add quality checks to an ML pipeline.
18. Create a data-quality monitoring dashboard.

---

# 📁 Project Structure

A recommended implementation structure:

```text
09-Data-Quality/
│
├── README.md
│
├── data/
│   ├── raw/
│   ├── validated/
│   └── processed/
│
├── src/
│   ├── quality_checks.py
│   ├── schema.py
│   ├── validation.py
│   └── reports.py
│
├── tests/
│   ├── test_schema.py
│   ├── test_quality.py
│   └── test_validation.py
│
└── reports/
    └── quality_report.csv
```

---

# 🚀 End-to-End Example

```python
import numpy as np
import pandas as pd


def create_dataset():
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 4],
        "age": [25, 31, None, 200, 40],
        "salary": [50000, 65000, 72000, -1000, 90000],
        "city": [
            "Pune",
            "Mumbai",
            "Pune",
            "Delhi",
            " Mumbai "
        ],
        "target": [0, 1, 0, 1, None],
    })


def quality_report(df):
    report = pd.DataFrame(index=df.columns)

    report["dtype"] = df.dtypes.astype(str)
    report["missing_count"] = df.isna().sum()
    report["missing_rate"] = df.isna().mean()
    report["unique_count"] = df.nunique(dropna=False)
    report["unique_rate"] = (
        df.nunique(dropna=False) / len(df)
    )

    return report


def validate_dataset(df):
    errors = {}

    # Missing customer IDs
    errors["missing_customer_id"] = int(
        df["customer_id"].isna().sum()
    )

    # Duplicate customer IDs
    errors["duplicate_customer_id"] = int(
        df["customer_id"].duplicated().sum()
    )

    # Invalid ages
    errors["invalid_age"] = int(
        (~df["age"].between(0, 120)).fillna(True).sum()
    )

    # Invalid salaries
    errors["invalid_salary"] = int(
        (df["salary"] < 0).sum()
    )

    # Infinite values
    numeric = df.select_dtypes(
        include=np.number
    )

    errors["infinite_values"] = int(
        np.isinf(numeric).sum().sum()
    )

    # Missing target
    errors["missing_target"] = int(
        df["target"].isna().sum()
    )

    return errors


def main():
    df = create_dataset()

    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print(df)

    print("\n" + "=" * 60)
    print("QUALITY REPORT")
    print("=" * 60)
    print(quality_report(df))

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    errors = validate_dataset(df)

    for name, count in errors.items():
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
```

---

# 🔬 Recommended Validation Strategy

A mature data-quality system should have multiple levels.

## Level 1 — Structural

```text
Does the dataset have the expected columns?
```

## Level 2 — Type

```text
Are columns the expected types?
```

## Level 3 — Field

```text
Are individual values valid?
```

## Level 4 — Relationship

```text
Do related fields agree?
```

## Level 5 — Distribution

```text
Does the dataset look statistically reasonable?
```

## Level 6 — Business

```text
Does the data satisfy domain rules?
```

## Level 7 — Temporal

```text
Was the information available at prediction time?
```

## Level 8 — Production

```text
Has the incoming data changed unexpectedly?
```

---

# 🛡️ Data Quality and Machine Learning

Data quality affects every stage of the ML lifecycle.

```text
Data Collection
      ↓
Data Quality
      ↓
EDA
      ↓
Cleaning
      ↓
Feature Engineering
      ↓
Training
      ↓
Evaluation
      ↓
Deployment
      ↓
Monitoring
```

A quality problem at the beginning can propagate through the entire pipeline.

For example:

```text
Incorrect Labels
       ↓
Incorrect Training Data
       ↓
Incorrect Model
       ↓
Incorrect Predictions
       ↓
Incorrect Business Decisions
```

---

# 📊 Quality Monitoring in Production

Production monitoring can track:

```text
Metric                  Current     Threshold
------------------------------------------------
Missing Rate             2.1%        < 5%
Duplicate Rate           0.3%        < 1%
Invalid Age Rate         0.1%        < 0.5%
Schema Errors            0           0
Unknown Categories       2           < 10
Feature Drift            Low         Defined threshold
```

When thresholds are exceeded:

```text
Metric Violation
      ↓
Alert
      ↓
Investigation
      ↓
Root Cause Analysis
      ↓
Correction
      ↓
Revalidation
```

---

# 🧠 Root Cause Analysis

Finding a bad value is only the first step.

Example:

```text
Problem:
20% missing salaries
```

Possible causes:

```text
Database migration
       │
       ├── Column mapping changed
       │
       ├── API stopped returning salary
       │
       ├── New customers do not provide salary
       │
       └── Collection form changed
```

The correct solution depends on the cause.

This is why:

> **Data quality is an engineering and analytical problem, not simply a cleaning problem.**

---

# 🔐 Data Quality and Security

Quality checks should also consider security-sensitive data.

Examples:

* Unexpected columns
* Sensitive information
* Malformed input
* Unexpected payloads
* Injection-like strings
* Unauthorized data sources
* Schema manipulation

Never assume that unexpected data is harmless simply because it can be parsed.

---

# 🧩 Data Quality vs Model Quality

These are different.

### Data Quality

```text
Is the input data suitable?
```

### Model Quality

```text
Does the model perform well?
```

A high-performing model on a poor-quality test set may provide misleading results.

Therefore:

```text
Good Data
   +
Good Validation
   +
Good Model
   =
More Reliable ML System
```

---

# 📋 Data Quality Checklist

Before using a dataset:

### Structure

* [ ] Expected columns exist
* [ ] Unexpected columns identified
* [ ] Row count checked
* [ ] Data types verified

### Completeness

* [ ] Missing values measured
* [ ] Required fields checked
* [ ] Missingness patterns investigated

### Uniqueness

* [ ] Duplicate rows checked
* [ ] Primary identifiers checked
* [ ] Business-key duplicates checked

### Validity

* [ ] Numeric ranges checked
* [ ] Categories validated
* [ ] Dates validated
* [ ] Strings normalized where appropriate

### Consistency

* [ ] Related columns agree
* [ ] Units are consistent
* [ ] Dates follow expected order

### Integrity

* [ ] Foreign-key relationships checked
* [ ] Referential integrity validated

### ML-Specific

* [ ] Target checked
* [ ] Class distribution checked
* [ ] Leakage investigated
* [ ] Train/test contamination checked
* [ ] Distribution differences investigated

### Production

* [ ] Validation rules automated
* [ ] Quality thresholds defined
* [ ] Failures logged
* [ ] Monitoring implemented

---

# 🗺️ Roadmap

Your data understanding journey now looks like:

```text
04-Data-Collection-and-Understanding/
│
├── 01-Data-Sources/
│
├── 02-CSV-Data/
│
├── 03-Excel-Data/
│
├── 04-JSON-Data/
│
├── 05-SQL-Data/
│
├── 06-API-Data/
│
├── 07-Web-Scraping/
│
├── 08-Dataset-Inspection/
│
└── 09-Data-Quality/
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
   Model Development
```

The key progression is:

```text
Collect
  ↓
Inspect
  ↓
Validate
  ↓
Understand
  ↓
Clean
  ↓
Explore
  ↓
Engineer
  ↓
Model
```

---

# 🎯 Key Takeaways

1. **Data quality is foundational to Machine Learning.**
2. Always inspect data before modeling.
3. Completeness measures missing information.
4. Uniqueness detects duplicate records.
5. Validity checks whether values follow rules.
6. Accuracy concerns whether values represent reality.
7. Consistency checks relationships between values.
8. Timeliness matters for real-time and time-sensitive systems.
9. Outliers are not automatically errors.
10. Missing values require investigation before imputation.
11. Schema validation protects downstream pipelines.
12. Business rules capture domain-specific correctness.
13. Data leakage can produce misleadingly strong model performance.
14. Data quality should be measured continuously in production.
15. A single quality score should not replace detailed diagnostics.
16. Automated validation makes data quality reproducible.
17. Good data quality improves the reliability of the entire ML lifecycle.

---

# 🚀 Next Step

After understanding Data Quality, the next major topic is:

```text
09-Data-Quality
        ↓
10-Data-Cleaning
```

In **Data Cleaning**, you will learn how to transform identified quality problems into reliable, analysis-ready datasets using:

* Missing-value handling
* Duplicate removal
* Type conversion
* Outlier treatment
* Category normalization
* String cleaning
* Date cleaning
* Data transformation
* Validation
* Reproducible cleaning pipelines

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

GitHub:
`https://github.com/Kishor055`

---

# 🤝 Contributing

Contributions are welcome!

If you find an error or want to improve this learning material:

1. Fork the repository.
2. Create a feature branch.
3. Improve the documentation or examples.
4. Test your code.
5. Commit your changes.
6. Open a Pull Request.

Example:

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b improve-data-quality

git add .

git commit -m "Improve data quality documentation"

git push origin improve-data-quality
```

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork it
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples
* 📚 Continue through the complete ML roadmap

---

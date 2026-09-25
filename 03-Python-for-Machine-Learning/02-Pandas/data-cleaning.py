"""
Pandas Data Cleaning for Machine Learning
=========================================

File:
03-Python-for-Machine-Learning/02-Pandas/data-cleaning.py

Description:
A practical, beginner-to-advanced guide to data cleaning with Pandas.

Data cleaning is one of the most important stages of a Machine Learning
workflow. Real-world datasets commonly contain:

```
- Missing values
- Duplicate records
- Incorrect data types
- Inconsistent text
- Invalid values
- Outliers
- Formatting problems
- Inconsistent categorical labels
- Impossible numerical values
- Date/time inconsistencies
```

This module demonstrates how to identify, analyze, clean, validate,
and prepare tabular data for downstream Machine Learning workflows.

Topics Covered:
1. Creating a messy dataset
2. Inspecting data
3. Detecting missing values
4. Handling missing numerical values
5. Handling missing categorical values
6. Detecting duplicate records
7. Cleaning column names
8. Cleaning text data
9. Standardizing categorical values
10. Converting data types safely
11. Handling invalid numerical values
12. Detecting impossible values
13. Outlier detection using IQR
14. Outlier handling strategies
15. Date/time cleaning
16. Feature engineering during cleaning
17. Data validation
18. Train/test leakage awareness
19. Reusable cleaning functions
20. Complete end-to-end workflow

Requirements:
Python 3.9+
pandas
numpy

Optional:
scikit-learn

Install:
pip install pandas numpy scikit-learn

Run:
python data-cleaning.py

Important ML Principle:
Cleaning rules should be based on domain knowledge whenever possible.
A statistically unusual value is not automatically an incorrect value.

Author:
Kishor Patil
"""

from **future** import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

# =============================================================================

# 01. CREATE A MESSY DATASET

# =============================================================================

def create_sample_dataset() -> pd.DataFrame:
"""
Create a deliberately messy dataset for demonstration.

```
The dataset contains:
    - Missing values
    - Duplicate rows
    - Inconsistent capitalization
    - Extra whitespace
    - Invalid age
    - Invalid salary
    - Mixed date formats
    - Inconsistent city names
    - Invalid categorical values
"""

data = {
    " Customer Name ": [
        "Alice",
        " Bob ",
        "CHARLIE",
        "David",
        "Eva",
        "Alice",
        "Frank",
        None,
    ],
    "Age": [
        25,
        31,
        None,
        -5,
        29,
        25,
        150,
        42,
    ],
    "Gender": [
        "Female",
        "male",
        "FEMALE",
        "M",
        "F",
        "Female",
        "Unknown",
        None,
    ],
    "City": [
        "Mumbai",
        "mumbai ",
        "Pune",
        " PUNE",
        "Nashik",
        "Mumbai",
        "Pune",
        None,
    ],
    "Salary": [
        55000,
        62000,
        58000,
        -1000,
        None,
        55000,
        99999999,
        70000,
    ],
    "Join Date": [
        "2024-01-10",
        "10/02/2024",
        "2024-03-15",
        "invalid-date",
        "2024-05-20",
        "2024-01-10",
        "2024-07-10",
        None,
    ],
    "Department": [
        "IT",
        "it",
        "HR",
        "Finance",
        "HR",
        "IT",
        "IT",
        "finance",
    ],
}

return pd.DataFrame(data)
```

# =============================================================================

# 02. DATASET INSPECTION

# =============================================================================

def inspect_dataset(df: pd.DataFrame) -> None:
"""
Display a useful first-level dataset inspection.
"""

```
print("\n" + "=" * 80)
print("DATASET INSPECTION")
print("=" * 80)

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())
```

# =============================================================================

# 03. CLEAN COLUMN NAMES

# =============================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
"""
Standardize DataFrame column names.

```
Transformations:
    - Remove leading/trailing whitespace
    - Convert to lowercase
    - Replace spaces with underscores
"""

cleaned = df.copy()

cleaned.columns = (
    cleaned.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

return cleaned
```

# =============================================================================

# 04. CLEAN TEXT COLUMNS

# =============================================================================

def clean_text_columns(
df: pd.DataFrame,
columns: Iterable[str],
) -> pd.DataFrame:
"""
Strip whitespace and normalize text columns.

```
Missing values are preserved.
"""

cleaned = df.copy()

for column in columns:
    if column not in cleaned.columns:
        continue

    cleaned[column] = (
        cleaned[column]
        .astype("string")
        .str.strip()
    )

return cleaned
```

# =============================================================================

# 05. STANDARDIZE CATEGORICAL VALUES

# =============================================================================

def standardize_gender(df: pd.DataFrame) -> pd.DataFrame:
"""
Standardize gender labels.

```
Example mappings:

    female -> Female
    FEMALE -> Female
    f -> Female
    male -> Male
    m -> Male

Unknown/unrecognized values are converted to <NA>.
"""

cleaned = df.copy()

mapping = {
    "female": "Female",
    "f": "Female",
    "male": "Male",
    "m": "Male",
}

cleaned["gender"] = (
    cleaned["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map(mapping)
)

return cleaned
```

def standardize_city(df: pd.DataFrame) -> pd.DataFrame:
"""
Standardize city names.
"""

```
cleaned = df.copy()

cleaned["city"] = (
    cleaned["city"]
    .astype("string")
    .str.strip()
    .str.title()
)

return cleaned
```

def standardize_department(df: pd.DataFrame) -> pd.DataFrame:
"""
Standardize department names.
"""

```
cleaned = df.copy()

cleaned["department"] = (
    cleaned["department"]
    .astype("string")
    .str.strip()
    .str.title()
)

return cleaned
```

# =============================================================================

# 06. HANDLE MISSING VALUES

# =============================================================================

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
"""
Return a useful missing-value report.

```
Columns:
    missing_count
    missing_percentage
    data_type
"""

report = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": (
        df.isna().mean() * 100
    ),
    "data_type": df.dtypes.astype(str),
})

report = report.sort_values(
    "missing_count",
    ascending=False,
)

return report
```

def fill_numeric_missing_values(
df: pd.DataFrame,
columns: Iterable[str],
strategy: str = "median",
) -> pd.DataFrame:
"""
Fill missing numerical values.

```
Supported strategies:
    - mean
    - median
    - zero

Median is often a useful default when numerical features
contain outliers.
"""

cleaned = df.copy()

for column in columns:
    if column not in cleaned.columns:
        continue

    if strategy == "mean":
        value = cleaned[column].mean()

    elif strategy == "median":
        value = cleaned[column].median()

    elif strategy == "zero":
        value = 0

    else:
        raise ValueError(
            "strategy must be 'mean', 'median', or 'zero'"
        )

    cleaned[column] = cleaned[column].fillna(value)

return cleaned
```

def fill_categorical_missing_values(
df: pd.DataFrame,
columns: Iterable[str],
strategy: str = "mode",
fill_value: str = "Unknown",
) -> pd.DataFrame:
"""
Fill missing categorical values.

```
Supported strategies:
    - mode
    - constant
"""

cleaned = df.copy()

for column in columns:
    if column not in cleaned.columns:
        continue

    if strategy == "mode":
        mode = cleaned[column].mode(dropna=True)

        if not mode.empty:
            value = mode.iloc[0]
        else:
            value = fill_value

    elif strategy == "constant":
        value = fill_value

    else:
        raise ValueError(
            "strategy must be 'mode' or 'constant'"
        )

    cleaned[column] = cleaned[column].fillna(value)

return cleaned
```

# =============================================================================

# 07. REMOVE DUPLICATES

# =============================================================================

def remove_duplicate_rows(
df: pd.DataFrame,
subset: Optional[list[str]] = None,
) -> pd.DataFrame:
"""
Remove duplicate rows.

```
Parameters
----------
df:
    Input DataFrame.

subset:
    Optional list of columns used to determine duplicates.
"""

return df.drop_duplicates(
    subset=subset,
    keep="first",
).reset_index(drop=True)
```

# =============================================================================

# 08. SAFE NUMERIC CONVERSION

# =============================================================================

def convert_to_numeric(
df: pd.DataFrame,
columns: Iterable[str],
) -> pd.DataFrame:
"""
Convert selected columns to numeric values.

```
Invalid values become NaN instead of raising an exception.
"""

cleaned = df.copy()

for column in columns:
    if column not in cleaned.columns:
        continue

    cleaned[column] = pd.to_numeric(
        cleaned[column],
        errors="coerce",
    )

return cleaned
```

# =============================================================================

# 09. HANDLE IMPOSSIBLE VALUES

# =============================================================================

def replace_invalid_ages(
df: pd.DataFrame,
minimum: int = 0,
maximum: int = 120,
) -> pd.DataFrame:
"""
Replace impossible age values with NaN.

```
Note:
    Valid age ranges depend on the domain. This function is
    only an educational example.
"""

cleaned = df.copy()

invalid = (
    (cleaned["age"] < minimum)
    | (cleaned["age"] > maximum)
)

cleaned.loc[invalid, "age"] = np.nan

return cleaned
```

def replace_invalid_salary(
df: pd.DataFrame,
minimum: float = 0,
) -> pd.DataFrame:
"""
Replace negative salaries with NaN.

```
Domain-specific validation should be used in real projects.
"""

cleaned = df.copy()

cleaned.loc[
    cleaned["salary"] < minimum,
    "salary",
] = np.nan

return cleaned
```

# =============================================================================

# 10. OUTLIER DETECTION

# =============================================================================

def calculate_iqr_bounds(
series: pd.Series,
multiplier: float = 1.5,
) -> tuple[float, float]:
"""
Calculate IQR-based lower and upper bounds.

```
Formula:

    IQR = Q3 - Q1

    Lower = Q1 - multiplier * IQR
    Upper = Q3 + multiplier * IQR
"""

q1 = series.quantile(0.25)
q3 = series.quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - multiplier * iqr
upper_bound = q3 + multiplier * iqr

return lower_bound, upper_bound
```

def detect_outliers_iqr(
df: pd.DataFrame,
column: str,
multiplier: float = 1.5,
) -> pd.DataFrame:
"""
Return rows identified as IQR outliers.
"""

```
lower_bound, upper_bound = calculate_iqr_bounds(
    df[column].dropna(),
    multiplier=multiplier,
)

mask = (
    (df[column] < lower_bound)
    | (df[column] > upper_bound)
)

return df.loc[mask].copy()
```

def cap_outliers_iqr(
df: pd.DataFrame,
column: str,
multiplier: float = 1.5,
) -> pd.DataFrame:
"""
Cap values outside IQR bounds.

```
This is also called winsorization-like clipping.

Important:
    Outlier treatment should be chosen based on the problem.
    Do not automatically remove every statistical outlier.
"""

cleaned = df.copy()

lower_bound, upper_bound = calculate_iqr_bounds(
    cleaned[column].dropna(),
    multiplier=multiplier,
)

cleaned[column] = cleaned[column].clip(
    lower=lower_bound,
    upper=upper_bound,
)

return cleaned
```

# =============================================================================

# 11. DATE/TIME CLEANING

# =============================================================================

def clean_date_column(
df: pd.DataFrame,
column: str,
) -> pd.DataFrame:
"""
Convert a date column to pandas datetime.

```
Invalid values become NaT.
"""

cleaned = df.copy()

cleaned[column] = pd.to_datetime(
    cleaned[column],
    errors="coerce",
)

return cleaned
```

# =============================================================================

# 12. FEATURE ENGINEERING

# =============================================================================

def create_date_features(
df: pd.DataFrame,
date_column: str,
) -> pd.DataFrame:
"""
Create common date-based features.
"""

```
result = df.copy()

result["join_year"] = result[date_column].dt.year
result["join_month"] = result[date_column].dt.month
result["join_day"] = result[date_column].dt.day
result["join_day_of_week"] = (
    result[date_column].dt.dayofweek
)

return result
```

# =============================================================================

# 13. DATA VALIDATION

# =============================================================================

def validate_clean_dataset(df: pd.DataFrame) -> dict:
"""
Run basic validation checks.

```
Returns a dictionary containing validation results.
"""

return {
    "rows": len(df),
    "columns": len(df.columns),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "negative_ages": int(
        (df["age"] < 0).sum()
    ) if "age" in df.columns else None,
    "negative_salaries": int(
        (df["salary"] < 0).sum()
    ) if "salary" in df.columns else None,
}
```

def print_validation_report(df: pd.DataFrame) -> None:
"""
Print validation results in a readable format.
"""

```
report = validate_clean_dataset(df)

print("\n" + "=" * 80)
print("VALIDATION REPORT")
print("=" * 80)

for key, value in report.items():
    print(f"{key:20}: {value}")
```

# =============================================================================

# 14. REUSABLE CLEANING PIPELINE

# =============================================================================

def clean_customer_dataset(
df: pd.DataFrame,
) -> pd.DataFrame:
"""
Complete reusable cleaning workflow.

```
Pipeline:

    1. Copy input
    2. Clean column names
    3. Clean text
    4. Standardize categories
    5. Convert numeric columns
    6. Validate impossible values
    7. Handle missing numerical values
    8. Handle missing categorical values
    9. Remove duplicates
    10. Convert dates
    11. Add date features

Returns
-------
pd.DataFrame
    Cleaned dataset.
"""

cleaned = df.copy()

# -------------------------------------------------------------------------
# Step 1: Column names
# -------------------------------------------------------------------------

cleaned = clean_column_names(cleaned)

# -------------------------------------------------------------------------
# Step 2: Text cleaning
# -------------------------------------------------------------------------

cleaned = clean_text_columns(
    cleaned,
    columns=[
        "customer_name",
        "gender",
        "city",
        "department",
    ],
)

# -------------------------------------------------------------------------
# Step 3: Categorical standardization
# -------------------------------------------------------------------------

cleaned = standardize_gender(cleaned)
cleaned = standardize_city(cleaned)
cleaned = standardize_department(cleaned)

# -------------------------------------------------------------------------
# Step 4: Numeric conversion
# -------------------------------------------------------------------------

cleaned = convert_to_numeric(
    cleaned,
    columns=["age", "salary"],
)

# -------------------------------------------------------------------------
# Step 5: Invalid value handling
# -------------------------------------------------------------------------

cleaned = replace_invalid_ages(cleaned)
cleaned = replace_invalid_salary(cleaned)

# -------------------------------------------------------------------------
# Step 6: Date conversion
# -------------------------------------------------------------------------

cleaned = clean_date_column(
    cleaned,
    "join_date",
)

# -------------------------------------------------------------------------
# Step 7: Remove duplicates
# -------------------------------------------------------------------------

cleaned = remove_duplicate_rows(cleaned)

# -------------------------------------------------------------------------
# Step 8: Fill numerical missing values
# -------------------------------------------------------------------------

cleaned = fill_numeric_missing_values(
    cleaned,
    columns=["age", "salary"],
    strategy="median",
)

# -------------------------------------------------------------------------
# Step 9: Fill categorical missing values
# -------------------------------------------------------------------------

cleaned = fill_categorical_missing_values(
    cleaned,
    columns=[
        "customer_name",
        "gender",
        "city",
        "department",
    ],
    strategy="mode",
)

# -------------------------------------------------------------------------
# Step 10: Date features
# -------------------------------------------------------------------------

cleaned = create_date_features(
    cleaned,
    "join_date",
)

return cleaned
```

# =============================================================================

# 15. TRAIN/TEST LEAKAGE AWARENESS

# =============================================================================

def demonstrate_train_test_principle() -> None:
"""
Explain the correct order of train/test preprocessing.

```
Data-dependent preprocessing should generally be fitted using
training data only and then applied to validation/test data.

This function demonstrates the principle conceptually.
"""

print("\n" + "=" * 80)
print("TRAIN/TEST PREPROCESSING PRINCIPLE")
print("=" * 80)

print(
    """
```

Correct workflow:

```
Raw Dataset
     |
     v
Train/Test Split
     |
     +------------------+
     |                  |
     v                  v
  Training            Test
     |
     v
Learn preprocessing
parameters from train
     |
     +------------------+
     |                  |
     v                  v
Transform train     Transform test
     |
     v
Train model
```

Avoid:

```
Raw Dataset
     |
     v
Calculate statistics
using ALL data
     |
     v
Train/Test Split
```

The second approach can introduce information leakage.
"""
)

# =============================================================================

# 16. SCALING AND ENCODING NOTE

# =============================================================================

def explain_ml_preprocessing_boundary() -> None:
"""
Explain which transformations belong in Pandas and which are often
better handled by an ML preprocessing pipeline.
"""

```
print("\n" + "=" * 80)
print("PANDAS VS ML PREPROCESSING")
print("=" * 80)

print(
    """
```

Pandas is excellent for:

```
- Loading data
- Inspecting data
- Cleaning text
- Removing duplicates
- Fixing obvious invalid values
- Parsing dates
- Exploratory transformations
- Creating domain-specific features
```

Scikit-learn pipelines are often preferable for:

```
- StandardScaler
- MinMaxScaler
- OneHotEncoder
- Imputation learned from training data
- Model-specific preprocessing
```

This separation helps make ML workflows reproducible
and reduces the risk of train/test leakage.
"""
)

# =============================================================================

# 17. COMPLETE DATA CLEANING DEMONSTRATION

# =============================================================================

def run_complete_example() -> pd.DataFrame:
"""
Run the complete data-cleaning demonstration.
"""

```
print("\n" + "#" * 80)
print("# COMPLETE PANDAS DATA CLEANING WORKFLOW")
print("#" * 80)

# -------------------------------------------------------------------------
# Create messy dataset
# -------------------------------------------------------------------------

raw_df = create_sample_dataset()

print("\nRAW DATASET")
print("-" * 80)
print(raw_df)

# -------------------------------------------------------------------------
# Inspect
# -------------------------------------------------------------------------

inspect_dataset(raw_df)

# -------------------------------------------------------------------------
# Missing-value report
# -------------------------------------------------------------------------

print("\nMISSING VALUE REPORT")
print("-" * 80)

raw_report = missing_value_report(raw_df)

print(raw_report)

# -------------------------------------------------------------------------
# Clean dataset
# -------------------------------------------------------------------------

cleaned_df = clean_customer_dataset(raw_df)

# -------------------------------------------------------------------------
# Final output
# -------------------------------------------------------------------------

print("\nCLEANED DATASET")
print("-" * 80)
print(cleaned_df)

# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

print_validation_report(cleaned_df)

return cleaned_df
```

# =============================================================================

# 18. COMMON DATA CLEANING CHECKLIST

# =============================================================================

def print_cleaning_checklist() -> None:
"""
Print a practical data-cleaning checklist.
"""

```
print("\n" + "=" * 80)
print("DATA CLEANING CHECKLIST")
print("=" * 80)

checklist = [
    "Inspect dataset shape",
    "Inspect column names",
    "Inspect data types",
    "Check missing values",
    "Check duplicate records",
    "Check unique categorical values",
    "Normalize text formatting",
    "Convert data types",
    "Validate numerical ranges",
    "Validate categorical values",
    "Parse dates",
    "Investigate outliers",
    "Handle missing values",
    "Remove or correct duplicates",
    "Create useful features",
    "Check for data leakage",
    "Validate final dataset",
    "Document cleaning decisions",
]

for number, item in enumerate(checklist, start=1):
    print(f"{number:2}. {item}")
```

# =============================================================================

# 19. MAIN

# =============================================================================

def main() -> None:
"""
Entry point for the data-cleaning demonstration.
"""

```
cleaned_df = run_complete_example()

print_cleaning_checklist()

demonstrate_train_test_principle()

explain_ml_preprocessing_boundary()

print("\n" + "=" * 80)
print("FINAL DATA TYPES")
print("=" * 80)
print(cleaned_df.dtypes)

print("\n" + "=" * 80)
print("FINAL DATASET")
print("=" * 80)
print(cleaned_df.to_string(index=False))

print("\nData cleaning workflow completed successfully.")
```

if **name** == "**main**":
main()

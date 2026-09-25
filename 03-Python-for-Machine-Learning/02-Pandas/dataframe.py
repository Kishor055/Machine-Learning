"""
Pandas DataFrame for Machine Learning
=====================================

File:
03-Python-for-Machine-Learning/02-Pandas/dataframe.py

Description:
A complete beginner-to-advanced guide to Pandas DataFrames.

A DataFrame is Pandas' primary two-dimensional data structure. It is
essential for working with structured datasets in Data Science and
Machine Learning.

Topics Covered:
1. Creating DataFrames
2. DataFrame structure
3. Index and columns
4. Accessing rows and columns
5. loc and iloc
6. Adding and removing columns
7. Adding and removing rows
8. Updating values
9. Filtering
10. Sorting
11. Renaming
12. Data types
13. Missing values
14. Duplicate values
15. Unique values
16. GroupBy and aggregation
17. Apply and Map
18. Merge and Join
19. Concatenation
20. Pivot tables
21. Sampling
22. Statistical operations
23. Correlation
24. Feature engineering
25. NumPy integration
26. ML dataset preparation

Requirements:
Python 3.9+
pandas
numpy

Install:
pip install pandas numpy

Run:
python dataframe.py

Author:
Kishor Patil
"""

from **future** import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

# =============================================================================

# 01. CREATE A BASIC DATAFRAME

# =============================================================================

def create_basic_dataframe() -> pd.DataFrame:
"""
Create a basic DataFrame from a dictionary.
"""

```
data = {
    "name": ["Alice", "Bob", "Charlie", "David"],
    "age": [22, 25, 28, 24],
    "score": [85, 91, 88, 76],
}

return pd.DataFrame(data)
```

def demonstrate_basic_dataframe() -> None:
"""
Demonstrate basic DataFrame creation.
"""

```
print("\n" + "=" * 80)
print("01. BASIC DATAFRAME")
print("=" * 80)

df = create_basic_dataframe()

print(df)

print("\nType:")
print(type(df))

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nIndex:")
print(df.index)
```

# =============================================================================

# 02. CREATE DATAFRAME FROM DIFFERENT SOURCES

# =============================================================================

def create_from_list_of_dicts() -> pd.DataFrame:
"""
Create a DataFrame from a list of dictionaries.
"""

```
records = [
    {"name": "Alice", "age": 22, "score": 85},
    {"name": "Bob", "age": 25, "score": 91},
    {"name": "Charlie", "age": 28, "score": 88},
]

return pd.DataFrame(records)
```

def create_from_numpy() -> pd.DataFrame:
"""
Create a DataFrame from a NumPy array.
"""

```
array = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90],
])

return pd.DataFrame(
    array,
    columns=["A", "B", "C"],
)
```

def create_from_records() -> pd.DataFrame:
"""
Create a DataFrame using structured records.
"""

```
records = [
    ("Alice", 22, 85),
    ("Bob", 25, 91),
    ("Charlie", 28, 88),
]

return pd.DataFrame(
    records,
    columns=["name", "age", "score"],
)
```

# =============================================================================

# 03. DATAFRAME STRUCTURE

# =============================================================================

def demonstrate_structure(df: pd.DataFrame) -> None:
"""
Explore DataFrame metadata and structure.
"""

```
print("\n" + "=" * 80)
print("03. DATAFRAME STRUCTURE")
print("=" * 80)

print("\nDataFrame:")
print(df)

print("\nShape:")
print(df.shape)

print("\nNumber of rows:")
print(df.shape[0])

print("\nNumber of columns:")
print(df.shape[1])

print("\nColumns:")
print(df.columns)

print("\nIndex:")
print(df.index)

print("\nData types:")
print(df.dtypes)

print("\nMemory usage:")
print(df.memory_usage(deep=True))
```

# =============================================================================

# 04. DATAFRAME INSPECTION

# =============================================================================

def demonstrate_inspection(df: pd.DataFrame) -> None:
"""
Demonstrate common DataFrame inspection methods.
"""

```
print("\n" + "=" * 80)
print("04. DATAFRAME INSPECTION")
print("=" * 80)

print("\nFirst rows:")
print(df.head())

print("\nLast rows:")
print(df.tail())

print("\nFirst 2 rows:")
print(df.head(2))

print("\nLast 2 rows:")
print(df.tail(2))

print("\nInfo:")
df.info()

print("\nDescriptive statistics:")
print(df.describe())
```

# =============================================================================

# 05. COLUMN SELECTION

# =============================================================================

def demonstrate_column_selection(df: pd.DataFrame) -> None:
"""
Demonstrate different column-selection techniques.
"""

```
print("\n" + "=" * 80)
print("05. COLUMN SELECTION")
print("=" * 80)

print("\nSingle column:")
print(df["name"])

print("\nMultiple columns:")
print(df[["name", "score"]])

print("\nColumn using attribute notation:")
print(df.score)
```

# =============================================================================

# 06. ROW SELECTION

# =============================================================================

def demonstrate_row_selection(df: pd.DataFrame) -> None:
"""
Demonstrate row selection with loc and iloc.
"""

```
print("\n" + "=" * 80)
print("06. ROW SELECTION")
print("=" * 80)

print("\nFirst row using iloc:")
print(df.iloc[0])

print("\nFirst two rows:")
print(df.iloc[0:2])

print("\nLast row:")
print(df.iloc[-1])

print("\nFirst row using loc:")
print(df.loc[0])

print("\nRows 1 through 2:")
print(df.loc[1:2])
```

# =============================================================================

# 07. ROW + COLUMN SELECTION

# =============================================================================

def demonstrate_loc_iloc(df: pd.DataFrame) -> None:
"""
Demonstrate two-dimensional selection.
"""

```
print("\n" + "=" * 80)
print("07. LOC AND ILOC")
print("=" * 80)

print("\nloc - rows and named columns:")
print(
    df.loc[
        0:2,
        ["name", "age"],
    ]
)

print("\niloc - rows and positional columns:")
print(
    df.iloc[
        0:3,
        0:2,
    ]
)

print("\nSingle value with loc:")
print(df.loc[0, "name"])

print("\nSingle value with iloc:")
print(df.iloc[0, 0])
```

# =============================================================================

# 08. ADDING COLUMNS

# =============================================================================

def demonstrate_adding_columns(df: pd.DataFrame) -> pd.DataFrame:
"""
Demonstrate multiple ways to add DataFrame columns.
"""

```
print("\n" + "=" * 80)
print("08. ADDING COLUMNS")
print("=" * 80)

result = df.copy()

# Direct assignment
result["passed"] = result["score"] >= 50

# Arithmetic feature
result["score_percentage"] = result["score"]

# assign()
result = result.assign(
    age_squared=result["age"] ** 2
)

print(result)

return result
```

# =============================================================================

# 09. REMOVING COLUMNS

# =============================================================================

def demonstrate_removing_columns(df: pd.DataFrame) -> pd.DataFrame:
"""
Demonstrate column removal.
"""

```
print("\n" + "=" * 80)
print("09. REMOVING COLUMNS")
print("=" * 80)

result = df.copy()

result["temporary"] = 100

print("\nBefore:")
print(result)

result = result.drop(
    columns=["temporary"]
)

print("\nAfter:")
print(result)

return result
```

# =============================================================================

# 10. ADDING ROWS

# =============================================================================

def demonstrate_adding_rows(df: pd.DataFrame) -> pd.DataFrame:
"""
Demonstrate adding rows with pd.concat().
"""

```
print("\n" + "=" * 80)
print("10. ADDING ROWS")
print("=" * 80)

new_row = pd.DataFrame([
    {
        "name": "Eva",
        "age": 26,
        "score": 93,
    }
])

result = pd.concat(
    [df, new_row],
    ignore_index=True,
)

print(result)

return result
```

# =============================================================================

# 11. REMOVING ROWS

# =============================================================================

def demonstrate_removing_rows(df: pd.DataFrame) -> pd.DataFrame:
"""
Demonstrate row removal.
"""

```
print("\n" + "=" * 80)
print("11. REMOVING ROWS")
print("=" * 80)

result = df.drop(index=0)

print(result)

return result
```

# =============================================================================

# 12. UPDATING VALUES

# =============================================================================

def demonstrate_updating_values(df: pd.DataFrame) -> pd.DataFrame:
"""
Demonstrate updating individual and multiple values.
"""

```
print("\n" + "=" * 80)
print("12. UPDATING VALUES")
print("=" * 80)

result = df.copy()

# Update one cell
result.loc[0, "score"] = 90

# Update multiple values using a condition
result.loc[
    result["score"] < 80,
    "score"
] = 80

print(result)

return result
```

# =============================================================================

# 13. FILTERING

# =============================================================================

def demonstrate_filtering(df: pd.DataFrame) -> None:
"""
Demonstrate boolean filtering.
"""

```
print("\n" + "=" * 80)
print("13. FILTERING")
print("=" * 80)

print("\nAge > 24:")
print(df[df["age"] > 24])

print("\nScore >= 85:")
print(df[df["score"] >= 85])

print("\nAge > 24 AND score > 85:")
print(
    df[
        (df["age"] > 24)
        & (df["score"] > 85)
    ]
)

print("\nAge > 24 OR score > 90:")
print(
    df[
        (df["age"] > 24)
        | (df["score"] > 90)
    ]
)
```

# =============================================================================

# 14. QUERY

# =============================================================================

def demonstrate_query(df: pd.DataFrame) -> None:
"""
Demonstrate DataFrame.query().
"""

```
print("\n" + "=" * 80)
print("14. QUERY")
print("=" * 80)

result = df.query(
    "age >= 25 and score >= 85"
)

print(result)
```

# =============================================================================

# 15. SORTING

# =============================================================================

def demonstrate_sorting(df: pd.DataFrame) -> None:
"""
Demonstrate sorting DataFrames.
"""

```
print("\n" + "=" * 80)
print("15. SORTING")
print("=" * 80)

print("\nSort by age:")
print(
    df.sort_values("age")
)

print("\nSort by score descending:")
print(
    df.sort_values(
        "score",
        ascending=False,
    )
)

print("\nSort by multiple columns:")
print(
    df.sort_values(
        ["age", "score"],
        ascending=[True, False],
    )
)
```

# =============================================================================

# 16. RENAMING

# =============================================================================

def demonstrate_renaming(df: pd.DataFrame) -> None:
"""
Demonstrate column renaming.
"""

```
print("\n" + "=" * 80)
print("16. RENAMING")
print("=" * 80)

result = df.rename(
    columns={
        "name": "student_name",
        "score": "exam_score",
    }
)

print(result)
```

# =============================================================================

# 17. DATA TYPES

# =============================================================================

def demonstrate_data_types(df: pd.DataFrame) -> None:
"""
Demonstrate DataFrame data types.
"""

```
print("\n" + "=" * 80)
print("17. DATA TYPES")
print("=" * 80)

print(df.dtypes)

print("\nConvert score to float:")
result = df.copy()
result["score"] = result["score"].astype(float)

print(result.dtypes)
```

# =============================================================================

# 18. UNIQUE VALUES

# =============================================================================

def demonstrate_unique_values() -> None:
"""
Demonstrate unique-value operations.
"""

```
print("\n" + "=" * 80)
print("18. UNIQUE VALUES")
print("=" * 80)

df = pd.DataFrame({
    "department": [
        "IT",
        "HR",
        "IT",
        "Finance",
        "HR",
        "IT",
    ]
})

print("\nUnique values:")
print(df["department"].unique())

print("\nNumber of unique values:")
print(df["department"].nunique())

print("\nValue counts:")
print(df["department"].value_counts())
```

# =============================================================================

# 19. DUPLICATES

# =============================================================================

def demonstrate_duplicates() -> None:
"""
Demonstrate duplicate detection and removal.
"""

```
print("\n" + "=" * 80)
print("19. DUPLICATES")
print("=" * 80)

df = pd.DataFrame({
    "name": [
        "Alice",
        "Bob",
        "Alice",
        "Charlie",
        "Bob",
    ],
    "score": [
        85,
        90,
        85,
        88,
        90,
    ],
})

print("\nOriginal:")
print(df)

print("\nDuplicate mask:")
print(df.duplicated())

print("\nDuplicate count:")
print(df.duplicated().sum())

print("\nAfter removing duplicates:")
print(
    df.drop_duplicates(
        ignore_index=True
    )
)
```

# =============================================================================

# 20. MISSING VALUES

# =============================================================================

def demonstrate_missing_values() -> None:
"""
Demonstrate missing-value detection and handling.
"""

```
print("\n" + "=" * 80)
print("20. MISSING VALUES")
print("=" * 80)

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "David"],
    "age": [22, np.nan, 28, np.nan],
    "score": [85, 90, np.nan, 88],
})

print("\nDataset:")
print(df)

print("\nMissing values:")
print(df.isna())

print("\nMissing-value count:")
print(df.isna().sum())

print("\nFilled age using median:")
result = df.copy()

result["age"] = result["age"].fillna(
    result["age"].median()
)

print(result)
```

# =============================================================================

# 21. APPLY

# =============================================================================

def demonstrate_apply(df: pd.DataFrame) -> None:
"""
Demonstrate apply() for custom transformations.
"""

```
print("\n" + "=" * 80)
print("21. APPLY")
print("=" * 80)

result = df.copy()

result["grade"] = result["score"].apply(
    lambda score: (
        "A" if score >= 90
        else "B" if score >= 80
        else "C"
    )
)

print(result)
```

# =============================================================================

# 22. MAP

# =============================================================================

def demonstrate_map() -> None:
"""
Demonstrate Series.map().
"""

```
print("\n" + "=" * 80)
print("22. MAP")
print("=" * 80)

df = pd.DataFrame({
    "gender": [
        "Male",
        "Female",
        "Female",
        "Male",
    ]
})

mapping = {
    "Male": 0,
    "Female": 1,
}

df["gender_code"] = df["gender"].map(mapping)

print(df)
```

# =============================================================================

# 23. GROUPBY

# =============================================================================

def create_employee_dataframe() -> pd.DataFrame:
"""
Create an employee dataset for GroupBy demonstrations.
"""

```
return pd.DataFrame({
    "name": [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eva",
        "Frank",
    ],
    "department": [
        "IT",
        "HR",
        "IT",
        "Finance",
        "HR",
        "IT",
    ],
    "salary": [
        70000,
        60000,
        80000,
        65000,
        62000,
        75000,
    ],
    "experience": [
        2,
        3,
        5,
        4,
        3,
        6,
    ],
})
```

def demonstrate_groupby() -> None:
"""
Demonstrate grouping and aggregation.
"""

```
print("\n" + "=" * 80)
print("23. GROUPBY")
print("=" * 80)

df = create_employee_dataframe()

print("\nAverage salary by department:")
print(
    df.groupby("department")["salary"].mean()
)

print("\nMultiple aggregations:")
print(
    df.groupby("department")["salary"].agg(
        ["mean", "median", "min", "max", "count"]
    )
)

print("\nAverage salary and experience:")
print(
    df.groupby("department")[
        ["salary", "experience"]
    ].mean()
)
```

# =============================================================================

# 24. AGGREGATION

# =============================================================================

def demonstrate_aggregation() -> None:
"""
Demonstrate common DataFrame aggregation methods.
"""

```
print("\n" + "=" * 80)
print("24. AGGREGATION")
print("=" * 80)

df = create_employee_dataframe()

numeric = df[
    ["salary", "experience"]
]

print("\nMean:")
print(numeric.mean())

print("\nMedian:")
print(numeric.median())

print("\nMinimum:")
print(numeric.min())

print("\nMaximum:")
print(numeric.max())

print("\nStandard deviation:")
print(numeric.std())

print("\nVariance:")
print(numeric.var())
```

# =============================================================================

# 25. MERGE

# =============================================================================

def demonstrate_merge() -> None:
"""
Demonstrate merging DataFrames using a common key.
"""

```
print("\n" + "=" * 80)
print("25. MERGE")
print("=" * 80)

students = pd.DataFrame({
    "student_id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
})

scores = pd.DataFrame({
    "student_id": [1, 2, 3],
    "score": [85, 91, 88],
})

result = pd.merge(
    students,
    scores,
    on="student_id",
    how="inner",
)

print(result)
```

# =============================================================================

# 26. CONCATENATION

# =============================================================================

def demonstrate_concat() -> None:
"""
Demonstrate vertical and horizontal concatenation.
"""

```
print("\n" + "=" * 80)
print("26. CONCATENATION")
print("=" * 80)

first = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "score": [85, 90],
})

second = pd.DataFrame({
    "name": ["Charlie", "David"],
    "score": [88, 92],
})

vertical = pd.concat(
    [first, second],
    ignore_index=True,
)

print("\nVertical concatenation:")
print(vertical)

extra = pd.DataFrame({
    "passed": [True, True, True, True]
})

horizontal = pd.concat(
    [vertical, extra],
    axis=1,
)

print("\nHorizontal concatenation:")
print(horizontal)
```

# =============================================================================

# 27. JOIN

# =============================================================================

def demonstrate_join() -> None:
"""
Demonstrate DataFrame.join().
"""

```
print("\n" + "=" * 80)
print("27. JOIN")
print("=" * 80)

left = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
    },
    index=[1, 2, 3],
)

right = pd.DataFrame(
    {
        "score": [85, 90, 88],
    },
    index=[1, 2, 3],
)

result = left.join(right)

print(result)
```

# =============================================================================

# 28. PIVOT TABLE

# =============================================================================

def demonstrate_pivot_table() -> None:
"""
Demonstrate pivot tables.
"""

```
print("\n" + "=" * 80)
print("28. PIVOT TABLE")
print("=" * 80)

df = pd.DataFrame({
    "department": [
        "IT",
        "IT",
        "HR",
        "HR",
        "Finance",
        "Finance",
    ],
    "gender": [
        "Male",
        "Female",
        "Male",
        "Female",
        "Male",
        "Female",
    ],
    "salary": [
        70000,
        75000,
        60000,
        65000,
        62000,
        68000,
    ],
})

pivot = pd.pivot_table(
    df,
    values="salary",
    index="department",
    columns="gender",
    aggfunc="mean",
)

print(pivot)
```

# =============================================================================

# 29. STATISTICS

# =============================================================================

def demonstrate_statistics(df: pd.DataFrame) -> None:
"""
Demonstrate statistical operations.
"""

```
print("\n" + "=" * 80)
print("29. STATISTICS")
print("=" * 80)

print("\nMean:")
print(df.mean(numeric_only=True))

print("\nMedian:")
print(df.median(numeric_only=True))

print("\nStandard deviation:")
print(df.std(numeric_only=True))

print("\nVariance:")
print(df.var(numeric_only=True))

print("\nQuantiles:")
print(
    df["score"].quantile(
        [0.25, 0.50, 0.75]
    )
)
```

# =============================================================================

# 30. CORRELATION

# =============================================================================

def demonstrate_correlation() -> None:
"""
Demonstrate correlation between numerical features.
"""

```
print("\n" + "=" * 80)
print("30. CORRELATION")
print("=" * 80)

df = pd.DataFrame({
    "age": [20, 25, 30, 35, 40],
    "experience": [1, 3, 5, 8, 10],
    "salary": [30000, 40000, 50000, 65000, 80000],
})

print(df.corr())
```

# =============================================================================

# 31. SAMPLING

# =============================================================================

def demonstrate_sampling(df: pd.DataFrame) -> None:
"""
Demonstrate random sampling.
"""

```
print("\n" + "=" * 80)
print("31. SAMPLING")
print("=" * 80)

print("\nRandom sample:")
print(
    df.sample(
        n=2,
        random_state=42,
    )
)

print("\n50% sample:")
print(
    df.sample(
        frac=0.5,
        random_state=42,
    )
)
```

# =============================================================================

# 32. RESET AND SET INDEX

# =============================================================================

def demonstrate_index_operations(df: pd.DataFrame) -> None:
"""
Demonstrate setting and resetting indexes.
"""

```
print("\n" + "=" * 80)
print("32. INDEX OPERATIONS")
print("=" * 80)

result = df.set_index("name")

print("\nSet name as index:")
print(result)

print("\nReset index:")
print(result.reset_index())
```

# =============================================================================

# 33. COPY VS VIEW

# =============================================================================

def demonstrate_copy() -> None:
"""
Demonstrate why explicit copies can be useful.
"""

```
print("\n" + "=" * 80)
print("33. COPY VS ORIGINAL")
print("=" * 80)

original = create_basic_dataframe()

subset = original[
    ["name", "score"]
].copy()

subset["score"] = subset["score"] + 5

print("\nOriginal:")
print(original)

print("\nCopied subset:")
print(subset)

print(
    "\nUsing .copy() makes the intention to create an "
    "independent object explicit."
)
```

# =============================================================================

# 34. FEATURE ENGINEERING

# =============================================================================

def demonstrate_feature_engineering() -> pd.DataFrame:
"""
Demonstrate simple ML-oriented feature engineering.
"""

```
print("\n" + "=" * 80)
print("34. FEATURE ENGINEERING")
print("=" * 80)

df = pd.DataFrame({
    "age": [22, 25, 30, 35],
    "income": [40000, 50000, 65000, 80000],
    "experience": [1, 3, 6, 10],
})

df["income_per_year_experience"] = (
    df["income"]
    / df["experience"]
)

df["age_squared"] = df["age"] ** 2

df["is_senior"] = df["age"] >= 30

print(df)

return df
```

# =============================================================================

# 35. ONE-HOT ENCODING

# =============================================================================

def demonstrate_one_hot_encoding() -> None:
"""
Demonstrate Pandas get_dummies().

```
For production ML pipelines, sklearn's OneHotEncoder is often preferable
because it integrates directly with train/test preprocessing pipelines.
"""

print("\n" + "=" * 80)
print("35. ONE-HOT ENCODING")
print("=" * 80)

df = pd.DataFrame({
    "city": [
        "Mumbai",
        "Pune",
        "Nashik",
        "Mumbai",
    ]
})

encoded = pd.get_dummies(
    df,
    columns=["city"],
    dtype=int,
)

print(encoded)
```

# =============================================================================

# 36. CONVERT DATAFRAME TO NUMPY

# =============================================================================

def demonstrate_numpy_conversion(df: pd.DataFrame) -> None:
"""
Convert a DataFrame to a NumPy array.
"""

```
print("\n" + "=" * 80)
print("36. DATAFRAME TO NUMPY")
print("=" * 80)

numeric = df[
    ["age", "score"]
]

array = numeric.to_numpy()

print(array)

print("\nArray type:")
print(type(array))

print("\nArray shape:")
print(array.shape)
```

# =============================================================================

# 37. ML DATASET PREPARATION

# =============================================================================

def prepare_ml_dataset() -> tuple[pd.DataFrame, pd.Series]:
"""
Create and prepare a simple ML-style dataset.

```
Returns:
    X: Feature DataFrame
    y: Target Series
"""

df = pd.DataFrame({
    "age": [22, 25, 28, 31, 35, 40],
    "experience": [1, 3, 5, 7, 10, 15],
    "income": [
        35000,
        45000,
        55000,
        65000,
        80000,
        100000,
    ],
    "purchased": [
        0,
        0,
        1,
        1,
        1,
        1,
    ],
})

X = df.drop(
    columns=["purchased"]
)

y = df["purchased"]

return X, y
```

def demonstrate_ml_dataset_preparation() -> None:
"""
Demonstrate separation of features and target.
"""

```
print("\n" + "=" * 80)
print("37. ML DATASET PREPARATION")
print("=" * 80)

X, y = prepare_ml_dataset()

print("\nFeatures X:")
print(X)

print("\nTarget y:")
print(y)

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)
```

# =============================================================================

# 38. DATAFRAME VALIDATION

# =============================================================================

def validate_dataframe(df: pd.DataFrame) -> dict:
"""
Return basic DataFrame validation information.
"""

```
return {
    "rows": len(df),
    "columns": len(df.columns),
    "missing_values": int(
        df.isna().sum().sum()
    ),
    "duplicate_rows": int(
        df.duplicated().sum()
    ),
    "column_names": df.columns.tolist(),
    "data_types": df.dtypes.astype(str).to_dict(),
}
```

def demonstrate_validation(df: pd.DataFrame) -> None:
"""
Display DataFrame validation information.
"""

```
print("\n" + "=" * 80)
print("38. DATAFRAME VALIDATION")
print("=" * 80)

report = validate_dataframe(df)

for key, value in report.items():
    print(f"\n{key}:")
    print(value)
```

# =============================================================================

# 39. COMPLETE DATAFRAME WORKFLOW

# =============================================================================

def complete_dataframe_workflow() -> pd.DataFrame:
"""
Demonstrate a realistic DataFrame workflow.
"""

```
print("\n" + "#" * 80)
print("# COMPLETE DATAFRAME WORKFLOW")
print("#" * 80)

df = pd.DataFrame({
    "name": [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eva",
    ],
    "age": [
        22,
        25,
        28,
        31,
        26,
    ],
    "department": [
        "IT",
        "HR",
        "IT",
        "Finance",
        "HR",
    ],
    "salary": [
        50000,
        60000,
        70000,
        65000,
        62000,
    ],
})

print("\n1. Original dataset:")
print(df)

# Add derived feature
df["salary_k"] = df["salary"] / 1000

print("\n2. Feature engineering:")
print(df)

# Filter
high_salary = df[
    df["salary"] >= 60000
]

print("\n3. High-salary employees:")
print(high_salary)

# GroupBy
department_summary = (
    df.groupby("department")
    .agg(
        average_salary=("salary", "mean"),
        employee_count=("name", "count"),
    )
)

print("\n4. Department summary:")
print(department_summary)

# Sort
df = df.sort_values(
    "salary",
    ascending=False,
)

print("\n5. Sorted dataset:")
print(df)

return df
```

# =============================================================================

# 40. MAIN

# =============================================================================

def main() -> None:
"""
Run the complete DataFrame tutorial.
"""

```
# -------------------------------------------------------------------------
# Basic DataFrame
# -------------------------------------------------------------------------

df = create_basic_dataframe()

demonstrate_basic_dataframe()

# -------------------------------------------------------------------------
# DataFrame creation methods
# -------------------------------------------------------------------------

print("\n" + "=" * 80)
print("02. DATAFRAME CREATION METHODS")
print("=" * 80)

print("\nFrom list of dictionaries:")
print(create_from_list_of_dicts())

print("\nFrom NumPy array:")
print(create_from_numpy())

print("\nFrom records:")
print(create_from_records())

# -------------------------------------------------------------------------
# Structure and inspection
# -------------------------------------------------------------------------

demonstrate_structure(df)
demonstrate_inspection(df)

# -------------------------------------------------------------------------
# Selection
# -------------------------------------------------------------------------

demonstrate_column_selection(df)
demonstrate_row_selection(df)
demonstrate_loc_iloc(df)

# -------------------------------------------------------------------------
# Modification
# -------------------------------------------------------------------------

modified_df = demonstrate_adding_columns(df)
demonstrate_removing_columns(modified_df)
demonstrate_adding_rows(df)
demonstrate_removing_rows(df)
demonstrate_updating_values(df)

# -------------------------------------------------------------------------
# Filtering and sorting
# -------------------------------------------------------------------------

demonstrate_filtering(df)
demonstrate_query(df)
demonstrate_sorting(df)

# -------------------------------------------------------------------------
# Metadata and cleaning
# -------------------------------------------------------------------------

demonstrate_renaming(df)
demonstrate_data_types(df)
demonstrate_unique_values()
demonstrate_duplicates()
demonstrate_missing_values()

# -------------------------------------------------------------------------
# Transformations
# -------------------------------------------------------------------------

demonstrate_apply(df)
demonstrate_map()

# -------------------------------------------------------------------------
# Grouping and aggregation
# -------------------------------------------------------------------------

demonstrate_groupby()
demonstrate_aggregation()

# -------------------------------------------------------------------------
# Combining DataFrames
# -------------------------------------------------------------------------

demonstrate_merge()
demonstrate_concat()
demonstrate_join()

# -------------------------------------------------------------------------
# Advanced operations
# -------------------------------------------------------------------------

demonstrate_pivot_table()
demonstrate_statistics(df)
demonstrate_correlation()
demonstrate_sampling(df)
demonstrate_index_operations(df)
demonstrate_copy()

# -------------------------------------------------------------------------
# Machine Learning operations
# -------------------------------------------------------------------------

demonstrate_feature_engineering()
demonstrate_one_hot_encoding()
demonstrate_numpy_conversion(df)
demonstrate_ml_dataset_preparation()

# -------------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------------

demonstrate_validation(df)

# -------------------------------------------------------------------------
# Complete workflow
# -------------------------------------------------------------------------

final_df = complete_dataframe_workflow()

print("\n" + "=" * 80)
print("FINAL DATAFRAME")
print("=" * 80)
print(final_df)

print("\n" + "=" * 80)
print("TUTORIAL COMPLETED")
print("=" * 80)

print(
    """
```

You have covered:

```
✓ DataFrame creation
✓ Structure and metadata
✓ Inspection
✓ Column selection
✓ Row selection
✓ loc / iloc
✓ Adding columns
✓ Removing columns
✓ Adding rows
✓ Removing rows
✓ Updating values
✓ Filtering
✓ Querying
✓ Sorting
✓ Renaming
✓ Data types
✓ Unique values
✓ Duplicates
✓ Missing values
✓ Apply / Map
✓ GroupBy
✓ Aggregation
✓ Merge / Join
✓ Concatenation
✓ Pivot tables
✓ Statistics
✓ Correlation
✓ Sampling
✓ Index operations
✓ Feature engineering
✓ One-hot encoding
✓ NumPy integration
✓ ML dataset preparation
✓ Data validation
```

"""
)

if **name** == "**main**":
main()

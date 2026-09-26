"""
Pandas Join Tutorial
====================

File:
03-Python-for-Machine-Learning/02-Pandas/join.py

Description:
A complete beginner-to-advanced guide to joining DataFrames with Pandas.

Topics Covered:
1. What is a join?
2. Creating sample DataFrames
3. Basic DataFrame.join()
4. Joining on index
5. Joining on a column
6. Inner join
7. Left join
8. Right join
9. Outer join
10. Join with suffixes
11. Join multiple DataFrames
12. Joining DataFrames with different indexes
13. Joining after set_index()
14. Join vs merge
15. Joining with MultiIndex
16. Joining with duplicate keys
17. Handling missing values after joins
18. Validating join results
19. Avoiding accidental many-to-many joins
20. Joining for Machine Learning datasets
21. Preventing data leakage
22. Reusable join functions
23. Complete real-world workflow
24. Common mistakes
25. Join cheat sheet

Requirements:
pip install pandas numpy

Run:
python join.py
"""

# ============================================================

# 1. IMPORT LIBRARIES

# ============================================================

import numpy as np
import pandas as pd

# ============================================================

# 2. DISPLAY SETTINGS

# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

def section(title: str) -> None:
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

# ============================================================

# 3. CREATE SAMPLE DATA

# ============================================================

def create_sample_data():
"""
Create sample DataFrames used throughout the tutorial.
"""

```
employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103, 104, 105],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Ethan"],
        "department_id": [10, 20, 10, 30, 20],
    }
)

departments = pd.DataFrame(
    {
        "department_id": [10, 20, 30, 40],
        "department": [
            "Data Science",
            "Engineering",
            "Marketing",
            "Finance",
        ],
        "location": [
            "Mumbai",
            "Pune",
            "Bangalore",
            "Delhi",
        ],
    }
)

salaries = pd.DataFrame(
    {
        "employee_id": [101, 102, 103, 104],
        "salary": [70000, 85000, 72000, 65000],
    }
)

return employees, departments, salaries
```

# ============================================================

# 4. BASIC DATAFRAME.JOIN()

# ============================================================

def basic_join():
"""
Demonstrate the simplest DataFrame.join() operation.

```
DataFrame.join() primarily joins using the index.
"""

section("4. BASIC DATAFRAME.JOIN()")

left = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
    },
    index=[101, 102, 103],
)

right = pd.DataFrame(
    {
        "salary": [70000, 85000, 72000],
    },
    index=[101, 102, 103],
)

print("Left DataFrame:")
print(left)

print("\nRight DataFrame:")
print(right)

result = left.join(right)

print("\nJoined DataFrame:")
print(result)
```

# ============================================================

# 5. JOIN USING INDEX

# ============================================================

def join_using_index():
"""
Join two DataFrames using their indexes.
"""

```
section("5. JOIN USING INDEX")

employees = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
        "department": ["Data Science", "Engineering", "Marketing"],
    },
    index=[101, 102, 103],
)

salaries = pd.DataFrame(
    {
        "salary": [70000, 85000, 65000],
    },
    index=[101, 102, 103],
)

result = employees.join(salaries)

print(result)
```

# ============================================================

# 6. LEFT JOIN

# ============================================================

def left_join():
"""
A left join keeps every row from the left DataFrame.

```
Missing matches from the right DataFrame become NaN.
"""

section("6. LEFT JOIN")

employees = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
    },
    index=[101, 102, 103, 104],
)

salaries = pd.DataFrame(
    {
        "salary": [70000, 85000, 72000],
    },
    index=[101, 102, 103],
)

result = employees.join(
    salaries,
    how="left",
)

print(result)
```

# ============================================================

# 7. INNER JOIN

# ============================================================

def inner_join():
"""
An inner join keeps only matching index values.
"""

```
section("7. INNER JOIN")

employees = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
    },
    index=[101, 102, 103, 104],
)

salaries = pd.DataFrame(
    {
        "salary": [70000, 85000, 72000],
    },
    index=[101, 102, 103],
)

result = employees.join(
    salaries,
    how="inner",
)

print(result)
```

# ============================================================

# 8. RIGHT JOIN

# ============================================================

def right_join():
"""
A right join keeps every row from the right DataFrame.
"""

```
section("8. RIGHT JOIN")

employees = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
    },
    index=[101, 102, 103],
)

salaries = pd.DataFrame(
    {
        "salary": [70000, 85000, 72000, 65000],
    },
    index=[101, 102, 103, 104],
)

result = employees.join(
    salaries,
    how="right",
)

print(result)
```

# ============================================================

# 9. OUTER JOIN

# ============================================================

def outer_join():
"""
An outer join keeps all index values from both DataFrames.
"""

```
section("9. OUTER JOIN")

left = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
    },
    index=[101, 102, 103],
)

right = pd.DataFrame(
    {
        "salary": [70000, 85000, 90000],
    },
    index=[102, 103, 104],
)

result = left.join(
    right,
    how="outer",
)

print(result)
```

# ============================================================

# 10. JOIN WITH MISSING VALUES

# ============================================================

def join_with_missing_values():
"""
Demonstrate missing values created by joins.
"""

```
section("10. JOIN WITH MISSING VALUES")

employees = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
    },
    index=[101, 102, 103, 104],
)

salaries = pd.DataFrame(
    {
        "salary": [70000, 85000],
    },
    index=[101, 102],
)

result = employees.join(salaries, how="left")

print(result)

print("\nMissing values:")
print(result.isna())

print("\nMissing-value count:")
print(result.isna().sum())
```

# ============================================================

# 11. FILL MISSING VALUES AFTER JOIN

# ============================================================

def fill_missing_after_join():
"""
Fill missing values created by an outer/left join.
"""

```
section("11. FILL MISSING VALUES AFTER JOIN")

employees = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
    },
    index=[101, 102, 103, 104],
)

salaries = pd.DataFrame(
    {
        "salary": [70000, 85000],
    },
    index=[101, 102],
)

result = employees.join(salaries)

result["salary"] = result["salary"].fillna(0)

print(result)
```

# ============================================================

# 12. JOIN ON A COLUMN

# ============================================================

def join_on_column():
"""
DataFrame.join() normally works with indexes.

```
To join using a column, use `on=` on the left DataFrame
and the matching key must be the index of the right DataFrame.
"""

section("12. JOIN ON A COLUMN")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103, 104],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "department_id": [10, 20, 10, 30],
    }
)

departments = pd.DataFrame(
    {
        "department_id": [10, 20, 30],
        "department": [
            "Data Science",
            "Engineering",
            "Marketing",
        ],
    }
).set_index("department_id")

result = employees.join(
    departments,
    on="department_id",
    how="left",
)

print(result)
```

# ============================================================

# 13. SET INDEX BEFORE JOINING

# ============================================================

def set_index_before_join():
"""
A common workflow is:

```
    1. Select the join key.
    2. Set it as the index.
    3. Perform the join.
    4. Reset the index if necessary.
"""

section("13. SET INDEX BEFORE JOINING")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

salaries = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "salary": [70000, 85000, 72000],
    }
)

employees_indexed = employees.set_index("employee_id")
salaries_indexed = salaries.set_index("employee_id")

result = employees_indexed.join(salaries_indexed)

print(result)

print("\nAfter reset_index():")
print(result.reset_index())
```

# ============================================================

# 14. JOIN WITH SUFFIXES

# ============================================================

def join_with_suffixes():
"""
Use lsuffix and rsuffix when both DataFrames contain
columns with the same name.
"""

```
section("14. JOIN WITH SUFFIXES")

employee_info = pd.DataFrame(
    {
        "name": ["Alice", "Bob"],
        "status": ["Active", "Active"],
    },
    index=[101, 102],
)

employee_status = pd.DataFrame(
    {
        "status": ["Permanent", "Contract"],
    },
    index=[101, 102],
)

result = employee_info.join(
    employee_status,
    lsuffix="_info",
    rsuffix="_employment",
)

print(result)
```

# ============================================================

# 15. JOIN MULTIPLE DATAFRAMES

# ============================================================

def join_multiple_dataframes():
"""
Multiple DataFrames can be joined in one operation.
"""

```
section("15. JOIN MULTIPLE DATAFRAMES")

employees = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
    },
    index=[101, 102, 103],
)

salaries = pd.DataFrame(
    {
        "salary": [70000, 85000, 72000],
    },
    index=[101, 102, 103],
)

performance = pd.DataFrame(
    {
        "performance_score": [92, 88, 95],
    },
    index=[101, 102, 103],
)

result = employees.join(
    [salaries, performance],
    how="left",
)

print(result)
```

# ============================================================

# 16. JOIN DIFFERENT INDEX VALUES

# ============================================================

def join_different_indexes():
"""
Demonstrate how Pandas aligns rows by index labels.
"""

```
section("16. JOIN DIFFERENT INDEX VALUES")

left = pd.DataFrame(
    {
        "feature_a": [10, 20, 30],
    },
    index=["A", "B", "C"],
)

right = pd.DataFrame(
    {
        "feature_b": [100, 200, 300],
    },
    index=["B", "C", "D"],
)

print("Left:")
print(left)

print("\nRight:")
print(right)

print("\nInner join:")
print(left.join(right, how="inner"))

print("\nOuter join:")
print(left.join(right, how="outer"))
```

# ============================================================

# 17. JOIN VS MERGE

# ============================================================

def join_vs_merge():
"""
Explain the difference between DataFrame.join() and merge().

```
JOIN:
    Primarily index-based.

MERGE:
    Primarily key/column-based.

Example:
    df1.join(df2)
    df1.merge(df2, on="employee_id")
"""

section("17. JOIN VS MERGE")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

salaries = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "salary": [70000, 85000, 72000],
    }
)

print("Using merge():")
merged = employees.merge(
    salaries,
    on="employee_id",
    how="left",
)
print(merged)

print("\nUsing join():")
joined = employees.set_index("employee_id").join(
    salaries.set_index("employee_id"),
    how="left",
)
print(joined.reset_index())
```

# ============================================================

# 18. JOIN WITH MULTIINDEX

# ============================================================

def multiindex_join():
"""
Demonstrate joining DataFrames using MultiIndex.
"""

```
section("18. JOIN WITH MULTIINDEX")

index = pd.MultiIndex.from_tuples(
    [
        ("Mumbai", 2024),
        ("Mumbai", 2025),
        ("Pune", 2024),
        ("Pune", 2025),
    ],
    names=["city", "year"],
)

sales = pd.DataFrame(
    {
        "sales": [100, 120, 90, 110],
    },
    index=index,
)

profit = pd.DataFrame(
    {
        "profit": [20, 30, 15, 25],
    },
    index=index,
)

result = sales.join(profit)

print(result)
```

# ============================================================

# 19. JOIN WITH DUPLICATE INDEX

# ============================================================

def duplicate_index_join():
"""
Duplicate indexes can produce multiple rows.

```
This can create a one-to-many or many-to-many relationship.
"""

section("19. JOIN WITH DUPLICATE INDEX")

employees = pd.DataFrame(
    {
        "name": ["Alice", "Bob"],
    },
    index=[101, 102],
)

transactions = pd.DataFrame(
    {
        "transaction": ["T1", "T2", "T3"],
        "amount": [100, 200, 300],
    },
    index=[101, 101, 102],
)

result = employees.join(
    transactions,
    how="left",
)

print(result)

print(
    "\nNotice that employee 101 appears multiple times "
    "because its index exists twice in the right DataFrame."
)
```

# ============================================================

# 20. CHECK INDEX DUPLICATES

# ============================================================

def check_duplicate_indexes():
"""
Always inspect index uniqueness when building datasets
where one-to-one relationships are expected.
"""

```
section("20. CHECK DUPLICATE INDEXES")

df = pd.DataFrame(
    {
        "value": [10, 20, 30, 40],
    },
    index=[1, 2, 2, 3],
)

print(df)

print("\nIs index unique?")
print(df.index.is_unique)

print("\nDuplicated index labels:")
print(df.index[df.index.duplicated()].unique())
```

# ============================================================

# 21. VALIDATE JOIN RESULT

# ============================================================

def validate_join_result():
"""
Validate important assumptions after joining.
"""

```
section("21. VALIDATE JOIN RESULT")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

salaries = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "salary": [70000, 85000, 72000],
    }
)

result = employees.merge(
    salaries,
    on="employee_id",
    how="left",
    validate="one_to_one",
)

print(result)

print("\nValidation successful.")
print("Every employee has at most one salary record.")
```

# ============================================================

# 22. HANDLE JOIN DATA TYPES

# ============================================================

def handle_join_data_types():
"""
Join keys should use compatible data types.

```
For example:
    int64 and string/object keys should not be treated
    as the same key.
"""

section("22. HANDLE JOIN DATA TYPES")

left = pd.DataFrame(
    {
        "employee_id": ["101", "102", "103"],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

right = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "salary": [70000, 85000, 72000],
    }
)

print("Left dtypes:")
print(left.dtypes)

print("\nRight dtypes:")
print(right.dtypes)

# Normalize the join key.
left["employee_id"] = pd.to_numeric(
    left["employee_id"],
    errors="coerce",
).astype("Int64")

right["employee_id"] = right["employee_id"].astype("Int64")

result = left.merge(
    right,
    on="employee_id",
    how="left",
)

print("\nAfter normalizing data types:")
print(result)
```

# ============================================================

# 23. JOIN AFTER SORTING

# ============================================================

def join_after_sorting():
"""
Sorting is not required for a join, but it can improve
readability and consistency of the final dataset.
"""

```
section("23. JOIN AFTER SORTING")

left = pd.DataFrame(
    {
        "name": ["Charlie", "Alice", "Bob"],
    },
    index=[103, 101, 102],
)

right = pd.DataFrame(
    {
        "salary": [72000, 70000, 85000],
    },
    index=[103, 101, 102],
)

result = left.join(right)

print("Joined:")
print(result)

print("\nSorted by index:")
print(result.sort_index())
```

# ============================================================

# 24. JOIN AND REINDEX

# ============================================================

def join_and_reindex():
"""
Reindexing can be used to explicitly align DataFrames
before joining.
"""

```
section("24. JOIN AND REINDEX")

customers = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
    },
    index=[101, 102, 103],
)

orders = pd.DataFrame(
    {
        "orders": [5, 8],
    },
    index=[101, 103],
)

result = customers.join(orders)

print(result)
```

# ============================================================

# 25. JOIN AND COLUMN SELECTION

# ============================================================

def join_selected_columns():
"""
Select only required columns before joining.

```
This reduces unnecessary data and improves readability.
"""

section("25. JOIN SELECTED COLUMNS")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 28],
    }
)

departments = pd.DataFrame(
    {
        "department_id": [10, 20, 30],
        "department": ["Data Science", "Engineering", "Marketing"],
        "location": ["Mumbai", "Pune", "Delhi"],
        "budget": [1000000, 2000000, 500000],
    }
)

departments = departments[
    ["department_id", "department"]
].set_index("department_id")

employees["department_id"] = [10, 20, 30]

result = employees.join(
    departments,
    on="department_id",
    how="left",
)

print(result)
```

# ============================================================

# 26. REAL-WORLD EMPLOYEE JOIN

# ============================================================

def real_world_employee_join():
"""
Combine employee, department, salary, and performance data.
"""

```
section("26. REAL-WORLD EMPLOYEE JOIN")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103, 104],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "department_id": [10, 20, 10, 30],
    }
)

departments = pd.DataFrame(
    {
        "department_id": [10, 20, 30],
        "department": [
            "Data Science",
            "Engineering",
            "Marketing",
        ],
    }
).set_index("department_id")

salaries = pd.DataFrame(
    {
        "employee_id": [101, 102, 103, 104],
        "salary": [70000, 85000, 72000, 65000],
    }
).set_index("employee_id")

performance = pd.DataFrame(
    {
        "employee_id": [101, 102, 103, 104],
        "performance_score": [92, 88, 95, 81],
    }
).set_index("employee_id")

result = (
    employees
    .join(departments, on="department_id", how="left")
    .set_index("employee_id")
    .join(salaries, how="left")
    .join(performance, how="left")
)

print(result)
```

# ============================================================

# 27. JOIN FOR MACHINE LEARNING DATA

# ============================================================

def machine_learning_join():
"""
Joining is extremely common when preparing ML datasets.

```
Example:

    customer table
    +
    transaction features
    +
    demographic features
    =
    training dataset

Important:
    The join must use information available at the correct
    prediction time.
"""

section("27. JOIN FOR MACHINE LEARNING DATA")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "age": [25, 32, 41, 29],
        "city": ["Mumbai", "Pune", "Delhi", "Mumbai"],
    }
).set_index("customer_id")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "total_spend": [1200, 3500, 2100, 1800],
        "transaction_count": [5, 12, 8, 6],
    }
).set_index("customer_id")

result = customers.join(
    transactions,
    how="left",
)

print(result)
```

# ============================================================

# 28. ML FEATURE DATASET

# ============================================================

def create_ml_features_with_join():
"""
Create a feature matrix from multiple DataFrames.
"""

```
section("28. CREATE ML FEATURES WITH JOIN")

customer_features = pd.DataFrame(
    {
        "age": [25, 32, 41, 29],
        "income": [40000, 60000, 85000, 50000],
    },
    index=[1, 2, 3, 4],
)

purchase_features = pd.DataFrame(
    {
        "purchase_count": [5, 12, 8, 6],
        "avg_order_value": [240, 291.67, 262.5, 300],
    },
    index=[1, 2, 3, 4],
)

engagement_features = pd.DataFrame(
    {
        "website_visits": [20, 50, 32, 27],
        "email_clicks": [4, 15, 8, 6],
    },
    index=[1, 2, 3, 4],
)

X = (
    customer_features
    .join(purchase_features)
    .join(engagement_features)
)

print("Feature matrix:")
print(X)

print("\nShape:")
print(X.shape)
```

# ============================================================

# 29. JOIN TARGET VARIABLE

# ============================================================

def join_target_variable():
"""
Join features and target while keeping the relationship
between X and y explicit.
"""

```
section("29. JOIN TARGET VARIABLE")

X = pd.DataFrame(
    {
        "age": [25, 32, 41, 29],
        "income": [40000, 60000, 85000, 50000],
    },
    index=[1, 2, 3, 4],
)

y = pd.DataFrame(
    {
        "churn": [0, 1, 0, 1],
    },
    index=[1, 2, 3, 4],
)

dataset = X.join(y)

print(dataset)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget column:")
print(y.columns.tolist())
```

# ============================================================

# 30. DATA LEAKAGE WARNING

# ============================================================

def data_leakage_warning():
"""
Explain why joins can introduce data leakage.

```
Example:
    If a prediction is supposed to be made on January 1,
    joining information generated on January 10 creates
    future-information leakage.

Safe ML workflow:
    1. Define prediction timestamp.
    2. Filter source tables by available time.
    3. Aggregate historical information.
    4. Join only valid features.
    5. Split train/test.
    6. Fit preprocessing on training data only.
"""

section("30. DATA LEAKAGE WARNING")

print(
    """
IMPORTANT:

A technically correct join can still create an incorrect
Machine Learning dataset.

Before joining data, ask:

    - Was this information available at prediction time?
    - Does the feature contain future information?
    - Was the target accidentally included?
    - Are aggregates calculated using future rows?
    - Is the same entity appearing multiple times?

Correct join != automatically correct ML dataset.
"""
)
```

# ============================================================

# 31. JOIN WITH DATE-BASED DATA

# ============================================================

def join_with_dates():
"""
Demonstrate joining datasets that contain dates.

```
Date-based ML joins require careful temporal logic.
"""

section("31. JOIN WITH DATES")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "signup_date": pd.to_datetime(
            ["2025-01-01", "2025-02-15", "2025-03-10"]
        ),
    }
).set_index("customer_id")

activity = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "last_activity": pd.to_datetime(
            ["2025-03-01", "2025-03-15", "2025-04-01"]
        ),
    }
).set_index("customer_id")

result = customers.join(activity)

print(result)
```

# ============================================================

# 32. REUSABLE JOIN FUNCTION

# ============================================================

def join_features(
base_df: pd.DataFrame,
feature_df: pd.DataFrame,
how: str = "left",
) -> pd.DataFrame:
"""
Reusable helper for index-based feature joining.

```
Parameters
----------
base_df:
    Main/base DataFrame.

feature_df:
    Additional feature DataFrame.

how:
    Join strategy such as left, right, inner, or outer.

Returns
-------
pd.DataFrame
    Joined DataFrame.
"""

if not isinstance(base_df, pd.DataFrame):
    raise TypeError("base_df must be a pandas DataFrame.")

if not isinstance(feature_df, pd.DataFrame):
    raise TypeError("feature_df must be a pandas DataFrame.")

valid_hows = {"left", "right", "inner", "outer"}

if how not in valid_hows:
    raise ValueError(
        f"how must be one of {sorted(valid_hows)}"
    )

return base_df.join(
    feature_df,
    how=how,
)
```

# ============================================================

# 33. REUSABLE KEY-BASED JOIN

# ============================================================

def merge_features(
base_df: pd.DataFrame,
feature_df: pd.DataFrame,
key: str,
how: str = "left",
) -> pd.DataFrame:
"""
Reusable column-based joining helper.

```
Uses DataFrame.merge() internally because merge() is
generally more natural when joining on ordinary columns.
"""

if key not in base_df.columns:
    raise KeyError(f"'{key}' not found in base_df.")

if key not in feature_df.columns:
    raise KeyError(f"'{key}' not found in feature_df.")

return base_df.merge(
    feature_df,
    on=key,
    how=how,
)
```

# ============================================================

# 34. REUSABLE FUNCTION DEMO

# ============================================================

def reusable_function_demo():
"""
Demonstrate reusable join functions.
"""

```
section("34. REUSABLE FUNCTION DEMO")

base = pd.DataFrame(
    {
        "age": [25, 30, 35],
    },
    index=[1, 2, 3],
)

features = pd.DataFrame(
    {
        "income": [40000, 60000, 80000],
    },
    index=[1, 2, 3],
)

result = join_features(base, features)

print(result)

print("\nKey-based merge:")

left = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

right = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "spend": [1000, 2000, 1500],
    }
)

merged = merge_features(
    left,
    right,
    key="customer_id",
)

print(merged)
```

# ============================================================

# 35. CHECK JOIN QUALITY

# ============================================================

def check_join_quality():
"""
Basic quality checks after joining.
"""

```
section("35. CHECK JOIN QUALITY")

left = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "age": [25, 30, 35, 40],
    }
)

right = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "income": [40000, 50000, 70000, 90000],
    }
)

result = left.merge(
    right,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

print(result)

print("\nRows:", len(result))
print("Columns:", len(result.columns))
print("Missing values:")
print(result.isna().sum())

print("\nDuplicate customer IDs:")
print(result["customer_id"].duplicated().sum())
```

# ============================================================

# 36. JOIN WITH COLUMN SELECTION

# ============================================================

def efficient_join():
"""
Keep only the columns required for the final dataset.

```
This is useful for:
    - readability
    - memory usage
    - avoiding duplicate columns
    - ML feature pipelines
"""

section("36. EFFICIENT JOIN")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
    }
).set_index("customer_id")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "total_spend": [1000, 2000, 1500],
        "internal_note": ["A", "B", "C"],
    }
).set_index("customer_id")

transactions = transactions[
    ["total_spend"]
]

result = customers.join(transactions)

print(result)
```

# ============================================================

# 37. COMMON JOIN MISTAKES

# ============================================================

def common_mistakes():
"""
Display common mistakes and their solutions.
"""

```
section("37. COMMON JOIN MISTAKES")

print(
    """
1. Joining the wrong key
   Problem:
       DataFrames are joined using unrelated columns.

   Solution:
       Verify the business meaning of the key.

2. Different key data types
   Problem:
       One key is integer while the other is string.

   Solution:
       Normalize data types before joining.

3. Duplicate keys
   Problem:
       Unexpected row multiplication occurs.

   Solution:
       Check:
           df.index.is_unique
       or:
           df["key"].duplicated().sum()

4. Ignoring missing values
   Problem:
       An outer/left join creates NaN values.

   Solution:
       Inspect and handle missing values explicitly.

5. Using join() when merge() is clearer
   Problem:
       Ordinary column-based relationships become difficult
       to understand.

   Solution:
       Prefer merge() for column-to-column joins.

6. Forgetting suffixes
   Problem:
       Both DataFrames contain columns with the same name.

   Solution:
       Use suffixes or select required columns first.

7. Data leakage
   Problem:
       Future information enters an ML feature table.

   Solution:
       Apply temporal/business rules before joining.

8. Assuming row order represents relationships
   Problem:
       DataFrames are combined by position rather than key.

   Solution:
       Use explicit keys and indexes.
"""
)
```

# ============================================================

# 38. JOIN CHEAT SHEET

# ============================================================

def join_cheat_sheet():
"""
Display a concise Pandas join reference.
"""

```
section("38. JOIN CHEAT SHEET")

cheat_sheet = pd.DataFrame(
    {
        "Operation": [
            "Basic join",
            "Left join",
            "Inner join",
            "Right join",
            "Outer join",
            "Join on column",
            "Join multiple DataFrames",
            "Column-based relationship",
        ],
        "Syntax": [
            "df1.join(df2)",
            "df1.join(df2, how='left')",
            "df1.join(df2, how='inner')",
            "df1.join(df2, how='right')",
            "df1.join(df2, how='outer')",
            "df1.join(df2, on='key')",
            "df1.join([df2, df3])",
            "df1.merge(df2, on='key')",
        ],
    }
)

print(cheat_sheet.to_string(index=False))
```

# ============================================================

# 39. COMPLETE JOIN WORKFLOW

# ============================================================

def complete_join_workflow():
"""
Demonstrate a production-style workflow.

```
Workflow:
    1. Inspect data.
    2. Identify relationship.
    3. Normalize keys.
    4. Check uniqueness.
    5. Select required columns.
    6. Join.
    7. Validate row count.
    8. Inspect missing values.
    9. Validate business rules.
    10. Prepare ML dataset.
"""

section("39. COMPLETE JOIN WORKFLOW")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "age": [25, 30, 35, 40],
        "city": ["Mumbai", "Pune", "Delhi", "Mumbai"],
    }
)

purchases = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "total_spend": [1200, 2500, 1800, 3200],
        "orders": [5, 8, 4, 10],
    }
)

print("Step 1: Inspect")
print(customers)
print()
print(purchases)

print("\nStep 2: Check key uniqueness")
print("Customers unique:", customers["customer_id"].is_unique)
print("Purchases unique:", purchases["customer_id"].is_unique)

print("\nStep 3: Select required features")
purchases = purchases[
    ["customer_id", "total_spend", "orders"]
]

print(purchases)

print("\nStep 4: Join")
dataset = customers.merge(
    purchases,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

print(dataset)

print("\nStep 5: Validate")
print("Rows:", len(dataset))
print("Missing values:")
print(dataset.isna().sum())

print("\nStep 6: Feature matrix")
X = dataset[
    ["age", "total_spend", "orders"]
]

print(X)
```

# ============================================================

# 40. MAIN FUNCTION

# ============================================================

def main():
"""
Run the complete Pandas join tutorial.
"""

```
print("=" * 80)
print("PANDAS JOIN TUTORIAL")
print("=" * 80)

print(
    """
This tutorial demonstrates how to combine DataFrames
using Pandas join operations.

Core idea:

    DataFrame.join()
        -> primarily index-based

    DataFrame.merge()
        -> primarily key/column-based
"""
)

# --------------------------------------------------------
# Basic joins
# --------------------------------------------------------

basic_join()
join_using_index()
left_join()
inner_join()
right_join()
outer_join()

# --------------------------------------------------------
# Missing values and keys
# --------------------------------------------------------

join_with_missing_values()
fill_missing_after_join()
join_on_column()
set_index_before_join()
join_with_suffixes()
join_multiple_dataframes()

# --------------------------------------------------------
# Advanced joins
# --------------------------------------------------------

join_different_indexes()
join_vs_merge()
multiindex_join()
duplicate_index_join()
check_duplicate_indexes()
validate_join_result()
handle_join_data_types()

# --------------------------------------------------------
# Data preparation
# --------------------------------------------------------

join_after_sorting()
join_and_reindex()
join_selected_columns()

# --------------------------------------------------------
# Real-world and ML workflows
# --------------------------------------------------------

real_world_employee_join()
machine_learning_join()
create_ml_features_with_join()
join_target_variable()
data_leakage_warning()
join_with_dates()

# --------------------------------------------------------
# Reusable functions
# --------------------------------------------------------

reusable_function_demo()
check_join_quality()
efficient_join()

# --------------------------------------------------------
# Reference
# --------------------------------------------------------

common_mistakes()
join_cheat_sheet()
complete_join_workflow()

section("TUTORIAL COMPLETE")

print(
    """
Key Takeaways:

1. DataFrame.join() is primarily index-based.
2. merge() is usually clearer for column-based joins.
3. Choose the join type based on the required relationship.
4. Always inspect duplicate keys.
5. Normalize key data types before joining.
6. Check missing values after joins.
7. Use validate= when using merge() to verify relationships.
8. Select only the columns required for your dataset.
9. In Machine Learning, prevent future information from
   entering the feature dataset.
10. A successful join is not necessarily a correct ML join.
"""
)
```

# ============================================================

# 41. SCRIPT ENTRY POINT

# ============================================================

if **name** == "**main**":
main()
"""

## Notes

Recommended learning order:

```
concat.py
    ↓
join.py
    ↓
merge.py
    ↓
groupby.py
    ↓
feature-engineering.py
    ↓
preprocessing.py
```

For Machine Learning projects, remember:

```
Raw Data
   ↓
Clean Data
   ↓
Validate Keys
   ↓
Join / Merge
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Preprocessing Pipeline
   ↓
Model Training
```

The most important rule is to understand the relationship
between datasets before joining them.
"""

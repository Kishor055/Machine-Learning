"""
Pandas Merge Tutorial
=====================

File:
03-Python-for-Machine-Learning/02-Pandas/merge.py

Description:
A complete beginner-to-advanced guide to combining Pandas
DataFrames using DataFrame.merge().

Why merge() matters:
Real-world Machine Learning datasets are rarely stored in
one table. Customer information, transactions, products,
events, labels, and features are often stored separately.

```
Pandas merge() allows us to combine these datasets using
common keys.
```

Topics Covered:
1. What is merging?
2. Creating sample datasets
3. Basic merge()
4. Merge on one column
5. Merge on multiple columns
6. Inner merge
7. Left merge
8. Right merge
9. Outer merge
10. Cross merge
11. Left-on / right-on
12. Merging on indexes
13. Index-to-column merge
14. Column-to-index merge
15. Multiple DataFrame merges
16. Suffixes
17. Indicator column
18. Merge validation
19. One-to-one relationships
20. One-to-many relationships
21. Many-to-one relationships
22. Many-to-many relationships
23. Duplicate-key problems
24. Data-type mismatches
25. Missing values
26. Sorting merged data
27. Merging time/date data
28. Machine Learning feature tables
29. Data leakage prevention
30. Reusable merge functions
31. Merge quality checks
32. Production-style workflow
33. Common mistakes
34. Merge cheat sheet

Requirements:
pip install pandas numpy

Run:
python merge.py
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
Create reusable sample DataFrames.

```
Returns
-------
tuple
    employees, departments, salaries
"""

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

# 4. WHAT IS MERGING?

# ============================================================

def what_is_merge():
"""
Explain the basic concept of merging.
"""

```
section("4. WHAT IS MERGING?")

print(
    """
Merging combines rows from two DataFrames using one or more
common keys.

Example:

    employees
    -------------------------
    employee_id | name
    101         | Alice
    102         | Bob

    salaries
    -------------------------
    employee_id | salary
    101         | 70000
    102         | 85000

    Merge result
    -------------------------------
    employee_id | name | salary
    101         | Alice | 70000
    102         | Bob   | 85000

General syntax:

    df1.merge(
        df2,
        on="key",
        how="left"
    )

Common join types:

    inner
    left
    right
    outer
    cross
"""
)
```

# ============================================================

# 5. BASIC MERGE

# ============================================================

def basic_merge():
"""
Demonstrate a basic inner merge.
"""

```
section("5. BASIC MERGE")

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
)

print("Employees:")
print(employees)

print("\nSalaries:")
print(salaries)

print("\nMerged:")
print(result)
```

# ============================================================

# 6. MERGE ON ONE COLUMN

# ============================================================

def merge_on_one_column():
"""
Merge using a single key.
"""

```
section("6. MERGE ON ONE COLUMN")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

orders = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "orders": [5, 8, 3],
    }
)

result = customers.merge(
    orders,
    on="customer_id",
    how="left",
)

print(result)
```

# ============================================================

# 7. MERGE ON MULTIPLE COLUMNS

# ============================================================

def merge_on_multiple_columns():
"""
Merge using more than one key.

```
Both columns must match for a row to match.
"""

section("7. MERGE ON MULTIPLE COLUMNS")

left = pd.DataFrame(
    {
        "customer_id": [1, 1, 2, 2],
        "year": [2024, 2025, 2024, 2025],
        "sales": [1000, 1200, 800, 900],
    }
)

right = pd.DataFrame(
    {
        "customer_id": [1, 1, 2, 2],
        "year": [2024, 2025, 2024, 2025],
        "profit": [200, 250, 150, 180],
    }
)

result = left.merge(
    right,
    on=["customer_id", "year"],
    how="inner",
)

print(result)
```

# ============================================================

# 8. INNER MERGE

# ============================================================

def inner_merge():
"""
Inner merge keeps only matching keys.
"""

```
section("8. INNER MERGE")

left = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "name": ["A", "B", "C", "D"],
    }
)

right = pd.DataFrame(
    {
        "id": [3, 4, 5, 6],
        "score": [80, 90, 75, 85],
    }
)

result = left.merge(
    right,
    on="id",
    how="inner",
)

print(result)
```

# ============================================================

# 9. LEFT MERGE

# ============================================================

def left_merge():
"""
Left merge keeps every row from the left DataFrame.
"""

```
section("9. LEFT MERGE")

left = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "name": ["A", "B", "C", "D"],
    }
)

right = pd.DataFrame(
    {
        "id": [2, 3],
        "score": [80, 90],
    }
)

result = left.merge(
    right,
    on="id",
    how="left",
)

print(result)
```

# ============================================================

# 10. RIGHT MERGE

# ============================================================

def right_merge():
"""
Right merge keeps every row from the right DataFrame.
"""

```
section("10. RIGHT MERGE")

left = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
    }
)

right = pd.DataFrame(
    {
        "id": [2, 3, 4],
        "score": [80, 90, 95],
    }
)

result = left.merge(
    right,
    on="id",
    how="right",
)

print(result)
```

# ============================================================

# 11. OUTER MERGE

# ============================================================

def outer_merge():
"""
Outer merge keeps every key from both DataFrames.
"""

```
section("11. OUTER MERGE")

left = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
    }
)

right = pd.DataFrame(
    {
        "id": [3, 4, 5],
        "score": [80, 90, 95],
    }
)

result = left.merge(
    right,
    on="id",
    how="outer",
)

print(result)
```

# ============================================================

# 12. CROSS MERGE

# ============================================================

def cross_merge():
"""
Cross merge creates the Cartesian product.

```
Every row from the left is combined with every row
from the right.

WARNING:
    The result can become extremely large.
"""

section("12. CROSS MERGE")

products = pd.DataFrame(
    {
        "product": ["Laptop", "Phone"],
    }
)

colors = pd.DataFrame(
    {
        "color": ["Black", "White", "Silver"],
    }
)

result = products.merge(
    colors,
    how="cross",
)

print(result)

print("\nNumber of combinations:", len(result))
```

# ============================================================

# 13. LEFT_ON AND RIGHT_ON

# ============================================================

def left_on_right_on():
"""
Use left_on and right_on when the key columns have
different names.
"""

```
section("13. LEFT_ON AND RIGHT_ON")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

salaries = pd.DataFrame(
    {
        "emp_id": [101, 102, 103],
        "salary": [70000, 85000, 72000],
    }
)

result = employees.merge(
    salaries,
    left_on="employee_id",
    right_on="emp_id",
    how="left",
)

print(result)
```

# ============================================================

# 14. DROP REDUNDANT KEY COLUMN

# ============================================================

def remove_redundant_key():
"""
When different key names are used, the merged result can
contain both columns.

```
Remove the redundant key if it is no longer required.
"""

section("14. REMOVE REDUNDANT KEY COLUMN")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

salaries = pd.DataFrame(
    {
        "emp_id": [101, 102, 103],
        "salary": [70000, 85000, 72000],
    }
)

result = employees.merge(
    salaries,
    left_on="employee_id",
    right_on="emp_id",
    how="left",
)

result = result.drop(columns="emp_id")

print(result)
```

# ============================================================

# 15. MERGE USING INDEX

# ============================================================

def merge_using_index():
"""
Merge using the index of both DataFrames.
"""

```
section("15. MERGE USING INDEX")

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

result = employees.merge(
    salaries,
    left_index=True,
    right_index=True,
    how="inner",
)

print(result)
```

# ============================================================

# 16. COLUMN TO INDEX MERGE

# ============================================================

def column_to_index_merge():
"""
Merge a column from one DataFrame with the index of another.
"""

```
section("16. COLUMN TO INDEX MERGE")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

salaries = pd.DataFrame(
    {
        "salary": [70000, 85000, 72000],
    },
    index=[101, 102, 103],
)

result = employees.merge(
    salaries,
    left_on="employee_id",
    right_index=True,
    how="left",
)

print(result)
```

# ============================================================

# 17. INDEX TO COLUMN MERGE

# ============================================================

def index_to_column_merge():
"""
Merge an index from one DataFrame with a column in another.
"""

```
section("17. INDEX TO COLUMN MERGE")

employees = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
    },
    index=[101, 102, 103],
)

salaries = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "salary": [70000, 85000, 72000],
    }
)

result = employees.merge(
    salaries,
    left_index=True,
    right_on="employee_id",
    how="left",
)

print(result)
```

# ============================================================

# 18. MERGE WITH SUFFIXES

# ============================================================

def merge_with_suffixes():
"""
Use suffixes when both DataFrames contain columns
with the same non-key name.
"""

```
section("18. MERGE WITH SUFFIXES")

left = pd.DataFrame(
    {
        "employee_id": [101, 102],
        "department": ["Engineering", "Data Science"],
        "status": ["Active", "Active"],
    }
)

right = pd.DataFrame(
    {
        "employee_id": [101, 102],
        "department": ["Engineering", "Research"],
        "status": ["Permanent", "Contract"],
    }
)

result = left.merge(
    right,
    on="employee_id",
    how="left",
    suffixes=("_company", "_hr"),
)

print(result)
```

# ============================================================

# 19. MERGE WITH INDICATOR

# ============================================================

def merge_with_indicator():
"""
indicator=True adds a `_merge` column showing where
each row originated.

```
Possible values:
    left_only
    right_only
    both
"""

section("19. MERGE WITH INDICATOR")

left = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
    }
)

right = pd.DataFrame(
    {
        "id": [2, 3, 4],
        "score": [80, 90, 95],
    }
)

result = left.merge(
    right,
    on="id",
    how="outer",
    indicator=True,
)

print(result)

print("\nOnly in left:")
print(result[result["_merge"] == "left_only"])

print("\nOnly in right:")
print(result[result["_merge"] == "right_only"])

print("\nPresent in both:")
print(result[result["_merge"] == "both"])
```

# ============================================================

# 20. ONE-TO-ONE MERGE

# ============================================================

def one_to_one_merge():
"""
One employee should have exactly one employee profile.
"""

```
section("20. ONE-TO-ONE MERGE")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

profiles = pd.DataFrame(
    {
        "employee_id": [101, 102, 103],
        "email": [
            "alice@example.com",
            "bob@example.com",
            "charlie@example.com",
        ],
    }
)

result = employees.merge(
    profiles,
    on="employee_id",
    how="left",
    validate="one_to_one",
)

print(result)
```

# ============================================================

# 21. ONE-TO-MANY MERGE

# ============================================================

def one_to_many_merge():
"""
One customer can have multiple orders.

```
Therefore:
    customers -> one row per customer
    orders    -> multiple rows per customer
"""

section("21. ONE-TO-MANY MERGE")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2],
        "name": ["Alice", "Bob"],
    }
)

orders = pd.DataFrame(
    {
        "order_id": [1001, 1002, 1003],
        "customer_id": [1, 1, 2],
        "amount": [500, 800, 1200],
    }
)

result = customers.merge(
    orders,
    on="customer_id",
    how="left",
    validate="one_to_many",
)

print(result)
```

# ============================================================

# 22. MANY-TO-ONE MERGE

# ============================================================

def many_to_one_merge():
"""
Many employees can belong to one department.

```
employees -> many rows per department
departments -> one row per department
"""

section("22. MANY-TO-ONE MERGE")

employees = pd.DataFrame(
    {
        "employee_id": [101, 102, 103, 104],
        "department_id": [10, 10, 20, 20],
    }
)

departments = pd.DataFrame(
    {
        "department_id": [10, 20],
        "department": ["Engineering", "Data Science"],
    }
)

result = employees.merge(
    departments,
    on="department_id",
    how="left",
    validate="many_to_one",
)

print(result)
```

# ============================================================

# 23. MANY-TO-MANY MERGE

# ============================================================

def many_to_many_merge():
"""
Demonstrate a many-to-many relationship.

```
WARNING:
    Many-to-many merges can multiply rows dramatically.
"""

section("23. MANY-TO-MANY MERGE")

left = pd.DataFrame(
    {
        "student": ["Alice", "Alice", "Bob"],
        "course": ["Python", "Python", "Python"],
        "attempt": [1, 2, 1],
    }
)

right = pd.DataFrame(
    {
        "student": ["Alice", "Alice", "Bob"],
        "course": ["Python", "Python", "Python"],
        "project": ["A", "B", "C"],
    }
)

result = left.merge(
    right,
    on=["student", "course"],
    how="inner",
    validate="many_to_many",
)

print(result)
```

# ============================================================

# 24. DUPLICATE KEY PROBLEM

# ============================================================

def duplicate_key_problem():
"""
Duplicate keys can cause unexpected row multiplication.

```
Example:
    left key occurs twice
    right key occurs three times

    Result:
        2 × 3 = 6 rows for that key
"""

section("24. DUPLICATE KEY PROBLEM")

left = pd.DataFrame(
    {
        "id": [1, 1],
        "feature": ["A", "B"],
    }
)

right = pd.DataFrame(
    {
        "id": [1, 1, 1],
        "score": [10, 20, 30],
    }
)

result = left.merge(
    right,
    on="id",
    how="inner",
)

print(result)

print("\nResult rows:", len(result))
```

# ============================================================

# 25. CHECK DUPLICATE KEYS

# ============================================================

def check_duplicate_keys():
"""
Inspect duplicate merge keys before performing a merge.
"""

```
section("25. CHECK DUPLICATE KEYS")

df = pd.DataFrame(
    {
        "customer_id": [1, 2, 2, 3, 4],
        "value": [10, 20, 25, 30, 40],
    }
)

print(df)

print("\nAre keys unique?")
print(df["customer_id"].is_unique)

print("\nNumber of duplicate rows:")
print(df["customer_id"].duplicated().sum())

print("\nDuplicated records:")
print(
    df[df["customer_id"].duplicated(keep=False)]
)
```

# ============================================================

# 26. DATA TYPE MISMATCH

# ============================================================

def handle_data_type_mismatch():
"""
Merge keys should use compatible data types.

```
A common real-world problem is:
    left key  -> string
    right key -> integer
"""

section("26. DATA TYPE MISMATCH")

customers = pd.DataFrame(
    {
        "customer_id": ["1", "2", "3"],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

purchases = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "spend": [1000, 2000, 1500],
    }
)

print("Before normalization:")
print(customers.dtypes)
print(purchases.dtypes)

customers["customer_id"] = pd.to_numeric(
    customers["customer_id"],
    errors="coerce",
).astype("Int64")

purchases["customer_id"] = purchases["customer_id"].astype(
    "Int64"
)

result = customers.merge(
    purchases,
    on="customer_id",
    how="left",
)

print("\nAfter normalization:")
print(result)
```

# ============================================================

# 27. HANDLE MISSING VALUES

# ============================================================

def handle_missing_values():
"""
Missing keys generally do not match ordinary values.

```
After a left/outer merge, non-matching columns can contain NaN.
"""

section("27. HANDLE MISSING VALUES")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "Diana"],
    }
)

purchases = pd.DataFrame(
    {
        "customer_id": [1, 2],
        "spend": [1000, 2000],
    }
)

result = customers.merge(
    purchases,
    on="customer_id",
    how="left",
)

print("Merged:")
print(result)

print("\nMissing values:")
print(result.isna().sum())

result["spend"] = result["spend"].fillna(0)

print("\nAfter filling missing spend:")
print(result)
```

# ============================================================

# 28. SORT MERGED DATA

# ============================================================

def sort_merged_data():
"""
Sort the merged result for easier analysis.
"""

```
section("28. SORT MERGED DATA")

left = pd.DataFrame(
    {
        "id": [3, 1, 2],
        "name": ["Charlie", "Alice", "Bob"],
    }
)

right = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "score": [80, 90, 85],
    }
)

result = left.merge(
    right,
    on="id",
    how="left",
)

result = result.sort_values(
    "score",
    ascending=False,
)

print(result)
```

# ============================================================

# 29. MERGE MULTIPLE DATAFRAMES

# ============================================================

def merge_multiple_dataframes():
"""
Merge multiple datasets sequentially.
"""

```
section("29. MERGE MULTIPLE DATAFRAMES")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

purchases = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "spend": [1000, 2000, 1500],
    }
)

engagement = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "visits": [20, 50, 30],
    }
)

result = (
    customers
    .merge(
        purchases,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        engagement,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
)

print(result)
```

# ============================================================

# 30. MERGE USING INDEX AND COLUMN

# ============================================================

def mixed_index_column_merge():
"""
Demonstrate merging a DataFrame column with another
DataFrame's index.
"""

```
section("30. MIXED INDEX/COLUMN MERGE")

orders = pd.DataFrame(
    {
        "order_id": [1001, 1002, 1003],
        "customer_id": [1, 2, 3],
        "amount": [500, 800, 1200],
    }
)

customers = pd.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
        "city": ["Mumbai", "Pune", "Delhi"],
    },
    index=[1, 2, 3],
)

result = orders.merge(
    customers,
    left_on="customer_id",
    right_index=True,
    how="left",
)

print(result)
```

# ============================================================

# 31. MERGE WITH CATEGORY DATA

# ============================================================

def merge_category_data():
"""
Merge entity-level records with category metadata.
"""

```
section("31. MERGE CATEGORY DATA")

products = pd.DataFrame(
    {
        "product_id": [101, 102, 103, 104],
        "category_id": [1, 2, 1, 3],
        "price": [500, 1000, 750, 1200],
    }
)

categories = pd.DataFrame(
    {
        "category_id": [1, 2, 3],
        "category": ["Electronics", "Books", "Home"],
    }
)

result = products.merge(
    categories,
    on="category_id",
    how="left",
    validate="many_to_one",
)

print(result)
```

# ============================================================

# 32. DATE-BASED MERGE

# ============================================================

def merge_date_data():
"""
Merge datasets containing date columns.

```
Dates should be converted to proper datetime values
before comparison or joining.
"""

section("32. DATE-BASED MERGE")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "date": [
            "2025-01-01",
            "2025-02-01",
            "2025-03-01",
        ],
        "segment": ["A", "B", "A"],
    }
)

activity = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "date": [
            "2025-01-01",
            "2025-02-01",
            "2025-03-01",
        ],
        "visits": [20, 35, 40],
    }
)

customers["date"] = pd.to_datetime(customers["date"])
activity["date"] = pd.to_datetime(activity["date"])

result = customers.merge(
    activity,
    on=["customer_id", "date"],
    how="left",
)

print(result)
```

# ============================================================

# 33. MERGE AS-OF

# ============================================================

def merge_asof_example():
"""
Demonstrate merge_asof() for nearest time-based matching.

```
merge_asof() is useful when an exact timestamp match is not
required.

Example:
    Match each transaction with the most recent known price.
"""

section("33. MERGE_ASOF")

transactions = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(
            [
                "2025-01-01 10:00",
                "2025-01-01 10:05",
                "2025-01-01 10:10",
            ]
        ),
        "quantity": [2, 3, 1],
    }
)

prices = pd.DataFrame(
    {
        "timestamp": pd.to_datetime(
            [
                "2025-01-01 09:55",
                "2025-01-01 10:03",
                "2025-01-01 10:08",
            ]
        ),
        "price": [100, 105, 110],
    }
)

transactions = transactions.sort_values("timestamp")
prices = prices.sort_values("timestamp")

result = pd.merge_asof(
    transactions,
    prices,
    on="timestamp",
    direction="backward",
)

print(result)
```

# ============================================================

# 34. MACHINE LEARNING FEATURE MERGE

# ============================================================

def machine_learning_feature_merge():
"""
Combine multiple feature tables into one ML dataset.
"""

```
section("34. MACHINE LEARNING FEATURE MERGE")

customer_features = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "age": [25, 32, 41, 29],
        "income": [40000, 60000, 85000, 50000],
    }
)

transaction_features = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "total_spend": [1200, 3500, 2100, 1800],
        "transaction_count": [5, 12, 8, 6],
    }
)

engagement_features = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "website_visits": [20, 50, 32, 27],
        "email_clicks": [4, 15, 8, 6],
    }
)

X = (
    customer_features
    .merge(
        transaction_features,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        engagement_features,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
)

print("ML Feature Dataset:")
print(X)

print("\nFeature matrix:")
print(
    X[
        [
            "age",
            "income",
            "total_spend",
            "transaction_count",
            "website_visits",
            "email_clicks",
        ]
    ]
)
```

# ============================================================

# 35. MERGE TARGET LABEL

# ============================================================

def merge_target_label():
"""
Merge target labels with a feature table.

```
Keeping the target table separate can make the data pipeline
easier to audit.
"""

section("35. MERGE TARGET LABEL")

features = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "age": [25, 32, 41, 29],
        "income": [40000, 60000, 85000, 50000],
    }
)

labels = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "churn": [0, 1, 0, 1],
    }
)

dataset = features.merge(
    labels,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

print(dataset)

X = dataset[
    [
        "age",
        "income",
    ]
]

y = dataset["churn"]

print("\nX:")
print(X)

print("\ny:")
print(y)
```

# ============================================================

# 36. DATA LEAKAGE

# ============================================================

def data_leakage_warning():
"""
Explain how merges can introduce ML data leakage.
"""

```
section("36. DATA LEAKAGE WARNING")

print(
    """
A merge can be technically correct but still create
a scientifically invalid Machine Learning dataset.

Example:

    Prediction date:
        January 1

    Transaction table:
        Contains transactions from January 1 -> January 31

    If all January transactions are merged into the feature
    table, the model may receive information that was not
    available when the prediction was supposed to be made.

Before merging ML features, verify:

    1. Prediction timestamp.
    2. Feature availability timestamp.
    3. Aggregation window.
    4. Target definition.
    5. Entity relationship.
    6. Duplicate keys.
    7. Train/test boundaries.

Rule:

    Future information must never become an input feature.
"""
)
```

# ============================================================

# 37. MERGE QUALITY CHECKS

# ============================================================

def merge_quality_checks():
"""
Perform practical post-merge validation.
"""

```
section("37. MERGE QUALITY CHECKS")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "age": [25, 30, 35, 40],
    }
)

purchases = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "spend": [1000, 2000, 1500, 3000],
    }
)

result = customers.merge(
    purchases,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

print(result)

print("\nRows:", len(result))
print("Columns:", len(result.columns))

print("\nMissing values:")
print(result.isna().sum())

print("\nDuplicate keys:")
print(result["customer_id"].duplicated().sum())

print("\nKey uniqueness:")
print(result["customer_id"].is_unique)
```

# ============================================================

# 38. REUSABLE MERGE FUNCTION

# ============================================================

def merge_dataframes(
left: pd.DataFrame,
right: pd.DataFrame,
on,
how: str = "left",
validate: str | None = None,
) -> pd.DataFrame:
"""
Reusable wrapper around DataFrame.merge().

```
Parameters
----------
left:
    Left DataFrame.

right:
    Right DataFrame.

on:
    Column name or list of column names.

how:
    Merge strategy.

validate:
    Optional relationship validation.

Returns
-------
pd.DataFrame
    Merged DataFrame.
"""

if not isinstance(left, pd.DataFrame):
    raise TypeError("left must be a pandas DataFrame.")

if not isinstance(right, pd.DataFrame):
    raise TypeError("right must be a pandas DataFrame.")

valid_hows = {
    "left",
    "right",
    "inner",
    "outer",
    "cross",
}

if how not in valid_hows:
    raise ValueError(
        f"how must be one of {sorted(valid_hows)}"
    )

if how == "cross" and on is not None:
    raise ValueError(
        "Cross merge should not use the 'on' parameter."
    )

if how == "cross":
    return left.merge(
        right,
        how="cross",
    )

return left.merge(
    right,
    on=on,
    how=how,
    validate=validate,
)
```

# ============================================================

# 39. REUSABLE FUNCTION DEMO

# ============================================================

def reusable_function_demo():
"""
Demonstrate the reusable merge helper.
"""

```
section("39. REUSABLE FUNCTION DEMO")

left = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

right = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "score": [90, 85, 95],
    }
)

result = merge_dataframes(
    left,
    right,
    on="id",
    how="left",
    validate="one_to_one",
)

print(result)
```

# ============================================================

# 40. EFFICIENT MERGE

# ============================================================

def efficient_merge():
"""
Select only required columns before merging.

```
Benefits:
    - less memory
    - fewer duplicate columns
    - clearer output
    - simpler feature datasets
"""

section("40. EFFICIENT MERGE")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "internal_note": ["A", "B", "C"],
    }
)

purchases = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "total_spend": [1000, 2000, 1500],
        "order_count": [5, 8, 4],
        "internal_code": ["X", "Y", "Z"],
    }
)

customers_selected = customers[
    ["customer_id", "age"]
]

purchases_selected = purchases[
    [
        "customer_id",
        "total_spend",
        "order_count",
    ]
]

result = customers_selected.merge(
    purchases_selected,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

print(result)
```

# ============================================================

# 41. MERGE AND RESET INDEX

# ============================================================

def merge_and_reset_index():
"""
Reset the index when the index is not part of the desired
final dataset.
"""

```
section("41. MERGE AND RESET INDEX")

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

result = left.merge(
    right,
    left_index=True,
    right_index=True,
    how="left",
)

print("Merged:")
print(result)

result = result.reset_index()

print("\nAfter reset_index():")
print(result)
```

# ============================================================

# 42. COMPLETE REAL-WORLD WORKFLOW

# ============================================================

def complete_merge_workflow():
"""
Demonstrate a production-style merge workflow.

```
Steps:
    1. Inspect source tables.
    2. Identify the business key.
    3. Normalize key data types.
    4. Check key uniqueness.
    5. Select required columns.
    6. Merge.
    7. Validate relationship.
    8. Check row count.
    9. Check missing values.
    10. Check duplicates.
    11. Check business rules.
    12. Prepare ML features.
"""

section("42. COMPLETE REAL-WORLD WORKFLOW")

customers = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "age": [25, 32, 41, 29],
        "city": ["Mumbai", "Pune", "Delhi", "Mumbai"],
    }
)

transactions = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "total_spend": [1200, 3500, 2100, 1800],
        "transaction_count": [5, 12, 8, 6],
    }
)

engagement = pd.DataFrame(
    {
        "customer_id": [1, 2, 3, 4],
        "website_visits": [20, 50, 32, 27],
        "email_clicks": [4, 15, 8, 6],
    }
)

# --------------------------------------------------------
# Step 1: Inspect
# --------------------------------------------------------

print("Customer data:")
print(customers)

print("\nTransaction data:")
print(transactions)

print("\nEngagement data:")
print(engagement)

# --------------------------------------------------------
# Step 2: Check keys
# --------------------------------------------------------

print("\nKey uniqueness:")
print(
    "Customers:",
    customers["customer_id"].is_unique,
)

print(
    "Transactions:",
    transactions["customer_id"].is_unique,
)

print(
    "Engagement:",
    engagement["customer_id"].is_unique,
)

# --------------------------------------------------------
# Step 3: Select required columns
# --------------------------------------------------------

transactions = transactions[
    [
        "customer_id",
        "total_spend",
        "transaction_count",
    ]
]

engagement = engagement[
    [
        "customer_id",
        "website_visits",
        "email_clicks",
    ]
]

# --------------------------------------------------------
# Step 4: Merge
# --------------------------------------------------------

dataset = (
    customers
    .merge(
        transactions,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        engagement,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
)

print("\nFinal merged dataset:")
print(dataset)

# --------------------------------------------------------
# Step 5: Validate
# --------------------------------------------------------

print("\nRows:", len(dataset))

print("\nMissing values:")
print(dataset.isna().sum())

print("\nDuplicate customer IDs:")
print(
    dataset["customer_id"].duplicated().sum()
)

# --------------------------------------------------------
# Step 6: Prepare X
# --------------------------------------------------------

X = dataset[
    [
        "age",
        "total_spend",
        "transaction_count",
        "website_visits",
        "email_clicks",
    ]
]

print("\nMachine Learning feature matrix X:")
print(X)
```

# ============================================================

# 43. COMMON MERGE MISTAKES

# ============================================================

def common_mistakes():
"""
Display common Pandas merge mistakes and solutions.
"""

```
section("43. COMMON MERGE MISTAKES")

print(
    """
1. WRONG MERGE KEY
   Problem:
       Joining unrelated columns.

   Solution:
       Understand the business meaning of the key.

--------------------------------------------------------

2. DIFFERENT DATA TYPES
   Problem:
       One key is string and another is integer.

   Solution:
       Normalize key types before merging.

--------------------------------------------------------

3. DUPLICATE KEYS
   Problem:
       Unexpected row multiplication.

   Solution:
       Check:
           df["key"].is_unique

       And use:
           validate="one_to_one"

       or:
           validate="many_to_one"

--------------------------------------------------------

4. WRONG JOIN TYPE
   Problem:
       Rows disappear unexpectedly.

   Solution:
       Understand:

           inner -> only matches
           left  -> all left rows
           right -> all right rows
           outer -> all rows

--------------------------------------------------------

5. UNEXPECTED NaN VALUES
   Problem:
       Some keys have no matching record.

   Solution:
       Investigate unmatched keys before filling values.

--------------------------------------------------------

6. DUPLICATE COLUMNS
   Problem:
       Both tables contain columns with the same name.

   Solution:
       Use:
           suffixes=("_left", "_right")

       Or select required columns first.

--------------------------------------------------------

7. MANY-TO-MANY EXPLOSION
   Problem:
       A merge creates far more rows than expected.

   Solution:
       Check key uniqueness and relationship cardinality.

--------------------------------------------------------

8. CROSS MERGE
   Problem:
       Dataset becomes extremely large.

   Solution:
       Use how="cross" only when a Cartesian product
       is explicitly required.

--------------------------------------------------------

9. DATA LEAKAGE
   Problem:
       Future information enters ML features.

   Solution:
       Apply time-aware filtering before merging.

--------------------------------------------------------

10. NO VALIDATION
    Problem:
        Incorrect relationships go unnoticed.

    Solution:
        Use the validate parameter whenever possible.
"""
)
```

# ============================================================

# 44. MERGE CHEAT SHEET

# ============================================================

def merge_cheat_sheet():
"""
Display a compact merge reference.
"""

```
section("44. MERGE CHEAT SHEET")

cheat_sheet = pd.DataFrame(
    {
        "Task": [
            "Basic merge",
            "Inner merge",
            "Left merge",
            "Right merge",
            "Outer merge",
            "Multiple keys",
            "Different key names",
            "Index-to-index",
            "Column-to-index",
            "Indicator",
            "Validate relationship",
            "Cross merge",
        ],
        "Syntax": [
            "df1.merge(df2, on='id')",
            "df1.merge(df2, on='id', how='inner')",
            "df1.merge(df2, on='id', how='left')",
            "df1.merge(df2, on='id', how='right')",
            "df1.merge(df2, on='id', how='outer')",
            "on=['id', 'date']",
            "left_on='id', right_on='emp_id'",
            "left_index=True, right_index=True",
            "left_on='id', right_index=True",
            "indicator=True",
            "validate='one_to_one'",
            "how='cross'",
        ],
    }
)

print(cheat_sheet.to_string(index=False))
```

# ============================================================

# 45. MAIN FUNCTION

# ============================================================

def main():
"""
Run the complete Pandas merge tutorial.
"""

```
print("=" * 80)
print("PANDAS MERGE TUTORIAL")
print("=" * 80)

what_is_merge()

# Basic merge operations
basic_merge()
merge_on_one_column()
merge_on_multiple_columns()

# Join types
inner_merge()
left_merge()
right_merge()
outer_merge()
cross_merge()

# Key handling
left_on_right_on()
remove_redundant_key()

# Index operations
merge_using_index()
column_to_index_merge()
index_to_column_merge()

# Advanced features
merge_with_suffixes()
merge_with_indicator()

# Relationship validation
one_to_one_merge()
one_to_many_merge()
many_to_one_merge()
many_to_many_merge()

# Data quality
duplicate_key_problem()
check_duplicate_keys()
handle_data_type_mismatch()
handle_missing_values()
sort_merged_data()

# Multiple datasets
merge_multiple_dataframes()
mixed_index_column_merge()
merge_category_data()

# Time-based data
merge_date_data()
merge_asof_example()

# Machine Learning
machine_learning_feature_merge()
merge_target_label()
data_leakage_warning()

# Quality and reusable functions
merge_quality_checks()
reusable_function_demo()
efficient_merge()
merge_and_reset_index()

# Complete workflow
complete_merge_workflow()

# Reference
common_mistakes()
merge_cheat_sheet()

section("TUTORIAL COMPLETE")

print(
    """
Key Takeaways:

1. merge() combines DataFrames using explicit keys.
2. Use `on=` when both DataFrames share the same key name.
3. Use `left_on=` and `right_on=` for different key names.
4. Use `how=` to control which rows are preserved.
5. Use multiple columns when a single key is insufficient.
6. Check duplicate keys before merging.
7. Use `validate=` to enforce expected relationships.
8. Use `indicator=True` to investigate unmatched records.
9. Use `suffixes=` to handle overlapping column names.
10. Be especially careful with many-to-many merges.
11. Normalize key data types before merging.
12. In ML, ensure joins cannot introduce future information.
13. A successful merge should always be followed by validation.

Remember:

    Correct syntax
         +
    Correct key
         +
    Correct relationship
         +
    Correct time logic
         +
    Validation
         =
    Reliable dataset
"""
)
```

# ============================================================

# 46. SCRIPT ENTRY POINT

# ============================================================

if **name** == "**main**":
main()

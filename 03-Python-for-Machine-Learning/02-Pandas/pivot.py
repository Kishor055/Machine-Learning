"""
Pandas Pivot Tutorial
=====================

File:
03-Python-for-Machine-Learning/02-Pandas/pivot.py

Description:
A complete beginner-to-advanced guide to reshaping and
summarizing tabular data with Pandas.

```
This tutorial focuses on:
    - pivot()
    - pivot_table()
    - melt()
    - stack()
    - unstack()
    - MultiIndex
    - aggregation
    - missing values
    - margins and totals
    - sorting pivot tables
    - reshaping for Machine Learning
    - avoiding data leakage
```

Why Pivoting Matters:
Real-world data is often stored in "long" format, while
analysis and reporting may require "wide" format.

```
Example:

    Long format
    -----------------------------
    customer | month | sales
    A        | Jan   | 100
    A        | Feb   | 120
    B        | Jan   | 150
    B        | Feb   | 180

    Wide format
    -----------------------------
    customer | Jan | Feb
    A        | 100 | 120
    B        | 150 | 180
```

Requirements:
pip install pandas numpy

Run:
python pivot.py
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
pd.set_option("display.width", 140)
pd.set_option("display.max_rows", 100)

def section(title: str) -> None:
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

# ============================================================

# 3. CREATE SAMPLE SALES DATA

# ============================================================

def create_sales_data() -> pd.DataFrame:
"""
Create a reusable sales dataset.

```
Returns
-------
pd.DataFrame
    Long-format sales data.
"""

return pd.DataFrame(
    {
        "date": pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-02",
                "2025-01-01",
                "2025-01-02",
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-03",
            ]
        ),
        "region": [
            "West",
            "West",
            "East",
            "East",
            "North",
            "North",
            "West",
            "East",
        ],
        "product": [
            "Laptop",
            "Phone",
            "Laptop",
            "Phone",
            "Laptop",
            "Phone",
            "Laptop",
            "Phone",
        ],
        "salesperson": [
            "Alice",
            "Bob",
            "Charlie",
            "Diana",
            "Ethan",
            "Fiona",
            "Alice",
            "Diana",
        ],
        "sales": [
            1000,
            700,
            1200,
            800,
            900,
            600,
            1100,
            950,
        ],
        "quantity": [
            2,
            3,
            2,
            4,
            1,
            2,
            2,
            5,
        ],
    }
)
```

# ============================================================

# 4. UNDERSTANDING LONG AND WIDE DATA

# ============================================================

def long_vs_wide():
"""
Explain long-format and wide-format datasets.
"""

```
section("4. LONG VS WIDE DATA")

long_data = pd.DataFrame(
    {
        "student": [
            "Alice",
            "Alice",
            "Bob",
            "Bob",
        ],
        "subject": [
            "Math",
            "Science",
            "Math",
            "Science",
        ],
        "score": [
            90,
            85,
            80,
            88,
        ],
    }
)

print("Long format:")
print(long_data)

wide_data = long_data.pivot(
    index="student",
    columns="subject",
    values="score",
)

print("\nWide format:")
print(wide_data)
```

# ============================================================

# 5. BASIC PIVOT

# ============================================================

def basic_pivot():
"""
Demonstrate DataFrame.pivot().

```
pivot() reshapes data without aggregation.
"""

section("5. BASIC PIVOT")

df = pd.DataFrame(
    {
        "student": [
            "Alice",
            "Alice",
            "Bob",
            "Bob",
        ],
        "subject": [
            "Math",
            "Science",
            "Math",
            "Science",
        ],
        "score": [
            90,
            85,
            80,
            88,
        ],
    }
)

print("Original:")
print(df)

result = df.pivot(
    index="student",
    columns="subject",
    values="score",
)

print("\nPivoted:")
print(result)
```

# ============================================================

# 6. PIVOT INDEX

# ============================================================

def pivot_index():
"""
Use the index parameter to define rows.
"""

```
section("6. PIVOT INDEX")

df = pd.DataFrame(
    {
        "region": ["West", "West", "East", "East"],
        "product": ["Laptop", "Phone", "Laptop", "Phone"],
        "sales": [1000, 700, 1200, 800],
    }
)

result = df.pivot(
    index="region",
    columns="product",
    values="sales",
)

print(result)
```

# ============================================================

# 7. PIVOT COLUMNS

# ============================================================

def pivot_columns():
"""
The columns parameter determines the new column labels.
"""

```
section("7. PIVOT COLUMNS")

df = pd.DataFrame(
    {
        "region": ["West", "West", "East", "East"],
        "product": ["Laptop", "Phone", "Laptop", "Phone"],
        "sales": [1000, 700, 1200, 800],
    }
)

result = df.pivot(
    index="region",
    columns="product",
    values="sales",
)

print(result)
```

# ============================================================

# 8. PIVOT VALUES

# ============================================================

def pivot_values():
"""
The values parameter determines what values populate
the resulting table.
"""

```
section("8. PIVOT VALUES")

df = pd.DataFrame(
    {
        "region": ["West", "West", "East", "East"],
        "product": ["Laptop", "Phone", "Laptop", "Phone"],
        "sales": [1000, 700, 1200, 800],
        "quantity": [2, 3, 2, 4],
    }
)

sales_pivot = df.pivot(
    index="region",
    columns="product",
    values="sales",
)

quantity_pivot = df.pivot(
    index="region",
    columns="product",
    values="quantity",
)

print("Sales:")
print(sales_pivot)

print("\nQuantity:")
print(quantity_pivot)
```

# ============================================================

# 9. MULTIPLE VALUES WITH PIVOT

# ============================================================

def multiple_values_pivot():
"""
Pivot multiple value columns.

```
This produces a MultiIndex column structure.
"""

section("9. MULTIPLE VALUES WITH PIVOT")

df = pd.DataFrame(
    {
        "region": ["West", "West", "East", "East"],
        "product": ["Laptop", "Phone", "Laptop", "Phone"],
        "sales": [1000, 700, 1200, 800],
        "quantity": [2, 3, 2, 4],
    }
)

result = df.pivot(
    index="region",
    columns="product",
    values=["sales", "quantity"],
)

print(result)
```

# ============================================================

# 10. PIVOT_TABLE

# ============================================================

def basic_pivot_table():
"""
pivot_table() is similar to pivot(), but supports
aggregation when duplicate combinations exist.
"""

```
section("10. BASIC PIVOT_TABLE")

df = pd.DataFrame(
    {
        "region": [
            "West",
            "West",
            "West",
            "East",
        ],
        "product": [
            "Laptop",
            "Laptop",
            "Phone",
            "Laptop",
        ],
        "sales": [
            1000,
            500,
            700,
            1200,
        ],
    }
)

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="sum",
)

print(result)
```

# ============================================================

# 11. WHY PIVOT CAN FAIL WITH DUPLICATES

# ============================================================

def duplicate_pivot_problem():
"""
pivot() requires each index/column combination to be unique.

```
If duplicates exist, use pivot_table() with an aggregation
function.
"""

section("11. DUPLICATE PIVOT PROBLEM")

df = pd.DataFrame(
    {
        "region": ["West", "West", "West"],
        "product": ["Laptop", "Laptop", "Phone"],
        "sales": [1000, 500, 700],
    }
)

print("Original data:")
print(df)

print(
    """
There are two rows for:

    region = West
    product = Laptop

Therefore, pivot() cannot determine which value should
occupy that single cell.

Use pivot_table() when aggregation is required.
"""
)

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="sum",
)

print("Using pivot_table():")
print(result)
```

# ============================================================

# 12. SUM AGGREGATION

# ============================================================

def pivot_sum():
"""
Aggregate duplicate observations using sum.
"""

```
section("12. PIVOT_TABLE SUM")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="sum",
)

print(result)
```

# ============================================================

# 13. MEAN AGGREGATION

# ============================================================

def pivot_mean():
"""
Calculate average values inside each group.
"""

```
section("13. PIVOT_TABLE MEAN")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="mean",
)

print(result)
```

# ============================================================

# 14. COUNT AGGREGATION

# ============================================================

def pivot_count():
"""
Count observations for each region/product combination.
"""

```
section("14. PIVOT_TABLE COUNT")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="count",
)

print(result)
```

# ============================================================

# 15. MIN/MAX AGGREGATION

# ============================================================

def pivot_min_max():
"""
Calculate minimum and maximum values.
"""

```
section("15. PIVOT_TABLE MIN/MAX")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc=["min", "max"],
)

print(result)
```

# ============================================================

# 16. MULTIPLE AGGREGATION FUNCTIONS

# ============================================================

def multiple_aggregation_functions():
"""
Apply multiple aggregation functions simultaneously.
"""

```
section("16. MULTIPLE AGGREGATION FUNCTIONS")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc=["sum", "mean", "count"],
)

print(result)
```

# ============================================================

# 17. MULTIPLE VALUE COLUMNS

# ============================================================

def multiple_value_columns():
"""
Aggregate multiple numerical columns.
"""

```
section("17. MULTIPLE VALUE COLUMNS")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values=["sales", "quantity"],
    aggfunc="sum",
)

print(result)
```

# ============================================================

# 18. MULTIPLE INDEX LEVELS

# ============================================================

def multiple_index_levels():
"""
Create a pivot table using multiple index columns.
"""

```
section("18. MULTIPLE INDEX LEVELS")

df = create_sales_data()

result = df.pivot_table(
    index=["region", "salesperson"],
    columns="product",
    values="sales",
    aggfunc="sum",
)

print(result)
```

# ============================================================

# 19. MULTIPLE COLUMN LEVELS

# ============================================================

def multiple_column_levels():
"""
Use multiple columns to create hierarchical column labels.
"""

```
section("19. MULTIPLE COLUMN LEVELS")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns=["product", "salesperson"],
    values="sales",
    aggfunc="sum",
)

print(result)
```

# ============================================================

# 20. FILL MISSING VALUES

# ============================================================

def fill_missing_pivot_values():
"""
Use fill_value to replace missing combinations.
"""

```
section("20. FILL MISSING PIVOT VALUES")

df = pd.DataFrame(
    {
        "region": ["West", "West", "East"],
        "product": ["Laptop", "Phone", "Laptop"],
        "sales": [1000, 700, 1200],
    }
)

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="sum",
    fill_value=0,
)

print(result)
```

# ============================================================

# 21. DROPNA

# ============================================================

def pivot_dropna():
"""
Control whether empty combinations are removed.
"""

```
section("21. PIVOT DROPNA")

df = pd.DataFrame(
    {
        "region": ["West", "East"],
        "product": ["Laptop", "Phone"],
        "sales": [1000, 800],
    }
)

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="sum",
    dropna=False,
    fill_value=0,
)

print(result)
```

# ============================================================

# 22. MARGINS / TOTALS

# ============================================================

def pivot_margins():
"""
Add row and column totals using margins=True.
"""

```
section("22. PIVOT MARGINS")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Total",
)

print(result)
```

# ============================================================

# 23. SORT PIVOT TABLE

# ============================================================

def sort_pivot_table():
"""
Sort pivot table rows and columns.
"""

```
section("23. SORT PIVOT TABLE")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="sum",
    fill_value=0,
)

result = result.sort_index()

print("Sorted by index:")
print(result)

print("\nSorted by Laptop sales:")
print(
    result.sort_values(
        by="Laptop",
        ascending=False,
    )
)
```

# ============================================================

# 24. PIVOT TABLE WITH CUSTOM AGGREGATION

# ============================================================

def custom_aggregation():
"""
Use custom functions with pivot_table().
"""

```
section("24. CUSTOM AGGREGATION")

df = create_sales_data()

def sales_range(series: pd.Series) -> float:
    """Return max - min."""
    return series.max() - series.min()

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc=sales_range,
    fill_value=0,
)

print(result)
```

# ============================================================

# 25. PIVOT TABLE WITH NAMED AGGREGATION

# ============================================================

def named_aggregation():
"""
Use groupby + named aggregation when precise output
column names are required.

```
pivot_table is convenient for reporting, while groupby
can offer more control over the resulting schema.
"""

section("25. NAMED AGGREGATION")

df = create_sales_data()

result = (
    df.groupby(["region", "product"], as_index=False)
    .agg(
        total_sales=("sales", "sum"),
        average_sales=("sales", "mean"),
        total_quantity=("quantity", "sum"),
        transaction_count=("sales", "count"),
    )
)

print(result)
```

# ============================================================

# 26. PIVOT AND RESET_INDEX

# ============================================================

def pivot_reset_index():
"""
Convert pivot-table index levels back into ordinary columns.
"""

```
section("26. PIVOT AND RESET_INDEX")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values="sales",
    aggfunc="sum",
    fill_value=0,
)

print("Pivot table:")
print(result)

result = result.reset_index()

print("\nAfter reset_index():")
print(result)
```

# ============================================================

# 27. FLATTEN MULTIINDEX COLUMNS

# ============================================================

def flatten_multiindex_columns():
"""
Flatten hierarchical columns created by pivot_table().
"""

```
section("27. FLATTEN MULTIINDEX COLUMNS")

df = create_sales_data()

result = df.pivot_table(
    index="region",
    columns="product",
    values=["sales", "quantity"],
    aggfunc="sum",
    fill_value=0,
)

print("Original MultiIndex columns:")
print(result)

result.columns = [
    "_".join(
        str(part)
        for part in column
        if str(part) != ""
    )
    for column in result.columns
]

result = result.reset_index()

print("\nFlattened columns:")
print(result)
```

# ============================================================

# 28. MELT

# ============================================================

def basic_melt():
"""
Convert wide-format data back to long-format data.

```
melt() is conceptually the reverse of pivoting.
"""

section("28. MELT")

wide = pd.DataFrame(
    {
        "student": ["Alice", "Bob"],
        "Math": [90, 80],
        "Science": [85, 88],
    }
)

print("Wide:")
print(wide)

long = wide.melt(
    id_vars="student",
    var_name="subject",
    value_name="score",
)

print("\nLong:")
print(long)
```

# ============================================================

# 29. PIVOT -> MELT ROUND TRIP

# ============================================================

def pivot_melt_round_trip():
"""
Demonstrate a common reshape workflow:

```
    Long
      ↓
    Pivot
      ↓
    Wide
      ↓
    Melt
      ↓
    Long
"""

section("29. PIVOT -> MELT ROUND TRIP")

long_data = pd.DataFrame(
    {
        "student": ["Alice", "Alice", "Bob", "Bob"],
        "subject": ["Math", "Science", "Math", "Science"],
        "score": [90, 85, 80, 88],
    }
)

wide_data = long_data.pivot(
    index="student",
    columns="subject",
    values="score",
)

restored = (
    wide_data
    .reset_index()
    .melt(
        id_vars="student",
        var_name="subject",
        value_name="score",
    )
)

print("Original:")
print(long_data)

print("\nWide:")
print(wide_data)

print("\nRestored long format:")
print(restored)
```

# ============================================================

# 30. STACK

# ============================================================

def stack_example():
"""
stack() moves column levels into the row index.
"""

```
section("30. STACK")

df = pd.DataFrame(
    {
        "Math": [90, 80],
        "Science": [85, 88],
    },
    index=["Alice", "Bob"],
)

print("Original:")
print(df)

result = df.stack()

print("\nStacked:")
print(result)
```

# ============================================================

# 31. UNSTACK

# ============================================================

def unstack_example():
"""
unstack() moves index levels into columns.
"""

```
section("31. UNSTACK")

df = pd.DataFrame(
    {
        "score": [90, 85, 80, 88],
    },
    index=pd.MultiIndex.from_tuples(
        [
            ("Alice", "Math"),
            ("Alice", "Science"),
            ("Bob", "Math"),
            ("Bob", "Science"),
        ],
        names=["student", "subject"],
    ),
)

print("Original:")
print(df)

result = df.unstack()

print("\nUnstacked:")
print(result)
```

# ============================================================

# 32. PIVOT WITH DATETIME

# ============================================================

def datetime_pivot():
"""
Pivot data using dates.

```
Useful for:
    - daily sales
    - monthly revenue
    - customer activity
    - time-series features
"""

section("32. DATETIME PIVOT")

df = pd.DataFrame(
    {
        "date": pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-01",
                "2025-01-02",
                "2025-01-02",
                "2025-01-03",
                "2025-01-03",
            ]
        ),
        "product": [
            "Laptop",
            "Phone",
            "Laptop",
            "Phone",
            "Laptop",
            "Phone",
        ],
        "sales": [
            1000,
            700,
            1200,
            800,
            1100,
            950,
        ],
    }
)

result = df.pivot_table(
    index="date",
    columns="product",
    values="sales",
    aggfunc="sum",
    fill_value=0,
)

print(result)
```

# ============================================================

# 33. MONTHLY PIVOT

# ============================================================

def monthly_pivot():
"""
Create month-based pivot features.
"""

```
section("33. MONTHLY PIVOT")

df = pd.DataFrame(
    {
        "date": pd.to_datetime(
            [
                "2025-01-10",
                "2025-01-20",
                "2025-02-10",
                "2025-02-20",
                "2025-03-10",
                "2025-03-20",
            ]
        ),
        "region": [
            "West",
            "East",
            "West",
            "East",
            "West",
            "East",
        ],
        "sales": [
            1000,
            1200,
            1400,
            1300,
            1600,
            1500,
        ],
    }
)

df["month"] = df["date"].dt.to_period("M")

result = df.pivot_table(
    index="region",
    columns="month",
    values="sales",
    aggfunc="sum",
    fill_value=0,
)

print(result)
```

# ============================================================

# 34. CROSSTAB

# ============================================================

def crosstab_example():
"""
pd.crosstab() is useful for frequency tables.

```
It is related to pivot_table() but specialized for
counting combinations.
"""

section("34. CROSSTAB")

df = pd.DataFrame(
    {
        "gender": [
            "Female",
            "Male",
            "Female",
            "Male",
            "Female",
        ],
        "department": [
            "Engineering",
            "Engineering",
            "Marketing",
            "Marketing",
            "Engineering",
        ],
    }
)

result = pd.crosstab(
    df["gender"],
    df["department"],
)

print(result)
```

# ============================================================

# 35. CROSSTAB WITH NORMALIZATION

# ============================================================

def crosstab_normalized():
"""
Normalize a crosstab to percentages.
"""

```
section("35. CROSSTAB NORMALIZED")

df = pd.DataFrame(
    {
        "gender": [
            "Female",
            "Male",
            "Female",
            "Male",
            "Female",
            "Male",
        ],
        "department": [
            "Engineering",
            "Engineering",
            "Marketing",
            "Marketing",
            "Engineering",
            "Marketing",
        ],
    }
)

result = pd.crosstab(
    df["gender"],
    df["department"],
    normalize="index",
)

print(result)
```

# ============================================================

# 36. PIVOT FOR CUSTOMER FEATURES

# ============================================================

def customer_feature_pivot():
"""
Transform transaction-level data into customer-level features.

```
This is a common Machine Learning feature-engineering task.
"""

section("36. CUSTOMER FEATURE PIVOT")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "category": [
            "Electronics",
            "Books",
            "Electronics",
            "Books",
            "Home",
            "Electronics",
            "Home",
            "Books",
        ],
        "amount": [
            1000,
            300,
            500,
            250,
            600,
            1200,
            400,
            350,
        ],
    }
)

features = transactions.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="sum",
    fill_value=0,
)

print(features)
```

# ============================================================

# 37. TRANSACTION COUNT FEATURES

# ============================================================

def transaction_count_features():
"""
Create category-level transaction count features.
"""

```
section("37. TRANSACTION COUNT FEATURES")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "category": [
            "Electronics",
            "Books",
            "Electronics",
            "Books",
            "Home",
            "Electronics",
            "Home",
            "Books",
        ],
        "amount": [
            1000,
            300,
            500,
            250,
            600,
            1200,
            400,
            350,
        ],
    }
)

features = transactions.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="count",
    fill_value=0,
)

features.columns = [
    f"{column.lower()}_transaction_count"
    for column in features.columns
]

print(features)
```

# ============================================================

# 38. AVERAGE SPEND FEATURES

# ============================================================

def average_spend_features():
"""
Create category-level average spending features.
"""

```
section("38. AVERAGE SPEND FEATURES")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "category": [
            "Electronics",
            "Books",
            "Electronics",
            "Books",
            "Home",
            "Electronics",
            "Home",
            "Books",
        ],
        "amount": [
            1000,
            300,
            500,
            250,
            600,
            1200,
            400,
            350,
        ],
    }
)

features = transactions.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="mean",
    fill_value=0,
)

features.columns = [
    f"{column.lower()}_avg_spend"
    for column in features.columns
]

print(features)
```

# ============================================================

# 39. MULTIPLE ML FEATURES

# ============================================================

def multiple_ml_features():
"""
Create multiple feature types from transaction data.
"""

```
section("39. MULTIPLE ML FEATURES")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "category": [
            "Electronics",
            "Books",
            "Electronics",
            "Books",
            "Home",
            "Electronics",
            "Home",
            "Books",
        ],
        "amount": [
            1000,
            300,
            500,
            250,
            600,
            1200,
            400,
            350,
        ],
    }
)

total_spend = transactions.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="sum",
    fill_value=0,
)

transaction_count = transactions.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="count",
    fill_value=0,
)

total_spend.columns = [
    f"{column.lower()}_total_spend"
    for column in total_spend.columns
]

transaction_count.columns = [
    f"{column.lower()}_count"
    for column in transaction_count.columns
]

features = total_spend.join(transaction_count)

print(features)
```

# ============================================================

# 40. PIVOT AND MACHINE LEARNING TARGET

# ============================================================

def pivot_with_target():
"""
Combine pivot-generated features with a target variable.
"""

```
section("40. PIVOT WITH TARGET")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 1, 2, 2, 3, 3],
        "category": [
            "Electronics",
            "Books",
            "Electronics",
            "Books",
            "Electronics",
            "Books",
        ],
        "amount": [
            1000,
            300,
            500,
            250,
            1200,
            400,
        ],
    }
)

features = transactions.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="sum",
    fill_value=0,
)

targets = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "churn": [0, 1, 0],
    }
).set_index("customer_id")

dataset = features.join(targets)

print(dataset)

X = dataset.drop(columns="churn")
y = dataset["churn"]

print("\nX:")
print(X)

print("\ny:")
print(y)
```

# ============================================================

# 41. DATA LEAKAGE WARNING

# ============================================================

def data_leakage_warning():
"""
Explain temporal leakage when generating pivot features.
"""

```
section("41. DATA LEAKAGE WARNING")

print(
    """
Pivoting is often used for Machine Learning feature
engineering.

However, aggregation can accidentally include future data.

Example:

    Prediction date:
        2025-01-10

    Transaction history:
        2025-01-01 -> 2025-01-31

If you calculate:

    total_spend = sum(all January transactions)

then transactions after January 10 are included.

Those transactions were not available at prediction time.

Correct approach:

    1. Define the prediction timestamp.
    2. Filter records available before that timestamp.
    3. Aggregate only historical records.
    4. Generate features.
    5. Join the target.
    6. Split data appropriately.

Important:

    A mathematically correct pivot can still create an
    invalid Machine Learning feature.
"""
)
```

# ============================================================

# 42. TIME-AWARE FEATURE PIVOT

# ============================================================

def time_aware_feature_pivot():
"""
Demonstrate a simple time-aware aggregation.

```
Only transactions before the cutoff are used.
"""

section("42. TIME-AWARE FEATURE PIVOT")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 1, 1, 2, 2, 3],
        "date": pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-05",
                "2025-01-15",
                "2025-01-03",
                "2025-01-20",
                "2025-01-08",
            ]
        ),
        "category": [
            "Books",
            "Electronics",
            "Books",
            "Home",
            "Electronics",
            "Books",
        ],
        "amount": [
            300,
            1000,
            500,
            600,
            800,
            350,
        ],
    }
)

cutoff = pd.Timestamp("2025-01-10")

historical = transactions[
    transactions["date"] < cutoff
].copy()

features = historical.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="sum",
    fill_value=0,
)

print("Historical records:")
print(historical)

print("\nFeatures generated before cutoff:")
print(features)
```

# ============================================================

# 43. PIVOT QUALITY CHECK

# ============================================================

def pivot_quality_check():
"""
Validate a feature table after pivoting.
"""

```
section("43. PIVOT QUALITY CHECK")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 1, 2, 2, 3],
        "category": [
            "Books",
            "Electronics",
            "Books",
            "Home",
            "Electronics",
        ],
        "amount": [
            300,
            1000,
            250,
            600,
            1200,
        ],
    }
)

features = transactions.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="sum",
    fill_value=0,
)

print("Feature table:")
print(features)

print("\nShape:")
print(features.shape)

print("\nMissing values:")
print(features.isna().sum())

print("\nDuplicate customer IDs:")
print(features.index.duplicated().sum())

print("\nNumeric columns:")
print(features.dtypes)
```

# ============================================================

# 44. REUSABLE PIVOT FUNCTION

# ============================================================

def create_pivot_features(
df: pd.DataFrame,
index: str,
columns: str,
values: str,
aggfunc="sum",
fill_value=0,
) -> pd.DataFrame:
"""
Create a reusable pivot-based feature table.

```
Parameters
----------
df:
    Source DataFrame.

index:
    Column identifying entities.

columns:
    Column whose unique values become feature columns.

values:
    Numerical column to aggregate.

aggfunc:
    Aggregation function.

fill_value:
    Value used for missing combinations.

Returns
-------
pd.DataFrame
    Pivoted feature DataFrame.
"""

required_columns = {index, columns, values}

missing = required_columns.difference(df.columns)

if missing:
    raise KeyError(
        f"Missing required columns: {sorted(missing)}"
    )

result = df.pivot_table(
    index=index,
    columns=columns,
    values=values,
    aggfunc=aggfunc,
    fill_value=fill_value,
)

return result
```

# ============================================================

# 45. REUSABLE FUNCTION DEMO

# ============================================================

def reusable_function_demo():
"""
Demonstrate the reusable pivot feature function.
"""

```
section("45. REUSABLE FUNCTION DEMO")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 1, 2, 2, 3],
        "category": [
            "Books",
            "Electronics",
            "Books",
            "Home",
            "Electronics",
        ],
        "amount": [
            300,
            1000,
            250,
            600,
            1200,
        ],
    }
)

result = create_pivot_features(
    transactions,
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="sum",
    fill_value=0,
)

print(result)
```

# ============================================================

# 46. COMMON PIVOT MISTAKES

# ============================================================

def common_mistakes():
"""
Display common pivot and reshape mistakes.
"""

```
section("46. COMMON PIVOT MISTAKES")

print(
    """
1. USING pivot() WITH DUPLICATE COMBINATIONS

   Problem:
       Multiple rows have the same index/column combination.

   Solution:
       Use pivot_table() with an aggregation function.


2. IGNORING MISSING COMBINATIONS

   Problem:
       Some entity/category combinations do not exist.

   Solution:
       Use fill_value=0 when zero is semantically correct.


3. CONFUSING SUM AND MEAN

   Problem:
       The wrong aggregation function changes the meaning
       of the feature.

   Solution:
       Choose aggregation based on the business question.


4. CREATING TOO MANY FEATURES

   Problem:
       High-cardinality categories create thousands of columns.

   Solution:
       Group rare categories or use a more appropriate
       encoding strategy.


5. FORGETTING THE INDEX

   Problem:
       The entity identifier remains hidden in the index.

   Solution:
       Use reset_index() when a normal column is required.


6. MULTIINDEX COLUMNS

   Problem:
       pivot_table() produces hierarchical column labels.

   Solution:
       Keep them when useful or flatten them carefully.


7. DATA LEAKAGE

   Problem:
       Future observations are included in aggregated features.

   Solution:
       Apply time-based filtering before pivoting.


8. USING PIVOT WHEN GROUPBY IS CLEARER

   Problem:
       Complex aggregation logic becomes difficult to read.

   Solution:
       Consider groupby().agg() for highly customized
       feature engineering.


9. FILLING MISSING VALUES WITHOUT THINKING

   Problem:
       Missing may mean "unknown", not zero.

   Solution:
       Understand the meaning of missing data before
       replacing it.


10. IGNORING CATEGORY CARDINALITY

    Problem:
        Pivoting a high-cardinality feature creates an
        extremely wide dataset.

    Solution:
        Inspect the number of unique categories first.
"""
)
```

# ============================================================

# 47. PIVOT CHEAT SHEET

# ============================================================

def pivot_cheat_sheet():
"""
Display a concise Pandas reshaping reference.
"""

```
section("47. PIVOT CHEAT SHEET")

cheat_sheet = pd.DataFrame(
    {
        "Operation": [
            "Basic pivot",
            "Pivot table",
            "Sum",
            "Mean",
            "Multiple aggregations",
            "Fill missing",
            "Totals",
            "Melt",
            "Stack",
            "Unstack",
            "Cross-tabulation",
        ],
        "Syntax": [
            "df.pivot(index='A', columns='B', values='C')",
            "df.pivot_table(index='A', columns='B', values='C')",
            "aggfunc='sum'",
            "aggfunc='mean'",
            "aggfunc=['sum', 'mean']",
            "fill_value=0",
            "margins=True",
            "df.melt(...)",
            "df.stack()",
            "df.unstack()",
            "pd.crosstab(...)",
        ],
    }
)

print(cheat_sheet.to_string(index=False))
```

# ============================================================

# 48. COMPLETE REAL-WORLD WORKFLOW

# ============================================================

def complete_pivot_workflow():
"""
Demonstrate a production-style pivot workflow.

```
Workflow:

    Raw transactions
          ↓
    Validate columns
          ↓
    Clean data
          ↓
    Filter valid time window
          ↓
    Aggregate
          ↓
    Pivot
          ↓
    Fill missing combinations
          ↓
    Flatten columns if needed
          ↓
    Validate features
          ↓
    Join target
          ↓
    Train ML model
"""

section("48. COMPLETE REAL-WORLD WORKFLOW")

transactions = pd.DataFrame(
    {
        "customer_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "date": pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-05",
                "2025-01-08",
                "2025-01-02",
                "2025-01-07",
                "2025-01-03",
                "2025-01-06",
                "2025-01-09",
            ]
        ),
        "category": [
            "Books",
            "Electronics",
            "Books",
            "Home",
            "Books",
            "Electronics",
            "Home",
            "Books",
        ],
        "amount": [
            300,
            1000,
            250,
            600,
            250,
            1200,
            400,
            350,
        ],
    }
)

# --------------------------------------------------------
# Step 1: Inspect
# --------------------------------------------------------

print("Raw transaction data:")
print(transactions)

# --------------------------------------------------------
# Step 2: Validate
# --------------------------------------------------------

print("\nData types:")
print(transactions.dtypes)

print("\nMissing values:")
print(transactions.isna().sum())

# --------------------------------------------------------
# Step 3: Define cutoff
# --------------------------------------------------------

cutoff = pd.Timestamp("2025-01-10")

historical = transactions[
    transactions["date"] < cutoff
].copy()

# --------------------------------------------------------
# Step 4: Pivot
# --------------------------------------------------------

features = historical.pivot_table(
    index="customer_id",
    columns="category",
    values="amount",
    aggfunc="sum",
    fill_value=0,
)

# --------------------------------------------------------
# Step 5: Flatten columns
# --------------------------------------------------------

features.columns = [
    f"{str(column).lower()}_total_spend"
    for column in features.columns
]

# --------------------------------------------------------
# Step 6: Reset index
# --------------------------------------------------------

features = features.reset_index()

print("\nGenerated feature table:")
print(features)

# --------------------------------------------------------
# Step 7: Validate
# --------------------------------------------------------

print("\nFeature shape:")
print(features.shape)

print("\nMissing values:")
print(features.isna().sum())

print("\nDuplicate customers:")
print(features["customer_id"].duplicated().sum())

# --------------------------------------------------------
# Step 8: Target
# --------------------------------------------------------

target = pd.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "churn": [0, 1, 0],
    }
)

# --------------------------------------------------------
# Step 9: Join target
# --------------------------------------------------------

dataset = features.merge(
    target,
    on="customer_id",
    how="left",
    validate="one_to_one",
)

print("\nFinal ML dataset:")
print(dataset)

# --------------------------------------------------------
# Step 10: Separate X and y
# --------------------------------------------------------

X = dataset.drop(
    columns=["customer_id", "churn"]
)

y = dataset["churn"]

print("\nX:")
print(X)

print("\ny:")
print(y)
```

# ============================================================

# 49. MAIN FUNCTION

# ============================================================

def main():
"""
Run the complete Pandas pivot tutorial.
"""

```
print("=" * 80)
print("PANDAS PIVOT & DATA RESHAPING TUTORIAL")
print("=" * 80)

# --------------------------------------------------------
# Fundamentals
# --------------------------------------------------------

long_vs_wide()
basic_pivot()
pivot_index()
pivot_columns()
pivot_values()
multiple_values_pivot()

# --------------------------------------------------------
# Pivot tables
# --------------------------------------------------------

basic_pivot_table()
duplicate_pivot_problem()
pivot_sum()
pivot_mean()
pivot_count()
pivot_min_max()
multiple_aggregation_functions()
multiple_value_columns()
multiple_index_levels()
multiple_column_levels()
fill_missing_pivot_values()
pivot_dropna()
pivot_margins()
sort_pivot_table()
custom_aggregation()
named_aggregation()

# --------------------------------------------------------
# Reshaping
# --------------------------------------------------------

pivot_reset_index()
flatten_multiindex_columns()
basic_melt()
pivot_melt_round_trip()
stack_example()
unstack_example()

# --------------------------------------------------------
# Time-based analysis
# --------------------------------------------------------

datetime_pivot()
monthly_pivot()

# --------------------------------------------------------
# Frequency tables
# --------------------------------------------------------

crosstab_example()
crosstab_normalized()

# --------------------------------------------------------
# Machine Learning feature engineering
# --------------------------------------------------------

customer_feature_pivot()
transaction_count_features()
average_spend_features()
multiple_ml_features()
pivot_with_target()
data_leakage_warning()
time_aware_feature_pivot()
pivot_quality_check()

# --------------------------------------------------------
# Reusable functions
# --------------------------------------------------------

reusable_function_demo()

# --------------------------------------------------------
# Reference
# --------------------------------------------------------

common_mistakes()
pivot_cheat_sheet()

# --------------------------------------------------------
# Complete workflow
# --------------------------------------------------------

complete_pivot_workflow()

section("TUTORIAL COMPLETE")

print(
    """
Key Takeaways:

1. pivot() reshapes data without aggregation.
2. pivot() requires unique index/column combinations.
3. pivot_table() supports aggregation.
4. Use sum, mean, count, min, max, or custom functions
   depending on the problem.
5. fill_value can replace missing combinations.
6. margins=True adds totals.
7. melt() converts wide data back into long format.
8. stack() and unstack() manipulate MultiIndex structures.
9. crosstab() is useful for frequency tables.
10. Pivot tables are powerful for Machine Learning
    feature engineering.
11. Always control the time window when generating
    historical ML features.
12. Validate the resulting feature table before training.
13. Avoid creating extremely wide datasets from
    high-cardinality categories.

Remember:

    Long Data
         ↓
    Group / Aggregate
         ↓
    Pivot
         ↓
    Feature Table
         ↓
    Validate
         ↓
    Machine Learning
"""
)
```

# ============================================================

# 50. SCRIPT ENTRY POINT

# ============================================================

if **name** == "**main**":
main()

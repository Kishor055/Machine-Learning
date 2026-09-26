"""
Pandas Indexing — Beginner to Advanced
======================================

File:
03-Python-for-Machine-Learning/02-Pandas/indexing.py

Description:
A comprehensive practical guide to indexing and selecting data
with Pandas.

Topics Covered:
1. Understanding the Pandas Index
2. Creating a DataFrame
3. Default RangeIndex
4. Custom indexes
5. Accessing the index
6. Selecting columns
7. Selecting rows with []
8. Selecting rows with loc[]
9. Selecting rows with iloc[]
10. loc vs iloc
11. Selecting individual values with at[]
12. Selecting individual values with iat[]
13. Boolean indexing
14. Multiple conditions
15. Slicing rows
16. Slicing columns
17. Selecting specific rows and columns
18. Setting values with loc
19. Setting values with iloc
20. Index replacement
21. reset_index()
22. set_index()
23. reindex()
24. Sorting by index
25. Dropping index labels
26. Duplicate indexes
27. MultiIndex
28. Selecting from MultiIndex
29. Index levels
30. DateTimeIndex
31. Time-based indexing
32. Index membership
33. Callable indexing
34. Indexing after filtering
35. Copy vs view
36. ML-oriented indexing
37. Avoiding data leakage
38. Common mistakes
39. Indexing cheat sheet
40. Complete workflow

Requirements:
pip install pandas numpy

Run:
python indexing.py

Author:
Kishor Patil

Core Concepts:
loc[]
Label-based indexing.

```
iloc[]
    Integer-position-based indexing.

at[]
    Fast scalar access by label.

iat[]
    Fast scalar access by integer position.
```

Important:
loc and iloc are not interchangeable.

```
loc uses labels.
iloc uses integer positions.
```

"""

# =============================================================================

# 1. IMPORT LIBRARIES

# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================

# 2. CREATE SAMPLE DATA

# =============================================================================

def create_sample_data() -> pd.DataFrame:
"""
Create a sample employee DataFrame.

```
Returns:
    pd.DataFrame:
        Employee dataset.
"""

data = {
    "employee_id": range(101, 111),
    "name": [
        "Aarav",
        "Priya",
        "Rahul",
        "Sneha",
        "Vikram",
        "Ananya",
        "Rohan",
        "Neha",
        "Karan",
        "Meera",
    ],
    "department": [
        "IT",
        "HR",
        "IT",
        "Finance",
        "IT",
        "Marketing",
        "Finance",
        "IT",
        "HR",
        "Marketing",
    ],
    "city": [
        "Pune",
        "Mumbai",
        "Pune",
        "Nashik",
        "Mumbai",
        "Pune",
        "Nashik",
        "Mumbai",
        "Pune",
        "Nagpur",
    ],
    "age": [
        24,
        29,
        35,
        42,
        27,
        31,
        38,
        np.nan,
        45,
        26,
    ],
    "salary": [
        65000,
        55000,
        90000,
        75000,
        62000,
        70000,
        85000,
        68000,
        95000,
        58000,
    ],
    "experience": [
        1,
        3,
        8,
        10,
        2,
        5,
        7,
        4,
        12,
        2,
    ],
    "performance_score": [
        82,
        75,
        91,
        88,
        79,
        95,
        86,
        72,
        94,
        80,
    ],
    "is_remote": [
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
        False,
        False,
    ],
}

return pd.DataFrame(data)
```

# =============================================================================

# 3. DISPLAY DATAFRAME

# =============================================================================

def display_dataframe(df: pd.DataFrame) -> None:
"""Display the sample DataFrame and its index."""

```
print("\n" + "=" * 80)
print("DATAFRAME")
print("=" * 80)

print(df.to_string())

print("\nIndex:")
print(df.index)

print("\nIndex type:")
print(type(df.index).__name__)
```

# =============================================================================

# 4. UNDERSTANDING THE INDEX

# =============================================================================

def explain_index() -> None:
"""Explain the purpose of the Pandas index."""

```
print("\n" + "=" * 80)
print("4. UNDERSTANDING THE PANDAS INDEX")
print("=" * 80)

print(
    """
```

The Pandas index identifies rows in a Series or DataFrame.

Example:

```
    name       salary
0   Aarav      65000
1   Priya      55000
2   Rahul      90000
```

Here:

```
0, 1, 2
```

are index labels.

The index can be:

```
- RangeIndex
- integer labels
- string labels
- dates
- MultiIndex
- custom identifiers
```

Important:

```
An index is not necessarily the same thing as a row number.
```

For example:

```
index = [101, 205, 310]
```

The second row has:

```
position = 1
label = 205
```

This distinction is why loc[] and iloc[] behave differently.
"""
)

# =============================================================================

# 5. RANGEINDEX

# =============================================================================

def range_index_example(df: pd.DataFrame) -> None:
"""Demonstrate the default RangeIndex."""

```
print("\n" + "=" * 80)
print("5. RANGEINDEX")
print("=" * 80)

print("Index:")
print(df.index)

print("\nStart:")
print(df.index.start)

print("\nStop:")
print(df.index.stop)

print("\nStep:")
print(df.index.step)
```

# =============================================================================

# 6. ACCESSING INDEX

# =============================================================================

def accessing_index(df: pd.DataFrame) -> None:
"""Demonstrate index-related properties."""

```
print("\n" + "=" * 80)
print("6. ACCESSING THE INDEX")
print("=" * 80)

print("Index values:")
print(df.index.to_list())

print("\nFirst index label:")
print(df.index[0])

print("\nLast index label:")
print(df.index[-1])

print("\nIndex name:")
print(df.index.name)
```

# =============================================================================

# 7. COLUMN SELECTION

# =============================================================================

def column_indexing(df: pd.DataFrame) -> None:
"""Demonstrate basic column selection."""

```
print("\n" + "=" * 80)
print("7. COLUMN INDEXING")
print("=" * 80)

# Select one column.
salary = df["salary"]

print("\nSingle column:")
print(salary)

# Select multiple columns.
selected = df[
    [
        "name",
        "department",
        "salary",
    ]
]

print("\nMultiple columns:")
print(selected.to_string(index=False))
```

# =============================================================================

# 8. ROW SELECTION USING []

# =============================================================================

def row_selection_brackets(df: pd.DataFrame) -> None:
"""Demonstrate slicing rows using DataFrame brackets."""

```
print("\n" + "=" * 80)
print("8. ROW SELECTION USING []")
print("=" * 80)

first_five = df[:5]

print("\nFirst five rows:")
print(first_five.to_string(index=False))

middle_rows = df[2:7]

print("\nRows 2 through 6:")
print(middle_rows.to_string(index=False))
```

# =============================================================================

# 9. LOC — LABEL BASED INDEXING

# =============================================================================

def loc_basic(df: pd.DataFrame) -> None:
"""
Demonstrate basic loc[] operations.

```
loc is label-based.
"""

print("\n" + "=" * 80)
print("9. LOC — LABEL-BASED INDEXING")
print("=" * 80)

# Row with label 2.
row = df.loc[2]

print("\nRow with index label 2:")
print(row)

# Multiple labels.
rows = df.loc[
    [1, 3, 5]
]

print("\nRows with labels 1, 3, 5:")
print(rows.to_string(index=False))
```

# =============================================================================

# 10. LOC RANGE SLICING

# =============================================================================

def loc_slicing(df: pd.DataFrame) -> None:
"""
Demonstrate loc slicing.

```
Important:
    loc label-based slices are inclusive of the end label.
"""

print("\n" + "=" * 80)
print("10. LOC SLICING")
print("=" * 80)

result = df.loc[2:5]

print("\nloc[2:5]:")
print(result.to_string(index=False))

print(
    "\nNotice: both labels 2 and 5 are included."
)
```

# =============================================================================

# 11. ILOC — POSITION BASED INDEXING

# =============================================================================

def iloc_basic(df: pd.DataFrame) -> None:
"""
Demonstrate iloc[].

```
iloc is integer-position-based.
"""

print("\n" + "=" * 80)
print("11. ILOC — POSITION-BASED INDEXING")
print("=" * 80)

# First row.
first_row = df.iloc[0]

print("\nFirst row:")
print(first_row)

# Third row.
third_row = df.iloc[2]

print("\nThird row:")
print(third_row)

# Multiple positions.
rows = df.iloc[
    [0, 2, 4]
]

print("\nRows at positions 0, 2, 4:")
print(rows.to_string(index=False))
```

# =============================================================================

# 12. ILOC SLICING

# =============================================================================

def iloc_slicing(df: pd.DataFrame) -> None:
"""
Demonstrate iloc slicing.

```
Unlike loc, iloc uses Python-style exclusive end positions.
"""

print("\n" + "=" * 80)
print("12. ILOC SLICING")
print("=" * 80)

result = df.iloc[2:5]

print("\niloc[2:5]:")
print(result.to_string(index=False))

print(
    "\nPositions 2, 3, and 4 are selected; position 5 is excluded."
)
```

# =============================================================================

# 13. LOC VS ILOC

# =============================================================================

def loc_vs_iloc(df: pd.DataFrame) -> None:
"""Compare label-based and position-based indexing."""

```
print("\n" + "=" * 80)
print("13. LOC VS ILOC")
print("=" * 80)

print(
    """
```

loc:
Uses labels.

```
df.loc[2]
```

iloc:
Uses integer positions.

```
df.iloc[2]
```

With a default index:

```
label 2
position 2
```

may refer to the same row.

But with a custom index:

```
labels = [101, 205, 310]
```

then:

```
df.loc[205]
    means label 205.

df.iloc[1]
    means second row.
```

Therefore:

```
loc  -> label
iloc -> position
```

"""
)

```
custom_df = df.set_index("employee_id")

print("\nCustom-index DataFrame:")
print(
    custom_df[
        ["name", "salary"]
    ].to_string()
)

print("\nloc[105] — employee ID 105:")
print(custom_df.loc[105])

print("\niloc[4] — fifth row:")
print(custom_df.iloc[4])
```

# =============================================================================

# 14. LOC WITH COLUMNS

# =============================================================================

def loc_rows_and_columns(df: pd.DataFrame) -> None:
"""Select rows and columns using loc."""

```
print("\n" + "=" * 80)
print("14. LOC WITH ROWS AND COLUMNS")
print("=" * 80)

result = df.loc[
    2:6,
    [
        "name",
        "department",
        "salary",
    ],
]

print(result.to_string(index=False))
```

# =============================================================================

# 15. ILOC WITH ROWS AND COLUMNS

# =============================================================================

def iloc_rows_and_columns(df: pd.DataFrame) -> None:
"""Select rows and columns using iloc."""

```
print("\n" + "=" * 80)
print("15. ILOC WITH ROWS AND COLUMNS")
print("=" * 80)

result = df.iloc[
    2:7,
    0:4,
]

print(result.to_string(index=False))
```

# =============================================================================

# 16. AT — SCALAR LABEL ACCESS

# =============================================================================

def at_example(df: pd.DataFrame) -> None:
"""
Demonstrate at[].

```
at[] is optimized for accessing a single scalar value
using row and column labels.
"""

print("\n" + "=" * 80)
print("16. AT — SCALAR LABEL ACCESS")
print("=" * 80)

value = df.at[
    2,
    "salary",
]

print("\nSalary at row label 2:")
print(value)
```

# =============================================================================

# 17. IAT — SCALAR POSITION ACCESS

# =============================================================================

def iat_example(df: pd.DataFrame) -> None:
"""
Demonstrate iat[].

```
iat[] accesses one scalar value by integer position.
"""

print("\n" + "=" * 80)
print("17. IAT — SCALAR POSITION ACCESS")
print("=" * 80)

value = df.iat[
    2,
    5,
]

print("\nValue at row position 2, column position 5:")
print(value)
```

# =============================================================================

# 18. BOOLEAN INDEXING

# =============================================================================

def boolean_indexing(df: pd.DataFrame) -> None:
"""Filter rows using a Boolean condition."""

```
print("\n" + "=" * 80)
print("18. BOOLEAN INDEXING")
print("=" * 80)

condition = df["salary"] > 70000

result = df.loc[
    condition
]

print("\nSalary > 70,000:")
print(
    result[
        ["name", "salary"]
    ].to_string(index=False)
)
```

# =============================================================================

# 19. MULTIPLE BOOLEAN CONDITIONS

# =============================================================================

def multiple_boolean_conditions(
df: pd.DataFrame,
) -> None:
"""Use AND, OR, and NOT with loc."""

```
print("\n" + "=" * 80)
print("19. MULTIPLE BOOLEAN CONDITIONS")
print("=" * 80)

result = df.loc[
    (df["salary"] > 70000)
    & (df["performance_score"] >= 85)
]

print("\nSalary > 70K AND performance >= 85:")
print(
    result[
        [
            "name",
            "salary",
            "performance_score",
        ]
    ].to_string(index=False)
)

result = df.loc[
    df["department"].isin(
        ["IT", "Finance"]
    )
]

print("\nIT OR Finance:")
print(
    result[
        ["name", "department"]
    ].to_string(index=False)
)
```

# =============================================================================

# 20. CONDITIONAL INDEXING WITH LOC

# =============================================================================

def conditional_loc(
df: pd.DataFrame,
) -> None:
"""Use loc for both filtering and selecting columns."""

```
print("\n" + "=" * 80)
print("20. CONDITIONAL LOC")
print("=" * 80)

result = df.loc[
    df["experience"] >= 5,
    [
        "name",
        "experience",
        "salary",
    ],
]

print("\nEmployees with >= 5 years experience:")
print(result.to_string(index=False))
```

# =============================================================================

# 21. SET VALUES WITH LOC

# =============================================================================

def set_values_with_loc(
df: pd.DataFrame,
) -> None:
"""
Demonstrate safe conditional assignment.

```
.loc is the preferred approach for conditional updates.
"""

print("\n" + "=" * 80)
print("21. SET VALUES WITH LOC")
print("=" * 80)

result = df.copy()

result.loc[
    result["salary"] >= 90000,
    "salary_level",
] = "High"

result.loc[
    result["salary"] < 90000,
    "salary_level",
] = "Standard"

print(
    result[
        [
            "name",
            "salary",
            "salary_level",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 22. SET VALUES WITH ILOC

# =============================================================================

def set_values_with_iloc(
df: pd.DataFrame,
) -> None:
"""Demonstrate position-based assignment with iloc."""

```
print("\n" + "=" * 80)
print("22. SET VALUES WITH ILOC")
print("=" * 80)

result = df.copy()

salary_column_position = result.columns.get_loc(
    "salary"
)

result.iloc[
    0,
    salary_column_position,
] = 70000

print(
    result[
        [
            "name",
            "salary",
        ]
    ].head().to_string(index=False)
)
```

# =============================================================================

# 23. SET SCALAR VALUES WITH AT

# =============================================================================

def set_scalar_with_at(
df: pd.DataFrame,
) -> None:
"""Update one scalar value using at[]."""

```
print("\n" + "=" * 80)
print("23. SET SCALAR VALUE WITH AT")
print("=" * 80)

result = df.copy()

result.at[
    0,
    "salary",
] = 72000

print(
    result.loc[
        0,
        [
            "name",
            "salary",
        ],
    ]
)
```

# =============================================================================

# 24. SET SCALAR VALUES WITH IAT

# =============================================================================

def set_scalar_with_iat(
df: pd.DataFrame,
) -> None:
"""Update one scalar value using iat[]."""

```
print("\n" + "=" * 80)
print("24. SET SCALAR VALUE WITH IAT")
print("=" * 80)

result = df.copy()

salary_column_position = result.columns.get_loc(
    "salary"
)

result.iat[
    0,
    salary_column_position,
] = 73000

print(
    result.loc[
        0,
        [
            "name",
            "salary",
        ],
    ]
)
```

# =============================================================================

# 25. SET CUSTOM INDEX

# =============================================================================

def set_custom_index(
df: pd.DataFrame,
) -> None:
"""Use employee_id as the DataFrame index."""

```
print("\n" + "=" * 80)
print("25. SET_CUSTOM_INDEX")
print("=" * 80)

result = df.set_index(
    "employee_id"
)

print(result.to_string())
```

# =============================================================================

# 26. SET INDEX WITHOUT REMOVING COLUMN

# =============================================================================

def set_index_keep_column(
df: pd.DataFrame,
) -> None:
"""Demonstrate keeping the source column."""

```
print("\n" + "=" * 80)
print("26. SET INDEX — KEEP COLUMN")
print("=" * 80)

result = df.set_index(
    "employee_id",
    drop=False,
)

print(
    result[
        [
            "employee_id",
            "name",
            "salary",
        ]
    ].head().to_string()
)
```

# =============================================================================

# 27. RESET INDEX

# =============================================================================

def reset_index_example(
df: pd.DataFrame,
) -> None:
"""Demonstrate reset_index()."""

```
print("\n" + "=" * 80)
print("27. RESET_INDEX")
print("=" * 80)

indexed = df.set_index(
    "employee_id"
)

print("\nCustom index:")
print(indexed.head())

reset = indexed.reset_index()

print("\nAfter reset_index():")
print(reset.head())
```

# =============================================================================

# 28. RESET INDEX DROP

# =============================================================================

def reset_index_drop(
df: pd.DataFrame,
) -> None:
"""Reset index while discarding the old index."""

```
print("\n" + "=" * 80)
print("28. RESET_INDEX DROP")
print("=" * 80)

indexed = df.set_index(
    "employee_id"
)

result = indexed.reset_index(
    drop=True
)

print(result.head())
```

# =============================================================================

# 29. REINDEX

# =============================================================================

def reindex_example(
df: pd.DataFrame,
) -> None:
"""
Demonstrate reindex().

```
reindex() conforms a DataFrame to a specified index.
"""

print("\n" + "=" * 80)
print("29. REINDEX")
print("=" * 80)

small_df = df[
    [
        "name",
        "salary",
    ]
].head(5)

small_df.index = [
    101,
    102,
    103,
    104,
    105,
]

desired_index = [
    101,
    102,
    103,
    104,
    105,
    106,
]

result = small_df.reindex(
    desired_index
)

print("\nOriginal:")
print(small_df)

print("\nAfter reindex:")
print(result)
```

# =============================================================================

# 30. REINDEX COLUMNS

# =============================================================================

def reindex_columns(
df: pd.DataFrame,
) -> None:
"""Reorder and select columns using reindex()."""

```
print("\n" + "=" * 80)
print("30. REINDEX COLUMNS")
print("=" * 80)

columns = [
    "name",
    "salary",
    "department",
    "performance_score",
]

result = df.reindex(
    columns=columns
)

print(result.head().to_string(index=False))
```

# =============================================================================

# 31. SORT BY INDEX

# =============================================================================

def sort_index_example(
df: pd.DataFrame,
) -> None:
"""Demonstrate sorting by index."""

```
print("\n" + "=" * 80)
print("31. SORT_INDEX")
print("=" * 80)

custom = df.set_index(
    "employee_id"
)

shuffled = custom.sample(
    frac=1,
    random_state=42,
)

print("\nShuffled index:")
print(
    shuffled[
        ["name", "salary"]
    ].head()
)

sorted_df = shuffled.sort_index()

print("\nSorted index:")
print(
    sorted_df[
        ["name", "salary"]
    ].head()
)
```

# =============================================================================

# 32. DROP INDEX LABELS

# =============================================================================

def drop_index_labels(
df: pd.DataFrame,
) -> None:
"""Remove rows using their index labels."""

```
print("\n" + "=" * 80)
print("32. DROP INDEX LABELS")
print("=" * 80)

result = df.drop(
    index=[1, 3, 5]
)

print(
    result[
        [
            "name",
            "salary",
        ]
    ].to_string()
)
```

# =============================================================================

# 33. DUPLICATE INDEX

# =============================================================================

def duplicate_index_example() -> None:
"""Demonstrate duplicate index labels."""

```
print("\n" + "=" * 80)
print("33. DUPLICATE INDEX")
print("=" * 80)

df = pd.DataFrame(
    {
        "name": [
            "Aarav",
            "Priya",
            "Rahul",
            "Sneha",
        ],
        "salary": [
            65000,
            55000,
            90000,
            75000,
        ],
    },
    index=[
        "A",
        "B",
        "A",
        "C",
    ],
)

print("\nDataFrame with duplicate index labels:")
print(df)

print("\nRows with index label 'A':")
print(df.loc["A"])

print(
    "\nIs index unique?",
    df.index.is_unique,
)

print(
    "\nDuplicate index labels:",
    df.index[df.index.duplicated()].to_list(),
)
```

# =============================================================================

# 34. MULTIINDEX

# =============================================================================

def create_multiindex(
df: pd.DataFrame,
) -> pd.DataFrame:
"""
Create a MultiIndex using department and city.

```
MultiIndex provides hierarchical indexing.
"""

result = df.set_index(
    [
        "department",
        "city",
    ]
)

return result
```

# =============================================================================

# 35. MULTIINDEX BASIC

# =============================================================================

def multiindex_basic(
df: pd.DataFrame,
) -> None:
"""Demonstrate MultiIndex creation and inspection."""

```
print("\n" + "=" * 80)
print("35. MULTIINDEX")
print("=" * 80)

multi = create_multiindex(df)

print("\nMultiIndex DataFrame:")
print(
    multi[
        [
            "name",
            "salary",
        ]
    ].to_string()
)

print("\nIndex:")
print(multi.index)

print("\nIndex names:")
print(multi.index.names)
```

# =============================================================================

# 36. MULTIINDEX SELECTION

# =============================================================================

def multiindex_selection(
df: pd.DataFrame,
) -> None:
"""Select rows from a MultiIndex."""

```
print("\n" + "=" * 80)
print("36. MULTIINDEX SELECTION")
print("=" * 80)

multi = create_multiindex(df)

print("\nAll IT employees:")
print(
    multi.loc[
        "IT"
    ][
        [
            "name",
            "salary",
        ]
    ]
)

print("\nIT employees in Pune:")
print(
    multi.loc[
        ("IT", "Pune")
    ][
        [
            "name",
            "salary",
        ]
    ]
)
```

# =============================================================================

# 37. MULTIINDEX SORTING

# =============================================================================

def multiindex_sorting(
df: pd.DataFrame,
) -> None:
"""Sort a MultiIndex."""

```
print("\n" + "=" * 80)
print("37. MULTIINDEX SORTING")
print("=" * 80)

multi = create_multiindex(df)

sorted_multi = multi.sort_index()

print(
    sorted_multi[
        [
            "name",
            "salary",
        ]
    ].to_string()
)
```

# =============================================================================

# 38. INDEX MEMBERSHIP

# =============================================================================

def index_membership(
df: pd.DataFrame,
) -> None:
"""Check whether labels exist in the index."""

```
print("\n" + "=" * 80)
print("38. INDEX MEMBERSHIP")
print("=" * 80)

print(
    "Does label 3 exist?",
    3 in df.index,
)

print(
    "Does label 100 exist?",
    100 in df.index,
)

custom = df.set_index(
    "employee_id"
)

print(
    "Does employee 105 exist?",
    105 in custom.index,
)
```

# =============================================================================

# 39. INDEX NAME

# =============================================================================

def index_name_example(
df: pd.DataFrame,
) -> None:
"""Set and inspect an index name."""

```
print("\n" + "=" * 80)
print("39. INDEX NAME")
print("=" * 80)

result = df.copy()

result.index.name = "row_id"

print(
    result[
        [
            "name",
            "salary",
        ]
    ].head()
)

print("\nIndex name:")
print(result.index.name)
```

# =============================================================================

# 40. DATETIME INDEX

# =============================================================================

def datetime_index_example(
df: pd.DataFrame,
) -> None:
"""
Demonstrate a DateTimeIndex.

```
This is especially useful for time-series data.
"""

print("\n" + "=" * 80)
print("40. DATETIME INDEX")
print("=" * 80)

result = df.copy()

result["joining_date"] = pd.to_datetime(
    [
        "2021-01-15",
        "2020-06-10",
        "2017-03-20",
        "2015-08-12",
        "2022-01-05",
        "2019-11-18",
        "2018-04-25",
        "2021-07-01",
        "2013-02-14",
        "2022-09-30",
    ]
)

result = result.set_index(
    "joining_date"
).sort_index()

print(
    result[
        [
            "name",
            "department",
            "salary",
        ]
    ].to_string()
)
```

# =============================================================================

# 41. TIME-BASED INDEXING

# =============================================================================

def time_based_indexing(
df: pd.DataFrame,
) -> None:
"""Filter records using a DateTimeIndex."""

```
print("\n" + "=" * 80)
print("41. TIME-BASED INDEXING")
print("=" * 80)

result = df.copy()

result["joining_date"] = pd.to_datetime(
    [
        "2021-01-15",
        "2020-06-10",
        "2017-03-20",
        "2015-08-12",
        "2022-01-05",
        "2019-11-18",
        "2018-04-25",
        "2021-07-01",
        "2013-02-14",
        "2022-09-30",
    ]
)

result = result.set_index(
    "joining_date"
).sort_index()

selected = result.loc[
    "2019-01-01":"2021-12-31"
]

print(
    "\nEmployees who joined between 2019 and 2021:"
)

print(
    selected[
        [
            "name",
            "department",
            "salary",
        ]
    ].to_string()
)
```

# =============================================================================

# 42. CALLABLE INDEXING

# =============================================================================

def callable_indexing(
df: pd.DataFrame,
) -> None:
"""
Demonstrate callable indexing.

```
A callable receives the DataFrame and returns an indexing object.
"""

print("\n" + "=" * 80)
print("42. CALLABLE INDEXING")
print("=" * 80)

result = df.loc[
    lambda data:
    data["salary"] > 70000,
    lambda data:
    [
        "name",
        "department",
        "salary",
    ],
]

print(result.to_string(index=False))
```

# =============================================================================

# 43. INDEXING AFTER FILTERING

# =============================================================================

def indexing_after_filtering(
df: pd.DataFrame,
) -> None:
"""Demonstrate reset_index after filtering."""

```
print("\n" + "=" * 80)
print("43. INDEXING AFTER FILTERING")
print("=" * 80)

filtered = df.loc[
    df["salary"] > 70000
]

print("\nFiltered DataFrame:")
print(
    filtered[
        [
            "name",
            "salary",
        ]
    ]
)

reset = filtered.reset_index(
    drop=True
)

print("\nAfter reset_index(drop=True):")
print(
    reset[
        [
            "name",
            "salary",
        ]
    ]
)
```

# =============================================================================

# 44. COPY VS VIEW

# =============================================================================

def copy_vs_view(
df: pd.DataFrame,
) -> None:
"""
Demonstrate safe copying after filtering.

```
Use .copy() when creating an independent DataFrame that will
later be modified.
"""

print("\n" + "=" * 80)
print("44. COPY VS VIEW")
print("=" * 80)

filtered = df.loc[
    df["salary"] > 70000
].copy()

filtered["bonus"] = (
    filtered["salary"] * 0.10
)

print(
    filtered[
        [
            "name",
            "salary",
            "bonus",
        ]
    ].to_string(index=False)
)

print(
    """
```

Recommended pattern:

```
filtered = df.loc[
    condition
].copy()
```

This makes your intention explicit and avoids many
chained-assignment problems.
"""
)

# =============================================================================

# 45. CHAINED INDEXING WARNING

# =============================================================================

def chained_indexing_warning() -> None:
"""Explain why chained indexing should be avoided."""

```
print("\n" + "=" * 80)
print("45. CHAINED INDEXING WARNING")
print("=" * 80)

print(
    """
```

Avoid patterns like:

```
df[df["salary"] > 70000]["salary"] = 0
```

This is chained indexing.

Prefer:

```
df.loc[
    df["salary"] > 70000,
    "salary",
] = 0
```

For a separate DataFrame:

```
filtered = df.loc[
    df["salary"] > 70000
].copy()
```

Then modify:

```
filtered["bonus"] = (
    filtered["salary"] * 0.10
)
```

Rule:

```
Use .loc for conditional assignment.
Use .copy() when an independent object is intended.
```

"""
)

# =============================================================================

# 46. ML DATASET INDEXING

# =============================================================================

def ml_dataset_indexing(
df: pd.DataFrame,
) -> None:
"""
Demonstrate indexing for ML features and target.

```
X:
    Feature matrix.

y:
    Target vector.

The same row index should be preserved so that X and y remain
aligned.
"""

print("\n" + "=" * 80)
print("46. MACHINE LEARNING DATASET INDEXING")
print("=" * 80)

feature_columns = [
    "age",
    "salary",
    "experience",
    "performance_score",
]

target_column = "is_remote"

clean_data = df.loc[
    df["age"].notna()
].copy()

X = clean_data.loc[
    :,
    feature_columns,
]

y = clean_data.loc[
    :,
    target_column,
]

print("\nFeature matrix X:")
print(X.to_string())

print("\nTarget y:")
print(y.to_string())

print("\nX and y indexes aligned:")
print(
    X.index.equals(y.index)
)
```

# =============================================================================

# 47. ML INDEX RESET

# =============================================================================

def ml_reset_index(
df: pd.DataFrame,
) -> None:
"""
Show when resetting an index can be useful before ML.

```
Most scikit-learn estimators work with array-like data and do not
require a meaningful Pandas index, but keeping index alignment
during preprocessing can still be useful.
"""

print("\n" + "=" * 80)
print("47. ML RESET INDEX")
print("=" * 80)

filtered = df.loc[
    df["salary"] >= 65000
].copy()

print("\nOriginal filtered index:")
print(filtered.index.to_list())

filtered = filtered.reset_index(
    drop=True
)

print("\nReset index:")
print(filtered.index.to_list())
```

# =============================================================================

# 48. INDEXING AND DATA LEAKAGE

# =============================================================================

def indexing_data_leakage_warning() -> None:
"""
Explain why indexing itself does not prevent ML leakage.
"""

```
print("\n" + "=" * 80)
print("48. INDEXING AND DATA LEAKAGE")
print("=" * 80)

print(
    """
```

An index is an identifier, not a protection against data leakage.

For example:

```
train = df.loc[train_index]
test = df.loc[test_index]
```

is only safe if train_index and test_index were created correctly.

Be careful with:

```
- time-series data
- duplicate entities
- repeated customers
- grouped observations
- target-derived indexes
- future observations
```

For time-dependent ML:

```
Training data should generally come from the past,
while validation/test data should represent future periods.
```

Do not randomly mix future information into training data when
the real prediction problem is time-dependent.
"""
)

# =============================================================================

# 49. INDEX VALIDATION

# =============================================================================

def index_validation(
df: pd.DataFrame,
) -> None:
"""Demonstrate useful index validation checks."""

```
print("\n" + "=" * 80)
print("49. INDEX VALIDATION")
print("=" * 80)

print(
    "Index unique:",
    df.index.is_unique,
)

print(
    "Index monotonic increasing:",
    df.index.is_monotonic_increasing,
)

print(
    "Index monotonic decreasing:",
    df.index.is_monotonic_decreasing,
)

print(
    "Number of index entries:",
    len(df.index),
)
```

# =============================================================================

# 50. REUSABLE INDEXING FUNCTIONS

# =============================================================================

def select_high_salary(
df: pd.DataFrame,
minimum_salary: float,
) -> pd.DataFrame:
"""
Select employees above a salary threshold.

```
Args:
    df: Input DataFrame.
    minimum_salary: Minimum salary.

Returns:
    Independent filtered DataFrame.
"""

return df.loc[
    df["salary"] >= minimum_salary
].copy()
```

def select_columns(
df: pd.DataFrame,
columns: list[str],
) -> pd.DataFrame:
"""
Select a specified list of columns.

```
Args:
    df: Input DataFrame.
    columns: Column names.

Returns:
    DataFrame containing requested columns.
"""

missing_columns = [
    column
    for column in columns
    if column not in df.columns
]

if missing_columns:
    raise KeyError(
        f"Missing columns: {missing_columns}"
    )

return df.loc[
    :,
    columns,
].copy()
```

def select_department(
df: pd.DataFrame,
department: str,
) -> pd.DataFrame:
"""
Select employees belonging to one department.
"""

```
return df.loc[
    df["department"] == department
].copy()
```

# =============================================================================

# 51. REUSABLE FUNCTIONS DEMO

# =============================================================================

def reusable_functions_demo(
df: pd.DataFrame,
) -> None:
"""Demonstrate reusable indexing functions."""

```
print("\n" + "=" * 80)
print("51. REUSABLE INDEXING FUNCTIONS")
print("=" * 80)

high_salary = select_high_salary(
    df,
    minimum_salary=80000,
)

print("\nSalary >= 80K:")
print(
    high_salary[
        [
            "name",
            "salary",
        ]
    ].to_string(index=False)
)

selected_columns = select_columns(
    df,
    [
        "name",
        "department",
        "salary",
    ],
)

print("\nSelected columns:")
print(
    selected_columns.head()
    .to_string(index=False)
)

it_employees = select_department(
    df,
    "IT",
)

print("\nIT employees:")
print(
    it_employees[
        [
            "name",
            "department",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 52. COMMON INDEXING MISTAKES

# =============================================================================

def common_mistakes() -> None:
"""Display common indexing mistakes and corrections."""

```
print("\n" + "=" * 80)
print("52. COMMON INDEXING MISTAKES")
print("=" * 80)

print(
    """
```

1. Confusing loc and iloc

   loc:
   label-based

   iloc:
   position-based

2. Assuming the index is always row numbers

   A DataFrame can have:

   ```
    index = [101, 205, 310]
   ```

   These are labels, not positions.

3. Forgetting loc slices are inclusive

   df.loc[2:5]

   includes label 5.

4. Forgetting iloc slices are exclusive

   df.iloc[2:5]

   excludes position 5.

5. Using chained indexing

   Avoid:

   ```
    df[df["salary"] > 70000]["salary"] = 0
   ```

   Prefer:

   ```
    df.loc[
        df["salary"] > 70000,
        "salary",
    ] = 0
   ```

6. Modifying a filtered object without copy()

   Prefer:

   ```
    filtered = df.loc[
        condition
    ].copy()
   ```

7. Ignoring duplicate indexes

   Check:

   ```
    df.index.is_unique
   ```

8. Breaking X/y alignment

   When working with ML data:

   ```
    X.index.equals(y.index)
   ```

   should be checked when index alignment matters.

9. Using an index as if it were a feature automatically

   An index is metadata unless intentionally converted into
   a meaningful feature.

10. Resetting indexes blindly

    Resetting an index may remove meaningful identifiers.

    Use:

    ```
    reset_index(drop=True)
    ```

    only when discarding the old labels is intentional.
    """
    )

# =============================================================================

# 53. INDEXING CHEAT SHEET

# =============================================================================

def indexing_cheat_sheet() -> None:
"""Display a concise Pandas indexing reference."""

```
print("\n" + "=" * 80)
print("53. INDEXING CHEAT SHEET")
print("=" * 80)

print(
    """
```

## SELECT COLUMNS

Single column:
df["salary"]

Multiple columns:
df[["name", "salary"]]

## SELECT ROWS

First rows:
df[:5]

loc by label:
df.loc[5]

Multiple labels:
df.loc[[1, 3, 5]]

iloc by position:
df.iloc[5]

Multiple positions:
df.iloc[[1, 3, 5]]

## SLICING

loc:
df.loc[2:5]

iloc:
df.iloc[2:5]

## ROWS + COLUMNS

loc:
df.loc[2:5, ["name", "salary"]]

iloc:
df.iloc[2:5, 0:3]

## SCALAR ACCESS

Label:
df.at[2, "salary"]

Position:
df.iat[2, 5]

## BOOLEAN INDEXING

df.loc[df["salary"] > 70000]

df.loc[
(df["salary"] > 70000)
& (df["performance_score"] > 85)
]

## SET VALUES

df.loc[
df["salary"] > 90000,
"level",
] = "High"

## INDEX MANAGEMENT

Set:
df.set_index("employee_id")

Reset:
df.reset_index()

Reset without old index:
df.reset_index(drop=True)

Sort:
df.sort_index()

Reorder:
df.reindex(...)

## MULTIINDEX

df.set_index(
["department", "city"]
)

df.loc["IT"]

df.loc[("IT", "Pune")]

## DATETIME INDEX

df.set_index("date")

df.loc["2025"]

df.loc["2025-01":"2025-06"]

## VALIDATION

df.index.is_unique

df.index.is_monotonic_increasing

label in df.index
"""
)

# =============================================================================

# 54. COMPLETE REAL-WORLD WORKFLOW

# =============================================================================

def complete_workflow(
df: pd.DataFrame,
) -> None:
"""
Demonstrate a realistic indexing workflow.

```
Scenario:
    Prepare a high-quality subset for analysis while preserving
    employee IDs and selecting only required features.
"""

print("\n" + "=" * 80)
print("54. COMPLETE INDEXING WORKFLOW")
print("=" * 80)

# Step 1: Filter valid records.
filtered = df.loc[
    df["age"].notna()
    & (df["salary"] >= 60000)
    & (df["performance_score"] >= 80)
].copy()

# Step 2: Use employee ID as an explicit index.
filtered = filtered.set_index(
    "employee_id"
)

# Step 3: Select required columns.
selected = filtered.loc[
    :,
    [
        "name",
        "department",
        "city",
        "age",
        "salary",
        "experience",
        "performance_score",
    ],
]

# Step 4: Sort using index.
selected = selected.sort_index()

print(
    "\nFinal indexed dataset:"
)

print(
    selected.to_string()
)

print(
    "\nIndex is unique:",
    selected.index.is_unique,
)
```

# =============================================================================

# 55. MAIN FUNCTION

# =============================================================================

def main() -> None:
"""Run the complete Pandas indexing tutorial."""

```
print("=" * 80)
print("PANDAS INDEXING — BEGINNER TO ADVANCED")
print("=" * 80)

df = create_sample_data()

display_dataframe(df)
explain_index()
range_index_example(df)
accessing_index(df)

column_indexing(df)
row_selection_brackets(df)

loc_basic(df)
loc_slicing(df)

iloc_basic(df)
iloc_slicing(df)

loc_vs_iloc(df)

loc_rows_and_columns(df)
iloc_rows_and_columns(df)

at_example(df)
iat_example(df)

boolean_indexing(df)
multiple_boolean_conditions(df)
conditional_loc(df)

set_values_with_loc(df)
set_values_with_iloc(df)
set_scalar_with_at(df)
set_scalar_with_iat(df)

set_custom_index(df)
set_index_keep_column(df)
reset_index_example(df)
reset_index_drop(df)

reindex_example(df)
reindex_columns(df)
sort_index_example(df)
drop_index_labels(df)

duplicate_index_example()

multiindex_basic(df)
multiindex_selection(df)
multiindex_sorting(df)

index_membership(df)
index_name_example(df)

datetime_index_example(df)
time_based_indexing(df)

callable_indexing(df)
indexing_after_filtering(df)
copy_vs_view(df)
chained_indexing_warning()

ml_dataset_indexing(df)
ml_reset_index(df)
indexing_data_leakage_warning()

index_validation(df)

reusable_functions_demo(df)
common_mistakes()
indexing_cheat_sheet()
complete_workflow(df)

print("\n" + "=" * 80)
print("INDEXING TUTORIAL COMPLETED")
print("=" * 80)
```

# =============================================================================

# 56. SCRIPT ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()

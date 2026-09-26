"""
Pandas Series — Complete Guide
==============================

Path:
03-Python-for-Machine-Learning/02-Pandas/series.py

A Pandas Series is a one-dimensional labeled data structure.

This module demonstrates:
1. Creating Series
2. Series attributes
3. Custom indexes
4. Accessing elements
5. loc / iloc / at / iat
6. Slicing
7. Boolean filtering
8. Adding and updating values
9. Missing values
10. Data types
11. Type conversion
12. Mathematical operations
13. Statistics
14. String operations
15. Sorting
16. Unique values and value counts
17. apply() and map()
18. Combining Series
19. Alignment
20. DateTime Series
21. Categorical Series
22. Series from dictionaries
23. Series and NumPy
24. Feature engineering
25. ML-oriented Series operations
26. Data validation
27. Common mistakes
28. Complete real-world workflow

Requirements:
pip install pandas numpy

Run:
python series.py
"""

# =============================================================================

# 1. IMPORTS

# =============================================================================

import numpy as np
import pandas as pd

# =============================================================================

# 2. DISPLAY SETTINGS

# =============================================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

def section(title: str) -> None:
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

# =============================================================================

# 3. WHAT IS A SERIES?

# =============================================================================

def demonstrate_basic_series() -> None:
"""
A Series is a one-dimensional labeled array.

```
Conceptually:

    Index       Value
    -----------------
    0           10
    1           20
    2           30
    3           40
"""

section("3. BASIC SERIES")

series = pd.Series([10, 20, 30, 40])

print(series)
print("\nType:", type(series))
```

# =============================================================================

# 4. CREATE SERIES FROM A LIST

# =============================================================================

def create_from_list() -> None:
"""Create a Series from a Python list."""

```
section("4. SERIES FROM LIST")

scores = pd.Series([85, 90, 78, 92, 88])

print(scores)
```

# =============================================================================

# 5. CREATE SERIES WITH CUSTOM INDEX

# =============================================================================

def create_with_custom_index() -> None:
"""Create a Series with meaningful labels."""

```
section("5. CUSTOM INDEX")

scores = pd.Series(
    [85, 90, 78, 92],
    index=["Alice", "Bob", "Charlie", "David"],
    name="score",
)

print(scores)
```

# =============================================================================

# 6. CREATE SERIES FROM DICTIONARY

# =============================================================================

def create_from_dictionary() -> None:
"""
Dictionary keys become the Series index.
Dictionary values become the Series values.
"""

```
section("6. SERIES FROM DICTIONARY")

ages = pd.Series(
    {
        "Alice": 21,
        "Bob": 22,
        "Charlie": 20,
        "David": 23,
    },
    name="age",
)

print(ages)
```

# =============================================================================

# 7. CREATE SERIES FROM SCALAR

# =============================================================================

def create_from_scalar() -> None:
"""
A scalar value requires an index.

```
The same value is repeated for every index label.
"""

section("7. SERIES FROM SCALAR")

status = pd.Series(
    "Active",
    index=["User_1", "User_2", "User_3"],
    name="status",
)

print(status)
```

# =============================================================================

# 8. CREATE SERIES FROM NUMPY ARRAY

# =============================================================================

def create_from_numpy() -> None:
"""Create a Series from a NumPy array."""

```
section("8. SERIES FROM NUMPY ARRAY")

values = np.array([10, 20, 30, 40, 50])

series = pd.Series(values, name="values")

print(series)
```

# =============================================================================

# 9. SERIES ATTRIBUTES

# =============================================================================

def demonstrate_attributes() -> None:
"""Explore important Series metadata."""

```
section("9. SERIES ATTRIBUTES")

series = pd.Series(
    [85, 90, 78, 92],
    index=["Alice", "Bob", "Charlie", "David"],
    name="score",
)

print("Series:")
print(series)

print("\nname:")
print(series.name)

print("\nindex:")
print(series.index)

print("\nvalues:")
print(series.values)

print("\nshape:")
print(series.shape)

print("\nsize:")
print(series.size)

print("\ndtype:")
print(series.dtype)

print("\nndim:")
print(series.ndim)

print("\nempty:")
print(series.empty)
```

# =============================================================================

# 10. INDEXING WITH []

# =============================================================================

def basic_indexing() -> None:
"""Access Series values using labels and positions."""

```
section("10. BASIC INDEXING")

scores = pd.Series(
    [85, 90, 78, 92],
    index=["Alice", "Bob", "Charlie", "David"],
)

print(scores)

print("\nBob:")
print(scores["Bob"])

print("\nMultiple labels:")
print(scores[["Alice", "David"]])
```

# =============================================================================

# 11. LOC

# =============================================================================

def demonstrate_loc() -> None:
"""
loc performs label-based selection.
"""

```
section("11. LOC — LABEL-BASED INDEXING")

scores = pd.Series(
    [85, 90, 78, 92],
    index=["Alice", "Bob", "Charlie", "David"],
)

print("Single label:")
print(scores.loc["Bob"])

print("\nMultiple labels:")
print(scores.loc[["Alice", "Charlie"]])

print("\nLabel slice:")
print(scores.loc["Bob":"David"])
```

# =============================================================================

# 12. ILOC

# =============================================================================

def demonstrate_iloc() -> None:
"""
iloc performs integer-position-based selection.
"""

```
section("12. ILOC — POSITION-BASED INDEXING")

scores = pd.Series(
    [85, 90, 78, 92],
    index=["Alice", "Bob", "Charlie", "David"],
)

print("First element:")
print(scores.iloc[0])

print("\nThird element:")
print(scores.iloc[2])

print("\nFirst three elements:")
print(scores.iloc[:3])
```

# =============================================================================

# 13. AT AND IAT

# =============================================================================

def demonstrate_at_iat() -> None:
"""
at  -> fast scalar access by label
iat -> fast scalar access by position
"""

```
section("13. AT AND IAT")

scores = pd.Series(
    [85, 90, 78, 92],
    index=["Alice", "Bob", "Charlie", "David"],
)

print("at:")
print(scores.at["Bob"])

print("\niat:")
print(scores.iat[1])
```

# =============================================================================

# 14. SLICING

# =============================================================================

def demonstrate_slicing() -> None:
"""Demonstrate Series slicing."""

```
section("14. SLICING")

series = pd.Series(
    [10, 20, 30, 40, 50],
    index=["A", "B", "C", "D", "E"],
)

print("Position slice:")
print(series.iloc[1:4])

print("\nLabel slice:")
print(series.loc["B":"D"])

print("\nEvery second value:")
print(series.iloc[::2])
```

# =============================================================================

# 15. BOOLEAN FILTERING

# =============================================================================

def boolean_filtering() -> None:
"""Filter values using boolean conditions."""

```
section("15. BOOLEAN FILTERING")

scores = pd.Series(
    [85, 45, 72, 91, 63, 88],
    index=["A", "B", "C", "D", "E", "F"],
)

passed = scores[scores >= 50]

print("All scores:")
print(scores)

print("\nScores >= 50:")
print(passed)

print("\nScores between 60 and 90:")
print(scores[scores.between(60, 90)])
```

# =============================================================================

# 16. MULTIPLE CONDITIONS

# =============================================================================

def multiple_conditions() -> None:
"""Combine multiple boolean conditions."""

```
section("16. MULTIPLE CONDITIONS")

scores = pd.Series([45, 55, 65, 75, 85, 95])

filtered = scores[(scores >= 60) & (scores <= 90)]

print(filtered)
```

# =============================================================================

# 17. ADDING VALUES

# =============================================================================

def adding_values() -> None:
"""Add a new labeled value."""

```
section("17. ADDING VALUES")

scores = pd.Series(
    [85, 90, 78],
    index=["Alice", "Bob", "Charlie"],
)

scores.loc["David"] = 92

print(scores)
```

# =============================================================================

# 18. UPDATING VALUES

# =============================================================================

def updating_values() -> None:
"""Update existing values safely."""

```
section("18. UPDATING VALUES")

scores = pd.Series(
    [85, 90, 78],
    index=["Alice", "Bob", "Charlie"],
)

scores.loc["Bob"] = 95

print(scores)
```

# =============================================================================

# 19. CONDITIONAL UPDATE

# =============================================================================

def conditional_update() -> None:
"""Update values based on a condition."""

```
section("19. CONDITIONAL UPDATE")

scores = pd.Series([45, 55, 72, 91, 63])

scores.loc[scores < 50] = 50

print(scores)
```

# =============================================================================

# 20. DELETE VALUES

# =============================================================================

def deleting_values() -> None:
"""Remove values using drop()."""

```
section("20. DELETING VALUES")

scores = pd.Series(
    [85, 90, 78, 92],
    index=["Alice", "Bob", "Charlie", "David"],
)

updated = scores.drop("Charlie")

print(updated)
```

# =============================================================================

# 21. MISSING VALUES

# =============================================================================

def missing_values() -> None:
"""Work with missing values."""

```
section("21. MISSING VALUES")

scores = pd.Series(
    [85, np.nan, 78, None, 92],
    index=["A", "B", "C", "D", "E"],
)

print(scores)

print("\nMissing mask:")
print(scores.isna())

print("\nNumber of missing values:")
print(scores.isna().sum())

print("\nNon-missing values:")
print(scores.notna())
```

# =============================================================================

# 22. FILL MISSING VALUES

# =============================================================================

def fill_missing_values() -> None:
"""Fill missing values using common strategies."""

```
section("22. FILL MISSING VALUES")

scores = pd.Series([85, np.nan, 78, np.nan, 92])

print("Original:")
print(scores)

print("\nFill with constant:")
print(scores.fillna(0))

print("\nFill with mean:")
print(scores.fillna(scores.mean()))

print("\nForward fill:")
print(scores.ffill())

print("\nBackward fill:")
print(scores.bfill())
```

# =============================================================================

# 23. DROP MISSING VALUES

# =============================================================================

def drop_missing_values() -> None:
"""Remove missing observations."""

```
section("23. DROP MISSING VALUES")

scores = pd.Series([85, np.nan, 78, np.nan, 92])

cleaned = scores.dropna()

print(cleaned)
```

# =============================================================================

# 24. DATA TYPES

# =============================================================================

def demonstrate_dtypes() -> None:
"""Explore and convert Series data types."""

```
section("24. DATA TYPES")

numbers = pd.Series([10, 20, 30, 40])

print("dtype:")
print(numbers.dtype)

decimal_numbers = numbers.astype(float)

print("\nConverted dtype:")
print(decimal_numbers.dtype)
```

# =============================================================================

# 25. SAFE NUMERIC CONVERSION

# =============================================================================

def safe_numeric_conversion() -> None:
"""
Convert messy values into numeric data.

```
errors="coerce" converts invalid values into NaN.
"""

section("25. SAFE NUMERIC CONVERSION")

values = pd.Series(["100", "200", "invalid", "300"])

numeric = pd.to_numeric(values, errors="coerce")

print("Original:")
print(values)

print("\nConverted:")
print(numeric)
```

# =============================================================================

# 26. MATHEMATICAL OPERATIONS

# =============================================================================

def mathematical_operations() -> None:
"""Perform vectorized arithmetic."""

```
section("26. MATHEMATICAL OPERATIONS")

prices = pd.Series([100, 200, 300, 400])

print("Prices:")
print(prices)

print("\nAdd 10:")
print(prices + 10)

print("\nMultiply by 2:")
print(prices * 2)

print("\nDiscount 10%:")
print(prices * 0.90)

print("\nSquare:")
print(prices**2)
```

# =============================================================================

# 27. UNIVERSAL FUNCTIONS

# =============================================================================

def universal_functions() -> None:
"""Use NumPy functions with Series."""

```
section("27. NUMPY FUNCTIONS")

values = pd.Series([1, 4, 9, 16, 25])

print("Square root:")
print(np.sqrt(values))

print("\nLogarithm:")
print(np.log(values))
```

# =============================================================================

# 28. STATISTICAL OPERATIONS

# =============================================================================

def statistical_operations() -> None:
"""Calculate common descriptive statistics."""

```
section("28. STATISTICS")

scores = pd.Series([72, 85, 91, 67, 88, 95, 78])

print("Count:", scores.count())
print("Sum:", scores.sum())
print("Mean:", scores.mean())
print("Median:", scores.median())
print("Minimum:", scores.min())
print("Maximum:", scores.max())
print("Standard deviation:", scores.std())
print("Variance:", scores.var())
print("Quantile 25%:", scores.quantile(0.25))
print("Quantile 75%:", scores.quantile(0.75))
```

# =============================================================================

# 29. DESCRIBE

# =============================================================================

def describe_series() -> None:
"""Generate a statistical summary."""

```
section("29. DESCRIBE")

scores = pd.Series([72, 85, 91, 67, 88, 95, 78])

print(scores.describe())
```

# =============================================================================

# 30. MIN/MAX INDEX

# =============================================================================

def min_max_index() -> None:
"""Find the labels corresponding to minimum and maximum values."""

```
section("30. MIN/MAX INDEX")

scores = pd.Series(
    [72, 85, 91, 67, 88],
    index=["A", "B", "C", "D", "E"],
)

print("Minimum value:", scores.min())
print("Minimum label:", scores.idxmin())

print("\nMaximum value:", scores.max())
print("Maximum label:", scores.idxmax())
```

# =============================================================================

# 31. UNIQUE VALUES

# =============================================================================

def unique_values() -> None:
"""Find unique values."""

```
section("31. UNIQUE VALUES")

cities = pd.Series(
    ["Pune", "Mumbai", "Pune", "Delhi", "Mumbai", "Pune"]
)

print("Unique values:")
print(cities.unique())

print("\nNumber of unique values:")
print(cities.nunique())
```

# =============================================================================

# 32. VALUE COUNTS

# =============================================================================

def value_counts() -> None:
"""Count frequency of categorical values."""

```
section("32. VALUE COUNTS")

departments = pd.Series(
    [
        "AI",
        "ML",
        "AI",
        "Data Science",
        "ML",
        "AI",
        "Data Science",
    ]
)

print(departments.value_counts())

print("\nNormalized:")
print(departments.value_counts(normalize=True))
```

# =============================================================================

# 33. SORTING

# =============================================================================

def sorting_series() -> None:
"""Sort Series values and indexes."""

```
section("33. SORTING")

scores = pd.Series(
    [85, 72, 95, 63],
    index=["Alice", "Bob", "Charlie", "David"],
)

print("Sort by values:")
print(scores.sort_values())

print("\nDescending:")
print(scores.sort_values(ascending=False))

print("\nSort by index:")
print(scores.sort_index())
```

# =============================================================================

# 34. STRING OPERATIONS

# =============================================================================

def string_operations() -> None:
"""Use vectorized string operations through .str."""

```
section("34. STRING OPERATIONS")

names = pd.Series(
    [" alice ", "BOB", "Charlie", " david "]
)

print("Original:")
print(names)

print("\nStrip:")
print(names.str.strip())

print("\nLowercase:")
print(names.str.lower())

print("\nUppercase:")
print(names.str.upper())

print("\nTitle case:")
print(names.str.title())

print("\nString length:")
print(names.str.len())
```

# =============================================================================

# 35. STRING FILTERING

# =============================================================================

def string_filtering() -> None:
"""Filter strings using vectorized string methods."""

```
section("35. STRING FILTERING")

names = pd.Series(
    ["Alice", "Bob", "Anita", "Charlie", "Andrew"]
)

print("Names beginning with A:")
print(names[names.str.startswith("A")])

print("\nNames containing 'li':")
print(names[names.str.contains("li", case=False, na=False)])
```

# =============================================================================

# 36. APPLY

# =============================================================================

def demonstrate_apply() -> None:
"""
apply() applies a Python function to each value.

```
Prefer vectorized operations when available because they
are generally clearer and faster.
"""

section("36. APPLY")

scores = pd.Series([45, 67, 82, 91, 38])

def grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 60:
        return "C"
    return "D"

grades = scores.apply(grade)

print("Scores:")
print(scores)

print("\nGrades:")
print(grades)
```

# =============================================================================

# 37. MAP

# =============================================================================

def demonstrate_map() -> None:
"""Map categorical values to new values."""

```
section("37. MAP")

gender = pd.Series(
    ["M", "F", "M", "F", "M"]
)

mapped = gender.map(
    {
        "M": "Male",
        "F": "Female",
    }
)

print("Original:")
print(gender)

print("\nMapped:")
print(mapped)
```

# =============================================================================

# 38. REPLACE

# =============================================================================

def demonstrate_replace() -> None:
"""Replace selected values."""

```
section("38. REPLACE")

status = pd.Series(
    ["Y", "N", "Y", "Y", "N"]
)

cleaned = status.replace(
    {
        "Y": "Yes",
        "N": "No",
    }
)

print(cleaned)
```

# =============================================================================

# 39. ALIGNMENT

# =============================================================================

def demonstrate_alignment() -> None:
"""
Pandas aligns Series by index labels during operations.

```
This is one of the most important differences between
Pandas and ordinary NumPy arrays.
"""

section("39. SERIES ALIGNMENT")

sales = pd.Series(
    [100, 200, 300],
    index=["A", "B", "C"],
)

expenses = pd.Series(
    [40, 80, 120],
    index=["B", "C", "D"],
)

print("Sales:")
print(sales)

print("\nExpenses:")
print(expenses)

print("\nSales - Expenses:")
print(sales - expenses)
```

# =============================================================================

# 40. REINDEX

# =============================================================================

def demonstrate_reindex() -> None:
"""Rearrange or introduce index labels."""

```
section("40. REINDEX")

scores = pd.Series(
    [85, 90, 78],
    index=["Alice", "Bob", "Charlie"],
)

reordered = scores.reindex(
    ["Charlie", "Alice", "Bob", "David"]
)

print(reordered)
```

# =============================================================================

# 41. INDEX NAME AND SERIES NAME

# =============================================================================

def demonstrate_names() -> None:
"""Set meaningful names for Series and its index."""

```
section("41. SERIES AND INDEX NAMES")

scores = pd.Series(
    [85, 90, 78],
    index=["Alice", "Bob", "Charlie"],
    name="score",
)

scores.index.name = "student"

print(scores)

print("\nSeries name:", scores.name)
print("Index name:", scores.index.name)
```

# =============================================================================

# 42. COPY

# =============================================================================

def demonstrate_copy() -> None:
"""
copy() creates an independent Series.

```
This is useful when a transformation should not modify
the original object.
"""

section("42. COPY")

original = pd.Series([10, 20, 30])

copied = original.copy()

copied.loc[0] = 999

print("Original:")
print(original)

print("\nCopied and modified:")
print(copied)
```

# =============================================================================

# 43. CONCATENATING SERIES

# =============================================================================

def concatenate_series() -> None:
"""Combine multiple Series."""

```
section("43. CONCATENATING SERIES")

first = pd.Series([10, 20, 30])
second = pd.Series([40, 50, 60])

combined = pd.concat(
    [first, second],
    ignore_index=True,
)

print(combined)
```

# =============================================================================

# 44. COMPARING SERIES

# =============================================================================

def compare_series() -> None:
"""Compare two aligned Series."""

```
section("44. COMPARING SERIES")

first = pd.Series(
    [10, 20, 30],
    index=["A", "B", "C"],
)

second = pd.Series(
    [10, 25, 30],
    index=["A", "B", "C"],
)

print("Equality:")
print(first == second)

print("\nGreater than:")
print(first > second)
```

# =============================================================================

# 45. DUPLICATES

# =============================================================================

def duplicate_values() -> None:
"""Detect and remove duplicate values."""

```
section("45. DUPLICATES")

values = pd.Series([10, 20, 20, 30, 10, 40])

print("Duplicate mask:")
print(values.duplicated())

print("\nWithout duplicates:")
print(values.drop_duplicates())
```

# =============================================================================

# 46. RANKING

# =============================================================================

def ranking() -> None:
"""Rank numerical values."""

```
section("46. RANKING")

scores = pd.Series(
    [85, 92, 78, 95, 88],
    index=["A", "B", "C", "D", "E"],
)

print(scores.rank(ascending=False))
```

# =============================================================================

# 47. CUMULATIVE OPERATIONS

# =============================================================================

def cumulative_operations() -> None:
"""Calculate cumulative sums, products, minimums and maximums."""

```
section("47. CUMULATIVE OPERATIONS")

sales = pd.Series([100, 200, 150, 300])

print("Cumulative sum:")
print(sales.cumsum())

print("\nCumulative maximum:")
print(sales.cummax())

print("\nCumulative minimum:")
print(sales.cummin())
```

# =============================================================================

# 48. PERCENT CHANGE

# =============================================================================

def percent_change() -> None:
"""Calculate percentage change between consecutive observations."""

```
section("48. PERCENT CHANGE")

revenue = pd.Series(
    [1000, 1200, 1500, 1350, 1800]
)

print(revenue.pct_change())
```

# =============================================================================

# 49. ROLLING WINDOW

# =============================================================================

def rolling_window() -> None:
"""
Rolling statistics are useful for time-series feature engineering.
"""

```
section("49. ROLLING WINDOW")

sales = pd.Series(
    [100, 120, 140, 130, 160, 180]
)

rolling_mean = sales.rolling(window=3).mean()

print("Sales:")
print(sales)

print("\n3-period rolling mean:")
print(rolling_mean)
```

# =============================================================================

# 50. DATETIME SERIES

# =============================================================================

def datetime_series() -> None:
"""Create and work with datetime values."""

```
section("50. DATETIME SERIES")

dates = pd.Series(
    pd.to_datetime(
        [
            "2026-01-01",
            "2026-02-15",
            "2026-03-20",
            "2026-04-10",
        ]
    )
)

print(dates)

print("\nYear:")
print(dates.dt.year)

print("\nMonth:")
print(dates.dt.month)

print("\nDay:")
print(dates.dt.day)

print("\nDay of week:")
print(dates.dt.dayofweek)
```

# =============================================================================

# 51. DATETIME FEATURE ENGINEERING

# =============================================================================

def datetime_feature_engineering() -> None:
"""Extract ML-friendly features from dates."""

```
section("51. DATETIME FEATURE ENGINEERING")

dates = pd.Series(
    pd.to_datetime(
        [
            "2026-01-05",
            "2026-02-14",
            "2026-03-21",
            "2026-04-18",
        ]
    )
)

features = pd.DataFrame(
    {
        "date": dates,
        "year": dates.dt.year,
        "month": dates.dt.month,
        "day": dates.dt.day,
        "day_of_week": dates.dt.dayofweek,
        "is_weekend": dates.dt.dayofweek >= 5,
    }
)

print(features)
```

# =============================================================================

# 52. CATEGORICAL SERIES

# =============================================================================

def categorical_series() -> None:
"""
Convert repeated string values to categorical dtype.

```
This can reduce memory usage and explicitly represent
categorical variables.
"""

section("52. CATEGORICAL SERIES")

departments = pd.Series(
    ["AI", "ML", "AI", "Data Science", "ML"],
    dtype="category",
)

print(departments)

print("\nDtype:")
print(departments.dtype)

print("\nCategories:")
print(departments.cat.categories)
```

# =============================================================================

# 53. CATEGORICAL ORDERING

# =============================================================================

def ordered_categories() -> None:
"""Create an ordered categorical Series."""

```
section("53. ORDERED CATEGORIES")

levels = pd.Series(
    ["Medium", "High", "Low", "Medium", "High"]
)

ordered = pd.Categorical(
    levels,
    categories=["Low", "Medium", "High"],
    ordered=True,
)

series = pd.Series(ordered)

print(series)

print("\nSorted:")
print(series.sort_values())
```

# =============================================================================

# 54. SERIES TO NUMPY

# =============================================================================

def series_to_numpy() -> None:
"""Convert a Series to a NumPy array."""

```
section("54. SERIES TO NUMPY")

scores = pd.Series([85, 90, 78, 92])

array = scores.to_numpy()

print("Series:")
print(scores)

print("\nNumPy array:")
print(array)

print("\nArray type:")
print(type(array))
```

# =============================================================================

# 55. SERIES TO LIST

# =============================================================================

def series_to_list() -> None:
"""Convert Series values to a Python list."""

```
section("55. SERIES TO LIST")

scores = pd.Series([85, 90, 78, 92])

values = scores.tolist()

print(values)
print(type(values))
```

# =============================================================================

# 56. SERIES TO DICTIONARY

# =============================================================================

def series_to_dictionary() -> None:
"""Convert a labeled Series to a dictionary."""

```
section("56. SERIES TO DICTIONARY")

scores = pd.Series(
    [85, 90, 78],
    index=["Alice", "Bob", "Charlie"],
)

dictionary = scores.to_dict()

print(dictionary)
```

# =============================================================================

# 57. SERIES TO DATAFRAME

# =============================================================================

def series_to_dataframe() -> None:
"""Convert a Series into a one-column DataFrame."""

```
section("57. SERIES TO DATAFRAME")

scores = pd.Series(
    [85, 90, 78],
    index=["Alice", "Bob", "Charlie"],
    name="score",
)

dataframe = scores.to_frame()

print(dataframe)
```

# =============================================================================

# 58. FEATURE ENGINEERING

# =============================================================================

def feature_engineering() -> None:
"""
Create machine-learning features from a numerical Series.
"""

```
section("58. FEATURE ENGINEERING")

income = pd.Series(
    [25000, 35000, 50000, 75000, 100000],
    name="income",
)

features = pd.DataFrame(
    {
        "income": income,
        "income_log": np.log1p(income),
        "income_squared": income**2,
        "income_k": income / 1000,
        "high_income": income >= 75000,
    }
)

print(features)
```

# =============================================================================

# 59. BINNING CONTINUOUS VALUES

# =============================================================================

def binning_values() -> None:
"""
Convert continuous numerical values into categories.

```
Binning can be useful for exploratory analysis and some
modeling workflows.
"""

section("59. BINNING VALUES")

ages = pd.Series(
    [18, 22, 27, 35, 42, 58, 67],
    name="age",
)

bins = [0, 18, 30, 45, 60, 100]
labels = [
    "Teen",
    "Young Adult",
    "Adult",
    "Middle Age",
    "Senior",
]

age_groups = pd.cut(
    ages,
    bins=bins,
    labels=labels,
    include_lowest=True,
)

print(
    pd.DataFrame(
        {
            "age": ages,
            "age_group": age_groups,
        }
    )
)
```

# =============================================================================

# 60. NORMALIZATION

# =============================================================================

def normalization() -> None:
"""
Min-Max normalization:

```
    x_scaled = (x - min) / (max - min)

This transforms values approximately into [0, 1].
"""

section("60. MIN-MAX NORMALIZATION")

values = pd.Series(
    [10, 20, 30, 40, 50],
    name="value",
)

normalized = (
    (values - values.min())
    / (values.max() - values.min())
)

print(
    pd.DataFrame(
        {
            "original": values,
            "normalized": normalized,
        }
    )
)
```

# =============================================================================

# 61. STANDARDIZATION

# =============================================================================

def standardization() -> None:
"""
Standardization:

```
    z = (x - mean) / standard_deviation

Useful when algorithms benefit from centered/scaled features.

For production ML workflows, fit preprocessing parameters on
training data only and apply them to validation/test data.
"""

section("61. STANDARDIZATION")

values = pd.Series(
    [10, 20, 30, 40, 50],
    name="value",
)

standardized = (
    (values - values.mean())
    / values.std()
)

print(
    pd.DataFrame(
        {
            "original": values,
            "standardized": standardized,
        }
    )
)
```

# =============================================================================

# 62. OUTLIER DETECTION WITH IQR

# =============================================================================

def detect_outliers() -> None:
"""
Detect potential outliers using the IQR rule.

```
IQR = Q3 - Q1

Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
"""

section("62. OUTLIER DETECTION")

values = pd.Series(
    [10, 12, 11, 13, 12, 14, 100],
    name="value",
)

q1 = values.quantile(0.25)
q3 = values.quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = values[
    (values < lower_bound)
    | (values > upper_bound)
]

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)

print("\nPotential outliers:")
print(outliers)
```

# =============================================================================

# 63. ML TARGET SERIES

# =============================================================================

def ml_target_series() -> None:
"""
A Series is commonly used as a target vector y.

```
Example:
    X -> DataFrame containing features
    y -> Series containing the target
"""

section("63. SERIES AS ML TARGET")

data = pd.DataFrame(
    {
        "age": [21, 25, 32, 40, 29],
        "income": [25000, 35000, 55000, 80000, 42000],
        "purchased": [0, 1, 1, 1, 0],
    }
)

X = data[["age", "income"]]
y = data["purchased"]

print("Features X:")
print(X)

print("\nTarget y:")
print(y)

print("\nX type:", type(X))
print("y type:", type(y))
```

# =============================================================================

# 64. TARGET DISTRIBUTION

# =============================================================================

def target_distribution() -> None:
"""Inspect a classification target."""

```
section("64. TARGET DISTRIBUTION")

target = pd.Series(
    [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    name="target",
)

print(target.value_counts())
print("\nProportion:")
print(target.value_counts(normalize=True))
```

# =============================================================================

# 65. CLASS IMBALANCE CHECK

# =============================================================================

def class_imbalance_check() -> None:
"""Check class distribution in a target Series."""

```
section("65. CLASS DISTRIBUTION")

target = pd.Series(
    [0] * 90 + [1] * 10,
    name="target",
)

distribution = target.value_counts(normalize=True)

print(distribution)

print("\nClass counts:")
print(target.value_counts())
```

# =============================================================================

# 66. MISSING TARGET VALUES

# =============================================================================

def missing_target_values() -> None:
"""
Missing target values often require special handling.

```
Do not blindly fill a target with its mean or mode.
The appropriate strategy depends on the modeling problem.
"""

section("66. MISSING TARGET VALUES")

target = pd.Series(
    [1, 0, 1, np.nan, 0, 1],
    name="target",
)

print("Target:")
print(target)

print("\nMissing target count:")
print(target.isna().sum())

print("\nRows with known target:")
print(target.dropna())
```

# =============================================================================

# 67. SERIES VALIDATION

# =============================================================================

def validate_series(series: pd.Series) -> dict:
"""
Return basic validation information for a Series.
"""

```
return {
    "name": series.name,
    "dtype": str(series.dtype),
    "rows": len(series),
    "missing": int(series.isna().sum()),
    "unique": int(series.nunique(dropna=True)),
}
```

def demonstrate_validation() -> None:
"""Validate a Series."""

```
section("67. SERIES VALIDATION")

scores = pd.Series(
    [85, 90, np.nan, 78, 92],
    name="score",
)

report = validate_series(scores)

for key, value in report.items():
    print(f"{key}: {value}")
```

# =============================================================================

# 68. REUSABLE NUMERIC CLEANING FUNCTION

# =============================================================================

def clean_numeric_series(
series: pd.Series,
fill_missing: bool = True,
) -> pd.Series:
"""
Clean a numerical Series.

```
Steps:
    1. Convert values to numeric.
    2. Optionally fill missing values with the median.

Important:
    In ML, the median should generally be learned from the
    training split and then reused for validation/test data.
"""

cleaned = pd.to_numeric(
    series,
    errors="coerce",
)

if fill_missing:
    cleaned = cleaned.fillna(cleaned.median())

return cleaned
```

def demonstrate_cleaning_function() -> None:
"""Demonstrate reusable numerical cleaning."""

```
section("68. REUSABLE CLEANING FUNCTION")

raw = pd.Series(
    ["100", "200", "invalid", "300", None],
    name="income",
)

cleaned = clean_numeric_series(raw)

print("Raw:")
print(raw)

print("\nCleaned:")
print(cleaned)
```

# =============================================================================

# 69. SERIES AND DATA LEAKAGE

# =============================================================================

def demonstrate_data_leakage_concept() -> None:
"""
Explain an important ML preprocessing rule.

```
Incorrect:
    Calculate preprocessing statistics from the entire dataset
    before train/test splitting.

Correct:
    1. Split the dataset.
    2. Learn preprocessing parameters from training data.
    3. Apply those parameters to validation/test data.

Example:
    Median imputation, normalization, standardization and
    outlier thresholds should be learned without using test data.
"""

section("69. DATA LEAKAGE")

print(
    "ML preprocessing should be fitted using training data only."
)
print(
    "Then apply the learned transformation to validation/test data."
)
print(
    "Avoid calculating preprocessing statistics from the full dataset."
)
```

# =============================================================================

# 70. COMPLETE REAL-WORLD SERIES WORKFLOW

# =============================================================================

def complete_workflow() -> None:
"""
Complete example:

```
    raw values
        ↓
    numeric conversion
        ↓
    missing-value handling
        ↓
    validation
        ↓
    feature engineering
        ↓
    ML-ready Series
"""

section("70. COMPLETE REAL-WORLD WORKFLOW")

raw_income = pd.Series(
    ["25000", "35000", "invalid", None, "75000", "100000"],
    name="income",
)

print("1. Raw data:")
print(raw_income)

income = pd.to_numeric(
    raw_income,
    errors="coerce",
)

print("\n2. Numeric conversion:")
print(income)

median_income = income.median()

income = income.fillna(median_income)

print("\n3. Missing values filled using median:")
print(income)

income_k = income / 1000
log_income = np.log1p(income)

result = pd.DataFrame(
    {
        "income": income,
        "income_k": income_k,
        "log_income": log_income,
    }
)

print("\n4. Engineered features:")
print(result)

print("\n5. Validation:")
print(validate_series(income))
```

# =============================================================================

# 71. COMMON SERIES MISTAKES

# =============================================================================

def common_mistakes() -> None:
"""Print common mistakes to avoid."""

```
section("71. COMMON MISTAKES")

mistakes = [
    "Confusing labels with integer positions.",
    "Using iloc when label-based selection is required.",
    "Using loc when positional selection is required.",
    "Ignoring missing values before numerical calculations.",
    "Using chained indexing for assignments.",
    "Forgetting that Series operations align by index.",
    "Modifying a filtered object without understanding copy semantics.",
    "Using Python loops when vectorized Series operations exist.",
    "Calculating ML preprocessing statistics using test data.",
    "Treating categorical values as numerical values without justification.",
]

for number, mistake in enumerate(mistakes, start=1):
    print(f"{number}. {mistake}")
```

# =============================================================================

# 72. SERIES CHEAT SHEET

# =============================================================================

def cheat_sheet() -> None:
"""Print a compact Series reference."""

```
section("72. SERIES CHEAT SHEET")

cheat_sheet_data = {
    "Create": "pd.Series([...])",
    "Custom index": "pd.Series(data, index=[...])",
    "Dictionary": "pd.Series({...})",
    "Label access": "series.loc[label]",
    "Position access": "series.iloc[position]",
    "Fast label access": "series.at[label]",
    "Fast position access": "series.iat[position]",
    "Filter": "series[series > value]",
    "Missing check": "series.isna()",
    "Fill missing": "series.fillna(value)",
    "Drop missing": "series.dropna()",
    "Unique values": "series.unique()",
    "Frequency": "series.value_counts()",
    "Sort values": "series.sort_values()",
    "Sort index": "series.sort_index()",
    "Statistics": "series.describe()",
    "Apply function": "series.apply(function)",
    "Map values": "series.map(mapping)",
    "Convert numeric": "pd.to_numeric(series, errors='coerce')",
    "Convert dtype": "series.astype(dtype)",
    "To DataFrame": "series.to_frame()",
    "To NumPy": "series.to_numpy()",
    "To list": "series.tolist()",
    "To dictionary": "series.to_dict()",
    "Rolling mean": "series.rolling(window=3).mean()",
    "Percentage change": "series.pct_change()",
}

for operation, syntax in cheat_sheet_data.items():
    print(f"{operation:22} -> {syntax}")
```

# =============================================================================

# 73. MAIN

# =============================================================================

def main() -> None:
"""Run the complete Pandas Series tutorial."""

```
demonstrate_basic_series()
create_from_list()
create_with_custom_index()
create_from_dictionary()
create_from_scalar()
create_from_numpy()

demonstrate_attributes()

basic_indexing()
demonstrate_loc()
demonstrate_iloc()
demonstrate_at_iat()
demonstrate_slicing()

boolean_filtering()
multiple_conditions()

adding_values()
updating_values()
conditional_update()
deleting_values()

missing_values()
fill_missing_values()
drop_missing_values()

demonstrate_dtypes()
safe_numeric_conversion()

mathematical_operations()
universal_functions()
statistical_operations()
describe_series()
min_max_index()

unique_values()
value_counts()
sorting_series()

string_operations()
string_filtering()

demonstrate_apply()
demonstrate_map()
demonstrate_replace()

demonstrate_alignment()
demonstrate_reindex()
demonstrate_names()
demonstrate_copy()

concatenate_series()
compare_series()
duplicate_values()
ranking()
cumulative_operations()
percent_change()
rolling_window()

datetime_series()
datetime_feature_engineering()

categorical_series()
ordered_categories()

series_to_numpy()
series_to_list()
series_to_dictionary()
series_to_dataframe()

feature_engineering()
binning_values()
normalization()
standardization()
detect_outliers()

ml_target_series()
target_distribution()
class_imbalance_check()
missing_target_values()

demonstrate_validation()
demonstrate_cleaning_function()
demonstrate_data_leakage_concept()

complete_workflow()

common_mistakes()
cheat_sheet()
```

# =============================================================================

# 74. SCRIPT ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()
"""

# =============================================================================

# QUICK REFERENCE

# =============================================================================

#

# Series creation:

# pd.Series([10, 20, 30])

#

# Custom labels:

# pd.Series([10, 20, 30], index=["A", "B", "C"])

#

# Dictionary:

# pd.Series({"A": 10, "B": 20})

#

# Access:

# series.loc["A"]      # label

# series.iloc[0]       # position

# series.at["A"]       # fast label

# series.iat[0]        # fast position

#

# Filtering:

# series[series > 50]

#

# Missing values:

# series.isna()

# series.fillna(...)

# series.dropna()

#

# Statistics:

# series.mean()

# series.median()

# series.std()

# series.describe()

#

# Categories:

# series.value_counts()

# series.unique()

#

# Transformation:

# series.map(...)

# series.apply(...)

#

# ML:

# X = dataframe[features]

# y = dataframe["target"]

#

# Key rule:

# In machine learning, preprocessing parameters should be

# learned from training data only to avoid data leakage.

#

# =============================================================================

"""

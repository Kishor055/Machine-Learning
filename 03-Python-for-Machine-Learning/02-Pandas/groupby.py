"""
Pandas GroupBy — Beginner to Advanced
=====================================

File:
03-Python-for-Machine-Learning/02-Pandas/groupby.py

Description:
A practical and comprehensive guide to Pandas GroupBy operations.

Topics Covered:
1. Creating a sample dataset
2. Understanding split-apply-combine
3. Basic groupby()
4. Selecting grouped columns
5. Aggregation: sum, mean, min, max, count
6. Multiple aggregations
7. agg()
8. Named aggregations
9. Grouping by multiple columns
10. Grouping by index
11. Grouping by categorical data
12. Group filtering
13. transform()
14. transform() vs agg()
15. Group-level calculations
16. Ranking within groups
17. Percentage of group total
18. Grouped cumulative operations
19. Grouped sorting
20. Grouped statistics
21. Handling missing values
22. reset_index()
23. as_index=False
24. size() vs count()
25. first(), last(), nth()
26. value_counts()
27. Grouped pivot-style analysis
28. ML-oriented feature engineering
29. Avoiding data leakage
30. Reusable GroupBy functions
31. Common mistakes
32. Complete real-world workflow

Requirements:
pip install pandas numpy

Run:
python groupby.py

Author:
Kishor Patil

Core Concept:
GroupBy follows the:

```
    SPLIT → APPLY → COMBINE

pattern.

SPLIT:
    Divide data into groups.

APPLY:
    Perform calculations on each group.

COMBINE:
    Combine the results into a new Series or DataFrame.
```

Example:

```
df.groupby("department")["salary"].mean()

This means:

    1. Split employees by department.
    2. Calculate salary mean for each department.
    3. Combine the results.
```

Important ML Note:
Group-level statistics can become data leakage if they are calculated
using validation/test/future data. In production ML pipelines, fit
data-dependent transformations on training data only.
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
Create a realistic employee dataset.

```
Returns:
    pd.DataFrame:
        Employee dataset suitable for GroupBy demonstrations.
"""

data = {
    "employee_id": range(101, 116),

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
        "Aditya",
        "Pooja",
        "Arjun",
        "Isha",
        "Kabir",
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
        "IT",
        "Finance",
        "Marketing",
        "HR",
        "IT",
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
        "Pune",
        "Mumbai",
        "Pune",
        "Nashik",
        "Mumbai",
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
        82000,
        78000,
        72000,
        60000,
        98000,
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
        6,
        9,
        4,
        5,
        13,
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
        89,
        92,
        87,
        78,
        96,
    ],

    "projects_completed": [
        3,
        2,
        7,
        6,
        4,
        8,
        6,
        3,
        9,
        4,
        7,
        8,
        5,
        4,
        10,
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
        True,
        False,
        True,
        False,
        True,
    ],

    "joining_date": [
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
        "2019-05-12",
        "2016-10-20",
        "2020-02-17",
        "2021-11-05",
        "2012-04-10",
    ],
}

df = pd.DataFrame(data)

df["joining_date"] = pd.to_datetime(
    df["joining_date"]
)

return df
```

# =============================================================================

# 3. DISPLAY DATASET

# =============================================================================

def display_dataset(df: pd.DataFrame) -> None:
"""Display the complete dataset."""

```
print("\n" + "=" * 80)
print("DATASET")
print("=" * 80)

print(df.to_string(index=False))
```

# =============================================================================

# 4. UNDERSTANDING SPLIT-APPLY-COMBINE

# =============================================================================

def explain_split_apply_combine() -> None:
"""Explain the fundamental GroupBy concept."""

```
print("\n" + "=" * 80)
print("4. SPLIT → APPLY → COMBINE")
print("=" * 80)

print(
    """
```

GroupBy follows three major steps:

1. SPLIT
   Divide the dataset into groups.

   Example:
   department = IT
   department = HR
   department = Finance
   department = Marketing

2. APPLY
   Perform a calculation for every group.

   Examples:
   mean salary
   total salary
   maximum performance
   employee count

3. COMBINE
   Return the results as a Series or DataFrame.

Example:

```
df.groupby("department")["salary"].mean()
```

Conceptually:

```
IT         → average salary
HR         → average salary
Finance    → average salary
Marketing  → average salary
```

"""
)

# =============================================================================

# 5. BASIC GROUPBY

# =============================================================================

def basic_groupby(df: pd.DataFrame) -> None:
"""Calculate basic statistics by department."""

```
print("\n" + "=" * 80)
print("5. BASIC GROUPBY")
print("=" * 80)

average_salary = (
    df.groupby("department")["salary"].mean()
)

print("\nAverage salary by department:")
print(average_salary)
```

# =============================================================================

# 6. GROUPBY SUM

# =============================================================================

def groupby_sum(df: pd.DataFrame) -> None:
"""Calculate total salary by department."""

```
print("\n" + "=" * 80)
print("6. GROUPBY SUM")
print("=" * 80)

result = (
    df.groupby("department")["salary"].sum()
)

print("\nTotal salary by department:")
print(result)
```

# =============================================================================

# 7. GROUPBY COUNT

# =============================================================================

def groupby_count(df: pd.DataFrame) -> None:
"""Count employees in each department."""

```
print("\n" + "=" * 80)
print("7. GROUPBY COUNT")
print("=" * 80)

result = (
    df.groupby("department")["employee_id"].count()
)

print("\nEmployee count by department:")
print(result)
```

# =============================================================================

# 8. GROUPBY MIN AND MAX

# =============================================================================

def groupby_min_max(df: pd.DataFrame) -> None:
"""Find minimum and maximum values by group."""

```
print("\n" + "=" * 80)
print("8. GROUPBY MIN AND MAX")
print("=" * 80)

result = (
    df.groupby("department")["salary"]
    .agg(["min", "max"])
)

print("\nMinimum and maximum salary:")
print(result)
```

# =============================================================================

# 9. MULTIPLE AGGREGATIONS

# =============================================================================

def multiple_aggregations(df: pd.DataFrame) -> None:
"""Calculate multiple statistics at once."""

```
print("\n" + "=" * 80)
print("9. MULTIPLE AGGREGATIONS")
print("=" * 80)

result = (
    df.groupby("department")["salary"]
    .agg([
        "count",
        "mean",
        "median",
        "min",
        "max",
        "sum",
        "std",
    ])
)

print("\nSalary statistics by department:")
print(result)
```

# =============================================================================

# 10. AGG() ON MULTIPLE COLUMNS

# =============================================================================

def aggregate_multiple_columns(df: pd.DataFrame) -> None:
"""Aggregate several columns with different functions."""

```
print("\n" + "=" * 80)
print("10. AGG() ON MULTIPLE COLUMNS")
print("=" * 80)

result = (
    df.groupby("department")
    .agg({
        "salary": ["mean", "min", "max"],
        "experience": ["mean", "max"],
        "performance_score": ["mean", "max"],
        "projects_completed": ["sum", "mean"],
    })
)

print("\nMultiple column aggregation:")
print(result)
```

# =============================================================================

# 11. NAMED AGGREGATIONS

# =============================================================================

def named_aggregations(df: pd.DataFrame) -> None:
"""
Use named aggregations for cleaner output columns.
"""

```
print("\n" + "=" * 80)
print("11. NAMED AGGREGATIONS")
print("=" * 80)

result = (
    df.groupby("department")
    .agg(
        employee_count=("employee_id", "count"),
        average_salary=("salary", "mean"),
        highest_salary=("salary", "max"),
        average_experience=("experience", "mean"),
        average_performance=("performance_score", "mean"),
        total_projects=("projects_completed", "sum"),
    )
    .reset_index()
)

print("\nClean aggregated DataFrame:")
print(result.to_string(index=False))
```

# =============================================================================

# 12. GROUP BY MULTIPLE COLUMNS

# =============================================================================

def multiple_group_columns(df: pd.DataFrame) -> None:
"""Group by department and city."""

```
print("\n" + "=" * 80)
print("12. GROUP BY MULTIPLE COLUMNS")
print("=" * 80)

result = (
    df.groupby(
        ["department", "city"]
    )["salary"]
    .mean()
)

print("\nAverage salary by department and city:")
print(result)
```

# =============================================================================

# 13. MULTIPLE GROUP COLUMNS WITH RESET_INDEX

# =============================================================================

def multiple_group_columns_dataframe(
df: pd.DataFrame,
) -> None:
"""Return grouped results as a normal DataFrame."""

```
print("\n" + "=" * 80)
print("13. MULTIPLE GROUP COLUMNS + RESET_INDEX")
print("=" * 80)

result = (
    df.groupby(
        ["department", "city"],
        as_index=False,
    )
    .agg(
        employee_count=("employee_id", "count"),
        average_salary=("salary", "mean"),
    )
)

print("\nGrouped DataFrame:")
print(result.to_string(index=False))
```

# =============================================================================

# 14. GROUPING BY INDEX

# =============================================================================

def grouping_by_index(df: pd.DataFrame) -> None:
"""Demonstrate grouping using an index level."""

```
print("\n" + "=" * 80)
print("14. GROUPING BY INDEX")
print("=" * 80)

indexed = df.set_index("department")

result = (
    indexed.groupby(level="department")["salary"]
    .mean()
)

print("\nAverage salary using index level:")
print(result)
```

# =============================================================================

# 15. GROUPING BY BOOLEAN

# =============================================================================

def grouping_boolean_column(df: pd.DataFrame) -> None:
"""Group records based on a Boolean column."""

```
print("\n" + "=" * 80)
print("15. GROUPING BY BOOLEAN")
print("=" * 80)

result = (
    df.groupby("is_remote", as_index=False)
    .agg(
        employee_count=("employee_id", "count"),
        average_salary=("salary", "mean"),
        average_performance=("performance_score", "mean"),
    )
)

print("\nRemote vs office employees:")
print(result.to_string(index=False))
```

# =============================================================================

# 16. GROUPING BY CATEGORICAL DATA

# =============================================================================

def grouping_categorical_data(
df: pd.DataFrame,
) -> None:
"""
Demonstrate categorical grouping.

```
observed=True tells Pandas to return only categories that
actually occur in the data.
"""

print("\n" + "=" * 80)
print("16. CATEGORICAL GROUPING")
print("=" * 80)

categorical_df = df.copy()

categorical_df["department"] = pd.Categorical(
    categorical_df["department"],
    categories=[
        "IT",
        "HR",
        "Finance",
        "Marketing",
        "Operations",
    ],
)

result = (
    categorical_df
    .groupby(
        "department",
        observed=True,
    )["salary"]
    .mean()
)

print("\nAverage salary by categorical department:")
print(result)
```

# =============================================================================

# 17. GROUP SIZE

# =============================================================================

def group_size(df: pd.DataFrame) -> None:
"""Compare size() and count()."""

```
print("\n" + "=" * 80)
print("17. GROUP SIZE")
print("=" * 80)

size_result = (
    df.groupby("department")
    .size()
)

count_result = (
    df.groupby("department")["salary"]
    .count()
)

print("\nsize():")
print(size_result)

print("\ncount() on salary:")
print(count_result)

print(
    """
```

Difference:

```
size()
    Counts rows in each group.

count()
    Counts non-missing values in the selected column.
```

Therefore, size() and count() can differ when missing values exist.
"""
)

# =============================================================================

# 18. FIRST, LAST, NTH

# =============================================================================

def first_last_nth(df: pd.DataFrame) -> None:
"""Demonstrate first(), last(), and nth()."""

```
print("\n" + "=" * 80)
print("18. FIRST(), LAST(), NTH()")
print("=" * 80)

grouped = df.groupby("department")

print("\nFirst employee in each department:")
print(
    grouped.first()[
        ["name", "salary"]
    ]
)

print("\nLast employee in each department:")
print(
    grouped.last()[
        ["name", "salary"]
    ]
)

print("\nSecond employee in each department:")
print(
    grouped.nth(1)[
        ["name", "salary"]
    ]
)
```

# =============================================================================

# 19. GROUPBY FILTER

# =============================================================================

def group_filter(df: pd.DataFrame) -> None:
"""
Keep only groups satisfying a condition.

```
Example:
    Keep departments whose average salary is > 70K.
"""

print("\n" + "=" * 80)
print("19. GROUPBY FILTER")
print("=" * 80)

result = (
    df.groupby(
        "department",
        group_keys=False,
    )
    .filter(
        lambda group:
        group["salary"].mean() > 70000
    )
)

print("\nDepartments with average salary > 70K:")
print(
    result[
        ["name", "department", "salary"]
    ].to_string(index=False)
)
```

# =============================================================================

# 20. GROUPBY TRANSFORM

# =============================================================================

def group_transform(df: pd.DataFrame) -> None:
"""
Demonstrate transform().

```
Unlike agg(), transform() returns a result aligned with
the original DataFrame.

This makes transform() extremely useful for feature engineering.
"""

print("\n" + "=" * 80)
print("20. GROUPBY TRANSFORM")
print("=" * 80)

result = df.copy()

result["department_avg_salary"] = (
    result.groupby("department")["salary"]
    .transform("mean")
)

result["salary_vs_department_avg"] = (
    result["salary"]
    - result["department_avg_salary"]
)

print("\nSalary compared with department average:")
print(
    result[
        [
            "name",
            "department",
            "salary",
            "department_avg_salary",
            "salary_vs_department_avg",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 21. TRANSFORM VS AGG

# =============================================================================

def transform_vs_aggregation(df: pd.DataFrame) -> None:
"""Explain the difference between aggregation and transformation."""

```
print("\n" + "=" * 80)
print("21. TRANSFORM VS AGG")
print("=" * 80)

aggregation = (
    df.groupby("department")["salary"]
    .mean()
)

transformation = (
    df.groupby("department")["salary"]
    .transform("mean")
)

print("\nAggregation:")
print(aggregation)

print("\nTransform:")
print(transformation.to_string(index=False))

print(
    """
```

Key difference:

```
agg()
    Returns one result per group.

transform()
    Returns one value per original row.
```

Use:

```
agg()
    For summaries and reports.

transform()
    For adding group-level features back to the original dataset.
```

"""
)

# =============================================================================

# 22. GROUP RANKING

# =============================================================================

def group_ranking(df: pd.DataFrame) -> None:
"""Rank employees within their departments."""

```
print("\n" + "=" * 80)
print("22. GROUP RANKING")
print("=" * 80)

result = df.copy()

result["department_salary_rank"] = (
    result.groupby("department")["salary"]
    .rank(
        ascending=False,
        method="dense",
    )
)

result = result.sort_values(
    [
        "department",
        "department_salary_rank",
    ]
)

print("\nSalary rank within department:")
print(
    result[
        [
            "name",
            "department",
            "salary",
            "department_salary_rank",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 23. GROUP PERCENTAGE

# =============================================================================

def group_percentage(df: pd.DataFrame) -> None:
"""
Calculate each employee's percentage of the total
department salary.
"""

```
print("\n" + "=" * 80)
print("23. GROUP PERCENTAGE")
print("=" * 80)

result = df.copy()

department_total = (
    result.groupby("department")["salary"]
    .transform("sum")
)

result["salary_percentage_of_department"] = (
    result["salary"]
    / department_total
    * 100
)

print(
    result[
        [
            "name",
            "department",
            "salary",
            "salary_percentage_of_department",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 24. GROUP CUMULATIVE SUM

# =============================================================================

def group_cumulative_sum(df: pd.DataFrame) -> None:
"""
Calculate cumulative salary within each department.

```
Sorting first is important when the cumulative operation
depends on time or another ordered variable.
"""

print("\n" + "=" * 80)
print("24. GROUP CUMULATIVE SUM")
print("=" * 80)

result = df.sort_values(
    [
        "department",
        "joining_date",
    ]
).copy()

result["department_cumulative_salary"] = (
    result.groupby("department")["salary"]
    .cumsum()
)

print(
    result[
        [
            "name",
            "department",
            "joining_date",
            "salary",
            "department_cumulative_salary",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 25. GROUP CUMULATIVE COUNT

# =============================================================================

def group_cumulative_count(df: pd.DataFrame) -> None:
"""Calculate employee sequence within each department."""

```
print("\n" + "=" * 80)
print("25. GROUP CUMULATIVE COUNT")
print("=" * 80)

result = df.sort_values(
    [
        "department",
        "joining_date",
    ]
).copy()

result["employee_sequence"] = (
    result.groupby("department")
    .cumcount()
    \+ 1
)

print(
    result[
        [
            "name",
            "department",
            "joining_date",
            "employee_sequence",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 26. GROUPED SORTING

# =============================================================================

def grouped_sorting(df: pd.DataFrame) -> None:
"""Sort employees within departments by salary."""

```
print("\n" + "=" * 80)
print("26. GROUPED SORTING")
print("=" * 80)

result = df.sort_values(
    by=[
        "department",
        "salary",
    ],
    ascending=[
        True,
        False,
    ],
)

print(
    result[
        [
            "name",
            "department",
            "salary",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 27. TOP EMPLOYEE PER GROUP

# =============================================================================

def top_employee_per_group(
df: pd.DataFrame,
) -> None:
"""Find the highest-paid employee in every department."""

```
print("\n" + "=" * 80)
print("27. TOP EMPLOYEE PER GROUP")
print("=" * 80)

result = (
    df.loc[
        df.groupby("department")["salary"]
        .idxmax()
    ]
    .sort_values("department")
)

print(
    result[
        [
            "name",
            "department",
            "salary",
            "performance_score",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 28. LOWEST EMPLOYEE PER GROUP

# =============================================================================

def lowest_employee_per_group(
df: pd.DataFrame,
) -> None:
"""Find the lowest-paid employee in each department."""

```
print("\n" + "=" * 80)
print("28. LOWEST EMPLOYEE PER GROUP")
print("=" * 80)

result = (
    df.loc[
        df.groupby("department")["salary"]
        .idxmin()
    ]
    .sort_values("department")
)

print(
    result[
        [
            "name",
            "department",
            "salary",
            "performance_score",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 29. VALUE COUNTS WITHIN GROUP

# =============================================================================

def grouped_value_counts(
df: pd.DataFrame,
) -> None:
"""Count city occurrences within each department."""

```
print("\n" + "=" * 80)
print("29. GROUPED VALUE COUNTS")
print("=" * 80)

result = (
    df.groupby("department")["city"]
    .value_counts()
)

print("\nCity distribution within departments:")
print(result)
```

# =============================================================================

# 30. GROUPED BOOLEAN ANALYSIS

# =============================================================================

def grouped_boolean_analysis(
df: pd.DataFrame,
) -> None:
"""Analyze remote employees within each department."""

```
print("\n" + "=" * 80)
print("30. GROUPED BOOLEAN ANALYSIS")
print("=" * 80)

result = (
    df.groupby("department")
    .agg(
        total_employees=("employee_id", "count"),
        remote_employees=("is_remote", "sum"),
        remote_rate=("is_remote", "mean"),
    )
    .reset_index()
)

result["remote_rate_percent"] = (
    result["remote_rate"] * 100
)

print(
    result[
        [
            "department",
            "total_employees",
            "remote_employees",
            "remote_rate_percent",
        ]
    ].to_string(index=False)
)
```

# =============================================================================

# 31. GROUPED STATISTICS

# =============================================================================

def grouped_statistics(
df: pd.DataFrame,
) -> None:
"""Generate a compact statistical report."""

```
print("\n" + "=" * 80)
print("31. GROUPED STATISTICS")
print("=" * 80)

result = (
    df.groupby("department")
    .agg(
        employees=("employee_id", "size"),
        salary_mean=("salary", "mean"),
        salary_median=("salary", "median"),
        salary_std=("salary", "std"),
        experience_mean=("experience", "mean"),
        performance_mean=("performance_score", "mean"),
        projects_total=("projects_completed", "sum"),
    )
    .round(2)
)

print(result)
```

# =============================================================================

# 32. MISSING VALUES AND GROUPBY

# =============================================================================

def missing_values_groupby(
df: pd.DataFrame,
) -> None:
"""
Demonstrate how missing values affect grouping.

```
dropna=True is the default behavior for group keys.
"""

print("\n" + "=" * 80)
print("32. MISSING VALUES AND GROUPBY")
print("=" * 80)

working_df = df.copy()

working_df.loc[
    working_df["employee_id"] == 108,
    "city",
] = np.nan

default_result = (
    working_df.groupby("city")
    .size()
)

include_missing = (
    working_df.groupby(
        "city",
        dropna=False,
    )
    .size()
)

print("\nDefault GroupBy — missing city excluded:")
print(default_result)

print("\nGroupBy(dropna=False) — missing city included:")
print(include_missing)
```

# =============================================================================

# 33. AS_INDEX=False

# =============================================================================

def as_index_example(
df: pd.DataFrame,
) -> None:
"""Demonstrate as_index=False."""

```
print("\n" + "=" * 80)
print("33. AS_INDEX=False")
print("=" * 80)

result = (
    df.groupby(
        "department",
        as_index=False,
    )
    .agg(
        average_salary=("salary", "mean"),
        employee_count=("employee_id", "count"),
    )
)

print(result)
```

# =============================================================================

# 34. RESET_INDEX

# =============================================================================

def reset_index_example(
df: pd.DataFrame,
) -> None:
"""Demonstrate reset_index()."""

```
print("\n" + "=" * 80)
print("34. RESET_INDEX")
print("=" * 80)

result = (
    df.groupby("department")["salary"]
    .mean()
    .reset_index(name="average_salary")
)

print(result)
```

# =============================================================================

# 35. GROUPBY + PIVOT-STYLE ANALYSIS

# =============================================================================

def grouped_pivot_analysis(
df: pd.DataFrame,
) -> None:
"""Create a department-city salary summary."""

```
print("\n" + "=" * 80)
print("35. GROUPBY + PIVOT-STYLE ANALYSIS")
print("=" * 80)

result = (
    df.groupby(
        ["department", "city"],
        as_index=False,
    )["salary"]
    .mean()
    .pivot(
        index="department",
        columns="city",
        values="salary",
    )
)

print("\nAverage salary by department and city:")
print(result.round(2))
```

# =============================================================================

# 36. FEATURE ENGINEERING WITH GROUPBY

# =============================================================================

def group_feature_engineering(
df: pd.DataFrame,
) -> None:
"""
Create ML-oriented features using group statistics.

```
Features:
    - department average salary
    - salary difference from department average
    - department average performance
    - salary percentile within department
"""

print("\n" + "=" * 80)
print("36. GROUP FEATURE ENGINEERING")
print("=" * 80)

result = df.copy()

result["department_avg_salary"] = (
    result.groupby("department")["salary"]
    .transform("mean")
)

result["salary_difference_from_department_avg"] = (
    result["salary"]
    - result["department_avg_salary"]
)

result["department_avg_performance"] = (
    result.groupby("department")["performance_score"]
    .transform("mean")
)

result["salary_percentile_in_department"] = (
    result.groupby("department")["salary"]
    .rank(
        pct=True,
    )
)

print(
    result[
        [
            "name",
            "department",
            "salary",
            "department_avg_salary",
            "salary_difference_from_department_avg",
            "department_avg_performance",
            "salary_percentile_in_department",
        ]
    ].round(2).to_string(index=False)
)
```

# =============================================================================

# 37. GROUPED NORMALIZATION

# =============================================================================

def grouped_normalization(
df: pd.DataFrame,
) -> None:
"""
Normalize salary within each department using a z-score.

```
Formula:

    z = (x - group_mean) / group_std

This is useful when groups have very different scales.
"""

print("\n" + "=" * 80)
print("37. GROUPED NORMALIZATION")
print("=" * 80)

result = df.copy()

group = result.groupby("department")["salary"]

group_mean = group.transform("mean")
group_std = group.transform("std")

result["salary_group_zscore"] = (
    (result["salary"] - group_mean)
    / group_std
)

print(
    result[
        [
            "name",
            "department",
            "salary",
            "salary_group_zscore",
        ]
    ].round(2).to_string(index=False)
)
```

# =============================================================================

# 38. MACHINE LEARNING GROUP FEATURES

# =============================================================================

def ml_group_features(
df: pd.DataFrame,
) -> pd.DataFrame:
"""
Create a compact set of group-based features.

```
Important:
    In a real ML project, group statistics must be calculated
    in a leakage-safe manner. For example, statistics used for
    validation/test rows should not be learned from those same
    validation/test target outcomes.
"""

result = df.copy()

result["department_avg_salary"] = (
    result.groupby("department")["salary"]
    .transform("mean")
)

result["department_avg_performance"] = (
    result.groupby("department")["performance_score"]
    .transform("mean")
)

result["department_employee_count"] = (
    result.groupby("department")["employee_id"]
    .transform("count")
)

result["city_avg_salary"] = (
    result.groupby("city")["salary"]
    .transform("mean")
)

return result
```

# =============================================================================

# 39. DATA LEAKAGE WARNING

# =============================================================================

def demonstrate_data_leakage_warning() -> None:
"""
Explain why group-based features can leak information.
"""

```
print("\n" + "=" * 80)
print("39. GROUPBY AND DATA LEAKAGE")
print("=" * 80)

print(
    """
```

GroupBy can create powerful ML features, but these features must be
created carefully.

Example:

```
df["department_target_mean"] = (
    df.groupby("department")["target"]
    .transform("mean")
)
```

This can leak target information because every row's feature may
include the target value of that same row.

For supervised ML:

```
Avoid calculating target-derived group statistics on the full
dataset before splitting.
```

Safer approaches include:

```
1. Split data first.
2. Fit group statistics on training data.
3. Apply those statistics to validation/test data.
4. Use cross-validation when appropriate.
5. Consider out-of-fold target encoding for supervised features.
```

The key rule is:

```
Never let information from validation/test/future observations
influence features used to train the model.
```

"""
)

# =============================================================================

# 40. REUSABLE GROUPBY FUNCTIONS

# =============================================================================

def department_summary(
df: pd.DataFrame,
) -> pd.DataFrame:
"""
Return a reusable department summary.

```
Returns:
    pd.DataFrame:
        One row per department.
"""

return (
    df.groupby(
        "department",
        as_index=False,
    )
    .agg(
        employee_count=("employee_id", "count"),
        average_salary=("salary", "mean"),
        median_salary=("salary", "median"),
        average_experience=("experience", "mean"),
        average_performance=("performance_score", "mean"),
        total_projects=("projects_completed", "sum"),
    )
    .round(2)
)
```

def city_summary(
df: pd.DataFrame,
) -> pd.DataFrame:
"""
Return salary and performance statistics by city.
"""

```
return (
    df.groupby(
        "city",
        as_index=False,
    )
    .agg(
        employee_count=("employee_id", "count"),
        average_salary=("salary", "mean"),
        average_performance=("performance_score", "mean"),
    )
    .round(2)
)
```

def top_department(
df: pd.DataFrame,
) -> pd.DataFrame:
"""
Return the highest-average-salary department.

```
Note:
    This function reports the highest average salary; it does not
    imply that the department is generally "best."
"""

summary = department_summary(df)

return (
    summary.sort_values(
        "average_salary",
        ascending=False,
    )
    .head(1)
)
```

# =============================================================================

# 41. REUSABLE FUNCTIONS DEMONSTRATION

# =============================================================================

def reusable_functions_demo(
df: pd.DataFrame,
) -> None:
"""Demonstrate reusable GroupBy functions."""

```
print("\n" + "=" * 80)
print("41. REUSABLE GROUPBY FUNCTIONS")
print("=" * 80)

print("\nDepartment summary:")
print(
    department_summary(df)
    .to_string(index=False)
)

print("\nCity summary:")
print(
    city_summary(df)
    .to_string(index=False)
)

print("\nDepartment with highest average salary:")
print(
    top_department(df)
    .to_string(index=False)
)
```

# =============================================================================

# 42. COMMON GROUPBY MISTAKES

# =============================================================================

def common_mistakes() -> None:
"""Display common GroupBy mistakes and their corrections."""

```
print("\n" + "=" * 80)
print("42. COMMON GROUPBY MISTAKES")
print("=" * 80)

print(
    """
```

1. Forgetting which column is being aggregated

   Clear:
   df.groupby("department")["salary"].mean()

2. Confusing count() and size()

   count():
   Counts non-null values in a selected column.

   size():
   Counts rows in each group.

3. Using agg() when transform() is needed

   agg():
   One result per group.

   transform():
   One result per original row.

4. Forgetting reset_index()

   GroupBy often produces the grouping column as an index.

   Use:
   result.reset_index()

   Or:
   groupby(..., as_index=False)

5. Ignoring missing group keys

   By default, missing group keys can be excluded.

   Use:
   groupby(..., dropna=False)

   when missing groups need to be retained.

6. Sorting before cumulative calculations incorrectly

   For time-dependent calculations:

   ```
    Sort first.
    Then group.
    Then calculate cumulative values.
   ```

7. Creating target-based group features before splitting

   This can create data leakage.

   Split first and calculate training-derived statistics safely.

8. Using loops unnecessarily

   Prefer:

   ```
    groupby()
    agg()
    transform()
    rank()
    cumsum()
   ```

   over manual Python loops for most tabular operations.
   """
   )

# =============================================================================

# 43. GROUPBY CHEAT SHEET

# =============================================================================

def groupby_cheat_sheet() -> None:
"""Display a quick GroupBy reference."""

```
print("\n" + "=" * 80)
print("43. GROUPBY CHEAT SHEET")
print("=" * 80)

print(
    """
```

Basic:
df.groupby("department")

Mean:
df.groupby("department")["salary"].mean()

Sum:
df.groupby("department")["salary"].sum()

Count:
df.groupby("department")["employee_id"].count()

Rows:
df.groupby("department").size()

Multiple:
df.groupby("department")["salary"].agg(
["mean", "min", "max"]
)

Multiple columns:
df.groupby("department").agg({
"salary": "mean",
"experience": "mean",
})

Named aggregation:
df.groupby("department").agg(
average_salary=("salary", "mean"),
employee_count=("employee_id", "count"),
)

Multiple group keys:
df.groupby(["department", "city"])["salary"].mean()

Keep normal DataFrame:
df.groupby(
"department",
as_index=False,
).mean(numeric_only=True)

Filter groups:
df.groupby("department").filter(
lambda group: group["salary"].mean() > 70000
)

Transform:
df.groupby("department")["salary"].transform("mean")

Rank:
df.groupby("department")["salary"].rank()

Cumulative sum:
df.groupby("department")["salary"].cumsum()

First:
df.groupby("department").first()

Last:
df.groupby("department").last()

Nth:
df.groupby("department").nth(1)

Top row per group:
df.loc[
df.groupby("department")["salary"].idxmax()
]

Reset index:
df.groupby("department")["salary"].mean().reset_index()

Missing groups:
df.groupby(
"department",
dropna=False,
)

Categorical groups:
df.groupby(
"department",
observed=True,
)
"""
)

# =============================================================================

# 44. COMPLETE REAL-WORLD WORKFLOW

# =============================================================================

def complete_workflow(
df: pd.DataFrame,
) -> None:
"""
Complete department analytics workflow.

```
Scenario:
    Build an HR analytics report containing:
        - employee count
        - average salary
        - median salary
        - average experience
        - average performance
        - remote employee percentage
        - total projects
"""

print("\n" + "=" * 80)
print("44. COMPLETE REAL-WORLD GROUPBY WORKFLOW")
print("=" * 80)

report = (
    df.groupby(
        "department",
        as_index=False,
    )
    .agg(
        employee_count=("employee_id", "count"),
        average_salary=("salary", "mean"),
        median_salary=("salary", "median"),
        average_experience=("experience", "mean"),
        average_performance=(
            "performance_score",
            "mean",
        ),
        total_projects=(
            "projects_completed",
            "sum",
        ),
        remote_rate=("is_remote", "mean"),
    )
)

report["remote_rate_percent"] = (
    report["remote_rate"] * 100
)

report = report.drop(
    columns="remote_rate"
)

report = report.sort_values(
    "average_salary",
    ascending=False,
)

numeric_columns = [
    "average_salary",
    "median_salary",
    "average_experience",
    "average_performance",
    "remote_rate_percent",
]

report[numeric_columns] = (
    report[numeric_columns].round(2)
)

print("\nFinal HR analytics report:")
print(report.to_string(index=False))
```

# =============================================================================

# 45. MAIN FUNCTION

# =============================================================================

def main() -> None:
"""Run the complete Pandas GroupBy tutorial."""

```
print("=" * 80)
print("PANDAS GROUPBY — BEGINNER TO ADVANCED")
print("=" * 80)

df = create_sample_data()

display_dataset(df)

explain_split_apply_combine()

basic_groupby(df)
groupby_sum(df)
groupby_count(df)
groupby_min_max(df)
multiple_aggregations(df)
aggregate_multiple_columns(df)
named_aggregations(df)

multiple_group_columns(df)
multiple_group_columns_dataframe(df)
grouping_by_index(df)
grouping_boolean_column(df)
grouping_categorical_data(df)

group_size(df)
first_last_nth(df)
group_filter(df)

group_transform(df)
transform_vs_aggregation(df)
group_ranking(df)
group_percentage(df)
group_cumulative_sum(df)
group_cumulative_count(df)

grouped_sorting(df)
top_employee_per_group(df)
lowest_employee_per_group(df)
grouped_value_counts(df)
grouped_boolean_analysis(df)
grouped_statistics(df)

missing_values_groupby(df)
as_index_example(df)
reset_index_example(df)
grouped_pivot_analysis(df)

group_feature_engineering(df)
grouped_normalization(df)
ml_group_features(df)

demonstrate_data_leakage_warning()

reusable_functions_demo(df)
common_mistakes()
groupby_cheat_sheet()
complete_workflow(df)

print("\n" + "=" * 80)
print("GROUPBY TUTORIAL COMPLETED")
print("=" * 80)
```

# =============================================================================

# 46. SCRIPT ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()

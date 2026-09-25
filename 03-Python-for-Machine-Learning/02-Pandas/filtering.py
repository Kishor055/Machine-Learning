"""
Pandas Filtering — Beginner to Advanced
=======================================

File:
03-Python-for-Machine-Learning/02-Pandas/filtering.py

Description:
A practical guide to filtering rows and columns in Pandas.

Topics Covered:
1. Creating a sample DataFrame
2. Basic row filtering
3. Comparison operators
4. Multiple conditions
5. AND / OR / NOT conditions
6. Filtering with isin()
7. Filtering with between()
8. Filtering with query()
9. String filtering with str methods
10. Missing-value filtering
11. Duplicate-aware filtering
12. Date-based filtering
13. Numeric range filtering
14. Filtering categorical data
15. Filtering with callable conditions
16. Filtering rows and columns together
17. Advanced boolean masks
18. Filtering using index
19. ML-oriented dataset filtering
20. Avoiding common filtering mistakes
21. Reusable filtering functions

Requirements:
pip install pandas numpy

Run:
python filtering.py

Author:
Kishor Patil

Learning Goal:
Learn how to efficiently select the exact rows and columns needed
for data analysis, data cleaning, and machine learning workflows.

Important:
Pandas boolean conditions use:
&  -> AND
|  -> OR
~  -> NOT

```
Do NOT use:
    and
    or
    not

Parentheses around each condition are strongly recommended.
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
Create a realistic employee dataset for filtering demonstrations.

```
Returns:
    pd.DataFrame:
        Sample employee dataset.
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
    ],
}

df = pd.DataFrame(data)

# Convert text dates into Pandas datetime objects.
df["joining_date"] = pd.to_datetime(df["joining_date"])

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

# 4. BASIC COMPARISON FILTERING

# =============================================================================

def basic_comparison_filtering(df: pd.DataFrame) -> None:
"""
Demonstrate basic comparison operators.

```
Operators:
    >   greater than
    <   less than
    >=  greater than or equal
    <=  less than or equal
    ==  equal
    !=  not equal
"""

print("\n" + "=" * 80)
print("4. BASIC COMPARISON FILTERING")
print("=" * 80)

# Employees older than 30.
result = df[df["age"] > 30]

print("\nEmployees older than 30:")
print(result[["name", "age", "department"]].to_string(index=False))

# Employees with salary greater than or equal to 80,000.
result = df[df["salary"] >= 80000]

print("\nEmployees earning >= 80,000:")
print(result[["name", "salary"]].to_string(index=False))

# Employees with exactly 5 years of experience.
result = df[df["experience"] == 5]

print("\nEmployees with exactly 5 years of experience:")
print(result[["name", "experience"]].to_string(index=False))

# Employees who are not from Pune.
result = df[df["city"] != "Pune"]

print("\nEmployees not from Pune:")
print(result[["name", "city"]].to_string(index=False))
```

# =============================================================================

# 5. FILTER USING BOOLEAN MASK

# =============================================================================

def boolean_mask_example(df: pd.DataFrame) -> None:
"""
Demonstrate how Pandas filtering works internally.

```
A condition such as:

    df["salary"] > 70000

produces a Boolean Series:

    True
    False
    True
    ...

Pandas uses this Boolean mask to select rows.
"""

print("\n" + "=" * 80)
print("5. BOOLEAN MASK")
print("=" * 80)

mask = df["salary"] > 70000

print("\nBoolean mask:")
print(mask.to_string())

filtered_df = df[mask]

print("\nFiltered DataFrame:")
print(filtered_df[["name", "salary"]].to_string(index=False))
```

# =============================================================================

# 6. MULTIPLE CONDITIONS — AND

# =============================================================================

def multiple_conditions_and(df: pd.DataFrame) -> None:
"""
Filter using multiple conditions with &.

```
Example:
    Age > 30 AND salary > 70000
"""

print("\n" + "=" * 80)
print("6. MULTIPLE CONDITIONS — AND")
print("=" * 80)

result = df[
    (df["age"] > 30)
    & (df["salary"] > 70000)
]

print("\nAge > 30 AND salary > 70,000:")
print(
    result[
        ["name", "age", "salary"]
    ].to_string(index=False)
)
```

# =============================================================================

# 7. MULTIPLE CONDITIONS — OR

# =============================================================================

def multiple_conditions_or(df: pd.DataFrame) -> None:
"""
Filter using OR.

```
Example:
    Department is IT OR Finance
"""

print("\n" + "=" * 80)
print("7. MULTIPLE CONDITIONS — OR")
print("=" * 80)

result = df[
    (df["department"] == "IT")
    | (df["department"] == "Finance")
]

print("\nEmployees from IT OR Finance:")
print(
    result[
        ["name", "department", "salary"]
    ].to_string(index=False)
)
```

# =============================================================================

# 8. NOT CONDITION

# =============================================================================

def not_condition(df: pd.DataFrame) -> None:
"""
Demonstrate the ~ operator.

```
Example:
    NOT IT department
"""

print("\n" + "=" * 80)
print("8. NOT CONDITION")
print("=" * 80)

result = df[
    ~(df["department"] == "IT")
]

print("\nEmployees who are NOT in IT:")
print(
    result[
        ["name", "department"]
    ].to_string(index=False)
)
```

# =============================================================================

# 9. ISIN()

# =============================================================================

def isin_filtering(df: pd.DataFrame) -> None:
"""
Use isin() when filtering against multiple possible values.

```
This is cleaner than writing many OR conditions.
"""

print("\n" + "=" * 80)
print("9. ISIN() FILTERING")
print("=" * 80)

departments = ["IT", "Finance"]

result = df[
    df["department"].isin(departments)
]

print("\nDepartments in IT or Finance:")
print(
    result[
        ["name", "department"]
    ].to_string(index=False)
)

cities = ["Pune", "Mumbai"]

result = df[
    df["city"].isin(cities)
]

print("\nEmployees from Pune or Mumbai:")
print(
    result[
        ["name", "city"]
    ].to_string(index=False)
)
```

# =============================================================================

# 10. NEGATIVE ISIN()

# =============================================================================

def negative_isin(df: pd.DataFrame) -> None:
"""Filter rows whose value is NOT present in a list."""

```
print("\n" + "=" * 80)
print("10. NEGATIVE ISIN()")
print("=" * 80)

excluded_departments = ["HR", "Marketing"]

result = df[
    ~df["department"].isin(excluded_departments)
]

print("\nEmployees outside HR and Marketing:")
print(
    result[
        ["name", "department"]
    ].to_string(index=False)
)
```

# =============================================================================

# 11. BETWEEN()

# =============================================================================

def between_filtering(df: pd.DataFrame) -> None:
"""
Demonstrate between() for range filtering.

```
By default, both boundaries are inclusive.
"""

print("\n" + "=" * 80)
print("11. BETWEEN() FILTERING")
print("=" * 80)

result = df[
    df["salary"].between(60000, 80000)
]

print("\nSalary between 60,000 and 80,000:")
print(
    result[
        ["name", "salary"]
    ].to_string(index=False)
)

result = df[
    df["age"].between(25, 35)
]

print("\nAge between 25 and 35:")
print(
    result[
        ["name", "age"]
    ].to_string(index=False)
)
```

# =============================================================================

# 12. STRING FILTERING

# =============================================================================

def string_filtering(df: pd.DataFrame) -> None:
"""
Demonstrate string filtering using Pandas .str methods.

```
Common methods:
    contains()
    startswith()
    endswith()
    match()
    len()
"""

print("\n" + "=" * 80)
print("12. STRING FILTERING")
print("=" * 80)

# Names beginning with "A".
result = df[
    df["name"].str.startswith("A")
]

print("\nNames starting with A:")
print(
    result[
        ["name"]
    ].to_string(index=False)
)

# Cities containing "a".
result = df[
    df["city"].str.contains(
        "a",
        case=False,
        na=False,
    )
]

print("\nCities containing the letter 'a':")
print(
    result[
        ["name", "city"]
    ].to_string(index=False)
)

# Names containing "ra".
result = df[
    df["name"].str.contains(
        "ra",
        case=False,
        na=False,
    )
]

print("\nNames containing 'ra':")
print(
    result[
        ["name"]
    ].to_string(index=False)
)
```

# =============================================================================

# 13. MISSING VALUE FILTERING

# =============================================================================

def missing_value_filtering(df: pd.DataFrame) -> None:
"""
Filter rows with missing and non-missing values.

```
Important:
    Never use:
        df["age"] == np.nan

    Use:
        df["age"].isna()

    or:
        df["age"].notna()
"""

print("\n" + "=" * 80)
print("13. MISSING VALUE FILTERING")
print("=" * 80)

missing_age = df[
    df["age"].isna()
]

print("\nEmployees with missing age:")
print(
    missing_age[
        ["name", "age"]
    ].to_string(index=False)
)

valid_age = df[
    df["age"].notna()
]

print("\nEmployees with known age:")
print(
    valid_age[
        ["name", "age"]
    ].to_string(index=False)
)
```

# =============================================================================

# 14. BOOLEAN COLUMN FILTERING

# =============================================================================

def boolean_filtering(df: pd.DataFrame) -> None:
"""Filter rows using Boolean columns."""

```
print("\n" + "=" * 80)
print("14. BOOLEAN COLUMN FILTERING")
print("=" * 80)

remote_employees = df[
    df["is_remote"]
]

print("\nRemote employees:")
print(
    remote_employees[
        ["name", "is_remote"]
    ].to_string(index=False)
)

office_employees = df[
    ~df["is_remote"]
]

print("\nOffice employees:")
print(
    office_employees[
        ["name", "is_remote"]
    ].to_string(index=False)
)
```

# =============================================================================

# 15. QUERY()

# =============================================================================

def query_filtering(df: pd.DataFrame) -> None:
"""
Demonstrate DataFrame.query().

```
query() can make complex filters easier to read.

Example:
    df.query("age > 30 and salary > 70000")
"""

print("\n" + "=" * 80)
print("15. QUERY() FILTERING")
print("=" * 80)

result = df.query(
    "age > 30 and salary > 70000"
)

print("\nAge > 30 AND salary > 70,000:")
print(
    result[
        ["name", "age", "salary"]
    ].to_string(index=False)
)

result = df.query(
    "department == 'IT' or department == 'Finance'"
)

print("\nIT OR Finance:")
print(
    result[
        ["name", "department"]
    ].to_string(index=False)
)
```

# =============================================================================

# 16. QUERY() WITH VARIABLES

# =============================================================================

def query_with_variables(df: pd.DataFrame) -> None:
"""
Demonstrate external Python variables in query().

```
The @ symbol references a Python variable.
"""

print("\n" + "=" * 80)
print("16. QUERY() WITH VARIABLES")
print("=" * 80)

minimum_salary = 75000
target_department = "IT"

result = df.query(
    "salary >= @minimum_salary and department == @target_department"
)

print(
    f"\nSalary >= {minimum_salary:,} "
    f"and department == {target_department}:"
)

print(
    result[
        ["name", "department", "salary"]
    ].to_string(index=False)
)
```

# =============================================================================

# 17. FILTERING USING LOC

# =============================================================================

def loc_filtering(df: pd.DataFrame) -> None:
"""
Use .loc for conditional row selection and column selection.

```
Pattern:

    df.loc[condition, columns]
"""

print("\n" + "=" * 80)
print("17. LOC FILTERING")
print("=" * 80)

condition = (
    (df["salary"] > 70000)
    & (df["performance_score"] >= 85)
)

result = df.loc[
    condition,
    [
        "name",
        "salary",
        "performance_score",
    ],
]

print("\nHigh salary + strong performance:")
print(result.to_string(index=False))
```

# =============================================================================

# 18. FILTER ROWS AND COLUMNS TOGETHER

# =============================================================================

def rows_and_columns(df: pd.DataFrame) -> None:
"""Select specific rows based on a condition and only required columns."""

```
print("\n" + "=" * 80)
print("18. FILTER ROWS AND COLUMNS TOGETHER")
print("=" * 80)

result = df.loc[
    df["department"] == "IT",
    ["name", "city", "salary"],
]

print("\nIT employees — selected columns only:")
print(result.to_string(index=False))
```

# =============================================================================

# 19. INDEX-BASED FILTERING

# =============================================================================

def index_filtering(df: pd.DataFrame) -> None:
"""Demonstrate filtering using DataFrame index."""

```
print("\n" + "=" * 80)
print("19. INDEX-BASED FILTERING")
print("=" * 80)

indexed_df = df.set_index("employee_id")

result = indexed_df.loc[
    indexed_df.index >= 105
]

print("\nEmployees with ID >= 105:")
print(result.to_string())
```

# =============================================================================

# 20. DATE FILTERING

# =============================================================================

def date_filtering(df: pd.DataFrame) -> None:
"""Filter records using datetime conditions."""

```
print("\n" + "=" * 80)
print("20. DATE FILTERING")
print("=" * 80)

start_date = pd.Timestamp("2019-01-01")

result = df[
    df["joining_date"] >= start_date
]

print("\nEmployees who joined on or after 2019:")
print(
    result[
        ["name", "joining_date"]
    ].to_string(index=False)
)

result = df[
    df["joining_date"].between(
        "2018-01-01",
        "2021-12-31",
    )
]

print("\nEmployees who joined between 2018 and 2021:")
print(
    result[
        ["name", "joining_date"]
    ].to_string(index=False)
)
```

# =============================================================================

# 21. MULTI-RANGE FILTERING

# =============================================================================

def multi_range_filtering(df: pd.DataFrame) -> None:
"""Combine several numeric ranges."""

```
print("\n" + "=" * 80)
print("21. MULTI-RANGE FILTERING")
print("=" * 80)

condition = (
    df["salary"].between(60000, 70000)
    | df["salary"].between(85000, 100000)
)

result = df.loc[
    condition,
    ["name", "salary"],
]

print("\nSalary in either 60K–70K OR 85K–100K:")
print(result.to_string(index=False))
```

# =============================================================================

# 22. PERFORMANCE-BASED FILTERING

# =============================================================================

def performance_filtering(df: pd.DataFrame) -> None:
"""Create a business-style filtering rule."""

```
print("\n" + "=" * 80)
print("22. PERFORMANCE-BASED FILTERING")
print("=" * 80)

condition = (
    (df["performance_score"] >= 90)
    & (df["experience"] >= 5)
)

result = df.loc[
    condition,
    [
        "name",
        "experience",
        "performance_score",
        "salary",
    ],
]

print("\nExperienced employees with score >= 90:")
print(result.to_string(index=False))
```

# =============================================================================

# 23. COMPLEX FILTER

# =============================================================================

def complex_filter(df: pd.DataFrame) -> None:
"""
Combine multiple filtering techniques.

```
Business rule:

    Employee must:
        - be in IT or Finance
        - have salary >= 65,000
        - have performance >= 80
        - NOT be from Nagpur
"""

print("\n" + "=" * 80)
print("23. COMPLEX FILTER")
print("=" * 80)

condition = (
    df["department"].isin(["IT", "Finance"])
    & (df["salary"] >= 65000)
    & (df["performance_score"] >= 80)
    & (df["city"] != "Nagpur")
)

result = df.loc[
    condition,
    [
        "name",
        "department",
        "city",
        "salary",
        "performance_score",
    ],
]

print("\nEmployees matching all business rules:")
print(result.to_string(index=False))
```

# =============================================================================

# 24. FILTERING WITH CALLABLE

# =============================================================================

def callable_filtering(df: pd.DataFrame) -> None:
"""
Demonstrate callable-based filtering.

```
A callable receives the DataFrame and returns a Boolean mask.
"""

print("\n" + "=" * 80)
print("24. CALLABLE FILTERING")
print("=" * 80)

result = df.loc[
    lambda data: (
        (data["salary"] > 70000)
        & (data["performance_score"] > 85)
    )
]

print("\nSalary > 70K and performance > 85:")
print(
    result[
        ["name", "salary", "performance_score"]
    ].to_string(index=False)
)
```

# =============================================================================

# 25. FILTERING COLUMNS

# =============================================================================

def column_filtering(df: pd.DataFrame) -> None:
"""Demonstrate column filtering."""

```
print("\n" + "=" * 80)
print("25. COLUMN FILTERING")
print("=" * 80)

# Select columns by explicit names.
columns = [
    "name",
    "department",
    "salary",
]

result = df[columns]

print("\nSelected columns:")
print(result.to_string(index=False))

# Select columns containing "score".
score_columns = df.filter(
    like="score"
)

print("\nColumns containing 'score':")
print(score_columns.to_string(index=False))
```

# =============================================================================

# 26. FILTER COLUMNS USING REGEX

# =============================================================================

def regex_column_filtering(df: pd.DataFrame) -> None:
"""Filter columns using regular-expression matching."""

```
print("\n" + "=" * 80)
print("26. REGEX COLUMN FILTERING")
print("=" * 80)

result = df.filter(
    regex="salary|score|experience"
)

print("\nColumns matching salary, score, or experience:")
print(result.to_string(index=False))
```

# =============================================================================

# 27. FILTER USING VALUE COUNTS

# =============================================================================

def frequency_based_filtering(df: pd.DataFrame) -> None:
"""
Filter based on frequency.

```
Example:
    Keep only departments appearing at least 2 times.
"""

print("\n" + "=" * 80)
print("27. FREQUENCY-BASED FILTERING")
print("=" * 80)

counts = df["department"].value_counts()

valid_departments = counts[
    counts >= 2
].index

result = df[
    df["department"].isin(valid_departments)
]

print("\nDepartments appearing at least twice:")
print(
    result[
        ["name", "department"]
    ].to_string(index=False)
)
```

# =============================================================================

# 28. TOP-N FILTERING

# =============================================================================

def top_n_filtering(df: pd.DataFrame) -> None:
"""Select top-N records after sorting."""

```
print("\n" + "=" * 80)
print("28. TOP-N FILTERING")
print("=" * 80)

top_5 = (
    df.sort_values(
        "salary",
        ascending=False,
    )
    .head(5)
)

print("\nTop 5 salaries:")
print(
    top_5[
        ["name", "salary"]
    ].to_string(index=False)
)

top_performers = (
    df.nlargest(
        3,
        "performance_score",
    )
)

print("\nTop 3 performance scores:")
print(
    top_performers[
        ["name", "performance_score"]
    ].to_string(index=False)
)
```

# =============================================================================

# 29. FILTER AFTER GROUPBY

# =============================================================================

def groupby_filtering(df: pd.DataFrame) -> None:
"""
Filter groups using groupby().filter().

```
Example:
    Keep departments whose average salary exceeds 70K.
"""

print("\n" + "=" * 80)
print("29. GROUPBY FILTERING")
print("=" * 80)

result = df.groupby(
    "department",
    group_keys=False,
).filter(
    lambda group: group["salary"].mean() > 70000
)

print("\nDepartments with average salary > 70K:")
print(
    result[
        ["name", "department", "salary"]
    ].to_string(index=False)
)
```

# =============================================================================

# 30. FILTERING FOR MACHINE LEARNING

# =============================================================================

def ml_filtering(df: pd.DataFrame) -> None:
"""
Demonstrate dataset filtering before ML.

```
Example:
    Select records with:
        - known age
        - salary >= 60K
        - performance >= 75

In a real ML project, filtering rules should be based on
domain requirements rather than arbitrary thresholds.
"""

print("\n" + "=" * 80)
print("30. MACHINE LEARNING FILTERING")
print("=" * 80)

ml_data = df.loc[
    df["age"].notna()
    & (df["salary"] >= 60000)
    & (df["performance_score"] >= 75)
].copy()

features = [
    "age",
    "salary",
    "experience",
    "performance_score",
]

target = "is_remote"

X = ml_data[features]
y = ml_data[target]

print("\nFiltered ML dataset:")
print(ml_data.to_string(index=False))

print("\nFeatures (X):")
print(X.to_string(index=False))

print("\nTarget (y):")
print(y.to_string(index=False))
```

# =============================================================================

# 31. AVOID DATA LEAKAGE

# =============================================================================

def demonstrate_leakage_warning() -> None:
"""
Explain an important ML filtering rule.

```
Filtering based on information that would only be known after
the prediction event can create target leakage.

Example:
    Predict employee attrition.

Incorrect:
    Filter or select employees using a future column such as
    "exit_date" before training.

Correct:
    Only use information available at prediction time.
"""

print("\n" + "=" * 80)
print("31. MACHINE LEARNING DATA LEAKAGE")
print("=" * 80)

print(
    """
```

Data leakage warning:

```
Filtering is not automatically safe just because the syntax is correct.

Before filtering an ML dataset, ask:

1. Was this information available at prediction time?
2. Does the filter use the target variable?
3. Does the filter use future information?
4. Was the filtering threshold calculated using the test set?
5. Could this operation indirectly reveal the target?
```

Example:

```
BAD:
    salary_threshold = df["salary"].quantile(0.90)

if df contains both training and test data.

BETTER:
    Calculate data-dependent thresholds using the training data only.

Training:
    threshold = X_train["salary"].quantile(0.90)

Then apply that fixed threshold to validation/test data.
```

"""
)

# =============================================================================

# 32. REUSABLE FILTERING FUNCTIONS

# =============================================================================

def filter_high_performers(
df: pd.DataFrame,
minimum_score: float = 85,
) -> pd.DataFrame:
"""
Return employees meeting a minimum performance score.

```
Args:
    df: Input DataFrame.
    minimum_score: Minimum accepted performance score.

Returns:
    Filtered DataFrame copy.
"""

return df.loc[
    df["performance_score"] >= minimum_score
].copy()
```

def filter_salary_range(
df: pd.DataFrame,
minimum_salary: float,
maximum_salary: float,
) -> pd.DataFrame:
"""
Return employees whose salary falls within a given range.

```
Args:
    df: Input DataFrame.
    minimum_salary: Lower salary boundary.
    maximum_salary: Upper salary boundary.

Returns:
    Filtered DataFrame copy.
"""

return df.loc[
    df["salary"].between(
        minimum_salary,
        maximum_salary,
    )
].copy()
```

def filter_departments(
df: pd.DataFrame,
departments: list[str],
) -> pd.DataFrame:
"""
Return employees belonging to selected departments.

```
Args:
    df: Input DataFrame.
    departments: List of accepted departments.

Returns:
    Filtered DataFrame copy.
"""

return df.loc[
    df["department"].isin(departments)
].copy()
```

# =============================================================================

# 33. REUSABLE FUNCTION DEMONSTRATION

# =============================================================================

def reusable_filtering_demo(df: pd.DataFrame) -> None:
"""Demonstrate reusable filtering functions."""

```
print("\n" + "=" * 80)
print("33. REUSABLE FILTERING FUNCTIONS")
print("=" * 80)

high_performers = filter_high_performers(
    df,
    minimum_score=90,
)

print("\nPerformance >= 90:")
print(
    high_performers[
        ["name", "performance_score"]
    ].to_string(index=False)
)

salary_range = filter_salary_range(
    df,
    minimum_salary=60000,
    maximum_salary=70000,
)

print("\nSalary between 60K and 70K:")
print(
    salary_range[
        ["name", "salary"]
    ].to_string(index=False)
)

selected_departments = filter_departments(
    df,
    ["IT", "Finance"],
)

print("\nIT and Finance:")
print(
    selected_departments[
        ["name", "department"]
    ].to_string(index=False)
)
```

# =============================================================================

# 34. COMMON FILTERING MISTAKES

# =============================================================================

def common_mistakes() -> None:
"""Print common Pandas filtering mistakes and corrections."""

```
print("\n" + "=" * 80)
print("34. COMMON FILTERING MISTAKES")
print("=" * 80)

print(
    """
```

1. Using Python's `and` instead of Pandas `&`

   WRONG:
   df[(df["age"] > 30) and (df["salary"] > 70000)]

   CORRECT:
   df[(df["age"] > 30) & (df["salary"] > 70000)]

2. Forgetting parentheses

   WRONG:
   df[df["age"] > 30 & df["salary"] > 70000]

   CORRECT:
   df[(df["age"] > 30) & (df["salary"] > 70000)]

3. Comparing NaN with ==

   WRONG:
   df[df["age"] == np.nan]

   CORRECT:
   df[df["age"].isna()]

4. Using `in` directly on a Series

   WRONG:
   df[df["department"] in ["IT", "HR"]]

   CORRECT:
   df[df["department"].isin(["IT", "HR"])]

5. Filtering and then modifying without a clear copy

   Risky:
   filtered = df[df["salary"] > 70000]
   filtered["bonus"] = 5000

   Safer:
   filtered = df.loc[df["salary"] > 70000].copy()
   filtered["bonus"] = 5000

6. Calculating ML thresholds using the complete dataset

   This can cause data leakage.

   Prefer:
   Calculate data-dependent rules on training data only.
   """
   )

# =============================================================================

# 35. FILTERING CHEAT SHEET

# =============================================================================

def filtering_cheat_sheet() -> None:
"""Display a quick Pandas filtering reference."""

```
print("\n" + "=" * 80)
print("35. FILTERING CHEAT SHEET")
print("=" * 80)

print(
    """
```

Basic:
df[df["age"] > 30]

Equal:
df[df["department"] == "IT"]

Not equal:
df[df["department"] != "IT"]

AND:
df[(df["age"] > 30) & (df["salary"] > 70000)]

OR:
df[(df["department"] == "IT") | (df["department"] == "HR")]

NOT:
df[~(df["department"] == "IT")]

Multiple values:
df[df["department"].isin(["IT", "HR"])]

Exclude values:
df[~df["department"].isin(["IT", "HR"])]

Range:
df[df["salary"].between(60000, 80000)]

Missing:
df[df["age"].isna()]

Not missing:
df[df["age"].notna()]

String contains:
df[df["name"].str.contains("ra", case=False, na=False)]

Starts with:
df[df["name"].str.startswith("A")]

Ends with:
df[df["name"].str.endswith("a")]

Query:
df.query("age > 30 and salary > 70000")

LOC:
df.loc[df["salary"] > 70000, ["name", "salary"]]

Date:
df[df["joining_date"] >= "2020-01-01"]

Top N:
df.nlargest(5, "salary")

Group filtering:
df.groupby("department").filter(
lambda group: group["salary"].mean() > 70000
)

Column filtering:
df.filter(like="salary")

Regex columns:
df.filter(regex="salary|score")
"""
)

# =============================================================================

# 36. COMPLETE FILTERING WORKFLOW

# =============================================================================

def complete_workflow(df: pd.DataFrame) -> None:
"""
Demonstrate a realistic end-to-end filtering workflow.

```
Scenario:
    Find experienced high-performing IT/Finance employees
    from Pune or Mumbai with salary between 65K and 100K.
"""

print("\n" + "=" * 80)
print("36. COMPLETE REAL-WORLD FILTERING WORKFLOW")
print("=" * 80)

filtered = df.loc[
    df["age"].notna()
    & df["department"].isin(["IT", "Finance"])
    & df["city"].isin(["Pune", "Mumbai"])
    & df["salary"].between(65000, 100000)
    & (df["experience"] >= 5)
    & (df["performance_score"] >= 85)
].copy()

filtered = filtered.sort_values(
    by=[
        "performance_score",
        "salary",
    ],
    ascending=[
        False,
        False,
    ],
)

print("\nFinal filtered dataset:")
print(filtered.to_string(index=False))

print(
    f"\nRecords selected: {len(filtered)}"
)
```

# =============================================================================

# 37. MAIN FUNCTION

# =============================================================================

def main() -> None:
"""Run all Pandas filtering demonstrations."""

```
print("=" * 80)
print("PANDAS FILTERING — BEGINNER TO ADVANCED")
print("=" * 80)

df = create_sample_data()

display_dataset(df)

basic_comparison_filtering(df)
boolean_mask_example(df)
multiple_conditions_and(df)
multiple_conditions_or(df)
not_condition(df)
isin_filtering(df)
negative_isin(df)
between_filtering(df)
string_filtering(df)
missing_value_filtering(df)
boolean_filtering(df)
query_filtering(df)
query_with_variables(df)
loc_filtering(df)
rows_and_columns(df)
index_filtering(df)
date_filtering(df)
multi_range_filtering(df)
performance_filtering(df)
complex_filter(df)
callable_filtering(df)
column_filtering(df)
regex_column_filtering(df)
frequency_based_filtering(df)
top_n_filtering(df)
groupby_filtering(df)
ml_filtering(df)
demonstrate_leakage_warning()

reusable_filtering_demo(df)
common_mistakes()
filtering_cheat_sheet()
complete_workflow(df)

print("\n" + "=" * 80)
print("FILTERING TUTORIAL COMPLETED")
print("=" * 80)
```

# =============================================================================

# 38. SCRIPT ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()

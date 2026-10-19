# 🔧 Incorrect Data Types in Python

🌱 **GROW → INSPECT → IDENTIFY → CONVERT → VALIDATE → BUILD**

Welcome to **Incorrect Data Types**, the third topic in the Data Cleaning section of the Machine Learning repository. This guide explains how to identify incorrect data types, convert columns safely, validate transformed data, and prepare reliable datasets for data analysis and machine learning.

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [What Are Data Types?](#-what-are-data-types)
3. [Why Correct Data Types Matter](#-why-correct-data-types-matter)
4. [Common Data Types in Pandas](#-common-data-types-in-pandas)
5. [Inspecting Data Types](#-inspecting-data-types)
6. [Identifying Incorrect Data Types](#-identifying-incorrect-data-types)
7. [Converting Data Types](#-converting-data-types)
8. [Handling Numeric Values Stored as Strings](#-handling-numeric-values-stored-as-strings)
9. [Converting Dates and Times](#-converting-dates-and-times)
10. [Working with Boolean Values](#-working-with-boolean-values)
11. [Handling Categorical Data](#-handling-categorical-data)
12. [Nullable Data Types](#-nullable-data-types)
13. [Handling Conversion Errors](#-handling-conversion-errors)
14. [Validating Converted Data](#-validating-converted-data)
15. [Complete Data Cleaning Example](#-complete-data-cleaning-example)
16. [Data Types and Machine Learning](#-data-types-and-machine-learning)
17. [Common Mistakes](#-common-mistakes)
18. [Best Practices](#-best-practices)
19. [Mini Projects](#-mini-projects)
20. [Practice Exercises](#-practice-exercises)
21. [Quick Revision](#-quick-revision)
22. [Project Structure](#-project-structure)
23. [Learning Roadmap](#-learning-roadmap)

---

## 🎯 Introduction

Data cleaning is an essential step in every data science and machine learning project. One common problem is **incorrect data types**, where a column contains values in a format that does not match their intended meaning.

For example, a dataset may store student marks as strings instead of numbers.

```python
import pandas as pd

df = pd.DataFrame({
    "Student": ["Amit", "Priya", "Rahul"],
    "Age": ["20", "21", "19"],
    "Marks": ["85.5", "92.0", "78.5"]
})

print(df.dtypes)
```

The `Age` and `Marks` columns may be stored as `object` or `string` rather than numeric types.

This can cause problems when calculating averages, comparing values, creating charts, or training machine learning models.

**Goal:** Convert each column to the correct data type without losing information or silently introducing errors.

## 🧠 What Are Data Types?

A data type defines the kind of value a variable or DataFrame column stores.

| Data type | Meaning | Example |
|---|---|---|
| Integer | Whole numbers | `20`, `100` |
| Float | Decimal numbers | `85.5`, `3.14` |
| String | Text values | `"Python"` |
| Boolean | True or false | `True`, `False` |
| Datetime | Date and time values | `2026-10-19` |
| Category | Values from a limited set | `"Beginner"`, `"Advanced"` |
| Object | General Python objects | Mixed or arbitrary values |

### Example

```python
age = 21
marks = 89.5
name = "Kishor"
is_enrolled = True
```

Inspect the types using Python's `type()` function.

```python
print(type(age))
print(type(marks))
print(type(name))
print(type(is_enrolled))
```

Output:

```text
<class 'int'>
<class 'float'>
<class 'str'>
<class 'bool'>
```

Remember that a value's appearance does not determine its type. The value `"100"` is text, while `100` is an integer.

## 🚀 Why Correct Data Types Matter

Incorrect data types can cause:

- Mathematical calculations to fail or produce unexpected results.
- Dates to be sorted alphabetically instead of chronologically.
- Memory usage to increase unnecessarily.
- Machine learning estimators to reject unsuitable inputs.
- Incorrect comparisons and filtering.
- Data validation errors during deployment.
- Inconsistent behavior between training and production datasets.

For example:

```python
values = ["10", "20", "30"]

print(values[0] + values[1])
# 1020

numbers = [10, 20, 30]

print(numbers[0] + numbers[1])
# 30
```

The first operation joins strings; the second adds numbers.

## 🧰 Common Data Types in Pandas

Pandas provides several data types suitable for structured datasets.

| Pandas dtype | Purpose |
|---|---|
| `int64` | Standard integer values |
| `float64` | Decimal values |
| `object` | General Python objects, often mixed or textual data |
| `string` | Text data |
| `bool` | Boolean values without missing-value support |
| `boolean` | Nullable Boolean values |
| `Int64` | Nullable integer values |
| `Float64` | Nullable floating-point values |
| `datetime64[ns]` | Date and time values |
| `category` | Repeated values from a defined category set |

Note: `Int64` with a capital `I` is Pandas' nullable integer type. It differs from NumPy's `int64`.

## 🔍 Inspecting Data Types

Before changing a dataset, understand its current structure.

### 1. Check all column data types

```python
print(df.dtypes)
```

### 2. Display a complete summary

```python
df.info()
```

### 3. Inspect sample values

```python
print(df.head())
print(df.sample(min(5, len(df)), random_state=42))
```

### 4. Check the shape

```python
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
```

### 5. Find columns with object data types

```python
object_columns = df.select_dtypes(include=["object"]).columns

print(object_columns.tolist())
```

### 6. Inspect values in a particular column

```python
print(df["Age"].value_counts(dropna=False))
```

**Important:** An `object` column is not necessarily incorrect. It may legitimately contain names, addresses, identifiers, or mixed Python objects. Always check the column's intended meaning before converting it.

## 🕵️ Identifying Incorrect Data Types

Consider this dataset:

```python
import pandas as pd

df = pd.DataFrame({
    "Student": ["Amit", "Priya", "Rahul", "Neha"],
    "Age": ["20", "21", "19", "22"],
    "Marks": ["85.5", "92.0", "78.5", "88.0"],
    "Attendance": ["90%", "95%", "80%", "92%"],
    "Admission_Date": [
        "2026-01-10",
        "2026-01-15",
        "2026-02-01",
        "2026-02-10"
    ],
    "Passed": ["Yes", "Yes", "No", "Yes"]
})

print(df.dtypes)
```

Potential data type issues:

| Column | Current representation | Intended representation |
|---|---|---|
| `Student` | Text | String |
| `Age` | Text | Integer |
| `Marks` | Text | Float |
| `Attendance` | Text with `%` | Numeric percentage |
| `Admission_Date` | Text | Datetime |
| `Passed` | Text | Boolean or categorical |

The intended type depends on how the column will be used. For example, student IDs should usually remain identifiers rather than numeric measurements.

## 🔄 Converting Data Types

### 1. Convert using `astype()`

Use `astype()` when the values are already compatible with the target type.

```python
df["Age"] = df["Age"].astype("int64")
df["Marks"] = df["Marks"].astype("float64")

print(df.dtypes)
```

For a string column:

```python
df["Student"] = df["Student"].astype("string")
```

**Limitation:** `astype()` normally raises an error when values cannot be converted.

```python
values = pd.Series(["10", "20", "unknown"])

# Raises an error because "unknown" is not an integer.
# values.astype("int64")
```

Clean or handle invalid values before conversion.

### 2. Convert using `pd.to_numeric()`

`pd.to_numeric()` is useful for converting numeric strings into numbers.

```python
df["Age"] = pd.to_numeric(df["Age"])
df["Marks"] = pd.to_numeric(df["Marks"])

print(df[["Age", "Marks"]].dtypes)
```

Handle invalid values:

```python
values = pd.Series(["10", "20", "unknown", "30"])

converted = pd.to_numeric(values, errors="coerce")

print(converted)
```

Output:

```text
0    10.0
1    20.0
2     NaN
3    30.0
dtype: float64
```

`errors="coerce"` converts unparseable values to missing values (`NaN`). It does not decide whether those missing values should be removed, imputed, or retained.

### 3. Convert using `convert_dtypes()`

Pandas can infer more appropriate nullable data types.

```python
df = df.convert_dtypes()

print(df.dtypes)
```

This is useful for general type inference, but it does not replace semantic validation. A date stored as text, for example, may still require explicit datetime parsing.

### 4. Convert multiple columns

```python
df = df.astype({
    "Age": "int64",
    "Marks": "float64"
})
```

Use this approach only when all values in the selected columns are valid for the requested types.

## 🔢 Handling Numeric Values Stored as Strings

Real-world data often contains currency symbols, thousands separators, whitespace, or percentage signs.

Example:

```python
df = pd.DataFrame({
    "Salary": ["₹50,000", "₹65,000", "₹72,500"],
    "Marks": ["85.5", "92.0", "78.5"],
    "Attendance": ["90%", "95%", "80%"]
})
```

### 1. Remove currency symbols and commas

```python
salary_text = (
    df["Salary"]
    .astype("string")
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["Salary"] = pd.to_numeric(salary_text, errors="coerce")

print(df["Salary"])
```

This example assumes the currency symbol is `₹`. Other currencies may require different cleaning rules.

### 2. Convert percentage strings

```python
attendance_text = (
    df["Attendance"]
    .astype("string")
    .str.replace("%", "", regex=False)
    .str.strip()
)

df["Attendance"] = pd.to_numeric(
    attendance_text,
    errors="coerce"
)

print(df["Attendance"])
```

This stores values on a 0–100 scale, such as `90` for 90%.

If your application requires a 0–1 fraction instead, divide by 100:

```python
df["Attendance_Fraction"] = df["Attendance"] / 100
```

### 3. Validate numeric ranges

```python
invalid_age = ~df["Age"].between(0, 120)
```

Use range checks appropriate to the column. For a percentage stored on a 0–100 scale:

```python
invalid_attendance = ~df["Attendance"].between(0, 100)

print(df.loc[invalid_attendance])
```

A value may be successfully converted to a number but still be logically invalid. Conversion and validation are separate steps.

## 📅 Converting Dates and Times

Dates stored as strings can be difficult to sort, filter, and analyze correctly.

### 1. Convert a date column

```python
df = pd.DataFrame({
    "Student": ["Amit", "Priya", "Rahul"],
    "Admission_Date": [
        "2026-01-10",
        "2026-01-15",
        "2026-02-01"
    ]
})

df["Admission_Date"] = pd.to_datetime(
    df["Admission_Date"],
    errors="coerce"
)

print(df.dtypes)
```

### 2. Parse a known date format

```python
df["Admission_Date"] = pd.to_datetime(
    df["Admission_Date"],
    format="%Y-%m-%d",
    errors="coerce"
)
```

Common format directives:

| Directive | Meaning | Example |
|---|---|---|
| `%Y` | Four-digit year | `2026` |
| `%m` | Month | `10` |
| `%d` | Day | `19` |
| `%H` | Hour, 24-hour clock | `14` |
| `%M` | Minute | `30` |
| `%S` | Second | `45` |

### 3. Extract date components

```python
df["Admission_Year"] = df["Admission_Date"].dt.year
df["Admission_Month"] = df["Admission_Date"].dt.month
df["Admission_Day"] = df["Admission_Date"].dt.day
```

### 4. Handle invalid dates

```python
dates = pd.Series([
    "2026-01-10",
    "not-a-date",
    "2026-02-30"
])

parsed_dates = pd.to_datetime(
    dates,
    errors="coerce"
)

print(parsed_dates)
```

Unparseable or invalid dates become `NaT`, Pandas' missing datetime value. Review these rows before deciding how to handle them.

**Timezone note:** For timestamps containing timezone information, use an appropriate timezone-aware conversion strategy. Do not discard timezone information unless that is an intentional business rule.

## ✅ Working with Boolean Values

Boolean columns should contain true/false values rather than arbitrary text when they represent binary conditions.

Example:

```python
df = pd.DataFrame({
    "Passed": ["Yes", "No", "Yes", "No"]
})
```

Map known values explicitly:

```python
mapping = {
    "Yes": True,
    "No": False
}

df["Passed"] = df["Passed"].map(mapping).astype("boolean")

print(df)
print(df.dtypes)
```

If an unexpected value appears, `.map()` produces a missing value for that entry. Check for such values before treating the conversion as complete.

```python
df = pd.DataFrame({
    "Passed": ["Yes", "No", "Unknown"]
})

mapping = {"Yes": True, "No": False}

mapped = df["Passed"].map(mapping)

unrecognized = df.loc[mapped.isna(), "Passed"]

print("Unrecognized values:")
print(unrecognized)
```

Do not convert arbitrary strings directly to Boolean values using `astype(bool)`: non-empty strings such as `"No"` evaluate to `True` in Python.

## 🏷️ Handling Categorical Data

Columns such as department, city, course, and membership level may contain a limited set of repeated values.

```python
df = pd.DataFrame({
    "Student": ["Amit", "Priya", "Rahul", "Neha"],
    "Department": ["Computer Science", "IT", "Computer Science", "IT"]
})

df["Department"] = df["Department"].astype("category")

print(df.dtypes)
print(df["Department"].cat.categories)
```

Categorical types can reduce memory usage when a column has relatively few distinct values.

However, categories do not automatically represent meaningful numeric order. For example, `Beginner`, `Intermediate`, and `Advanced` have an order only if the application defines one.

For machine learning, categorical columns often require encoding. Use appropriate encoders inside a fitted preprocessing pipeline rather than assuming that changing a Pandas dtype makes the values model-ready.

## 🧩 Nullable Data Types

Traditional integer columns generally cannot represent missing values directly. Pandas may therefore convert an integer column to floating-point when a missing value is introduced.

```python
df = pd.DataFrame({
    "Age": [20, 21, None, 23]
})

print(df["Age"].dtype)
```

Use nullable integer types when whole-number values and missing values must coexist:

```python
df["Age"] = df["Age"].astype("Int64")

print(df)
print(df["Age"].dtype)
```

Other useful nullable types include:

```python
df["Age"] = df["Age"].astype("Int64")
df["Name"] = df["Name"].astype("string")
df["Passed"] = df["Passed"].astype("boolean")
```

Remember that Pandas nullable types may use `pd.NA` rather than NumPy's `NaN`. Use Pandas missing-value methods such as `.isna()` and `.notna()` instead of relying on direct equality comparisons.

## ⚠️ Handling Conversion Errors

There are three important error-handling modes in `pd.to_numeric()` and `pd.to_datetime()`.

| Mode | Behavior | Appropriate use |
|---|---|---|
| `errors="raise"` | Raises an exception for invalid input | Strict validation |
| `errors="coerce"` | Converts invalid values to missing values | Controlled cleaning workflows |
| `errors="ignore"` | Historical option that may preserve original values | Avoid as a general cleaning strategy |

For modern Pandas workflows, prefer `raise` or `coerce` with explicit validation. `errors="ignore"` is deprecated in recent Pandas versions for these conversion functions.

### Identify invalid numeric values before conversion

```python
values = pd.Series(["100", "200", "invalid", "300"])

numeric_values = pd.to_numeric(values, errors="coerce")

invalid_mask = values.notna() & numeric_values.isna()

print("Invalid values:")
print(values[invalid_mask])
```

This approach helps identify the original values that failed conversion.

### Preserve the original data

For important cleaning operations, retain a copy of the raw column.

```python
df["Age_Raw"] = df["Age"]

df["Age"] = pd.to_numeric(
    df["Age_Raw"],
    errors="coerce"
)
```

This makes it easier to audit changes and investigate conversion failures.

## 🔬 Validating Converted Data

After conversion, verify both the data type and the values.

### 1. Check data types

```python
print(df.dtypes)
```

### 2. Count missing values

```python
print(df.isna().sum())
```

### 3. Check numeric summaries

```python
print(df.select_dtypes(include="number").describe())
```

### 4. Check allowed ranges

```python
invalid_marks = ~df["Marks"].between(0, 100)

print(df.loc[invalid_marks])
```

### 5. Check whether a column is numeric

```python
is_numeric = pd.api.types.is_numeric_dtype(df["Marks"])

print(is_numeric)
```

### 6. Assert expected types

```python
assert pd.api.types.is_numeric_dtype(df["Marks"])
assert pd.api.types.is_datetime64_any_dtype(df["Admission_Date"])
```

Assertions are useful for checks during development. For production data pipelines, consider explicit validation functions that report meaningful errors and prevent invalid datasets from silently proceeding.

## 🛠️ Complete Data Cleaning Example

The following example demonstrates a complete workflow using a student dataset.

```python
import pandas as pd


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate common student dataset data types."""

    cleaned = df.copy()

    # Standardize text columns.
    cleaned["Student"] = (
        cleaned["Student"]
        .astype("string")
        .str.strip()
    )

    # Convert age to a nullable integer.
    age_numeric = pd.to_numeric(
        cleaned["Age"],
        errors="coerce"
    )

    invalid_age = age_numeric.notna() & (
        (age_numeric < 0) | (age_numeric > 120)
    )

    if invalid_age.any():
        raise ValueError("Age contains values outside the allowed range.")

    cleaned["Age"] = age_numeric.astype("Int64")

    # Convert marks to numeric values.
    cleaned["Marks"] = pd.to_numeric(
        cleaned["Marks"],
        errors="coerce"
    )

    invalid_marks = cleaned["Marks"].notna() & (
        ~cleaned["Marks"].between(0, 100)
    )

    if invalid_marks.any():
        raise ValueError("Marks must be between 0 and 100.")

    # Convert admission dates.
    cleaned["Admission_Date"] = pd.to_datetime(
        cleaned["Admission_Date"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    # Convert attendance from percentage text to a 0–100 number.
    attendance_text = (
        cleaned["Attendance"]
        .astype("string")
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    cleaned["Attendance"] = pd.to_numeric(
        attendance_text,
        errors="coerce"
    )

    invalid_attendance = cleaned["Attendance"].notna() & (
        ~cleaned["Attendance"].between(0, 100)
    )

    if invalid_attendance.any():
        raise ValueError("Attendance must be between 0 and 100.")

    # Map known status values to nullable Booleans.
    passed_text = (
        cleaned["Passed"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    passed_mapping = {
        "yes": True,
        "no": False
    }

    cleaned["Passed"] = passed_text.map(
        passed_mapping
    ).astype("boolean")

    return cleaned


def main():
    raw_data = pd.DataFrame({
        "Student": ["Amit", "Priya", "Rahul", "Neha"],
        "Age": ["20", "21", "19", "unknown"],
        "Marks": ["85.5", "92", "78.5", "88"],
        "Admission_Date": [
            "2026-01-10",
            "2026-01-15",
            "2026-02-01",
            "2026-02-10"
        ],
        "Attendance": ["90%", "95%", "80%", "92%"],
        "Passed": ["Yes", "Yes", "No", "Yes"]
    })

    print("BEFORE CLEANING")
    print(raw_data.dtypes)

    cleaned_data = clean_student_data(raw_data)

    print("\nAFTER CLEANING")
    print(cleaned_data.dtypes)

    print("\nCLEANED DATA")
    print(cleaned_data)

    print("\nMISSING VALUES")
    print(cleaned_data.isna().sum())

    cleaned_data.to_csv(
        "cleaned_student_data.csv",
        index=False
    )

    print("\nSaved cleaned_student_data.csv")


if __name__ == "__main__":
    main()
```

### What this example demonstrates

- Text normalization using `.str.strip()`.
- Numeric conversion with `pd.to_numeric()`.
- Nullable integer conversion with `Int64`.
- Date parsing with `pd.to_datetime()`.
- Percentage cleanup.
- Boolean mapping with explicit allowed values.
- Range validation for marks, age, and attendance.
- Missing-value inspection.
- Saving the cleaned dataset to CSV.

**Important:** The example intentionally preserves missing values created from invalid entries. For example, `"unknown"` becomes a missing age. A real project should decide whether to reject the record, correct the source value, or impute the missing age. It should not silently assume the value.

## 🤖 Data Types and Machine Learning

Correct data types are important, but **correct dtype does not automatically mean a feature is ready for machine learning**.

| Feature | Typical preparation |
|---|---|
| Age | Numeric conversion and missing-value handling |
| Salary | Numeric conversion and possibly scaling |
| Admission date | Datetime parsing, then meaningful date features |
| Department | Categorical encoding |
| Passed | Boolean or target label, depending on the task |
| Student ID | Usually excluded from features or handled as an identifier |
| Marks | Numeric validation and target/feature definition |

### Avoid data leakage

When building machine learning models:

1. Split the dataset into training and testing sets.
2. Fit imputers, encoders, scalers, and learned transformations on training data only.
3. Apply the fitted transformations to validation and test data.
4. Use a pipeline to keep preprocessing consistent.

For example:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

numeric_features = ["Age", "Attendance"]
categorical_features = ["Department"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

This illustrates how type-appropriate preprocessing can be combined into a pipeline. Before fitting, ensure the input columns have been converted to suitable types and that the target variable is defined correctly.

## ❌ Common Mistakes

### Mistake 1: Converting identifiers into numbers

A student ID such as `"00125"` may lose its leading zeros if converted to an integer.

**Solution:** Keep identifiers as strings unless numeric operations are genuinely required.

### Mistake 2: Assuming all object columns are incorrect

Names and addresses are expected to contain text.

**Solution:** Determine the semantic meaning of each column before converting it.

### Mistake 3: Ignoring failed conversions

```python
pd.to_numeric(df["Marks"], errors="coerce")
```

This may produce missing values.

**Solution:** Count and investigate newly introduced missing values before proceeding.

### Mistake 4: Converting `"Yes"` and `"No"` directly with `astype(bool)`

Non-empty strings evaluate to `True`, including `"No"`.

**Solution:** Use an explicit mapping.

### Mistake 5: Treating a numeric value as valid just because conversion succeeded

For example, `-50` can be converted to a number but may be invalid as an age.

**Solution:** Add business-rule checks and range validation.

### Mistake 6: Losing raw data

Overwriting columns can make errors difficult to investigate.

**Solution:** Keep the original dataset unchanged and work on a copy.

### Mistake 7: Parsing ambiguous dates without checking their format

`01/02/2026` can mean January 2 or February 1, depending on the date convention.

**Solution:** Specify the expected format and validate the result.

### Mistake 8: Assuming categorical conversion encodes features for ML

Pandas `category` is a storage representation, not a universal model encoding.

**Solution:** Select an appropriate encoder based on the model and feature meaning.

## 🌟 Best Practices

- Inspect the dataset before modifying it.
- Define the intended dtype for every important column.
- Preserve raw data when traceability matters.
- Use `pd.to_numeric()` for numeric conversion.
- Use `pd.to_datetime()` for dates and timestamps.
- Use explicit mappings for Boolean values.
- Prefer nullable types when missing values are allowed.
- Validate ranges, formats, and allowed categories after conversion.
- Track how many values fail conversion.
- Use clear, reusable cleaning functions.
- Separate data cleaning from model-specific feature engineering.
- Use Scikit-Learn pipelines to prevent preprocessing leakage.
- Test cleaning functions with missing, malformed, and unexpected values.
- Record cleaning rules so the same logic can be applied consistently in production.

## 💡 Mini Projects

### Project 1: Student Data Type Cleaner

Create a student dataset containing:

- Age stored as strings.
- Marks stored as strings.
- Attendance with percentage signs.
- Admission dates stored as text.
- Pass status stored as `"Yes"` and `"No"`.

**Tasks:**
1. Inspect the original dtypes.
2. Convert every column to its intended type.
3. Identify conversion failures.
4. Validate numeric ranges.
5. Export the cleaned CSV.

### Project 2: Sales Dataset Cleaner

Create a sales dataset with columns for product, quantity, unit price, order date, and payment status.

**Tasks:**
1. Remove currency symbols and thousands separators.
2. Convert quantity to an integer where appropriate.
3. Convert unit price to a numeric type.
4. Parse order dates.
5. Map valid payment statuses.
6. Validate that quantity and price satisfy business rules.

### Project 3: Real-World Dataset Validation

Choose a public dataset and document:

- Original data types.
- Expected data types.
- Conversion rules.
- Invalid values found.
- Missing values introduced during conversion.
- Final validation results.

Produce a report comparing the dataset before and after cleaning.

## 📝 Practice Exercises

1. Create a DataFrame with age values stored as strings and convert them to integers.
2. Convert marks such as `"85.5"` and `"92.0"` to floating-point numbers.
3. Convert `"₹50,000"` to a numeric salary.
4. Remove `%` from attendance values and convert them to numbers.
5. Convert a date column into `datetime64`.
6. Convert `"Yes"` and `"No"` into nullable Boolean values.
7. Identify values that fail numeric conversion.
8. Convert an integer column with missing values to `Int64`.
9. Identify ages outside a valid range.
10. Write a reusable function that checks a DataFrame against an expected dtype specification.

## ⚡ Quick Revision

| Task | Method |
|---|---|
| Inspect data types | `df.dtypes` |
| Inspect dataset summary | `df.info()` |
| Convert a column | `df["col"].astype("string")` |
| Convert numeric text | `pd.to_numeric()` |
| Convert dates | `pd.to_datetime()` |
| Infer nullable dtypes | `df.convert_dtypes()` |
| Find invalid conversions | `errors="coerce"` and a failure mask |
| Check missing values | `df.isna().sum()` |
| Select numeric columns | `df.select_dtypes(include="number")` |
| Check numeric dtype | `pd.api.types.is_numeric_dtype()` |
| Check datetime dtype | `pd.api.types.is_datetime64_any_dtype()` |
| Validate ranges | `.between()` |
| Save a cleaned dataset | `df.to_csv("cleaned.csv", index=False)` |

## 📂 Project Structure

Recommended repository structure:

```text
05-Data-Cleaning/
│
├── README.md
│
├── 01-Missing-Values/
│   └── README.md
│
├── 02-Duplicates/
│   └── README.md
│
└── 03-Incorrect-Data-Types/
    ├── README.md
    ├── incorrect_data_types.py
    ├── student_data.csv
    └── cleaned_student_data.csv
```

Suggested files:

- `README.md` — concepts, examples, and instructions.
- `incorrect_data_types.py` — reusable conversion and validation logic.
- `student_data.csv` — sample raw dataset.
- `cleaned_student_data.csv` — cleaned output generated by the script.

Do not commit sensitive personal information or confidential datasets to a public repository.

## 🗺️ Learning Roadmap

Follow this progression to build practical data-cleaning skills:

1. Understand Python and Pandas data types.
2. Inspect dataset structure with `dtypes` and `info()`.
3. Identify columns that do not match their intended meaning.
4. Convert strings into numeric and datetime types.
5. Handle invalid values and missing data.
6. Validate data against business rules.
7. Build reusable data-cleaning functions.
8. Integrate preprocessing into machine learning pipelines.
9. Test, document, and maintain the cleaning workflow.

## 🎯 Key Takeaways

- Data types define how values are represented and processed.
- Incorrect data types can break calculations and data analysis.
- `astype()`, `pd.to_numeric()`, and `pd.to_datetime()` solve different conversion needs.
- Successful conversion does not guarantee valid data.
- Invalid values and newly introduced missing values must be investigated.
- Identifiers, categorical variables, and numeric measurements should be treated according to their meaning.
- Reliable preprocessing includes conversion, validation, testing, and documentation.

---

## 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

GitHub: [Kishor055](https://github.com/Kishor055)

Repository: [Machine-Learning](https://github.com/Kishor055/Machine-Learning)

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Add examples or improve the documentation.
4. Test your code.
5. Submit a pull request with a clear description.

## ⭐ Support

If this resource helps you learn Python data cleaning and machine learning, consider giving the repository a star on GitHub.

**Keep learning, keep building, and keep growing!** 🌱

---

*Part of the Machine Learning learning path — Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Model Training → Evaluation → Deployment.*

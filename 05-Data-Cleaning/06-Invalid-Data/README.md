# 🚨 Invalid Data Detection and Handling in Python

🌱 **GROW → INSPECT → DETECT → VALIDATE → CORRECT → VERIFY**

Welcome to **Invalid Data Detection and Handling**, part of the Data Cleaning module in the Machine Learning learning path. This guide explains how to identify values that violate data types, expected formats, allowed ranges, business rules, or relationships between columns using Python and Pandas.

Invalid data can lead to inaccurate analysis, unreliable predictions, failed data pipelines, and unexpected application behavior. Learning to detect and handle these problems is essential for building trustworthy datasets and machine learning systems.

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [What Is Invalid Data?](#-what-is-invalid-data)
3. [Invalid Data vs Missing Values vs Outliers](#-invalid-data-vs-missing-values-vs-outliers)
4. [Types of Invalid Data](#-types-of-invalid-data)
5. [Create a Sample Dataset](#-create-a-sample-dataset)
6. [Inspect Data Quality](#-inspect-data-quality)
7. [Validate Data Types](#-validate-data-types)
8. [Validate Numeric Ranges](#-validate-numeric-ranges)
9. [Validate Categorical Values](#-validate-categorical-values)
10. [Validate Dates and Times](#-validate-dates-and-times)
11. [Validate Text and Identifiers](#-validate-text-and-identifiers)
12. [Validate Relationships Between Columns](#-validate-relationships-between-columns)
13. [Handle Invalid Values](#-handle-invalid-values)
14. [Build Reusable Validation Functions](#-build-reusable-validation-functions)
15. [Complete Data Validation Program](#-complete-data-validation-program)
16. [Invalid Data in Machine Learning](#-invalid-data-in-machine-learning)
17. [Common Mistakes](#-common-mistakes)
18. [Best Practices](#-best-practices)
19. [Mini Projects](#-mini-projects)
20. [Practice Exercises](#-practice-exercises)
21. [Quick Revision](#-quick-revision)
22. [Project Structure](#-project-structure)
23. [Learning Roadmap](#-learning-roadmap)

---

## 🎯 Introduction

Data collected from spreadsheets, APIs, databases, forms, and third-party systems may contain values that do not follow the expected data rules.

Consider a student dataset:

| Student | Age | Marks | Attendance |
|---|---:|---:|---:|
| Amit | 20 | 85 | 90 |
| Priya | 21 | 92 | 95 |
| Rahul | -5 | 78 | 80 |
| Neha | 19 | 150 | 88 |
| Kiran | 22 | 84 | 120 |

Potential issues include:

- Rahul's negative age may violate the application's age rules.
- Neha's marks exceed the assumed maximum of 100.
- Kiran's attendance exceeds the assumed 100% limit.

These values may be valid numbers at the Python level, but they violate the example dataset's business rules.

The purpose of validation is to identify such problems, explain why they are invalid, and determine how they should be handled.

### Learning objectives

By completing this guide, you will learn to:

- Detect invalid values in structured datasets.
- Distinguish invalid data from missing values and statistical outliers.
- Validate data types, ranges, categories, dates, and identifiers.
- Apply conditional and cross-column validation rules.
- Report invalid records without silently deleting information.
- Build reusable data validation functions.
- Prepare reliable data for machine learning.
- Prevent invalid records from silently entering production pipelines.

## 🧠 What Is Invalid Data?

**Invalid data** is information that violates a defined requirement, schema, format, logical constraint, or business rule.

A value may be syntactically valid but semantically invalid.

For example:

```python
age = -10
marks = 150
```

Both values are valid Python integers. However, they may be invalid under the rules of a student dataset.

Another example:

```python
email = "student.example.com"
```

This is a valid Python string, but it does not match a typical email-address format.

Validation requires knowing what values are acceptable for the specific application.

### Examples of invalid data

| Field | Example | Possible issue |
|---|---|---|
| Age | `-5` | Outside the allowed range |
| Marks | `150` | Exceeds the maximum score |
| Attendance | `120` | Exceeds the percentage range |
| Date | `"2026-02-30"` | Impossible calendar date |
| Status | `"Completedd"` | Not in the approved vocabulary |
| Student ID | `" "` | Empty identifier |
| Quantity | `-10` | Invalid for a non-returned purchase |
| Email | `"student.example.com"` | Missing expected email structure |
| End date | Earlier than start date | Violates a date relationship |
| Total price | Different from quantity × unit price | Possible data-integrity problem |

Some rules depend on context. A negative quantity may be valid in a system that represents returns as negative numbers. Always define validation rules from the intended data model rather than applying assumptions blindly.

## 🚀 Why Data Validation Matters

Invalid data can cause:

- Incorrect business reports and summaries.
- Failed numeric calculations.
- Inaccurate statistical analysis.
- Inconsistent model training.
- Incorrect predictions.
- Failed API requests and database operations.
- Broken data pipelines.
- Unexpected behavior in deployed applications.
- Increased debugging and maintenance costs.

### Example

```python
import pandas as pd

df = pd.DataFrame({
    "Student": ["Amit", "Priya", "Rahul", "Neha"],
    "Marks": [85, 92, 150, 78]
})

print("Average marks:", df["Marks"].mean())
```

The calculation runs successfully, but the average is misleading if marks must be between 0 and 100.

Validate the domain range:

```python
invalid_mask = ~df["Marks"].between(0, 100)

print("Invalid records:")
print(df.loc[invalid_mask])
```

Validation helps distinguish calculations that are technically possible from results that are meaningful.

## 🔎 Invalid Data vs Missing Values vs Outliers

These concepts are related, but they are not identical.

| Concept | Definition | Example |
|---|---|---|
| Invalid data | Violates a defined rule | Marks of `150` when the maximum is 100 |
| Missing value | Information is unavailable or absent | Age is `NaN` |
| Statistical outlier | Unusual compared with the data distribution | A valid annual salary far above the median |
| Inconsistent value | Represents information in an unexpected format | `"mumbai"` versus `"Mumbai"` |
| Duplicate record | Repeats another observation according to the relevant duplicate rule | Same student record entered twice |

A value can belong to more than one category. For instance, a salary of zero might be statistically unusual and also invalid under a specific employment-data rule.

**Key principle:** A statistical anomaly is not automatically invalid. Invalidity should be established through a documented rule, trusted reference, or verified source.

## 🧩 Types of Invalid Data

### 1. Type-related invalid data

A column expected to contain numbers may contain text.

```text
Age = ["20", "21", "unknown", "25"]
```

Some numeric strings can be converted; the value `"unknown"` requires review.

### 2. Range-related invalid data

Values fall outside the acceptable range.

```text
Marks = [-10, 85, 92, 150]
```

For an examination scored from 0 to 100, `-10` and `150` violate the assumed range.

### 3. Format-related invalid data

Values fail the expected format.

```text
Student_ID = ["S001", "S002", "invalid-id"]
```

The appropriate rule depends on the identifier specification.

### 4. Categorical invalid data

A field contains a value not allowed by its vocabulary.

```text
Payment_Status = ["Paid", "Pending", "Completedd"]
```

The third label may be a typo or an unrecognized status.

### 5. Date-related invalid data

Dates may contain impossible dates or violate ordering rules.

```text
Start_Date = "2026-05-10"
End_Date   = "2026-04-10"
```

If the end date must not precede the start date, the record violates a business rule.

### 6. Cross-column invalid data

Two or more fields contradict one another.

Examples:

- An order is marked `"Delivered"` without a required delivery date.
- An employee's termination date precedes their joining date.
- An invoice total differs from its calculated line-item total beyond the permitted tolerance.

### 7. Referential-integrity problems

A record references an entity that does not exist in the trusted reference dataset.

Example: an order contains a customer ID that is not present in the customer table.

### 8. Duplicate or conflicting identifiers

Two records may use the same identifier even though the identifier is expected to be unique.

Whether this is invalid depends on the schema. A customer may legitimately have multiple transactions, while a transaction ID may need to be unique.

---

## 🧪 Create a Sample Dataset

The following dataset intentionally includes several validation problems.

```python
import pandas as pd

df = pd.DataFrame({
    "Student_ID": [
        "S001", "S002", "S003", "S004",
        "S005", "S006", "S007", "S008"
    ],
    "Name": [
        "Amit", "Priya", "Rahul", "Neha",
        "Kiran", "", "Meera", "Arjun"
    ],
    "Age": [20, 21, -5, 19, 22, 24, 150, 23],
    "Marks": [85, 92, 78, 150, -10, 88, 74, 91],
    "Attendance": [90, 95, 80, 88, 75, 120, 85, 92],
    "Status": [
        "Passed", "Passed", "Failed", "Passed",
        "Completedd", "Passed", "Failed", "Passed"
    ],
    "Admission_Date": [
        "2026-01-10", "2026-01-15", "2026-02-01",
        "2026-02-30", "2026-03-05", "2026-03-12",
        "2026-03-20", "2026-04-01"
    ]
})

print(df)
print(df.info())
```

Potential problems include:

- Empty student name.
- Negative and implausibly large ages.
- Marks outside the permitted range.
- Attendance greater than 100%.
- An unapproved status value.
- An invalid calendar date.

These are examples of validation failures, not proof that the original records should be deleted.

## 🔍 Inspect Data Quality

Start by understanding the data before applying validation rules.

### 1. Display the dataset

```python
print(df.head())
print(df.tail())
```

### 2. Check dimensions

```python
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
```

### 3. Inspect data types

```python
print(df.dtypes)
df.info()
```

### 4. Count missing values

```python
print(df.isna().sum())
```

Empty strings are not automatically treated as missing values by Pandas. Check them separately:

```python
blank_counts = (
    df.astype("string")
    .apply(lambda column: column.str.strip().eq("").sum())
)

print(blank_counts)
```

### 5. Inspect unique categories

```python
print(df["Status"].value_counts(dropna=False))
```

### 6. Check for duplicate identifiers

```python
duplicate_ids = df["Student_ID"].duplicated(keep=False)

print(df.loc[duplicate_ids])
```

Whether duplicate IDs are invalid depends on the dataset's uniqueness rule. Do not remove records until the identifier semantics have been established.

## 🔢 Validate Data Types

Data types should match their intended meaning.

### Convert numeric values safely

```python
df["Age_Numeric"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)
```

The `errors="coerce"` option converts values that cannot be interpreted as numeric into missing values.

Identify the original values that failed conversion:

```python
original = df["Age"].astype("string")

converted = pd.to_numeric(
    original,
    errors="coerce"
)

failed_conversion = (
    original.notna()
    & original.str.strip().ne("")
    & converted.isna()
)

print("Values that failed conversion:")
print(original[failed_conversion])
```

A failed conversion should be reviewed separately from an actual missing value.

### Check the resulting dtype

```python
print(df["Age_Numeric"].dtype)

is_numeric = pd.api.types.is_numeric_dtype(
    df["Age_Numeric"]
)

print("Is numeric:", is_numeric)
```

### Convert dates safely

```python
df["Admission_Date_Parsed"] = pd.to_datetime(
    df["Admission_Date"],
    format="%Y-%m-%d",
    errors="coerce"
)
```

Identify invalid or unparseable dates:

```python
original_dates = df["Admission_Date"].astype("string")

parsed_dates = pd.to_datetime(
    original_dates,
    format="%Y-%m-%d",
    errors="coerce"
)

invalid_dates = (
    original_dates.notna()
    & original_dates.str.strip().ne("")
    & parsed_dates.isna()
)

print(original_dates[invalid_dates])
```

Pandas converts invalid dates such as `"2026-02-30"` to `NaT` when `errors="coerce"` is used.

**Important:** Do not silently lose the raw values. Preserve them when investigating failed conversions.

## 📏 Validate Numeric Ranges

Range validation identifies values outside a field's allowed limits.

### 1. Validate age

```python
invalid_age = ~df["Age"].between(0, 120)

print(df.loc[invalid_age, ["Student_ID", "Age"]])
```

The range of 0–120 is an illustrative data-quality rule, not a universal rule for every age-related system.

### 2. Validate marks

```python
invalid_marks = ~df["Marks"].between(0, 100)

print(df.loc[invalid_marks, ["Student_ID", "Marks"]])
```

This assumes the assessment uses a 0–100 scoring scale.

### 3. Validate attendance

```python
invalid_attendance = ~df["Attendance"].between(0, 100)

print(df.loc[invalid_attendance, ["Student_ID", "Attendance"]])
```

This assumes attendance is stored as a percentage from 0 to 100 rather than a fraction from 0 to 1.

### 4. Validate multiple columns

```python
numeric_ranges = {
    "Age": (0, 120),
    "Marks": (0, 100),
    "Attendance": (0, 100)
}

for column, (minimum, maximum) in numeric_ranges.items():
    invalid_mask = ~df[column].between(minimum, maximum)

    print(f"\nInvalid values in {column}:")
    print(df.loc[invalid_mask, ["Student_ID", column]])
```

For datasets containing missing values, decide whether missingness is itself a validation failure. By default, missing values generally should be handled as a separate rule rather than interpreted as an ordinary out-of-range number.

### 5. Create reusable range validation

```python
def validate_range(
    series: pd.Series,
    minimum: float,
    maximum: float
) -> pd.Series:
    """Return True for nonmissing values outside the allowed range."""

    numeric = pd.to_numeric(series, errors="coerce")

    return (
        numeric.notna()
        & ~numeric.between(minimum, maximum)
    )


print(validate_range(df["Marks"], 0, 100))
```

This function returns a Boolean mask identifying numeric values outside the allowed limits. Failed conversions need a separate validation check.

## 🏷️ Validate Categorical Values

Categorical fields often have a known set of allowed values.

For example:

```python
allowed_statuses = {
    "Passed",
    "Failed",
    "Pending"
}
```

Find unapproved values:

```python
invalid_status = (
    df["Status"].notna()
    & ~df["Status"].isin(allowed_statuses)
)

print(df.loc[invalid_status, ["Student_ID", "Status"]])
```

This identifies unexpected labels such as `"Completedd"`.

### Standardize formatting before validating

If the approved vocabulary is case-sensitive only by convention, normalize formatting first:

```python
status_normalized = (
    df["Status"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

allowed_statuses_normalized = {
    "passed",
    "failed",
    "pending"
}

invalid_status = (
    status_normalized.notna()
    & ~status_normalized.isin(allowed_statuses_normalized)
)

print(df.loc[invalid_status, ["Student_ID", "Status"]])
```

Do not add a value to the allowed set solely because it appears in the incoming dataset. Update the schema only after determining that the category is legitimate.

### Map known synonyms carefully

```python
status_mapping = {
    "passed": "Passed",
    "pass": "Passed",
    "failed": "Failed",
    "fail": "Failed",
    "pending": "Pending"
}

normalized_status = (
    df["Status"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

standardized_status = normalized_status.map(status_mapping)

unmapped = (
    normalized_status.notna()
    & normalized_status.str.strip().ne("")
    & standardized_status.isna()
)

print("Unmapped values:")
print(df.loc[unmapped, ["Student_ID", "Status"]])
```

Using `map()` allows unrecognized values to be detected. Unmapped values should be corrected, accepted through a reviewed schema change, or quarantined according to the application's policy.

## 📅 Validate Dates and Times

Date validation includes parsing the expected format and checking relationships between dates.

### 1. Parse dates with a known format

```python
dates = pd.Series([
    "2026-01-10",
    "2026-02-30",
    "not-a-date",
    "2026-03-15"
])

parsed_dates = pd.to_datetime(
    dates,
    format="%Y-%m-%d",
    errors="coerce"
)

print(parsed_dates)
```

Invalid values become `NaT`.

### 2. Validate date ranges

Suppose admission dates must fall within a particular reporting period:

```python
start_date = pd.Timestamp("2026-01-01")
end_date = pd.Timestamp("2026-12-31")

date_outside_period = (
    parsed_dates.notna()
    & ~parsed_dates.between(start_date, end_date)
)

print(parsed_dates[date_outside_period])
```

Use date limits that match the business process. A date outside a reporting period may be unexpected without being intrinsically invalid.

### 3. Validate relationships between dates

```python
admissions = pd.DataFrame({
    "Student_ID": ["S001", "S002", "S003"],
    "Start_Date": ["2026-01-10", "2026-03-15", "2026-04-01"],
    "End_Date": ["2026-02-10", "2026-03-01", None]
})

admissions["Start_Date"] = pd.to_datetime(
    admissions["Start_Date"],
    errors="coerce"
)

admissions["End_Date"] = pd.to_datetime(
    admissions["End_Date"],
    errors="coerce"
)

invalid_order = (
    admissions["Start_Date"].notna()
    & admissions["End_Date"].notna()
    & (admissions["End_Date"] < admissions["Start_Date"])
)

print(admissions.loc[invalid_order])
```

The second record violates the condition that an end date must not precede its start date.

A missing end date might be valid for an ongoing activity. Missingness rules should therefore be defined separately from date-order rules.

## 📝 Validate Text and Identifiers

Text fields and identifiers often require their own validation rules.

### 1. Find blank names

```python
blank_name = (
    df["Name"].isna()
    | df["Name"].astype("string").str.strip().eq("")
)

print(df.loc[blank_name, ["Student_ID", "Name"]])
```

### 2. Validate student ID format

Suppose a student ID must contain the letter `S` followed by exactly three digits.

```python
id_text = df["Student_ID"].astype("string")

invalid_id_format = ~id_text.str.fullmatch(
    r"S\d{3}",
    na=False
)

print(df.loc[invalid_id_format, ["Student_ID"]])
```

Examples of valid values under this specific rule are `"S001"` and `"S123"`.

If lowercase IDs or leading spaces are permitted by the source specification, normalize those values before applying the format check.

### 3. Validate unique identifiers

```python
duplicate_id_mask = df["Student_ID"].duplicated(keep=False)

print(df.loc[duplicate_id_mask])
```

A duplicate key is invalid only if the schema requires that the key be unique. For transaction or event datasets, repeated customer IDs may be entirely valid.

### 4. Validate basic email structure

```python
email = pd.Series([
    "student@example.com",
    "invalid-email",
    "person@domain",
    None
], dtype="string")

basic_email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

invalid_email = (
    email.notna()
    & ~email.str.fullmatch(basic_email_pattern, na=False)
)

print(email[invalid_email])
```

This is a basic structural check, not a complete email-standard implementation. Passing the pattern does not prove that the address exists or can receive messages.

## 🔗 Validate Relationships Between Columns

A value can be acceptable on its own but invalid when compared with another field.

### 1. Validate total price

Suppose an order contains quantity, unit price, and total price.

```python
orders = pd.DataFrame({
    "Quantity": [2, 3, 4],
    "Unit_Price": [100.0, 50.0, 25.0],
    "Total_Price": [200.0, 200.0, 100.0]
})

expected_total = orders["Quantity"] * orders["Unit_Price"]

invalid_total = ~(
    (orders["Total_Price"] - expected_total).abs() <= 0.01
)

print(orders.loc[invalid_total])
```

The tolerance of `0.01` is illustrative. Choose a tolerance that matches the currency precision and rounding policy.

In financial systems, monetary values are often best handled using a documented decimal and rounding strategy rather than unconstrained floating-point arithmetic.

### 2. Validate logically related fields

```python
records = pd.DataFrame({
    "Status": ["Delivered", "Pending", "Delivered"],
    "Delivery_Date": ["2026-01-10", None, None]
})

records["Delivery_Date"] = pd.to_datetime(
    records["Delivery_Date"],
    errors="coerce"
)

invalid_delivery = (
    records["Status"].eq("Delivered")
    & records["Delivery_Date"].isna()
)

print(records.loc[invalid_delivery])
```

This assumes every delivered order must have a recorded delivery date. Confirm that requirement before applying it to real data.

### 3. Validate referential integrity

Suppose each order must reference an existing customer.

```python
customers = pd.DataFrame({
    "Customer_ID": ["C001", "C002", "C003"]
})

orders = pd.DataFrame({
    "Order_ID": ["O001", "O002", "O003"],
    "Customer_ID": ["C001", "C999", "C002"]
})

valid_customer_ids = set(customers["Customer_ID"])

invalid_reference = ~orders["Customer_ID"].isin(valid_customer_ids)

print(orders.loc[invalid_reference])
```

`C999` is not found in the supplied customer reference table. Whether the order should be rejected, held for review, or matched to a corrected customer ID depends on the workflow.

## 🧹 Handle Invalid Values

Detection and handling are different steps. First identify invalid records; then choose a treatment.

### 1. Correct a confirmed error

If a trusted source confirms the correct value, correct it and record the change.

```python
# Illustrative example only:
# Use this assignment only after verifying the correct mark.
df.loc[df["Student_ID"] == "S004", "Marks"] = 95
```

Never infer the correct value from the desired output alone.

### 2. Replace invalid values with missing values

If an invalid value cannot be corrected, it may be appropriate to mark it as missing.

```python
df["Marks_Cleaned"] = df["Marks"].where(
    df["Marks"].between(0, 100),
    pd.NA
)
```

This preserves the original column and creates a cleaned version. It assumes missingness is preferable to retaining the invalid mark in that derived field.

### 3. Filter invalid records

```python
valid_mask = (
    df["Age"].between(0, 120)
    & df["Marks"].between(0, 100)
    & df["Attendance"].between(0, 100)
)

valid_records = df.loc[valid_mask].copy()
invalid_records = df.loc[~valid_mask].copy()
```

This example separates records into valid and invalid groups under the specified range rules. It does not mean that invalid records should be deleted permanently.

### 4. Map a confirmed spelling error

```python
df["Status_Cleaned"] = df["Status"].replace({
    "Completedd": "Passed"
})
```

This is appropriate only if `"Completedd"` is confirmed to mean `"Passed"` in this dataset. If the intended status is unclear, retain it for review.

### 5. Impute missing or rejected values

Imputation is appropriate only when the application allows it and the method is justified.

For example, numeric imputation can be performed with Pandas:

```python
median_marks = df["Marks_Cleaned"].median()

df["Marks_Imputed"] = df["Marks_Cleaned"].fillna(median_marks)
```

This is a simple example. For machine learning, calculate learned imputation statistics from the training data only, generally using a Scikit-Learn pipeline.

### 6. Quarantine invalid records

For production data pipelines, a useful pattern is to preserve records that fail validation in a separate output for review.

```python
invalid_records.to_csv(
    "invalid_records.csv",
    index=False
)
```

A quarantine process helps prevent silent data loss and supports traceability.

## 🧰 Build Reusable Validation Functions

Reusable validation functions make data quality checks easier to maintain and test.

### 1. Numeric range validator

```python
import pandas as pd


def validate_numeric_range(
    series: pd.Series,
    minimum: float,
    maximum: float
) -> pd.Series:
    """Flag parsed numeric values outside an allowed range."""

    numeric = pd.to_numeric(series, errors="coerce")

    return (
        numeric.notna()
        & ~numeric.between(minimum, maximum)
    )
```

This flags successfully parsed numeric values outside the range. Values that cannot be converted should be checked separately.

### 2. Allowed-value validator

```python
def validate_allowed_values(
    series: pd.Series,
    allowed_values: set[str]
) -> pd.Series:
    """Flag nonmissing values outside the allowed vocabulary."""

    return (
        series.notna()
        & ~series.isin(allowed_values)
    )
```

Use this validator after applying any required formatting normalization.

### 3. Required-value validator

```python
def validate_required_text(series: pd.Series) -> pd.Series:
    """Flag missing or blank text values."""

    text = series.astype("string")

    return (
        text.isna()
        | text.str.strip().eq("").fillna(False)
    )
```

### 4. Build an issue report

```python
def build_validation_report(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Return one row per validation issue."""

    issues = []

    age_numeric = pd.to_numeric(df["Age"], errors="coerce")
    marks_numeric = pd.to_numeric(df["Marks"], errors="coerce")
    attendance_numeric = pd.to_numeric(
        df["Attendance"],
        errors="coerce"
    )

    checks = {
        "Age outside range": (
            age_numeric.notna()
            & ~age_numeric.between(0, 120)
        ),
        "Marks outside range": (
            marks_numeric.notna()
            & ~marks_numeric.between(0, 100)
        ),
        "Attendance outside range": (
            attendance_numeric.notna()
            & ~attendance_numeric.between(0, 100)
        ),
        "Blank student name": validate_required_text(df["Name"])
    }

    for issue_name, mask in checks.items():
        for index in df.index[mask]:
            issues.append({
                "Row_Index": index,
                "Student_ID": df.at[index, "Student_ID"],
                "Issue": issue_name
            })

    return pd.DataFrame(
        issues,
        columns=["Row_Index", "Student_ID", "Issue"]
    )
```

Run the report:

```python
validation_report = build_validation_report(df)

print(validation_report)
```

This pattern separates validation logic from handling decisions. It can be extended with date parsing, allowed categories, identifier rules, and cross-column checks.

## 🛠️ Complete Data Validation Program

Save the following code as `invalid_data.py`.

This example is self-contained. It creates a sample student dataset, performs multiple validation checks, generates an issue report, separates invalid records, and saves the results.

```python
import pandas as pd


def validate_student_data(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate student records and return the issue report and bad rows."""

    required_columns = {
        "Student_ID",
        "Name",
        "Age",
        "Marks",
        "Attendance",
        "Status",
        "Admission_Date"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    issues = []

    def add_issue(
        mask: pd.Series,
        column: str,
        issue_name: str
    ) -> None:
        """Record each failed validation for review."""
        for index in df.index[mask.fillna(False)]:
            issues.append({
                "Row_Index": index,
                "Student_ID": df.at[index, "Student_ID"],
                "Column": column,
                "Issue": issue_name,
                "Value": df.at[index, column]
            })

    # Required text values.
    for column in ["Student_ID", "Name"]:
        text = df[column].astype("string")

        invalid_required = (
            text.isna()
            | text.str.strip().eq("").fillna(False)
        )

        add_issue(
            invalid_required,
            column,
            "Missing or blank required value"
        )

    # Identifier format: S followed by three digits.
    student_id = df["Student_ID"].astype("string").str.strip()

    invalid_id = ~student_id.str.fullmatch(
        r"S\d{3}",
        na=False
    )

    add_issue(
        invalid_id,
        "Student_ID",
        "Invalid identifier format"
    )

    # Numeric conversion and invalid conversion reporting.
    numeric_columns = {
        "Age": ("Age", 0, 120),
        "Marks": ("Marks", 0, 100),
        "Attendance": ("Attendance", 0, 100)
    }

    numeric_data = {}

    for column, (label, minimum, maximum) in numeric_columns.items():
        original = df[column].astype("string")
        numeric = pd.to_numeric(df[column], errors="coerce")

        failed_conversion = (
            original.notna()
            & original.str.strip().ne("")
            & numeric.isna()
        )

        add_issue(
            failed_conversion,
            column,
            "Failed numeric conversion"
        )

        outside_range = (
            numeric.notna()
            & ~numeric.between(minimum, maximum)
        )

        add_issue(
            outside_range,
            column,
            f"Outside permitted range [{minimum}, {maximum}]"
        )

        numeric_data[column] = numeric

    # Allowed category vocabulary.
    allowed_statuses = {
        "Passed",
        "Failed",
        "Pending"
    }

    status_normalized = (
        df["Status"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    allowed_statuses_normalized = {
        value.casefold()
        for value in allowed_statuses
    }

    invalid_status = (
        status_normalized.notna()
        & status_normalized.ne("")
        & ~status_normalized.isin(allowed_statuses_normalized)
    )

    add_issue(
        invalid_status,
        "Status",
        "Unrecognized category"
    )

    # Date parsing.
    parsed_dates = pd.to_datetime(
        df["Admission_Date"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    original_dates = df["Admission_Date"].astype("string")

    invalid_date = (
        original_dates.notna()
        & original_dates.str.strip().ne("")
        & parsed_dates.isna()
    )

    add_issue(
        invalid_date,
        "Admission_Date",
        "Invalid or unparseable date"
    )

    # Detect duplicate identifiers.
    duplicate_id = student_id.duplicated(keep=False)

    add_issue(
        duplicate_id,
        "Student_ID",
        "Duplicate identifier"
    )

    issue_report = pd.DataFrame(
        issues,
        columns=[
            "Row_Index",
            "Student_ID",
            "Column",
            "Issue",
            "Value"
        ]
    )

    if issue_report.empty:
        invalid_indices = []
    else:
        invalid_indices = issue_report["Row_Index"].unique().tolist()

    invalid_records = df.loc[df.index.isin(invalid_indices)].copy()

    return issue_report, invalid_records


def main() -> None:
    raw_data = pd.DataFrame({
        "Student_ID": [
            "S001", "S002", "S003", "S004",
            "S005", "S006", "S007", "S008"
        ],
        "Name": [
            "Amit", "Priya", "Rahul", "Neha",
            "Kiran", "", "Meera", "Arjun"
        ],
        "Age": [20, 21, -5, 19, 22, 24, 150, 23],
        "Marks": [85, 92, 78, 150, -10, 88, 74, 91],
        "Attendance": [90, 95, 80, 88, 75, 120, 85, 92],
        "Status": [
            "Passed", "Passed", "Failed", "Passed",
            "Completedd", "Passed", "Failed", "Passed"
        ],
        "Admission_Date": [
            "2026-01-10", "2026-01-15", "2026-02-01",
            "2026-02-30", "2026-03-05", "2026-03-12",
            "2026-03-20", "2026-04-01"
        ]
    })

    print("ORIGINAL DATA")
    print(raw_data)

    print("\nDATA TYPES")
    print(raw_data.dtypes)

    issue_report, invalid_records = validate_student_data(raw_data)

    print("\nVALIDATION ISSUE REPORT")
    if issue_report.empty:
        print("No validation issues found.")
    else:
        print(issue_report.to_string(index=False))

    print("\nINVALID RECORDS FOR REVIEW")
    if invalid_records.empty:
        print("No invalid records found.")
    else:
        print(invalid_records.to_string(index=False))

    # Preserve the original input and the audit reports.
    raw_data.to_csv("student_data.csv", index=False)
    issue_report.to_csv("validation_issues.csv", index=False)
    invalid_records.to_csv("invalid_records.csv", index=False)

    print("\nSaved:")
    print("- student_data.csv")
    print("- validation_issues.csv")
    print("- invalid_records.csv")


if __name__ == "__main__":
    main()
```

### What this program demonstrates

- Checking for required columns.
- Validating required text fields.
- Checking identifier format and uniqueness.
- Detecting numeric conversion failures.
- Validating numeric ranges.
- Checking a controlled category vocabulary.
- Parsing dates with an explicit format.
- Generating a row-level issue report.
- Exporting invalid records for review.

The program reports issues instead of automatically changing or deleting the source data. A single row may have multiple issues, so the issue report can contain multiple entries for one student. This is intentional and helps explain every validation failure.

**Note:** The identifier format and category vocabulary are specific to this example. Adjust them to match your real dataset's schema.

## 🤖 Invalid Data in Machine Learning

Invalid records can interfere with preprocessing, model training, evaluation, and deployment.

Examples include:

- Negative age values that violate the data definition.
- Missing values in features that a model cannot accept.
- Unexpected categories that break an encoder.
- Incorrect dates that generate meaningless date-derived features.
- Target values outside the intended prediction domain.
- Data type inconsistencies between training and production.

### Recommended workflow

1. Define expected columns, types, ranges, and categories.
2. Validate raw data before modeling.
3. Correct invalid values using trusted evidence where possible.
4. Reject or quarantine records that cannot be resolved safely.
5. Separate deterministic validation from learned preprocessing.
6. Split data appropriately for model evaluation.
7. Fit learned imputers, encoders, and scalers on training data only.
8. Monitor prediction inputs for new data quality issues.

### Use a Scikit-Learn pipeline

Learned preprocessing should normally be fitted on the training partition only.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["Age", "Attendance"]
categorical_features = ["City"]

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

This example defines a preprocessing and classification pipeline. It expects an input dataset with `Age`, `Attendance`, `City`, and a separately defined target variable. Fit the pipeline on the training partition after applying the required validation rules.

`handle_unknown="ignore"` prevents an unseen category from crashing the encoder, but it does not make an unexpected value correct. Use monitoring and validation to identify categories that may require review.

### Data validation versus data preprocessing

| Task | Typical stage |
|---|---|
| Reject an impossible date | Input validation |
| Check a required identifier | Input validation |
| Correct a verified source error | Data cleaning |
| Impute a missing numeric feature using the training median | Learned preprocessing |
| Encode categorical features | Preprocessing pipeline |
| Scale numerical features | Preprocessing pipeline |
| Monitor changes in invalid-record rates | Production data quality monitoring |

**Important:** Do not use the held-out test set to learn statistical cleaning decisions that influence training. Deterministic rules defined by the domain can be applied consistently, but learned thresholds and preprocessing statistics should not use held-out information.

## ❌ Common Mistakes

1. **Assuming every unusual value is invalid.** Unusual values may be valid observations.
2. **Validating only data types.** Numeric values can still violate business rules.
3. **Ignoring failed conversions.** `errors="coerce"` may introduce missing values that need review.
4. **Deleting invalid rows without an audit trail.** This loses information and can hide systematic problems.
5. **Using broad rules without domain knowledge.** A negative value may be valid in some applications, such as representing a return.
6. **Treating every unknown category as a typo.** It may represent a legitimate new state.
7. **Ignoring relationships between columns.** Individually valid values may contradict one another.
8. **Confusing missing values with invalid values.** Each requires its own validation rule.
9. **Fitting learned preprocessing on the full dataset.** This can cause evaluation leakage.
10. **Failing to monitor incoming data.** New invalid values can appear after deployment.

## 🌟 Best Practices

- Define a clear schema before validation.
- Preserve raw data and record changes.
- Separate validation rules from correction and deletion logic.
- Use domain knowledge to establish allowed ranges.
- Check missing values, blank strings, and failed conversions separately.
- Validate category vocabularies using approved reference values.
- Use explicit date formats where the source format is known.
- Check uniqueness and referential integrity when required.
- Validate relationships between columns, not only individual fields.
- Generate actionable validation reports.
- Quarantine problematic records when immediate correction is not possible.
- Test validation functions with boundary values and missing inputs.
- Version important rules so results remain reproducible.
- Revalidate data whenever source schemas or business rules change.
- Monitor error rates and recurring problems in production pipelines.

## 💡 Mini Projects

### Project 1: Student Data Validation

Create a student dataset containing age, marks, attendance, status, and admission date.

Tasks:

1. Define allowed ranges and categories.
2. Check types and required fields.
3. Detect invalid numeric values.
4. Validate dates and student identifiers.
5. Generate a validation issue report.
6. Export invalid records for review.

### Project 2: E-commerce Order Validation

Create an order dataset with order ID, customer ID, quantity, unit price, total amount, and payment status.

Tasks:

1. Validate required identifiers.
2. Check quantity and price constraints.
3. Verify calculated totals.
4. Check allowed payment statuses.
5. Validate customer references.
6. Separate invalid orders into a review dataset.

### Project 3: Employee Dataset Validation

Create an employee dataset with employee ID, age, salary, department, joining date, and termination date.

Tasks:

1. Validate employee ID format and uniqueness.
2. Define valid numeric ranges.
3. Check approved departments.
4. Ensure termination dates do not precede joining dates.
5. Distinguish missing termination dates from invalid dates.
6. Produce an issue summary.

### Project 4: ML Dataset Quality Checker

Build a reusable validator for a machine learning dataset.

Tasks:

1. Validate required feature columns.
2. Detect missing or unconvertible numerical values.
3. Check expected categories.
4. Validate the target variable.
5. Report invalid rows and reasons.
6. Integrate validation into a reproducible training workflow.

## 📝 Practice Exercises

1. Create a dataset containing a negative age and marks above 100.
2. Write a function that validates a numeric range.
3. Identify values that fail numeric conversion.
4. Validate categories against an approved set.
5. Parse dates and identify impossible calendar dates.
6. Find blank values in required text columns.
7. Validate a student ID with a regular expression.
8. Identify duplicate identifiers when uniqueness is required.
9. Check whether an order's total price matches its line items.
10. Write a validation report that records row index, column, issue, and original value.
11. Explain the difference between invalid data and a statistical outlier.
12. Describe how data validation and machine learning preprocessing differ.

## ⚡ Quick Revision

| Task | Pandas or Python approach |
|---|---|
| Inspect structure | `df.info()` |
| Count missing values | `df.isna().sum()` |
| Convert numeric data safely | `pd.to_numeric(..., errors="coerce")` |
| Parse dates | `pd.to_datetime(..., errors="coerce")` |
| Validate a numeric range | `series.between(minimum, maximum)` |
| Check allowed categories | `series.isin(allowed_values)` |
| Detect blank text | `series.astype("string").str.strip().eq("")` |
| Validate a text pattern | `series.str.fullmatch(pattern, na=False)` |
| Identify duplicate values | `series.duplicated(keep=False)` |
| Compare related columns | Boolean conditions and tolerances |
| Validate foreign-key references | `.isin(reference_ids)` |
| Create an issue report | A list of validation issues converted to a DataFrame |
| Export issues | `report.to_csv("validation_issues.csv", index=False)` |

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
├── 03-Incorrect-Data-Types/
│   └── README.md
│
├── 04-Inconsistent-Values/
│   └── README.md
│
├── 05-Outliers/
│   └── README.md
│
└── 06-Invalid-Data/
    ├── README.md
    ├── invalid_data.py
    ├── student_data.csv
    ├── validation_issues.csv
    └── invalid_records.csv
```

| File | Purpose |
|---|---|
| `README.md` | Concepts, validation rules, and examples |
| `invalid_data.py` | Reusable data validation program |
| `student_data.csv` | Original sample dataset |
| `validation_issues.csv` | Detailed validation failure report |
| `invalid_records.csv` | Records that require investigation |

The CSV output files can be generated by running the Python script. They do not need to be committed unless they are useful as reproducible learning examples. Avoid committing sensitive or confidential datasets.

## 🗺️ Learning Roadmap

Follow this progression to build practical data validation skills:

1. Understand the dataset schema and business rules.
2. Inspect the structure, types, and missing-value patterns.
3. Validate data types and numeric ranges.
4. Validate categorical vocabularies and text formats.
5. Validate dates, identifiers, and required fields.
6. Check logical relationships between columns.
7. Generate clear validation reports.
8. Correct verified errors and quarantine unresolved records.
9. Integrate validation into machine learning pipelines.
10. Test, monitor, and maintain validation rules as requirements evolve.

## 🎯 Key Takeaways

- Invalid data violates a defined schema, rule, format, or business constraint.
- Missing values, inconsistent values, outliers, and invalid values are related but distinct concepts.
- Successful numeric conversion does not guarantee business validity.
- Range, category, date, identifier, and cross-column checks all serve different purposes.
- Validation should identify problems before decisions about correction, imputation, or removal are made.
- Keeping raw values and recording validation issues improves traceability.
- Machine learning pipelines need consistent validation and preprocessing.
- Reliable validation is an ongoing part of data quality management, not a one-time cleaning operation.

---

## 👨‍💻 Author

**Kishor Patil**

Python | Data Science | Machine Learning | Artificial Intelligence

GitHub: [Kishor055](https://github.com/Kishor055)

Repository: [Machine-Learning](https://github.com/Kishor055/Machine-Learning)

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Add examples, improvements, or corrections.
4. Test your code.
5. Submit a pull request with a clear description.

## ⭐ Support

If this guide helps you learn data cleaning and machine learning, consider giving the repository a star on GitHub.

**Keep learning, keep building, and keep growing!** 🌱

---

*Part of the Machine Learning learning path: Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Model Training → Evaluation → Deployment.*

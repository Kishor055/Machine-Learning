# 🧹 Inconsistent Values in Python

🌱 **GROW → INSPECT → IDENTIFY → STANDARDIZE → VALIDATE → BUILD**

Welcome to **Inconsistent Values**, part of the Data Cleaning module in the Machine Learning learning path. This guide explains how to identify and correct inconsistent representations of the same information using Python and Pandas.

Inconsistent values are common in real-world datasets. For example, the same city may appear as `"Mumbai"`, `"mumbai"`, `" MUMBAI "`, or `"Bombay"`. These values may refer to the same place, but a computer can treat them as different categories until they are standardized.

This guide covers text normalization, category mapping, spelling corrections, whitespace removal, unit standardization, validation, reusable cleaning functions, and machine learning workflows.

---

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [What Are Inconsistent Values?](#-what-are-inconsistent-values)
3. [Why Inconsistent Values Matter](#-why-inconsistent-values-matter)
4. [Types of Inconsistent Values](#-types-of-inconsistent-values)
5. [Create a Sample Dataset](#-create-a-sample-dataset)
6. [Inspect Inconsistent Values](#-inspect-inconsistent-values)
7. [Standardize Text Values](#-standardize-text-values)
8. [Handle Whitespace and Special Characters](#-handle-whitespace-and-special-characters)
9. [Standardize Categorical Values](#-standardize-categorical-values)
10. [Correct Known Spelling Variations](#-correct-known-spelling-variations)
11. [Standardize Yes/No Values](#-standardize-yesno-values)
12. [Standardize Units and Numeric Representations](#-standardize-units-and-numeric-representations)
13. [Handle Missing and Placeholder Values](#-handle-missing-and-placeholder-values)
14. [Use Mapping and Replacement](#-use-mapping-and-replacement)
15. [Validate Standardized Values](#-validate-standardized-values)
16. [Complete Data Cleaning Example](#-complete-data-cleaning-example)
17. [Inconsistent Values in Machine Learning](#-inconsistent-values-in-machine-learning)
18. [Common Mistakes](#-common-mistakes)
19. [Best Practices](#-best-practices)
20. [Mini Projects](#-mini-projects)
21. [Practice Exercises](#-practice-exercises)
22. [Quick Revision](#-quick-revision)
23. [Project Structure](#-project-structure)
24. [Learning Roadmap](#-learning-roadmap)

---

## 🎯 Introduction

Data collected from multiple sources is rarely perfectly consistent. Different users, forms, applications, and organizations may record the same information in different ways.

For example, a student dataset might contain these department names:

```python
departments = [
    "Computer Science",
    "computer science",
    " COMPUTER SCIENCE ",
    "Comp Science",
    "CSE"
]
```

These values may represent one department, but they cannot safely be assumed to be equivalent without understanding the dataset's terminology and business rules.

The goal of cleaning is to produce consistent, meaningful values without changing the underlying meaning of the data.

### Learning objectives

By the end of this guide, you will be able to:

- Identify inconsistent values in a DataFrame.
- Normalize capitalization, whitespace, and text representations.
- Standardize categorical values using mappings.
- Correct known spelling variations.
- Handle multiple representations of Boolean values.
- Recognize missing-value placeholders.
- Convert different representations into a common unit.
- Validate the cleaned dataset.
- Build reusable cleaning functions.
- Prepare consistent data for machine learning.

## 🧠 What Are Inconsistent Values?

**Inconsistent values** occur when the same concept is represented in different ways, or when a dataset contains values that violate a defined format, vocabulary, or business rule.

### Example

| Raw value | Standardized value |
|---|---|
| `"mumbai"` | `"Mumbai"` |
| `" MUMBAI "` | `"Mumbai"` |
| `"Female"` | `"Female"` |
| `"F"` | `"Female"` |
| `"yes"` | `"Yes"` |
| `"Y"` | `"Yes"` |
| `"N/A"` | Missing value, when defined as a placeholder |
| `"Comp Sci"` | `"Computer Science"`, if confirmed by the domain rules |

The standardized values depend on the rules of the dataset. For example, `"Bombay"` may be an accepted historical name, an alias, or a distinct user-provided string depending on the application's purpose.

**Important:** Never replace values solely because they look different. First establish whether they have the same meaning.

## 🚀 Why Inconsistent Values Matter

Inconsistent values can lead to:

- Duplicate categories in reports and charts.
- Incorrect group-by and aggregation results.
- Inaccurate counts and percentages.
- Failed data validation.
- Poor data integration across systems.
- Unexpected machine learning features.
- Increased complexity in dashboards and applications.
- Inconsistent behavior between training and production data.

For example:

```python
import pandas as pd

df = pd.DataFrame({
    "City": ["Mumbai", "mumbai", "MUMBAI", "Pune", "Pune"]
})

print(df["City"].value_counts())
```

The city may appear as three separate categories instead of one.

Normalize the case and trim surrounding whitespace:

```python
df["City"] = (
    df["City"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

print(df["City"].value_counts())
```

The three Mumbai representations now share the same normalized representation: `"mumbai"`.

If the final display should use title case, that can be applied separately where appropriate. Title casing is not suitable for every type of text, especially names, abbreviations, product codes, and identifiers.

## 🧩 Types of Inconsistent Values

### 1. Case inconsistency

```text
Python
PYTHON
python
PyThOn
```

A case-normalization rule can map these to a common form.

### 2. Whitespace inconsistency

```text
"Mumbai"
" Mumbai"
"Mumbai "
"  Mumbai  "
```

Leading and trailing whitespace can usually be removed.

### 3. Abbreviation inconsistency

```text
"Computer Science"
"Comp Sci"
"CSE"
```

Use an approved mapping only after confirming that the abbreviations represent the same category in the dataset.

### 4. Spelling inconsistency

```text
"Maharahstra"
"Maharashtra"
"Mahaarashtra"
```

Known errors can be corrected using an explicit mapping or a reviewed correction table.

### 5. Different representations of Boolean values

```text
"Yes"
"yes"
"Y"
"1"
"True"
```

These may express the same condition in some datasets, but the meaning of values like `"1"` must be confirmed.

### 6. Inconsistent missing-value placeholders

```text
""
"N/A"
"NA"
"Not Available"
"Unknown"
```

Some are placeholders for missing information; others may describe a genuine category. Define this distinction before replacing them.

### 7. Unit inconsistency

```text
"5 km"
"5000 m"
```

Both represent the same distance, but a consistent unit is necessary for reliable calculations.

### 8. Format inconsistency

```text
"2026-10-19"
"19/10/2026"
```

These may represent the same date, provided that the date convention is known.

### 9. Domain-rule inconsistency

```text
Attendance = 120
Marks = -15
```

These values are numeric, but they may violate the rules of the dataset. Type conversion alone cannot detect every business-rule violation.

---

## 🧪 Create a Sample Dataset

Start with a dataset containing multiple kinds of inconsistencies.

```python
import pandas as pd

df = pd.DataFrame({
    "Student_ID": ["S001", "S002", "S003", "S004", "S005", "S006"],
    "Name": [" Amit ", "PRIYA", "rahul", "Neha", "  AMIT", "Kiran"],
    "City": ["mumbai", "MUMBAI ", "Pune", " pune ", "Mumbai", "PUNE"],
    "Department": [
        "Computer Science",
        "computer science",
        "Comp Sci",
        "CSE",
        "IT",
        "Information Technology"
    ],
    "Gender": ["M", "Female", "male", "F", "FEMALE", "Male"],
    "Passed": ["Yes", "Y", "No", "N", "yes", "TRUE"],
    "Attendance": ["90%", "85", "80 %", "95%", "N/A", "75%"]
})

print(df)
print(df.info())
```

This dataset has:

- Inconsistent letter case.
- Extra whitespace.
- Multiple department labels.
- Short and long gender labels.
- Multiple representations of pass status.
- Percentage values stored as text.
- A possible missing-value placeholder.

The next sections address each issue systematically.

## 🔍 Inspect Inconsistent Values

Before changing anything, inspect the raw values.

### 1. Find unique values

```python
print(df["City"].unique())
print(df["Department"].unique())
print(df["Gender"].unique())
```

### 2. Count each representation

```python
print(df["City"].value_counts(dropna=False))
print(df["Department"].value_counts(dropna=False))
```

### 3. Inspect all categorical columns

```python
categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns

for column in categorical_columns:
    print(f"\n--- {column} ---")
    print(df[column].value_counts(dropna=False))
```

### 4. Compare values after basic normalization

```python
city_comparison = pd.DataFrame({
    "Original": df["City"],
    "Normalized": (
        df["City"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
})

print(city_comparison)
```

### 5. Review suspiciously similar categories

```python
print(df["Department"].value_counts(dropna=False))
```

Two strings that appear similar may still have different meanings. Review the dataset documentation or ask the data owner before merging them.

---

## 🔤 Standardize Text Values

Text normalization makes the formatting of text more consistent.

### 1. Convert text to lowercase

```python
df["City"] = df["City"].astype("string").str.lower()
```

This is useful for case-insensitive comparisons.

### 2. Convert text to uppercase

```python
df["Student_ID"] = (
    df["Student_ID"]
    .astype("string")
    .str.upper()
)
```

Uppercase normalization can be useful for identifiers when the source system treats them as case-insensitive.

Do not uppercase all text columns blindly. Names, addresses, and display labels may require different rules.

### 3. Use case folding for case-insensitive matching

```python
df["City_Normalized"] = (
    df["City"]
    .astype("string")
    .str.casefold()
)
```

`casefold()` is designed for stronger Unicode-aware case-insensitive matching than `lower()`. It is particularly useful when text may contain characters from multiple languages.

### 4. Standardize display capitalization

For ordinary English city labels:

```python
df["City_Display"] = (
    df["City"]
    .astype("string")
    .str.strip()
    .str.title()
)
```

Output examples:

```text
mumbai  -> Mumbai
PUNE    -> Pune
```

**Caution:** `.str.title()` can alter proper names and acronyms in undesirable ways. Use explicit mappings for names whose capitalization must be preserved.

## 🧹 Handle Whitespace and Special Characters

### 1. Remove leading and trailing whitespace

```python
df["Name"] = (
    df["Name"]
    .astype("string")
    .str.strip()
)
```

### 2. Replace repeated spaces

```python
df["Name"] = df["Name"].str.replace(
    r"\s+",
    " ",
    regex=True
)
```

This replaces runs of whitespace with one ordinary space.

### 3. Remove unwanted surrounding whitespace from several columns

```python
text_columns = ["Name", "City", "Department", "Gender", "Passed"]

for column in text_columns:
    df[column] = df[column].astype("string").str.strip()
```

### 4. Normalize case and whitespace together

```python
df["City"] = (
    df["City"]
    .astype("string")
    .str.strip()
    .str.casefold()
)
```

Keep text normalization appropriate to the field. Removing punctuation or symbols from every column may damage product codes, email addresses, IDs, or meaningful names.

---

## 🏷️ Standardize Categorical Values

Categorical standardization is the process of mapping alternative representations to a canonical value.

Suppose the department column contains these values:

```python
print(df["Department"].value_counts(dropna=False))
```

The goal is to standardize approved variants of `"Computer Science"` and `"Information Technology"`.

### 1. Normalize formatting first

```python
df["Department"] = (
    df["Department"]
    .astype("string")
    .str.strip()
    .str.casefold()
)
```

### 2. Apply an explicit mapping

```python
department_mapping = {
    "computer science": "Computer Science",
    "comp sci": "Computer Science",
    "cse": "Computer Science",
    "it": "Information Technology",
    "information technology": "Information Technology"
}

df["Department"] = df["Department"].replace(
    department_mapping
)

print(df["Department"].value_counts(dropna=False))
```

The mapping merges abbreviations only because this example defines them as equivalent.

### 3. Find unrecognized categories

For a field with a known set of allowed values, compare observed values with the allowed set.

```python
allowed_departments = {
    "Computer Science",
    "Information Technology"
}

unexpected = set(df["Department"].dropna()) - allowed_departments

print("Unexpected departments:", unexpected)
```

Run this check after standardization. If categories are still unexpected, investigate rather than replacing them automatically.

### Recommended rule

Use a documented vocabulary for fields such as:

- Department
- Gender, where the collection schema defines its categories
- Employment status
- Payment status
- Product type
- Customer segment
- Education level

The vocabulary should support every legitimate value and should not discard meaningful distinctions.

---

## ✍️ Correct Known Spelling Variations

Spelling errors and aliases can create artificial categories.

Example:

```python
df = pd.DataFrame({
    "State": [
        "Maharashtra",
        "maharashtra",
        "Maharahstra",
        "Maharastra",
        "Karnataka"
    ]
})
```

### 1. Normalize the text

```python
df["State_Normalized"] = (
    df["State"]
    .astype("string")
    .str.strip()
    .str.casefold()
)
```

### 2. Apply reviewed corrections

```python
state_mapping = {
    "maharashtra": "Maharashtra",
    "maharahstra": "Maharashtra",
    "maharastra": "Maharashtra",
    "karnataka": "Karnataka"
}

df["State_Cleaned"] = df["State_Normalized"].replace(
    state_mapping
)

print(df)
```

### 3. Keep a correction table for auditability

```python
corrections = pd.DataFrame({
    "Raw_Value": ["maharahstra", "maharastra"],
    "Canonical_Value": ["Maharashtra", "Maharashtra"],
    "Reason": ["Known spelling error", "Known spelling variation"]
})

print(corrections)
```

This approach documents why a value changed and makes cleaning rules easier to review.

### What about fuzzy matching?

Fuzzy matching and edit-distance algorithms can identify values that look similar, such as `"Maharashtra"` and `"Maharahstra"`.

Use similarity scoring to generate **candidates for review**, not as an automatic guarantee that two categories are equivalent. Similar strings can represent genuinely different entities.

---

## ✅ Standardize Yes/No Values

Binary values are often recorded using different labels.

Example input values:

```python
statuses = ["Yes", "Y", "yes", "TRUE", "No", "N", "false", "Unknown"]
```

Create a mapping after defining the permitted representations.

```python
status_mapping = {
    "yes": True,
    "y": True,
    "true": True,
    "no": False,
    "n": False,
    "false": False
}

status_series = pd.Series(statuses, dtype="string")

normalized_status = (
    status_series
    .str.strip()
    .str.casefold()
)

mapped_status = normalized_status.map(status_mapping).astype("boolean")

result = pd.DataFrame({
    "Original": status_series,
    "Normalized": normalized_status,
    "Standardized": mapped_status
})

print(result)
```

`"Unknown"` becomes a missing Boolean value because it is not in the mapping.

Check the unknown values explicitly:

```python
unknown_mask = normalized_status.notna() & mapped_status.isna()

print("Values requiring review:")
print(status_series[unknown_mask])
```

Do not assume that values like `"1"` and `"0"` always mean yes and no. Confirm the source system's definition first.

---

## 📏 Standardize Units and Numeric Representations

Values can be inconsistent even when every entry can be parsed as a number.

For example, distances may be recorded in kilometres and metres.

### 1. Convert different distance units to kilometres

```python
import pandas as pd

df = pd.DataFrame({
    "Distance": [5, 5000, 2.5],
    "Unit": ["km", "m", "km"]
})

unit_to_km = {
    "km": 1.0,
    "m": 0.001
}

df["Unit"] = df["Unit"].astype("string").str.strip().str.casefold()

df["Distance_KM"] = (
    pd.to_numeric(df["Distance"], errors="coerce")
    * df["Unit"].map(unit_to_km)
)

print(df)
```

Both `5 km` and `5000 m` become `5.0 km`.

### 2. Convert percentage text to numeric values

```python
df = pd.DataFrame({
    "Attendance": ["90%", "85", "80 %", "95%"]
})

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

print(df)
```

Here, the standardized values use a 0–100 scale. If the application expects proportions, divide these values by 100 and document that convention.

### 3. Validate ranges after conversion

```python
invalid = ~df["Attendance"].between(0, 100)

print(df.loc[invalid])
```

Conversion handles representation; range checks handle validity. Both may be necessary.

---

## 🚫 Handle Missing and Placeholder Values

Some datasets represent missing information with strings instead of actual missing values.

Example:

```python
df = pd.DataFrame({
    "Name": ["Amit", "Priya", "Rahul", "Neha"],
    "Phone": ["9876543210", "N/A", "", "Not Available"],
    "City": ["Mumbai", "Pune", "Unknown", "Nashik"]
})
```

### 1. Define placeholders carefully

```python
missing_placeholders = [
    "",
    "N/A",
    "NA",
    "Not Available"
]
```

### 2. Replace only approved missing placeholders

```python
df["Phone"] = df["Phone"].replace(
    missing_placeholders,
    pd.NA
)
```

For case-insensitive text placeholders, normalize the strings before replacement.

```python
phone_text = df["Phone"].astype("string").str.strip()

missing_mask = phone_text.str.casefold().isin({
    "",
    "n/a",
    "na",
    "not available"
})

df["Phone"] = phone_text.mask(missing_mask, pd.NA)
```

### 3. Inspect the result

```python
print(df.isna().sum())
```

**Important:** `"Unknown"` may be a legitimate category rather than a missing value. Treat it as missing only when the data definition explicitly says that it represents unavailable information.

---

## 🧰 Use Mapping and Replacement

Pandas provides several useful tools for standardization.

### `replace()`

Use `replace()` when known values should be substituted.

```python
df["Gender"] = df["Gender"].replace({
    "M": "Male",
    "male": "Male",
    "F": "Female",
    "FEMALE": "Female"
})
```

Normalize case and whitespace first to reduce the number of mapping entries required.

### `map()`

Use `map()` when you want values not present in the mapping to become missing and then be reviewed.

```python
gender_mapping = {
    "m": "Male",
    "male": "Male",
    "f": "Female",
    "female": "Female"
}

gender_normalized = (
    df["Gender"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

df["Gender_Standardized"] = gender_normalized.map(
    gender_mapping
)
```

### Comparison

| Method | Behavior |
|---|---|
| `replace()` | Replaces specified values and leaves unmatched values unchanged |
| `map()` | Maps known values and typically returns missing values for unmatched values |
| `.str.strip()` | Removes surrounding whitespace |
| `.str.casefold()` | Normalizes text for case-insensitive comparison |
| `.str.replace()` | Replaces text patterns or literal substrings |
| `.isin()` | Checks membership in an allowed set |

Choose the method based on whether unknown values should be preserved, flagged, or treated as missing.

---

## 🔬 Validate Standardized Values

After cleaning, validate the output against defined rules.

### 1. Check unique values

```python
print(df["City"].value_counts(dropna=False))
print(df["Department"].value_counts(dropna=False))
```

Unexpected categories after standardization may indicate an incomplete mapping or a legitimate value that must be added to the schema.

### 2. Validate against an allowed set

```python
allowed_statuses = {"Active", "Inactive", "Pending"}

invalid_mask = (
    df["Status"].notna()
    & ~df["Status"].isin(allowed_statuses)
)

print(df.loc[invalid_mask])
```

This example assumes a `Status` column exists and that the allowed vocabulary is correct for the application.

### 3. Check newly created missing values

Preserve the original values when conversion or mapping can introduce missing data.

```python
original = df["Department"].copy()

mapping = {
    "computer science": "Computer Science",
    "it": "Information Technology"
}

normalized = (
    original
    .astype("string")
    .str.strip()
    .str.casefold()
)

standardized = normalized.map(mapping)

unmapped = original.notna() & standardized.isna()

print("Unmapped original values:")
print(original[unmapped])
```

This identifies values that could not be mapped using the supplied rules.

### 4. Check expected data types

```python
print(df.dtypes)
```

### 5. Check duplicate categories after normalization

```python
normalized_categories = (
    df["City"]
    .dropna()
    .astype("string")
    .str.strip()
    .str.casefold()
)

print(normalized_categories.value_counts())
```

Review counts before and after cleaning to make sure the transformations have the intended effect.

---

## 🛠️ Complete Data Cleaning Example

The following example creates a dataset with inconsistent student records, standardizes the values, validates the results, and saves the output.

Save it as `inconsistent_values.py`.

```python
import pandas as pd


def standardize_text(series: pd.Series) -> pd.Series:
    """Trim whitespace, collapse repeated spaces, and normalize case."""
    return (
        series
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize selected student dataset columns."""

    required_columns = {
        "Student_ID",
        "Name",
        "City",
        "Department",
        "Gender",
        "Passed",
        "Attendance"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    cleaned = df.copy()

    # Standardize identifiers without converting them to numbers.
    cleaned["Student_ID"] = (
        cleaned["Student_ID"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # Normalize names for consistent whitespace and case.
    # Title case is a display rule and may need adjustment for real names.
    cleaned["Name"] = (
        cleaned["Name"]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.title()
    )

    # Standardize known city variants.
    city = standardize_text(cleaned["City"])

    city_mapping = {
        "mumbai": "Mumbai",
        "bombay": "Mumbai",
        "pune": "Pune",
        "bangalore": "Bengaluru",
        "bengaluru": "Bengaluru"
    }

    cleaned["City"] = city.map(city_mapping).astype("string")

    # Standardize known department labels.
    department = standardize_text(cleaned["Department"])

    department_mapping = {
        "computer science": "Computer Science",
        "comp sci": "Computer Science",
        "cse": "Computer Science",
        "it": "Information Technology",
        "information technology": "Information Technology"
    }

    cleaned["Department"] = department.map(
        department_mapping
    ).astype("string")

    # Standardize approved gender labels.
    gender = standardize_text(cleaned["Gender"])

    gender_mapping = {
        "m": "Male",
        "male": "Male",
        "f": "Female",
        "female": "Female"
    }

    cleaned["Gender"] = gender.map(gender_mapping).astype("string")

    # Standardize binary pass/fail values.
    passed = standardize_text(cleaned["Passed"])

    passed_mapping = {
        "yes": True,
        "y": True,
        "true": True,
        "no": False,
        "n": False,
        "false": False
    }

    cleaned["Passed"] = passed.map(passed_mapping).astype("boolean")

    # Standardize attendance to a numeric 0–100 scale.
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

    # Convert explicit missing placeholders to missing values.
    for column in ["City", "Department", "Gender"]:
        cleaned[column] = cleaned[column].replace(
            ["", "NA", "N/A", "Unknown"],
            pd.NA
        )

    return cleaned


def validate_student_data(df: pd.DataFrame) -> None:
    """Validate standardized student data and report issues."""

    allowed_cities = {"Mumbai", "Pune", "Bengaluru"}
    allowed_departments = {
        "Computer Science",
        "Information Technology"
    }
    allowed_genders = {"Male", "Female"}

    checks = {
        "Unexpected cities": (
            df["City"].notna()
            & ~df["City"].isin(allowed_cities)
        ),
        "Unexpected departments": (
            df["Department"].notna()
            & ~df["Department"].isin(allowed_departments)
        ),
        "Unexpected gender values": (
            df["Gender"].notna()
            & ~df["Gender"].isin(allowed_genders)
        ),
        "Invalid attendance values": (
            df["Attendance"].notna()
            & ~df["Attendance"].between(0, 100)
        )
    }

    for label, mask in checks.items():
        count = int(mask.sum())
        print(f"{label}: {count}")

        if count:
            print(df.loc[mask])

    print("\nMissing values:")
    print(df.isna().sum())


def main() -> None:
    raw_data = pd.DataFrame({
        "Student_ID": [
            "s001", "S002", " s003 ", "S004", "S005", "S006"
        ],
        "Name": [
            " Amit ", "PRIYA", "rahul", " Neha  ", "KIRAN", "Meera"
        ],
        "City": [
            "mumbai", "MUMBAI ", "Pune", " pune ",
            "Bombay", "Bangalore"
        ],
        "Department": [
            "Computer Science", "computer science", "Comp Sci",
            "CSE", "IT", "Information Technology"
        ],
        "Gender": [
            "M", "Female", "male", "F", "FEMALE", "Male"
        ],
        "Passed": [
            "Yes", "Y", "No", "N", "yes", "TRUE"
        ],
        "Attendance": [
            "90%", "85", "80 %", "95%", "N/A", "75%"
        ]
    })

    print("BEFORE CLEANING")
    print(raw_data)
    print("\nOriginal data types:")
    print(raw_data.dtypes)

    cleaned_data = clean_student_data(raw_data)

    print("\nAFTER CLEANING")
    print(cleaned_data)
    print("\nCleaned data types:")
    print(cleaned_data.dtypes)

    print("\nVALIDATION")
    validate_student_data(cleaned_data)

    cleaned_data.to_csv(
        "cleaned_student_data.csv",
        index=False
    )

    print("\nSaved cleaned_student_data.csv")


if __name__ == "__main__":
    main()
```

### Important considerations

- The mapping is intentionally explicit. Review the canonical values and aliases before using the script on real datasets.
- Unmapped categories become missing in this example, making it important to review the resulting missing-value counts.
- `"N/A"` in attendance becomes an unparseable value and therefore `NaN`.
- City aliases such as `"Bombay"` and `"Mumbai"` are merged only because this sample defines them as equivalent.
- Gender categories are determined by the sample schema; real applications should preserve the categories actually supported by the source and use case.
- The sample validation reports problems but does not automatically reject every invalid row. Production systems should define what happens when validation fails.

---

## 🤖 Inconsistent Values in Machine Learning

Inconsistent representations can change the features seen by a machine learning model.

For example, an uncleaned categorical feature may treat the following as three separate categories:

```text
Pune
pune
 PUNE
```

After case and whitespace normalization, they can be represented consistently as `"pune"`.

However, **standardization rules must be learned from domain knowledge or defined in advance**, not invented from test-set patterns.

### Recommended workflow

1. Inspect training data and establish a clear data schema.
2. Define canonical category names and valid aliases.
3. Standardize raw values using approved deterministic rules.
4. Identify and report unknown categories.
5. Fit learned preprocessing transformations on training data only.
6. Apply the same transformation rules during validation, testing, and inference.
7. Monitor newly appearing categories after deployment.

### Pandas cleaning versus Scikit-Learn preprocessing

Deterministic cleaning—such as trimming whitespace or applying a fixed, approved alias mapping—can often be performed consistently before splitting, provided the rules are independent of dataset-wide statistical patterns and the target.

Learned transformations—such as imputing values using a median, inferring categories from observed data, or selecting features based on statistics—should generally be fitted on the training data only.

For learned transformations, Scikit-Learn pipelines help keep training and inference behavior consistent.

### Example: Consistent categorical encoding

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    "Age": [20, 21, 19, 22, 23, 20, 24, 21],
    "City": [
        "Mumbai", "mumbai", "Pune", "PUNE",
        "Mumbai", "pune", "Mumbai", "Pune"
    ],
    "Passed": [1, 1, 0, 1, 1, 0, 1, 0]
})

# Apply a predefined deterministic normalization rule.
df["City"] = (
    df["City"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

X = df[["Age", "City"]]
y = df["Passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer([
    (
        "numeric",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        ["Age"]
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        ["City"]
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

print("Test score:", model.score(X_test, y_test))
```

The example uses a small, artificial dataset to demonstrate the workflow. A real model requires sufficient representative observations and suitable evaluation.

`handle_unknown="ignore"` prevents an unseen category from crashing the encoder during transformation. It does not automatically determine the meaning of that category; production systems should still monitor unexpected values.

---

## ❌ Common Mistakes

### 1. Replacing values without confirming their meaning

Two labels that look similar may identify different things.

**Solution:** Use documented alias mappings and domain knowledge.

### 2. Applying `.str.title()` to every text column

This may damage abbreviations, product codes, or proper names.

**Solution:** Define formatting rules for each field.

### 3. Treating every unknown value as missing

An unrecognized category may be a valid new value.

**Solution:** Flag unexpected values for review rather than discarding them automatically.

### 4. Using `replace()` without validating its results

A replacement may leave many variants untouched.

**Solution:** Check the unique values and counts after the transformation.

### 5. Using fuzzy matching as an automatic correction system

Similar-looking strings are not always equivalent.

**Solution:** Use fuzzy matching to suggest candidates for human or rule-based review.

### 6. Combining categories that have different meanings

Merging `"Full Time"` and `"Part Time"` would be incorrect even if the labels have similar formatting.

**Solution:** Maintain a clear category definition.

### 7. Removing meaningful special characters

Characters such as hyphens, apostrophes, or symbols may be part of IDs, names, or product codes.

**Solution:** Clean only the characters that the column's specification identifies as irrelevant.

### 8. Ignoring data quality after transformation

A standardized column may still contain invalid values.

**Solution:** Validate the allowed vocabulary, value ranges, and missing-value counts.

### 9. Using different cleaning logic for training and deployment

Different normalization steps can result in different categories at prediction time.

**Solution:** Reuse versioned cleaning rules and preprocessing pipelines.

---

## 🌟 Best Practices

- Preserve the original dataset and work on a copy.
- Profile categorical columns before cleaning.
- Define canonical values for important categories.
- Use explicit mappings for known aliases and misspellings.
- Normalize case and whitespace consistently.
- Keep a record of rules, corrections, and their purpose.
- Distinguish missing values from legitimate values such as `"Unknown"`.
- Validate all values against a documented schema.
- Review unmapped values rather than silently discarding them.
- Standardize measurement units before combining or comparing values.
- Test transformations with valid, invalid, missing, and unexpected inputs.
- Keep deterministic normalization rules consistent across training and inference.
- Version important cleaning rules to support reproducibility.
- Monitor incoming data for new categories and unexpected representations.

## 💡 Mini Projects

### Project 1: Student Dataset Standardization

Create a dataset with inconsistent city, department, gender, and pass-status values.

**Tasks:**
1. Display unique values and counts before cleaning.
2. Standardize whitespace and case.
3. Create canonical mappings for approved category variants.
4. Identify unknown and unmapped values.
5. Validate the resulting categories.
6. Save the cleaned dataset to CSV.

### Project 2: E-commerce Data Cleaning

Create a product dataset with inconsistent category names, brand spellings, and price representations.

**Tasks:**
1. Normalize category names.
2. Define approved aliases for product categories.
3. Parse price strings into numbers.
4. Identify unknown brands.
5. Validate price and stock ranges.
6. Generate a data quality report.

### Project 3: Survey Response Standardization

Build a survey dataset with inconsistent answers such as `"Yes"`, `"Y"`, `"yes"`, `"No"`, `"N"`, and `"Unknown"`.

**Tasks:**
1. Define valid responses.
2. Standardize case and whitespace.
3. Map the approved representations.
4. Report values outside the vocabulary.
5. Preserve missing and unknown responses according to the survey specification.
6. Document the cleaning rules.

### Project 4: Unit Standardization

Create a dataset containing distances, weights, or lengths in different units.

**Tasks:**
1. Identify the unit associated with each value.
2. Convert every value into one canonical unit.
3. Validate the conversion.
4. Compare values before and after normalization.
5. Document the unit convention in the dataset schema.

---

## 📝 Practice Exercises

1. Create a DataFrame containing `"Mumbai"`, `"mumbai"`, `"MUMBAI"`, and `" Mumbai "`. Standardize the values.
2. Write a function that trims whitespace and collapses repeated spaces.
3. Map `"CSE"`, `"Comp Sci"`, and `"Computer Science"` to a canonical department name.
4. Convert `"M"`, `"male"`, `"F"`, and `"female"` to the categories defined by a gender field's schema.
5. Standardize `"Yes"`, `"Y"`, `"TRUE"`, `"No"`, and `"N"` using an explicit mapping.
6. Identify values that do not match an allowed category list.
7. Convert a dataset containing metres and kilometres into a single distance unit.
8. Replace only those placeholder strings that the data specification defines as missing.
9. Build a function that returns both a cleaned DataFrame and a report of unexpected values.
10. Compare category counts before and after standardization and explain every significant change.

## ⚡ Quick Revision

| Task | Pandas approach |
|---|---|
| Find distinct values | `series.unique()` |
| Count categories | `series.value_counts(dropna=False)` |
| Remove surrounding spaces | `series.str.strip()` |
| Collapse repeated whitespace | `series.str.replace(r"\s+", " ", regex=True)` |
| Normalize case | `series.str.casefold()` |
| Replace known aliases | `series.replace(mapping)` |
| Map approved labels | `series.map(mapping)` |
| Check allowed values | `series.isin(allowed_values)` |
| Find missing values | `series.isna()` |
| Check column missing counts | `df.isna().sum()` |
| Convert numeric text | `pd.to_numeric()` |
| Validate numeric ranges | `series.between(minimum, maximum)` |
| Save cleaned data | `df.to_csv("cleaned.csv", index=False)` |

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
└── 04-Inconsistent-Values/
    ├── README.md
    ├── inconsistent_values.py
    ├── student_data.csv
    └── cleaned_student_data.csv
```

| File | Purpose |
|---|---|
| `README.md` | Concepts, examples, and learning guide |
| `inconsistent_values.py` | Standardization and validation logic |
| `student_data.csv` | Sample raw dataset |
| `cleaned_student_data.csv` | Cleaned output generated by the script |

Only include the output CSV if you intend to track sample results. Avoid committing confidential or personally identifiable data to a public repository.

## 🗺️ Learning Roadmap

Follow this sequence to strengthen your data-cleaning skills:

1. Inspect raw data and understand its schema.
2. Identify formatting and category inconsistencies.
3. Normalize whitespace and letter case.
4. Build mappings for known aliases and spelling variations.
5. Standardize missing values and units according to explicit rules.
6. Validate results against an allowed vocabulary and business rules.
7. Build reusable functions and tests.
8. Integrate consistent transformations into a machine learning workflow.
9. Monitor data quality as new records arrive.

## 🎯 Key Takeaways

- Inconsistent values represent differences in format, spelling, categories, or conventions.
- Similar-looking values should only be merged when they have the same meaning.
- Pandas provides tools such as `.str.strip()`, `.str.casefold()`, `.replace()`, `.map()`, and `.isin()` for standardization.
- Explicit mappings are generally safer than broad, unreviewed substitutions.
- Unknown categories and missing-value placeholders must be handled according to the dataset's specification.
- Unit conversion and business-rule validation are important parts of data quality.
- Reliable cleaning requires validation, documentation, testing, and consistency between training and production.

---

## 👨‍💻 Author

**Kishor Patil**

Python | Data Science | Machine Learning | AI

GitHub: [Kishor055](https://github.com/Kishor055)

Repository: [Machine-Learning](https://github.com/Kishor055/Machine-Learning)

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Add improvements, examples, or corrections.
4. Test the code.
5. Submit a pull request with a clear description.

## ⭐ Support

If this resource helps you learn data cleaning and machine learning, consider giving the repository a star on GitHub.

**Keep learning, keep building, and keep growing!** 🌱

---

*Part of the Machine Learning learning path: Data Collection → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Model Training → Evaluation → Deployment.*

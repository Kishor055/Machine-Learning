# 📄 CSV Data

> A practical and professional guide to understanding, collecting, loading, validating, analyzing, and preparing CSV datasets for Machine Learning.

CSV (**Comma-Separated Values**) is one of the most widely used formats for exchanging tabular data.

It is simple enough for humans to read, supported by almost every programming language, and commonly used to move data between:

* Applications
* Databases
* Spreadsheets
* APIs
* Data pipelines
* Analytics systems
* Machine Learning projects

However, a CSV file is only a **representation of data**.

Before using it for Machine Learning, you must understand:

```text
File Structure
     ↓
Encoding
     ↓
Delimiter
     ↓
Header
     ↓
Schema
     ↓
Data Types
     ↓
Missing Values
     ↓
Duplicates
     ↓
Invalid Values
     ↓
Data Quality
     ↓
Machine Learning
```

---

# 📚 Table of Contents

* [What is CSV?](#-what-is-csv)
* [Why CSV Matters](#-why-csv-is-important)
* [CSV Structure](#-csv-file-structure)
* [Basic CSV Example](#-basic-csv-example)
* [Rows and Columns](#-rows-and-columns)
* [Headers](#-headers)
* [Delimiters](#-delimiters)
* [Quoting](#-quoting)
* [Escaping](#-escaping)
* [Encoding](#-encoding)
* [CSV Data Types](#-csv-data-types)
* [Missing Values](#-missing-values)
* [CSV Limitations](#-csv-limitations)
* [Python CSV Module](#-python-csv-module)
* [Reading CSV with Python](#-reading-csv-with-python)
* [Writing CSV with Python](#-writing-csv-with-python)
* [Pandas and CSV](#-pandas-and-csv)
* [Reading CSV with Pandas](#-reading-csv-with-pandas)
* [Inspecting a CSV Dataset](#-inspecting-a-csv-dataset)
* [Selecting Columns](#-selecting-columns)
* [Filtering Rows](#-filtering-rows)
* [Handling Missing Values](#-handling-missing-values)
* [Data Type Conversion](#-data-type-conversion)
* [Dates and Times](#-dates-and-times)
* [Cleaning CSV Data](#-cleaning-csv-data)
* [Duplicates](#-duplicates)
* [Outliers](#-outliers)
* [Validation](#-csv-validation)
* [Large CSV Files](#-working-with-large-csv-files)
* [Chunk Processing](#-chunk-processing)
* [Compressed CSV](#-compressed-csv)
* [Encoding Problems](#-handling-encoding-problems)
* [Malformed CSV](#-handling-malformed-csv-files)
* [Security Considerations](#-csv-security-considerations)
* [Data Leakage](#-csv-data-and-data-leakage)
* [CSV Data Pipeline](#-csv-data-pipeline)
* [Recommended Directory Structure](#-recommended-directory-structure)
* [CSV Quality Checklist](#-csv-quality-checklist)
* [Common Mistakes](#-common-mistakes)
* [Mini Projects](#-mini-projects)
* [Best Practices](#-best-practices)
* [Key Takeaways](#-key-takeaways)

---

# 📖 What is CSV?

CSV stands for:

> **Comma-Separated Values**

A CSV file stores tabular data as plain text.

Example:

```text
id,name,age,salary
1,Alice,25,45000
2,Bob,30,60000
3,Charlie,28,52000
```

Each line usually represents a record.

Each separator represents a field.

Conceptually:

```text
CSV File
   │
   ├── Row 1
   ├── Row 2
   ├── Row 3
   │
   └── Columns
        ├── id
        ├── name
        ├── age
        └── salary
```

---

# 🎯 Why CSV Is Important

CSV remains common because it is:

* Simple
* Portable
* Human-readable
* Easy to generate
* Easy to inspect
* Supported by many tools
* Easy to version-control
* Compatible with Python
* Compatible with databases
* Common in ML datasets

Typical workflow:

```text
Database
    ↓
Export CSV
    ↓
Python / Pandas
    ↓
Data Cleaning
    ↓
EDA
    ↓
Feature Engineering
    ↓
Machine Learning
```

---

# 🧱 CSV File Structure

A typical CSV contains:

```text
Header
Record
Record
Record
...
```

Example:

```text
customer_id,age,city,purchases
1001,24,Mumbai,5
1002,31,Pune,8
1003,28,Nashik,3
```

The structure can be visualized as:

```text
                 CSV
                  │
        ┌─────────┴─────────┐
        │                   │
     Header              Records
        │                   │
        ▼             ┌─────┼─────┐
    Column Names      Row   Row   Row
```

---

# 📝 Basic CSV Example

Create a file named:

```text
customers.csv
```

Contents:

```csv
customer_id,name,age,city,purchases
1001,Alice,25,Mumbai,5
1002,Bob,31,Pune,8
1003,Charlie,28,Nashik,3
1004,David,35,Mumbai,12
```

This represents:

| customer_id | name    | age | city   | purchases |
| ----------: | ------- | --: | ------ | --------: |
|        1001 | Alice   |  25 | Mumbai |         5 |
|        1002 | Bob     |  31 | Pune   |         8 |
|        1003 | Charlie |  28 | Nashik |         3 |
|        1004 | David   |  35 | Mumbai |        12 |

---

# 📊 Rows and Columns

CSV data can be represented as:

```text
Rows    → observations / records
Columns → variables / features
```

For example:

```text
customer_id | age | income | churn
------------|-----|--------|------
1001        | 25  | 45000  | 0
1002        | 31  | 62000  | 1
1003        | 28  | 51000  | 0
```

Here:

* Each row represents a customer.
* `age`, `income`, and `churn` are variables.
* `churn` could be the target variable.

---

# 🏷️ Headers

A header identifies the columns.

Example:

```csv
id,name,age,salary
```

Without a header:

```csv
1,Alice,25,45000
2,Bob,30,60000
```

Python may interpret the first row as data unless instructed otherwise.

With Pandas:

```python
import pandas as pd

df = pd.read_csv(
    "data.csv",
    header=None,
)
```

You can provide names:

```python
df = pd.read_csv(
    "data.csv",
    header=None,
    names=["id", "name", "age", "salary"],
)
```

---

# 🔣 Delimiters

CSV does not always use commas.

Common delimiters include:

```text
,
;
\t
|
```

Example semicolon-separated file:

```text
id;name;age
1;Alice;25
2;Bob;30
```

Read it with:

```python
df = pd.read_csv(
    "data.csv",
    sep=";",
)
```

For tab-separated files:

```python
df = pd.read_csv(
    "data.tsv",
    sep="\t",
)
```

> Always inspect the actual file format instead of assuming that every `.csv` uses commas.

---

# 📝 Quoting

Fields containing commas may need quotation marks.

Example:

```csv
id,name,address
1,Alice,"Mumbai, Maharashtra"
```

Without quoting, the comma inside the address could be interpreted as another separator.

Pandas generally handles standard CSV quoting automatically.

---

# 🔐 Escaping

Special characters may require escaping depending on the CSV producer and parser.

Example:

```csv
id,name,comment
1,Alice,"She said ""Hello"""
```

The CSV representation uses doubled quotes to represent a quote character inside a quoted field.

When working with unusual CSV files, inspect how the source system generates the file before changing parser settings.

---

# 🌍 Encoding

Encoding determines how text is represented as bytes.

Common encodings include:

```text
UTF-8
UTF-8-SIG
UTF-16
Latin-1
Windows-1252
```

UTF-8 is generally a strong default for modern data exchange.

Read explicitly when necessary:

```python
df = pd.read_csv(
    "customers.csv",
    encoding="utf-8",
)
```

If a source contains a UTF-8 byte-order mark:

```python
df = pd.read_csv(
    "customers.csv",
    encoding="utf-8-sig",
)
```

---

# 🔢 CSV Data Types

CSV itself does not provide the same strong schema system as a database table.

For example:

```csv
id,age,income,active
1001,25,45000,True
```

When loaded into Python, you need to verify how each column was interpreted.

Use:

```python
print(df.dtypes)
```

Typical Pandas types include:

```text
int64
float64
bool
object
string
datetime64
category
```

---

# ❓ Missing Values

CSV files may represent missing information in many ways:

```text
(empty)
NA
N/A
NaN
null
NULL
?
unknown
```

Example:

```csv
id,name,age
1,Alice,25
2,Bob,
3,Charlie,NA
```

Specify recognized missing-value markers:

```python
df = pd.read_csv(
    "data.csv",
    na_values=["NA", "N/A", "NULL", "?"],
)
```

Always understand what a missing value means.

For example:

```text
NULL
```

might mean:

* Not collected
* Not applicable
* Unknown
* Failed measurement

These meanings are not necessarily interchangeable.

---

# ⚠️ CSV Limitations

CSV is useful, but it has limitations.

## 1. Weak Schema

A CSV does not reliably store rich data types.

---

## 2. No Relationships

CSV does not naturally represent:

```text
Foreign Keys
Relationships
Constraints
Indexes
```

---

## 3. Large Files Can Be Inefficient

Parsing very large CSV files can require substantial:

* CPU
* RAM
* Disk I/O

---

## 4. Date Types Are Not Native

A CSV stores dates as text.

Example:

```text
2026-09-29
```

Your application must interpret it.

---

## 5. Nested Data Is Awkward

Complex structures such as:

```json
{
  "customer": {
    "orders": [...]
  }
}
```

are not naturally represented in a simple CSV table.

---

# 🐍 Python CSV Module

Python includes a built-in `csv` module.

Import it:

```python
import csv
```

Advantages:

* No external dependency
* Fine-grained control
* Useful for simple CSV processing
* Good for streaming row-by-row

---

# 📥 Reading CSV with Python

```python
import csv

with open(
    "customers.csv",
    "r",
    newline="",
    encoding="utf-8",
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        print(row)
```

`DictReader` maps column names to values.

---

# 📤 Writing CSV with Python

```python
import csv

rows = [
    {"id": 1, "name": "Alice", "age": 25},
    {"id": 2, "name": "Bob", "age": 30},
]

with open(
    "customers.csv",
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["id", "name", "age"],
    )

    writer.writeheader()
    writer.writerows(rows)
```

---

# 🐼 Pandas and CSV

For Machine Learning and data analysis, Pandas is usually more convenient than manually processing every CSV row.

```python
import pandas as pd

df = pd.read_csv("customers.csv")
```

Pandas provides:

* Parsing
* Type inference
* Missing-value handling
* Filtering
* Aggregation
* Statistics
* Transformation
* Integration with NumPy and Scikit-Learn

---

# 📥 Reading CSV with Pandas

## Basic

```python
import pandas as pd

df = pd.read_csv("customers.csv")
```

## Custom separator

```python
df = pd.read_csv(
    "customers.csv",
    sep=";",
)
```

## Specific columns

```python
df = pd.read_csv(
    "customers.csv",
    usecols=[
        "customer_id",
        "age",
        "income",
    ],
)
```

## Limit rows

```python
df = pd.read_csv(
    "customers.csv",
    nrows=1000,
)
```

## Parse dates

```python
df = pd.read_csv(
    "transactions.csv",
    parse_dates=["transaction_date"],
)
```

---

# 🔍 Inspecting a CSV Dataset

Never start modeling immediately after loading the file.

Start with:

```python
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
```

Then:

```python
print(df.describe())
```

For categorical columns:

```python
print(df.describe(include="object"))
```

Check missing values:

```python
print(df.isna().sum())
```

Check duplicates:

```python
print(df.duplicated().sum())
```

---

# 📊 Dataset Profiling

A useful first inspection:

```python
def inspect_dataset(df):
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isna().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nSample:")
    print(df.head())
```

Use:

```python
inspect_dataset(df)
```

---

# 🧱 Selecting Columns

Single column:

```python
age = df["age"]
```

Multiple columns:

```python
subset = df[
    [
        "age",
        "income",
        "city",
    ]
]
```

---

# 🔎 Filtering Rows

Example:

```python
high_income = df.loc[
    df["income"] > 50000
]
```

Multiple conditions:

```python
filtered = df.loc[
    (df["age"] >= 25)
    & (df["income"] > 50000)
]
```

Use parentheses around individual conditions.

---

# 🧹 Handling Missing Values

First measure them:

```python
missing = df.isna().sum()

print(missing)
```

## Remove rows

```python
clean_df = df.dropna()
```

## Fill numerical values

```python
df["age"] = df["age"].fillna(
    df["age"].median()
)
```

## Fill categorical values

```python
df["city"] = df["city"].fillna(
    "Unknown"
)
```

> Missing-value handling should be based on the meaning of the variable and the modeling objective rather than applying one rule to every column.

---

# 🔢 Data Type Conversion

CSV parsing can infer an incorrect or undesirable type.

Example:

```python
df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce",
)
```

Invalid values become missing:

```text
"25"   → 25
"31"   → 31
"unknown" → NaN
```

For categorical strings:

```python
df["city"] = df["city"].astype("string")
```

---

# 📅 Dates and Times

CSV stores dates as text.

Example:

```text
2026-09-29
```

Convert explicitly:

```python
df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
)
```

Check invalid dates:

```python
invalid_dates = df["date"].isna().sum()

print("Invalid dates:", invalid_dates)
```

Once parsed, useful features can be created:

```python
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
```

---

# 🧼 Cleaning CSV Data

A typical cleaning workflow:

```text
CSV
 ↓
Load
 ↓
Inspect
 ↓
Standardize Column Names
 ↓
Fix Data Types
 ↓
Handle Missing Values
 ↓
Remove Duplicates
 ↓
Validate Ranges
 ↓
Handle Invalid Records
 ↓
Create Features
 ↓
Save Processed Dataset
```

Example:

```python
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
```

---

# 🔁 Duplicates

Check:

```python
duplicate_count = df.duplicated().sum()

print(
    "Duplicate rows:",
    duplicate_count,
)
```

Remove exact duplicate rows:

```python
df = df.drop_duplicates()
```

For key-based duplicates:

```python
df = df.drop_duplicates(
    subset=["customer_id"],
    keep="last",
)
```

Be careful:

> Duplicate-looking rows are not always erroneous. In transaction datasets, multiple records for the same customer may be legitimate.

---

# 📈 Outliers

CSV data can contain extreme values.

Example:

```text
age:
18
22
25
31
240
```

The value `240` may require investigation.

A simple IQR-based check:

```python
q1 = df["age"].quantile(0.25)
q3 = df["age"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df.loc[
    (df["age"] < lower)
    | (df["age"] > upper)
]
```

Do not automatically delete every outlier.

An extreme observation may be:

* An error
* A rare legitimate event
* A special population
* A measurement issue

---

# ✅ CSV Validation

Validation should happen before Machine Learning.

## Schema Validation

```python
required_columns = {
    "customer_id",
    "age",
    "income",
    "churn",
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )
```

---

# 🔢 Range Validation

Example:

```python
invalid_age = df.loc[
    (df["age"] < 0)
    | (df["age"] > 120)
]

print(
    "Invalid age records:",
    len(invalid_age),
)
```

---

# 🎯 Target Validation

Suppose `churn` should contain only:

```text
0
1
```

Validate:

```python
allowed_values = {0, 1}

actual_values = set(
    df["churn"]
    .dropna()
    .unique()
)

unexpected = (
    actual_values - allowed_values
)

if unexpected:
    raise ValueError(
        f"Unexpected target values: {unexpected}"
    )
```

---

# 🧪 Complete CSV Validation Example

```python
def validate_customer_csv(df):
    required_columns = {
        "customer_id",
        "age",
        "income",
        "churn",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if df["customer_id"].duplicated().any():
        raise ValueError(
            "customer_id contains duplicates."
        )

    if (
        df["age"].dropna()
        .lt(0)
        .any()
    ):
        raise ValueError(
            "Negative ages detected."
        )

    allowed_targets = {0, 1}

    actual_targets = set(
        df["churn"]
        .dropna()
        .unique()
    )

    if not actual_targets.issubset(
        allowed_targets
    ):
        raise ValueError(
            "Unexpected target values detected."
        )

    return True
```

---

# 💾 Writing CSV Files

Save a DataFrame:

```python
df.to_csv(
    "processed_customers.csv",
    index=False,
)
```

The `index=False` argument prevents the Pandas index from becoming an extra CSV column.

---

# 📦 CSV Compression

Large datasets can be compressed.

Example:

```python
df.to_csv(
    "customers.csv.gz",
    index=False,
    compression="gzip",
)
```

Read:

```python
df = pd.read_csv(
    "customers.csv.gz",
    compression="gzip",
)
```

Compression can reduce disk usage and transfer size.

---

# 🐘 Working with Large CSV Files

A large CSV may not fit comfortably into RAM.

Instead of:

```python
df = pd.read_csv("huge_dataset.csv")
```

consider:

```python
for chunk in pd.read_csv(
    "huge_dataset.csv",
    chunksize=100_000,
):
    process(chunk)
```

This allows incremental processing.

---

# 🔄 Chunk Processing

Example:

```python
import pandas as pd

total_rows = 0

for chunk in pd.read_csv(
    "transactions.csv",
    chunksize=50_000,
):
    total_rows += len(chunk)

print("Total rows:", total_rows)
```

You can aggregate incrementally:

```python
total_sales = 0.0

for chunk in pd.read_csv(
    "transactions.csv",
    chunksize=50_000,
):
    total_sales += chunk["amount"].sum()

print("Total sales:", total_sales)
```

This approach is useful when the complete dataset does not need to exist in memory simultaneously.

---

# 🌍 Handling Encoding Problems

A common error:

```text
UnicodeDecodeError
```

Possible approach:

```python
df = pd.read_csv(
    "data.csv",
    encoding="latin-1",
)
```

But do not blindly change encodings.

Prefer to determine the source system's actual encoding when possible.

---

# ⚠️ Handling Malformed CSV Files

Potential problems include:

```text
Unclosed quotation
Unexpected delimiter
Missing fields
Extra fields
Corrupted rows
Mixed encodings
```

First investigate the source.

For a controlled recovery workflow, Pandas can be configured to skip problematic records:

```python
df = pd.read_csv(
    "data.csv",
    on_bad_lines="skip",
)
```

However:

> Skipping bad records silently can remove important information.

A production pipeline should log rejected records and investigate why they failed.

---

# 🔐 CSV Security Considerations

CSV files can contain untrusted content.

Potential risks include:

* Malicious formulas
* Unexpected file content
* Sensitive information
* Path manipulation in automated workflows
* Extremely large fields
* Malformed input

## Spreadsheet Formula Injection

If CSV data is later opened in spreadsheet software, values beginning with characters such as:

```text
=
+
-
@
```

may be interpreted as formulas by some spreadsheet applications.

Treat externally supplied CSV content as untrusted data.

For systems exporting CSV for spreadsheet consumption, apply an appropriate output-sanitization strategy.

---

# 🚨 CSV Data and Data Leakage

Loading a CSV does not cause leakage by itself.

Leakage occurs when the dataset contains information that would not be available at prediction time or when preprocessing uses information from evaluation data.

Example:

```text
customer_id
age
income
monthly_spend
cancellation_date
churn
```

If predicting churn **before cancellation**, using `cancellation_date` may reveal future information.

A useful question is:

> **Would this column be available at the exact moment the prediction is supposed to be made?**

---

# 🏗️ CSV Data Pipeline

A professional ingestion workflow:

```text
                CSV Source
                    │
                    ▼
             File Validation
                    │
                    ▼
               Schema Check
                    │
                    ▼
              Load Raw Data
                    │
                    ▼
              Type Validation
                    │
                    ▼
           Missing-Value Analysis
                    │
                    ▼
            Duplicate Detection
                    │
                    ▼
             Range Validation
                    │
                    ▼
              Data Cleaning
                    │
                    ▼
             Feature Engineering
                    │
                    ▼
             Processed Dataset
                    │
                    ▼
             Machine Learning
```

---

# 🗂️ Recommended Directory Structure

```text
02-CSV-Data/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   └── customers.csv
│   │
│   ├── interim/
│   │   └── customers_cleaned.csv
│   │
│   └── processed/
│       └── customers_model_ready.csv
│
├── scripts/
│   ├── load_csv.py
│   ├── validate_csv.py
│   └── clean_csv.py
│
└── metadata/
    └── dataset.yaml
```

---

# 🧪 Recommended CSV Learning Workflow

Follow this progression:

```text
CSV Basics
    ↓
File Structure
    ↓
Headers & Delimiters
    ↓
Encoding
    ↓
Python csv Module
    ↓
Pandas read_csv()
    ↓
Dataset Inspection
    ↓
Data Types
    ↓
Missing Values
    ↓
Duplicates
    ↓
Data Validation
    ↓
Cleaning
    ↓
Large Files
    ↓
CSV → ML
```

---

# 📊 CSV → Pandas → Machine Learning

The complete relationship:

```text
CSV
 │
 │ read_csv()
 ▼
Pandas DataFrame
 │
 ├── Inspect
 ├── Clean
 ├── Validate
 ├── Transform
 └── Engineer Features
 │
 ▼
Train/Test Split
 │
 ▼
Preprocessing
 │
 ▼
Scikit-Learn Pipeline
 │
 ▼
Model
 │
 ▼
Evaluation
```

---

# 📋 CSV Quality Checklist

Before using a CSV for Machine Learning:

### File

* [ ] File exists
* [ ] File can be opened
* [ ] Encoding is known
* [ ] Delimiter is known
* [ ] Header is understood

### Structure

* [ ] Number of rows checked
* [ ] Number of columns checked
* [ ] Column names inspected
* [ ] Expected schema verified

### Data Types

* [ ] Numeric columns validated
* [ ] Categorical columns validated
* [ ] Boolean columns validated
* [ ] Date columns parsed
* [ ] Unexpected types identified

### Data Quality

* [ ] Missing values measured
* [ ] Duplicates checked
* [ ] Invalid values checked
* [ ] Outliers investigated
* [ ] Unique values inspected

### ML Readiness

* [ ] Target identified
* [ ] Leakage checked
* [ ] Features understood
* [ ] Train/test strategy defined
* [ ] Preprocessing strategy defined

### Governance

* [ ] Source documented
* [ ] Dataset version recorded
* [ ] License checked
* [ ] Privacy considerations reviewed
* [ ] Raw file preserved

---

# ❌ Common Mistakes

## 1. Assuming Every CSV Uses Commas

Some files use:

```text
;
|
\t
```

Inspect the file before loading.

---

## 2. Ignoring Encoding

International text can fail when the wrong encoding is used.

---

## 3. Trusting Automatically Inferred Types

Always inspect:

```python
print(df.dtypes)
```

---

## 4. Treating Empty Strings as Valid Data

An empty string may represent missing information.

---

## 5. Dropping All Missing Rows

This can unnecessarily remove a large percentage of your dataset.

Investigate missingness first.

---

## 6. Removing Every Outlier

Some outliers are legitimate observations.

---

## 7. Overwriting the Raw CSV

Keep the original dataset unchanged.

---

## 8. Silently Skipping Malformed Rows

If rows are rejected, log and investigate them.

---

## 9. Loading Huge CSVs Directly Into Memory

Use chunk processing when appropriate.

---

## 10. Training Before Validation

A CSV that loads successfully is not necessarily a valid ML dataset.

---

# 🧩 Mini Projects

## Project 1 — CSV Explorer

Build a Python program that displays:

```text
Filename
Rows
Columns
Column Names
Data Types
Missing Values
Duplicate Rows
Numeric Statistics
```

---

## Project 2 — CSV Validator

Create a validator that checks:

```text
Required Columns
Data Types
Age Range
Target Values
Duplicate IDs
Missing Required Values
```

---

## Project 3 — CSV Cleaning Pipeline

Build:

```text
Raw CSV
 ↓
Load
 ↓
Clean Column Names
 ↓
Convert Types
 ↓
Handle Missing Values
 ↓
Remove Invalid Records
 ↓
Save Processed CSV
```

---

## Project 4 — Large CSV Processor

Use `chunksize` to calculate:

* Total rows
* Total revenue
* Average transaction value
* Number of unique customers

without loading the complete dataset into memory.

---

## Project 5 — CSV-to-ML Pipeline

Build:

```text
customers.csv
      ↓
Validation
      ↓
Cleaning
      ↓
EDA
      ↓
Train/Test Split
      ↓
Preprocessing
      ↓
Model
      ↓
Evaluation
```

---

# 🧠 Best Practices

### Prefer explicit configuration

Instead of relying entirely on inference:

```python
df = pd.read_csv(
    "customers.csv",
    encoding="utf-8",
)
```

Specify additional parsing options when the source requires them.

### Validate immediately

Do not wait until model training to discover bad records.

### Keep raw data immutable

```text
raw → interim → processed
```

### Document assumptions

Record:

* What missing values mean
* Which rows were removed
* Why values were transformed
* Which columns are targets

### Use reproducible transformations

The same raw CSV should produce the same processed dataset when the pipeline and dependencies are unchanged.

### Prefer pipelines for ML preprocessing

For model training, use Scikit-Learn pipelines to reduce leakage risk and keep transformations reproducible.

---

# 🔬 Example End-to-End Workflow

```python
import pandas as pd

# 1. Load
df = pd.read_csv(
    "customers.csv",
    encoding="utf-8",
)

# 2. Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# 3. Convert numeric values
df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce",
)

df["income"] = pd.to_numeric(
    df["income"],
    errors="coerce",
)

# 4. Parse dates
df["signup_date"] = pd.to_datetime(
    df["signup_date"],
    errors="coerce",
)

# 5. Inspect missing values
print(df.isna().sum())

# 6. Remove exact duplicates
df = df.drop_duplicates()

# 7. Save processed data
df.to_csv(
    "customers_processed.csv",
    index=False,
)
```

This is an **exploratory data-cleaning example**.

For ML training, preprocessing decisions such as imputation, scaling, and encoding should generally be fitted using the training data only, preferably through a Scikit-Learn pipeline.

---

# 🏆 Key Takeaways

After completing this section, you should understand:

* What CSV files are
* How CSV files are structured
* How headers work
* How delimiters work
* Why quoting matters
* Why encoding matters
* How CSV differs from database tables
* How to read CSV files with Python
* How to read CSV files with Pandas
* How to inspect CSV datasets
* How to validate schemas
* How to detect missing values
* How to identify duplicates
* How to convert data types
* How to parse dates
* How to detect invalid values
* How to investigate outliers
* How to process large CSV files
* How to handle compressed CSV files
* How to preserve raw datasets
* How to document CSV provenance
* How CSV data enters a Machine Learning pipeline
* Why validation must happen before modeling
* Why data leakage must be considered

---

# 🚀 Next Step

Once you understand CSV data, continue to the next stage:

```text
CSV Data
   ↓
Data Collection
   ↓
Data Understanding
   ↓
Data Quality
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Machine Learning
```

The ultimate goal is not simply to load a CSV file.

The goal is to transform:

```text
Raw CSV
   ↓
Reliable Dataset
   ↓
Well-Understood Data
   ↓
Validated Features
   ↓
Reproducible ML Pipeline
```

> **A successful Machine Learning project begins long before `model.fit()`. It begins with understanding exactly what your data contains, where it came from, and whether it can be trusted.**

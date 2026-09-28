# 📊 Excel Data for Machine Learning

Excel is one of the most common formats used to store, exchange, and analyze real-world business data. Although machine learning workflows often use CSV, Parquet, databases, or APIs, many datasets begin life as an **Excel workbook (`.xlsx` / `.xls`)**.

This section teaches how to work with Excel data professionally—from reading worksheets and cleaning messy spreadsheets to validating data and preparing it for machine learning.

> **Goal:** Learn how to reliably extract, inspect, clean, validate, transform, and prepare Excel data for downstream data analysis and machine learning workflows.

---

## 📚 Table of Contents

* [1. What Is Excel Data?](#1-what-is-excel-data)
* [2. Why Excel Matters in Machine Learning](#2-why-excel-matters-in-machine-learning)
* [3. Excel File Formats](#3-excel-file-formats)
* [4. Workbook Structure](#4-workbook-structure)
* [5. Excel vs CSV](#5-excel-vs-csv)
* [6. Python Tools for Excel](#6-python-tools-for-excel)
* [7. Reading Excel with Pandas](#7-reading-excel-with-pandas)
* [8. Reading Specific Sheets](#8-reading-specific-sheets)
* [9. Inspecting an Excel Dataset](#9-inspecting-an-excel-dataset)
* [10. Selecting Columns and Rows](#10-selecting-columns-and-rows)
* [11. Filtering Excel Data](#11-filtering-excel-data)
* [12. Handling Missing Values](#12-handling-missing-values)
* [13. Cleaning Excel Data](#13-cleaning-excel-data)
* [14. Converting Data Types](#14-converting-data-types)
* [15. Working with Dates](#15-working-with-dates)
* [16. Handling Duplicate Records](#16-handling-duplicate-records)
* [17. Handling Excel Headers](#17-handling-excel-headers)
* [18. Working with Multiple Sheets](#18-working-with-multiple-sheets)
* [19. Combining Excel Worksheets](#19-combining-excel-worksheets)
* [20. Reading Multiple Excel Files](#20-reading-multiple-excel-files)
* [21. Writing Data to Excel](#21-writing-data-to-excel)
* [22. Multiple Sheets with Pandas](#22-multiple-sheets-with-pandas)
* [23. Excel Formatting and Data Quality](#23-excel-formatting-and-data-quality)
* [24. Merged Cells and Blank Rows](#24-merged-cells-and-blank-rows)
* [25. Validating Excel Data](#25-validating-excel-data)
* [26. Data Leakage](#26-data-leakage)
* [27. Preparing Excel Data for Machine Learning](#27-preparing-excel-data-for-machine-learning)
* [28. Excel Data Pipeline](#28-excel-data-pipeline)
* [29. Large Excel Files](#29-large-excel-files)
* [30. Common Excel Problems](#30-common-excel-problems)
* [31. Security Considerations](#31-security-considerations)
* [32. Recommended Project Structure](#32-recommended-project-structure)
* [33. Practical Exercises](#33-practical-exercises)
* [34. Mini Projects](#34-mini-projects)
* [35. Best Practices](#35-best-practices)
* [36. Professional Workflow](#36-professional-workflow)
* [37. Learning Roadmap](#37-learning-roadmap)
* [38. Key Takeaways](#38-key-takeaways)

---

# 1. What Is Excel Data?

Excel is a spreadsheet-based data format commonly stored in:

```text
.xlsx
.xls
.xlsm
```

An Excel workbook can contain:

* multiple worksheets
* rows and columns
* formulas
* formatting
* tables
* charts
* dates
* numeric values
* text
* hyperlinks
* named ranges
* metadata

A simple workbook might look like:

| Customer ID | Age | Income | City   | Purchased |
| ----------- | --: | -----: | ------ | --------: |
| 101         |  25 |  45000 | Pune   |       Yes |
| 102         |  32 |  62000 | Mumbai |        No |
| 103         |  29 |  55000 | Nashik |       Yes |

For machine learning, this can eventually become:

```text
Features (X)
    ↓
Age
Income
City

Target (y)
    ↓
Purchased
```

---

# 2. Why Excel Matters in Machine Learning

Many real-world organizations still exchange data through Excel.

Examples include:

* sales reports
* customer records
* financial reports
* employee data
* inventory
* survey results
* healthcare research datasets
* academic datasets
* marketing reports
* experiment results
* operational data

A typical workflow might be:

```text
Excel Workbook
      ↓
Read with Pandas
      ↓
Inspect
      ↓
Validate
      ↓
Clean
      ↓
Transform
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Machine Learning Pipeline
      ↓
Model
```

Therefore, knowing how to reliably process Excel files is an important practical data skill.

---

# 3. Excel File Formats

## `.xlsx`

Modern Excel workbook format.

```text
data.xlsx
```

This is the most common format for modern Excel files.

---

## `.xls`

Older Excel format.

```text
data.xls
```

Legacy workbooks may require additional dependencies.

---

## `.xlsm`

Excel workbook containing macros.

```text
data.xlsm
```

Be careful when processing macro-enabled files because macros may contain executable logic.

---

# 4. Workbook Structure

An Excel file is usually a **workbook**.

A workbook can contain multiple **worksheets**.

```text
sales.xlsx
│
├── Sales
├── Customers
├── Products
└── Summary
```

For example:

### Sales

| Order ID | Customer ID | Amount |
| -------- | ----------- | -----: |
| 1        | C001        |    500 |
| 2        | C002        |    800 |

### Customers

| Customer ID | Age | City   |
| ----------- | --: | ------ |
| C001        |  25 | Pune   |
| C002        |  31 | Mumbai |

These worksheets can later be joined using a key such as:

```text
Customer ID
```

---

# 5. Excel vs CSV

| Feature                | Excel                       | CSV                           |
| ---------------------- | --------------------------- | ----------------------------- |
| Multiple sheets        | ✅                           | ❌                             |
| Formatting             | ✅                           | ❌                             |
| Formulas               | ✅                           | ❌                             |
| Charts                 | ✅                           | ❌                             |
| Simple text format     | ❌                           | ✅                             |
| Human-friendly         | ✅                           | ✅                             |
| Easy version control   | ⚠️                          | ✅                             |
| ML interoperability    | ✅                           | ✅                             |
| Large-scale processing | ⚠️                          | Better alternatives available |
| Database ingestion     | Usually requires conversion | Easy                          |

Excel is useful for **human collaboration and reporting**.

CSV is often simpler for **data interchange and machine learning pipelines**.

For larger analytical workloads, formats such as **Parquet** are often preferable.

---

# 6. Python Tools for Excel

The most common Python tools are:

```text
Pandas
OpenPyXL
XLRD
XlsxWriter
```

## Pandas

Best for:

* reading Excel into DataFrames
* cleaning data
* analysis
* transformation
* feature engineering

```python
import pandas as pd
```

## OpenPyXL

Useful for working with `.xlsx` workbooks at a lower level.

```python
from openpyxl import load_workbook
```

## XlsxWriter

Useful for generating formatted Excel workbooks.

```python
import xlsxwriter
```

---

# 7. Reading Excel with Pandas

Install Pandas and the Excel engine:

```bash
pip install pandas openpyxl
```

Read an Excel file:

```python
import pandas as pd

df = pd.read_excel("data.xlsx")

print(df)
```

---

## Specify the Engine

```python
df = pd.read_excel(
    "data.xlsx",
    engine="openpyxl"
)
```

For modern `.xlsx` files, `openpyxl` is a common choice.

---

# 8. Reading Specific Sheets

Suppose the workbook contains:

```text
Sales
Customers
Products
```

Read one sheet:

```python
df = pd.read_excel(
    "data.xlsx",
    sheet_name="Sales"
)
```

Read another:

```python
customers = pd.read_excel(
    "data.xlsx",
    sheet_name="Customers"
)
```

---

## Read Sheet by Index

```python
df = pd.read_excel(
    "data.xlsx",
    sheet_name=0
)
```

The first worksheet has index:

```text
0
```

---

## Read Multiple Sheets

```python
sheets = pd.read_excel(
    "data.xlsx",
    sheet_name=["Sales", "Customers"]
)

sales = sheets["Sales"]
customers = sheets["Customers"]
```

---

## Read All Sheets

```python
sheets = pd.read_excel(
    "data.xlsx",
    sheet_name=None
)

for name, df in sheets.items():
    print(name)
    print(df.head())
```

This returns a dictionary:

```text
{
    "Sales": DataFrame,
    "Customers": DataFrame,
    "Products": DataFrame
}
```

---

# 9. Inspecting an Excel Dataset

After loading data, inspect it before modifying anything.

```python
import pandas as pd

df = pd.read_excel("data.xlsx")

print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
```

Useful commands:

```python
df.head()
df.tail()
df.sample(5)
df.shape
df.columns
df.dtypes
df.info()
df.describe()
```

---

## Inspect Missing Values

```python
print(df.isna().sum())
```

Percentage of missing values:

```python
missing_percentage = (
    df.isna().mean() * 100
)

print(missing_percentage)
```

---

# 10. Selecting Columns and Rows

Select one column:

```python
ages = df["Age"]
```

Select multiple columns:

```python
subset = df[
    ["Age", "Income", "City"]
]
```

Use `.loc` for label-based selection:

```python
subset = df.loc[
    df["Age"] > 25,
    ["Age", "Income"]
]
```

Use `.iloc` for positional selection:

```python
subset = df.iloc[0:10, 0:3]
```

---

# 11. Filtering Excel Data

Suppose:

```text
Age > 30
```

Then:

```python
filtered = df[
    df["Age"] > 30
]
```

Multiple conditions:

```python
filtered = df[
    (df["Age"] > 25)
    & (df["Income"] > 50000)
]
```

OR condition:

```python
filtered = df[
    (df["City"] == "Pune")
    | (df["City"] == "Mumbai")
]
```

Use `.isin()` for multiple categories:

```python
filtered = df[
    df["City"].isin(["Pune", "Mumbai"])
]
```

---

# 12. Handling Missing Values

Excel users often represent missing values as:

```text
blank
N/A
NA
-
?
Unknown
```

These should be standardized.

Inspect missing data:

```python
print(df.isna().sum())
```

---

## Replace Custom Missing Values

```python
df = df.replace(
    ["N/A", "NA", "-", "?"],
    pd.NA
)
```

---

## Drop Rows

```python
df = df.dropna()
```

Use this carefully.

Dropping every row containing a missing value may remove a large amount of useful data.

---

## Fill Numeric Values

```python
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)
```

---

## Fill Categorical Values

```python
df["City"] = df["City"].fillna(
    "Unknown"
)
```

For machine learning, imputation should generally be performed as part of the training pipeline so that statistics are learned only from the training data.

---

# 13. Cleaning Excel Data

Excel files often contain inconsistent values.

Example:

```text
Mumbai
mumbai
MUMBAI
 Mumbai
Mumbai 
```

Normalize text:

```python
df["City"] = (
    df["City"]
    .astype("string")
    .str.strip()
    .str.lower()
)
```

Now:

```text
Mumbai
mumbai
MUMBAI
```

become:

```text
mumbai
```

---

## Remove Extra Whitespace

```python
df["Name"] = (
    df["Name"]
    .astype("string")
    .str.strip()
)
```

---

## Standardize Categories

```python
city_mapping = {
    "bombay": "mumbai",
    "mum": "mumbai"
}

df["City"] = df["City"].replace(city_mapping)
```

---

# 14. Converting Data Types

Excel may store numeric-looking values as text.

Example:

```text
"45000"
"52000"
"61000"
```

Convert them:

```python
df["Income"] = pd.to_numeric(
    df["Income"],
    errors="coerce"
)
```

Invalid values become missing:

```text
"unknown" → NaN
```

---

## Convert Integer-Like Data

```python
df["Age"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
).astype("Int64")
```

`Int64` is Pandas' nullable integer dtype.

---

# 15. Working with Dates

Excel dates can sometimes arrive as strings or timestamps.

Read a date column:

```python
df = pd.read_excel(
    "data.xlsx",
    parse_dates=["Order Date"]
)
```

Or convert after loading:

```python
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    errors="coerce"
)
```

---

## Extract Date Features

```python
df["year"] = df["Order Date"].dt.year
df["month"] = df["Order Date"].dt.month
df["day"] = df["Order Date"].dt.day
df["day_of_week"] = (
    df["Order Date"].dt.dayofweek
)
```

These features can be useful for machine learning.

---

# 16. Handling Duplicate Records

Find duplicate rows:

```python
duplicates = df[
    df.duplicated()
]

print(duplicates)
```

Remove exact duplicates:

```python
df = df.drop_duplicates()
```

Duplicate detection can also use selected columns:

```python
df = df.drop_duplicates(
    subset=["Customer ID"]
)
```

Be careful: duplicate rows are not always errors.

For example, multiple purchases by the same customer may be valid.

---

# 17. Handling Excel Headers

A common Excel problem is a title above the actual table.

Example:

```text
Sales Report - 2026

Customer ID | Amount | City
001         | 500    | Pune
002         | 700    | Mumbai
```

If the actual header is on row 3:

```python
df = pd.read_excel(
    "data.xlsx",
    header=2
)
```

Remember that Pandas uses zero-based indexing.

Therefore:

```text
Excel row 3 → header=2
```

---

## No Header

If the workbook contains no column names:

```python
df = pd.read_excel(
    "data.xlsx",
    header=None
)
```

Assign names:

```python
df.columns = [
    "customer_id",
    "age",
    "income"
]
```

---

# 18. Working with Multiple Sheets

A professional workflow often loads each sheet independently.

```python
import pandas as pd

workbook = pd.read_excel(
    "company_data.xlsx",
    sheet_name=None
)

sales = workbook["Sales"]
customers = workbook["Customers"]
products = workbook["Products"]
```

You can then validate each DataFrame separately.

```python
print(sales.shape)
print(customers.shape)
print(products.shape)
```

---

# 19. Combining Excel Worksheets

Suppose two sheets contain the same structure:

```text
January
February
```

You can concatenate them:

```python
combined = pd.concat(
    [january, february],
    ignore_index=True
)
```

---

## Combining Many Sheets

```python
sheets = pd.read_excel(
    "sales.xlsx",
    sheet_name=None
)

combined = pd.concat(
    sheets.values(),
    ignore_index=True
)
```

This is useful when each worksheet represents:

```text
January
February
March
April
...
```

---

# 20. Reading Multiple Excel Files

Suppose:

```text
data/
├── sales_2024.xlsx
├── sales_2025.xlsx
└── sales_2026.xlsx
```

You can process them programmatically.

```python
from pathlib import Path
import pandas as pd

files = Path("data").glob("*.xlsx")

frames = []

for file in files:
    df = pd.read_excel(file)
    df["source_file"] = file.name
    frames.append(df)

combined = pd.concat(
    frames,
    ignore_index=True
)
```

Adding the source file is useful for:

* debugging
* auditing
* traceability
* data-quality investigation

---

# 21. Writing Data to Excel

Save a DataFrame:

```python
df.to_excel(
    "cleaned_data.xlsx",
    index=False
)
```

The `index=False` argument prevents the Pandas index from becoming an unnecessary Excel column.

---

## Specify the Engine

```python
df.to_excel(
    "cleaned_data.xlsx",
    engine="openpyxl",
    index=False
)
```

---

# 22. Multiple Sheets with Pandas

Use `ExcelWriter`:

```python
import pandas as pd

with pd.ExcelWriter(
    "output.xlsx",
    engine="openpyxl"
) as writer:

    sales.to_excel(
        writer,
        sheet_name="Sales",
        index=False
    )

    customers.to_excel(
        writer,
        sheet_name="Customers",
        index=False
    )
```

Result:

```text
output.xlsx
│
├── Sales
└── Customers
```

---

# 23. Excel Formatting and Data Quality

Formatting can make spreadsheets look clean while hiding data-quality problems.

Examples:

```text
₹50,000
50,000
50000
₹ 50,000
```

These may represent the same numeric value but are not necessarily stored identically.

Machine learning requires **consistent machine-readable values**.

Therefore, do not rely only on visual inspection.

Always inspect:

```python
print(df.dtypes)
print(df.head())
print(df.isna().sum())
```

---

# 24. Merged Cells and Blank Rows

Human-designed spreadsheets often contain:

* merged cells
* title rows
* blank rows
* section headers
* footnotes
* subtotals
* decorative formatting

For example:

```text
-------------------------
      SALES REPORT
-------------------------

Region: Maharashtra

Product | Sales
A       | 100
B       | 200

Total   | 300
```

This is readable for humans but not necessarily structured for machine learning.

The first step is to identify the actual tabular region.

---

## Professional Principle

> **Separate presentation-oriented spreadsheets from analysis-ready tables.**

A report designed for humans may need substantial restructuring before entering an ML pipeline.

---

# 25. Validating Excel Data

Cleaning should not be based only on assumptions.

Create validation rules.

Suppose:

```text
Age
Income
Target
```

Validation:

```python
required_columns = {
    "Age",
    "Income",
    "Target"
}

missing_columns = (
    required_columns - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
)
```

---

## Validate Numeric Ranges

```python
invalid_age = ~df["Age"].between(
    0,
    120
)

if invalid_age.any():
    print("Invalid age values detected.")
```

---

## Validate Missing Target Values

```python
if df["Target"].isna().any():
    raise ValueError(
        "Target contains missing values."
    )
```

---

## Validate Duplicate IDs

```python
if df["Customer ID"].duplicated().any():
    print("Duplicate customer IDs detected.")
```

---

# 26. Data Leakage

Data leakage occurs when information that would not legitimately be available at prediction time enters the model-training process.

Example:

```text
Customer
Age
Income
Purchase
Purchase Confirmation Date
```

Suppose you are predicting whether a customer will purchase.

Using:

```text
Purchase Confirmation Date
```

as a feature could leak information about the outcome.

---

## Another Common Leakage Example

Suppose you calculate:

```python
df["income_mean"] = df["Income"].mean()
```

using the entire dataset before splitting into training and test sets.

The test-set information has influenced the feature transformation.

This can lead to overly optimistic evaluation.

---

## Correct Principle

```text
Raw Data
   ↓
Train/Test Split
   ↓
Fit preprocessing on Training Data
   ↓
Transform Training Data
   ↓
Transform Test Data
   ↓
Train Model
   ↓
Evaluate
```

For production ML, use Scikit-Learn pipelines whenever possible.

---

# 27. Preparing Excel Data for Machine Learning

A typical workflow:

```text
Excel
 ↓
Load
 ↓
Inspect
 ↓
Validate
 ↓
Clean
 ↓
Select Features
 ↓
Separate Target
 ↓
Train/Test Split
 ↓
Preprocessing
 ↓
Model
 ↓
Evaluation
```

Example:

```python
import pandas as pd

df = pd.read_excel(
    "customer_data.xlsx"
)

X = df[
    ["Age", "Income"]
]

y = df["Purchased"]
```

Then:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

# 28. Excel Data Pipeline

A professional project should separate raw data from processed data.

```text
Raw Excel
    ↓
Ingestion
    ↓
Validation
    ↓
Cleaning
    ↓
Transformation
    ↓
Feature Engineering
    ↓
Modeling
```

Example project:

```text
project/
│
├── data/
│   ├── raw/
│   │   └── customers.xlsx
│   │
│   └── processed/
│       └── customers_clean.csv
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── load_data.py
│   ├── validate.py
│   ├── clean.py
│   └── features.py
│
├── models/
│
├── tests/
│
└── README.md
```

---

# 29. Large Excel Files

Excel is not generally the first choice for very large datasets.

Potential problems include:

* high memory usage
* slow parsing
* workbook complexity
* multiple sheets
* formulas
* formatting overhead
* limited scalability

For larger datasets, consider:

```text
CSV
Parquet
SQL databases
Data warehouses
Object storage
```

---

## Better Analytical Formats

For machine-learning workloads, Parquet is often useful because it provides:

* columnar storage
* compression
* efficient analytical reads
* typed columns
* compatibility with modern data tools

Example:

```python
df.to_parquet(
    "customers.parquet"
)
```

---

# 30. Common Excel Problems

## Problem 1: Wrong Header

```text
Report Title
Generated: 2026

Name | Age | Salary
```

Solution:

```python
pd.read_excel(
    "data.xlsx",
    header=2
)
```

---

## Problem 2: Numbers Stored as Text

```text
"50000"
"60000"
```

Solution:

```python
df["Salary"] = pd.to_numeric(
    df["Salary"],
    errors="coerce"
)
```

---

## Problem 3: Inconsistent Categories

```text
Pune
pune
PUNE
 Pune
```

Solution:

```python
df["City"] = (
    df["City"]
    .astype("string")
    .str.strip()
    .str.lower()
)
```

---

## Problem 4: Blank Rows

Remove completely empty rows:

```python
df = df.dropna(
    how="all"
)
```

---

## Problem 5: Duplicate Rows

```python
df = df.drop_duplicates()
```

---

## Problem 6: Unexpected Columns

Always inspect:

```python
print(df.columns.tolist())
```

before building downstream logic.

---

# 31. Security Considerations

Excel files can contain more than ordinary tabular data.

Potential concerns include:

* macros
* external links
* formulas
* malicious workbook content
* sensitive personal information
* hidden worksheets
* hidden rows/columns

Do not automatically trust files from unknown sources.

---

## Sensitive Data

Excel files may contain:

```text
Names
Emails
Phone numbers
Addresses
Financial information
Employee information
Customer records
```

Only collect and process information that is appropriate for your use case.

---

## Spreadsheet Formula Injection

When exporting user-controlled strings back into spreadsheets, values beginning with characters such as:

```text
=
+
-
@
```

can potentially be interpreted as formulas by spreadsheet software.

When generating Excel files from untrusted input, treat user-controlled values carefully.

---

# 32. Recommended Project Structure

A professional Excel-processing project can use:

```text
03-Excel-Data/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   └── sample.xlsx
│   │
│   └── processed/
│       └── cleaned.xlsx
│
├── notebooks/
│   └── excel-eda.ipynb
│
├── scripts/
│   ├── read_excel.py
│   ├── clean_excel.py
│   └── validate_excel.py
│
└── tests/
    └── test_validation.py
```

---

# 33. Practical Exercises

## Exercise 1 — Load Excel

Load:

```text
customers.xlsx
```

and display:

* first 5 rows
* shape
* columns
* data types

---

## Exercise 2 — Missing Values

Find:

```text
missing count
missing percentage
```

for every column.

---

## Exercise 3 — Data Cleaning

Standardize:

```text
City
Gender
Occupation
```

---

## Exercise 4 — Dates

Convert:

```text
Registration Date
```

into a Pandas datetime column.

Create:

```text
year
month
day_of_week
```

---

## Exercise 5 — Multiple Sheets

Create:

```text
customers.xlsx
```

with:

```text
Customers
Orders
Products
```

Load all three worksheets.

---

## Exercise 6 — Combine Monthly Data

Create:

```text
January
February
March
```

worksheets and combine them into one DataFrame.

---

## Exercise 7 — Validation

Create rules for:

```text
Age
Salary
Customer ID
```

and detect invalid records.

---

# 34. Mini Projects

## 🛒 Project 1 — Sales Analysis

Input:

```text
sales.xlsx
```

Analyze:

* total sales
* average order value
* sales by city
* sales by product
* monthly sales
* top customers

---

## 👥 Project 2 — Customer Segmentation

Input:

```text
customers.xlsx
```

Features:

```text
Age
Income
Spending Score
Purchase Frequency
```

Prepare the dataset for clustering.

Possible algorithm:

```text
K-Means
```

---

## 💳 Project 3 — Loan Approval Dataset

Input:

```text
loan_applications.xlsx
```

Features:

```text
Age
Income
Credit Score
Loan Amount
Employment
```

Target:

```text
Loan Approved
```

Prepare the data for classification.

---

## 🏠 Project 4 — House Price Prediction

Input:

```text
housing.xlsx
```

Features:

```text
Area
Bedrooms
Bathrooms
Location
Age
```

Target:

```text
Price
```

Prepare the dataset for regression.

---

## 📊 Project 5 — Multi-Sheet Business Dataset

Create a workbook containing:

```text
Customers
Products
Orders
Employees
```

Build a pipeline that:

1. loads each sheet
2. validates schemas
3. cleans values
4. joins related tables
5. creates analytical features
6. exports a machine-learning-ready dataset

---

# 35. Best Practices

### 1. Keep Raw Data Untouched

Do not overwrite:

```text
raw/customers.xlsx
```

Instead create:

```text
processed/customers_clean.xlsx
```

---

### 2. Validate Before Cleaning

Understand the original dataset before changing it.

---

### 3. Keep Data Types Explicit

Do not assume that Excel's visual formatting represents the actual stored type.

---

### 4. Track Data Provenance

Record:

```text
source file
sheet
load date
transformation version
```

---

### 5. Avoid Manual Cleaning

Prefer reproducible Python code:

```python
df["City"] = (
    df["City"]
    .astype("string")
    .str.strip()
    .str.lower()
)
```

rather than manually editing hundreds of spreadsheet cells.

---

### 6. Separate EDA From Production Pipelines

Exploratory analysis may involve:

```python
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)
```

But a production ML pipeline should learn the median from training data only.

---

### 7. Use Version Control

Store code in Git.

Avoid committing sensitive raw datasets.

---

### 8. Document Assumptions

For example:

```text
Age must be between 0 and 120.
Income must be non-negative.
Customer ID must be unique.
```

---

# 36. Professional Workflow

A reliable Excel-to-ML workflow looks like:

```text
                    ┌─────────────────┐
                    │ Excel Workbook  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Data Ingestion   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Schema Checking  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Data Validation  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Data Cleaning    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Feature Creation │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Train/Test Split │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ ML Preprocessing│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Training   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Evaluation       │
                    └─────────────────┘
```

---

# 37. Learning Roadmap

Follow this progression:

```text
Excel Basics
     ↓
Workbook / Worksheet Structure
     ↓
Pandas read_excel()
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Missing Values
     ↓
Type Conversion
     ↓
Date Processing
     ↓
Multiple Worksheets
     ↓
Multiple Files
     ↓
Data Validation
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Scikit-Learn Pipelines
     ↓
Machine Learning
```

---

# 38. Key Takeaways

After completing this section, you should understand how to:

* read Excel workbooks with Pandas
* load specific worksheets
* load multiple worksheets
* inspect Excel datasets
* select and filter records
* handle missing values
* clean inconsistent categories
* convert data types
* process dates
* remove duplicate records
* handle non-standard headers
* combine worksheets
* process multiple Excel files
* export DataFrames to Excel
* validate data quality
* identify spreadsheet-specific problems
* understand Excel security concerns
* prevent data leakage
* organize Excel ingestion pipelines
* prepare Excel data for machine learning

The key principle is:

> **An Excel workbook is a source of data—not automatically a machine-learning-ready dataset.**

Professional ML workflows turn human-oriented spreadsheet data into **validated, reproducible, structured, and model-ready data**.

---

## 🔗 Connection to the Next Section

The data-collection journey now looks like:

```text
01-Data-Sources
       ↓
02-CSV-Data
       ↓
03-Excel-Data
       ↓
04-JSON-Data
       ↓
05-APIs
       ↓
06-Databases
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Machine Learning
```

The next step is **JSON Data**, where you will learn how to work with nested, semi-structured data commonly returned by APIs and modern applications.

---

## 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

GitHub:
`https://github.com/Kishor055`

---

## 🤝 Contributing

Contributions are welcome.

If you find an issue or have an improvement:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test your examples.
5. Commit your changes.
6. Open a Pull Request.

Example:

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/improve-excel-guide
```

---

## ⭐ Support

If this repository helps you learn Machine Learning, consider giving it a ⭐ on GitHub.

Happy Learning! 🚀

**Python → Data → Features → Models → Machine Learning**

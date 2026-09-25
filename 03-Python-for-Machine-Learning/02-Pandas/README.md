# 🐼 Pandas for Machine Learning

> A complete, practical, and structured guide to **Pandas** for data manipulation, cleaning, exploration, preprocessing, and machine learning workflows in Python.

---

## 📌 Overview

**Pandas** is one of the most important Python libraries for data analysis and machine learning.

It provides powerful data structures and tools for:

* 📊 Working with tabular data
* 🔍 Exploring datasets
* 🧹 Cleaning messy data
* 🔄 Transforming data
* 🧩 Combining multiple datasets
* 📈 Performing statistical analysis
* 🕐 Working with time-series data
* 🤖 Preparing datasets for Machine Learning
* 💾 Reading and writing data in different formats

The two fundamental Pandas data structures are:

* `Series` → One-dimensional labeled data
* `DataFrame` → Two-dimensional labeled tabular data

---

# 🎯 Learning Objectives

By completing this section, you will learn how to:

* Understand Pandas architecture
* Create and manipulate Series
* Create and manipulate DataFrames
* Load datasets from files
* Inspect datasets
* Select rows and columns
* Filter data
* Sort data
* Handle missing values
* Remove duplicates
* Transform columns
* Apply custom functions
* Group and aggregate data
* Merge and join datasets
* Concatenate datasets
* Work with categorical data
* Work with dates and times
* Analyze numerical and categorical features
* Detect and handle outliers
* Prepare datasets for Machine Learning
* Build reproducible data preprocessing pipelines

---

# 📚 Table of Contents

* [What is Pandas?](#-what-is-pandas)
* [Why Pandas for Machine Learning?](#-why-pandas-for-machine-learning)
* [Installation](#-installation)
* [Importing Pandas](#-importing-pandas)
* [Series](#-series)
* [DataFrame](#-dataframe)
* [Creating DataFrames](#-creating-dataframes)
* [Reading Data](#-reading-data)
* [Writing Data](#-writing-data)
* [Dataset Inspection](#-dataset-inspection)
* [Selecting Columns](#-selecting-columns)
* [Selecting Rows](#-selecting-rows)
* [Filtering Data](#-filtering-data)
* [Sorting Data](#-sorting-data)
* [Adding and Removing Columns](#-adding-and-removing-columns)
* [Renaming Columns](#-renaming-columns)
* [Missing Data](#-missing-data)
* [Duplicate Data](#-duplicate-data)
* [Data Type Conversion](#-data-type-conversion)
* [String Operations](#-string-operations)
* [Apply, Map and Applymap](#-apply-map-and-applymap)
* [GroupBy](#-groupby)
* [Aggregation](#-aggregation)
* [Merge](#-merge)
* [Join](#-join)
* [Concatenation](#-concatenation)
* [Pivot Tables](#-pivot-tables)
* [Categorical Data](#-categorical-data)
* [Date and Time](#-date-and-time)
* [Statistical Analysis](#-statistical-analysis)
* [Correlation](#-correlation)
* [Outlier Detection](#-outlier-detection)
* [Feature Engineering](#-feature-engineering)
* [Pandas and Machine Learning](#-pandas-and-machine-learning)
* [Best Practices](#-best-practices)
* [Common Mistakes](#-common-mistakes)
* [Useful Methods](#-useful-methods)
* [Practice Problems](#-practice-problems)
* [Learning Roadmap](#-learning-roadmap)

---

# 🐼 What is Pandas?

Pandas is an open-source Python library designed for:

> **Data manipulation and data analysis.**

It is built on top of NumPy and provides high-level data structures that make working with structured data much easier.

### Example

```python
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [21, 25, 23],
    "Score": [85, 92, 88]
}

df = pd.DataFrame(data)

print(df)
```

Output:

```text
      Name  Age  Score
0    Alice   21     85
1      Bob   25     92
2  Charlie   23     88
```

---

# 🚀 Why Pandas for Machine Learning?

Machine Learning is not only about training models.

A large part of a real-world ML project involves:

```text
Raw Data
   ↓
Load Data
   ↓
Inspect Data
   ↓
Clean Data
   ↓
Transform Data
   ↓
Feature Engineering
   ↓
Exploratory Data Analysis
   ↓
Train/Test Split
   ↓
Machine Learning Model
```

Pandas is heavily used in the first stages of this pipeline.

### Typical ML workflow

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df.describe())

df = df.drop_duplicates()

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Gender"] = df["Gender"].fillna("Unknown")
```

---

# 📦 Installation

Install Pandas using pip:

```bash
pip install pandas
```

Recommended environment:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Verify installation:

```python
import pandas as pd

print(pd.__version__)
```

---

# 📥 Importing Pandas

The standard convention is:

```python
import pandas as pd
```

Then use:

```python
pd.DataFrame()
pd.Series()
pd.read_csv()
```

---

# 📊 Series

A `Series` is a one-dimensional labeled data structure.

```python
import pandas as pd

scores = pd.Series([85, 90, 78, 92])

print(scores)
```

Output:

```text
0    85
1    90
2    78
3    92
dtype: int64
```

---

## Creating a Series

### From a list

```python
numbers = pd.Series([10, 20, 30, 40])
```

### With custom indexes

```python
scores = pd.Series(
    [85, 90, 95],
    index=["Alice", "Bob", "Charlie"]
)
```

### Accessing values

```python
print(scores["Alice"])
```

---

# 🧱 DataFrame

A `DataFrame` is Pandas' primary data structure.

It represents data in rows and columns.

```text
        Name  Age  Salary
0      Alice   22   50000
1        Bob   25   60000
2    Charlie   28   70000
```

Example:

```python
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [22, 25, 28],
    "Salary": [50000, 60000, 70000]
}

df = pd.DataFrame(data)

print(df)
```

---

# 🏗️ Creating DataFrames

## From Dictionary

```python
data = {
    "Name": ["Alice", "Bob"],
    "Age": [22, 25]
}

df = pd.DataFrame(data)
```

## From List of Dictionaries

```python
data = [
    {"Name": "Alice", "Age": 22},
    {"Name": "Bob", "Age": 25}
]

df = pd.DataFrame(data)
```

## From NumPy Array

```python
import numpy as np
import pandas as pd

data = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

df = pd.DataFrame(
    data,
    columns=["A", "B"]
)
```

---

# 📂 Reading Data

Pandas supports many file formats.

---

## CSV

```python
df = pd.read_csv("data.csv")
```

---

## Excel

```python
df = pd.read_excel("data.xlsx")
```

---

## JSON

```python
df = pd.read_json("data.json")
```

---

## SQL

```python
df = pd.read_sql(query, connection)
```

---

# 💾 Writing Data

## CSV

```python
df.to_csv("output.csv", index=False)
```

## Excel

```python
df.to_excel("output.xlsx", index=False)
```

## JSON

```python
df.to_json("output.json")
```

---

# 🔍 Dataset Inspection

Before modifying a dataset, inspect it.

---

## `head()`

Displays the first rows.

```python
df.head()
```

---

## `tail()`

Displays the last rows.

```python
df.tail()
```

---

## `shape`

Returns:

```text
(rows, columns)
```

```python
print(df.shape)
```

---

## `columns`

```python
print(df.columns)
```

---

## `index`

```python
print(df.index)
```

---

## `dtypes`

```python
print(df.dtypes)
```

---

## `info()`

Provides a detailed dataset summary.

```python
df.info()
```

---

## `describe()`

Generates descriptive statistics.

```python
df.describe()
```

For categorical columns:

```python
df.describe(include="object")
```

For all columns:

```python
df.describe(include="all")
```

---

# 🎯 Selecting Columns

## Single Column

```python
df["Age"]
```

Returns a Series.

---

## Multiple Columns

```python
df[["Name", "Age"]]
```

Returns a DataFrame.

---

# 🎯 Selecting Rows

Pandas provides two important indexers:

* `.loc[]`
* `.iloc[]`

---

# `.loc[]`

Used for label-based selection.

```python
df.loc[0]
```

Select specific columns:

```python
df.loc[0, "Name"]
```

Select multiple rows:

```python
df.loc[0:2]
```

Select rows and columns:

```python
df.loc[0:2, ["Name", "Age"]]
```

---

# `.iloc[]`

Used for integer-position-based selection.

```python
df.iloc[0]
```

First row:

```python
df.iloc[0]
```

First three rows:

```python
df.iloc[0:3]
```

First two columns:

```python
df.iloc[:, 0:2]
```

---

# 🔎 Filtering Data

Filtering is one of the most important Pandas operations for Machine Learning.

```python
df[df["Age"] > 25]
```

Multiple conditions:

```python
df[
    (df["Age"] > 25) &
    (df["Salary"] > 60000)
]
```

OR condition:

```python
df[
    (df["Age"] > 25) |
    (df["Salary"] > 60000)
]
```

NOT condition:

```python
df[~(df["Age"] > 25)]
```

---

# 🔍 `isin()`

```python
df[df["City"].isin(["Mumbai", "Pune"])]
```

---

# 🔍 `between()`

```python
df[df["Age"].between(20, 30)]
```

---

# 🔍 `query()`

Pandas provides a readable filtering syntax:

```python
df.query("Age > 25")
```

Multiple conditions:

```python
df.query("Age > 25 and Salary > 60000")
```

---

# 🔃 Sorting Data

Sort by one column:

```python
df.sort_values("Age")
```

Descending:

```python
df.sort_values("Age", ascending=False)
```

Multiple columns:

```python
df.sort_values(
    ["Department", "Salary"],
    ascending=[True, False]
)
```

---

# ➕ Adding Columns

```python
df["Bonus"] = df["Salary"] * 0.10
```

Create a feature:

```python
df["BMI"] = df["Weight"] / (df["Height"] ** 2)
```

---

# ➖ Removing Columns

```python
df.drop("Bonus", axis=1)
```

Multiple columns:

```python
df.drop(
    ["Bonus", "TemporaryColumn"],
    axis=1
)
```

Recommended explicit syntax:

```python
df.drop(
    columns=["Bonus", "TemporaryColumn"]
)
```

---

# ✏️ Renaming Columns

```python
df.rename(
    columns={
        "Name": "Student_Name",
        "Age": "Student_Age"
    }
)
```

Rename all columns:

```python
df.columns = [
    "name",
    "age",
    "salary"
]
```

---

# 🧹 Missing Data

Real-world datasets frequently contain missing values.

Example:

```text
Name      Age     Salary
Alice     22      50000
Bob       NaN     60000
Charlie   25      NaN
```

---

## Detect Missing Values

```python
df.isna()
```

Count missing values:

```python
df.isna().sum()
```

Alternative:

```python
df.isnull().sum()
```

---

# 🗑️ Removing Missing Values

Remove rows containing missing values:

```python
df.dropna()
```

Remove columns:

```python
df.dropna(axis=1)
```

---

# 🩹 Filling Missing Values

Fill with constant:

```python
df["City"] = df["City"].fillna("Unknown")
```

Mean:

```python
df["Age"] = df["Age"].fillna(
    df["Age"].mean()
)
```

Median:

```python
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)
```

Mode:

```python
df["City"] = df["City"].fillna(
    df["City"].mode()[0]
)
```

---

# 🧠 Mean vs Median

For approximately symmetric data:

```text
Mean
```

may be suitable.

For data containing strong outliers:

```text
Median
```

is often more robust.

The appropriate choice depends on the feature and the data-generating process.

---

# ♻️ Duplicate Data

Check duplicates:

```python
df.duplicated()
```

Count:

```python
df.duplicated().sum()
```

Remove duplicates:

```python
df = df.drop_duplicates()
```

---

# 🔄 Data Type Conversion

Check types:

```python
df.dtypes
```

Convert to integer:

```python
df["Age"] = df["Age"].astype(int)
```

Convert to float:

```python
df["Salary"] = df["Salary"].astype(float)
```

Convert to string:

```python
df["Name"] = df["Name"].astype(str)
```

---

# 🧹 Numeric Conversion

Useful when datasets contain invalid numeric values:

```python
df["Salary"] = pd.to_numeric(
    df["Salary"],
    errors="coerce"
)
```

Invalid values become `NaN`.

---

# 🔤 String Operations

Pandas provides vectorized string operations through `.str`.

Convert to lowercase:

```python
df["Name"] = df["Name"].str.lower()
```

Uppercase:

```python
df["Name"].str.upper()
```

Remove whitespace:

```python
df["Name"].str.strip()
```

Contains:

```python
df[df["Name"].str.contains("john", case=False, na=False)]
```

Replace:

```python
df["City"] = df["City"].str.replace(
    "Mumbai",
    "Bombay",
    regex=False
)
```

---

# ⚙️ Apply, Map and Transform

## `map()`

Useful for transforming values in a Series.

```python
gender_map = {
    "M": 0,
    "F": 1
}

df["Gender_Code"] = df["Gender"].map(gender_map)
```

---

## `apply()`

Apply a function to a Series:

```python
df["Age_Group"] = df["Age"].apply(
    lambda age: "Young" if age < 30 else "Adult"
)
```

---

## `apply()` on DataFrame

```python
df["Salary_K"] = df["Salary"].apply(
    lambda salary: salary / 1000
)
```

---

# 📊 GroupBy

`groupby()` is one of the most powerful Pandas operations.

Suppose:

```text
Department   Salary
IT            70000
IT            80000
HR            60000
HR            65000
```

Calculate average salary:

```python
df.groupby("Department")["Salary"].mean()
```

---

## Multiple Aggregations

```python
df.groupby("Department")["Salary"].agg(
    ["mean", "median", "min", "max", "count"]
)
```

---

## GroupBy Multiple Columns

```python
df.groupby(
    ["Department", "City"]
)["Salary"].mean()
```

---

# 🧮 Aggregation

Common aggregation functions:

```python
df["Salary"].mean()
df["Salary"].median()
df["Salary"].sum()
df["Salary"].min()
df["Salary"].max()
df["Salary"].std()
df["Salary"].var()
df["Salary"].count()
```

---

# 🔗 Merge

`merge()` combines DataFrames using common keys.

Example:

```python
students = pd.DataFrame({
    "student_id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"]
})

scores = pd.DataFrame({
    "student_id": [1, 2, 3],
    "score": [85, 90, 95]
})
```

Merge:

```python
result = pd.merge(
    students,
    scores,
    on="student_id"
)
```

---

# 🔀 Merge Types

Pandas supports:

```text
inner
left
right
outer
cross
```

Example:

```python
pd.merge(
    left,
    right,
    on="id",
    how="inner"
)
```

---

## Inner Join

Returns matching records.

```python
pd.merge(
    df1,
    df2,
    on="id",
    how="inner"
)
```

---

## Left Join

Keeps every row from the left DataFrame.

```python
pd.merge(
    df1,
    df2,
    on="id",
    how="left"
)
```

---

## Outer Join

Keeps rows from both DataFrames.

```python
pd.merge(
    df1,
    df2,
    on="id",
    how="outer"
)
```

---

# 🔗 Join

`join()` primarily works with DataFrame indexes.

```python
df1.join(df2)
```

---

# 🧩 Concatenation

Combine DataFrames vertically:

```python
result = pd.concat(
    [df1, df2],
    axis=0,
    ignore_index=True
)
```

Combine horizontally:

```python
result = pd.concat(
    [df1, df2],
    axis=1
)
```

---

# 📐 Pivot Tables

Pivot tables summarize data across multiple dimensions.

```python
pd.pivot_table(
    df,
    values="Salary",
    index="Department",
    columns="Gender",
    aggfunc="mean"
)
```

---

# 🏷️ Categorical Data

Categorical columns contain a limited number of repeated values.

Examples:

```text
Gender
City
Department
Product Category
Education Level
```

Convert to categorical:

```python
df["Department"] = df["Department"].astype("category")
```

Inspect categories:

```python
df["Department"].cat.categories
```

---

# 🕐 Date and Time

Convert a column to datetime:

```python
df["Date"] = pd.to_datetime(df["Date"])
```

Extract year:

```python
df["Year"] = df["Date"].dt.year
```

Month:

```python
df["Month"] = df["Date"].dt.month
```

Day:

```python
df["Day"] = df["Date"].dt.day
```

Day of week:

```python
df["DayOfWeek"] = df["Date"].dt.dayofweek
```

---

# 📈 Statistical Analysis

Pandas provides many statistical operations.

```python
df.mean(numeric_only=True)
df.median(numeric_only=True)
df.std(numeric_only=True)
df.var(numeric_only=True)
df.min(numeric_only=True)
df.max(numeric_only=True)
```

---

## Quantiles

```python
df["Salary"].quantile(0.25)
df["Salary"].quantile(0.50)
df["Salary"].quantile(0.75)
```

---

## Value Counts

For categorical analysis:

```python
df["Department"].value_counts()
```

Normalized:

```python
df["Department"].value_counts(
    normalize=True
)
```

---

# 🔗 Correlation

Correlation measures the strength and direction of a linear relationship between numerical variables.

```python
df.corr(numeric_only=True)
```

Example:

```python
correlation = df[
    ["Age", "Salary", "Experience"]
].corr()

print(correlation)
```

### Important

Correlation does **not** by itself establish causation.

---

# 📦 Outlier Detection

One common approach uses the Interquartile Range (IQR).

```python
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Salary"] < lower_bound) |
    (df["Salary"] > upper_bound)
]
```

---

# 🛠️ Feature Engineering

Feature engineering transforms raw data into useful ML features.

Example:

```python
df["BMI"] = (
    df["Weight"] /
    (df["Height"] ** 2)
)
```

Extract information from dates:

```python
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
```

Create interaction features:

```python
df["Income_Per_Year"] = (
    df["Income"] /
    df["Experience"]
)
```

---

# 🤖 Pandas and Machine Learning

A typical ML workflow looks like:

```text
Dataset
   │
   ▼
Pandas
   │
   ├── Load
   ├── Inspect
   ├── Clean
   ├── Transform
   ├── Feature Engineering
   └── Prepare
        │
        ▼
   NumPy / Scikit-learn
        │
        ▼
   Machine Learning Model
```

---

# 🧪 Example ML Dataset Preparation

```python
import pandas as pd

df = pd.read_csv("customers.csv")

# Inspect
print(df.head())
print(df.info())

# Remove duplicates
df = df.drop_duplicates()

# Missing numerical values
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

# Missing categorical values
df["City"] = df["City"].fillna(
    df["City"].mode()[0]
)

# Convert categorical values
df = pd.get_dummies(
    df,
    columns=["City"],
    drop_first=True
)

print(df.head())
```

---

# ⚠️ Data Leakage

One of the most important concepts in ML preprocessing is **data leakage**.

Data leakage occurs when information from outside the training data improperly influences model training.

For example, calculating a normalization statistic using the entire dataset before splitting into training and test sets can allow information from the test set to influence preprocessing.

### Safer workflow

```text
Raw Dataset
     │
     ▼
Train/Test Split
     │
     ├──────────────┐
     ▼              ▼
Training Data     Test Data
     │
     ▼
Fit preprocessing
     │
     ▼
Transform training
     │
     ▼
Transform test
```

For production ML workflows, `scikit-learn` preprocessing pipelines are often preferable for this reason.

---

# 🔢 One-Hot Encoding

Categorical variables can be converted into indicator columns.

```python
df = pd.get_dummies(
    df,
    columns=["City"],
    drop_first=True
)
```

Example:

```text
City
----
Mumbai
Pune
Mumbai
```

can become:

```text
City_Pune
---------
0
1
0
```

For more complex ML pipelines, `sklearn.preprocessing.OneHotEncoder` is often preferable because it integrates directly with model pipelines.

---

# 📊 Pandas + NumPy

Pandas works closely with NumPy.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6]
})

array = df.to_numpy()

print(array)
```

---

# 🔄 DataFrame ↔ NumPy

DataFrame → NumPy:

```python
array = df.to_numpy()
```

NumPy → DataFrame:

```python
df = pd.DataFrame(array)
```

---

# 🧠 Important Pandas Concepts

| Concept             | Purpose                         |
| ------------------- | ------------------------------- |
| `Series`            | One-dimensional labeled data    |
| `DataFrame`         | Two-dimensional tabular data    |
| `loc`               | Label-based selection           |
| `iloc`              | Position-based selection        |
| `groupby()`         | Group and analyze               |
| `merge()`           | Combine using keys              |
| `concat()`          | Stack/combine objects           |
| `pivot_table()`     | Summarize multidimensional data |
| `fillna()`          | Handle missing values           |
| `dropna()`          | Remove missing values           |
| `drop_duplicates()` | Remove duplicates               |
| `astype()`          | Change data type                |
| `sort_values()`     | Sort records                    |
| `value_counts()`    | Frequency analysis              |
| `describe()`        | Statistical summary             |

---

# 🧪 Complete Example

```python
import pandas as pd

# ---------------------------------------------------------
# 1. Create Dataset
# ---------------------------------------------------------

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "Age": [22, 25, None, 30, 28],
    "Department": [
        "IT",
        "HR",
        "IT",
        "Finance",
        "HR"
    ],
    "Salary": [
        50000,
        60000,
        55000,
        None,
        65000
    ]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# 2. Inspect Dataset
# ---------------------------------------------------------

print(df.head())
print(df.info())
print(df.describe())

# ---------------------------------------------------------
# 3. Missing Values
# ---------------------------------------------------------

print(df.isna().sum())

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Salary"] = df["Salary"].fillna(
    df["Salary"].median()
)

# ---------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------

df["Salary_K"] = df["Salary"] / 1000

# ---------------------------------------------------------
# 5. Filtering
# ---------------------------------------------------------

high_salary = df[
    df["Salary"] > 55000
]

print(high_salary)

# ---------------------------------------------------------
# 6. Sorting
# ---------------------------------------------------------

df = df.sort_values(
    "Salary",
    ascending=False
)

# ---------------------------------------------------------
# 7. GroupBy
# ---------------------------------------------------------

department_salary = (
    df.groupby("Department")["Salary"]
    .mean()
)

print(department_salary)

# ---------------------------------------------------------
# 8. Final Dataset
# ---------------------------------------------------------

print(df)
```

---

# 🧠 Pandas Performance Tips

For larger datasets:

### Prefer vectorized operations

Instead of:

```python
for index, row in df.iterrows():
    ...
```

prefer:

```python
df["Salary_K"] = df["Salary"] / 1000
```

Vectorized operations are generally clearer and faster.

---

### Select only required columns

Instead of loading or processing unnecessary data:

```python
df = df[
    ["Age", "Salary", "Department"]
]
```

---

### Specify data types when appropriate

For CSV files:

```python
df = pd.read_csv(
    "data.csv",
    dtype={
        "Age": "Int64"
    }
)
```

---

### Use chunks for very large CSV files

```python
for chunk in pd.read_csv(
    "large_dataset.csv",
    chunksize=10000
):
    process(chunk)
```

---

# ⚠️ Common Mistakes

## 1. Confusing Series and DataFrame

```python
df["Age"]
```

returns a Series.

```python
df[["Age"]]
```

returns a DataFrame.

---

## 2. Using `inplace=True` everywhere

Instead of relying heavily on:

```python
df.drop(..., inplace=True)
```

explicit assignment can make transformations easier to reason about:

```python
df = df.drop(...)
```

---

## 3. Ignoring Missing Values

Always inspect:

```python
df.isna().sum()
```

before training a model.

---

## 4. Ignoring Data Types

Check:

```python
df.dtypes
```

Incorrect types can cause unexpected behavior.

---

## 5. Modifying Data Without Understanding It

Always inspect the dataset first:

```python
df.head()
df.info()
df.describe()
df.isna().sum()
```

---

## 6. Data Leakage

Do not calculate training-dependent preprocessing statistics from validation/test data.

Use proper train/test splitting and preprocessing pipelines.

---

# 🏆 Best Practices

### 1. Inspect before transforming

```python
df.head()
df.info()
df.describe()
```

### 2. Validate assumptions

Check:

```python
df.dtypes
df.isna().sum()
df.duplicated().sum()
```

### 3. Avoid unnecessary loops

Prefer vectorized Pandas operations.

### 4. Keep preprocessing reproducible

Document transformations.

### 5. Separate raw and processed data

```text
data/
├── raw/
└── processed/
```

### 6. Avoid data leakage

Fit data-dependent preprocessing only on training data.

### 7. Use meaningful column names

Prefer:

```text
customer_age
annual_income
purchase_amount
```

instead of:

```text
a
b
x1
```

---

# 📁 Recommended Project Structure

```text
02-Pandas/
│
├── README.md
│
├── 01-Series/
│   ├── README.md
│   └── series.py
│
├── 02-DataFrame/
│   ├── README.md
│   └── dataframe.py
│
├── 03-Data-Loading/
│   ├── README.md
│   └── data-loading.py
│
├── 04-Data-Inspection/
│   ├── README.md
│   └── inspection.py
│
├── 05-Selection-and-Filtering/
│   ├── README.md
│   └── selection.py
│
├── 06-Data-Cleaning/
│   ├── README.md
│   └── cleaning.py
│
├── 07-GroupBy-and-Aggregation/
│   ├── README.md
│   └── groupby.py
│
├── 08-Merge-and-Join/
│   ├── README.md
│   └── merge.py
│
├── 09-Time-Series/
│   ├── README.md
│   └── datetime.py
│
├── 10-Feature-Engineering/
│   ├── README.md
│   └── feature-engineering.py
│
└── 11-ML-Data-Preprocessing/
    ├── README.md
    └── preprocessing.py
```

---

# 🧪 Practice Problems

## Beginner

1. Create a Pandas Series containing five numbers.
2. Create a DataFrame containing student information.
3. Display the first five rows.
4. Display the last five rows.
5. Find the shape of a dataset.
6. Select a single column.
7. Select multiple columns.
8. Filter rows using a condition.
9. Sort a DataFrame.
10. Rename columns.

---

## Intermediate

11. Find missing values.
12. Replace missing numerical values with the median.
13. Replace missing categorical values with the mode.
14. Remove duplicate rows.
15. Convert column types.
16. Group data by category.
17. Calculate mean, median, minimum, and maximum.
18. Merge two DataFrames.
19. Concatenate multiple DataFrames.
20. Create a pivot table.

---

## Advanced

21. Detect outliers using IQR.
22. Perform feature engineering.
23. Extract date-based features.
24. Analyze categorical distributions.
25. Build a complete preprocessing workflow.
26. Process a large CSV in chunks.
27. Identify possible data leakage.
28. Prepare a dataset for Scikit-learn.
29. Compare raw and processed datasets.
30. Build a reusable data-cleaning pipeline.

---

# 🗺️ Learning Roadmap

```text
Python Basics
     │
     ▼
NumPy
     │
     ▼
Pandas
     │
     ├── Series
     │
     ├── DataFrames
     │
     ├── Data Loading
     │
     ├── Data Inspection
     │
     ├── Selection
     │
     ├── Filtering
     │
     ├── Cleaning
     │
     ├── GroupBy
     │
     ├── Merge / Join
     │
     ├── Time Series
     │
     └── Feature Engineering
             │
             ▼
       Data Visualization
             │
             ▼
       Machine Learning
             │
             ▼
       Model Evaluation
             │
             ▼
       Production ML
```

---

# 📌 Quick Reference

## Create

```python
pd.Series()
pd.DataFrame()
```

## Load

```python
pd.read_csv()
pd.read_excel()
pd.read_json()
```

## Inspect

```python
df.head()
df.tail()
df.info()
df.describe()
df.shape
df.dtypes
```

## Select

```python
df["column"]
df[["column1", "column2"]]
df.loc[]
df.iloc[]
```

## Filter

```python
df[df["column"] > value]
df.query()
df["column"].isin()
```

## Clean

```python
df.isna()
df.fillna()
df.dropna()
df.drop_duplicates()
```

## Transform

```python
df.astype()
df.apply()
df.map()
df.sort_values()
```

## Analyze

```python
df.groupby()
df.value_counts()
df.corr()
df.describe()
```

## Combine

```python
pd.concat()
pd.merge()
df.join()
```

## Save

```python
df.to_csv()
df.to_excel()
df.to_json()
```

---

# 🌍 Real-World Applications

Pandas is widely useful for:

* 🏦 Financial data analysis
* 🛒 Customer analytics
* 🏥 Healthcare datasets
* 📊 Business intelligence
* 🧠 Machine Learning
* 🤖 Artificial Intelligence
* 📈 Time-series analysis
* 🔎 Exploratory Data Analysis
* 🚨 Fraud detection
* 📦 Recommendation systems
* 🏭 Industrial analytics
* 🌐 Web data analysis

---

# 🔗 Pandas in the ML Ecosystem

```text
                    Python
                       │
             ┌─────────┴─────────┐
             │                   │
           NumPy              Pandas
             │                   │
       Numerical Data      Tabular Data
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
                Data Visualization
                       │
                       ▼
                Scikit-learn
                       │
                       ▼
              Machine Learning
                       │
                       ▼
                 Deep Learning
```

---

# 📚 Recommended Learning Order

Follow this order for the strongest foundation:

1. Python fundamentals
2. NumPy
3. Pandas Series
4. Pandas DataFrames
5. Data loading
6. Dataset inspection
7. Indexing and selection
8. Filtering
9. Data cleaning
10. Missing-value handling
11. GroupBy and aggregation
12. Merge and Join
13. Time-series operations
14. Feature engineering
15. Exploratory Data Analysis
16. Scikit-learn preprocessing
17. Machine Learning

---

# 🎓 Key Takeaways

After completing this section, you should be comfortable with:

* Creating Series and DataFrames
* Loading real-world datasets
* Inspecting data
* Selecting rows and columns
* Filtering records
* Sorting data
* Cleaning missing values
* Removing duplicates
* Converting data types
* Transforming columns
* Grouping and aggregating
* Merging datasets
* Working with dates
* Performing statistical analysis
* Detecting outliers
* Creating ML features
* Preparing datasets for Machine Learning
* Avoiding common preprocessing mistakes
* Building reproducible data workflows

---

# 🚀 Next Step

After mastering Pandas, continue with:

```text
Pandas
   ↓
Data Cleaning
   ↓
Data Visualization
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Scikit-learn
   ↓
Machine Learning
```

> **Pandas is the bridge between raw data and machine learning. Mastering data manipulation is one of the most valuable skills in an ML workflow.**

---

## 🧑‍💻 Author

**Kishor Patil**

Machine Learning • Python • Data Science • AI

---

## ⭐ Contributing

Contributions are welcome!

If you find an issue or have an improvement:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Commit your changes
5. Push the branch
6. Open a Pull Request

---

## 📄 License

This project is intended for educational and learning purposes.

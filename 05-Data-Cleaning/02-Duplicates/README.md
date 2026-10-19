# 🧹 Duplicate Data Handling in Python

**Duplicate Handling = Detect → Investigate → Validate → Remove or Retain → Verify → Maintain Data Quality**

Duplicate handling is an essential data-cleaning technique used to identify and manage repeated records in a dataset. Removing unwanted duplicates can improve data quality, reporting accuracy, and the reliability of machine learning workflows.

---

## 🌱 GROW → INSPECT → DETECT → CLEAN → VALIDATE → BUILD 🚀

> **Goal:** Learn how to identify duplicate rows and values, remove unwanted duplicates using Python and Pandas, preserve legitimate repeated records, and build reliable data-cleaning pipelines.

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [What Is Duplicate Data?](#-what-is-duplicate-data)
3. [Why Duplicate Handling Matters](#-why-duplicate-handling-matters)
4. [Types of Duplicates](#-types-of-duplicates)
5. [Create a Sample Dataset](#-create-a-sample-dataset)
6. [Inspect a Dataset](#-inspect-a-dataset)
7. [Detect Exact Duplicate Rows](#-detect-exact-duplicate-rows)
8. [Count Duplicate Records](#-count-duplicate-records)
9. [Detect Duplicates in Selected Columns](#-detect-duplicates-in-selected-columns)
10. [Keep First, Last, or All Duplicates](#-keep-first-last-or-all-duplicates)
11. [Remove Duplicate Rows](#-remove-duplicate-rows)
12. [Remove Duplicates Using Business Keys](#-remove-duplicates-using-business-keys)
13. [Handle Case and Whitespace Differences](#-handle-case-and-whitespace-differences)
14. [Detect Near-Duplicates](#-detect-near-duplicates)
15. [Duplicate Handling with NumPy](#-duplicate-handling-with-numpy)
16. [Duplicate Handling with SQL](#-duplicate-handling-with-sql)
17. [Duplicate Data and Machine Learning](#-duplicate-data-and-machine-learning)
18. [Avoid Data Leakage](#-avoid-data-leakage)
19. [Complete Data-Cleaning Example](#-complete-data-cleaning-example)
20. [Common Mistakes](#-common-mistakes)
21. [Best Practices](#-best-practices)
22. [Mini Projects](#-mini-projects)
23. [Practice Exercises](#-practice-exercises)
24. [Quick Revision](#-quick-revision)
25. [Project Structure](#-project-structure)
26. [Learning Roadmap](#-learning-roadmap)
27. [Key Takeaways](#-key-takeaways)
28. [What's Next](#-whats-next)

---

## 📌 Introduction

Real-world datasets often contain repeated records because of:

- Repeated form submissions.
- Multiple imports of the same file.
- Database synchronization issues.
- Incorrect data-entry procedures.
- Repeated API responses.
- Duplicate customer or student registrations.
- Event logging and system retries.

For example, a student dataset may contain the same student record twice. If both records are counted in a report, the number of students may be overstated.

However, **not every repeated row is an unwanted duplicate**. A customer can legitimately make several purchases, and a student can attend multiple classes.

The correct approach is to identify duplicates using the dataset's meaning and business rules before removing anything.

---

## 🎯 What Is Duplicate Data?

Duplicate data refers to records or values that repeat according to a defined comparison rule.

Consider the following dataset:

| Student ID | Name | Course |
|---|---|---|
| 101 | Amit | Python |
| 102 | Priya | Java |
| 101 | Amit | Python |
| 103 | Rahul | SQL |
| 102 | Priya | Java |

The rows for student IDs `101` and `102` are repeated exactly.

If each student should have only one record in this dataset, the repeated rows may be removed.

### Before cleaning

```text
Total rows: 5
Unique student records: 3
Duplicate rows after first occurrence: 2
```

### After cleaning

```text
Total rows: 3
Unique student records: 3
Duplicate rows after first occurrence: 0
```

This example assumes that each student should have one record for the specified course. A dataset containing course enrollments or attendance records may need a different uniqueness rule.

---

## 💡 Why Duplicate Handling Matters

### 1. Improves data quality

Removing confirmed unwanted duplicate records produces a more consistent dataset.

### 2. Prevents inflated counts

Duplicate records can make totals, averages, frequencies, and reports misleading.

### 3. Improves machine learning reliability

Repeated observations can unintentionally give certain examples more influence during training.

### 4. Reduces unnecessary storage

Removing redundant records can reduce storage and processing requirements.

### 5. Supports reliable analysis

Clean, well-defined records make exploratory data analysis easier to interpret.

**Important:** Duplicate removal is not automatically beneficial. Repeated observations may represent legitimate events, measurements, or transactions.

---

## 🔍 Types of Duplicates

### 1. Exact duplicates

Every compared column has the same value.

```text
101, Amit, Python
101, Amit, Python
```

### 2. Partial duplicates

Only selected columns match.

```text
101, Amit, Python
101, Amit, Java
```

Whether these are duplicates depends on the intended uniqueness rule.

### 3. Key-based duplicates

Two or more rows share a business key, such as a student ID, customer ID, or transaction ID.

```text
Student ID: 101
Student ID: 101
```

The rows may differ in other columns, so the correct record must be identified using an explicit rule.

### 4. Formatting duplicates

The same logical value appears in different formats.

```text
Kishor Patil
kishor patil
 Kishor Patil
```

These can often be identified after normalization of case and whitespace.

### 5. Near-duplicates

Two records are similar but not identical.

```text
Rahul Sharma
Rahul Sahrma
```

Detecting these requires additional rules, such as similarity matching or human review.

---

## 🧪 Create a Sample Dataset

Install Pandas if needed:

```bash
python -m pip install pandas numpy
```

Create a sample DataFrame containing exact duplicates and repeated student IDs.

```python
import pandas as pd


def main():
    data = {
        "student_id": [101, 102, 101, 103, 104, 102, 105],
        "name": [
            "Amit",
            "Priya",
            "Amit",
            "Rahul",
            "Sneha",
            "Priya",
            "Kiran",
        ],
        "course": [
            "Python",
            "Java",
            "Python",
            "SQL",
            "Python",
            "Java",
            "Machine Learning",
        ],
        "marks": [85, 90, 85, 78, 92, 90, 88],
    }

    df = pd.DataFrame(data)

    print("Original dataset:")
    print(df)

    print("\nShape:", df.shape)
    print("\nData types:")
    print(df.dtypes)


if __name__ == "__main__":
    main()
```

This dataset contains repeated complete rows for students `101` and `102`.

---

## 🔎 Inspect a Dataset

Before removing duplicates, inspect the structure and understand what each row represents.

```python
import pandas as pd

df = pd.read_csv("students.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())
```

Useful methods:

| Method | Purpose |
|---|---|
| `df.head()` | View the first rows |
| `df.shape` | Get rows and columns |
| `df.info()` | Inspect types and non-null counts |
| `df.dtypes` | Inspect column data types |
| `df.isna().sum()` | Count missing values |
| `df.describe()` | Summarize numerical columns |

Inspecting the dataset helps determine whether repeated rows are legitimate records or data-quality problems.

---

## 🔍 Detect Exact Duplicate Rows

Pandas provides `duplicated()` to identify repeated rows.

```python
import pandas as pd

df = pd.DataFrame(
    {
        "student_id": [101, 102, 101, 103, 102],
        "name": ["Amit", "Priya", "Amit", "Rahul", "Priya"],
        "course": ["Python", "Java", "Python", "SQL", "Java"],
    }
)

duplicate_mask = df.duplicated()

print(duplicate_mask)
print("\nDuplicate rows:")
print(df.loc[duplicate_mask])
```

By default, `duplicated()` marks later occurrences as duplicates and leaves the first occurrence unmarked.

### Display all rows that belong to duplicate groups

To display both the original occurrence and subsequent duplicates:

```python
duplicate_groups = df.duplicated(keep=False)

print(df.loc[duplicate_groups])
```

This is useful when reviewing the records before deciding what to remove.

### Visualize the logic

```text
Record A → First occurrence → Not marked
Record B → First occurrence → Not marked
Record A → Repeated record  → Marked
Record B → Repeated record  → Marked
```

The meaning of “duplicate” depends on which columns are compared.

---

## 🔢 Count Duplicate Records

Count the number of duplicate rows after the first occurrence:

```python
duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)
```

Count all rows that belong to duplicate groups, including their first occurrence:

```python
all_duplicate_rows = df.duplicated(keep=False).sum()

print("Rows in duplicate groups:", all_duplicate_rows)
```

These numbers can differ.

For example, if a record appears three times:

- `duplicated().sum()` counts two later occurrences.
- `duplicated(keep=False).sum()` counts all three rows.

### Calculate duplicate percentage

```python
total_rows = len(df)
duplicate_count = int(df.duplicated().sum())

duplicate_percentage = (
    duplicate_count / total_rows * 100
    if total_rows
    else 0.0
)

print(f"Duplicate rows: {duplicate_count}")
print(f"Duplicate percentage: {duplicate_percentage:.2f}%")
```

The duplicate percentage is meaningful only in relation to the chosen comparison rule.

---

## 🎯 Detect Duplicates in Selected Columns

Sometimes the complete row differs, but a subset of columns defines uniqueness.

For example, suppose student IDs should be unique.

```python
import pandas as pd

df = pd.DataFrame(
    {
        "student_id": [101, 102, 101, 103],
        "name": ["Amit", "Priya", "Amit Kumar", "Rahul"],
        "course": ["Python", "Java", "SQL", "Python"],
    }
)

duplicates = df.duplicated(
    subset=["student_id"],
    keep=False,
)

print(df.loc[duplicates])
```

Here, rows sharing a student ID are flagged even if their names or courses differ.

### Compare multiple columns

```python
duplicates = df.duplicated(
    subset=["name", "course"],
    keep=False,
)

print(df.loc[duplicates])
```

This flags rows that match on both `name` and `course`.

**Important:** A student ID may be unique for a student master table but repeated in an enrollment table. Always define the uniqueness rule according to the table's purpose.

---

## 🔁 Keep First, Last, or All Duplicates

The `keep` parameter controls how Pandas marks duplicate occurrences.

### Keep the first occurrence

```python
df.duplicated(keep="first")
```

The first occurrence is not marked; later matching occurrences are marked.

### Keep the last occurrence

```python
df.duplicated(keep="last")
```

All earlier matching occurrences are marked; the last is not.

### Mark every duplicate occurrence

```python
df.duplicated(keep=False)
```

Every row belonging to a duplicate group is marked.

### Comparison

| `keep` value | Behavior |
|---|---|
| `"first"` | Keeps the first occurrence unmarked |
| `"last"` | Keeps the last occurrence unmarked |
| `False` | Marks all occurrences |

These settings control detection. To actually remove rows, use `drop_duplicates()`.

---

## 🧹 Remove Duplicate Rows

Pandas provides `drop_duplicates()`.

### Remove exact duplicates

```python
df_clean = df.drop_duplicates()

print("Before:", df.shape)
print("After:", df_clean.shape)
```

The original DataFrame remains unchanged unless you assign the result back to it.

### Remove duplicates and keep the last occurrence

```python
df_clean = df.drop_duplicates(keep="last")
```

### Remove rows based on selected columns

```python
df_clean = df.drop_duplicates(
    subset=["student_id"],
    keep="first",
)
```

This keeps the first row for each student ID according to the current row order.

### Remove every row belonging to a duplicate group

If the goal is to retain only records that occur exactly once under the chosen comparison rule:

```python
duplicate_mask = df.duplicated(keep=False)
df_unique_only = df.loc[~duplicate_mask].copy()
```

This removes every row in a duplicate group rather than keeping one representative.

### Verify the results

```python
print("Rows before cleaning:", len(df))
print("Rows after cleaning:", len(df_clean))
print("Remaining exact duplicates:", df_clean.duplicated().sum())
```

Do not assume that removing exact duplicates also removes duplicates based on a particular ID. Verify using the same subset that defines uniqueness.

---

## 🗝️ Remove Duplicates Using Business Keys

A business key identifies a record according to the rules of the system.

Examples:

- Student master: `student_id`
- Customer master: `customer_id`
- Order table: `order_id`
- Product catalog: `product_id`
- Event table: a combination of `event_id` and timestamp

### Example

```python
import pandas as pd

df = pd.DataFrame(
    {
        "student_id": [101, 102, 101, 103],
        "name": ["Amit", "Priya", "Amit Kumar", "Rahul"],
        "updated_at": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-05",
            "2026-01-03",
        ],
    }
)

df["updated_at"] = pd.to_datetime(
    df["updated_at"],
    errors="coerce",
)

df_sorted = df.sort_values(
    "updated_at",
    na_position="first",
)

df_latest = df_sorted.drop_duplicates(
    subset=["student_id"],
    keep="last",
)

print(df_latest)
```

This example keeps the most recent dated record for each student ID because rows are sorted chronologically first.

Before using this approach, verify:

- The key actually identifies the entity.
- The timestamp represents the correct update time.
- Missing timestamps are handled according to an explicit policy.
- Conflicting records are reviewed when necessary.

**Warning:** Keeping the latest row is not always correct. For transaction and event tables, multiple rows with the same customer ID may represent legitimate independent events.

---

## 🔤 Handle Case and Whitespace Differences

These strings may refer to the same logical name:

```text
"Kishor Patil"
" kishor patil "
"KISHOR PATIL"
```

Exact string comparison treats them as different values.

Normalize text before comparing, but retain the original data if formatting itself matters.

```python
import pandas as pd

df = pd.DataFrame(
    {
        "name": [
            "Kishor Patil",
            " kishor patil ",
            "KISHOR PATIL",
            "Amit Sharma",
        ]
    }
)

df["name_normalized"] = (
    df["name"]
    .astype("string")
    .str.strip()
    .str.casefold()
)

print(df)
```

Detect repeated normalized names:

```python
duplicate_names = df.duplicated(
    subset=["name_normalized"],
    keep=False,
)

print(df.loc[duplicate_names])
```

### Remove repeated normalized names

```python
df_unique_names = df.drop_duplicates(
    subset=["name_normalized"],
    keep="first",
)
```

This is appropriate only if normalized name is a suitable uniqueness criterion. Two different people may have the same name, so names alone are often insufficient for identity matching.

---

## 🧠 Detect Near-Duplicates

Near-duplicates are records that appear similar but are not identical.

Example:

```text
Rahul Sharma
Rahul Sahrma
Rahul Sharma.
```

They may arise from spelling errors, inconsistent punctuation, or formatting differences.

### Common approaches

- Normalize case and whitespace.
- Standardize punctuation where appropriate.
- Compare normalized email addresses or phone numbers.
- Use domain-specific identifiers.
- Apply string similarity methods.
- Use record-linkage or entity-resolution tools.
- Review uncertain matches manually.

### Simple normalization example

```python
import pandas as pd

df = pd.DataFrame(
    {
        "name": [
            "Rahul Sharma",
            "rahul sharma ",
            "Rahul Sharma.",
            "Priya Patil",
        ]
    }
)

df["normalized_name"] = (
    df["name"]
    .astype("string")
    .str.strip()
    .str.casefold()
    .str.rstrip(".")
)

print(df)
print("\nPotential repeated names:")

mask = df.duplicated(
    subset=["normalized_name"],
    keep=False,
)

print(df.loc[mask])
```

This handles simple formatting differences, not general spelling mistakes.

For fuzzy matching, choose a similarity threshold based on the domain, then validate the results. Automatically merging records on name similarity alone can combine distinct people.

---

## 🔢 Duplicate Handling with NumPy

For simple numerical arrays, NumPy provides `unique()`.

### Find unique values

```python
import numpy as np

values = np.array([10, 20, 10, 30, 20, 40])

unique_values = np.unique(values)

print("Original:", values)
print("Unique values:", unique_values)
```

### Get unique values and counts

```python
import numpy as np

values = np.array([10, 20, 10, 30, 20, 40])

unique_values, counts = np.unique(
    values,
    return_counts=True,
)

print("Values:", unique_values)
print("Counts:", counts)
```

### Identify repeated values

```python
import numpy as np

values = np.array([10, 20, 10, 30, 20, 40])

unique_values, counts = np.unique(
    values,
    return_counts=True,
)

repeated_values = unique_values[counts > 1]

print("Repeated values:", repeated_values)
```

NumPy's `unique()` is useful for array-level analysis. For tabular records with named columns and business keys, Pandas is generally more convenient.

---

## 🗄️ Duplicate Handling with SQL

SQL can identify and manage duplicate records in relational databases.

### Find repeated email addresses

```sql
SELECT
    email,
    COUNT(*) AS occurrence_count
FROM customers
GROUP BY email
HAVING COUNT(*) > 1;
```

### Retrieve rows with duplicate emails

```sql
SELECT *
FROM customers
WHERE email IN (
    SELECT email
    FROM customers
    GROUP BY email
    HAVING COUNT(*) > 1
);
```

The query above assumes `email` is the field used to identify repeated records. Null values and database-specific behavior should be considered separately.

### Identify duplicates using a window function

```sql
SELECT *
FROM (
    SELECT
        customers.*,
        ROW_NUMBER() OVER (
            PARTITION BY email
            ORDER BY created_at, customer_id
        ) AS row_num
    FROM customers
) AS ranked
WHERE row_num > 1;
```

This identifies rows beyond the first according to the selected ordering.

### Prevent duplicates with a unique constraint

When a value must be unique, enforcing the rule at the database level is usually safer than relying on repeated cleanup.

```sql
ALTER TABLE customers
ADD CONSTRAINT unique_customer_email UNIQUE (email);
```

Use this only if email is genuinely required to be unique under your business rules. Syntax and null-handling details vary by database system.

---

## 🤖 Duplicate Data and Machine Learning

Duplicates can affect model training and evaluation.

### 1. Repeated observations can change training influence

If the same observation appears multiple times, it may receive more weight in the training process.

### 2. Duplicates can inflate evaluation scores

If identical or near-identical examples occur in both training and test sets, the test score may not represent performance on genuinely unseen data.

### 3. Repeated entities can cause leakage

If several rows belong to the same student, customer, patient, or device, a random row-level split may place records from the same entity into both training and test sets.

### 4. Duplicate targets may indicate conflicts

The same feature record may have different target labels. This could indicate a data error, an omitted feature, a changing target, or a legitimate ambiguity.

Investigate the cause rather than deleting conflicting rows automatically.

---

## 🚨 Avoid Data Leakage

Duplicate handling should be planned alongside dataset splitting.

For a supervised learning project, a safe workflow is often:

1. Define record identity and remove confirmed accidental duplicates using legitimate data-quality rules.
2. Separate the target from input features.
3. Split the data according to the deployment scenario.
4. Use group-aware splitting when records from the same entity must not cross splits.
5. Fit learned preprocessing steps using training data only.
6. Evaluate on genuinely unseen data.

### Example: Group-aware splitting

```python
from sklearn.model_selection import GroupShuffleSplit

# X: feature DataFrame
# y: target Series
# groups: entity ID for each row, such as student_id

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]
```

`GroupShuffleSplit` prevents the same group from being assigned to both sets in this split.

For time-dependent data, use a chronological split when that best represents how the model will be deployed.

**Important:** Do not blindly deduplicate across an entire dataset if repeated events are legitimate or if the cleaning decision would use information unavailable at prediction time.

---

## 🧪 Complete Data-Cleaning Example

This example detects exact duplicate rows, reports the results, removes duplicates, and saves the cleaned dataset.

```python
from pathlib import Path

import pandas as pd


def clean_duplicates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Remove exact duplicate rows while preserving the first occurrence.

    Returns:
        A cleaned copy and a summary of row counts.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    rows_before = len(df)
    duplicate_count = int(df.duplicated().sum())

    cleaned_df = df.drop_duplicates().copy()

    summary = {
        "rows_before": rows_before,
        "duplicate_rows_removed": duplicate_count,
        "rows_after": len(cleaned_df),
    }

    return cleaned_df, summary


def main() -> None:
    data = {
        "student_id": [101, 102, 101, 103, 104, 102],
        "name": [
            "Amit",
            "Priya",
            "Amit",
            "Rahul",
            "Sneha",
            "Priya",
        ],
        "course": [
            "Python",
            "Java",
            "Python",
            "SQL",
            "Python",
            "Java",
        ],
        "marks": [85, 90, 85, 78, 92, 90],
    }

    df = pd.DataFrame(data)

    print("Original dataset:")
    print(df)

    cleaned_df, summary = clean_duplicates(df)

    print("\nCleaning summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nCleaned dataset:")
    print(cleaned_df)

    # Save the cleaned data to a separate file.
    output_path = Path("students_cleaned.csv")
    cleaned_df.to_csv(output_path, index=False)

    print(f"\nSaved cleaned dataset to: {output_path.resolve()}")

    # Verify exact duplicates have been removed.
    assert cleaned_df.duplicated().sum() == 0


if __name__ == "__main__":
    main()
```

### What this program does

- Builds a sample student dataset.
- Counts exact duplicate rows.
- Removes repeated complete rows while keeping the first occurrence.
- Reports the number of rows before and after cleaning.
- Saves the cleaned dataset to a new CSV file.
- Verifies that no exact duplicate rows remain.

This example removes exact duplicate rows only. It does not enforce unique student IDs or remove records with repeated names.

---

## ❌ Common Mistakes

1. Removing every repeated value without checking its meaning.
2. Confusing repeated values in one column with duplicate records.
3. Removing records based on a non-unique field such as a person's name.
4. Keeping the first or last record without sorting by a meaningful timestamp.
5. Ignoring formatting differences that hide logical duplicates.
6. Automatically merging near-duplicates without validating matches.
7. Forgetting to compare records using the correct business key.
8. Modifying the original data without preserving an audit trail.
9. Failing to verify the number of records removed.
10. Removing legitimate repeated transactions or events.
11. Ignoring duplicate or near-duplicate observations across training and test sets.
12. Assuming `drop_duplicates()` guarantees that every business key is unique.

---

## ✅ Best Practices

- Understand the meaning of each row before cleaning.
- Define uniqueness rules explicitly.
- Prefer stable business keys when available.
- Inspect duplicate groups before deleting conflicting records.
- Normalize text carefully before comparing it.
- Preserve the original dataset when auditability matters.
- Record how many rows were removed and why.
- Verify the output using the same duplicate definition.
- Add database constraints for mandatory uniqueness rules.
- Use group-aware or time-aware splitting when the data requires it.
- Monitor duplicate rates in recurring data pipelines.
- Review near-duplicate matches instead of blindly merging records.

---

## 🚀 Mini Projects

### Project 1: Student Dataset Cleaning

**Objective:** Identify and remove accidental duplicate student records.

Tasks:

- Create a CSV file with student IDs, names, courses, and marks.
- Add exact duplicate rows.
- Detect duplicate rows with Pandas.
- Remove confirmed exact duplicates.
- Compare dataset shapes before and after cleaning.
- Export the cleaned dataset.

### Project 2: Customer Data Deduplication

**Objective:** Identify repeated customer records.

Tasks:

- Create a dataset containing customer IDs, names, and email addresses.
- Introduce case and whitespace differences.
- Normalize email addresses according to a defined policy.
- Identify possible duplicates.
- Review conflicting customer records before deciding which to retain.

### Project 3: E-Commerce Transactions

**Objective:** Distinguish accidental duplicates from legitimate repeat purchases.

Tasks:

- Create an order dataset with order IDs, customer IDs, product IDs, and timestamps.
- Introduce repeated order records.
- Identify a suitable uniqueness key.
- Keep legitimate multiple purchases.
- Explain why customer ID alone is not a suitable key for a transaction table.

### Project 4: Machine Learning Leakage Investigation

**Objective:** Investigate the effect of repeated examples on evaluation.

Tasks:

- Create a dataset containing repeated observations.
- Compare a random row-level split with an appropriate group-aware split.
- Check whether duplicate examples cross the train/test boundary.
- Train a suitable baseline model.
- Explain why an evaluation score can be misleading when related records appear in both splits.

---

## 📝 Practice Exercises

1. What is duplicate data?
2. What is the difference between exact duplicates and partial duplicates?
3. How does `DataFrame.duplicated()` work?
4. What is the difference between `keep="first"` and `keep="last"`?
5. How can you identify all rows belonging to duplicate groups?
6. How do you remove duplicates based on selected columns?
7. Why is a business key important?
8. How can whitespace and case differences hide duplicates?
9. What is the difference between deduplication and fuzzy matching?
10. Why might duplicate observations cause machine learning leakage?
11. When should repeated records be retained?
12. How can a SQL unique constraint prevent duplicate entries?

---

## ⚡ Quick Revision

| Method | Purpose |
|---|---|
| `df.duplicated()` | Identify repeated rows |
| `df.duplicated(keep=False)` | Mark all rows in duplicate groups |
| `df.duplicated(subset=[...])` | Detect duplicates using selected columns |
| `df.drop_duplicates()` | Remove repeated rows |
| `df.drop_duplicates(subset=[...])` | Remove duplicates according to selected columns |
| `df.shape` | Inspect dataset dimensions |
| `np.unique()` | Find unique array values |
| SQL `GROUP BY` | Identify repeated key values |
| SQL `ROW_NUMBER()` | Rank rows within duplicate groups |
| Unique constraint | Enforce a database uniqueness rule |

---

## 📁 Project Structure

```text
05-Data-Cleaning/
│
├── README.md
│
├── 01-Missing-Values/
│   └── README.md
│
└── 02-Duplicates/
    ├── README.md
    ├── detect_duplicates.py
    ├── remove_duplicates.py
    ├── subset_duplicates.py
    ├── normalize_text.py
    ├── duplicate_analysis.py
    └── clean_dataset.py
```

---

## 🗺️ Learning Roadmap

1. Inspect the dataset.
2. Understand exact and partial duplicates.
3. Detect repeated records with Pandas.
4. Count duplicate rows.
5. Choose the correct uniqueness key.
6. Remove confirmed accidental duplicates.
7. Normalize text before comparing values.
8. Investigate near-duplicates.
9. Understand duplicate-related machine learning leakage.
10. Automate and validate the cleaning workflow.

---

## 🎯 Key Takeaways

- Duplicate handling is a fundamental part of data cleaning.
- Pandas provides `duplicated()` and `drop_duplicates()` for common tasks.
- The correct duplicate definition depends on the dataset's purpose.
- Business keys are usually more reliable than names alone.
- Formatting normalization can reveal repeated logical values.
- Near-duplicate detection requires additional validation.
- Repeated events are not necessarily errors.
- Duplicate records can distort reports and machine learning evaluation.
- Always verify and document the cleaning results.

---

## 🔜 What's Next?

Continue your data-cleaning journey with **Data Type Conversion**, where you will learn how to correct column types, convert strings into numeric values, parse dates, and handle invalid conversions.

Explore the complete repository: [Machine Learning — GitHub](https://github.com/Kishor055/Machine-Learning)

---

## 👨‍💻 Author

**Kishor Patil**

GitHub: [Kishor055](https://github.com/Kishor055)

## 🤝 Contributing

Contributions are welcome! Improve the examples, correct mistakes, add exercises, or suggest new data-cleaning techniques.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Submit a pull request with a clear description.

## ⭐ Support

If this repository helps you learn Python and Machine Learning, consider giving it a star on GitHub.

## 📄 License

Refer to the repository's root `LICENSE` file for licensing terms.

---

**Clean data. Reliable analysis. Better machine learning. 🚀**

# 🗄️ SQL Data for Machine Learning

SQL (**Structured Query Language**) is one of the most important skills for working with real-world machine learning data.

While CSV, Excel, and JSON files are common data sources, production data is frequently stored in **relational databases**.

Examples include:

* Customer databases
* Banking systems
* E-commerce platforms
* Healthcare systems
* Inventory systems
* Education platforms
* Financial applications
* SaaS products
* Enterprise data warehouses
* Analytics platforms

Machine learning engineers often spend significant time **querying, joining, filtering, aggregating, validating, and extracting data from databases** before a model is ever trained.

> **Goal:** Learn how to use SQL to discover, retrieve, validate, transform, aggregate, and prepare relational data for machine learning workflows.

---

## 📚 Table of Contents

* [1. What Is SQL?](#1-what-is-sql)
* [2. Why SQL Matters in Machine Learning](#2-why-sql-matters-in-machine-learning)
* [3. What Is a Relational Database?](#3-what-is-a-relational-database)
* [4. Database Terminology](#4-database-terminology)
* [5. Tables, Rows, and Columns](#5-tables-rows-and-columns)
* [6. Primary Keys](#6-primary-keys)
* [7. Foreign Keys](#7-foreign-keys)
* [8. Relationships Between Tables](#8-relationships-between-tables)
* [9. SQL vs CSV vs Excel vs JSON](#9-sql-vs-csv-vs-excel-vs-json)
* [10. SQL Command Categories](#10-sql-command-categories)
* [11. Creating a Database](#11-creating-a-database)
* [12. Creating Tables](#12-creating-tables)
* [13. SQL Data Types](#13-sql-data-types)
* [14. Inserting Data](#14-inserting-data)
* [15. Selecting Data](#15-selecting-data)
* [16. Filtering with WHERE](#16-filtering-with-where)
* [17. Comparison Operators](#17-comparison-operators)
* [18. Logical Operators](#18-logical-operators)
* [19. Sorting Data](#19-sorting-data)
* [20. Limiting Results](#20-limiting-results)
* [21. Handling NULL](#21-handling-null)
* [22. DISTINCT](#22-distinct)
* [23. Aggregate Functions](#23-aggregate-functions)
* [24. GROUP BY](#24-group-by)
* [25. HAVING](#25-having)
* [26. SQL Joins](#26-sql-joins)
* [27. INNER JOIN](#27-inner-join)
* [28. LEFT JOIN](#28-left-join)
* [29. RIGHT and FULL OUTER JOIN](#29-right-and-full-outer-join)
* [30. SELF JOIN](#30-self-join)
* [31. CROSS JOIN](#31-cross-join)
* [32. Subqueries](#32-subqueries)
* [33. Common Table Expressions](#33-common-table-expressions)
* [34. CASE Expressions](#34-case-expressions)
* [35. String Functions](#35-string-functions)
* [36. Date and Time Operations](#36-date-and-time-operations)
* [37. Window Functions](#37-window-functions)
* [38. SQL for Feature Engineering](#38-sql-for-feature-engineering)
* [39. Avoiding Data Leakage](#39-avoiding-data-leakage)
* [40. Data Validation with SQL](#40-data-validation-with-sql)
* [41. SQL and Python](#41-sql-and-python)
* [42. SQLite with Python](#42-sqlite-with-python)
* [43. Pandas and SQL](#43-pandas-and-sql)
* [44. Parameterized Queries](#44-parameterized-queries)
* [45. Extracting ML Datasets](#45-extracting-ml-datasets)
* [46. Large Datasets](#46-large-datasets)
* [47. Database Security](#47-database-security)
* [48. Recommended Project Structure](#48-recommended-project-structure)
* [49. Practical Exercises](#49-practical-exercises)
* [50. Mini Projects](#50-mini-projects)
* [51. Common Mistakes](#51-common-mistakes)
* [52. Best Practices](#52-best-practices)
* [53. Professional ML Data Workflow](#53-professional-ml-data-workflow)
* [54. Learning Roadmap](#54-learning-roadmap)
* [55. Key Takeaways](#55-key-takeaways)

---

# 1. What Is SQL?

SQL stands for:

> **Structured Query Language**

SQL is used to interact with relational databases.

Typical operations include:

```text
Create data
Read data
Update data
Delete data
```

These are commonly known as:

> **CRUD**

```text
Create
Read
Update
Delete
```

For machine learning, the most common operation is usually **reading and transforming data**.

Example:

```sql
SELECT
    age,
    income,
    city
FROM customers;
```

---

# 2. Why SQL Matters in Machine Learning

Machine learning datasets are often distributed across multiple database tables.

For example:

```text
customers
orders
products
payments
```

A model may need information from all of them.

A typical workflow:

```text
Database
   ↓
SQL Query
   ↓
Join Tables
   ↓
Filter Records
   ↓
Aggregate Data
   ↓
Create Features
   ↓
Python / Pandas
   ↓
ML Pipeline
   ↓
Model
```

SQL therefore acts as a bridge between:

```text
Raw Enterprise Data
        ↓
Machine Learning Dataset
```

---

# 3. What Is a Relational Database?

A relational database stores data in **tables**.

Example:

```text
Customers
┌────┬─────────┬─────┬─────────┐
│ id │ name    │ age │ city    │
├────┼─────────┼─────┼─────────┤
│ 1  │ Alice   │ 25  │ Pune    │
│ 2  │ Bob     │ 31  │ Mumbai  │
└────┴─────────┴─────┴─────────┘
```

Another table:

```text
Orders
┌────┬─────────────┬────────┐
│ id │ customer_id │ amount │
├────┼─────────────┼────────┤
│ 10 │ 1           │ 500    │
│ 11 │ 2           │ 800    │
└────┴─────────────┴────────┘
```

The relationship is:

```text
Customers.id
      ↓
Orders.customer_id
```

---

# 4. Database Terminology

| Term        | Meaning                             |
| ----------- | ----------------------------------- |
| Database    | Collection of organized data        |
| Table       | Structured collection of records    |
| Row         | One record                          |
| Column      | One attribute                       |
| Primary Key | Unique identifier                   |
| Foreign Key | Reference to another table          |
| Query       | Request for data                    |
| Schema      | Structure of database objects       |
| Index       | Structure used to speed up lookups  |
| View        | Saved query-like database object    |
| Transaction | Logical unit of database operations |

---

# 5. Tables, Rows, and Columns

Consider:

```text
customers
```

| customer_id | name    | age | income |
| ----------: | ------- | --: | -----: |
|         101 | Alice   |  25 |  50000 |
|         102 | Bob     |  31 |  70000 |
|         103 | Charlie |  28 |  60000 |

### Table

```text
customers
```

### Columns

```text
customer_id
name
age
income
```

### Row

```text
101 | Alice | 25 | 50000
```

A machine learning dataset often maps naturally to:

```text
Rows    → observations
Columns → features / target / identifiers
```

---

# 6. Primary Keys

A primary key uniquely identifies each row.

Example:

```sql
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    income REAL
);
```

Here:

```text
customer_id
```

is the primary key.

A primary key should identify one record uniquely.

---

# 7. Foreign Keys

A foreign key connects one table to another.

Example:

```sql
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount REAL,
    FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
);
```

Relationship:

```text
customers
    │
    │ customer_id
    ↓
orders
```

Foreign keys help preserve relational integrity.

---

# 8. Relationships Between Tables

Common relationships include:

## One-to-One

```text
User
 ↓
Profile
```

One user has one profile.

---

## One-to-Many

```text
Customer
   ↓
Orders
```

One customer can have many orders.

---

## Many-to-Many

```text
Students
   ↕
Courses
```

A student can take many courses, and a course can contain many students.

This is usually represented through an intermediate table.

```text
student_courses
```

---

# 9. SQL vs CSV vs Excel vs JSON

| Feature           | SQL      | CSV      | Excel    | JSON                   |
| ----------------- | -------- | -------- | -------- | ---------------------- |
| Relational data   | ⭐⭐⭐      | Limited  | Limited  | Limited                |
| Multiple tables   | ✅        | ❌        | Sheets   | Nested structures      |
| Querying          | ✅        | Limited  | Limited  | Limited                |
| Large datasets    | ✅        | ⚠️       | ❌        | ⚠️                     |
| Concurrent access | ✅        | ❌        | Limited  | ❌                      |
| Transactions      | ✅        | ❌        | ❌        | ❌                      |
| APIs              | Indirect | Indirect | Indirect | ⭐⭐⭐                    |
| ML extraction     | ⭐⭐⭐      | ⭐⭐       | ⭐⭐       | ⭐⭐                     |
| Data integrity    | Strong   | Low      | Medium   | Depends on application |

A useful rule:

```text
CSV / Excel
    ↓
Small, portable datasets

JSON
    ↓
Nested / API-oriented data

SQL
    ↓
Relational, persistent, multi-user data
```

---

# 10. SQL Command Categories

SQL commands can be grouped into several categories.

## DDL — Data Definition Language

Defines database structures.

```sql
CREATE
ALTER
DROP
TRUNCATE
```

---

## DML — Data Manipulation Language

Changes data.

```sql
INSERT
UPDATE
DELETE
```

---

## DQL — Data Query Language

Retrieves data.

```sql
SELECT
```

---

## DCL — Data Control Language

Controls permissions.

```sql
GRANT
REVOKE
```

---

## TCL — Transaction Control Language

Controls transactions.

```sql
COMMIT
ROLLBACK
SAVEPOINT
```

---

# 11. Creating a Database

Different database systems use different approaches.

For learning, SQLite is convenient because it is included with Python.

Create a SQLite database:

```python
import sqlite3

connection = sqlite3.connect(
    "ml_data.db"
)

connection.close()
```

This creates:

```text
ml_data.db
```

---

# 12. Creating Tables

SQL:

```sql
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    income REAL,
    city TEXT
);
```

Execute it with Python:

```python
import sqlite3

connection = sqlite3.connect(
    "ml_data.db"
)

connection.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        income REAL,
        city TEXT
    )
""")

connection.commit()
connection.close()
```

---

# 13. SQL Data Types

Common SQL data types include:

```text
INTEGER
REAL
DECIMAL
TEXT
BOOLEAN
DATE
TIMESTAMP
BLOB
```

The exact behavior depends on the database system.

For example:

```sql
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    salary REAL,
    active BOOLEAN
);
```

---

# 14. Inserting Data

Insert one record:

```sql
INSERT INTO customers (
    customer_id,
    name,
    age,
    income,
    city
)
VALUES (
    101,
    'Alice',
    25,
    50000,
    'Pune'
);
```

Insert multiple records:

```sql
INSERT INTO customers (
    customer_id,
    name,
    age,
    income,
    city
)
VALUES
    (102, 'Bob', 31, 70000, 'Mumbai'),
    (103, 'Charlie', 28, 60000, 'Nashik');
```

---

# 15. Selecting Data

Select everything:

```sql
SELECT *
FROM customers;
```

Prefer selecting only required columns in production queries:

```sql
SELECT
    customer_id,
    age,
    income
FROM customers;
```

This makes the query:

* clearer
* more efficient
* easier to maintain

---

# 16. Filtering with WHERE

Select customers older than 30:

```sql
SELECT *
FROM customers
WHERE age > 30;
```

Income filter:

```sql
SELECT
    name,
    income
FROM customers
WHERE income >= 60000;
```

---

# 17. Comparison Operators

Common operators:

```text
=
!=
<>
>
<
>=
<=
```

Examples:

```sql
SELECT *
FROM customers
WHERE age = 25;
```

```sql
SELECT *
FROM customers
WHERE income >= 50000;
```

---

# 18. Logical Operators

## AND

```sql
SELECT *
FROM customers
WHERE age > 25
  AND income > 50000;
```

## OR

```sql
SELECT *
FROM customers
WHERE city = 'Pune'
   OR city = 'Mumbai';
```

## NOT

```sql
SELECT *
FROM customers
WHERE NOT city = 'Pune';
```

---

## IN

Instead of:

```sql
WHERE city = 'Pune'
   OR city = 'Mumbai'
```

use:

```sql
WHERE city IN (
    'Pune',
    'Mumbai'
);
```

---

## BETWEEN

```sql
SELECT *
FROM customers
WHERE income BETWEEN 50000 AND 80000;
```

---

## LIKE

```sql
SELECT *
FROM customers
WHERE name LIKE 'A%';
```

This finds names beginning with `A`.

---

# 19. Sorting Data

Sort ascending:

```sql
SELECT *
FROM customers
ORDER BY income ASC;
```

Descending:

```sql
SELECT *
FROM customers
ORDER BY income DESC;
```

Multiple columns:

```sql
SELECT *
FROM customers
ORDER BY city ASC, income DESC;
```

---

# 20. Limiting Results

Return only 10 records:

```sql
SELECT *
FROM customers
LIMIT 10;
```

This is useful for exploration.

For example:

```sql
SELECT *
FROM customers
ORDER BY income DESC
LIMIT 10;
```

This retrieves the highest-income records.

---

# 21. Handling NULL

SQL uses:

```text
NULL
```

to represent missing/unknown values.

Incorrect:

```sql
WHERE income = NULL
```

Correct:

```sql
WHERE income IS NULL;
```

Not null:

```sql
WHERE income IS NOT NULL;
```

---

## Replace NULL with a Value

Many SQL systems support:

```sql
COALESCE(income, 0)
```

Example:

```sql
SELECT
    name,
    COALESCE(income, 0) AS income
FROM customers;
```

Be careful: replacing missing values with `0` is a modeling decision, not merely a formatting operation.

---

# 22. DISTINCT

Find unique cities:

```sql
SELECT DISTINCT city
FROM customers;
```

Count unique cities:

```sql
SELECT COUNT(DISTINCT city)
FROM customers;
```

This is useful during exploratory data analysis.

---

# 23. Aggregate Functions

Common functions:

```text
COUNT()
SUM()
AVG()
MIN()
MAX()
```

Example:

```sql
SELECT COUNT(*)
FROM customers;
```

Average income:

```sql
SELECT AVG(income)
FROM customers;
```

Maximum income:

```sql
SELECT MAX(income)
FROM customers;
```

Total income:

```sql
SELECT SUM(income)
FROM customers;
```

---

# 24. GROUP BY

Suppose you want average income by city:

```sql
SELECT
    city,
    AVG(income) AS average_income
FROM customers
GROUP BY city;
```

Result:

| city   | average_income |
| ------ | -------------: |
| Mumbai |          70000 |
| Nashik |          60000 |
| Pune   |          50000 |

---

## Count Customers by City

```sql
SELECT
    city,
    COUNT(*) AS customer_count
FROM customers
GROUP BY city;
```

---

# 25. HAVING

`WHERE` filters rows before aggregation.

`HAVING` filters groups after aggregation.

Example:

```sql
SELECT
    city,
    AVG(income) AS average_income
FROM customers
GROUP BY city
HAVING AVG(income) > 60000;
```

Conceptually:

```text
WHERE
 ↓
GROUP BY
 ↓
HAVING
```

---

# 26. SQL Joins

Real-world ML data is rarely contained in one table.

Consider:

```text
customers
orders
products
```

A join combines related records.

```text
customers
     +
orders
     ↓
combined dataset
```

---

# 27. INNER JOIN

Return only matching records.

```sql
SELECT
    customers.customer_id,
    customers.name,
    orders.order_id,
    orders.amount
FROM customers
INNER JOIN orders
    ON customers.customer_id =
       orders.customer_id;
```

Conceptually:

```text
Customers ∩ Orders
```

Only records with matching keys appear.

---

# 28. LEFT JOIN

Keep all customers even if they have no orders.

```sql
SELECT
    customers.customer_id,
    customers.name,
    orders.order_id,
    orders.amount
FROM customers
LEFT JOIN orders
    ON customers.customer_id =
       orders.customer_id;
```

This is especially useful when constructing customer-level ML datasets.

For example:

```text
Customer without order
        ↓
order_count = 0
```

may be a meaningful feature.

---

# 29. RIGHT and FULL OUTER JOIN

Some database systems support:

```sql
RIGHT JOIN
```

and:

```sql
FULL OUTER JOIN
```

A `RIGHT JOIN` keeps all records from the right table.

A `FULL OUTER JOIN` keeps matching and non-matching records from both tables.

Support varies by database system, so always check the SQL dialect being used.

---

# 30. SELF JOIN

A table can be joined to itself.

Example:

```text
employees
```

contains:

```text
employee_id
name
manager_id
```

Query:

```sql
SELECT
    employee.name AS employee_name,
    manager.name AS manager_name
FROM employees AS employee
LEFT JOIN employees AS manager
    ON employee.manager_id =
       manager.employee_id;
```

This is useful for hierarchical data.

---

# 31. CROSS JOIN

A cross join creates combinations between two tables.

```sql
SELECT *
FROM products
CROSS JOIN regions;
```

If:

```text
products = 10 rows
regions  = 5 rows
```

the result can contain:

```text
10 × 5 = 50 rows
```

Use cross joins carefully because the result can grow rapidly.

---

# 32. Subqueries

A subquery is a query inside another query.

Example:

```sql
SELECT *
FROM customers
WHERE income > (
    SELECT AVG(income)
    FROM customers
);
```

This returns customers whose income is above the overall average.

---

# 33. Common Table Expressions

CTEs improve query readability.

Example:

```sql
WITH average_income AS (
    SELECT AVG(income) AS value
    FROM customers
)

SELECT
    customer_id,
    name,
    income
FROM customers
WHERE income > (
    SELECT value
    FROM average_income
);
```

CTEs are useful for complex feature-building queries.

---

# 34. CASE Expressions

`CASE` creates conditional values.

Example:

```sql
SELECT
    name,
    income,
    CASE
        WHEN income < 50000 THEN 'Low'
        WHEN income < 100000 THEN 'Medium'
        ELSE 'High'
    END AS income_group
FROM customers;
```

Result:

| name    | income | income_group |
| ------- | -----: | ------------ |
| Alice   |  40000 | Low          |
| Bob     |  75000 | Medium       |
| Charlie | 150000 | High         |

This can be useful for feature engineering.

---

# 35. String Functions

Common SQL string functions include:

```text
LOWER()
UPPER()
TRIM()
LENGTH()
SUBSTR()
REPLACE()
```

Example:

```sql
SELECT
    LOWER(name) AS normalized_name
FROM customers;
```

Trim whitespace:

```sql
SELECT
    TRIM(name) AS clean_name
FROM customers;
```

Exact function names and behavior can vary by database engine.

---

# 36. Date and Time Operations

Date operations vary significantly across SQL dialects.

Common concepts include:

```text
date extraction
date differences
date truncation
timestamp filtering
interval calculations
```

Example in a dialect that supports `EXTRACT`:

```sql
SELECT
    EXTRACT(YEAR FROM order_date) AS order_year
FROM orders;
```

In SQLite, date functions have different syntax:

```sql
SELECT
    strftime('%Y', order_date) AS order_year
FROM orders;
```

Always write SQL according to the database engine being used.

---

# 37. Window Functions

Window functions calculate values across related rows without collapsing them into groups.

Example:

```sql
SELECT
    customer_id,
    order_id,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id
    ) AS customer_total
FROM orders;
```

Each order remains a row, but the customer's total is added as a feature.

---

## Ranking

```sql
SELECT
    customer_id,
    amount,
    ROW_NUMBER() OVER (
        PARTITION BY customer_id
        ORDER BY amount DESC
    ) AS order_rank
FROM orders;
```

This can identify:

```text
largest order
second-largest order
third-largest order
```

---

## Running Total

```sql
SELECT
    order_date,
    amount,
    SUM(amount) OVER (
        ORDER BY order_date
    ) AS running_total
FROM orders;
```

Window functions are extremely useful for feature engineering and time-dependent datasets.

---

# 38. SQL for Feature Engineering

Suppose:

```text
customers
orders
```

You want customer-level features:

```text
total_orders
total_spending
average_order_value
maximum_order_value
```

SQL:

```sql
SELECT
    c.customer_id,
    COUNT(o.order_id) AS total_orders,
    COALESCE(SUM(o.amount), 0) AS total_spending,
    COALESCE(AVG(o.amount), 0) AS average_order_value,
    COALESCE(MAX(o.amount), 0) AS maximum_order_value
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
GROUP BY c.customer_id;
```

This produces a feature table:

| customer_id | total_orders | total_spending | average_order_value | maximum_order_value |
| ----------: | -----------: | -------------: | ------------------: | ------------------: |
|         101 |            5 |          25000 |                5000 |                9000 |
|         102 |            2 |           7000 |                3500 |                4000 |

This table can then be passed into a machine learning pipeline.

---

# 39. Avoiding Data Leakage

SQL can accidentally create leakage.

Suppose the target is:

```text
customer_churn
```

and you create features using data that occurred **after** the churn event.

For example:

```text
last_activity_after_churn
```

could reveal the outcome.

---

## Time-Aware Feature Engineering

A safer design uses a prediction timestamp:

```text
prediction_time
```

and only allows information available before that timestamp.

Conceptually:

```text
Historical Data
       │
       │ information available
       ↓
Prediction Time
       │
       ↓
Target Outcome
```

Never allow future information to flow backward into training features.

---

# 40. Data Validation with SQL

SQL is excellent for data-quality checks.

## Count Rows

```sql
SELECT COUNT(*)
FROM customers;
```

---

## Find NULL Values

```sql
SELECT COUNT(*)
FROM customers
WHERE age IS NULL;
```

---

## Find Invalid Ages

```sql
SELECT *
FROM customers
WHERE age < 0
   OR age > 120;
```

---

## Find Duplicate IDs

```sql
SELECT
    customer_id,
    COUNT(*) AS count
FROM customers
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

---

## Check Negative Income

```sql
SELECT *
FROM customers
WHERE income < 0;
```

These checks can become automated data-quality tests.

---

# 41. SQL and Python

Python can connect to many databases.

Common options include:

```text
SQLite
PostgreSQL
MySQL
SQL Server
Oracle
Cloud warehouses
```

Popular Python libraries include:

```text
sqlite3
SQLAlchemy
psycopg
mysql-connector-python
```

The exact library depends on the database.

---

# 42. SQLite with Python

SQLite is excellent for learning because Python includes:

```python
import sqlite3
```

Create a connection:

```python
import sqlite3

connection = sqlite3.connect(
    "ml_data.db"
)
```

Execute a query:

```python
cursor = connection.execute("""
    SELECT *
    FROM customers
""")

rows = cursor.fetchall()

for row in rows:
    print(row)
```

Close the connection:

```python
connection.close()
```

---

## Context Manager

Prefer:

```python
import sqlite3

with sqlite3.connect("ml_data.db") as connection:
    rows = connection.execute("""
        SELECT *
        FROM customers
    """).fetchall()
```

This makes connection handling cleaner.

---

# 43. Pandas and SQL

Pandas can read SQL query results directly.

```python
import pandas as pd
import sqlite3

with sqlite3.connect("ml_data.db") as connection:

    df = pd.read_sql_query(
        """
        SELECT
            customer_id,
            age,
            income,
            city
        FROM customers
        WHERE age >= 18
        """,
        connection
    )

print(df.head())
```

This is a common bridge between:

```text
SQL
 ↓
Pandas
 ↓
Machine Learning
```

---

# 44. Parameterized Queries

Never construct SQL queries by directly concatenating untrusted user input.

Avoid:

```python
query = (
    "SELECT * FROM customers "
    "WHERE city = '" + city + "'"
)
```

Use parameters instead.

SQLite example:

```python
query = """
    SELECT *
    FROM customers
    WHERE city = ?
"""

with sqlite3.connect("ml_data.db") as connection:
    df = pd.read_sql_query(
        query,
        connection,
        params=(city,)
    )
```

Parameterized queries help protect against SQL injection.

---

# 45. Extracting ML Datasets

A production ML dataset often comes from a carefully designed SQL query.

Example:

```sql
SELECT
    c.customer_id,
    c.age,
    c.income,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(o.amount), 0) AS total_spending,
    COALESCE(AVG(o.amount), 0) AS average_order_value
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id,
    c.age,
    c.income;
```

Python:

```python
import pandas as pd
import sqlite3

query = """
SELECT
    c.customer_id,
    c.age,
    c.income,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(o.amount), 0) AS total_spending,
    COALESCE(AVG(o.amount), 0) AS average_order_value
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id,
    c.age,
    c.income
"""

with sqlite3.connect("ml_data.db") as connection:
    features = pd.read_sql_query(
        query,
        connection
    )
```

Now:

```text
features
    ↓
Train/Test Split
    ↓
Preprocessing
    ↓
Model
```

---

# 46. Large Datasets

For large datasets, avoid blindly loading everything into memory.

Instead of:

```python
df = pd.read_sql_query(
    "SELECT * FROM huge_table",
    connection
)
```

consider selecting only required columns:

```sql
SELECT
    customer_id,
    age,
    income
FROM customers;
```

and filtering at the database level:

```sql
SELECT
    customer_id,
    age,
    income
FROM customers
WHERE age >= 18;
```

This is generally better than retrieving unnecessary rows and columns and filtering afterward.

---

## Chunked SQL Reads

Pandas can read query results in chunks:

```python
import pandas as pd

chunks = pd.read_sql_query(
    query,
    connection,
    chunksize=10000
)

for chunk in chunks:
    process(chunk)
```

This supports memory-conscious processing.

---

# 47. Database Security

Database access should follow security best practices.

### Never commit:

```text
database passwords
API keys
connection strings
private certificates
access tokens
```

---

## Use Environment Variables

Example:

```python
import os

database_url = os.environ[
    "DATABASE_URL"
]
```

For local development, a `.env` file may be used with appropriate tooling, but secrets should not be committed to version control.

---

## Principle of Least Privilege

An ML data pipeline should receive only the database permissions it actually needs.

For example:

```text
Read-only access
```

may be sufficient for a training-data extraction job.

Avoid giving unnecessary:

```text
DELETE
DROP
ALTER
UPDATE
```

permissions.

---

# 48. Recommended Project Structure

A professional SQL-based ML project can use:

```text
05-SQL-Data/
│
├── README.md
│
├── data/
│   └── sample/
│
├── sql/
│   ├── schema.sql
│   ├── seed.sql
│   ├── validation.sql
│   └── features.sql
│
├── src/
│   ├── database.py
│   ├── extract.py
│   ├── validate.py
│   └── features.py
│
├── notebooks/
│   └── sql-eda.ipynb
│
├── tests/
│   ├── test_queries.py
│   └── test_validation.py
│
├── .env.example
│
└── README.md
```

---

# 49. Practical Exercises

## Exercise 1 — Create a Database

Create:

```text
ml_data.db
```

with:

```text
customers
orders
products
```

---

## Exercise 2 — Basic Queries

Write SQL queries to find:

* all customers
* customers older than 30
* customers from Pune
* customers with income above 50000
* the 10 highest-income customers

---

## Exercise 3 — Aggregation

Calculate:

* total customers
* average age
* average income
* maximum income
* minimum income

---

## Exercise 4 — GROUP BY

Calculate:

```text
customers by city
average income by city
total orders by customer
total sales by product
```

---

## Exercise 5 — JOIN

Join:

```text
customers
orders
```

using:

```text
customer_id
```

---

## Exercise 6 — Feature Engineering

Create:

```text
total_orders
total_spending
average_order_value
maximum_order_value
```

for every customer.

---

## Exercise 7 — Data Quality

Find:

* duplicate customer IDs
* missing income
* invalid ages
* negative order amounts
* orders without matching customers

---

## Exercise 8 — Python Integration

Load the feature query into Pandas:

```python
pd.read_sql_query()
```

Then inspect:

```python
df.info()
df.describe()
df.isna().sum()
```

---

# 50. Mini Projects

## 🛒 Project 1 — E-Commerce Customer Analytics

Create:

```text
customers
products
orders
order_items
```

Generate:

```text
customer_id
order_count
total_spending
average_order_value
unique_products
```

Use the resulting dataset for customer segmentation.

---

## 💳 Project 2 — Loan Risk Dataset

Create tables:

```text
customers
loans
payments
```

Build a query that creates:

```text
age
income
loan_amount
payment_count
late_payment_count
total_paid
outstanding_amount
```

Prepare the resulting dataset for classification.

---

## 🏥 Project 3 — Healthcare Analytics

Create:

```text
patients
visits
lab_results
```

Build a dataset containing:

```text
patient_age
visit_count
average_lab_value
last_visit_date
```

Use appropriate privacy protections when working with real data.

---

## 📦 Project 4 — Inventory Prediction

Tables:

```text
products
inventory
sales
```

Create features such as:

```text
total_sales
average_daily_sales
current_inventory
sales_last_7_days
sales_last_30_days
```

Prepare the data for demand forecasting.

---

## 🚨 Project 5 — Fraud Detection Dataset

Tables:

```text
customers
transactions
devices
locations
```

Create transaction-level features:

```text
transaction_amount
customer_transaction_count
average_transaction_amount
device_transaction_count
location_transaction_count
```

Then prepare the dataset for anomaly detection or classification.

---

# 51. Common Mistakes

## ❌ Mistake 1 — `SELECT *` Everywhere

Avoid:

```sql
SELECT *
FROM huge_table;
```

when only a few columns are needed.

Prefer:

```sql
SELECT
    customer_id,
    income,
    age
FROM customers;
```

---

## ❌ Mistake 2 — Filtering After Downloading Everything

Avoid loading millions of rows into Pandas just to filter them.

Prefer:

```sql
WHERE
```

inside the database query.

---

## ❌ Mistake 3 — Incorrect Joins

A wrong join can silently duplicate rows.

Always verify:

```text
row count before join
row count after join
key uniqueness
relationship cardinality
```

---

## ❌ Mistake 4 — Ignoring NULL

SQL `NULL` is not equivalent to:

```text
0
""
"Unknown"
```

Treat missingness according to the meaning of the data.

---

## ❌ Mistake 5 — Leakage Through Historical Aggregations

A feature such as:

```text
customer_total_spending
```

can leak future information if it includes transactions occurring after the prediction timestamp.

---

## ❌ Mistake 6 — SQL Injection

Never concatenate untrusted input into SQL.

Use parameterized queries.

---

## ❌ Mistake 7 — Assuming SQL Dialects Are Identical

SQLite, PostgreSQL, MySQL, SQL Server, and other systems differ in:

* data types
* date functions
* string functions
* syntax
* window-function support
* JSON functions

Always verify the target database dialect.

---

# 52. Best Practices

### 1. Push Filtering to the Database

Prefer:

```sql
SELECT *
FROM customers
WHERE age >= 18;
```

over downloading everything and filtering afterward.

---

### 2. Select Only Required Columns

Prefer:

```sql
SELECT
    age,
    income,
    city
FROM customers;
```

instead of:

```sql
SELECT *
FROM customers;
```

when possible.

---

### 3. Validate Join Cardinality

Know whether your join is:

```text
one-to-one
one-to-many
many-to-one
many-to-many
```

before joining.

---

### 4. Make Queries Reproducible

Store important queries in:

```text
sql/
```

rather than hiding them inside notebooks.

---

### 5. Separate Extraction From Modeling

Use:

```text
SQL
 ↓
Dataset
 ↓
ML Pipeline
```

instead of mixing database operations throughout model-training code.

---

### 6. Version Control SQL

SQL is code.

Store important SQL scripts in Git.

---

### 7. Document Feature Definitions

For example:

```text
total_spending:
Sum of all completed transactions
available before prediction_time.
```

This is much better than simply naming a column:

```text
total_spending
```

without defining how it was calculated.

---

### 8. Test Data Queries

Test:

* expected row counts
* null rates
* duplicate keys
* valid ranges
* join behavior
* time boundaries

---

# 53. Professional ML Data Workflow

A production-oriented workflow looks like:

```text
                 ┌──────────────────┐
                 │ Relational DB    │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ SQL Extraction   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Data Validation  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Joins / Aggreg.  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Feature Dataset  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Time/Data Split  │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ ML Preprocessing │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Model Training   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │ Evaluation       │
                 └──────────────────┘
```

---

# 54. Learning Roadmap

Follow this progression:

```text
SQL Fundamentals
       ↓
Database Concepts
       ↓
Tables / Keys / Relationships
       ↓
SELECT
       ↓
WHERE
       ↓
ORDER BY
       ↓
GROUP BY
       ↓
HAVING
       ↓
Aggregate Functions
       ↓
JOINs
       ↓
Subqueries
       ↓
CTEs
       ↓
CASE
       ↓
Window Functions
       ↓
Data Validation
       ↓
Feature Engineering
       ↓
Python + SQL
       ↓
Pandas + SQL
       ↓
ML Dataset Extraction
       ↓
Production Data Pipelines
```

---

# 55. Key Takeaways

After completing this section, you should understand how to:

* understand relational databases
* work with tables, rows, and columns
* understand primary keys
* understand foreign keys
* model table relationships
* write basic SQL queries
* filter records
* sort data
* limit results
* handle `NULL`
* use `DISTINCT`
* calculate aggregates
* use `GROUP BY`
* use `HAVING`
* perform SQL joins
* use subqueries
* use CTEs
* use `CASE`
* work with dates
* use window functions
* create ML features with SQL
* validate database data
* identify data leakage
* connect Python to SQL databases
* use SQLite
* load SQL data into Pandas
* use parameterized queries
* process large datasets efficiently
* protect database credentials
* design reproducible SQL extraction pipelines
* build ML-ready datasets from relational data

The most important principle is:

> **Machine learning begins long before model training.**

A model can only learn from the data you provide to it. SQL gives you the ability to retrieve the **right records, from the right tables, using the right historical information**, and transform them into meaningful features without unnecessarily moving massive amounts of raw data into Python.

---

## 🔗 Connection to the Next Sections

Your data-collection progression now becomes:

```text
01-Data-Sources
        ↓
02-CSV-Data
        ↓
03-Excel-Data
        ↓
04-JSON-Data
        ↓
05-SQL-Data
        ↓
06-APIs / Data Collection
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

You have now moved from **file-based data** to **relational data systems**.

The next stage can focus on collecting data programmatically from **APIs**, where SQL and JSON concepts come together in real-world data pipelines.

---

## 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

GitHub:

```text
https://github.com/Kishor055
```

---

## 🤝 Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Improve the documentation or examples.
4. Test all SQL and Python examples.
5. Commit your changes.
6. Open a Pull Request.

Example:

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/improve-sql-guide
```

---

## ⭐ Support

If this repository helps you learn Machine Learning, consider giving it a ⭐ on GitHub.

Happy Learning! 🚀

**SQL → Data → Features → Models → Machine Learning**

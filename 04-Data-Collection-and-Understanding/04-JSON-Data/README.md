# 🧾 JSON Data for Machine Learning

JSON (**JavaScript Object Notation**) is one of the most widely used formats for exchanging and storing structured and semi-structured data.

Unlike CSV and traditional spreadsheets, JSON can represent **nested objects, arrays, hierarchical relationships, and variable structures**. This makes it especially important for machine learning and data engineering because JSON is commonly returned by:

* REST APIs
* Web applications
* Cloud services
* Databases
* IoT systems
* Mobile applications
* Social platforms
* SaaS applications
* Machine learning services
* External data providers

This section teaches how to work with JSON professionally—from reading simple JSON files to handling nested objects, arrays, JSON Lines, normalization, validation, API responses, and preparing JSON data for machine learning.

> **Goal:** Learn how to reliably load, inspect, parse, flatten, validate, transform, and prepare JSON data for downstream data analysis and machine learning workflows.

---

## 📚 Table of Contents

* [1. What Is JSON?](#1-what-is-json)
* [2. Why JSON Matters in Machine Learning](#2-why-json-matters-in-machine-learning)
* [3. JSON Data Types](#3-json-data-types)
* [4. JSON Objects](#4-json-objects)
* [5. JSON Arrays](#5-json-arrays)
* [6. Nested JSON](#6-nested-json)
* [7. JSON vs CSV vs Excel](#7-json-vs-csv-vs-excel)
* [8. JSON File Structure](#8-json-file-structure)
* [9. Python JSON Module](#9-python-json-module)
* [10. Reading JSON Files](#10-reading-json-files)
* [11. Writing JSON Files](#11-writing-json-files)
* [12. JSON Serialization and Deserialization](#12-json-serialization-and-deserialization)
* [13. Working with JSON Objects](#13-working-with-json-objects)
* [14. Working with JSON Arrays](#14-working-with-json-arrays)
* [15. Accessing Nested JSON](#15-accessing-nested-json)
* [16. Handling Missing JSON Keys](#16-handling-missing-json-keys)
* [17. JSON with Pandas](#17-json-with-pandas)
* [18. Reading JSON with `read_json()`](#18-reading-json-with-read_json)
* [19. JSON Records](#19-json-records)
* [20. Flattening Nested JSON](#20-flattening-nested-json)
* [21. `json_normalize()`](#21-json_normalize)
* [22. Nested Arrays](#22-nested-arrays)
* [23. JSON Lines / NDJSON](#23-json-lines--ndjson)
* [24. Reading JSON from URLs and APIs](#24-reading-json-from-urls-and-apis)
* [25. API Response Structure](#25-api-response-structure)
* [26. JSON Validation](#26-json-validation)
* [27. Cleaning JSON Data](#27-cleaning-json-data)
* [28. Data Types and Missing Values](#28-data-types-and-missing-values)
* [29. Duplicate Records](#29-duplicate-records)
* [30. Dates and Timestamps](#30-dates-and-timestamps)
* [31. JSON Schema](#31-json-schema)
* [32. Handling Large JSON Files](#32-handling-large-json-files)
* [33. JSON Security Considerations](#33-json-security-considerations)
* [34. Data Leakage](#34-data-leakage)
* [35. Preparing JSON Data for Machine Learning](#35-preparing-json-data-for-machine-learning)
* [36. JSON-to-DataFrame Pipeline](#36-json-to-dataframe-pipeline)
* [37. Recommended Project Structure](#37-recommended-project-structure)
* [38. Practical Exercises](#38-practical-exercises)
* [39. Mini Projects](#39-mini-projects)
* [40. Common Mistakes](#40-common-mistakes)
* [41. Best Practices](#41-best-practices)
* [42. Professional Workflow](#42-professional-workflow)
* [43. Learning Roadmap](#43-learning-roadmap)
* [44. Key Takeaways](#44-key-takeaways)

---

# 1. What Is JSON?

JSON stands for:

> **JavaScript Object Notation**

Despite its name, JSON is language-independent and is widely supported by Python, Java, JavaScript, Go, Java, C#, Rust, and many other programming languages.

A simple JSON object:

```json
{
  "name": "Kishor",
  "age": 22,
  "city": "Pune"
}
```

The structure contains:

```text
key → value
```

For example:

```text
name → "Kishor"
age  → 22
city → "Pune"
```

---

# 2. Why JSON Matters in Machine Learning

JSON is particularly important because modern applications frequently exchange data through APIs.

A typical workflow might look like:

```text
Application
     ↓
REST API
     ↓
JSON Response
     ↓
Python
     ↓
Pandas DataFrame
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Machine Learning
```

For example, an API might return:

```json
{
  "user": {
    "id": 101,
    "name": "Kishor",
    "location": {
      "city": "Pune",
      "country": "India"
    }
  }
}
```

The data is structured but not immediately tabular.

You may need to transform it into:

|  id | name   | city | country |
| --: | ------ | ---- | ------- |
| 101 | Kishor | Pune | India   |

This process is commonly called **JSON normalization** or **flattening**.

---

# 3. JSON Data Types

JSON supports six fundamental data types:

```text
Object
Array
String
Number
Boolean
Null
```

Example:

```json
{
  "name": "Kishor",
  "age": 22,
  "active": true,
  "skills": ["Python", "ML"],
  "address": {
    "city": "Pune"
  },
  "middle_name": null
}
```

Mapping to Python:

| JSON   | Python          |
| ------ | --------------- |
| object | `dict`          |
| array  | `list`          |
| string | `str`           |
| number | `int` / `float` |
| true   | `True`          |
| false  | `False`         |
| null   | `None`          |

---

# 4. JSON Objects

A JSON object is enclosed by:

```text
{}
```

Example:

```json
{
  "id": 101,
  "name": "Alice",
  "salary": 50000
}
```

Python equivalent:

```python
data = {
    "id": 101,
    "name": "Alice",
    "salary": 50000
}
```

Access a value:

```python
print(data["name"])
```

Output:

```text
Alice
```

---

# 5. JSON Arrays

An array is represented using:

```text
[]
```

Example:

```json
[
  "Python",
  "Pandas",
  "Scikit-Learn"
]
```

Python:

```python
skills = [
    "Python",
    "Pandas",
    "Scikit-Learn"
]
```

Access an item:

```python
print(skills[0])
```

Output:

```text
Python
```

---

## Array of Objects

This is very common in APIs:

```json
[
  {
    "id": 1,
    "name": "Alice"
  },
  {
    "id": 2,
    "name": "Bob"
  }
]
```

This structure maps naturally to rows in a DataFrame.

---

# 6. Nested JSON

JSON can contain objects inside objects.

Example:

```json
{
  "id": 101,
  "name": "Kishor",
  "address": {
    "city": "Pune",
    "state": "Maharashtra",
    "country": "India"
  }
}
```

The `address` value is itself an object.

Access it:

```python
data["address"]["city"]
```

Output:

```text
Pune
```

---

## Deeper Nesting

```json
{
  "user": {
    "profile": {
      "location": {
        "city": "Pune"
      }
    }
  }
}
```

Access:

```python
city = data["user"]["profile"]["location"]["city"]
```

Deeply nested JSON can become difficult to maintain manually.

This is where normalization tools become useful.

---

# 7. JSON vs CSV vs Excel

| Feature             | JSON   | CSV     | Excel   |
| ------------------- | ------ | ------- | ------- |
| Nested objects      | ✅      | ❌       | Limited |
| Arrays              | ✅      | ❌       | Limited |
| Multiple sheets     | ❌      | ❌       | ✅       |
| Human-friendly      | ✅      | ✅       | ✅       |
| API usage           | ⭐⭐⭐    | ⭐       | ⭐       |
| Machine learning    | ✅      | ✅       | ✅       |
| Hierarchical data   | ✅      | ❌       | Limited |
| Schema flexibility  | High   | Low     | Medium  |
| File size           | Medium | Small   | Larger  |
| Multiple data types | ✅      | Limited | ✅       |

### Use JSON when:

* data is hierarchical
* consuming APIs
* working with application data
* handling nested structures
* exchanging structured information

### Use CSV when:

* data is flat and tabular
* simplicity is important
* interoperability is the priority

### Use Excel when:

* humans need to review/edit the data
* multiple worksheets are useful
* reporting and formatting matter

---

# 8. JSON File Structure

A JSON file commonly looks like:

```text
data/
└── customers.json
```

Example:

```json
[
  {
    "id": 1,
    "name": "Alice",
    "age": 25
  },
  {
    "id": 2,
    "name": "Bob",
    "age": 31
  }
]
```

---

# 9. Python JSON Module

Python provides a built-in `json` module.

No external package is required.

```python
import json
```

The module provides functions such as:

```text
json.load()
json.loads()
json.dump()
json.dumps()
```

These can be divided into:

```text
File operations
    ↓
load()
dump()

String operations
    ↓
loads()
dumps()
```

---

# 10. Reading JSON Files

Suppose:

```text
customers.json
```

contains:

```json
[
  {
    "id": 1,
    "name": "Alice"
  },
  {
    "id": 2,
    "name": "Bob"
  }
]
```

Read it:

```python
import json

with open(
    "customers.json",
    "r",
    encoding="utf-8"
) as file:
    data = json.load(file)

print(data)
```

---

## Inspect the Result

```python
print(type(data))
```

Output:

```text
<class 'list'>
```

Because the top-level JSON structure is an array.

---

# 11. Writing JSON Files

Use `json.dump()`:

```python
import json

data = {
    "name": "Kishor",
    "age": 22,
    "skills": [
        "Python",
        "Machine Learning"
    ]
}

with open(
    "output.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        data,
        file,
        indent=4
    )
```

The resulting file is formatted for readability.

---

## Preserve Unicode

```python
json.dump(
    data,
    file,
    indent=4,
    ensure_ascii=False
)
```

This is useful when working with multilingual text.

---

# 12. JSON Serialization and Deserialization

### Serialization

Converting a Python object into JSON:

```text
Python
  ↓
JSON
```

Example:

```python
json_string = json.dumps(data)
```

---

### Deserialization

Converting JSON into a Python object:

```text
JSON
  ↓
Python
```

Example:

```python
data = json.loads(json_string)
```

---

## Example

```python
import json

data = {
    "name": "Alice",
    "age": 25
}

json_string = json.dumps(data)

print(json_string)

restored = json.loads(json_string)

print(restored)
```

---

# 13. Working with JSON Objects

Example:

```python
data = {
    "name": "Alice",
    "age": 25,
    "city": "Pune"
}
```

Access:

```python
print(data["name"])
print(data["age"])
```

Add:

```python
data["country"] = "India"
```

Update:

```python
data["age"] = 26
```

Remove:

```python
del data["city"]
```

---

# 14. Working with JSON Arrays

Example:

```python
users = [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
    {"id": 3, "name": "Charlie"}
]
```

Loop through records:

```python
for user in users:
    print(user["name"])
```

Output:

```text
Alice
Bob
Charlie
```

---

## Extract One Field

```python
names = [
    user["name"]
    for user in users
]
```

Result:

```python
[
    "Alice",
    "Bob",
    "Charlie"
]
```

---

# 15. Accessing Nested JSON

Consider:

```python
data = {
    "user": {
        "name": "Alice",
        "location": {
            "city": "Pune",
            "country": "India"
        }
    }
}
```

Access:

```python
city = data["user"]["location"]["city"]

print(city)
```

Output:

```text
Pune
```

---

## Using `.get()`

Direct indexing can raise `KeyError`.

Instead:

```python
city = (
    data
    .get("user", {})
    .get("location", {})
    .get("city")
)
```

This is safer when fields are optional.

---

# 16. Handling Missing JSON Keys

Consider:

```json
{
  "id": 101,
  "name": "Alice"
}
```

There is no `email`.

This would fail:

```python
email = data["email"]
```

Use:

```python
email = data.get("email")
```

Result:

```text
None
```

Or provide a default:

```python
email = data.get(
    "email",
    "Unknown"
)
```

---

## Why This Matters

API responses may change depending on:

* user permissions
* missing information
* optional fields
* API versions
* records
* errors
* feature flags

Production code should not assume every field always exists.

---

# 17. JSON with Pandas

Pandas can convert JSON into DataFrames.

```python
import pandas as pd

df = pd.read_json(
    "customers.json"
)

print(df)
```

For JSON containing records:

```json
[
  {
    "id": 1,
    "name": "Alice",
    "age": 25
  },
  {
    "id": 2,
    "name": "Bob",
    "age": 31
  }
]
```

The result becomes:

| id | name  | age |
| -: | ----- | --: |
|  1 | Alice |  25 |
|  2 | Bob   |  31 |

---

# 18. Reading JSON with `read_json()`

Basic usage:

```python
df = pd.read_json(
    "data.json"
)
```

Inspect:

```python
print(df.head())
print(df.shape)
print(df.dtypes)
```

---

## Reading a JSON String

```python
json_data = """
[
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
]
"""

df = pd.read_json(
    json_data,
    orient="records"
)
```

For newer Pandas versions, when passing literal JSON strings, wrapping the string in `StringIO` can make the intent explicit:

```python
from io import StringIO

df = pd.read_json(
    StringIO(json_data),
    orient="records"
)
```

---

# 19. JSON Records

One of the most useful JSON formats for tabular data is:

```text
records
```

Example:

```json
[
  {
    "id": 1,
    "name": "Alice",
    "age": 25
  },
  {
    "id": 2,
    "name": "Bob",
    "age": 31
  }
]
```

Each object represents one record.

Conceptually:

```text
JSON Object → DataFrame Row
JSON Key    → DataFrame Column
```

---

## Convert DataFrame to JSON

```python
json_data = df.to_json(
    orient="records",
    indent=2
)
```

---

# 20. Flattening Nested JSON

Consider:

```json
{
  "id": 101,
  "name": "Alice",
  "address": {
    "city": "Pune",
    "state": "Maharashtra"
  }
}
```

A flat table might be:

|  id | name  | address.city | address.state |
| --: | ----- | ------------ | ------------- |
| 101 | Alice | Pune         | Maharashtra   |

This process is called:

> **Flattening nested JSON**

---

# 21. `json_normalize()`

Pandas provides:

```python
pd.json_normalize()
```

Example:

```python
import pandas as pd

data = [
    {
        "id": 101,
        "name": "Alice",
        "address": {
            "city": "Pune",
            "state": "Maharashtra"
        }
    }
]

df = pd.json_normalize(data)

print(df)
```

Columns become:

```text
id
name
address.city
address.state
```

---

## Custom Separator

```python
df = pd.json_normalize(
    data,
    sep="_"
)
```

Now:

```text
address_city
address_state
```

This can be more convenient for machine-learning feature names.

---

# 22. Nested Arrays

Consider:

```json
{
  "id": 101,
  "name": "Alice",
  "orders": [
    {
      "product": "Laptop",
      "amount": 80000
    },
    {
      "product": "Mouse",
      "amount": 1000
    }
  ]
}
```

Here:

```text
user
 └── orders
      ├── order 1
      └── order 2
```

This cannot always be represented directly as one flat row without deciding how to aggregate the orders.

Possible transformations include:

```text
User-level table
Order-level table
User + aggregated order features
```

---

## Normalize Nested Records

```python
data = [
    {
        "id": 101,
        "name": "Alice",
        "orders": [
            {
                "product": "Laptop",
                "amount": 80000
            },
            {
                "product": "Mouse",
                "amount": 1000
            }
        ]
    }
]

orders = pd.json_normalize(
    data,
    record_path="orders",
    meta=["id", "name"]
)
```

Result:

| product | amount |  id | name  |
| ------- | -----: | --: | ----- |
| Laptop  |  80000 | 101 | Alice |
| Mouse   |   1000 | 101 | Alice |

---

# 23. JSON Lines / NDJSON

JSON Lines, commonly called **JSONL** or **NDJSON**, stores one JSON object per line.

Example:

```text
{"id":1,"name":"Alice","age":25}
{"id":2,"name":"Bob","age":31}
{"id":3,"name":"Charlie","age":28}
```

This is useful for:

* logs
* event streams
* large datasets
* machine learning datasets
* incremental processing

---

## Read JSON Lines

```python
import pandas as pd

df = pd.read_json(
    "data.jsonl",
    lines=True
)
```

---

## Write JSON Lines

```python
df.to_json(
    "output.jsonl",
    orient="records",
    lines=True
)
```

---

# 24. Reading JSON from URLs and APIs

Many APIs return JSON.

Example pattern:

```text
Client
  ↓
HTTP Request
  ↓
API
  ↓
JSON Response
```

Python's `requests` library is commonly used:

```bash
pip install requests
```

Example:

```python
import requests

response = requests.get(
    "https://example.com/api/data",
    timeout=30
)

response.raise_for_status()

data = response.json()

print(data)
```

The important steps are:

```text
Request
   ↓
Status Check
   ↓
Parse JSON
   ↓
Validate
   ↓
Transform
```

---

# 25. API Response Structure

An API response may contain metadata around the actual records:

```json
{
  "status": "success",
  "page": 1,
  "total": 100,
  "data": [
    {
      "id": 1,
      "name": "Alice"
    },
    {
      "id": 2,
      "name": "Bob"
    }
  ]
}
```

The actual records are inside:

```text
data
```

Access them:

```python
records = response_data["data"]
```

Then:

```python
df = pd.json_normalize(records)
```

---

## Handle API Errors

Do not assume every response is valid JSON containing the expected fields.

```python
response = requests.get(
    url,
    timeout=30
)

response.raise_for_status()

data = response.json()

if "data" not in data:
    raise ValueError(
        "Expected 'data' field was not found."
    )
```

---

# 26. JSON Validation

Before using JSON in ML workflows, validate:

* required fields
* data types
* allowed values
* ranges
* identifiers
* missing values
* nested structures

Example:

```python
required_fields = {
    "id",
    "name",
    "age"
}

missing = (
    required_fields - data.keys()
)

if missing:
    raise ValueError(
        f"Missing fields: {missing}"
    )
```

---

## Validate Records

```python
for record in records:
    if "id" not in record:
        raise ValueError(
            "Record is missing id."
        )
```

---

# 27. Cleaning JSON Data

JSON does not guarantee clean data.

You may encounter:

```text
"Pune"
"pune"
" Pune "
"PUNE"
```

Normalize:

```python
df["city"] = (
    df["city"]
    .astype("string")
    .str.strip()
    .str.lower()
)
```

---

## Numeric Conversion

```python
df["income"] = pd.to_numeric(
    df["income"],
    errors="coerce"
)
```

---

## Standardize Missing Values

```python
df = df.replace(
    ["N/A", "NA", "-", "?"],
    pd.NA
)
```

---

# 28. Data Types and Missing Values

JSON itself has a limited set of primitive data types.

When loaded into Pandas, additional type decisions are required.

For example:

```json
{
  "age": null
}
```

may become a missing value.

Inspect:

```python
print(df.dtypes)
```

Check missing values:

```python
print(df.isna().sum())
```

---

## Nullable Integer

When integers contain missing values:

```python
df["age"] = (
    pd.to_numeric(
        df["age"],
        errors="coerce"
    )
    .astype("Int64")
)
```

---

# 29. Duplicate Records

Find duplicates:

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

For identifier-based duplicates:

```python
df = df.drop_duplicates(
    subset=["id"]
)
```

Do not automatically remove duplicates without understanding what they represent.

In event data, multiple records for the same entity may be completely valid.

---

# 30. Dates and Timestamps

JSON APIs commonly return timestamps such as:

```text
2026-09-28T10:30:00Z
```

Convert:

```python
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
    utc=True
)
```

Extract useful features:

```python
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)
```

---

## Time-Zone Awareness

When data comes from multiple systems or countries, time zones matter.

Prefer timezone-aware timestamps when the source provides them.

Avoid silently converting timestamps without documenting the chosen timezone.

---

# 31. JSON Schema

For larger systems, JSON structure can be formally described using **JSON Schema**.

A simplified example:

```json
{
  "type": "object",
  "required": [
    "id",
    "name"
  ],
  "properties": {
    "id": {
      "type": "integer"
    },
    "name": {
      "type": "string"
    },
    "age": {
      "type": "integer",
      "minimum": 0
    }
  }
}
```

Schema validation can verify whether incoming JSON follows the expected structure.

A Python package such as `jsonschema` can be used:

```bash
pip install jsonschema
```

Example:

```python
from jsonschema import validate

validate(
    instance=data,
    schema=schema
)
```

This is particularly useful for:

* APIs
* data ingestion
* production pipelines
* automated testing
* data contracts

---

# 32. Handling Large JSON Files

Loading an enormous JSON file using:

```python
json.load(file)
```

may consume significant memory.

For large datasets, consider:

```text
JSON Lines / NDJSON
Streaming
Chunk processing
Databases
Object storage
Parquet
```

JSONL is particularly useful because each line can be processed independently.

Conceptually:

```text
Large File
│
├── Record 1 → process
├── Record 2 → process
├── Record 3 → process
├── ...
└── Record N → process
```

---

## Line-by-Line Processing

```python
import json

with open(
    "events.jsonl",
    "r",
    encoding="utf-8"
) as file:

    for line in file:
        record = json.loads(line)

        # Process one record
        print(record)
```

This avoids loading the entire file into memory at once.

---

# 33. JSON Security Considerations

JSON is a data format, but the surrounding application can still introduce security risks.

Potential concerns include:

* untrusted input
* excessive payload size
* deeply nested data
* malformed JSON
* sensitive information
* unsafe downstream processing
* API abuse
* authentication tokens
* personally identifiable information

---

## Validate Untrusted Data

Do not blindly assume:

```python
data["user"]["profile"]["age"]
```

exists.

Validate structure before accessing critical fields.

---

## Avoid Executing JSON Content

JSON should be parsed as data.

Do not treat arbitrary JSON strings as executable Python code.

Use:

```python
json.loads()
```

rather than unsafe dynamic evaluation techniques.

---

## Protect Secrets

Never place credentials directly inside JSON files committed to Git.

Avoid:

```json
{
  "api_key": "SECRET_KEY"
}
```

inside a public repository.

Use environment variables or a secret-management system instead.

---

# 34. Data Leakage

JSON ingestion can introduce the same machine-learning leakage problems as CSV or Excel.

Suppose an API returns:

```json
{
  "customer_id": 101,
  "income": 50000,
  "application_status": "approved",
  "approval_timestamp": "2026-09-20T10:00:00Z"
}
```

If the goal is to predict approval before the decision occurs, fields created after the decision may leak the target.

Always ask:

> **Would this information actually be available at prediction time?**

---

## Correct ML Workflow

```text
Raw JSON
   ↓
Validate
   ↓
Transform
   ↓
Split Data
   ↓
Fit Training Preprocessing
   ↓
Transform Validation/Test Data
   ↓
Train Model
   ↓
Evaluate
```

For production workflows, use reproducible preprocessing pipelines.

---

# 35. Preparing JSON Data for Machine Learning

Suppose the JSON contains:

```json
{
  "id": 101,
  "age": 25,
  "income": 50000,
  "location": {
    "city": "Pune",
    "state": "Maharashtra"
  },
  "purchased": true
}
```

First flatten it:

```python
import pandas as pd

df = pd.json_normalize(
    [data],
    sep="_"
)
```

Result:

```text
id
age
income
location_city
location_state
purchased
```

Separate features and target:

```python
X = df[
    [
        "age",
        "income",
        "location_city",
        "location_state"
    ]
]

y = df["purchased"]
```

---

# 36. JSON-to-DataFrame Pipeline

A practical pipeline:

```text
                JSON
                  ↓
          Parse JSON Structure
                  ↓
            Validate Schema
                  ↓
          Handle Missing Fields
                  ↓
           Normalize / Flatten
                  ↓
            Clean Data Types
                  ↓
          Remove/Investigate Duplicates
                  ↓
          Validate Business Rules
                  ↓
          Feature Engineering
                  ↓
             Train/Test Split
                  ↓
          ML Preprocessing Pipeline
                  ↓
               Model
```

Example:

```python
import json
import pandas as pd

with open(
    "customers.json",
    "r",
    encoding="utf-8"
) as file:
    records = json.load(file)

df = pd.json_normalize(
    records,
    sep="_"
)

df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)

df["income"] = pd.to_numeric(
    df["income"],
    errors="coerce"
)

print(df.head())
print(df.info())
```

---

# 37. Recommended Project Structure

A professional JSON-processing project can use:

```text
04-JSON-Data/
│
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── customers.json
│   │   └── events.jsonl
│   │
│   └── processed/
│       └── customers_clean.csv
│
├── schemas/
│   └── customer_schema.json
│
├── notebooks/
│   └── json-eda.ipynb
│
├── scripts/
│   ├── load_json.py
│   ├── normalize_json.py
│   ├── validate_json.py
│   └── transform_json.py
│
├── tests/
│   └── test_json_validation.py
│
└── README.md
```

---

# 38. Practical Exercises

## Exercise 1 — Basic JSON

Create:

```text
student.json
```

with:

```text
name
age
course
skills
```

Read it using Python's `json` module.

---

## Exercise 2 — JSON Array

Create 10 student records:

```json
[
  {
    "id": 1,
    "name": "Student 1"
  }
]
```

Convert them into a DataFrame.

---

## Exercise 3 — Nested JSON

Create:

```json
{
  "name": "Alice",
  "address": {
    "city": "Pune",
    "state": "Maharashtra"
  }
}
```

Flatten it using:

```python
pd.json_normalize()
```

---

## Exercise 4 — Nested Arrays

Create customer data containing:

```text
customer
 └── orders
      ├── order 1
      ├── order 2
      └── order 3
```

Convert the orders into a separate DataFrame.

---

## Exercise 5 — JSONL

Create:

```text
events.jsonl
```

with 1,000 records.

Read it using:

```python
pd.read_json(
    "events.jsonl",
    lines=True
)
```

---

## Exercise 6 — API Data

Use a public JSON API and:

1. request the data
2. validate the response
3. inspect the JSON structure
4. extract records
5. convert them to Pandas
6. clean the dataset
7. export it to CSV

---

# 39. Mini Projects

## 🌐 Project 1 — API Data Collector

Build a Python script that:

```text
API
 ↓
HTTP Request
 ↓
JSON
 ↓
Validation
 ↓
DataFrame
 ↓
CSV
```

Include:

* timeout handling
* HTTP error handling
* schema validation
* logging
* output file generation

---

## 🛒 Project 2 — E-Commerce JSON Dataset

Create JSON containing:

```text
Customers
Products
Orders
Payments
```

Normalize the data into separate tables.

Then create:

```text
customer_features
```

for machine learning.

---

## 🤖 Project 3 — Customer Purchase Prediction

JSON:

```text
customer.json
```

Features:

```text
age
income
city
purchase_frequency
average_order_value
```

Target:

```text
purchased
```

Prepare the JSON data for a classification model.

---

## 📡 Project 4 — IoT Event Processing

Create JSONL records containing:

```text
device_id
timestamp
temperature
humidity
pressure
status
```

Build a pipeline that:

1. reads events
2. validates fields
3. parses timestamps
4. detects invalid sensor values
5. creates statistical features
6. prepares data for anomaly detection

---

## 🧠 Project 5 — Nested JSON to ML Dataset

Create a deeply nested dataset:

```text
Customer
 ├── Profile
 ├── Location
 ├── Orders
 │    ├── Order
 │    └── Products
 └── Activity
```

Flatten the data into useful feature tables.

The goal is to understand that **data modeling decisions are required before nested data can become ML-ready**.

---

# 40. Common Mistakes

## ❌ Mistake 1 — Assuming JSON Is Always Flat

JSON can contain:

```text
objects
arrays
nested objects
arrays of objects
mixed structures
```

Always inspect the structure first.

---

## ❌ Mistake 2 — Using Direct Key Access Everywhere

Avoid blindly doing:

```python
data["user"]["profile"]["city"]
```

when fields are optional.

Prefer validated access or `.get()` where appropriate.

---

## ❌ Mistake 3 — Ignoring API Errors

Do not immediately call:

```python
response.json()
```

without checking the HTTP response.

Use:

```python
response.raise_for_status()
```

first.

---

## ❌ Mistake 4 — Flattening Without Understanding Relationships

Nested arrays may represent one-to-many relationships.

Blind flattening can duplicate parent information or distort the dataset.

Understand the underlying data model first.

---

## ❌ Mistake 5 — Loading Huge JSON Files Into Memory

Avoid:

```python
json.load()
```

for very large datasets when memory is limited.

Consider JSONL or streaming approaches.

---

## ❌ Mistake 6 — Mixing Training and Test Information

Do not calculate preprocessing statistics using the entire dataset before splitting.

This can create data leakage.

---

## ❌ Mistake 7 — Hard-Coding API Credentials

Never commit:

```text
API keys
passwords
tokens
private credentials
```

to GitHub.

---

# 41. Best Practices

### 1. Inspect Before Transforming

Understand the JSON structure before writing transformation logic.

---

### 2. Validate External Data

Treat API responses and downloaded JSON as untrusted input until validated.

---

### 3. Preserve Raw Data

Keep:

```text
data/raw/
```

separate from:

```text
data/processed/
```

---

### 4. Keep Transformations Reproducible

Prefer code over manually editing JSON files.

---

### 5. Document the Schema

Record:

```text
field
type
required/optional
description
allowed values
```

---

### 6. Handle Missing Fields

External data can evolve.

Use validation and defensive parsing.

---

### 7. Preserve Relationships

When flattening nested JSON, understand:

```text
one-to-one
one-to-many
many-to-many
```

relationships.

---

### 8. Keep API Collection Separate From ML Code

Prefer:

```text
API Collector
      ↓
Raw JSON
      ↓
Normalizer
      ↓
Processed Dataset
      ↓
ML Pipeline
```

instead of mixing API calls directly into model-training code.

---

### 9. Add Metadata

Record useful provenance such as:

```text
source
endpoint
collection timestamp
API version
schema version
processing version
```

---

### 10. Test Transformations

A JSON transformation should be tested against:

* valid records
* missing fields
* empty arrays
* unexpected types
* malformed records
* duplicate records
* schema changes

---

# 42. Professional Workflow

A production-style JSON ingestion workflow:

```text
┌──────────────────┐
│ External Source  │
│ / REST API       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Raw JSON / JSONL │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Schema Validation│
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Data Validation  │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Normalization    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Data Cleaning    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Feature Creation │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Train/Test Split │
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

# 43. Learning Roadmap

Follow this progression:

```text
JSON Basics
      ↓
Objects
      ↓
Arrays
      ↓
Nested JSON
      ↓
Python json Module
      ↓
Serialization
      ↓
Deserialization
      ↓
Pandas read_json()
      ↓
JSON Records
      ↓
json_normalize()
      ↓
Nested Arrays
      ↓
JSONL / NDJSON
      ↓
REST API Responses
      ↓
Schema Validation
      ↓
Data Cleaning
      ↓
Data Transformation
      ↓
Feature Engineering
      ↓
ML Preprocessing
      ↓
Machine Learning
```

---

# 44. Key Takeaways

After completing this section, you should understand how to:

* understand JSON syntax
* work with JSON objects
* work with JSON arrays
* handle nested JSON
* read JSON using Python
* write JSON files
* serialize Python objects
* deserialize JSON
* use `.get()` safely
* load JSON into Pandas
* understand JSON record orientation
* flatten nested objects
* use `pd.json_normalize()`
* process nested arrays
* work with JSON Lines
* consume JSON API responses
* handle API errors
* validate JSON structures
* clean JSON-derived datasets
* process dates and timestamps
* handle missing values
* detect duplicate records
* work with JSON Schema
* process large JSON datasets
* protect secrets and sensitive data
* identify machine-learning data leakage
* transform JSON into ML-ready tables
* design reproducible JSON ingestion pipelines

The most important concept is:

> **JSON describes structure; machine learning usually requires carefully designed tabular or feature representations.**

The job of a data scientist or ML engineer is not simply to load JSON. It is to understand the structure, preserve meaningful relationships, validate the data, transform it appropriately, and produce a reliable dataset for modeling.

---

## 🔗 Connection to the Next Sections

Your data-collection progression is now:

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

JSON is particularly important because it forms the bridge between **static datasets** and **live data collected from APIs and applications**.

The next section can build directly on this foundation by exploring **APIs and programmatic data collection**.

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
4. Test all Python examples.
5. Commit your changes.
6. Open a Pull Request.

Example:

```bash
git clone https://github.com/Kishor055/Machine-Learning.git

cd Machine-Learning

git checkout -b feature/improve-json-guide
```

---

## ⭐ Support

If this repository helps you learn Machine Learning, consider giving it a ⭐ on GitHub.

Happy Learning! 🚀

**Python → Data → JSON → Features → Models → Machine Learning**

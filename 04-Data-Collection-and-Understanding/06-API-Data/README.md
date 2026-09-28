# 🌐 API Data for Machine Learning

APIs (Application Programming Interfaces) are one of the most important ways to collect **live, structured, and machine-readable data** for Machine Learning and Data Science applications.

Instead of manually downloading datasets, APIs allow applications and ML pipelines to programmatically retrieve data from external services such as:

* 🌦️ Weather services
* 💰 Financial platforms
* 🗺️ Maps and location services
* 📰 News platforms
* 🛒 E-commerce systems
* 🏥 Healthcare systems
* 🏢 Business applications
* 📊 Government platforms
* 🤖 AI services
* 📱 Social platforms
* 🛰️ IoT systems
* ☁️ Cloud services

This section explains how to **discover, consume, validate, clean, store, and transform API data into ML-ready datasets**.

---

## 📚 Table of Contents

1. [What Is an API?](#-what-is-an-api)
2. [Why APIs Matter for Machine Learning](#-why-apis-matter-for-machine-learning)
3. [API Architecture](#-api-architecture)
4. [How an API Request Works](#-how-an-api-request-works)
5. [HTTP Fundamentals](#-http-fundamentals)
6. [HTTP Methods](#-http-methods)
7. [HTTP Status Codes](#-http-status-codes)
8. [API Endpoints](#-api-endpoints)
9. [Query Parameters](#-query-parameters)
10. [Path Parameters](#-path-parameters)
11. [Request Headers](#-request-headers)
12. [Request Body](#-request-body)
13. [JSON Data](#-json-data)
14. [Authentication](#-authentication)
15. [API Keys](#-api-keys)
16. [Bearer Tokens](#-bearer-tokens)
17. [OAuth](#-oauth)
18. [Python API Requests](#-python-api-requests)
19. [GET Requests](#-get-requests)
20. [POST Requests](#-post-requests)
21. [Handling JSON Responses](#-handling-json-responses)
22. [Query Parameters in Python](#-query-parameters-in-python)
23. [Headers in Python](#-headers-in-python)
24. [Timeouts and Error Handling](#-timeouts-and-error-handling)
25. [Retry Strategies](#-retry-strategies)
26. [Pagination](#-pagination)
27. [Rate Limits](#-rate-limits)
28. [API Documentation](#-api-documentation)
29. [Public APIs](#-public-apis)
30. [REST APIs](#-rest-apis)
31. [GraphQL APIs](#-graphql-apis)
32. [API Data → Pandas](#-api-data--pandas)
33. [Nested JSON](#-nested-json)
34. [Flattening API Data](#-flattening-api-data)
35. [Data Validation](#-data-validation)
36. [Data Cleaning](#-data-cleaning)
37. [Feature Engineering](#-feature-engineering)
38. [API Data Leakage](#-api-data-leakage)
39. [Historical Data](#-historical-data)
40. [Real-Time ML Data](#-real-time-ml-data)
41. [API Data Storage](#-api-data-storage)
42. [API Data Pipeline](#-api-data-pipeline)
43. [Caching](#-caching)
44. [Logging](#-logging)
45. [Security](#-security)
46. [Production Best Practices](#-production-best-practices)
47. [Common Mistakes](#-common-mistakes)
48. [Mini Projects](#-mini-projects)
49. [Exercises](#-exercises)
50. [Recommended Project Structure](#-recommended-project-structure)
51. [End-to-End Example](#-end-to-end-example)
52. [API → ML Workflow](#-api--ml-workflow)
53. [Learning Roadmap](#-learning-roadmap)
54. [Key Takeaways](#-key-takeaways)
55. [Next Step](#-next-step)

---

# 🔹 What Is an API?

**API** stands for **Application Programming Interface**.

An API provides a standardized way for one software application to communicate with another.

For example:

```text
Python Application
       │
       │ HTTP Request
       ▼
Weather API
       │
       │ JSON Response
       ▼
Python Application
       │
       ▼
Pandas DataFrame
       │
       ▼
Machine Learning Model
```

A simplified API request may look like:

```text
GET /weather?city=Mumbai
```

The server may return:

```json
{
  "city": "Mumbai",
  "temperature": 29.5,
  "humidity": 78,
  "wind_speed": 12.4
}
```

This information can then be transformed into features for an ML model.

---

# 🔹 Why APIs Matter for Machine Learning

Traditional ML projects often begin with static datasets:

```text
CSV
Excel
JSON
Database
```

Real-world ML systems frequently need continuously changing data:

```text
API
 ↓
Data Collection
 ↓
Validation
 ↓
Cleaning
 ↓
Feature Engineering
 ↓
ML Model
```

APIs are useful when models require:

* Current information
* Frequent updates
* External signals
* Real-time predictions
* Historical records
* Third-party datasets
* Automated data ingestion

### Example

A weather prediction application might collect:

```text
temperature
humidity
pressure
wind_speed
precipitation
cloud_cover
```

and use them as ML features.

---

# 🔹 API Architecture

A typical API architecture contains:

```text
Client
  │
  │ HTTP Request
  ▼
API Server
  │
  ├── Authentication
  ├── Validation
  ├── Business Logic
  └── Database
        │
        ▼
    API Response
```

The client might be:

* Python
* JavaScript
* Mobile application
* Web application
* ML pipeline
* Data engineering system

---

# 🔹 How an API Request Works

A typical interaction looks like:

```text
1. Client creates request
          ↓
2. Request sent to server
          ↓
3. Server authenticates request
          ↓
4. Server processes request
          ↓
5. Server retrieves data
          ↓
6. Server returns response
          ↓
7. Client validates response
          ↓
8. Data transformed for ML
```

---

# 🔹 HTTP Fundamentals

Most web APIs communicate using **HTTP** or **HTTPS**.

An HTTP request contains:

```text
Method
URL
Headers
Parameters
Body
```

Example:

```http
GET https://api.example.com/weather?city=Mumbai
Authorization: Bearer TOKEN
Accept: application/json
```

---

# 🔹 HTTP Methods

Common HTTP methods include:

| Method | Purpose            |
| ------ | ------------------ |
| GET    | Retrieve data      |
| POST   | Create/send data   |
| PUT    | Replace a resource |
| PATCH  | Partially update   |
| DELETE | Delete a resource  |

For ML data collection, `GET` is especially common.

---

# 🔹 HTTP Status Codes

API responses include status codes.

| Code | Meaning             |
| ---- | ------------------- |
| 200  | Success             |
| 201  | Created             |
| 204  | No content          |
| 400  | Bad request         |
| 401  | Unauthorized        |
| 403  | Forbidden           |
| 404  | Not found           |
| 408  | Request timeout     |
| 409  | Conflict            |
| 429  | Rate limit exceeded |
| 500  | Server error        |
| 502  | Bad gateway         |
| 503  | Service unavailable |
| 504  | Gateway timeout     |

A production API client should never assume every request succeeds.

---

# 🔹 API Endpoints

An **endpoint** is a specific URL through which an API exposes functionality.

Example:

```text
https://api.example.com/users
```

Another endpoint might be:

```text
https://api.example.com/products
```

Or:

```text
https://api.example.com/weather
```

Different endpoints usually represent different resources or operations.

---

# 🔹 Query Parameters

Query parameters provide additional information to an endpoint.

Example:

```text
https://api.example.com/weather?city=Mumbai&units=metric
```

Here:

```text
city=Mumbai
units=metric
```

are query parameters.

In Python:

```python
import requests

url = "https://api.example.com/weather"

params = {
    "city": "Mumbai",
    "units": "metric",
}

response = requests.get(url, params=params, timeout=10)

print(response.url)
```

Using `params` is safer and cleaner than manually constructing URLs.

---

# 🔹 Path Parameters

Path parameters are part of the URL path.

Example:

```text
https://api.example.com/users/123
```

Here:

```text
123
```

may represent a user ID.

Python:

```python
user_id = 123

url = f"https://api.example.com/users/{user_id}"

response = requests.get(url, timeout=10)
```

---

# 🔹 Request Headers

Headers provide additional metadata.

Example:

```python
headers = {
    "Accept": "application/json",
    "User-Agent": "ml-data-pipeline/1.0",
}

response = requests.get(
    "https://api.example.com/data",
    headers=headers,
    timeout=10,
)
```

Common headers include:

```text
Accept
Content-Type
Authorization
User-Agent
```

---

# 🔹 Request Body

POST and other write operations may send data in the request body.

Example:

```python
import requests

payload = {
    "city": "Mumbai",
    "units": "metric",
}

response = requests.post(
    "https://api.example.com/weather/query",
    json=payload,
    timeout=10,
)

print(response.status_code)
```

Using `json=payload` allows `requests` to serialize the dictionary as JSON.

---

# 🔹 JSON Data

JSON is one of the most common formats returned by APIs.

Example:

```json
{
  "id": 101,
  "name": "Kishor",
  "skills": [
    "Python",
    "Machine Learning",
    "SQL"
  ]
}
```

JSON maps naturally to Python:

```text
JSON object → dict
JSON array  → list
string      → str
number      → int / float
boolean     → bool
null        → None
```

---

# 🔹 Authentication

APIs may require authentication.

Common approaches include:

```text
API Key
Bearer Token
OAuth 2.0
Basic Authentication
Signed Requests
```

Never assume that an endpoint is public simply because its URL is accessible.

---

# 🔹 API Keys

An API key is a credential identifying the client.

Bad practice:

```python
API_KEY = "my-secret-key-123"
```

Do not commit secrets to GitHub.

Instead use environment variables:

```python
import os

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set")
```

Example shell:

```bash
export API_KEY="your-secret-key"
```

Windows PowerShell:

```powershell
$env:API_KEY="your-secret-key"
```

---

# 🔹 Bearer Tokens

A bearer token is commonly supplied through the `Authorization` header.

```python
import os
import requests

token = os.getenv("API_TOKEN")

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/json",
}

response = requests.get(
    "https://api.example.com/data",
    headers=headers,
    timeout=10,
)
```

---

# 🔹 OAuth

OAuth allows applications to obtain delegated access without directly handling another user's password.

OAuth is commonly used by:

* Cloud services
* Social platforms
* Enterprise applications
* Google services
* Microsoft services
* GitHub
* Other third-party platforms

OAuth flows can be more complex than API keys because they may involve:

```text
Authorization
↓
Access Token
↓
API Request
↓
Token Refresh
```

Always follow the provider's official documentation.

---

# 🔹 Python API Requests

The `requests` library is one of the simplest ways to consume REST APIs in Python.

Install:

```bash
pip install requests
```

Basic request:

```python
import requests

url = "https://api.example.com/data"

response = requests.get(url, timeout=10)

print(response.status_code)
print(response.text)
```

---

# 🔹 GET Requests

Example:

```python
import requests

def fetch_data(url: str) -> dict:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def main() -> None:
    data = fetch_data("https://api.example.com/data")
    print(data)


if __name__ == "__main__":
    main()
```

### Why `raise_for_status()`?

It converts unsuccessful HTTP responses into exceptions.

For example:

```text
404 → HTTPError
500 → HTTPError
```

This prevents silent failures.

---

# 🔹 POST Requests

Example:

```python
import requests

payload = {
    "name": "Kishor",
    "role": "student",
}

response = requests.post(
    "https://api.example.com/users",
    json=payload,
    timeout=10,
)

response.raise_for_status()

print(response.json())
```

---

# 🔹 Handling JSON Responses

Always validate that the response is usable.

```python
import requests

response = requests.get(
    "https://api.example.com/data",
    timeout=10,
)

response.raise_for_status()

data = response.json()

print(type(data))
```

Possible response:

```python
{
    "results": [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20}
    ]
}
```

---

# 🔹 Query Parameters in Python

Prefer:

```python
params = {
    "page": 1,
    "limit": 100,
    "category": "technology",
}

response = requests.get(
    url,
    params=params,
    timeout=10,
)
```

Instead of:

```python
url = (
    "https://api.example.com/data"
    "?page=1&limit=100&category=technology"
)
```

The first approach handles URL encoding more reliably.

---

# 🔹 Headers in Python

Example:

```python
headers = {
    "Accept": "application/json",
    "User-Agent": "ml-data-collector/1.0",
}

response = requests.get(
    url,
    headers=headers,
    timeout=10,
)
```

Authenticated API:

```python
headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/json",
}
```

---

# 🔹 Timeouts and Error Handling

Never make production requests without a timeout.

Bad:

```python
requests.get(url)
```

Better:

```python
requests.get(url, timeout=10)
```

Production-oriented example:

```python
import requests


def fetch_json(url: str) -> dict:
    try:
        response = requests.get(
            url,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    except requests.Timeout as exc:
        raise RuntimeError("API request timed out") from exc

    except requests.HTTPError as exc:
        raise RuntimeError(
            f"API returned HTTP error: {response.status_code}"
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError("API request failed") from exc
```

---

# 🔹 Retry Strategies

Temporary failures can occur because of:

* Network problems
* Server overload
* Temporary outages
* Rate limits
* Gateway errors

A production client can use retries with exponential backoff.

Conceptually:

```text
Attempt 1
   ↓
Failure
   ↓
Wait 1 second
   ↓
Attempt 2
   ↓
Failure
   ↓
Wait 2 seconds
   ↓
Attempt 3
```

With `requests`, retry behavior can be configured using `urllib3` adapters.

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session
```

Then:

```python
session = create_session()

response = session.get(
    "https://api.example.com/data",
    timeout=10,
)
```

Retries should be designed carefully. Retrying a non-idempotent operation such as an unsafe `POST` can create unintended side effects.

---

# 🔹 Pagination

Large APIs rarely return every record in one response.

Instead:

```text
Request page 1
     ↓
Request page 2
     ↓
Request page 3
     ↓
...
```

Common pagination methods:

```text
page + limit
offset + limit
cursor
next URL
```

Example:

```python
import requests


def fetch_all_pages(base_url: str) -> list[dict]:
    records = []
    page = 1

    while True:
        response = requests.get(
            base_url,
            params={
                "page": page,
                "limit": 100,
            },
            timeout=10,
        )

        response.raise_for_status()

        payload = response.json()
        batch = payload.get("results", [])

        if not batch:
            break

        records.extend(batch)
        page += 1

    return records
```

Always inspect the API documentation because pagination semantics differ between providers.

---

# 🔹 Rate Limits

APIs often limit requests.

For example:

```text
100 requests / minute
```

A response may indicate:

```text
429 Too Many Requests
```

Good practices:

* Respect provider limits.
* Use pagination efficiently.
* Avoid unnecessary requests.
* Cache responses.
* Implement backoff.
* Monitor request volume.

Never attempt to bypass API rate limits.

---

# 🔹 API Documentation

Before using an API, identify:

```text
Base URL
Endpoints
Authentication
Parameters
Headers
Request body
Response format
Pagination
Rate limits
Error codes
Terms of use
Data licensing
```

A good API workflow starts with documentation rather than trial-and-error requests.

---

# 🔹 Public APIs

Public APIs can provide useful datasets for learning and experimentation.

Examples of API categories:

```text
Weather
Geocoding
Finance
Government
Books
Movies
Space
Sports
Education
Research
```

Before using a public API, check:

* Usage limits
* Authentication requirements
* Data license
* Attribution requirements
* Commercial restrictions
* Retention policies
* Privacy requirements

---

# 🔹 REST APIs

REST APIs commonly expose resources through HTTP.

Example:

```text
GET    /users
GET    /users/123
POST   /users
PATCH  /users/123
DELETE /users/123
```

A REST response often uses JSON:

```json
{
  "id": 123,
  "name": "Alice",
  "active": true
}
```

---

# 🔹 GraphQL APIs

GraphQL allows clients to request specific fields.

Example:

```graphql
query {
  user(id: 123) {
    name
    email
  }
}
```

Instead of receiving an entire object, the client can request exactly the required fields.

GraphQL can be useful when:

* APIs contain deeply nested data
* Clients need flexible queries
* Over-fetching is a concern

However, GraphQL introduces its own concepts such as:

```text
Queries
Mutations
Schemas
Resolvers
Fragments
Variables
```

---

# 🔹 API Data → Pandas

API responses can easily become Pandas DataFrames.

```python
import pandas as pd
import requests


response = requests.get(
    "https://api.example.com/products",
    timeout=10,
)

response.raise_for_status()

data = response.json()

df = pd.DataFrame(data["results"])

print(df.head())
print(df.info())
```

Typical transformation:

```text
API
 ↓
JSON
 ↓
Python dict/list
 ↓
Pandas DataFrame
 ↓
Cleaning
 ↓
Feature Engineering
 ↓
ML Dataset
```

---

# 🔹 Nested JSON

APIs frequently return nested structures.

Example:

```json
{
  "id": 101,
  "user": {
    "name": "Kishor",
    "location": {
      "city": "Mumbai"
    }
  }
}
```

A direct DataFrame conversion may not produce the desired structure.

Use:

```python
import pandas as pd

data = [
    {
        "id": 101,
        "user": {
            "name": "Kishor",
            "location": {
                "city": "Mumbai"
            },
        },
    }
]

df = pd.json_normalize(data)

print(df)
```

---

# 🔹 Flattening API Data

`json_normalize()` is particularly useful for nested API responses.

Example:

```python
import pandas as pd

records = [
    {
        "id": 1,
        "profile": {
            "age": 22,
            "city": "Mumbai",
        },
    },
    {
        "id": 2,
        "profile": {
            "age": 25,
            "city": "Pune",
        },
    },
]

df = pd.json_normalize(records)

print(df)
```

Result:

```text
id | profile.age | profile.city
---|-------------|-------------
1  | 22          | Mumbai
2  | 25          | Pune
```

---

# 🔹 Data Validation

API data should never automatically be considered trustworthy.

Validate:

### Schema

```text
Required columns exist
Expected fields exist
Correct data types
```

### Range

```text
age >= 0
temperature within expected limits
percentage between 0 and 100
```

### Missing values

```text
None
null
NaN
empty strings
```

### Duplicates

```text
Duplicate IDs
Repeated API records
```

Example:

```python
required_columns = {
    "id",
    "age",
    "city",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )
```

---

# 🔹 Data Cleaning

Typical API cleaning steps:

```text
Raw API Data
     ↓
Remove duplicates
     ↓
Handle missing values
     ↓
Convert data types
     ↓
Parse dates
     ↓
Normalize categories
     ↓
Validate ranges
     ↓
Feature Engineering
```

Example:

```python
df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce",
)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce",
)
```

---

# 🔹 Feature Engineering

API data can be transformed into ML features.

Suppose an API returns:

```text
temperature
humidity
wind_speed
pressure
```

You might create:

```text
temperature_humidity_ratio
wind_pressure_ratio
temperature_squared
```

Example:

```python
df["temperature_humidity_ratio"] = (
    df["temperature"] /
    df["humidity"].replace(0, pd.NA)
)
```

Other common features include:

```text
Date → year
Date → month
Date → day
Timestamp → hour
Category → encoded value
Latitude + Longitude → geographic features
Text → NLP features
```

Feature engineering should be based on information that would actually be available at prediction time.

---

# 🔹 API Data Leakage

One of the most important ML concerns is **data leakage**.

Suppose you're predicting:

```text
Tomorrow's weather
```

but your training dataset contains:

```text
Tomorrow's actual temperature
```

The model is receiving information that would not have been available when the prediction was supposed to be made.

This creates unrealistic performance.

### Leakage Example

```text
Historical API Data
        │
        ├── Data available at prediction time
        │
        └── Data published later
```

Only the first category should be used for the corresponding prediction.

Always record:

```text
event_time
collection_time
publication_time
```

when timing matters.

---

# 🔹 Historical Data

APIs may expose:

```text
Current data
Historical data
Forecast data
```

These should not automatically be treated as equivalent.

For ML:

```text
Prediction timestamp
        ↓
What information was available?
        ↓
Use only information available at that time
```

This is especially important for:

* Finance
* Weather
* Demand forecasting
* Fraud detection
* Recommendation systems
* Operations forecasting

---

# 🔹 Real-Time ML Data

A real-time ML system may look like:

```text
External API
     ↓
API Collector
     ↓
Validation
     ↓
Feature Processing
     ↓
ML Model
     ↓
Prediction
     ↓
Application
```

Example:

```text
Weather API
     ↓
Current conditions
     ↓
Feature transformation
     ↓
Model
     ↓
Rain prediction
```

---

# 🔹 API Data Storage

Do not always send API responses directly into a model.

A robust pipeline often stores raw data first:

```text
API
 ↓
Raw JSON
 ↓
Data Validation
 ↓
Processed Dataset
 ↓
Feature Store / Database
 ↓
ML Model
```

Possible storage options:

```text
JSON
CSV
Parquet
SQLite
PostgreSQL
Data Warehouse
Object Storage
```

---

# 🔹 API Data Pipeline

A production-style pipeline might look like:

```text
             ┌───────────────┐
             │ External API  │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ API Collector │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Raw Storage   │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Validation    │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Cleaning      │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ Feature Eng.  │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │ ML Dataset    │
             └───────────────┘
```

---

# 🔹 Caching

Caching avoids repeatedly requesting the same data.

Without caching:

```text
Application
 ↓
API
 ↓
Same request
 ↓
API
 ↓
Same request
 ↓
API
```

With caching:

```text
Application
 ↓
Cache
 ├── HIT → Return cached data
 └── MISS → API → Store result
```

Caching can:

* Reduce API usage
* Improve performance
* Reduce costs
* Protect against temporary API outages

But cached data must have an appropriate freshness policy.

---

# 🔹 Logging

Production API pipelines should record useful operational information.

Example:

```text
Timestamp
Endpoint
HTTP status
Latency
Number of records
Retry count
Error type
```

Example:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

logger.info("Starting API collection")
logger.info("Fetched %d records", 250)
```

Never log secrets such as:

```text
API keys
Passwords
Access tokens
Private credentials
```

---

# 🔹 Security

Important API security practices:

### Never hard-code secrets

Bad:

```python
API_KEY = "secret-value"
```

Good:

```python
import os

API_KEY = os.environ["API_KEY"]
```

### Never commit `.env`

Add:

```text
.env
```

to:

```text
.gitignore
```

Example:

```gitignore
.env
*.key
*.pem
secrets/
```

### Use HTTPS

Prefer:

```text
https://
```

over:

```text
http://
```

for authenticated or sensitive traffic.

### Validate external data

Never assume API responses are safe or correctly formatted.

---

# 🔹 Production Best Practices

A professional API data collector should:

* Use timeouts.
* Validate HTTP responses.
* Validate JSON structure.
* Handle missing fields.
* Handle rate limits.
* Use retries carefully.
* Implement pagination.
* Cache when appropriate.
* Store raw responses when reproducibility matters.
* Track timestamps.
* Log failures.
* Protect credentials.
* Respect API terms.
* Validate data before ML processing.
* Monitor data quality.
* Avoid training/serving skew.
* Prevent temporal leakage.

---

# 🔹 Common Mistakes

### ❌ No timeout

```python
requests.get(url)
```

### ✅ Better

```python
requests.get(url, timeout=10)
```

---

### ❌ Hard-coded API key

```python
API_KEY = "123456789"
```

### ✅ Better

```python
import os

API_KEY = os.getenv("API_KEY")
```

---

### ❌ Assuming the response is valid

```python
data = response.json()
```

### ✅ Better

```python
response.raise_for_status()
data = response.json()
```

---

### ❌ Ignoring pagination

```text
First 100 records
        ↓
Train model
```

The dataset may be incomplete.

---

### ❌ Ignoring rate limits

Repeated requests can result in:

```text
429 Too Many Requests
```

---

### ❌ Mixing future information with historical features

This can create:

```text
Data Leakage
```

and unrealistic model performance.

---

# 🔹 Mini Projects

## 🟢 Project 1 — Weather Data Collector

Build a Python application that:

```text
Weather API
    ↓
Collect weather data
    ↓
Validate
    ↓
Pandas DataFrame
    ↓
CSV / Parquet
```

Suggested fields:

```text
timestamp
city
temperature
humidity
pressure
wind_speed
condition
```

---

## 🟢 Project 2 — API Dataset Builder

Build a reusable collector that supports:

```text
Pagination
Retries
Rate limits
Logging
JSON storage
CSV export
```

---

## 🟡 Project 3 — API → ML Pipeline

Build:

```text
API
 ↓
Raw JSON
 ↓
Pandas
 ↓
Cleaning
 ↓
Feature Engineering
 ↓
Train/Test Split
 ↓
Scikit-Learn Model
 ↓
Prediction
```

---

## 🟡 Project 4 — Historical Data Pipeline

Collect time-series data from an API and build:

```text
API
 ↓
Historical Dataset
 ↓
Time Features
 ↓
Lag Features
 ↓
Forecasting Model
```

Pay special attention to temporal leakage.

---

## 🔴 Project 5 — Production API Data Service

Create a production-oriented system containing:

```text
API client
Retry mechanism
Rate-limit handling
Caching
Logging
Validation
Database
Feature pipeline
ML model
Monitoring
```

---

# 🔹 Exercises

### Beginner

1. What is an API?
2. What is an endpoint?
3. What is JSON?
4. What does HTTP 200 mean?
5. What does HTTP 404 mean?
6. What is an API key?
7. Why should API keys not be committed to Git?

### Intermediate

8. Write a GET request using Python.
9. Add query parameters.
10. Add request headers.
11. Parse a JSON response.
12. Convert JSON into a DataFrame.
13. Flatten nested JSON.
14. Implement pagination.
15. Add timeout handling.

### Advanced

16. Implement exponential backoff.
17. Add API response caching.
18. Design a rate-limit-aware collector.
19. Store raw API responses.
20. Build an API → ML pipeline.
21. Detect schema changes.
22. Prevent temporal data leakage.
23. Add logging and monitoring.
24. Build an automated daily ingestion pipeline.

---

# 🔹 Recommended Project Structure

A clean API data collection project can look like:

```text
06-API-Data/
│
├── README.md
│
├── api_client.py
├── config.py
├── collector.py
├── validator.py
├── transformer.py
├── pipeline.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── logs/
│
├── tests/
│   ├── test_api_client.py
│   ├── test_validator.py
│   └── test_transformer.py
│
├── .env.example
├── .gitignore
└── requirements.txt
```

---

# 🔹 End-to-End Example

The following example demonstrates a reusable API collection pattern.

```python
"""
Generic API data collection example.

This example demonstrates:
- HTTP requests
- Timeouts
- Error handling
- JSON parsing
- Basic validation
- Pandas conversion
"""

from __future__ import annotations

import requests
import pandas as pd


def fetch_records(url: str) -> list[dict]:
    """Fetch records from an API endpoint."""

    response = requests.get(
        url,
        timeout=10,
    )

    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, list):
        raise ValueError(
            "Expected API response to be a list of records."
        )

    return payload


def validate_records(
    records: list[dict],
    required_columns: set[str],
) -> None:
    """Validate that every record contains required fields."""

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Record {index} is not a JSON object."
            )

        missing = required_columns - record.keys()

        if missing:
            raise ValueError(
                f"Record {index} is missing: {sorted(missing)}"
            )


def transform_records(records: list[dict]) -> pd.DataFrame:
    """Convert API records into a DataFrame."""

    df = pd.DataFrame(records)

    return df


def main() -> None:
    url = "https://api.example.com/data"

    records = fetch_records(url)

    validate_records(
        records,
        required_columns={"id"},
    )

    df = transform_records(records)

    print(df.head())
    print(df.info())


if __name__ == "__main__":
    main()
```

> Replace the example endpoint with an actual API and adapt the response schema according to its documentation.

---

# 🔹 API → ML Workflow

A complete Machine Learning workflow using API data can be represented as:

```text
┌─────────────────────┐
│      API Source     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   API Collection    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Raw Storage      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Data Validation   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Data Cleaning     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Engineering │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Train / Validation  │
│       / Test        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ML Model Training │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Evaluation      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Deployment      │
└─────────────────────┘
```

---

# 🔹 API Data Quality Checklist

Before using API data for ML, verify:

### Source

* [ ] API provider identified
* [ ] Documentation reviewed
* [ ] Data license reviewed
* [ ] Usage limits understood

### Collection

* [ ] Timeout configured
* [ ] Authentication implemented securely
* [ ] Pagination handled
* [ ] Rate limits respected
* [ ] Retries configured where appropriate

### Data

* [ ] Schema validated
* [ ] Missing values checked
* [ ] Duplicate records checked
* [ ] Data types validated
* [ ] Range constraints checked
* [ ] Timestamps validated

### ML

* [ ] Target defined
* [ ] Features defined
* [ ] Leakage checked
* [ ] Temporal ordering preserved when required
* [ ] Train/validation/test strategy defined
* [ ] Reproducibility considered

---

# 🔹 Learning Roadmap

Follow this progression:

```text
HTTP Basics
     ↓
REST APIs
     ↓
JSON
     ↓
Python Requests
     ↓
Authentication
     ↓
Pagination
     ↓
Rate Limits
     ↓
Error Handling
     ↓
Pandas
     ↓
Data Validation
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
API Data Pipelines
     ↓
Databases
     ↓
Machine Learning
     ↓
Production ML Systems
```

---

# 🔹 Key Takeaways

> APIs turn external services into programmatically accessible data sources.

Remember:

1. **API = interface between software systems.**
2. **HTTP is the communication protocol commonly used by web APIs.**
3. **JSON is one of the most common API response formats.**
4. **GET is commonly used to retrieve data.**
5. **Authentication protects restricted APIs.**
6. **Never hard-code secrets in source code.**
7. **Always configure request timeouts.**
8. **Handle HTTP errors explicitly.**
9. **Understand pagination before collecting large datasets.**
10. **Respect API rate limits.**
11. **Validate API responses before ML processing.**
12. **Store raw data when reproducibility matters.**
13. **Track timestamps for time-dependent datasets.**
14. **Prevent temporal data leakage.**
15. **Transform API data into reproducible ML features.**

---

# 🔹 Data Collection Roadmap

This section fits into the broader Machine Learning data workflow:

```text
04-Data-Collection-and-Understanding
│
├── 01-Data-Sources
│
├── 02-CSV-Data
│
├── 03-Excel-Data
│
├── 04-JSON-Data
│
├── 05-SQL-Data
│
└── 06-API-Data
        │
        ▼
   Data Understanding
        │
        ▼
    Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
 Feature Engineering
        │
        ▼
 Machine Learning
```

API data collection connects static datasets with **dynamic, automated, real-world ML systems**.

---

# 🔹 Next Step

After understanding API-based data collection, continue with:

```text
API Data
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

The next stage is to learn how to systematically **understand the structure, quality, distributions, relationships, and limitations of collected data** before building ML models.

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

GitHub:
[https://github.com/Kishor055](https://github.com/Kishor055)

---

# 🤝 Contributing

Contributions are welcome!

If you find an issue or have an improvement:

1. Fork the repository.
2. Create a feature branch.
3. Add your changes.
4. Test your examples.
5. Commit your changes.
6. Open a Pull Request.

Please keep contributions:

* Beginner-friendly
* Technically correct
* Well documented
* Reproducible
* Consistent with the repository structure

---

# ⭐ Support

If this Machine Learning learning repository helps you:

* ⭐ Star the repository
* 🍴 Fork it
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples

Every contribution helps make the learning path better for the community.

**Happy Learning & Building! 🚀🐍🤖**

# 📚 Data Sources

> Learn where Machine Learning data comes from, how to access it, how to evaluate it, and how to choose the right source for a reliable ML project.

Data is the foundation of every Machine Learning system.

A model can use sophisticated algorithms, but if the underlying data is incomplete, biased, outdated, incorrectly collected, or poorly understood, the resulting system can still perform poorly.

This section focuses on the first major question in a Machine Learning project:

> **Where does the data come from, and can we trust it?**

---

# 📖 Table of Contents

* [What Are Data Sources?](#-what-are-data-sources)
* [Why Data Sources Matter](#-why-data-sources-matter)
* [Data Collection Lifecycle](#-data-collection-lifecycle)
* [Types of Data Sources](#-types-of-data-sources)
* [Primary Data](#-primary-data)
* [Secondary Data](#-secondary-data)
* [Structured Data](#-structured-data)
* [Semi-Structured Data](#-semi-structured-data)
* [Unstructured Data](#-unstructured-data)
* [Public Datasets](#-public-datasets)
* [Government Data](#-government-data)
* [Research Datasets](#-research-datasets)
* [Open Data Platforms](#-open-data-platforms)
* [APIs](#-apis)
* [Databases](#-databases)
* [Files](#-files)
* [Web Data](#-web-data)
* [IoT and Sensor Data](#-iot-and-sensor-data)
* [Application Data](#-application-data)
* [Social and User-Generated Data](#-social-and-user-generated-data)
* [Business Data](#-business-data)
* [Synthetic Data](#-synthetic-data)
* [Data Source Evaluation](#-data-source-evaluation)
* [Data Quality Dimensions](#-data-quality-dimensions)
* [Data Provenance](#-data-provenance)
* [Data Documentation](#-data-documentation)
* [Data Licensing](#-data-licensing)
* [Privacy and Security](#-privacy-and-security)
* [Bias and Representativeness](#-bias-and-representativeness)
* [Training vs Evaluation Data](#-training-vs-evaluation-data)
* [Data Leakage](#-data-leakage)
* [Choosing a Data Source](#-choosing-a-data-source)
* [Source Comparison](#-source-comparison)
* [Practical Workflow](#-practical-workflow)
* [Source Selection Checklist](#-source-selection-checklist)
* [Common Mistakes](#-common-mistakes)
* [Mini Projects](#-mini-projects)
* [Key Takeaways](#-key-takeaways)

---

# 🔎 What Are Data Sources?

A **data source** is the origin from which data is obtained.

Examples include:

```text
CSV Files
Databases
APIs
Websites
Sensors
Mobile Applications
Business Systems
Surveys
Government Portals
Research Publications
Public Datasets
Cloud Storage
Synthetic Generators
```

A Machine Learning project may use one source or combine multiple sources.

For example:

```text
Customer Database
        +
Website Analytics
        +
Customer Support Tickets
        +
Marketing Data
        ↓
Unified Dataset
        ↓
Machine Learning
```

---

# 🎯 Why Data Sources Matter

The source of your data affects:

* Data quality
* Data completeness
* Data freshness
* Data availability
* Data consistency
* Data reliability
* Data privacy
* Data licensing
* Data bias
* Model performance
* Reproducibility
* Deployment feasibility

A useful ML system begins with understanding these properties **before model training**.

---

# 🔄 Data Collection Lifecycle

A professional data workflow typically looks like:

```text
                 Define Problem
                      │
                      ▼
               Identify Variables
                      │
                      ▼
                Find Data Sources
                      │
                      ▼
                Evaluate Sources
                      │
                      ▼
              Collect / Acquire Data
                      │
                      ▼
                 Validate Data
                      │
                      ▼
                Store Raw Data
                      │
                      ▼
              Document Provenance
                      │
                      ▼
                 Clean Data
                      │
                      ▼
               Explore Dataset
                      │
                      ▼
            Feature Engineering
                      │
                      ▼
               Machine Learning
```

The important point is:

> **Data collection happens before data cleaning and modeling.**

---

# 🗂️ Types of Data Sources

There are several ways to classify data sources.

## Based on Collection

```text
Primary Data
Secondary Data
```

## Based on Structure

```text
Structured
Semi-Structured
Unstructured
```

## Based on Accessibility

```text
Public
Private
Restricted
Commercial
Internal
```

## Based on Origin

```text
Human Generated
Machine Generated
Application Generated
Sensor Generated
Synthetic
```

---

# 🧪 Primary Data

Primary data is collected specifically for the current research or application.

Examples:

* Surveys
* Interviews
* Experiments
* User studies
* Sensor experiments
* A/B tests
* Direct observations

### Example

Suppose you want to predict student satisfaction.

You create a survey:

```text
Student ID
Age
Course
Study Hours
Attendance
Satisfaction Score
```

The data is collected specifically for your study.

### Advantages

* Designed for the specific problem
* Control over variables
* Better understanding of collection methodology
* Potentially better alignment with project requirements

### Disadvantages

* Expensive
* Time-consuming
* Requires planning
* May require participants
* Collection errors can occur

---

# 📦 Secondary Data

Secondary data was originally collected for another purpose but is reused for your project.

Examples:

* Government datasets
* Research datasets
* Public APIs
* Existing company records
* Historical datasets
* Open-source datasets

### Example

A government publishes annual population statistics.

You use that dataset to analyze demographic trends.

### Advantages

* Faster acquisition
* Lower collection cost
* Often large-scale
* Useful for exploratory projects

### Disadvantages

* Variables may not perfectly match your problem
* Documentation may be incomplete
* Collection methodology may be unclear
* Data may be outdated
* Licensing restrictions may apply

---

# 📊 Structured Data

Structured data follows a well-defined schema.

Typical example:

| customer_id | age | income | city   | churn |
| ----------- | --: | -----: | ------ | ----: |
| 1001        |  25 |  45000 | Pune   |     0 |
| 1002        |  31 |  62000 | Mumbai |     1 |
| 1003        |  28 |  51000 | Nashik |     0 |

Common formats:

* SQL tables
* CSV
* Excel
* Relational databases

Structured data is often convenient for classical Machine Learning.

---

# 🧩 Semi-Structured Data

Semi-structured data does not follow a strict relational table structure but still contains organizational information.

Examples:

* JSON
* XML
* YAML
* Log files
* API responses

Example JSON:

```json
{
  "customer": {
    "id": 1001,
    "name": "Example User",
    "orders": [
      {
        "product": "Laptop",
        "amount": 75000
      }
    ]
  }
}
```

Semi-structured data often requires parsing and normalization before modeling.

---

# 📝 Unstructured Data

Unstructured data does not naturally follow a tabular schema.

Examples:

```text
Images
Videos
Audio
Documents
Emails
PDFs
Text
Social Media Posts
```

Different ML approaches may be required:

```text
Text
 ↓
NLP

Images
 ↓
Computer Vision

Audio
 ↓
Speech / Audio ML

Video
 ↓
Computer Vision + Temporal Analysis
```

---

# 🌐 Public Datasets

Public datasets are openly accessible datasets provided for research, education, experimentation, or public use.

Examples include datasets for:

* Healthcare research
* Climate analysis
* Finance
* Transportation
* Education
* Demographics
* Sports
* Retail
* Natural language processing
* Computer vision

Before using a public dataset, check:

```text
Source
License
Version
Collection Date
Documentation
Variables
Target Definition
Known Limitations
```

---

# 🏛️ Government Data

Government organizations publish datasets covering areas such as:

* Population
* Economics
* Transportation
* Agriculture
* Education
* Public health
* Environment
* Infrastructure

Government data can be valuable because it may cover large populations and long time periods.

However, always inspect:

* Collection methodology
* Definitions
* Update frequency
* Geographic coverage
* Missing values
* Revisions
* Metadata
* Licensing terms

---

# 🔬 Research Datasets

Research datasets are commonly produced by:

* Universities
* Research laboratories
* Scientific organizations
* Academic collaborations

They are particularly useful for:

* Academic projects
* Benchmarking
* Reproducible research
* Experimental Machine Learning

Important metadata may include:

```text
Dataset Name
Authors
Institution
Publication
Collection Method
Sample Size
Features
Labels
Preprocessing
Limitations
License
Citation
```

---

# 🌍 Open Data Platforms

Popular categories of data platforms include:

| Platform Type            | Typical Data        |
| ------------------------ | ------------------- |
| Government portals       | Public statistics   |
| Research repositories    | Scientific datasets |
| ML repositories          | Benchmark datasets  |
| Competition platforms    | ML competitions     |
| Open-source repositories | Community datasets  |
| Data marketplaces        | Commercial datasets |

When evaluating a platform, do not assume that **publicly downloadable means unrestricted for every use case**.

Always check the dataset's license and terms.

---

# 🔌 APIs

An API allows software to request data programmatically.

Typical workflow:

```text
Application
    │
    │ Request
    ▼
API Server
    │
    │ Response
    ▼
JSON / XML / Other Data
    │
    ▼
Python
    │
    ▼
Pandas
    │
    ▼
Machine Learning
```

Example conceptual request:

```python
import requests

response = requests.get(
    "https://example.com/api/data",
    timeout=30,
)

response.raise_for_status()

data = response.json()
```

> Use the API provider's documented authentication, rate limits, terms of service, and data-use policies.

---

# 🗄️ Databases

Many real-world ML systems obtain data from databases.

Common database categories:

```text
Relational
    ↓
PostgreSQL
MySQL
SQLite
SQL Server

NoSQL
    ↓
MongoDB
Redis
Cassandra
```

Typical workflow:

```text
Database
   ↓
SQL Query
   ↓
Extract Dataset
   ↓
Pandas
   ↓
Validation
   ↓
Machine Learning
```

Example:

```python
import pandas as pd
import sqlite3

connection = sqlite3.connect("example.db")

df = pd.read_sql_query(
    """
    SELECT *
    FROM customers
    """,
    connection,
)

connection.close()
```

---

# 📄 Files

Files remain one of the most common ways to exchange datasets.

## CSV

```python
import pandas as pd

df = pd.read_csv("data.csv")
```

## Excel

```python
df = pd.read_excel("data.xlsx")
```

## JSON

```python
df = pd.read_json("data.json")
```

## Parquet

```python
df = pd.read_parquet("data.parquet")
```

Different formats have different trade-offs in:

* Size
* Speed
* Schema
* Compatibility
* Nested data support
* Compression

---

# 🌐 Web Data

Websites can contain valuable information, but collecting web data requires additional considerations.

Potential approaches include:

```text
Official API
    ↓
Structured Download
    ↓
Web Scraping
```

Prefer an official API or downloadable dataset when one is available.

Before collecting website data, consider:

* Terms of service
* Robots directives
* Copyright
* Privacy
* Rate limits
* Authentication
* Data licensing
* Website stability

Avoid treating arbitrary website content as automatically free to collect or reuse.

---

# 🤖 IoT and Sensor Data

Modern systems generate huge volumes of machine-generated data.

Examples:

```text
Temperature Sensors
GPS
Accelerometers
Smart Watches
Industrial Equipment
Cameras
Smart Homes
Vehicles
```

A sensor dataset may look like:

| timestamp | sensor_id | temperature | pressure |
| --------- | --------- | ----------: | -------: |
| 10:00     | S01       |        27.4 |     1012 |
| 10:01     | S01       |        27.6 |     1013 |
| 10:02     | S01       |        27.8 |     1012 |

Important considerations:

* Sampling frequency
* Missing readings
* Sensor drift
* Calibration
* Time synchronization
* Device failures
* Outliers

---

# 📱 Application Data

Applications generate data through user interactions.

Examples:

```text
Logins
Clicks
Purchases
Searches
Page Views
Session Duration
Feature Usage
Errors
Transactions
```

This information can be used for:

* Recommendation systems
* Churn prediction
* Fraud detection
* Personalization
* Product analytics

However, user-generated data may contain sensitive information and must be handled according to applicable privacy and organizational requirements.

---

# 👥 Social and User-Generated Data

Examples include:

* Posts
* Comments
* Reviews
* Ratings
* Discussions
* Images
* Videos

Potential ML applications:

```text
Sentiment Analysis
Topic Classification
Recommendation
Trend Analysis
Content Moderation
```

Important considerations include:

* Platform terms
* Privacy
* Consent
* Copyright
* Sampling bias
* Bot activity
* Deleted content
* Representativeness

---

# 🏢 Business Data

Organizations often have valuable internal data.

Examples:

```text
CRM
ERP
Sales
Finance
Inventory
Customer Support
Marketing
Website Analytics
Transactions
```

A company might combine:

```text
CRM
 +
Transactions
 +
Support Tickets
 +
Website Events
        ↓
Customer 360 Dataset
```

Business data often requires access controls and strict governance.

---

# 🧬 Synthetic Data

Synthetic data is artificially generated data designed to resemble certain properties of real data.

It can be generated using:

* Statistical distributions
* Simulation
* Generative models
* Rule-based systems

Example:

```python
import numpy as np

rng = np.random.default_rng(42)

ages = rng.integers(
    low=18,
    high=70,
    size=1000,
)
```

Synthetic data can be useful for:

* Prototyping
* Testing
* Demonstrations
* Algorithm development
* Privacy-sensitive development scenarios

However:

> Synthetic data is not automatically representative of the real-world population.

It should be validated against the intended use case.

---

# 🔍 Data Source Evaluation

Finding data is not enough.

You must determine whether the source is appropriate.

A useful evaluation framework is:

```text
                 Data Source
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    Quality       Reliability    Relevance
       │             │             │
       └─────────────┼─────────────┘
                     ▼
                Accessibility
                     │
                     ▼
                  License
                     │
                     ▼
                  Privacy
                     │
                     ▼
               Representativeness
```

---

# 📏 Data Quality Dimensions

Evaluate data using multiple dimensions.

## 1. Accuracy

Does the data correctly represent the underlying reality?

Example:

```text
Actual Age: 25
Recorded Age: 250
```

The record may be invalid.

---

## 2. Completeness

How much expected information is present?

```text
Total Records: 100,000
Missing Customer IDs: 15,000
```

This may indicate a serious quality issue depending on the use case.

---

## 3. Consistency

Are values represented consistently?

Example:

```text
Mumbai
mumbai
MUMBAI
Bombay
```

These may refer to the same location but are represented differently.

---

## 4. Timeliness

How recent is the data?

A dataset collected ten years ago may be unsuitable for a rapidly changing problem.

---

## 5. Validity

Do values follow expected rules?

Example:

```text
Age = -10
```

is generally invalid for a human-age field.

---

## 6. Uniqueness

Are records duplicated?

```text
Customer ID 1001
Customer ID 1001
```

Duplicate records may distort analysis.

---

# 🧾 Data Provenance

**Data provenance** describes where data came from and what happened to it.

A useful provenance record might contain:

```text
Source
Source URL / Identifier
Dataset Name
Version
Collection Date
Download Date
Provider
Transformation Steps
Filtering
Feature Engineering
Known Limitations
License
```

Example:

```yaml
dataset:
  name: customer_transactions
  version: "2026-01"
  source: internal_database
  collected_at: "2026-01-15"
  owner: analytics_team

transformations:
  - removed_duplicate_transactions
  - standardized_currency
  - parsed_timestamps
```

Good provenance improves:

* Reproducibility
* Debugging
* Auditing
* Collaboration
* Model governance

---

# 📖 Data Documentation

Before modeling, create a **data dictionary**.

Example:

| Column      | Type     | Description                | Missing Allowed |
| ----------- | -------- | -------------------------- | --------------- |
| customer_id | integer  | Unique customer identifier | No              |
| age         | integer  | Customer age               | Yes             |
| income      | float    | Annual income              | Yes             |
| city        | category | Customer city              | Yes             |
| churn       | integer  | Churn indicator            | No              |

A good dataset should ideally document:

* Column meanings
* Units
* Allowed values
* Data types
* Missing-value meanings
* Collection methodology
* Target definition
* Sampling methodology
* Known limitations

---

# ⚖️ Data Licensing

Before using external data, determine what the license permits.

Potential licensing models include:

```text
Public Domain
Open Data Licenses
Creative Commons
Research-Specific Licenses
Commercial Licenses
Proprietary Licenses
```

Check:

* Commercial-use permissions
* Attribution requirements
* Redistribution rules
* Modification rules
* Derivative-work requirements
* Dataset-specific restrictions

> A dataset being publicly downloadable does not necessarily mean it can be freely redistributed or commercially used.

---

# 🔐 Privacy and Security

Data collection may involve personal or sensitive information.

Examples:

```text
Name
Email
Phone Number
Address
Location
Financial Information
Authentication Data
Health Information
Biometric Data
```

Before collecting or using such data:

* Determine whether collection is lawful and appropriate
* Minimize unnecessary personal information
* Apply access controls
* Protect credentials and secrets
* Encrypt data where appropriate
* Follow applicable privacy requirements
* Define retention policies
* Remove or protect identifiers where appropriate

### Data Minimization

Only collect information that is actually required.

Instead of:

```text
Full Name
Exact Address
Phone
Email
Date of Birth
```

a model may only require:

```text
Age Group
City
Transaction Count
Average Order Value
```

depending on the use case.

---

# ⚠️ Bias and Representativeness

A dataset may not represent the population you ultimately care about.

Example:

```text
Target Population
        │
        ▼
Eligible Population
        │
        ▼
Collected Population
        │
        ▼
Available Dataset
```

Each step can introduce selection effects.

Potential sources of bias include:

* Sampling bias
* Geographic bias
* Demographic imbalance
* Survivorship bias
* Historical bias
* Measurement bias
* Labeling bias
* Selection bias

Ask:

> **Who is represented in this dataset, and who is missing?**

---

# 🧪 Training vs Evaluation Data

A data source may contain information that is appropriate for analysis but not necessarily appropriate for model training.

A typical workflow is:

```text
Collected Data
      │
      ▼
Initial Validation
      │
      ▼
Train / Validation / Test Strategy
      │
      ├───────────────┐
      ▼               ▼
 Training          Evaluation
 Data              Data
```

Evaluation data should represent the intended deployment conditions as closely as practical.

---

# 🚨 Data Leakage

Data leakage occurs when information unavailable at prediction time influences the model.

Example:

Suppose we want to predict whether a customer will cancel a subscription.

A feature called:

```text
cancellation_date
```

would reveal the future outcome if it is populated after cancellation.

Using it for prediction would produce misleadingly strong results.

---

# 🕒 Time-Based Leakage

Time-dependent problems require special attention.

Suppose:

```text
January → February → March → April
```

You want to predict April outcomes.

Using future information from April to construct January training features would be invalid.

A safer structure is:

```text
Past
 │
 ▼
Training Data
 │
 ▼
Validation Period
 │
 ▼
Future Test Period
```

Time-aware data collection and feature construction are essential for forecasting systems.

---

# 🎯 Choosing a Data Source

Use the following questions.

## Problem Relevance

* Does the source contain the required target?
* Does it contain useful predictors?
* Does the population match the problem?

## Quality

* Are values accurate?
* Are missing values documented?
* Are duplicates present?
* Are data types consistent?

## Freshness

* When was it collected?
* How often is it updated?

## Scale

* Number of records?
* Number of features?
* Historical coverage?

## Accessibility

* API?
* Download?
* Database?
* Authentication?
* Rate limits?

## Legal

* License?
* Attribution?
* Redistribution?
* Commercial-use restrictions?

## Privacy

* Personal information?
* Sensitive information?
* Required protections?

---

# 📊 Source Comparison

| Source           | Scale        | Freshness  | Control    | Cost        | Typical Use           |
| ---------------- | ------------ | ---------- | ---------- | ----------- | --------------------- |
| Public Dataset   | Medium–Large | Variable   | Low        | Low         | Learning / Research   |
| Government Data  | Large        | Variable   | Low        | Low         | Public Analysis       |
| Research Dataset | Variable     | Variable   | Low        | Low–Medium  | Research              |
| API              | Variable     | Often High | Medium     | Variable    | Applications          |
| Database         | Large        | High       | High       | Variable    | Business ML           |
| Sensor Data      | Very Large   | Very High  | High       | Medium–High | IoT                   |
| Survey           | Variable     | High       | High       | Medium      | Primary Research      |
| Synthetic Data   | Configurable | Generated  | High       | Low–Medium  | Testing / Prototyping |
| Web Data         | Variable     | Variable   | Low–Medium | Variable    | Research / Analytics  |

The "best" source depends on the project requirements rather than the source type alone.

---

# 🔬 Practical Workflow

A professional data-source workflow can be implemented as follows.

## Step 1 — Define the Problem

```text
What are we trying to predict,
classify, estimate, or understand?
```

---

## Step 2 — Define the Unit of Observation

Determine what one row represents.

Examples:

```text
One customer
One transaction
One patient visit
One image
One sensor reading
One website session
```

This is one of the most important decisions in dataset design.

---

## Step 3 — Define Required Variables

Create an initial data requirement:

```text
Target:
    customer_churn

Features:
    age
    tenure
    monthly_spend
    support_tickets
    subscription_type
```

---

## Step 4 — Identify Candidate Sources

Search for:

```text
Internal databases
Public datasets
Government sources
Research repositories
APIs
Surveys
Sensors
Existing applications
```

---

## Step 5 — Evaluate Each Source

Record:

```text
Source
Coverage
Quality
Freshness
License
Privacy
Accessibility
Known Bias
Limitations
```

---

## Step 6 — Acquire Raw Data

Keep the original data unchanged whenever possible.

```text
data/
├── raw/
├── interim/
└── processed/
```

---

## Step 7 — Validate

Check:

```text
Schema
Rows
Columns
Missing Values
Duplicates
Types
Ranges
Target
Time Coverage
```

---

## Step 8 — Document

Record:

```text
Where did it come from?
When was it collected?
Which version was used?
What transformations were performed?
What limitations exist?
```

---

# 🗃️ Recommended Data Directory

For ML projects:

```text
data/
│
├── raw/
│   └── original_dataset.csv
│
├── external/
│   └── external_source.csv
│
├── interim/
│   └── cleaned_intermediate.csv
│
└── processed/
    └── model_ready.csv
```

### Important Principle

```text
RAW DATA
    ↓
Never overwrite
    ↓
Transform
    ↓
INTERMEDIATE DATA
    ↓
Transform
    ↓
PROCESSED DATA
```

Keeping raw data immutable makes experiments easier to reproduce.

---

# 🧪 Source Metadata Template

Use a metadata document for every important dataset.

```yaml
dataset:
  name: ""
  description: ""
  version: ""

source:
  provider: ""
  location: ""
  access_method: ""

collection:
  collected_at: ""
  downloaded_at: ""
  geographic_scope: ""
  population: ""

schema:
  rows: 0
  columns: 0

quality:
  missing_values: ""
  duplicates: ""
  known_issues: ""

license:
  name: ""
  restrictions: ""

privacy:
  personal_data: false
  sensitive_data: false

limitations:
  - ""

notes:
  - ""
```

---

# ✅ Source Selection Checklist

Before using a dataset, verify:

### Source

* [ ] Provider identified
* [ ] Source location documented
* [ ] Dataset version recorded
* [ ] Download/access date recorded

### Relevance

* [ ] Dataset matches the ML problem
* [ ] Target is available
* [ ] Features are relevant
* [ ] Unit of observation is understood

### Quality

* [ ] Missing values inspected
* [ ] Duplicates inspected
* [ ] Data types inspected
* [ ] Invalid values checked
* [ ] Outliers investigated

### Freshness

* [ ] Collection date known
* [ ] Update frequency known
* [ ] Historical coverage understood

### Legal

* [ ] License identified
* [ ] Intended usage permitted
* [ ] Attribution requirements checked

### Privacy

* [ ] Personal data identified
* [ ] Sensitive information identified
* [ ] Access controls considered
* [ ] Data minimization considered

### Bias

* [ ] Population identified
* [ ] Sampling method understood
* [ ] Representation assessed
* [ ] Known limitations documented

---

# ❌ Common Mistakes

## 1. Choosing Data Because It Is Large

More rows do not automatically mean better data.

A smaller, relevant, high-quality dataset may be more useful.

---

## 2. Ignoring the Data Collection Process

Two datasets with identical columns can have very different quality depending on how the data was collected.

---

## 3. Ignoring Dataset Versions

External datasets can change.

Record:

```text
Dataset Version
Download Date
Source
```

---

## 4. Overwriting Raw Data

Avoid:

```text
raw.csv
    ↓
clean raw.csv
```

Prefer:

```text
raw/
    original.csv

processed/
    cleaned.csv
```

---

## 5. Ignoring Licensing

Never assume:

> "I can download it, therefore I can use it for anything."

Check the actual license and terms.

---

## 6. Ignoring Bias

A dataset may contain systematic representation problems.

Always ask:

> Who is missing?

---

## 7. Collecting Unnecessary Personal Data

Collect only what is required for the intended purpose.

---

## 8. Using Future Information

Especially in:

* Forecasting
* Finance
* Healthcare
* Customer churn
* Predictive maintenance
* Recommendation systems

Always ensure features are available at the prediction timestamp.

---

# 🧩 Mini Projects

## Project 1 — Dataset Source Audit

Choose a public dataset and document:

```text
Source
Provider
Version
License
Collection Method
Features
Target
Missing Values
Known Limitations
Bias Risks
```

---

## Project 2 — Multi-Source Dataset

Combine:

```text
CSV
+
API
+
Database
```

Create a unified dataset.

Document:

* Source identifiers
* Join keys
* Conflicting columns
* Missing records
* Transformation steps

---

## Project 3 — Data Quality Report

Create an automated report containing:

```text
Dataset Shape
Data Types
Missing Values
Duplicate Count
Unique Values
Numeric Statistics
Categorical Statistics
Potential Outliers
```

---

## Project 4 — Data Provenance Pipeline

Build:

```text
Source
 ↓
Download
 ↓
Raw Storage
 ↓
Validation
 ↓
Metadata
 ↓
Processed Dataset
```

Store the metadata alongside the dataset.

---

## Project 5 — Time-Aware Dataset

Collect historical data and construct:

```text
Training Period
Validation Period
Test Period
```

Make sure future observations do not influence past features.

---

# 🧠 Source Selection Decision Tree

```text
Do you already have internal data?
        │
       Yes
        │
        ▼
Can it legally and appropriately be used?
        │
   ┌────┴────┐
  Yes        No
   │          │
   ▼          ▼
Evaluate    Find another
quality     permitted source
   │
   ▼
Is the data sufficient?
   │
 ┌─┴─┐
Yes  No
 │    │
 ▼    ▼
Use  Supplement
     with another source
```

If no internal data exists:

```text
Public Dataset
      │
      ├── Government
      ├── Research
      ├── Open Data
      └── Competition Dataset
```

Or:

```text
API
Database
Survey
Sensor
Application Logs
Synthetic Data
```

---

# 📈 From Data Source to Machine Learning

The complete relationship is:

```text
                 DATA SOURCES
                      │
      ┌───────────────┼────────────────┐
      ▼               ▼                ▼
   Files            APIs           Databases
      │               │                │
      └───────────────┼────────────────┘
                      ▼
                Raw Dataset
                      │
                      ▼
               Data Validation
                      │
                      ▼
              Data Understanding
                      │
                      ▼
                 Data Cleaning
                      │
                      ▼
             Exploratory Analysis
                      │
                      ▼
              Feature Engineering
                      │
                      ▼
                ML Dataset
                      │
                      ▼
                Model Training
```

---

# 🏗️ Professional Data Engineering Principles

For serious Machine Learning projects:

### 1. Preserve Raw Data

Never destroy the original source unnecessarily.

### 2. Track Versions

Data changes over time.

### 3. Record Provenance

Know exactly where every dataset came from.

### 4. Validate Automatically

Do not rely only on manual inspection.

### 5. Separate Raw and Processed Data

Keep transformation stages clear.

### 6. Document Assumptions

Write down:

* Sampling assumptions
* Filtering rules
* Feature definitions
* Missing-value decisions
* Target definitions

### 7. Protect Sensitive Data

Apply appropriate security and privacy controls.

### 8. Reproduce the Dataset

Another developer should be able to understand how the model dataset was created.

---

# 🧭 Learning Roadmap

This section should be learned in the following progression:

```text
01
Understand Data Sources
        ↓
02
Identify Primary vs Secondary Data
        ↓
03
Understand Structured Data
        ↓
04
Understand Semi-Structured Data
        ↓
05
Understand Unstructured Data
        ↓
06
Explore Public & Research Datasets
        ↓
07
Understand APIs & Databases
        ↓
08
Evaluate Data Quality
        ↓
09
Understand Provenance
        ↓
10
Understand Licensing & Privacy
        ↓
11
Identify Bias & Representation Issues
        ↓
12
Select Appropriate Sources
        ↓
13
Acquire Raw Data
        ↓
14
Validate Data
        ↓
15
Move to Data Understanding
```

---

# 🔗 Connection With the Next Sections

Data Sources are only the first step.

The broader workflow becomes:

```text
04-Data-Collection-and-Understanding/
│
├── 01-Data-Sources/
│       │
│       ▼
├── 02-Data-Collection/
│       │
│       ▼
├── 03-Data-Understanding/
│       │
│       ▼
├── 04-Data-Quality/
│       │
│       ▼
├── 05-Exploratory-Data-Analysis/
│       │
│       ▼
└── 06-Data-Documentation/
```

This creates the foundation for the later:

```text
Data Preprocessing
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Model Evaluation
        ↓
Deployment
```

---

# 🏆 Key Takeaways

Remember these principles:

1. **Data is the foundation of Machine Learning.**
2. A dataset is only useful when it is relevant to the problem.
3. Large datasets are not automatically high-quality datasets.
4. Understand how data was collected.
5. Always inspect data quality.
6. Document dataset versions and provenance.
7. Check licenses before using external data.
8. Protect personal and sensitive information.
9. Consider whether the dataset represents the intended population.
10. Preserve raw data whenever possible.
11. Avoid future information entering historical predictions.
12. Treat data collection as an engineering process, not just a download step.
13. Build reproducible data acquisition and validation workflows.
14. Understand the dataset before selecting a Machine Learning algorithm.

---

# 📌 Final Principle

> **A Machine Learning model can only learn from the information you provide it. The quality, relevance, provenance, and representativeness of that information are therefore fundamental parts of the ML system itself.**

The objective of this section is not simply to **find data**.

It is to learn how to answer:

```text
Where did this data come from?
        ↓
Why was it collected?
        ↓
Who does it represent?
        ↓
How reliable is it?
        ↓
Can I legally and appropriately use it?
        ↓
What limitations does it have?
        ↓
Can I reproduce the dataset?
        ↓
Is it suitable for my Machine Learning problem?
```

Once those questions have clear answers, you have a much stronger foundation for the next stage:

**Data Collection → Data Understanding → Data Quality → Exploratory Data Analysis → Machine Learning.**

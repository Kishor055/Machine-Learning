
# 🕷️ Web Scraping for Machine Learning

Web scraping is the process of **programmatically collecting information from websites**.

For Machine Learning and Data Science, web scraping can transform publicly accessible web content into structured datasets for:

* 📊 Exploratory Data Analysis
* 🤖 Machine Learning
* 🧠 Natural Language Processing
* 🔎 Search and recommendation systems
* 📈 Market research
* 📰 News analysis
* 🛒 Product analysis
* 📚 Research
* 🗺️ Location intelligence
* 📉 Trend analysis

A typical workflow is:

```text
Website
   ↓
HTTP Request
   ↓
HTML
   ↓
HTML Parsing
   ↓
Data Extraction
   ↓
Cleaning
   ↓
Validation
   ↓
Structured Dataset
   ↓
Feature Engineering
   ↓
Machine Learning
```

This chapter focuses on **responsible, reproducible, and technically robust web data collection**.

---

# 📚 Table of Contents

1. [What Is Web Scraping?](#-what-is-web-scraping)
2. [Web Scraping vs Web Crawling](#-web-scraping-vs-web-crawling)
3. [Why Web Scraping Matters for ML](#-why-web-scraping-matters-for-ml)
4. [How the Web Works](#-how-the-web-works)
5. [HTML Fundamentals](#-html-fundamentals)
6. [HTML Elements](#-html-elements)
7. [Attributes](#-attributes)
8. [DOM](#-dom)
9. [CSS Selectors](#-css-selectors)
10. [XPath](#-xpath)
11. [Static vs Dynamic Websites](#-static-vs-dynamic-websites)
12. [HTTP Requests](#-http-requests)
13. [Python Requests](#-python-requests)
14. [User-Agent](#-user-agent)
15. [BeautifulSoup](#-beautifulsoup)
16. [Selecting Elements](#-selecting-elements)
17. [Extracting Text](#-extracting-text)
18. [Extracting Links](#-extracting-links)
19. [Extracting Images](#-extracting-images)
20. [Extracting Tables](#-extracting-tables)
21. [Cleaning Scraped Text](#-cleaning-scraped-text)
22. [Handling Missing Data](#-handling-missing-data)
23. [Pagination](#-pagination)
24. [Multiple Pages](#-multiple-pages)
25. [Sessions](#-sessions)
26. [Timeouts](#-timeouts)
27. [Error Handling](#-error-handling)
28. [Retries](#-retries)
29. [Rate Limiting](#-rate-limiting)
30. [robots.txt](#-robotstxt)
31. [Terms and Responsible Scraping](#-terms-and-responsible-scraping)
32. [Dynamic JavaScript Websites](#-dynamic-javascript-websites)
33. [Selenium](#-selenium)
34. [Playwright](#-playwright)
35. [When Not to Use Browser Automation](#-when-not-to-use-browser-automation)
36. [Web Scraping with Pandas](#-web-scraping-with-pandas)
37. [Scraping Tables](#-scraping-tables)
38. [Nested HTML](#-nested-html)
39. [Data Validation](#-data-validation)
40. [Data Cleaning](#-data-cleaning)
41. [Feature Engineering](#-feature-engineering)
42. [Text Data for NLP](#-text-data-for-nlp)
43. [Avoiding Data Leakage](#-avoiding-data-leakage)
44. [Data Provenance](#-data-provenance)
45. [Storage Formats](#-storage-formats)
46. [Scraping Pipeline](#-scraping-pipeline)
47. [Caching](#-caching)
48. [Logging](#-logging)
49. [Testing](#-testing)
50. [Production Architecture](#-production-architecture)
51. [Common Mistakes](#-common-mistakes)
52. [Mini Projects](#-mini-projects)
53. [Exercises](#-exercises)
54. [Project Structure](#-project-structure)
55. [End-to-End Example](#-end-to-end-example)
56. [ML Workflow](#-ml-workflow)
57. [Best Practices](#-best-practices)
58. [Learning Roadmap](#-learning-roadmap)
59. [Key Takeaways](#-key-takeaways)
60. [Next Step](#-next-step)

---

# 🔹 What Is Web Scraping?

Web scraping is the automated extraction of information from web pages.

For example, a webpage may visually display:

```text
Product
Laptop
Price
₹75,000
Rating
4.5
```

The underlying HTML might contain:

```html
<div class="product">
    <h2>Laptop</h2>
    <span class="price">₹75,000</span>
    <span class="rating">4.5</span>
</div>
```

A scraper converts the HTML into structured data:

```text
product | price | rating
--------|-------|-------
Laptop  | 75000 | 4.5
```

That dataset can then be analyzed or used in an ML pipeline.

---

# 🔹 Web Scraping vs Web Crawling

These terms are related but different.

### Web Scraping

Focuses on **extracting specific information**.

```text
Website
   ↓
Find product cards
   ↓
Extract names/prices/ratings
```

### Web Crawling

Focuses on **discovering and visiting URLs**.

```text
Homepage
   ↓
Links
   ↓
Page 2
   ↓
Page 3
   ↓
Page 4
```

A real system may combine both:

```text
Crawler
   ↓
Discover URLs
   ↓
Scraper
   ↓
Extract data
```

---

# 🔹 Why Web Scraping Matters for ML

Machine Learning depends on data.

Websites can provide large amounts of:

* Text
* Images
* Prices
* Reviews
* Ratings
* Product metadata
* Public statistics
* Articles
* Historical information
* Structured tables

Examples:

### NLP

```text
Articles
   ↓
Scraping
   ↓
Text Dataset
   ↓
NLP
   ↓
Classification
```

### Recommendation Systems

```text
Products
   ↓
Names
Descriptions
Categories
Ratings
   ↓
Recommendation Model
```

### Price Analysis

```text
Product
   ↓
Historical Prices
   ↓
Time-Series Dataset
   ↓
Forecasting
```

---

# 🔹 How the Web Works

A simplified browser workflow:

```text
User
 ↓
Browser
 ↓
DNS
 ↓
Web Server
 ↓
HTTP Request
 ↓
HTTP Response
 ↓
HTML
 ↓
Browser Rendering
```

A scraper often performs:

```text
Python
 ↓
HTTP Request
 ↓
Server
 ↓
HTML Response
 ↓
Parser
 ↓
Extracted Data
```

---

# 🔹 HTML Fundamentals

HTML stands for **HyperText Markup Language**.

Example:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Example</title>
</head>
<body>

    <h1>Hello World</h1>

    <p>This is a paragraph.</p>

</body>
</html>
```

A scraper needs to understand HTML structure to identify the desired elements.

---

# 🔹 HTML Elements

Common elements:

```html
<h1>Heading</h1>

<p>Paragraph</p>

<a href="/products">Products</a>

<img src="image.jpg" alt="Product">

<table>
    ...
</table>

<div>
    ...
</div>
```

Frequently scraped elements include:

```text
h1
h2
h3
p
a
img
span
div
li
table
```

---

# 🔹 Attributes

HTML elements may contain attributes.

Example:

```html
<a
    href="/products/123"
    class="product-link"
>
    Laptop
</a>
```

Important attributes:

```text
class
id
href
src
alt
data-*
```

Attributes often provide useful identifiers for scraping.

---

# 🔹 DOM

The browser represents HTML as a **Document Object Model (DOM)**.

Example:

```text
html
 ├── head
 │    └── title
 │
 └── body
      ├── h1
      ├── p
      └── div
           ├── h2
           └── span
```

Understanding the DOM makes it easier to locate elements reliably.

---

# 🔹 CSS Selectors

CSS selectors are commonly used to locate HTML elements.

Examples:

```css
p
```

Selects all paragraphs.

```css
.product
```

Selects elements with class `product`.

```css
#price
```

Selects the element with ID `price`.

```css
.product .price
```

Selects `.price` elements inside `.product`.

---

# 🔹 XPath

XPath provides another way to locate elements.

Example:

```xpath
//div[@class='product']
```

Another:

```xpath
//h2[contains(@class, 'title')]
```

XPath is particularly useful when the HTML structure requires more precise navigation.

---

# 🔹 Static vs Dynamic Websites

One of the most important concepts in web scraping is understanding how content is generated.

## Static Website

HTML already contains the data:

```text
Request
 ↓
HTML
 ↓
Data available
```

Tools such as:

```text
requests
BeautifulSoup
```

are often sufficient.

## Dynamic Website

The initial HTML may not contain all displayed data.

Instead:

```text
Request
 ↓
Initial HTML
 ↓
JavaScript
 ↓
API Requests
 ↓
Rendered Content
```

Browser automation may be required, or the underlying data endpoint may be a better source when legitimately accessible.

---

# 🔹 HTTP Requests

Python can request a webpage directly.

Install:

```bash
pip install requests
```

Example:

```python
import requests

url = "https://example.com"

response = requests.get(
    url,
    timeout=10,
)

print(response.status_code)
print(response.text[:500])
```

---

# 🔹 User-Agent

Some servers behave differently depending on the client.

A scraper can provide a descriptive User-Agent:

```python
import requests

headers = {
    "User-Agent": "ml-research-scraper/1.0"
}

response = requests.get(
    "https://example.com",
    headers=headers,
    timeout=10,
)

response.raise_for_status()
```

Do not use a User-Agent as a mechanism to evade access controls.

---

# 🔹 BeautifulSoup

BeautifulSoup is one of the most popular Python HTML parsers.

Install:

```bash
pip install beautifulsoup4
```

Example:

```python
import requests
from bs4 import BeautifulSoup

url = "https://example.com"

response = requests.get(
    url,
    timeout=10,
)

response.raise_for_status()

soup = BeautifulSoup(
    response.text,
    "html.parser",
)

print(soup.title.get_text(strip=True))
```

---

# 🔹 Selecting Elements

### Find one element

```python
heading = soup.find("h1")

if heading:
    print(heading.get_text(strip=True))
```

### Find multiple elements

```python
links = soup.find_all("a")

for link in links:
    print(link.get_text(strip=True))
```

### CSS selector

```python
products = soup.select(".product")
```

---

# 🔹 Extracting Text

Use:

```python
text = element.get_text(
    " ",
    strip=True,
)
```

Example:

```python
paragraphs = soup.find_all("p")

for paragraph in paragraphs:
    text = paragraph.get_text(
        " ",
        strip=True,
    )
    print(text)
```

The separator `" "` helps prevent words from being accidentally concatenated when nested elements are present.

---

# 🔹 Extracting Links

Links typically use the `href` attribute.

```python
for link in soup.find_all("a"):
    href = link.get("href")

    if href:
        print(href)
```

Extracting both text and URL:

```python
for link in soup.find_all("a"):
    text = link.get_text(" ", strip=True)
    href = link.get("href")

    print({
        "text": text,
        "url": href,
    })
```

---

# 🔹 Extracting Images

Images commonly use `src` and `alt`.

```python
for image in soup.find_all("img"):
    src = image.get("src")
    alt = image.get("alt")

    print({
        "src": src,
        "alt": alt,
    })
```

For image-based ML projects, image URLs may then be collected and processed according to the site's permissions and applicable terms.

---

# 🔹 Extracting Tables

Pandas can sometimes read HTML tables directly.

```python
import pandas as pd

tables = pd.read_html(
    "https://example.com/table"
)

for table in tables:
    print(table.head())
```

This is convenient when the table is genuinely represented as an HTML table.

---

# 🔹 Cleaning Scraped Text

Raw web text often contains:

```text
Extra whitespace
Newlines
Tabs
HTML entities
Navigation text
Advertisements
Repeated content
```

Example:

```python
import re


def clean_text(text: str) -> str:
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()
```

Example:

```python
raw_text = """
    Machine Learning

    is powerful.
"""

print(clean_text(raw_text))
```

Result:

```text
Machine Learning is powerful.
```

---

# 🔹 Handling Missing Data

Web pages frequently have incomplete records.

Example:

```python
price_element = card.select_one(".price")

price = (
    price_element.get_text(strip=True)
    if price_element
    else None
)
```

Never assume that every page has identical HTML.

---

# 🔹 Pagination

Many websites split content into pages:

```text
/products?page=1
/products?page=2
/products?page=3
```

A simple collector:

```python
import requests
from bs4 import BeautifulSoup


def scrape_pages(base_url: str, pages: int) -> list[dict]:
    records = []

    for page in range(1, pages + 1):
        response = requests.get(
            base_url,
            params={"page": page},
            timeout=10,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for card in soup.select(".product"):
            name_element = card.select_one(".name")
            price_element = card.select_one(".price")

            records.append({
                "name": (
                    name_element.get_text(strip=True)
                    if name_element
                    else None
                ),
                "price": (
                    price_element.get_text(strip=True)
                    if price_element
                    else None
                ),
            })

    return records
```

For a real website, selectors must be adapted to its HTML.

---

# 🔹 Multiple Pages

A robust scraper should separate:

```text
URL discovery
      ↓
Page fetching
      ↓
HTML parsing
      ↓
Record extraction
      ↓
Validation
```

This separation makes the code easier to test.

---

# 🔹 Sessions

Use `requests.Session()` when making multiple requests.

```python
import requests

session = requests.Session()

session.headers.update({
    "User-Agent": "ml-research-scraper/1.0",
})

response = session.get(
    "https://example.com",
    timeout=10,
)

response.raise_for_status()
```

A session can reuse connections and maintain cookies when appropriate.

---

# 🔹 Timeouts

Never rely on an unlimited network wait.

Bad:

```python
requests.get(url)
```

Better:

```python
requests.get(
    url,
    timeout=10,
)
```

For finer control:

```python
requests.get(
    url,
    timeout=(5, 20),
)
```

Here:

```text
5 seconds  → connection timeout
20 seconds → read timeout
```

---

# 🔹 Error Handling

Network requests can fail because of:

* Connection errors
* DNS problems
* Timeouts
* HTTP errors
* Server failures
* Invalid content

Example:

```python
import requests


def fetch_page(url: str) -> str:
    try:
        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        return response.text

    except requests.Timeout as exc:
        raise RuntimeError(
            f"Timeout while fetching {url}"
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Request failed for {url}"
        ) from exc
```

---

# 🔹 Retries

Temporary failures may succeed if retried after a delay.

A retry strategy can use:

```text
Attempt 1
   ↓
Failure
   ↓
Wait
   ↓
Attempt 2
   ↓
Failure
   ↓
Longer Wait
   ↓
Attempt 3
```

Example using `requests` adapters:

```python
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=[
            "GET",
            "HEAD",
        ],
    )

    adapter = HTTPAdapter(
        max_retries=retry,
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    return session
```

Retries should be limited and should not be used to circumvent intentional access restrictions.

---

# 🔹 Rate Limiting

Do not send requests as quickly as possible.

A responsible scraper should control its request rate.

Example:

```python
import time

time.sleep(1)
```

For larger projects, implement a configurable delay:

```python
REQUEST_DELAY = 1.0
```

Then:

```python
time.sleep(REQUEST_DELAY)
```

Better approaches may use:

* Fixed delays
* Exponential backoff
* Server-provided retry information
* Adaptive request scheduling

---

# 🔹 robots.txt

Many websites publish a:

```text
/robots.txt
```

file describing crawler-related rules.

For example:

```text
https://example.com/robots.txt
```

Before automated collection, inspect the site's crawler guidance and applicable terms.

`robots.txt` is not a universal legal permission system, so it should be considered alongside:

* Terms of service
* Copyright
* Privacy obligations
* Applicable laws
* Authentication/access controls
* Data licensing

---

# 🔹 Terms and Responsible Scraping

Web scraping should be performed responsibly.

Before collecting data, consider:

* Is the information publicly accessible?
* Does the site provide an API?
* Does the API provide the needed data?
* What do the site's terms say?
* Are there usage restrictions?
* Is personal information involved?
* Is authentication required?
* Are there technical access controls?
* Is the dataset licensed for your intended use?

### Prefer official APIs

If a website provides an official API for the data you need, it is often preferable because it provides:

```text
Structured data
Stable schemas
Authentication
Rate limits
Documentation
```

### Do not bypass access controls

Do not use scraping as a method to defeat:

* Authentication
* CAPTCHAs
* Paywalls
* Technical access restrictions
* Security mechanisms

---

# 🔹 Dynamic JavaScript Websites

Some pages display content generated by JavaScript.

For example:

```text
Initial HTML
     ↓
JavaScript executes
     ↓
API request
     ↓
Data received
     ↓
DOM updated
```

If `requests.get()` does not contain the displayed content, inspect the page architecture.

Possible approaches:

```text
Official API
     ↓
Preferred when available

Public data endpoint
     ↓
Use only where legitimately accessible

Browser automation
     ↓
When rendering is genuinely required
```

---

# 🔹 Selenium

Selenium automates browsers.

Install:

```bash
pip install selenium
```

Basic example:

```python
from selenium import webdriver
from selenium.webdriver.common.by import By


driver = webdriver.Chrome()

try:
    driver.get("https://example.com")

    heading = driver.find_element(
        By.TAG_NAME,
        "h1",
    )

    print(heading.text)

finally:
    driver.quit()
```

Selenium is useful when:

* JavaScript rendering is required
* Browser interaction is necessary
* Content appears only after page execution

It is usually heavier than direct HTTP requests.

---

# 🔹 Playwright

Playwright is another browser automation framework.

Install:

```bash
pip install playwright
playwright install
```

Example:

```python
from playwright.sync_api import sync_playwright


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        headless=True
    )

    page = browser.new_page()

    page.goto(
        "https://example.com",
        wait_until="domcontentloaded",
    )

    print(page.title())

    browser.close()
```

Playwright provides strong support for modern browser automation.

---

# 🔹 When Not to Use Browser Automation

Do not use Selenium or Playwright simply because they are available.

Prefer:

```text
Direct HTTP request
```

when:

* HTML contains the required data.
* An official API exists.
* A static endpoint is sufficient.
* Browser rendering provides no benefit.

Browser automation consumes more:

```text
CPU
Memory
Time
Network resources
```

A good engineering principle is:

> Use the simplest legitimate mechanism that provides the required data.

---

# 🔹 Web Scraping with Pandas

Pandas can directly parse certain HTML tables.

```python
import pandas as pd

tables = pd.read_html(
    "https://example.com/data"
)

print(len(tables))

for table in tables:
    print(table.head())
```

This is convenient for tabular pages.

However, it is not a replacement for a general HTML parser.

---

# 🔹 Scraping Tables

For a table like:

```html
<table>
    <tr>
        <th>Product</th>
        <th>Price</th>
    </tr>

    <tr>
        <td>Laptop</td>
        <td>75000</td>
    </tr>
</table>
```

Pandas can produce:

```text
Product | Price
--------|------
Laptop  | 75000
```

Always inspect the resulting schema:

```python
df.info()
df.head()
```

---

# 🔹 Nested HTML

HTML may contain deeply nested elements.

Example:

```html
<div class="product">
    <div class="details">
        <h2 class="name">Laptop</h2>

        <div class="pricing">
            <span class="price">
                ₹75,000
            </span>
        </div>
    </div>
</div>
```

CSS:

```python
name = soup.select_one(
    ".product .details .name"
)
```

However, prefer stable selectors rather than excessively long paths.

---

# 🔹 Data Validation

After scraping, validate the extracted dataset.

Check:

```text
Schema
Missing values
Duplicates
Data types
Ranges
Unique IDs
Date formats
Unexpected HTML changes
```

Example:

```python
required = {
    "name",
    "price",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )
```

---

# 🔹 Data Cleaning

Suppose scraped prices look like:

```text
₹75,000
₹1,20,000
₹99,999
```

Convert them into numbers:

```python
import pandas as pd


def parse_price(value):
    if pd.isna(value):
        return pd.NA

    cleaned = (
        str(value)
        .replace("₹", "")
        .replace(",", "")
        .strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


df["price"] = df["price"].map(parse_price)
```

Now:

```text
₹75,000 → 75000
```

---

# 🔹 Feature Engineering

Scraped data often requires transformation before ML.

Suppose:

```text
title
description
price
rating
review_count
```

Potential features:

```text
title_length
description_length
price_log
rating
review_count
```

Example:

```python
df["title_length"] = (
    df["title"]
    .fillna("")
    .str.len()
)
```

For numeric transformations:

```python
import numpy as np

df["log_review_count"] = np.log1p(
    df["review_count"]
)
```

---

# 🔹 Text Data for NLP

Web scraping is especially useful for NLP.

A pipeline could be:

```text
Web Pages
    ↓
Text Extraction
    ↓
Cleaning
    ↓
Tokenization
    ↓
Feature Extraction
    ↓
ML / NLP Model
```

Possible tasks:

* Sentiment analysis
* Topic classification
* Spam detection
* News classification
* Text clustering
* Search
* Summarization

Example dataset:

```text
text | category
-----|---------
article 1 | technology
article 2 | finance
article 3 | sports
```

---

# 🔹 Avoiding Data Leakage

Web data often changes over time.

Suppose you want to predict:

```text
Product price next week
```

You should not train using information published after the prediction timestamp.

Record metadata such as:

```text
source_url
collected_at
published_at
updated_at
```

A useful representation is:

```text
event_time
collection_time
publication_time
```

Then define precisely which timestamp determines feature availability.

---

# 🔹 Data Provenance

Every scraped dataset should ideally retain information about where it came from.

Useful metadata:

```text
source_url
source_name
collection_timestamp
page_timestamp
scraper_version
parser_version
schema_version
```

Example:

```python
record = {
    "name": "Laptop",
    "price": 75000,
    "source_url": "https://example.com/product/123",
    "collected_at": "2026-09-29T12:00:00Z",
}
```

This makes datasets easier to audit and reproduce.

---

# 🔹 Storage Formats

Possible storage formats:

### JSON

Good for preserving raw API-like structures.

```python
import json

with open(
    "raw_data.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        records,
        file,
        ensure_ascii=False,
        indent=2,
    )
```

### CSV

Good for simple tabular datasets.

```python
df.to_csv(
    "products.csv",
    index=False,
)
```

### Parquet

Good for analytical datasets.

```python
df.to_parquet(
    "products.parquet",
    index=False,
)
```

For larger ML pipelines, Parquet is often more efficient than CSV because it preserves types and supports column-oriented storage.

---

# 🔹 Scraping Pipeline

A clean scraping pipeline:

```text
                    ┌───────────────┐
                    │ Website       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ HTTP Client   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ HTML Parser   │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Extractor     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Validator     │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Cleaner       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Storage       │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ ML Pipeline   │
                    └───────────────┘
```

---

# 🔹 Caching

Caching reduces unnecessary network requests.

Concept:

```text
Request URL
     ↓
Check Cache
   ↙     ↘
 HIT     MISS
  ↓        ↓
Data     Website
           ↓
         Cache
```

Caching provides:

* Faster development
* Lower network usage
* Easier testing
* Reproducibility
* Reduced repeated requests

For example, during parser development, save a previously collected HTML page and test your parser locally instead of repeatedly requesting the live site.

---

# 🔹 Logging

Production scrapers should log operational information.

Example:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

logger.info(
    "Fetching page: %s",
    url,
)

logger.info(
    "Extracted %d records",
    len(records),
)
```

Useful metrics:

```text
pages requested
pages successful
pages failed
records extracted
records rejected
HTTP status distribution
request latency
retry count
```

Never log credentials or sensitive tokens.

---

# 🔹 Testing

Scrapers should be tested like normal software.

Test:

### Parser

```text
Does HTML produce expected records?
```

### Missing fields

```text
Does missing price produce None?
```

### HTML changes

```text
Does the parser fail clearly?
```

### Data validation

```text
Are invalid records rejected?
```

A good strategy is to keep representative HTML fixtures:

```text
tests/
├── fixtures/
│   ├── product_page.html
│   └── product_page_missing_price.html
│
├── test_parser.py
└── test_validator.py
```

---

# 🔹 Production Architecture

A production-oriented architecture can look like:

```text
                 ┌─────────────────┐
                 │ Web Source      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Scheduler       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Scraper Workers │
                 └────────┬────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
        ┌──────────────┐    ┌──────────────┐
        │ Raw Storage  │    │ Logs/Metrics │
        └──────┬───────┘    └──────────────┘
               │
               ▼
        ┌──────────────┐
        │ Validation   │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Cleaning     │
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ Feature Store│
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ ML Pipeline  │
        └──────────────┘
```

---

# 🔹 Common Mistakes

### ❌ Scraping without understanding HTML

Learn:

```text
HTML
DOM
CSS selectors
Attributes
```

first.

### ❌ Assuming every page has identical structure

Real websites may contain:

```text
Missing elements
Different templates
Advertisements
A/B tests
Empty fields
```

### ❌ No timeout

Always configure timeouts.

### ❌ No rate limiting

Avoid sending excessive requests.

### ❌ Ignoring errors

A scraper that silently loses records is difficult to trust.

### ❌ Storing only cleaned data

Keeping raw data can be valuable for debugging and reproducibility.

### ❌ Hard-coding selectors everywhere

Centralize selectors where practical.

### ❌ Ignoring data provenance

Always know:

```text
Where did this record come from?
When was it collected?
Which scraper version produced it?
```

### ❌ Using browser automation unnecessarily

Try direct HTTP retrieval first when appropriate.

### ❌ Training on leaked future information

Always consider temporal availability.

---

# 🔹 Mini Projects

## 🟢 Project 1 — Quote Scraper

Build a scraper that collects:

```text
Quote
Author
Tags
```

Store:

```text
quotes.csv
```

Skills:

* requests
* BeautifulSoup
* CSS selectors
* Pandas

---

## 🟢 Project 2 — Multi-Page Product Scraper

Collect:

```text
Product
Price
Rating
Category
URL
```

Implement:

```text
Pagination
Cleaning
Validation
CSV export
```

---

## 🟡 Project 3 — News Dataset

Collect publicly accessible article metadata:

```text
title
description
publication_date
category
source_url
```

Build an NLP dataset.

---

## 🟡 Project 4 — Review Sentiment Dataset

Collect appropriately licensed/public review data and build:

```text
Review
Rating
Date
Product
```

Then create:

```text
Text
 ↓
Cleaning
 ↓
TF-IDF
 ↓
Classification
```

---

## 🟡 Project 5 — Historical Price Dataset

Collect permitted product price observations over time.

Dataset:

```text
timestamp
product
price
category
```

Analyze:

```text
Price trends
Price changes
Volatility
Seasonality
```

---

## 🔴 Project 6 — Production Web Data Pipeline

Build:

```text
Scheduler
   ↓
Scraper
   ↓
Retry
   ↓
Rate Limiter
   ↓
Raw Storage
   ↓
Validation
   ↓
Cleaning
   ↓
Database
   ↓
Feature Engineering
   ↓
ML Model
```

Add:

* Logging
* Monitoring
* Tests
* Data quality checks
* Schema versioning

---

# 🔹 Exercises

### Beginner

1. What is web scraping?
2. What is HTML?
3. What is the DOM?
4. What is a CSS selector?
5. What is XPath?
6. What is BeautifulSoup?
7. How do you extract an `href`?
8. How do you extract text from an HTML element?

### Intermediate

9. Write a basic webpage scraper.
10. Extract all links from a page.
11. Extract all images.
12. Extract a table using Pandas.
13. Scrape multiple pages.
14. Implement pagination.
15. Add timeout handling.
16. Add retry handling.
17. Save the dataset as CSV.
18. Save raw HTML for reproducibility.

### Advanced

19. Build a session-based scraper.
20. Implement caching.
21. Add structured logging.
22. Create parser unit tests.
23. Detect schema changes.
24. Build a dynamic-page scraper.
25. Build an API-first data collector.
26. Add data-quality validation.
27. Add provenance metadata.
28. Build an end-to-end scraper → ML pipeline.
29. Design a temporal leakage prevention strategy.

---

# 🔹 Project Structure

A professional scraping project:

```text
07-Web-Scraping/
│
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── client.py
│   ├── parser.py
│   ├── extractor.py
│   ├── validator.py
│   ├── cleaner.py
│   ├── storage.py
│   └── pipeline.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│   ├── fixtures/
│   ├── test_client.py
│   ├── test_parser.py
│   └── test_validator.py
│
├── logs/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── pyproject.toml
```

---

# 🔹 End-to-End Example

The following example demonstrates a clean separation between:

```text
Fetching
Parsing
Extraction
Validation
Storage
```

```python
"""
Simple web scraping pipeline.

Demonstrates:
- HTTP requests
- HTML parsing
- CSS selectors
- Basic validation
- Pandas conversion
- CSV storage
"""

from __future__ import annotations

import logging

import pandas as pd
import requests
from bs4 import BeautifulSoup


logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def fetch_html(
    url: str,
    session: requests.Session,
) -> str:
    """Fetch HTML from a webpage."""

    response = session.get(
        url,
        timeout=10,
    )

    response.raise_for_status()

    return response.text


def parse_products(html: str) -> list[dict]:
    """Extract product records from HTML."""

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    records = []

    for card in soup.select(".product"):
        name_element = card.select_one(".name")
        price_element = card.select_one(".price")
        link_element = card.select_one("a")

        record = {
            "name": (
                name_element.get_text(
                    " ",
                    strip=True,
                )
                if name_element
                else None
            ),
            "price": (
                price_element.get_text(
                    " ",
                    strip=True,
                )
                if price_element
                else None
            ),
            "url": (
                link_element.get("href")
                if link_element
                else None
            ),
        }

        records.append(record)

    return records


def validate_records(
    records: list[dict],
) -> None:
    """Validate extracted records."""

    if not records:
        raise ValueError(
            "No records were extracted."
        )

    required_fields = {
        "name",
        "price",
        "url",
    }

    for index, record in enumerate(records):
        missing = (
            required_fields
            - record.keys()
        )

        if missing:
            raise ValueError(
                f"Record {index} is missing "
                f"{sorted(missing)}"
            )


def main() -> None:
    url = "https://example.com/products"

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "ml-research-scraper/1.0"
        )
    })

    logger.info(
        "Fetching %s",
        url,
    )

    html = fetch_html(
        url,
        session,
    )

    records = parse_products(html)

    validate_records(records)

    df = pd.DataFrame(records)

    df.to_csv(
        "products.csv",
        index=False,
    )

    logger.info(
        "Saved %d records",
        len(df),
    )


if __name__ == "__main__":
    main()
```

> The `.product`, `.name`, and `.price` selectors are placeholders. Always inspect the target site's HTML and adapt them to the actual structure.

---

# 🔹 ML Workflow

Web scraping becomes valuable for ML when it forms a reproducible data pipeline:

```text
                    Web
                     │
                     ▼
              ┌─────────────┐
              │   Scraper   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Raw Dataset │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Validation  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │  Cleaning   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   Features  │
              └──────┬──────┘
                     │
                     ▼
            ┌──────────────────┐
            │ Train / Validate │
            │      / Test      │
            └────────┬─────────┘
                     │
                     ▼
              ┌─────────────┐
              │ ML Training │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Evaluation  │
              └─────────────┘
```

---

# 🔹 Best Practices

## 1. Prefer official APIs

If a legitimate API provides the required information, consider using it before scraping rendered HTML.

## 2. Inspect before automating

Understand:

```text
HTML
DOM
Selectors
Pagination
Data source
```

before writing the scraper.

## 3. Use timeouts

Never allow network requests to hang indefinitely.

## 4. Respect request limits

Use controlled request rates.

## 5. Validate everything

Web content can change without warning.

## 6. Keep raw data

Raw HTML or raw extracted records can make debugging much easier.

## 7. Track provenance

Record:

```text
URL
Timestamp
Scraper version
Source
```

## 8. Test parsers

A small HTML change can break an extraction rule.

## 9. Separate concerns

Keep:

```text
HTTP client
Parser
Extractor
Validator
Storage
```

separate.

## 10. Think about ML leakage

Ask:

> "Would this information actually have been available when the model makes the prediction?"

---

# 🔹 Data Collection Quality Checklist

Before using scraped data for ML:

### Source

* [ ] Source identified
* [ ] Data purpose understood
* [ ] API availability checked
* [ ] Terms reviewed
* [ ] Applicable licensing considered

### Scraper

* [ ] Timeout configured
* [ ] Retry policy defined
* [ ] Rate limit respected
* [ ] Errors logged
* [ ] Parser tested

### Dataset

* [ ] Schema validated
* [ ] Missing values checked
* [ ] Duplicates checked
* [ ] Data types validated
* [ ] Outliers investigated
* [ ] URLs/source metadata preserved

### ML

* [ ] Target defined
* [ ] Features defined
* [ ] Leakage checked
* [ ] Temporal ordering considered
* [ ] Train/test strategy defined
* [ ] Reproducibility maintained

---

# 🔹 Learning Roadmap

Follow this progression:

```text
HTML
 ↓
DOM
 ↓
CSS Selectors
 ↓
XPath
 ↓
HTTP
 ↓
Requests
 ↓
BeautifulSoup
 ↓
Pagination
 ↓
Sessions
 ↓
Error Handling
 ↓
Rate Limiting
 ↓
Data Validation
 ↓
Pandas
 ↓
Dynamic Websites
 ↓
Browser Automation
 ↓
Data Pipelines
 ↓
Feature Engineering
 ↓
Machine Learning
```

---

# 🔹 Key Takeaways

> Web scraping is not simply downloading HTML. A reliable ML scraping system is a complete data engineering workflow.

Remember:

1. **Web scraping extracts information programmatically from websites.**
2. **HTML and DOM knowledge are essential.**
3. **CSS selectors and XPath locate page elements.**
4. **Requests is useful for static pages.**
5. **BeautifulSoup parses HTML efficiently.**
6. **Dynamic pages may require a different data-access strategy.**
7. **Browser automation should not be the default choice.**
8. **Always use request timeouts.**
9. **Handle failures and retries carefully.**
10. **Respect rate limits and access restrictions.**
11. **Review robots.txt, terms, licensing, privacy, and applicable law.**
12. **Never bypass authentication, CAPTCHAs, paywalls, or security controls.**
13. **Validate scraped data before using it for ML.**
14. **Preserve provenance and collection timestamps.**
15. **Store raw data when reproducibility matters.**
16. **Prevent temporal data leakage.**
17. **Treat scraping code as production software when it feeds production ML systems.**

---

# 🔹 Data Collection & Understanding Roadmap

This section completes the major external-data collection methods:

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
├── 06-API-Data
│
└── 07-Web-Scraping
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

The goal is not simply to **collect more data**, but to build datasets that are:

```text
Reliable
Reproducible
Relevant
Validated
Well-documented
Legally and ethically appropriate
Suitable for ML
```

---

# 🔹 Next Step

After Web Scraping, continue with **Data Understanding**.

The next stage should answer:

```text
What data did we collect?
        ↓
What does each column mean?
        ↓
What are the data types?
        ↓
How much data do we have?
        ↓
Are there missing values?
        ↓
Are there duplicates?
        ↓
Are there outliers?
        ↓
Are there biases?
        ↓
Is the target suitable?
        ↓
Is the data ready for ML?
```

This leads naturally into:

```text
Data Collection
      ↓
Data Understanding
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

# 👨‍💻 Author

**Kishor Patil**

Machine Learning | Python | Data Science | AI

GitHub:
[https://github.com/Kishor055](https://github.com/Kishor055)

---

# 🤝 Contributing

Contributions are welcome!

If you find an issue or want to improve this learning material:

1. Fork the repository.
2. Create a feature branch.
3. Add your improvements.
4. Test all code examples.
5. Commit your changes.
6. Open a Pull Request.

Please keep contributions:

* Beginner-friendly
* Technically accurate
* Reproducible
* Well documented
* Consistent with the repository structure

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute examples

**Learn → Build → Experiment → Validate → Deploy 🚀**

Happy Learning! 🐍🤖

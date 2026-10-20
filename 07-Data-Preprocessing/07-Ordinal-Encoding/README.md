# 🏷️ Ordinal Encoding in Python

🌱 **UNDERSTAND → DEFINE ORDER → ENCODE → VALIDATE → TRAIN → EVALUATE**

Welcome to **Ordinal Encoding**, part of the Data Preprocessing module in the Machine Learning learning path. This guide explains how to convert ordered categorical values into numerical representations using Python, Pandas, and Scikit-learn.

Ordinal encoding is useful when categories have a meaningful order, such as education levels, satisfaction ratings, experience levels, and product quality ratings. It helps prepare categorical features for machine learning while preserving their intended ranking.

This tutorial covers manual ordinal mapping, `pandas.map()`, Scikit-learn's `OrdinalEncoder`, category ordering, missing and unknown categories, preprocessing pipelines, common mistakes, and practical machine learning examples.

## 📚 Table of Contents

1. [Introduction](#-introduction)
2. [What Is Ordinal Encoding?](#-what-is-ordinal-encoding)
3. [Why Ordinal Encoding Matters](#-why-ordinal-encoding-matters)
4. [Understanding Ordinal Variables](#-understanding-ordinal-variables)
5. [Ordinal Encoding vs Other Encoding Methods](#-ordinal-encoding-vs-other-encoding-methods)
6. [Create a Sample Dataset](#-create-a-sample-dataset)
7. [Manual Ordinal Encoding](#-manual-ordinal-encoding)
8. [Ordinal Encoding with Pandas](#-ordinal-encoding-with-pandas)
9. [Ordinal Encoding with Scikit-learn](#-ordinal-encoding-with-scikit-learn)
10. [Encoding Multiple Columns](#-encoding-multiple-columns)
11. [Handling Missing and Unknown Categories](#-handling-missing-and-unknown-categories)
12. [Inverse Transform](#-inverse-transform)
13. [Ordinal Encoding with a Pipeline](#-ordinal-encoding-with-a-pipeline)
14. [Complete Machine Learning Example](#-complete-machine-learning-example)
15. [Common Mistakes](#-common-mistakes)
16. [Best Practices](#-best-practices)
17. [Mini Projects](#-mini-projects)
18. [Practice Exercises](#-practice-exercises)
19. [Quick Revision](#-quick-revision)
20. [Project Structure](#-project-structure)
21. [Learning Roadmap](#-learning-roadmap)
22. [References](#-references)

---

## 🎯 Introduction

Machine learning algorithms generally require numerical feature representations. However, real-world datasets often contain categorical information represented by text.

For example:

```python
education_levels = [
    "High School",
    "Bachelor",
    "Master",
    "Doctorate"
]
```

These categories have a meaningful order in the assumed education-level hierarchy. We can represent them numerically:

| Education level | Encoded value |
|---|---:|
| High School | 0 |
| Bachelor | 1 |
| Master | 2 |
| Doctorate | 3 |

This transformation is called **ordinal encoding**.

The assigned numbers represent the order of the categories, not necessarily the actual distance between them. A Doctorate encoded as `3` is not literally three times a High School qualification encoded as `1` in any meaningful measurement sense.

### Learning objectives

By the end of this guide, you will be able to:

- Understand ordinal and nominal categorical variables.
- Define the correct order of categories.
- Perform ordinal encoding manually and with Pandas.
- Use Scikit-learn's `OrdinalEncoder`.
- Encode several categorical columns with different orders.
- Handle missing and previously unseen categories.
- Integrate ordinal encoding into a preprocessing pipeline.
- Avoid common encoding mistakes and data leakage.
- Prepare categorical features for machine learning.

## 🧠 What Is Ordinal Encoding?

**Ordinal encoding** is a categorical encoding technique that assigns an integer or numerical code to each category according to a meaningful ranking.

For example, an experience-level feature may contain:

```text
Beginner
Intermediate
Advanced
Expert
```

A suitable encoding is:

```text
Beginner      → 0
Intermediate  → 1
Advanced      → 2
Expert        → 3
```

The encoding preserves the ranking:

\[
\text{Beginner} < \text{Intermediate} < \text{Advanced} < \text{Expert}
\]

However, it does not establish that the difference between Beginner and Intermediate is equal to the difference between Advanced and Expert.

That assumption depends on the feature's meaning and the model being used.

### Example

```python
import pandas as pd

df = pd.DataFrame({
    "Customer": ["Amit", "Priya", "Rahul", "Neha"],
    "Satisfaction": ["Low", "High", "Medium", "High"]
})

satisfaction_order = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["Satisfaction_Encoded"] = df["Satisfaction"].map(
    satisfaction_order
)

print(df)
```

Output:

```text
  Customer Satisfaction  Satisfaction_Encoded
0     Amit          Low                      0
1    Priya         High                      2
2    Rahul       Medium                      1
3     Neha         High                      2
```

The numerical representation preserves the specified order.

## 🚀 Why Ordinal Encoding Matters

Ordinal encoding can be helpful because it:

- Converts ordered categories into numerical features.
- Preserves a known category ranking.
- Uses one output column per encoded input column.
- Keeps the feature representation relatively compact.
- Works naturally for features such as education level, rating, and experience tier.
- Can be integrated into Scikit-learn preprocessing pipelines.

Consider a customer satisfaction dataset:

| Customer | Satisfaction |
|---|---|
| A | Low |
| B | Medium |
| C | High |
| D | Medium |

A model cannot directly use these strings with every estimator. An ordinal encoder can transform them into numbers while preserving the established order.

**Important:** Numerical codes do not automatically make a feature suitable for every model. Some algorithms interpret numerical differences and distances, so the encoding must match the problem.

## 📊 Understanding Ordinal Variables

A categorical variable is **ordinal** when its categories have a meaningful order.

Examples include:

| Feature | Ordered categories |
|---|---|
| Education | High School → Bachelor → Master → Doctorate |
| Satisfaction | Very Low → Low → Medium → High → Very High |
| Experience | Beginner → Intermediate → Advanced → Expert |
| Product quality | Poor → Fair → Good → Very Good → Excellent |
| T-shirt size | XS → S → M → L → XL |
| Credit rating tier | Low → Medium → High, according to a defined rating scheme |

Not all categorical variables are ordinal.

### Ordinal versus nominal variables

| Ordinal variable | Nominal variable |
|---|---|
| Categories have a meaningful order | Categories have no inherent order |
| Education level | City |
| Satisfaction rating | Department |
| Product quality | Colour |
| Experience tier | Country |
| Size category | Payment method |

For nominal categories, use an appropriate technique such as One-Hot Encoding rather than inventing an order.

For example, assigning Mumbai `0`, Pune `1`, and Delhi `2` does not mean Delhi is greater than Pune or that Pune lies numerically between Mumbai and Delhi.

## 🔄 Ordinal Encoding vs Other Encoding Methods

| Method | Purpose | Common use |
|---|---|---|
| Ordinal Encoding | Represents ordered categorical features with numerical ranks | Education level, ratings, experience tiers |
| One-Hot Encoding | Represents categories using separate indicator columns | City, department, colour |
| Label Encoding | Commonly encodes target class labels into integers | Classification target variable |
| Binary Encoding | Represents category identifiers using binary digits | Some high-cardinality categorical features |
| Frequency Encoding | Replaces categories with their frequencies or proportions | Certain high-cardinality features |
| Target Encoding | Uses target-related statistics to encode categories | Selected supervised-learning problems with careful leakage control |

### Ordinal Encoding vs One-Hot Encoding

Ordinal encoding:

```text
Low     → 0
Medium  → 1
High    → 2
```

One-Hot Encoding:

| Satisfaction | Low | Medium | High |
|---|---:|---:|---:|
| Low | 1 | 0 | 0 |
| Medium | 0 | 1 | 0 |
| High | 0 | 0 | 1 |

Ordinal encoding is appropriate when the order is meaningful. One-Hot Encoding is often preferable for nominal categories or when imposing a ranking would misrepresent the data.

### Ordinal Encoding vs Label Encoding

These methods are related but serve different purposes.

- `OrdinalEncoder` is designed for categorical input features, commonly represented by columns in `X`.
- `LabelEncoder` is commonly used to encode target labels in `y` for classification tasks.
- Neither technique automatically identifies the semantic order of categories.

For an ordered feature, explicitly define the category sequence rather than assuming the encoder will infer the intended ranking.

## 🧪 Create a Sample Dataset

Let's create a dataset that contains education levels, experience tiers, and satisfaction ratings.

```python
import pandas as pd

df = pd.DataFrame({
    "Student": [
        "Amit", "Priya", "Rahul", "Neha",
        "Kiran", "Meera", "Arjun", "Sneha"
    ],
    "Education": [
        "High School", "Bachelor", "Master", "Bachelor",
        "Doctorate", "High School", "Master", "Bachelor"
    ],
    "Experience": [
        "Beginner", "Intermediate", "Advanced", "Beginner",
        "Expert", "Intermediate", "Advanced", "Expert"
    ],
    "Satisfaction": [
        "Low", "Medium", "High", "Medium",
        "Very High", "Low", "High", "Very High"
    ]
})

print(df)
```

Inspect the categorical columns:

```python
print(df.dtypes)

for column in ["Education", "Experience", "Satisfaction"]:
    print(f"\n{column}:")
    print(df[column].value_counts())
```

Before encoding, define the order for each feature using domain knowledge or the dataset specification.

## 🛠️ Manual Ordinal Encoding

Manual mapping is a simple and transparent way to encode a known order.

### Step 1: Define the category order

```python
education_mapping = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "Doctorate": 3
}
```

### Step 2: Apply the mapping

```python
df["Education_Encoded"] = df["Education"].map(
    education_mapping
)

print(df[["Education", "Education_Encoded"]])
```

### Step 3: Encode experience levels

```python
experience_mapping = {
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2,
    "Expert": 3
}

df["Experience_Encoded"] = df["Experience"].map(
    experience_mapping
)
```

### Step 4: Encode satisfaction ratings

```python
satisfaction_mapping = {
    "Very Low": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

df["Satisfaction_Encoded"] = df["Satisfaction"].map(
    satisfaction_mapping
)
```

### Complete example

```python
import pandas as pd

df = pd.DataFrame({
    "Education": [
        "High School", "Bachelor", "Master", "Doctorate"
    ],
    "Experience": [
        "Beginner", "Intermediate", "Advanced", "Expert"
    ],
    "Satisfaction": [
        "Low", "Medium", "High", "Very High"
    ]
})

mappings = {
    "Education": {
        "High School": 0,
        "Bachelor": 1,
        "Master": 2,
        "Doctorate": 3
    },
    "Experience": {
        "Beginner": 0,
        "Intermediate": 1,
        "Advanced": 2,
        "Expert": 3
    },
    "Satisfaction": {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4
    }
}

for column, mapping in mappings.items():
    df[f"{column}_Encoded"] = df[column].map(mapping)

print(df)
```

### Advantages of manual encoding

- The category order is explicit.
- The mapping is easy to inspect and document.
- It is useful for business-defined ordinal scales.
- The same mapping can be reused across datasets.
- It allows custom category codes.

### Important limitation

Pandas `map()` returns missing values for inputs that are not present in the mapping, including unexpected categories.

Check for unmapped values before using the encoded column:

```python
column = "Education"
mapping = education_mapping

encoded = df[column].map(mapping)

unknown_mask = df[column].notna() & encoded.isna()

print("Unmapped categories:")
print(df.loc[unknown_mask, column].unique())
```

The mapping should be updated only when an additional category is verified and its position in the order is known.

## 🐼 Ordinal Encoding with Pandas

Pandas also supports ordered categorical data using `CategoricalDtype`.

This is useful when you want to define an explicit category order in a DataFrame.

### Example

```python
import pandas as pd

education_order = [
    "High School",
    "Bachelor",
    "Master",
    "Doctorate"
]

education_dtype = pd.CategoricalDtype(
    categories=education_order,
    ordered=True
)

df = pd.DataFrame({
    "Education": [
        "Bachelor",
        "High School",
        "Doctorate",
        "Master"
    ]
})

df["Education"] = df["Education"].astype(education_dtype)

print(df)
print(df["Education"].cat.categories)
print(df["Education"].cat.ordered)
```

### Compare ordered categories

```python
print(df["Education"] > "Bachelor")
```

Because the categories are ordered, Pandas can compare them according to the defined sequence, provided the comparison is valid for that categorical dtype.

### Get ordinal codes

```python
df["Education_Code"] = df["Education"].cat.codes

print(df)
```

The codes follow the declared category order:

```text
High School  → 0
Bachelor     → 1
Master       → 2
Doctorate    → 3
```

Pandas uses `-1` for missing categorical values in `.cat.codes`. Avoid interpreting that sentinel as an ordinary rank.

### Handle unknown categories

When converting to a fixed categorical vocabulary, values not included in the declared categories become missing:

```python
education_order = [
    "High School",
    "Bachelor",
    "Master",
    "Doctorate"
]

df = pd.DataFrame({
    "Education": ["Bachelor", "PhD", "Master"]
})

df["Education"] = df["Education"].astype(
    pd.CategoricalDtype(
        categories=education_order,
        ordered=True
    )
)

print(df)
```

Here, `"PhD"` becomes missing because it is absent from the declared category list. Investigate the value before deciding whether to add it to the schema and where it belongs in the ranking.

**Note:** Use manual mapping when custom integer codes or explicit auditing are important. Use `CategoricalDtype` when ordered category behavior is useful throughout Pandas operations.

## 🤖 Ordinal Encoding with Scikit-learn

Scikit-learn provides `OrdinalEncoder` to encode categorical input features.

### Basic example

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    "Satisfaction": [
        "Low",
        "Medium",
        "High",
        "Very High",
        "Low"
    ]
})

encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High", "Very High"]]
)

df["Satisfaction_Encoded"] = encoder.fit_transform(
    df[["Satisfaction"]]
)

print(df)
```

Output:

```text
  Satisfaction  Satisfaction_Encoded
0          Low                    0.0
1       Medium                    1.0
2         High                    2.0
3    Very High                    3.0
4          Low                    0.0
```

`OrdinalEncoder` accepts a two-dimensional feature matrix, so use `df[["Satisfaction"]]` rather than `df["Satisfaction"]` for this example.

### Why specify `categories`?

By default, `OrdinalEncoder(categories="auto")` infers categories from the training data. The resulting sequence may not match your domain's intended order.

For ordered variables, define the order explicitly:

```python
encoder = OrdinalEncoder(
    categories=[[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]]
)
```

The provided category sequence determines the code assignment.

### Inspect the learned categories

```python
print(encoder.categories_)
```

For example:

```text
[array(['Low', 'Medium', 'High', 'Very High'], dtype=object)]
```

The exact displayed type depends on the input and installed library version.

### Transform new data

```python
new_data = pd.DataFrame({
    "Satisfaction": ["High", "Low", "Very High"]
})

encoded_data = encoder.transform(new_data[["Satisfaction"]])

print(encoded_data)
```

Expected encoded values:

```text
[[2.]
 [0.]
 [3.]]
```

Use the fitted encoder for new data. Do not fit a new encoder independently on test or production data, because that can produce inconsistent codes.

## 🔢 Encoding Multiple Columns

A dataset can contain several ordinal features with different category orders.

For example:

- Education: High School → Bachelor → Master → Doctorate
- Experience: Beginner → Intermediate → Advanced → Expert
- Satisfaction: Very Low → Low → Medium → High → Very High

Each column needs its own defined order.

### Example

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    "Education": [
        "Bachelor",
        "Master",
        "High School",
        "Doctorate"
    ],
    "Experience": [
        "Intermediate",
        "Advanced",
        "Beginner",
        "Expert"
    ],
    "Satisfaction": [
        "Medium",
        "High",
        "Low",
        "Very High"
    ]
})

categories = [
    ["High School", "Bachelor", "Master", "Doctorate"],
    ["Beginner", "Intermediate", "Advanced", "Expert"],
    ["Very Low", "Low", "Medium", "High", "Very High"]
]

encoder = OrdinalEncoder(categories=categories)

encoded_array = encoder.fit_transform(df)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=[
        "Education_Encoded",
        "Experience_Encoded",
        "Satisfaction_Encoded"
    ],
    index=df.index
)

print(encoded_df)
```

### Important rule

The categories list must follow the same column order as the input DataFrame.

In this example, the first category list belongs to `Education`, the second to `Experience`, and the third to `Satisfaction`.

Avoid relying on accidental alphabetical sorting or on the order in which values happen to appear in the dataset.

## 🧩 Handling Missing and Unknown Categories

Production datasets often contain missing values and categories that were not present during model training.

These cases must be handled deliberately.

### 1. Handle unknown categories with `OrdinalEncoder`

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

train_data = pd.DataFrame({
    "Experience": [
        "Beginner",
        "Intermediate",
        "Advanced",
        "Expert"
    ]
})

test_data = pd.DataFrame({
    "Experience": [
        "Beginner",
        "Expert",
        "Specialist"
    ]
})

encoder = OrdinalEncoder(
    categories=[[
        "Beginner",
        "Intermediate",
        "Advanced",
        "Expert"
    ]],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

encoder.fit(train_data)

print(encoder.transform(test_data))
```

The `"Specialist"` value is not part of the defined order and is assigned `-1`.

**Caution:** A sentinel of `-1` does not mean the category is truly lower than Beginner. It is a technical code for an unknown value. Some models may interpret it as a rank below all known categories. In production, consider explicit unknown-category policies, monitoring, or a separately defined category where that approach is justified.

### 2. Handle missing categories

Scikit-learn's `OrdinalEncoder` supports `encoded_missing_value`. It can be used to assign a dedicated code to missing values.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    "Satisfaction": ["Low", "Medium", None, "High"]
})

encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-2
)

encoded = encoder.fit_transform(df[["Satisfaction"]])

print(encoded)
```

In this example:

- `"Low"` is encoded as `0`.
- `"Medium"` is encoded as `1`.
- `"High"` is encoded as `2`.
- An unknown nonmissing category is encoded as `-1`.
- A missing category is encoded as `-2`.

This example uses distinct sentinel values so that missing and unknown categories can be distinguished.

Missing-value handling and unknown-category handling are separate design decisions. The model may still need a justified strategy for how these sentinel codes should influence predictions.

### 3. Use an imputer before ordinal encoding

For many machine learning workflows, imputing missing categories is more convenient than passing special missing codes into the model.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OrdinalEncoder(
            categories=[["Low", "Medium", "High"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    )
])
```

The imputer learns the most frequent category during `fit()` and applies that learned value during transformation. The encoder then maps the category according to the declared order.

This approach is appropriate when replacing missing categories with the most frequent category is justified for the application. It is not always appropriate for every dataset.

## 🔁 Inverse Transform

The `inverse_transform()` method converts encoded values back into the original categorical labels.

### Example

```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    "Quality": ["Poor", "Good", "Excellent", "Fair"]
})

encoder = OrdinalEncoder(
    categories=[["Poor", "Fair", "Good", "Excellent"]]
)

encoded = encoder.fit_transform(df[["Quality"]])

decoded = encoder.inverse_transform(encoded)

print("Encoded values:")
print(encoded)

print("\nDecoded values:")
print(decoded)
```

Expected encoding:

```text
Poor       → 0
Fair       → 1
Good       → 2
Excellent  → 3
```

### Why inverse transformation is useful

- Interpreting model input transformations.
- Debugging preprocessing.
- Verifying category mappings.
- Converting encoded categories back to readable labels.

Inverse transformation is not a way to restore original information if it has already been lost or replaced by an unknown-value sentinel. Preserve the original data when auditability matters.

## 🔗 Ordinal Encoding with a Pipeline

For real machine learning projects, encoding should generally be part of a reproducible preprocessing workflow.

A typical dataset may contain:

- Numeric features, such as age and attendance.
- Ordinal features, such as education level.
- Nominal features, such as city or department.
- A target variable, such as pass/fail status.

Use `ColumnTransformer` to apply different transformations to different columns.

### Recommended strategy

| Feature type | Example | Preprocessing |
|---|---|---|
| Numerical | Age, Attendance | Imputation and optional scaling |
| Ordinal | Satisfaction, Education | Imputation and `OrdinalEncoder` with explicit order |
| Nominal | City, Department | Imputation and `OneHotEncoder` |
| Target | Pass/Fail | Appropriate target representation |

### Example pipeline structure

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

numeric_features = ["Age", "Attendance"]
ordinal_features = ["Education", "Satisfaction"]
nominal_features = ["City"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OrdinalEncoder(
            categories=[
                ["High School", "Bachelor", "Master", "Doctorate"],
                ["Low", "Medium", "High"]
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    )
])

nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("ordinal", ordinal_pipeline, ordinal_features),
    ("nominal", nominal_pipeline, nominal_features)
])
```

This preprocessing setup preserves known ordering for the two ordinal columns while avoiding an invented ranking for the city column.

The category list inside `OrdinalEncoder` follows the order of `ordinal_features`. Ensure that the declared labels match the actual schema used by your dataset.

## 🛠️ Complete Machine Learning Example

The following example demonstrates ordinal encoding, nominal encoding, numerical preprocessing, a training/test split, and classification in one reproducible workflow.

Save this code as `ordinal_encoding.py`.

```python
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def create_dataset() -> pd.DataFrame:
    """Create a small synthetic student dataset."""

    return pd.DataFrame({
        "Age": [
            18, 19, 20, 21, 22, 18, 23, 20, 24, 19,
            21, 22, 20, 23, 18, 24, 21, 22, 19, 23
        ],
        "Attendance": [
            55, 60, 65, 58, 70, 75, 80, 68, 85, 72,
            78, 82, 88, 92, 75, 95, 86, 90, 62, 89
        ],
        "Education": [
            "High School", "High School", "Bachelor",
            "High School", "Bachelor", "Bachelor",
            "Master", "Bachelor", "Master", "High School",
            "Bachelor", "Master", "Bachelor", "Doctorate",
            "High School", "Doctorate", "Master", "Bachelor",
            "High School", "Master"
        ],
        "Satisfaction": [
            "Low", "Low", "Medium", "Low", "Medium",
            "High", "High", "Medium", "Very High", "Medium",
            "High", "High", "Medium", "Very High", "Low",
            "Very High", "High", "Medium", "Low", "High"
        ],
        "City": [
            "Mumbai", "Pune", "Delhi", "Mumbai", "Pune",
            "Delhi", "Mumbai", "Pune", "Delhi", "Mumbai",
            "Pune", "Delhi", "Mumbai", "Pune", "Delhi",
            "Mumbai", "Pune", "Delhi", "Mumbai", "Pune"
        ],
        "Passed": [
            0, 0, 0, 0, 0, 1, 1, 0, 1, 0,
            1, 1, 0, 1, 0, 1, 1, 1, 0, 1
        ]
    })


def build_preprocessor() -> ColumnTransformer:
    """Create type-specific preprocessing pipelines."""

    numeric_features = ["Age", "Attendance"]
    ordinal_features = ["Education", "Satisfaction"]
    nominal_features = ["City"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                categories=[
                    [
                        "High School",
                        "Bachelor",
                        "Master",
                        "Doctorate"
                    ],
                    [
                        "Low",
                        "Medium",
                        "High",
                        "Very High"
                    ]
                ],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )
        )
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("ordinal", ordinal_pipeline, ordinal_features),
        ("nominal", nominal_pipeline, nominal_features)
    ])


def main() -> None:
    df = create_dataset()

    X = df.drop(columns=["Passed"])
    y = df["Passed"]

    print("Dataset shape:", df.shape)
    print("\nOriginal data:")
    print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Fit the encoder, imputer, scaler, and classifier on training data.
    model.fit(X_train, y_train)

    # Predict using the fitted preprocessing and model pipeline.
    predictions = model.predict(X_test)

    print("\nTraining accuracy:")
    print(accuracy_score(y_train, model.predict(X_train)))

    print("\nTest accuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Inspect the learned ordinal category order.
    fitted_preprocessor = model.named_steps["preprocessor"]
    fitted_ordinal_pipeline = fitted_preprocessor.named_transformers_["ordinal"]
    fitted_encoder = fitted_ordinal_pipeline.named_steps["encoder"]

    print("\nLearned ordinal category definitions:")
    print(fitted_encoder.categories_)


if __name__ == "__main__":
    main()
```

### What this project demonstrates

- Creating a reproducible student dataset.
- Defining the correct order of ordinal features.
- Encoding ordinal and nominal columns differently.
- Handling missing values through type-specific pipelines.
- Scaling numerical features.
- Splitting the dataset before fitting learned preprocessing.
- Training a Logistic Regression classifier.
- Evaluating the model on held-out test observations.
- Inspecting the fitted encoder's categories.

The dataset is small and artificial, so the model scores should not be interpreted as evidence of real-world predictive performance. The purpose is to demonstrate the preprocessing architecture.

## ⚠️ Common Mistakes

### 1. Assigning arbitrary orders

Incorrect:

```text
Mumbai → 0
Pune   → 1
Delhi  → 2
```

These cities do not have an inherent order.

**Solution:** Use One-Hot Encoding for nominal features unless a meaningful ordering is explicitly defined.

### 2. Assuming alphabetic order equals semantic order

Automatically inferred categories may be sorted alphabetically rather than according to your business definition.

**Solution:** Define the category sequence explicitly for ordinal variables.

### 3. Treating ordinal codes as equal measurements

An encoding of `0`, `1`, `2`, and `3` does not prove that adjacent categories are equally far apart.

**Solution:** Understand how the model interprets numerical features and compare appropriate encoding strategies.

### 4. Using `LabelEncoder` for input features

`LabelEncoder` is designed primarily for encoding target labels, not general input feature columns.

**Solution:** Use `OrdinalEncoder` for ordered categorical input features.

### 5. Ignoring unknown categories

New category values may appear after deployment.

**Solution:** Define an explicit unknown-category policy and monitor new values.

### 6. Losing missing values during mapping

Pandas mapping can produce missing values when the input is absent from the mapping.

**Solution:** Inspect unmapped values and distinguish missing inputs from unrecognized categories.

### 7. Fitting separate encoders on training and test data

Independent fitting may assign inconsistent codes.

**Solution:** Fit once on training data and reuse the fitted encoder.

### 8. Encoding nominal features as ordinal

Numerically encoding unordered categories creates an artificial ranking.

**Solution:** Select the encoding method based on the meaning of the variable.

### 9. Assuming ordinal encoding always improves model performance

A model may interpret ordinal codes as numeric distances or thresholds.

**Solution:** Evaluate the representation using an appropriate validation strategy.

### 10. Ignoring preprocessing leakage

Learning category representations from held-out data can contaminate a machine learning evaluation.

**Solution:** Place learned preprocessing inside a Scikit-learn pipeline and fit it on the training partition only.

## 🌟 Best Practices

- Understand each categorical feature before encoding it.
- Use ordinal encoding only when a meaningful order exists.
- Document the permitted categories and their order.
- Prefer explicit mappings or explicit `categories` definitions when order matters.
- Use One-Hot Encoding for nominal variables when appropriate.
- Preserve raw columns for traceability.
- Inspect unknown categories instead of silently accepting them.
- Define a deliberate strategy for missing values.
- Avoid treating unknown-category sentinels as genuine ranks.
- Fit learned transformations on training data only.
- Use `ColumnTransformer` for mixed numerical, ordinal, and nominal features.
- Compare preprocessing alternatives using validation data.
- Test category handling with missing and previously unseen values.
- Monitor for new categories after deployment.
- Save and reuse the fitted preprocessing pipeline for production predictions.

## 💡 Mini Projects

### Project 1: Student Education Encoding

Create a dataset containing education levels.

**Tasks:**

1. Define an education hierarchy.
2. Encode it using manual mapping.
3. Encode it using `OrdinalEncoder`.
4. Compare both outputs.
5. Test missing and unknown categories.
6. Document the ranking assumptions.

### Project 2: Customer Satisfaction Encoding

Create a customer survey dataset containing ratings from Very Low to Very High.

**Tasks:**

1. Establish a rating order.
2. Apply ordinal encoding.
3. Inspect category frequencies.
4. Detect unexpected rating labels.
5. Compare a model using ordinal encoding with a suitable alternative.

### Project 3: Employee Experience Encoding

Create an employee dataset with experience tiers, department, and salary.

**Tasks:**

1. Encode experience as an ordered feature.
2. Encode department as a nominal feature.
3. Build a `ColumnTransformer`.
4. Fit preprocessing on the training data.
5. Evaluate a regression or classification model.

### Project 4: End-to-End Categorical Preprocessing

Build a machine learning pipeline containing numeric, ordinal, and nominal features.

**Tasks:**

1. Inspect the feature types.
2. Define category orderings.
3. Impute missing values.
4. Apply ordinal and one-hot encoders to suitable columns.
5. Train a classification model.
6. Test unknown-category handling.
7. Evaluate performance without contaminating the test set.

## 📝 Practice Exercises

1. Encode `Low`, `Medium`, and `High` manually.
2. Encode `Beginner`, `Intermediate`, `Advanced`, and `Expert`.
3. Create an education ranking and apply it using `Series.map()`.
4. Use `CategoricalDtype` to define an ordered Pandas column.
5. Encode an ordered feature with Scikit-learn's `OrdinalEncoder`.
6. Encode three ordinal columns with different category sequences.
7. Identify values that fail a manual mapping.
8. Handle missing values and unknown categories separately.
9. Convert an encoded feature back using `inverse_transform()`.
10. Build a pipeline for numerical, ordinal, and nominal columns.
11. Explain why city names should not normally be ordinal-encoded.
12. Explain why ordinal codes do not imply equal distances between categories.
13. Demonstrate why an encoder should be fitted only on training data.
14. Compare ordinal and one-hot encodings for a suitable model.

## ⚡ Quick Revision

| Task | Method |
|---|---|
| Manual ordinal mapping | `series.map(mapping)` |
| Define ordered Pandas categories | `pd.CategoricalDtype(categories=..., ordered=True)` |
| Get Pandas category codes | `series.cat.codes` |
| Encode ordered features | `OrdinalEncoder(categories=...)` |
| Inspect categories | `encoder.categories_` |
| Encode unknown categories | `handle_unknown="use_encoded_value"` |
| Choose an unknown-value code | `unknown_value=-1`, when suitable |
| Encode missing categories separately | `encoded_missing_value=...` |
| Reverse encoded values | `encoder.inverse_transform()` |
| Encode nominal features | `OneHotEncoder()` |
| Encode classification target labels | `LabelEncoder`, when appropriate |
| Combine feature transformations | `ColumnTransformer` |
| Prevent common preprocessing leakage | `Pipeline` |
| Inspect unmapped values | Compare original values with mapped results |

## 📂 Project Structure

Recommended repository structure:

```text
07-Data-Preprocessing/
│
├── README.md
│
├── 01-Train-Test-Split/
│   └── README.md
│
├── 02-Feature-Scaling/
│   └── README.md
│
├── 03-Standardization/
│   └── README.md
│
├── 04-Normalization/
│   └── README.md
│
├── 05-Label-Encoding/
│   └── README.md
│
├── 06-One-Hot-Encoding/
│   └── README.md
│
└── 07-Ordinal-Encoding/
    ├── README.md
    ├── ordinal_encoding.py
    ├── manual_ordinal_encoding.py
    ├── pandas_ordinal_encoding.py
    ├── sklearn_ordinal_encoding.py
    ├── unknown_categories.py
    └── ordinal_encoding_pipeline.py
```

| File | Purpose |
|---|---|
| `README.md` | Theory, methods, examples, and best practices |
| `ordinal_encoding.py` | Complete machine learning example |
| `manual_ordinal_encoding.py` | Encoding with Python dictionaries |
| `pandas_ordinal_encoding.py` | Ordered categorical operations with Pandas |
| `sklearn_ordinal_encoding.py` | Explicit category encoding using Scikit-learn |
| `unknown_categories.py` | Missing and unseen-category handling |
| `ordinal_encoding_pipeline.py` | Mixed-feature preprocessing pipeline |

Create the individual Python files as you expand this topic. The examples in this README provide a starting point for each script.

## 🗺️ Learning Roadmap

Follow this progression to develop practical categorical preprocessing skills:

1. Understand categorical feature types.
2. Distinguish ordinal and nominal data.
3. Define meaningful category hierarchies.
4. Learn manual ordinal mapping.
5. Use Pandas ordered categorical data.
6. Learn Scikit-learn's `OrdinalEncoder`.
7. Handle missing and unknown categories.
8. Combine ordinal encoding with numeric preprocessing and One-Hot Encoding.
9. Integrate preprocessing into machine learning pipelines.
10. Evaluate representations and monitor incoming category changes.

## 🎯 Key Takeaways

- Ordinal encoding assigns numerical codes to categories with a meaningful order.
- The order must come from the data's semantics or a documented business rule.
- Explicit category lists are safer than automatically inferred orders for ordinal variables.
- Pandas mapping, `CategoricalDtype`, and Scikit-learn's `OrdinalEncoder` provide different ways to represent ordinal data.
- Unknown categories and missing values require deliberate handling.
- Nominal features should not be assigned arbitrary ranks.
- Encoded values do not necessarily represent equal numerical distances.
- Scikit-learn pipelines help prevent common preprocessing mistakes and data leakage.
- The best encoding method depends on the feature, model, and prediction objective.

## 👨‍💻 Author

**Kishor Patil**

Python | Data Science | Machine Learning | Artificial Intelligence

GitHub: [Kishor055](https://github.com/Kishor055)

Repository: [Machine-Learning](https://github.com/Kishor055/Machine-Learning)

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Add examples, improvements, or corrections.
4. Test the code.
5. Submit a pull request with a clear description.

## ⭐ Support

If this guide helps you learn data preprocessing and machine learning, consider starring the repository on GitHub.

**Keep learning, keep building, and keep growing!** 🌱

## 📖 References

- [Scikit-learn: OrdinalEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)
- [Scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn: OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Pandas: Categorical data](https://pandas.pydata.org/docs/user_guide/categorical.html)

---

*Part of the Machine Learning learning path: Data Collection → Data Cleaning → Exploratory Data Analysis → Data Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment.*

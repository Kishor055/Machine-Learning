# 🏷️ Label Encoding in Machine Learning

**Label Encoding = Identify Categories → Assign Numerical Labels → Transform Data → Prepare Features for ML**

Label encoding is a categorical data preprocessing technique that converts category labels into numerical values. It is useful for preparing target labels and certain categorical features for machine learning workflows.

---

## 🌱 GROW → LEARN → ENCODE → PREPROCESS → TRAIN → BUILD 🚀

> **Goal:** Understand label encoding, implement it using Python and Scikit-Learn, avoid accidental ordinal relationships, and select the right encoding method for your machine learning project.

## 📚 Table of Contents

1. Introduction
2. What Is Label Encoding?
3. Why Is Label Encoding Important?
4. How Label Encoding Works
5. Label Encoding Using Python
6. Label Encoding Using Pandas
7. Label Encoding Using Scikit-Learn
8. Label Encoding Target Variables
9. Encoding Categorical Features
10. Label Encoding vs One-Hot Encoding
11. Ordinal Encoding
12. Handling Unknown Categories
13. Avoiding Data Leakage
14. Label Encoding in ML Pipelines
15. Complete Classification Example
16. Common Mistakes
17. Best Practices
18. Mini Projects
19. Practice Exercises
20. Quick Revision
21. Project Structure
22. Learning Roadmap
23. Key Takeaways
24. What's Next?

---

## 📌 Introduction

Machine learning algorithms generally work with numerical data. However, real-world datasets often contain categorical values such as:

- Colors: `Red`, `Blue`, `Green`
- Departments: `HR`, `Finance`, `Engineering`
- Education levels: `School`, `Bachelor's`, `Master's`, `PhD`
- Customer types: `Regular`, `Premium`, `Enterprise`
- Target classes: `Spam`, `Not Spam`

Label encoding converts these labels into numeric representations.

### Example

Before encoding:

| Customer ID | Membership |
| ----------- | ---------- |
| 101         | Bronze     |
| 102         | Silver     |
| 103         | Gold       |
| 104         | Bronze     |

After encoding:

| Customer ID | Membership |
| ----------- | ---------- |
| 101         | 0          |
| 102         | 1          |
| 103         | 2          |
| 104         | 0          |

The assigned numbers are category identifiers. Unless the categories have a meaningful order and the encoding represents that order, the numbers should **not** be interpreted as measurements or rankings.

---

## 🎯 What Is Label Encoding?

Label encoding maps each distinct category to an integer.

For example:

```
Red    → 0
Blue   → 1
Green  → 2
Red    → 0
Green  → 2
```

The mapping is reused for repeated occurrences of the same category.

### Mathematical representation

Let (C) be a set of categories:

[\
C={c_1,c_2,\ldots,c_k}\
]

Label encoding defines a mapping:

[\
f\rightarrow{0,1,\ldots,k-1}\
]

Each category receives one integer label.

**Important:** The assigned integers do not automatically preserve semantic relationships. For example, encoding `Red = 0`, `Blue = 1`, and `Green = 2` does not mean that green is greater than blue.

---

## 💡 Why Is Label Encoding Important?

### 1. Converts text labels into integers

Some algorithms and machine learning interfaces require the target classes to be represented numerically.

### 2. Helps prepare classification targets

Labels such as `Pass` and `Fail` can be represented as `0` and `1` when that mapping is appropriate.

### 3. Reduces representation size

A single encoded column can represent many categories, unlike one-hot encoding, which creates a separate indicator column for each category.

### 4. Supports ordered categories

For genuinely ordinal features, an explicitly defined mapping can preserve a meaningful order.

### 5. Creates consistent mappings

A fitted encoder can reuse the same category mapping during prediction.

However, a compact integer representation is not always a good representation for a model. For nominal input features, arbitrary integers may introduce misleading relationships.

---

## 🔢 How Label Encoding Works

Consider a dataset:

```
colors = ["Red", "Blue", "Green", "Blue", "Red"]
```

### Step 1: Identify unique categories

```
Red, Blue, Green
```

### Step 2: Assign integer labels

For this example, define:

```
Blue  → 0
Green → 1
Red   → 2
```

### Step 3: Transform the values

```
Original: ["Red", "Blue", "Green", "Blue", "Red"]
Encoded:  [2, 0, 1, 0, 2]
```

The mapping is an implementation choice. Different encoders may assign different integers depending on their rules.

---

## 🐍 Label Encoding Using Python

You can implement a simple encoder using a dictionary.

```
def label_encode(values):
    """
    Encode categorical values using first-appearance order.

    Returns:
        encoded_values: List of integer labels.
        mapping: Dictionary mapping categories to integers.
    """
    mapping = {}
    encoded_values = []

    for value in values:
        if value not in mapping:
            mapping[value] = len(mapping)

        encoded_values.append(mapping[value])

    return encoded_values, mapping


def main():
    colors = ["Red", "Blue", "Green", "Blue", "Red"]

    encoded, mapping = label_encode(colors)

    print("Original values:", colors)
    print("Mapping:", mapping)
    print("Encoded values:", encoded)


if __name__ == "__main__":
    main()
```

Example output:

```
Original values: ['Red', 'Blue', 'Green', 'Blue', 'Red']
Mapping: {'Red': 0, 'Blue': 1, 'Green': 2}
Encoded values: [0, 1, 2, 1, 0]
```

### Advantages

- Easy to understand.
- Uses basic Python concepts.
- Makes the mapping explicit.
- Useful for learning how encoding works.

### Limitations

This simple implementation is intended for learning. Before using it in a production workflow, decide how to handle missing values, mixed data types, unknown categories, and consistent mappings across datasets.

---

## 🐼 Label Encoding Using Pandas

Pandas provides several ways to work with categorical values.

### Method 1: `factorize()`

```
import pandas as pd

colors = pd.Series(
    ["Red", "Blue", "Green", "Blue", "Red"]
)

encoded, categories = pd.factorize(colors)

print("Encoded values:", encoded)
print("Categories:", categories.tolist())
```

`factorize()` assigns integer codes based on its category-discovery behavior. By default, missing values receive code `-1`.

Be careful: `-1` can be a valid numeric value in other contexts. Handle missing categories explicitly before passing encoded data to a model.

### Method 2: `Categorical`

```
import pandas as pd

colors = pd.Series(
    ["Red", "Blue", "Green", "Blue", "Red"]
)

categorical = pd.Categorical(colors)

print("Category codes:", categorical.codes)
print("Categories:", categorical.categories.tolist())
```

Pandas categorical codes depend on the category ordering. They are not automatically meaningful numeric measurements.

### Method 3: Explicit mapping

When you want full control over the mapping:

```
import pandas as pd

df = pd.DataFrame(
    {
        "education": [
            "High School",
            "Bachelor",
            "Master",
            "PhD",
            "Bachelor",
        ]
    }
)

education_mapping = {
    "High School": 0,
    "Bachelor": 1,
    "Master": 2,
    "PhD": 3,
}

df["education_encoded"] = df["education"].map(
    education_mapping
)

print(df)
```

This explicit mapping is suitable only if the ordering is intentional and meaningful.

If a category is missing from the dictionary, `.map()` returns a missing value. Validate the mapping before training.

---

## 🤖 Label Encoding Using Scikit-Learn

Scikit-Learn provides `LabelEncoder` for encoding target labels.

### Basic example

```
from sklearn.preprocessing import LabelEncoder

colors = ["Red", "Blue", "Green", "Blue", "Red"]

encoder = LabelEncoder()

encoded_colors = encoder.fit_transform(colors)

print("Encoded values:", encoded_colors)
print("Classes:", encoder.classes_)
```

`LabelEncoder` learns the classes and provides a consistent integer mapping for them.

### Convert integers back to labels

```
from sklearn.preprocessing import LabelEncoder

labels = ["Cat", "Dog", "Bird", "Dog", "Cat"]

encoder = LabelEncoder()

encoded = encoder.fit_transform(labels)
decoded = encoder.inverse_transform(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)
```

### Inspect the mapping

```
from sklearn.preprocessing import LabelEncoder

labels = ["Bronze", "Silver", "Gold"]

encoder = LabelEncoder()
encoder.fit(labels)

for label, code in zip(
    encoder.classes_,
    range(len(encoder.classes_)),
):
    print(f"{label} -> {code}")
```

`LabelEncoder` generally orders its discovered classes according to its class-sorting behavior. Do not rely on a manually assumed mapping; inspect `classes_` when you need to understand the learned mapping.

**Best practice:** Use `LabelEncoder` primarily for a one-dimensional target variable `y`. For categorical input features `X`, consider `OneHotEncoder` or `OrdinalEncoder` as appropriate.

---

## 🎯 Label Encoding Target Variables

Target labels are the outcomes a supervised model learns to predict.

For example:

```
Pass
Fail
Pass
Pass
Fail
```

These can be encoded as:

```
Fail → 0
Pass → 1
```

### Example

```
from sklearn.preprocessing import LabelEncoder

y = [
    "Pass",
    "Fail",
    "Pass",
    "Pass",
    "Fail",
]

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print("Original target:", y)
print("Encoded target:", y_encoded)
print("Target classes:", target_encoder.classes_)
```

### Why encode target labels?

- Some model APIs expect numeric class labels.
- Evaluation tools can work with encoded classes.
- Integer class IDs can be convenient for downstream processing.

Many Scikit-Learn classifiers can also accept string target labels directly. Encoding the target is not always required.

### Recover original class names

```
predicted_codes = target_encoder.transform(
    ["Pass", "Fail"]
)

predicted_labels = target_encoder.inverse_transform(
    predicted_codes
)

print(predicted_labels)
```

For a deployed model, preserve the fitted target encoder with the model if you use one, so predictions can be converted back into the original labels consistently.

---

## 🧩 Encoding Categorical Features

A feature is an input used to predict the target. Not every categorical feature should be label-encoded.

Consider:

| Color | Target |
| ----- | ------ |
| Red   | Yes    |
| Blue  | No     |
| Green | Yes    |

If you encode the color values as `Red = 0`, `Blue = 1`, and `Green = 2`, a model may treat those numbers as having a numerical order or distance.

For many algorithms, this is misleading because colors are nominal categories.

### What should you use?

- **Nominal categories:** Usually use one-hot encoding.
- **Ordinal categories:** Use an explicitly ordered encoding when the ordering is meaningful.
- **Target labels:** `LabelEncoder` may be suitable, although many classifiers accept strings.
- **High-cardinality categories:** Consider one-hot encoding, hashing, or carefully validated alternatives based on the model and data.

---

## ⚖️ Label Encoding vs One-Hot Encoding

| Property                             | Label Encoding           | One-Hot Encoding                           |
| ------------------------------------ | ------------------------ | ------------------------------------------ |
| Output                               | Integer codes            | Binary indicator columns                   |
| Number of columns                    | Usually one              | One per category, subject to configuration |
| Implies numeric order?               | Can imply one to a model | Avoids a simple ordinal code               |
| Suitable for nominal input features? | Often not ideal          | Common choice                              |
| Suitable for target labels?          | Yes                      | Sometimes, depending on model interface    |
| Memory use                           | Usually compact          | Can be large with many categories          |

### Label encoding example

```
Color: Red, Blue, Green

Red   → 0
Blue  → 1
Green → 2
```

### One-hot encoding example

| Color | Red | Blue | Green |
| ----- | --- | ---- | ----- |
| Red   | 1   | 0    | 0     |
| Blue  | 0   | 1    | 0     |
| Green | 0   | 0    | 1     |

### Scikit-Learn one-hot encoding

```
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

X = pd.DataFrame(
    {
        "color": ["Red", "Blue", "Green", "Blue"]
    }
)

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)

X_encoded = encoder.fit_transform(X)

print(encoder.get_feature_names_out())
print(X_encoded)
```

`OneHotEncoder` is commonly the safer default for nominal categorical input features in linear models, distance-based models, and many other algorithms.

For compatibility with older Scikit-Learn versions, `sparse_output=False` may need to be replaced with `sparse=False`.

---

## 📐 Ordinal Encoding

Ordinal encoding is used when categories have a meaningful order.

Examples include:

```
Low < Medium < High
Beginner < Intermediate < Advanced
Poor < Fair < Good < Excellent
```

For a genuinely ordinal feature, you can explicitly define the mapping.

### Example using Scikit-Learn

```
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

X = np.array(
    [
        ["Low"],
        ["Medium"],
        ["High"],
        ["Medium"],
    ],
    dtype=object,
)

encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)

X_encoded = encoder.fit_transform(X)

print(X_encoded)
```

Output:

```
[[0.]
 [1.]
 [2.]
 [1.]]
```

The order is specified deliberately instead of being inferred from an arbitrary category list.

**Important:** Ordinal encoding gives the categories numeric codes, but the model may also interpret the numeric gaps as meaningful. This may be acceptable for some models and unsuitable for others.

---

## 🛡️ Handling Unknown Categories

An unknown category is a value that appears during validation, testing, or prediction but was not present when the encoder was fitted.

For example:

```
Training categories: Red, Blue, Green
New category:        Yellow
```

A fitted `LabelEncoder` raises an error when asked to transform an unseen label.

### Example

```
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(["Red", "Blue", "Green"])

# This raises ValueError because Yellow is unknown.
# encoder.transform(["Yellow"])
```

### Recommended strategies

1. Use `OneHotEncoder(handle_unknown="ignore")` for suitable nominal input features.
2. Use `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` when that behavior suits the model and pipeline.
3. Define an explicit unknown-category policy for custom encoders.
4. Monitor new categories in production.
5. Avoid silently mapping unknown categories to arbitrary valid labels.

Example:

```
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

X_train = np.array(
    [["Small"], ["Medium"], ["Large"]],
    dtype=object,
)

X_new = np.array(
    [["Medium"], ["Extra Large"]],
    dtype=object,
)

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

encoder.fit(X_train)

print(encoder.transform(X_new))
```

The unknown category receives `-1`. Make sure your chosen model can handle this representation appropriately.

---

## 🚨 Avoiding Data Leakage

Data leakage occurs when information that should be unavailable during training influences the learned model or its evaluation.

For label encoding, the risks depend on the method:

- Defining a fixed, domain-based mapping in advance is often safe.
- Learning categories from the complete dataset can reveal information about the test distribution.
- Target encoding and other target-dependent encodings require particular care because they can leak information from the target.
- Inconsistent mappings between training and inference can corrupt predictions.

### Recommended workflow

1. Split the dataset into training and test sets.
2. Fit preprocessing steps using training data only.
3. Apply the fitted transformations to validation and test data.
4. Use pipelines where possible.
5. Keep target-dependent encoding inside a properly cross-validated workflow.

For a simple target label mapping, the encoder does not learn predictive relationships between the input features and the target. Nevertheless, preserving a consistent mapping is important for interpreting predictions.

---

## 🔗 Label Encoding in ML Pipelines

Use `ColumnTransformer` and `Pipeline` to handle categorical and numerical input features together.

This example uses one-hot encoding for a nominal feature and Min-Max scaling for numerical features.

```
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def main():
    df = pd.DataFrame(
        {
            "age": [22, 35, 28, 42, 31, 26, 45, 39],
            "salary": [
                30000, 65000, 42000, 80000,
                55000, 38000, 90000, 72000,
            ],
            "city": [
                "Pune", "Mumbai", "Pune", "Delhi",
                "Mumbai", "Delhi", "Pune", "Mumbai",
            ],
            "purchased": [
                "No", "Yes", "No", "Yes",
                "Yes", "No", "Yes", "Yes",
            ],
        }
    )

    X = df[["age", "salary", "city"]]
    y = df["purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    numeric_features = ["age", "salary"]
    categorical_features = ["city"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    print("Test accuracy:", model.score(X_test, y_test))
    print("Predictions:", model.predict(X_test))


if __name__ == "__main__":
    main()
```

### Why this approach is better

- Numerical features are imputed and scaled.
- Nominal categorical features are imputed and one-hot encoded.
- The target remains as readable string labels, which Logistic Regression can handle through Scikit-Learn.
- The preprocessing steps are fitted during training.
- Unknown categories in the nominal feature are handled by the one-hot encoder.
- The full workflow can be saved and reused for prediction.

The small dataset here demonstrates the workflow, not a reliable estimate of real-world model performance.

---

## ❌ Common Mistakes

1. Applying `LabelEncoder` to every categorical feature automatically.
2. Treating arbitrary integer labels as meaningful numerical values.
3. Forgetting the difference between nominal and ordinal categories.
4. Assuming category codes have the same meaning across separately fitted encoders.
5. Ignoring unknown categories during inference.
6. Encoding missing values as ordinary categories without an explicit policy.
7. Losing the mapping needed to interpret model predictions.
8. Applying target-dependent encoding without cross-validation safeguards.
9. Fitting transformations inconsistently between training and production.
10. Assuming label encoding is always required for classification targets.

---

## ✅ Best Practices

- Identify whether each categorical variable is nominal or ordinal.
- Use explicit mappings for categories with known business meaning.
- Prefer one-hot encoding for many nominal input features.
- Use ordinal encoding only when the order is meaningful.
- Use `LabelEncoder` primarily for target labels.
- Preserve the fitted encoder and its class mapping when needed.
- Decide how unknown categories should be handled.
- Handle missing values consistently.
- Use pipelines for repeatable preprocessing.
- Validate model performance instead of assuming encoding improves accuracy.
- Document category definitions and mapping rules.

---

## 🚀 Mini Projects

### Project 1: Student Result Classification

**Objective:** Convert categorical target labels into numerical labels.

Tasks:

- Create a dataset with `Pass` and `Fail` labels.
- Encode the target using `LabelEncoder`.
- Inspect the mapping.
- Train a simple classifier.
- Convert predictions back to the original labels.

### Project 2: Customer Category Encoding

**Objective:** Compare encoding strategies for nominal categories.

Tasks:

- Create a dataset with city, membership type, and spending.
- Encode nominal features using one-hot encoding.
- Compare the resulting feature matrix with integer label encoding.
- Train a suitable model.
- Explain why arbitrary integer codes may be misleading.

### Project 3: Education Level Encoding

**Objective:** Encode a genuinely ordinal feature.

Tasks:

- Define the order from High School to PhD.
- Implement an explicit mapping.
- Use `OrdinalEncoder` with a defined category order.
- Inspect the transformed values.
- Explain why the order must be chosen intentionally.

### Project 4: Unknown Category Handling

**Objective:** Build a robust preprocessing workflow.

Tasks:

- Fit an encoder on a training dataset.
- Introduce a new category in a test dataset.
- Observe the behavior of different encoders.
- Implement an unknown-category policy.
- Verify that prediction-time transformations remain consistent.

---

## 📝 Practice Exercises

1. What is label encoding?
2. Why do machine learning algorithms often require numerical input?
3. Encode `["Apple", "Banana", "Orange", "Apple"]` using first-appearance order.
4. What is the difference between nominal and ordinal categorical data?
5. Why is label encoding not always suitable for nominal input features?
6. How does `LabelEncoder` differ from `OrdinalEncoder`?
7. What does `OneHotEncoder` do?
8. What happens when `LabelEncoder` encounters an unknown category?
9. Why is a fixed mapping useful in production?
10. How can pipelines make preprocessing more reliable?
11. When is ordinal encoding appropriate?
12. What is target leakage, and why is target encoding particularly sensitive to it?

---

## ⚡ Quick Revision

| Concept             | Key point                                                                   |
| ------------------- | --------------------------------------------------------------------------- |
| Label encoding      | Maps categories to integers                                                 |
| `LabelEncoder`      | Commonly used for target labels                                             |
| `OrdinalEncoder`    | Encodes categorical input features, including explicitly ordered categories |
| One-hot encoding    | Creates binary indicators for nominal categories                            |
| Nominal data        | Categories have no inherent order                                           |
| Ordinal data        | Categories have a meaningful order                                          |
| Unknown categories  | Need an explicit handling strategy                                          |
| Mapping consistency | Essential for reliable inference                                            |
| Data leakage        | Prevent by fitting preprocessing appropriately                              |
| Pipeline            | Combines transformations and model training                                 |

---

## 📁 Project Structure

```
07-Data-Preprocessing/
│
├── README.md
├── 01-Train-Test-Split/
│   └── README.md
├── 02-Feature-Scaling/
│   └── README.md
├── 03-Standardization/
│   └── README.md
├── 04-Normalization/
│   └── README.md
└── 05-Label-Encoding/
    ├── README.md
    ├── manual_label_encoding.py
    ├── pandas_label_encoding.py
    ├── sklearn_label_encoding.py
    ├── target_encoding.py
    ├── ordinal_encoding.py
    ├── one_hot_encoding.py
    └── encoding_pipeline.py
```

---

## 🗺️ Learning Roadmap

1. Understand categorical data.
2. Learn nominal and ordinal variables.
3. Implement label encoding with Python.
4. Explore Pandas category codes.
5. Use Scikit-Learn `LabelEncoder`.
6. Learn `OrdinalEncoder`.
7. Master `OneHotEncoder`.
8. Handle unknown and missing categories.
9. Build preprocessing pipelines.
10. Apply encoding in a complete ML project.

---

## 🎯 Key Takeaways

- Label encoding converts categories into integer codes.
- Numeric codes do not automatically represent a meaningful order.
- `LabelEncoder` is generally intended for target labels.
- `OrdinalEncoder` is designed for categorical input features.
- One-hot encoding is often preferable for nominal input features.
- Unknown categories and missing values require deliberate handling.
- Consistent mappings are essential for reliable predictions.
- Pipelines help keep preprocessing consistent and reduce leakage risks.

---

## 🔜 What's Next?

Continue your Machine Learning journey with **One-Hot Encoding**, where you will learn how to represent nominal categories using binary feature columns.

Explore the full repository: [Machine Learning — GitHub](https://github.com/Kishor055/Machine-Learning)

---

## 👨‍💻 Author

**Kishor Patil**

GitHub: [Kishor055](https://github.com/Kishor055)

## 🤝 Contributing

Contributions are welcome! Improve explanations, add examples, fix issues, or suggest additional exercises.

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Submit a pull request with a clear description.

## ⭐ Support

If this repository helps you learn Machine Learning, consider giving it a star on GitHub.

## 📄 License

Refer to the repository's root `LICENSE` file for licensing terms.

---

**Keep learning. Keep building. Keep growing. 🚀**

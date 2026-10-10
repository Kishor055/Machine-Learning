# 06 - One-Hot Encoding

## 📌 Overview

**One-Hot Encoding** is a feature encoding technique used in Machine Learning to convert categorical data into numerical format. Most machine learning algorithms require numerical input, so categorical values such as `"Red"`, `"Blue"`, and `"Green"` must be converted into numbers.

One-Hot Encoding creates a separate binary column for each unique category. A value of `1` indicates the presence of a category, while `0` indicates its absence.

## 🎯 Learning Objectives

- Understand categorical data and feature encoding.
- Learn how One-Hot Encoding works.
- Implement One-Hot Encoding using Pandas and Scikit-learn.
- Handle unseen categories in training and testing datasets.
- Avoid common encoding mistakes and data leakage.
- Understand the advantages and limitations of One-Hot Encoding.

## 📂 Project Structure

```
06-One-Hot-Encoding/
│
├── README.md
├── one_hot_encoding.py
├── one_hot_encoding_pandas.py
├── one_hot_encoding_sklearn.py
└── requirements.txt
```

## 📖 1. What Is One-Hot Encoding?

Suppose a dataset contains a categorical feature called `Color`.

Original data:

| Color |
| ----- |
| Red   |
| Blue  |
| Green |
| Red   |
| Blue  |

After One-Hot Encoding:

| Color_Red | Color_Blue | Color_Green |
| --------- | ---------- | ----------- |
| 1         | 0          | 0           |
| 0         | 1          | 0           |
| 0         | 0          | 1           |
| 1         | 0          | 0           |
| 0         | 1          | 0           |

Each category receives a separate binary column.

## 🧠 2. Why Do We Need One-Hot Encoding?

Categorical variables contain labels rather than numerical measurements.

For example:

```
colors = ["Red", "Blue", "Green"]
```

Assigning arbitrary numbers such as `Red = 0`, `Blue = 1`, and `Green = 2` may incorrectly suggest an order or numerical relationship.

One-Hot Encoding represents categories without introducing this artificial ordering.

### Common use cases

- City names
- Product categories
- Department names
- Color labels
- Gender categories
- Payment methods
- Education streams

## ⚙️ 3. How One-Hot Encoding Works

1. Identify the categorical feature.
2. Find its unique categories.
3. Create one binary column for each category.
4. Assign `1` to the matching category.
5. Assign `0` to all other categories.

For example:

```
Category: Payment_Method

Cash  → [1, 0, 0]
Card  → [0, 1, 0]
UPI   → [0, 0, 1]
```

## 🐼 4. Implementation Using Pandas

The `pd.get_dummies()` function converts categorical columns into indicator columns.

```
import pandas as pd

# Create a sample dataset
df = pd.DataFrame({
    "Student": ["Amit", "Priya", "Rahul", "Sneha", "Karan"],
    "City": ["Pune", "Mumbai", "Delhi", "Pune", "Mumbai"],
    "Marks": [85, 92, 78, 88, 80]
})

print("Original Dataset:")
print(df)

# Apply One-Hot Encoding
encoded_df = pd.get_dummies(
    df,
    columns=["City"],
    dtype=int
)

print("\nEncoded Dataset:")
print(encoded_df)
```

### Expected output

```
Original Dataset:

  Student    City  Marks
0    Amit    Pune     85
1   Priya  Mumbai     92
2   Rahul   Delhi     78
3   Sneha    Pune     88
4   Karan  Mumbai     80
```

Encoded columns:

```
Student  Marks  City_Delhi  City_Mumbai  City_Pune
Amit       85       0            0           1
Priya      92       0            1           0
Rahul      78       1            0           0
Sneha      88       0            0           1
Karan      80       0            1           0
```

**Note:** Pandas creates dummy columns for the categories present in the data being encoded.

## 🤖 5. Implementation Using Scikit-learn

Scikit-learn provides `OneHotEncoder` for machine learning pipelines.

```
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample categorical data
df = pd.DataFrame({
    "City": ["Pune", "Mumbai", "Delhi", "Pune", "Mumbai"]
})

# Create the encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=int
)

# Fit and transform the data
encoded_array = encoder.fit_transform(df[["City"]])

# Convert the output to a DataFrame
encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(["City"])
)

print(encoded_df)
```

### Expected output

```
   City_Delhi  City_Mumbai  City_Pune
0            0            0          1
1            0            1          0
2            1            0          0
3            0            0          1
4            0            1          0
```

**Compatibility note:** `sparse_output=False` is supported in Scikit-learn 1.2 and newer. For older versions, use `sparse=False`.

## 🔍 6. Important Parameters of OneHotEncoder

| Parameter        | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `categories`     | Specifies the categories to encode or uses automatic detection. |
| `drop`           | Drops a category, such as `drop="first"`.                       |
| `handle_unknown` | Controls how unseen categories are handled.                     |
| `sparse_output`  | Returns a sparse matrix when `True`.                            |
| `dtype`          | Sets the output data type.                                      |
| `min_frequency`  | Groups infrequent categories when configured.                   |
| `max_categories` | Limits the number of output categories when configured.         |

Example:

```
encoder = OneHotEncoder(
    drop="first",
    handle_unknown="ignore",
    sparse_output=False
)
```

Dropping a category can help avoid perfect multicollinearity in certain linear models with an intercept. However, it is not necessary for every machine learning algorithm.

## 🧪 7. One-Hot Encoding with Train-Test Split

Always learn the categories from the training data and apply the same transformation to the test data.

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({
    "City": [
        "Pune", "Mumbai", "Delhi", "Pune",
        "Mumbai", "Delhi", "Pune", "Mumbai"
    ],
    "Marks": [80, 90, 75, 85, 88, 70, 92, 81]
})

X = df[["City"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

print("Training shape:", X_train_encoded.shape)
print("Testing shape:", X_test_encoded.shape)
```

### Key principle

- `fit_transform()` learns categories from training data and encodes them.
- `transform()` applies the learned encoding to test data.
- `handle_unknown="ignore"` prevents errors when a test set contains an unseen category.

Never fit the encoder separately on the training and testing datasets, because this can produce inconsistent feature columns and introduce data leakage.

## 🔄 8. One-Hot Encoding vs Label Encoding

| Feature               | One-Hot Encoding             | Label Encoding                                                     |
| --------------------- | ---------------------------- | ------------------------------------------------------------------ |
| Output                | Multiple binary columns      | Integer labels                                                     |
| Artificial ordering   | Generally avoided            | May imply ordering                                                 |
| Dimensionality        | Increases columns            | Keeps one column                                                   |
| Best suited for       | Nominal categorical features | Target labels or genuinely ordinal features with suitable ordering |
| High-cardinality data | Can become expensive         | More compact                                                       |

**Important:** For ordinal features such as `Low`, `Medium`, and `High`, use an explicitly ordered encoding when the order is meaningful. Label encoding is also commonly used for classification targets.

## ⚠️ 9. Common Mistakes

1. **Encoding the entire dataset before splitting:** Fit the encoder only on training data.
2. **Using One-Hot Encoding for high-cardinality features without evaluation:** Thousands of unique values can create thousands of columns.
3. **Ignoring unseen categories:** Configure `handle_unknown="ignore"` or another suitable strategy.
4. **Encoding numerical measurements unnecessarily:** Continuous features such as age and salary usually do not require One-Hot Encoding.
5. **Assuming the column order:** Use `get_feature_names_out()` to identify the generated features.
6. **Using arbitrary numeric labels for nominal categories:** This can introduce a misleading numerical relationship.

## ✅ 10. Advantages

- Easy to understand and implement.
- Avoids artificial ranking among nominal categories.
- Works well with many common machine learning algorithms.
- Produces explicit, interpretable binary features.
- Integrates with Scikit-learn pipelines.

## ❌ 11. Disadvantages

- Increases the number of features.
- Can consume significant memory for high-cardinality variables.
- Produces sparse data when many categories are absent in each row.
- May require additional handling for unseen categories.
- Can introduce multicollinearity in some linear models when all categories are retained with an intercept.

## 💼 12. Real-World Applications

- **Customer segmentation:** Encode customer locations and membership types.
- **E-commerce:** Encode product categories and payment methods.
- **Education:** Encode departments, courses, and academic streams.
- **Banking:** Encode account types and transaction categories.
- **Healthcare:** Encode non-numeric categorical variables.
- **Recommendation systems:** Represent product or user categories as model features.

## 🎤 13. Interview Questions and Answers

### Q1. What is One-Hot Encoding?

One-Hot Encoding is a categorical feature encoding technique that creates a separate binary feature for each category.

### Q2. Why is One-Hot Encoding preferred over arbitrary integer encoding for nominal variables?

It avoids introducing a false numerical order among categories.

### Q3. What is the difference between `pd.get_dummies()` and `OneHotEncoder`?

`pd.get_dummies()` is a convenient Pandas transformation, while Scikit-learn's `OneHotEncoder` supports learning categories during fitting, consistent transformation, unknown-category handling, and machine learning pipelines.

### Q4. What does `handle_unknown="ignore"` do?

It allows the encoder to process unseen categories during transformation without raising an error. Those categories are represented by zeros across the encoder's known output columns for that feature.

### Q5. What is the dummy variable trap?

It occurs when redundant dummy variables create perfect multicollinearity, commonly when all categories are represented alongside an intercept in a linear model.

### Q6. When should One-Hot Encoding be avoided?

It may be unsuitable for high-cardinality features where the resulting feature matrix becomes too large. Consider alternatives based on the model, data, and evaluation results.

### Q7. Can One-Hot Encoding be applied to multiple columns?

Yes. Pass multiple categorical columns to `pd.get_dummies()` or provide multiple categorical features to Scikit-learn's encoder.

### Q8. Why should an encoder be fitted only on training data?

This prevents information from the test set from influencing the learned preprocessing configuration and ensures consistent feature representation.

## 📦 14. Requirements

Create a `requirements.txt` file:

```
pandas
scikit-learn
```

Install the dependencies:

```
pip install -r requirements.txt
```

## 🚀 15. How to Run

Run the Pandas implementation:

```
python one_hot_encoding_pandas.py
```

Run the Scikit-learn implementation:

```
python one_hot_encoding_sklearn.py
```

## 📚 16. Key Takeaways

- One-Hot Encoding converts nominal categorical features into binary columns.
- Pandas provides `pd.get_dummies()`.
- Scikit-learn provides `OneHotEncoder`.
- Fit preprocessing transformations on training data only.
- Use `handle_unknown` to manage categories not observed during fitting.
- Evaluate dimensionality and memory requirements before encoding high-cardinality features.

## 🔗 References

- [Scikit-learn — OneHotEncoder Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Pandas — ](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)[`get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)[ Documentation](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
- [Scikit-learn — Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)

---

**Author:** Kishor Patil

**Topic:** Machine Learning — Data Preprocessing

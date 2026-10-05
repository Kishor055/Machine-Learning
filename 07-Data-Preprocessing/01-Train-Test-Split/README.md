
# ✂️ Train-Test Split

> **Train-Test Split = Separate Data → Train the Model → Evaluate on Unseen Data → Measure Generalization**

A **Train-Test Split** is one of the most fundamental steps in Machine Learning preprocessing.

It divides a dataset into separate subsets so that a machine learning model can **learn from one portion of the data** and be **evaluated on data it has never seen before**.

---

# 🌱 GROW → SPLIT → TRAIN → VALIDATE → TEST → GENERALIZE 🚀

```text
                    📊 Complete Dataset
                           │
                           ▼
                  ┌─────────────────┐
                  │   Shuffle Data   │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       🧠 Training Data            🧪 Test Data
       70% / 80% / 90%             30% / 20% / 10%
              │                         │
              ▼                         │
       Train ML Model                  │
              │                         │
              ▼                         │
       Learned Parameters              │
              │                         │
              └────────────┬────────────┘
                           ▼
                    Test on Unseen Data
                           │
                           ▼
                     📈 Evaluation
                           │
                           ▼
                  🌍 Generalization
```

---

## 📚 Table of Contents

1. [What is Train-Test Split?](#-what-is-train-test-split)
2. [Why Do We Split Data?](#-why-do-we-split-data)
3. [Machine Learning Data Workflow](#-machine-learning-data-workflow)
4. [Training Dataset](#-training-dataset)
5. [Testing Dataset](#-testing-dataset)
6. [Validation Dataset](#-validation-dataset)
7. [Train-Test Split vs Train-Validation-Test Split](#-train-test-split-vs-train-validation-test-split)
8. [Common Split Ratios](#-common-split-ratios)
9. [Random State](#-random-state)
10. [Shuffling](#-shuffling)
11. [Data Leakage](#-data-leakage)
12. [Stratified Train-Test Split](#-stratified-train-test-split)
13. [Regression Data Splitting](#-regression-data-splitting)
14. [Time-Series Data Splitting](#-time-series-data-splitting)
15. [Group-Based Splitting](#-group-based-splitting)
16. [Train-Test Split with Pandas](#-train-test-split-with-pandas)
17. [Train-Test Split with Scikit-Learn](#-train-test-split-with-scikit-learn)
18. [Understanding `test_size`](#-understanding-test_size)
19. [Understanding `train_size`](#-understanding-train_size)
20. [Understanding `random_state`](#-understanding-random_state-1)
21. [Understanding `shuffle`](#-understanding-shuffle)
22. [Using `stratify`](#-using-stratify)
23. [Feature-Target Splitting](#-feature-target-splitting)
24. [Complete Classification Example](#-complete-classification-example)
25. [Complete Regression Example](#-complete-regression-example)
26. [Checking the Split](#-checking-the-split)
27. [Common Mistakes](#-common-mistakes)
28. [Best Practices](#-best-practices)
29. [Train-Test Split in ML Pipelines](#-train-test-split-in-ml-pipelines)
30. [Cross-Validation](#-cross-validation)
31. [When Not to Randomly Split](#-when-not-to-randomly-split)
32. [Practical Decision Guide](#-practical-decision-guide)
33. [Mini Projects](#-mini-projects)
34. [Exercises](#-exercises)
35. [Checklist](#-checklist)
36. [Project Structure](#-project-structure)
37. [Key Takeaways](#-key-takeaways)
38. [Next Step](#-next-step)

---

# 🎯 What is Train-Test Split?

Suppose we have a dataset containing **10,000 records**.

Instead of training our model on all 10,000 records, we divide the data:

```text
10,000 Records
     │
     ├── 8,000 → Training
     │
     └── 2,000 → Testing
```

The model learns from the **8,000 training samples**.

The remaining **2,000 samples** are used to evaluate how well the model performs on unseen data.

### Basic idea

```text
Training Data
     ↓
Model learns patterns
     ↓
Model makes predictions
     ↓
Test Data
     ↓
Evaluate predictions
     ↓
Estimate generalization
```

---

# 🤔 Why Do We Split Data?

If we train and evaluate a model using exactly the same data, we cannot reliably determine whether the model can generalize to new observations.

For example:

```text
Training Accuracy = 99%
```

This might look excellent.

But if:

```text
Test Accuracy = 68%
```

the model may have **overfit** the training data.

Therefore:

> **A model should be evaluated on data that was not used to train it.**

---

# 🔄 Machine Learning Data Workflow

A typical supervised learning workflow looks like:

```text
Raw Dataset
    │
    ▼
Data Understanding
    │
    ▼
Data Cleaning
    │
    ▼
Feature / Target Separation
    │
    ▼
Train-Test Split
    │
    ├───────────────┐
    ▼               ▼
Training Set     Test Set
    │               │
    ▼               │
Preprocessing      │
    │               │
    ▼               │
Model Training     │
    │               │
    └───────┬───────┘
            ▼
       Model Evaluation
            │
            ▼
      Model Improvement
```

---

# 🧠 Training Dataset

The **training dataset** is the portion of data used to learn model parameters.

For example:

```text
Dataset = 1,000 rows

Training = 800 rows
Testing  = 200 rows
```

The model uses the training data to learn:

* relationships between features
* coefficients
* decision boundaries
* patterns
* feature importance
* model parameters

### Example

```python
model.fit(X_train, y_train)
```

The model learns from:

```text
X_train → Features
y_train → Target
```

---

# 🧪 Testing Dataset

The test dataset contains observations that should remain unseen during model training.

Example:

```python
predictions = model.predict(X_test)
```

The predictions can then be compared against:

```python
y_test
```

to calculate evaluation metrics.

For example:

```python
accuracy = accuracy_score(y_test, predictions)
```

---

# 🔍 Validation Dataset

A validation dataset is used when we need a separate dataset for:

* model selection
* hyperparameter tuning
* threshold selection
* architecture selection
* feature selection

A common structure is:

```text
Complete Dataset
       │
       ▼
┌───────────────────────┐
│                       │
▼                       ▼
Training              Test
│
▼
Validation
```

More commonly:

```text
Dataset
│
├── Training
├── Validation
└── Test
```

---

# ⚖️ Train-Test Split vs Train-Validation-Test Split

## Two-way split

```text
Dataset
│
├── Training → 80%
└── Testing  → 20%
```

Useful for:

* simple projects
* introductory ML
* small experiments

---

## Three-way split

```text
Dataset
│
├── Training   → 70%
├── Validation → 15%
└── Testing    → 15%
```

Useful when:

* tuning hyperparameters
* comparing models
* performing multiple experiments

---

# 📊 Common Split Ratios

There is no universal ratio.

Common choices include:

| Training | Testing |
| -------: | ------: |
|      90% |     10% |
|      80% |     20% |
|      75% |     25% |
|      70% |     30% |

### Common default

```text
80% Training
20% Testing
```

But the appropriate split depends on:

* dataset size
* problem type
* class distribution
* temporal structure
* grouping structure
* computational cost

---

# 🎲 Random State

A train-test split usually involves randomness.

For example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

`random_state` makes the split **reproducible**.

Without it:

```text
Run 1 → different split
Run 2 → different split
Run 3 → different split
```

With it:

```text
Run 1 → same split
Run 2 → same split
Run 3 → same split
```

### Important

`42` is not a special machine learning number.

You can use:

```python
random_state=0
```

or:

```python
random_state=42
```

or:

```python
random_state=123
```

The important thing is **reproducibility**, not the specific number.

---

# 🔀 Shuffling

Shuffling randomizes the order of observations before splitting.

Example:

```text
Before:

A
B
C
D
E
F
G
H
```

After shuffling:

```text
D
A
G
C
H
B
F
E
```

Then the data can be divided into training and test subsets.

### Why shuffle?

It can prevent an accidental split based on the original ordering of data.

However:

> **Do not blindly shuffle time-dependent data.**

Time-series data generally requires chronological splitting.

---

# 🚨 Data Leakage

One of the most important concepts in train-test splitting is **data leakage**.

Data leakage occurs when information from outside the training data improperly influences the model during training.

Example:

```text
Entire Dataset
     │
     ▼
Scale Entire Dataset
     │
     ▼
Train-Test Split
```

This is problematic because the scaler has already seen information from the test set.

### Better approach

```text
Entire Dataset
     │
     ▼
Train-Test Split
     │
     ├──────────────┐
     ▼              ▼
Training          Testing
     │
     ▼
Fit Preprocessor
     │
     ├── transform Training
     │
     └── transform Testing
```

The key rule is:

> **Fit preprocessing transformations only on the training data.**

---

# ⚖️ Stratified Train-Test Split

For classification problems, class proportions can become unbalanced after random splitting.

Suppose:

```text
Original Dataset

Class A → 90%
Class B → 10%
```

A random split might accidentally produce:

```text
Training:
Class A → 95%
Class B → 5%

Testing:
Class A → 70%
Class B → 30%
```

This can create an unrepresentative test set.

### Solution

Use stratification:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

This attempts to preserve the target class distribution across the split.

---

# 📈 Regression Data Splitting

Regression problems have continuous targets.

For example:

```text
House Price
₹50,00,000
₹62,00,000
₹71,00,000
₹85,00,000
```

Traditional classification-style:

```python
stratify=y
```

is generally not directly applicable to continuous regression targets.

Instead, consider:

* random splitting when appropriate
* cross-validation
* domain-aware sampling
* grouping
* temporal splitting
* carefully designed bins only when justified

### Example

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# ⏳ Time-Series Data Splitting

Random splitting can be incorrect for time-dependent datasets.

Suppose we have:

```text
2022 → 2023 → 2024 → 2025 → 2026
```

A random split might put:

```text
2026 data → Training
2023 data → Testing
```

This can introduce future information into the training process.

Instead:

```text
Past
│
├─────────────── Training ───────────────┤
                                      │
                                      ▼
                              ┌─────────────┐
                              │    Test     │
                              └─────────────┘
                                      │
                                      ▼
                                    Future
```

Example:

```python
train = df[df["date"] < "2026-01-01"]
test = df[df["date"] >= "2026-01-01"]
```

The exact cutoff should depend on the problem.

---

# 👥 Group-Based Splitting

Sometimes multiple rows belong to the same entity.

Examples:

* multiple medical records from one patient
* multiple purchases from one customer
* multiple measurements from one machine
* multiple images from one person

If the same entity appears in both training and testing data, the model may indirectly see very similar information.

Example:

```text
Customer A
├── Transaction 1 → Training
├── Transaction 2 → Training
└── Transaction 3 → Test
```

This may create leakage.

Instead, keep groups together.

Conceptually:

```text
Customer A → Training
Customer B → Training
Customer C → Test
Customer D → Test
```

Scikit-learn provides group-aware splitting tools such as:

```python
GroupShuffleSplit
```

and:

```python
GroupKFold
```

---

# 🐼 Train-Test Split with Pandas

A simple manual split can be implemented with Pandas.

```python
import pandas as pd

df = pd.DataFrame({
    "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "score": [55, 60, 62, 65, 70, 72, 75, 80, 85, 90]
})

train_size = int(len(df) * 0.8)

train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print("Training:")
print(train_df)

print("\nTesting:")
print(test_df)
```

This approach is especially useful when preserving order is intentional.

---

# 🤖 Train-Test Split with Scikit-Learn

The recommended general-purpose approach is:

```python
from sklearn.model_selection import train_test_split
```

Basic example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This returns:

```text
X_train → Training features
X_test  → Testing features
y_train → Training targets
y_test  → Testing targets
```

---

# 📐 Understanding `test_size`

`test_size` determines how much data goes into the test set.

### Percentage

```python
test_size=0.2
```

means:

```text
20% → Test
80% → Train
```

### Another example

```python
test_size=0.3
```

means:

```text
30% → Test
70% → Train
```

### Integer

You can also provide the number of test samples:

```python
test_size=100
```

meaning:

```text
100 samples → Test
```

---

# 📐 Understanding `train_size`

You can explicitly specify training size.

```python
train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42
)
```

This means approximately:

```text
80% → Training
20% → Testing
```

Usually, specifying `test_size` is sufficient.

---

# 🎯 Understanding `random_state`

Example:

```python
random_state=42
```

This creates a deterministic random split.

For reproducible experiments:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

For production ML systems, reproducibility should extend beyond the split to:

* preprocessing
* model version
* feature definitions
* dependencies
* random seeds
* dataset version

---

# 🔀 Understanding `shuffle`

Default behavior:

```python
shuffle=True
```

Example:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)
```

For ordered data where randomization is inappropriate:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)
```

When `shuffle=False`, the split preserves the original order.

---

# 🏷️ Using `stratify`

For classification:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

### Without stratification

```text
Original:
Class 0 → 80%
Class 1 → 20%

Training:
Class 0 → 84%
Class 1 → 16%

Testing:
Class 0 → 64%
Class 1 → 36%
```

### With stratification

```text
Original:
Class 0 → 80%
Class 1 → 20%

Training:
Class 0 → ~80%
Class 1 → ~20%

Testing:
Class 0 → ~80%
Class 1 → ~20%
```

The proportions may not be perfectly identical for very small datasets, but stratification helps preserve the distribution.

---

# 🎯 Feature-Target Splitting

Before performing train-test splitting, separate:

```text
Features → X
Target   → y
```

Example:

```python
X = df.drop(columns=["target"])
y = df["target"]
```

Then:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Workflow

```text
DataFrame
   │
   ├───────────────┐
   ▼               ▼
Features X       Target y
   │               │
   └───────┬───────┘
           ▼
      Train-Test Split
           │
      ┌────┴────┐
      ▼         ▼
   Training    Testing
```

---

# 🧪 Complete Classification Example

```python
"""
Train-Test Split - Classification Example
------------------------------------------
Demonstrates:
1. Loading a dataset
2. Separating features and target
3. Performing a stratified train-test split
4. Training a model
5. Evaluating the model
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def main():
    # Load dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    print(f"Total samples: {len(X)}")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on unseen test data
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
```

---

# 📉 Complete Regression Example

```python
"""
Train-Test Split - Regression Example
--------------------------------------
Demonstrates a basic train-test split
for a regression problem.
"""

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def main():
    # Load dataset
    diabetes = load_diabetes()

    X = diabetes.data
    y = diabetes.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")


if __name__ == "__main__":
    main()
```

---

# 🔎 Checking the Split

Never assume the split worked correctly.

Check:

```python
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
```

Example:

```text
X_train: (800, 10)
X_test:  (200, 10)

y_train: (800,)
y_test:  (200,)
```

---

## Check Class Distribution

For classification:

```python
import pandas as pd

print("Original:")
print(pd.Series(y).value_counts(normalize=True))

print("\nTraining:")
print(pd.Series(y_train).value_counts(normalize=True))

print("\nTesting:")
print(pd.Series(y_test).value_counts(normalize=True))
```

This is especially useful when using:

```python
stratify=y
```

---

# 🚨 Common Mistakes

## 1. Training and testing on the same data

❌ Incorrect:

```python
model.fit(X, y)

predictions = model.predict(X)
```

This does not provide a reliable estimate of performance on unseen data.

---

## 2. Scaling before splitting

❌ Risky:

```python
scaler.fit_transform(X)
train_test_split(X, y)
```

The scaler has access to information from the entire dataset.

### Better

```text
Split
 ↓
Fit scaler on training data
 ↓
Transform training data
 ↓
Transform test data
```

---

## 3. Using test data for tuning

The test set should remain untouched while repeatedly experimenting with the model.

Otherwise:

```text
Test Set
   ↓
Tune Model
   ↓
Evaluate
   ↓
Tune Again
```

The test set is effectively becoming part of the development process.

---

## 4. Forgetting stratification

For classification:

```python
train_test_split(X, y, test_size=0.2)
```

may produce an undesirable class distribution, particularly with small or imbalanced datasets.

Consider:

```python
stratify=y
```

---

## 5. Randomly splitting time-series data

❌ Avoid:

```python
train_test_split(time_series_data)
```

when future observations must not influence training.

Prefer chronological splitting or time-series cross-validation.

---

## 6. Splitting related observations independently

If multiple rows belong to the same person, customer, machine, or subject, random splitting can leak information across subsets.

Use group-aware splitting when appropriate.

---

## 7. Splitting after preprocessing

The safest general pattern is:

```text
Raw Data
   ↓
Separate features and target
   ↓
Train-Test Split
   ↓
Fit preprocessing on training data
   ↓
Transform train/test
   ↓
Model training
```

---

# ✅ Best Practices

### 1. Always create a genuine holdout set

Use unseen data for final evaluation.

### 2. Make the split reproducible

Use:

```python
random_state=42
```

or another documented seed.

### 3. Use stratification for classification

```python
stratify=y
```

when appropriate.

### 4. Respect temporal order

Time-series problems require chronological evaluation.

### 5. Respect groups

Use group-aware splitting when observations are related.

### 6. Prevent preprocessing leakage

Fit transformations only on training data.

### 7. Do not repeatedly tune against the test set

Keep the test set for final evaluation.

### 8. Check your split

Inspect:

* shapes
* class distributions
* target distributions
* groups
* dates
* duplicates

---

# 🧩 Train-Test Split in ML Pipelines

A production-style workflow looks like:

```text
Raw Dataset
     │
     ▼
Feature / Target Separation
     │
     ▼
Train-Test Split
     │
     ├──────────────────┐
     ▼                  ▼
Training              Testing
     │                  │
     ▼                  │
Preprocessing           │
     │                  │
     ▼                  │
Model Training          │
     │                  │
     └─────────┬────────┘
               ▼
         Final Evaluation
```

For example, preprocessing can be placed inside a Scikit-Learn pipeline:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
```

The important principle is:

> **The pipeline is fitted using training data only.**

---

# 🔁 Cross-Validation

A single train-test split can be sensitive to the particular samples selected.

Cross-validation provides multiple training/validation evaluations.

Example:

```text
Dataset
│
├── Fold 1 → Validation
├── Fold 2 → Validation
├── Fold 3 → Validation
├── Fold 4 → Validation
└── Fold 5 → Validation
```

In 5-fold cross-validation:

```text
Round 1:
Train Train Train Train Validation

Round 2:
Train Train Train Validation Train

Round 3:
Train Train Validation Train Train

Round 4:
Train Validation Train Train Train

Round 5:
Validation Train Train Train Train
```

The scores can then be aggregated.

Example:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5
)

print("CV Scores:", scores)
print("Mean CV Score:", scores.mean())
```

The final untouched test set can then be used once for final evaluation.

---

# 🚫 When Not to Randomly Split

Random splitting is not appropriate for every problem.

| Problem                   | Preferred Strategy    |
| ------------------------- | --------------------- |
| Standard classification   | Random + stratified   |
| Standard regression       | Random                |
| Time series               | Chronological         |
| Customer-level data       | Group split           |
| Patient-level data        | Group split           |
| Repeated measurements     | Group split           |
| Imbalanced classification | Stratified            |
| Very small dataset        | Cross-validation      |
| Forecasting               | Time-based validation |

---

# 🧭 Practical Decision Guide

Ask these questions before splitting.

### Question 1

**Is this a classification problem?**

If yes:

```python
stratify=y
```

may be appropriate.

---

### Question 2

**Does the order of observations matter?**

If yes:

```text
Do not randomly shuffle.
```

Use temporal splitting.

---

### Question 3

**Do multiple observations belong to the same entity?**

If yes:

```text
Use group-aware splitting.
```

---

### Question 4

**Is the dataset very small?**

Consider:

```text
Cross-validation
```

rather than relying on a single split.

---

### Question 5

**Are you preprocessing the features?**

Split first.

Then fit preprocessing only on training data.

---

# 🛠️ Mini Projects

## Project 1 — Iris Classification

Use the Iris dataset.

Tasks:

* load dataset
* separate X and y
* create an 80/20 split
* use stratification
* train a classifier
* evaluate accuracy
* inspect class distributions

---

## Project 2 — House Price Prediction

Use a regression dataset.

Tasks:

* separate features and target
* create an 80/20 split
* train a regression model
* calculate MAE
* calculate RMSE
* calculate R²

---

## Project 3 — Imbalanced Classification

Create a binary classification dataset with:

```text
Class 0 → 95%
Class 1 → 5%
```

Compare:

```python
stratify=None
```

with:

```python
stratify=y
```

Analyze the resulting class distributions.

---

## Project 4 — Time-Series Split

Create a dataset containing:

```text
date
sales
```

Split the dataset chronologically.

Do not randomly shuffle the observations.

---

## Project 5 — Group-Aware Split

Create data containing:

```text
customer_id
feature_1
feature_2
purchase
```

Ensure that the same customer never appears in both training and testing datasets.

---

# 📝 Exercises

### Beginner

1. What is train-test splitting?
2. Why do we need a test set?
3. What does `test_size=0.2` mean?
4. What is `random_state`?
5. What is the purpose of shuffling?
6. What is a training dataset?
7. What is a testing dataset?

### Intermediate

8. Why is stratification useful?
9. What is data leakage?
10. Why should preprocessing happen after splitting?
11. When should `shuffle=False` be used?
12. What is the difference between validation and testing?
13. Why should the test set not be used for repeated model tuning?

### Advanced

14. Why can random splitting be problematic for time series?
15. Why should patient-level data often use group splitting?
16. How does cross-validation differ from a simple train-test split?
17. How can duplicate observations cause leakage?
18. How would you split data when multiple observations belong to the same customer?
19. Why should preprocessing be fitted only on training data?
20. How would you design a train/validation/test strategy for a small imbalanced dataset?

---

# 📌 Important Concepts

| Concept          | Meaning                                   |
| ---------------- | ----------------------------------------- |
| Training Set     | Data used to train the model              |
| Test Set         | Unseen data used for final evaluation     |
| Validation Set   | Data used during model development        |
| `test_size`      | Proportion or number of test samples      |
| `train_size`     | Proportion or number of training samples  |
| `random_state`   | Controls reproducibility                  |
| `shuffle`        | Controls randomization                    |
| `stratify`       | Preserves class proportions               |
| Data Leakage     | Unintended information flow into training |
| Cross-Validation | Multiple train/validation evaluations     |
| Group Split      | Keeps related observations together       |
| Temporal Split   | Preserves chronological order             |

---

# 📁 Project Structure

```text
07-Data-Preprocessing/
│
├── README.md
│
├── 01-Train-Test-Split/
│   ├── README.md
│   ├── basic_split.py
│   ├── classification_split.py
│   ├── regression_split.py
│   ├── stratified_split.py
│   ├── time_series_split.py
│   └── group_split.py
│
├── 02-Feature-Scaling/
├── 03-Encoding-Categorical-Data/
├── 04-Feature-Transformation/
├── 05-Feature-Selection/
├── 06-Feature-Engineering/
└── ...
```

---

# 🔍 End-to-End Example

```python
"""
End-to-End Train-Test Split Workflow
-------------------------------------
Demonstrates the standard workflow for
a supervised classification problem.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # -------------------------------------------------
    # 1. Load data
    # -------------------------------------------------
    iris = load_iris()

    X = iris.data
    y = iris.target

    # -------------------------------------------------
    # 2. Split data
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Dataset Information")
    print("-" * 40)
    print(f"Total samples : {len(X)}")
    print(f"Training      : {len(X_train)}")
    print(f"Testing       : {len(X_test)}")

    # -------------------------------------------------
    # 3. Build preprocessing + model pipeline
    # -------------------------------------------------
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # -------------------------------------------------
    # 4. Train only on training data
    # -------------------------------------------------
    pipeline.fit(X_train, y_train)

    # -------------------------------------------------
    # 5. Predict unseen test data
    # -------------------------------------------------
    predictions = pipeline.predict(X_test)

    # -------------------------------------------------
    # 6. Evaluate
    # -------------------------------------------------
    accuracy = accuracy_score(y_test, predictions)

    print("\nModel Performance")
    print("-" * 40)
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
```

---

# 🧠 Train-Test Split Mental Model

Think of the dataset as an exam.

```text
Training Data
    ↓
Study Material
    ↓
Model Learns
```

Then:

```text
Test Data
    ↓
Final Exam
    ↓
Model Must Perform Without Seeing Answers
```

If you give the model the test questions during training:

```text
❌ Unfair Evaluation
```

Similarly, if information from the test dataset influences preprocessing or model selection:

```text
❌ Data Leakage
```

The goal is:

```text
Train on Known Data
        ↓
Evaluate on Unknown Data
        ↓
Estimate Real-World Performance
```

---

# 🧪 Train-Test Split Checklist

Before training a model:

* [ ] Separate features and target
* [ ] Understand the dataset
* [ ] Identify the problem type
* [ ] Choose an appropriate split strategy
* [ ] Choose train/test proportions
* [ ] Set a reproducible random state when appropriate
* [ ] Use stratification for suitable classification problems
* [ ] Consider groups
* [ ] Consider time ordering
* [ ] Split before fitting preprocessing
* [ ] Check training/test shapes
* [ ] Check target distributions
* [ ] Check for duplicates and leakage
* [ ] Keep the final test set untouched during tuning
* [ ] Evaluate on unseen data

---

# 🌱 Roadmap

```text
Train-Test Split
       │
       ▼
Feature Scaling
       │
       ▼
Categorical Encoding
       │
       ▼
Feature Transformation
       │
       ▼
Feature Selection
       │
       ▼
Feature Engineering
       │
       ▼
Preprocessing Pipelines
       │
       ▼
Model Training
```

---

# 💡 Key Takeaways

> **1. Train-test splitting estimates how well a model generalizes.**

> **2. Training data is used to learn; test data is used for final evaluation.**

> **3. Never allow test information to influence training.**

> **4. Use `stratify=y` when appropriate for classification.**

> **5. Use chronological splitting for time-dependent problems.**

> **6. Use group-aware splitting when observations belong to the same entities.**

> **7. Fit preprocessing transformations only on training data.**

> **8. Keep the final test set untouched during model development.**

> **9. Use cross-validation when a single split is not reliable enough.**

> **10. The best split strategy depends on the structure of your data—not just a standard 80/20 rule.**

---

# 🚀 Next Step

After understanding train-test splitting, the next important preprocessing concept is:

## 📏 Feature Scaling

Learn how to transform numerical features into comparable scales using techniques such as:

* Standardization
* Min-Max Scaling
* Robust Scaling
* Normalization

```text
🌱 Split
   ↓
📏 Scale
   ↓
🔤 Encode
   ↓
🛠️ Transform
   ↓
🎯 Select Features
   ↓
⚙️ Engineer Features
   ↓
🤖 Train Model
```

---

# 👨‍💻 Author

**Kishor Patil**

Machine Learning • Python • Data Science • AI

GitHub: [Kishor055](https://github.com/Kishor055?utm_source=chatgpt.com)

---

# 🤝 Contributing

Contributions are welcome!

If you find an error or have an improvement:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Commit your changes
5. Open a Pull Request

---

# ⭐ Support

If this repository helps you learn Machine Learning:

* ⭐ Star the repository
* 🍴 Fork the repository
* 🐛 Report issues
* 💡 Suggest improvements
* 🚀 Share it with other learners

---

## 📄 License

This project is intended for educational purposes.

# 03 - Naive Bayes Classification

## 📌 Overview

Naive Bayes is a family of supervised machine learning algorithms based on **Bayes' Theorem**. It is primarily used for classification tasks and works by calculating the probability of a data point belonging to a particular class.

It is called *Naive* because it assumes that all features are conditionally independent of one another given the target class.

Naive Bayes is particularly effective for:
- Email spam detection.
- Sentiment analysis.
- Text classification.
- Document categorization.
- Medical diagnosis.
- News classification.

---

## 🧠 Bayes' Theorem

Bayes' Theorem calculates the probability of an event based on prior knowledge of related conditions.

**Mathematical Formula:**

\[
P(A|B)=\frac{P(B|A)\times P(A)}{P(B)}
\]

Where:

- **P(A|B):** Posterior Probability.
- **P(B|A):** Likelihood.
- **P(A):** Prior Probability.
- **P(B):** Evidence.

### How Naive Bayes Works

1. Calculate the prior probability of each class.
2. Calculate the likelihood of each feature given a class.
3. Apply Bayes' Theorem.
4. Calculate the posterior probability for each class.
5. Assign the observation to the class with the highest posterior probability.

---

## 📂 Project Structure

```text
03-Naive-Bayes/
│
├── README.md
├── naive-bayes.py
├── gaussian-naive-bayes.py
├── multinomial-naive-bayes.py
├── bernoulli-naive-bayes.py
├── from-scratch.py
├── practice.py
├── visualization.py
│
└── datasets/
    └── iris.csv
```

---

## 📚 Types of Naive Bayes Algorithms

### 1. Gaussian Naive Bayes

Used when features follow a continuous numerical distribution, typically assumed to be Gaussian (Normal).

**Applications:**
- Medical diagnosis.
- Iris classification.
- Sensor data classification.

**Scikit-learn:**
`GaussianNB()`

### 2. Multinomial Naive Bayes

Used for discrete data, particularly word counts and frequency-based text classification.

**Applications:**
- Spam detection.
- News categorization.
- Document classification.

**Scikit-learn:**
`MultinomialNB()`

### 3. Bernoulli Naive Bayes

Used for binary-valued features, where each feature represents the presence or absence of something.

**Applications:**
- Text classification.
- Spam filtering.
- Binary feature classification.

**Scikit-learn:**
`BernoulliNB()`

### 4. Complement Naive Bayes

A variation of Multinomial Naive Bayes designed to improve classification performance on imbalanced datasets.

**Scikit-learn:**
`ComplementNB()`

---

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical computations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning implementation |

---

## 🚀 Installation

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 💻 Implementation

### 1. Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

### 2. Load Dataset

```python
df = pd.read_csv("datasets/iris.csv")

print(df.head())
print(df.info())
print(df.describe())
```

### 3. Prepare Features and Target

```python
X = df.drop("species", axis=1)
y = df["species"]
```

### 4. Split Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### 5. Train Naive Bayes Model

```python
model = GaussianNB()

model.fit(X_train, y_train)
```

### 6. Make Predictions

```python
y_pred = model.predict(X_test)

print("Predictions:", y_pred)
```

### 7. Evaluate Model

```python
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

---

## 📊 Model Evaluation

Evaluate the classification model using the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Percentage of correct predictions |
| Precision | Correct positive predictions among predicted positives |
| Recall | Correct positive predictions among actual positives |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Comparison of actual and predicted classes |

---

## 📈 Visualization

Visualize the confusion matrix using Seaborn.

```python
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")

plt.show()
```

---

## ⚖️ Advantages and Disadvantages

### Advantages

- Simple and easy to implement.
- Computationally efficient.
- Works well with high-dimensional datasets.
- Requires relatively little training data.
- Particularly effective for text classification.
- Supports multiclass classification.

### Disadvantages

- Assumes conditional independence between features.
- Correlated features can reduce performance.
- Zero-frequency problems can occur without smoothing.
- Probability estimates may not be well calibrated.
- Performance depends on selecting an appropriate Naive Bayes variant.

---

## 🔍 Important Concepts

| Concept | Description |
|---|---|
| Prior Probability | Probability of a class before observing evidence |
| Likelihood | Probability of observing features given a class |
| Posterior Probability | Updated probability after observing evidence |
| Conditional Independence | Assumption that features are independent given the class |
| Laplace Smoothing | Technique used to handle zero-frequency probabilities |
| Maximum A Posteriori (MAP) | Selecting the class with the highest posterior probability |

---

## 🎯 Learning Objectives

After completing this section, you will be able to:

- Understand Bayes' Theorem and its mathematical foundation.
- Explain the working principle of Naive Bayes.
- Implement Gaussian, Multinomial, and Bernoulli Naive Bayes.
- Build classification models using Scikit-learn.
- Implement Naive Bayes from scratch.
- Understand conditional independence and Laplace smoothing.
- Evaluate classification performance using appropriate metrics.
- Apply Naive Bayes to real-world classification problems.

---

## 📝 Practice Exercises

1. Implement Gaussian Naive Bayes using the Iris dataset.
2. Build a spam email detection system using Multinomial Naive Bayes.
3. Implement Bernoulli Naive Bayes for binary text classification.
4. Implement Naive Bayes from scratch using NumPy.
5. Compare Gaussian, Multinomial, and Bernoulli Naive Bayes.
6. Experiment with Laplace smoothing.
7. Evaluate models using accuracy, precision, recall, and F1-score.

---

## 📚 References

- [Scikit-learn - Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
- [Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [Bernoulli Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)

---

**⭐ Learning Machine Learning one algorithm at a time!**

Part of my Machine Learning learning journey, focusing on understanding algorithms, mathematical concepts, practical implementations, and real-world applications.

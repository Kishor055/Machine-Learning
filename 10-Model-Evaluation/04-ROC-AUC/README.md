
# 04. ROC Curve and AUC (Area Under the Curve)

## 📌 Overview

ROC (Receiver Operating Characteristic) Curve and AUC (Area Under the Curve) are important evaluation techniques used to measure the performance of classification models in Machine Learning.

The ROC curve illustrates the ability of a classification model to distinguish between positive and negative classes at different classification thresholds.

AUC measures the overall ability of a model to distinguish between classes. A higher AUC value indicates better classification performance.

ROC-AUC is commonly used to evaluate:

- Logistic Regression
- Decision Trees
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Naive Bayes
- Gradient Boosting
- Neural Networks

---

## 🎯 Learning Objectives

After completing this topic, you will be able to:

- Understand ROC Curve and AUC.
- Understand True Positive Rate (TPR).
- Understand False Positive Rate (FPR).
- Calculate TPR and FPR.
- Understand classification thresholds.
- Interpret ROC curves.
- Calculate AUC scores.
- Implement ROC-AUC using Scikit-Learn.
- Visualize ROC curves using Matplotlib.
- Compare multiple classification models.
- Understand multiclass ROC-AUC.
- Apply ROC-AUC to real-world classification problems.

---

## 1. What is ROC Curve?

ROC stands for Receiver Operating Characteristic.

It is a graphical representation of a classification model's performance at different classification thresholds.

The ROC curve plots:

- **X-axis:** False Positive Rate (FPR)
- **Y-axis:** True Positive Rate (TPR)

It helps determine how effectively a model separates positive and negative classes.

### ROC Curve Structure

The ROC curve consists of:

1. True Positive Rate (TPR).
2. False Positive Rate (FPR).
3. Different classification thresholds.
4. Random classification baseline.
5. Perfect classification point.

### Important Points

- (0, 0): No positive predictions.
- (0, 1): Perfect classification.
- (1, 1): All observations predicted as positive.
- Diagonal line: Random classification.

---

## 2. What is AUC?

AUC stands for Area Under the Curve.

It represents the area under the ROC curve and provides a single numerical value to evaluate the classification model's ability to distinguish between positive and negative classes.

AUC values range between 0 and 1.

### AUC Interpretation

| AUC Score | Model Performance |
|---|---|
| 1.00 | Perfect Classification |
| 0.90 - 0.99 | Excellent |
| 0.80 - 0.89 | Very Good |
| 0.70 - 0.79 | Good |
| 0.60 - 0.69 | Poor |
| 0.50 - 0.59 | Very Poor / Near Random |
| Below 0.50 | Worse than Random (potentially reversed predictions) |

**Note:** These ranges are common informal guidelines, not universal standards. AUC must be interpreted according to the application and dataset.

---

## 3. True Positive Rate (TPR)

True Positive Rate is also called Recall or Sensitivity.

It measures the proportion of actual positive observations correctly identified by the model.

### Formula

```text
TPR = TP / (TP + FN)
```

Where:

- TP = True Positive
- FN = False Negative

### Example

Suppose a disease detection model identifies:

- True Positive = 80
- False Negative = 20

```text
TPR = 80 / (80 + 20)

TPR = 80 / 100

TPR = 0.80

TPR = 80%
```

Therefore, the model correctly identifies 80% of actual positive cases.

---

## 4. False Positive Rate (FPR)

False Positive Rate measures the proportion of actual negative observations incorrectly classified as positive.

### Formula

```text
FPR = FP / (FP + TN)
```

Where:

- FP = False Positive
- TN = True Negative

### Example

Suppose:

- False Positive = 10
- True Negative = 90

```text
FPR = 10 / (10 + 90)

FPR = 10 / 100

FPR = 0.10

FPR = 10%
```

Therefore, the model incorrectly identifies 10% of actual negative cases as positive.

---

## 5. Classification Threshold

A classification threshold is a probability cutoff used to convert predicted probabilities into class labels.

For binary classification:

```text
If Probability >= Threshold:
    Predict Positive
Else:
    Predict Negative
```

### Example

Suppose a model predicts the following probabilities:

| Observation | Predicted Probability |
|---|---:|
| A | 0.95 |
| B | 0.80 |
| C | 0.60 |
| D | 0.40 |
| E | 0.20 |

Using a threshold of 0.50:

- A = Positive
- B = Positive
- C = Positive
- D = Negative
- E = Negative

Using a threshold of 0.70:

- A = Positive
- B = Positive
- C = Negative
- D = Negative
- E = Negative

Changing the threshold changes the number of True Positives, False Positives, True Negatives, and False Negatives.

### Threshold Impact

| Threshold | TPR | FPR |
|---|---|---|
| Low | Generally Higher | Generally Higher |
| High | Generally Lower | Generally Lower |

The ROC curve is generated by evaluating the model across different thresholds.

---

## 6. Understanding ROC Curve

A ROC curve plots TPR against FPR for different classification thresholds.

### ROC Curve Interpretation

- A curve closer to the top-left corner indicates better classification performance.
- A curve close to the diagonal indicates performance similar to random guessing.
- AUC = 1 indicates perfect separation.
- AUC = 0.5 indicates random-ranking performance.

### Ideal ROC Curve

The ideal classifier reaches:

```text
FPR = 0
TPR = 1
```

This represents perfect classification.

---

## 7. Mathematical Interpretation of AUC

AUC represents the probability that a randomly selected positive observation receives a higher prediction score than a randomly selected negative observation.

Mathematically:

```text
AUC = Integral from 0 to 1 of TPR(FPR) d(FPR)
```

A higher AUC indicates better ranking ability.

---

## 8. Python Implementation Using Scikit-Learn

### Step 1: Import Required Libraries

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    auc,
    classification_report
)
```

### Step 2: Create a Classification Dataset

```python
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)
```

### Step 3: Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Step 4: Train Logistic Regression Model

```python
model = LogisticRegression()

model.fit(X_train, y_train)
```

### Step 5: Predict Probabilities

```python
# Predict probability of positive class
y_prob = model.predict_proba(X_test)[:, 1]
```

**Important:** ROC-AUC requires prediction scores or probabilities, not only predicted class labels.

### Step 6: Calculate ROC Curve

```python
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)
```

### Step 7: Calculate AUC Score

```python
roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("ROC-AUC Score:", roc_auc)
```

### Step 8: Visualize ROC Curve

```python
plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.2f})"
)

# Random classifier baseline
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.show()
```

---

## 9. Compare Multiple Classification Models

ROC-AUC can be used to compare the performance of different classification algorithms.

### Python Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(
        random_state=42
    ),
    "SVM": SVC(
        probability=True,
        random_state=42
    )
}

plt.figure(figsize=(9, 7))

for name, model in models.items():

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )

    score = roc_auc_score(y_test, y_prob)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {score:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.grid(alpha=0.3)

plt.show()
```

### Interpretation

- The model with the highest AUC generally has the best ranking ability.
- Models with curves closer to the top-left corner generally perform better.
- Model selection should also consider Precision, Recall, F1-Score, and application-specific error costs.

---

## 10. Multiclass ROC-AUC

ROC-AUC can also be applied to multiclass classification problems.

In multiclass classification, the model predicts one class from three or more possible classes.

Example:

- Class 0: Cat
- Class 1: Dog
- Class 2: Bird

Common approaches include:

### 10.1 One-vs-Rest (OvR)

In One-vs-Rest, one class is treated as positive and all remaining classes are treated as negative.

For three classes:

- Cat vs Rest
- Dog vs Rest
- Bird vs Rest

The ROC-AUC score is calculated for each class.

### 10.2 One-vs-One (OvO)

In One-vs-One, every possible pair of classes is compared.

For three classes:

- Cat vs Dog
- Cat vs Bird
- Dog vs Bird

### Python Implementation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)

auc_score = roc_auc_score(
    y_test,
    y_prob,
    multi_class="ovr",
    average="macro"
)

print("Multiclass ROC-AUC:", auc_score)
```

---

## 11. ROC-AUC vs Accuracy

| Feature | ROC-AUC | Accuracy |
|---|---|---|
| Meaning | Measures ranking ability | Measures correct predictions |
| Output | Score between 0 and 1 | Score between 0 and 1 |
| Threshold | Evaluates multiple thresholds | Uses a particular threshold |
| Class Imbalance | Often more informative than accuracy | Can be misleading |
| Main Purpose | Class discrimination | Overall correctness |
| Uses | Binary and multiclass classification | Binary and multiclass classification |

**Note:** ROC-AUC is not completely immune to class imbalance. For highly imbalanced datasets, Precision-Recall curves and Average Precision should also be considered.

---

## 12. ROC-AUC vs Precision-Recall Curve

| Feature | ROC Curve | Precision-Recall Curve |
|---|---|---|
| X-axis | False Positive Rate | Recall |
| Y-axis | True Positive Rate | Precision |
| Focus | Overall class discrimination | Positive class performance |
| Imbalanced Data | Can hide poor positive-class precision | Often more informative |
| Main Metric | ROC-AUC | Average Precision |

### When to Use ROC-AUC?

- When evaluating ranking performance across thresholds.
- When both positive and negative classes matter.
- When comparing binary classification models.

### When to Use Precision-Recall?

- When the positive class is rare.
- When false positives are particularly costly.
- When identifying positive cases is the main objective.

---

## 13. Advantages of ROC-AUC

1. Evaluates classification performance across multiple thresholds.
2. Provides a single numerical performance score.
3. Useful for comparing classification algorithms.
4. Measures the model's ability to distinguish between classes.
5. Does not require selecting a fixed classification threshold.
6. Useful for evaluating probability-based classification models.
7. Can be extended to multiclass classification.

---

## 14. Limitations of ROC-AUC

1. Does not directly provide the best classification threshold.
2. Can appear optimistic on highly imbalanced datasets.
3. Does not directly measure prediction calibration.
4. Does not consider the actual cost of false positives and false negatives.
5. A high AUC does not guarantee high Precision at a specific threshold.
6. Can be misleading when evaluated on a small or unrepresentative dataset.

---

## 15. Real-World Applications

| Application | Purpose |
|---|---|
| Healthcare | Disease detection |
| Banking | Credit risk assessment |
| Cybersecurity | Intrusion detection |
| Email Systems | Spam classification |
| Finance | Fraud detection |
| Marketing | Customer response prediction |
| Manufacturing | Defect detection |

---

## 16. Important Scikit-Learn Functions

| Function | Description |
|---|---|
| `roc_curve()` | Calculates FPR, TPR, and thresholds |
| `roc_auc_score()` | Calculates ROC-AUC score |
| `auc()` | Calculates area under a given curve |
| `RocCurveDisplay` | Visualizes ROC curves |
| `predict_proba()` | Returns class probabilities |
| `decision_function()` | Returns decision scores |

---

## 17. Practice Exercises

### Beginner Level

1. Create a binary classification dataset.
2. Train a Logistic Regression model.
3. Calculate ROC-AUC score.
4. Plot the ROC curve.
5. Identify TPR and FPR.

### Intermediate Level

1. Compare Logistic Regression and Random Forest using ROC-AUC.
2. Experiment with different classification thresholds.
3. Plot ROC curves for multiple models.
4. Compare ROC-AUC and Accuracy.
5. Generate a classification report.

### Advanced Level

1. Implement multiclass ROC-AUC.
2. Compare OvR and OvO strategies.
3. Evaluate ROC-AUC on an imbalanced dataset.
4. Compare ROC-AUC with Average Precision.
5. Select an appropriate classification threshold based on application requirements.

---

## 18. Best Practices

- Always use predicted probabilities or decision scores for ROC-AUC.
- Evaluate models using unseen test data.
- Use stratified train-test splitting for classification.
- Compare multiple models using the same test dataset.
- Consider Precision and Recall alongside ROC-AUC.
- Use Precision-Recall curves for highly imbalanced datasets.
- Select thresholds according to application-specific requirements.
- Avoid data leakage during model training and evaluation.

---

## 📚 Conclusion

ROC Curve and AUC are important evaluation techniques used to measure the performance of classification models.

The ROC curve represents the relationship between True Positive Rate and False Positive Rate at different classification thresholds.

AUC summarizes the ROC curve into a single numerical value, helping compare the ranking ability of different classification models.

Therefore, ROC-AUC is a valuable technique for evaluating classification algorithms in applications such as healthcare, fraud detection, cybersecurity, and spam filtering.

---

## 🔗 References

- [Scikit-Learn ROC Curve Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
- [Scikit-Learn ROC-AUC Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Scikit-Learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-Learn ROC Curve Display](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html)
- [Google Machine Learning - ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

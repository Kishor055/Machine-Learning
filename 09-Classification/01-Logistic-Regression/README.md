# Logistic Regression

## 📌 Overview

Logistic Regression is a supervised machine learning algorithm used primarily for **classification problems**. Despite its name, it is a classification algorithm that predicts the probability of an input belonging to a particular class.

It uses the logistic (sigmoid) function to transform a linear combination of input features into a probability between 0 and 1.

This folder covers the mathematical foundations, implementation, visualization, and practical applications of Logistic Regression.

## 🎯 Learning Objectives

By completing this module, you will learn how to:

- Understand Logistic Regression and its applications.
- Differentiate between regression and classification.
- Understand the sigmoid function and decision boundaries.
- Implement Logistic Regression using Scikit-learn.
- Build a Logistic Regression model from scratch.
- Understand binary and multiclass classification.
- Evaluate classification models using appropriate metrics.
- Visualize predictions and classification boundaries.
- Understand regularization and its importance.

## 🧠 What Is Logistic Regression?

Logistic Regression estimates the probability that an observation belongs to a specific class.

For binary classification, the predicted probability is:

\[
P(y=1\mid x)=\frac{1}{1+e^{-z}}
\]

where:

\[
z=w_0+w_1x_1+w_2x_2+\cdots+w_nx_n
\]

- \(P(y=1\mid x)\): Probability of belonging to class 1.
- \(e\): Euler's number.
- \(w_0\): Bias or intercept.
- \(w_1,\ldots,w_n\): Model coefficients.
- \(x_1,\ldots,x_n\): Input features.

## 📈 The Sigmoid Function

The sigmoid function maps any real-valued input to a value between 0 and 1.

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

A predicted probability can then be converted into a class label using a decision threshold.

For the default threshold of 0.5:

- Probability ≥ 0.5 → Class 1
- Probability < 0.5 → Class 0

The threshold can be adjusted depending on the application and the relative costs of false positives and false negatives.

## ⚙️ How Logistic Regression Works

1. Collect and preprocess the dataset.
2. Separate the input features and target labels.
3. Split the data into training and testing sets.
4. Initialize the model parameters.
5. Calculate a linear combination of the input features.
6. Apply the sigmoid function to estimate probabilities.
7. Learn the parameters by minimizing the log-loss function.
8. Generate predictions using a decision threshold.
9. Evaluate the model using classification metrics.

## 🧮 Cost Function: Log Loss

Logistic Regression commonly uses binary cross-entropy, also known as log loss.

\[
J(w)=-\frac{1}{m}\sum_{i=1}^{m}
\left[y_i\log(\hat y_i)+(1-y_i)\log(1-\hat y_i)\right]
\]

where:

- \(m\): Number of training examples.
- \(y_i\): Actual class label.
- \(\hat y_i\): Predicted probability.

The model learns parameters that minimize this loss, typically using an optimization algorithm such as gradient descent or a numerical solver.

## 📂 Suggested Folder Structure

```text
09-Classification/
└── 01-Logistic-Regression/
    ├── README.md
    ├── logistic-regression.py
    ├── from-scratch.py
    ├── visualization.py
    ├── practice.md
    └── requirements.txt
```

## 💻 Implementation Using Scikit-learn

The following example trains a Logistic Regression model on a binary classification dataset.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create a preprocessing and classification pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

# Train model
model.fit(X_train, y_train)

# Predict classes and probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
```

**Note:** The breast cancer dataset uses `0` for malignant and `1` for benign. The example predicts the probability of the benign class.

## 🛠️ Implementation From Scratch

A Logistic Regression model can be implemented using NumPy and gradient descent.

The main steps are:

1. Initialize the weights and bias.
2. Calculate the linear output.
3. Apply the sigmoid function.
4. Calculate binary cross-entropy loss.
5. Compute the gradients.
6. Update the parameters iteratively.
7. Predict class probabilities and labels.

The weight update rule is:

\[
w:=w-\alpha\frac{\partial J}{\partial w}
\]

where \(\alpha\) is the learning rate.

For a binary Logistic Regression model, the gradients are:

\[
\frac{\partial J}{\partial w}=\frac{1}{m}X^T(\hat y-y)
\]

\[
\frac{\partial J}{\partial b}=\frac{1}{m}\sum_{i=1}^{m}(\hat y_i-y_i)
\]

These equations provide the foundation for implementing the algorithm without using a machine learning library.

## 📊 Model Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Fraction of all predictions that are correct |
| Precision | Fraction of predicted positives that are actual positives |
| Recall | Fraction of actual positives correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Summarizes true and false predictions |
| ROC-AUC | Measures ranking performance across classification thresholds |
| Log Loss | Measures the quality of predicted probabilities |

Accuracy alone may be misleading when the classes are imbalanced. Precision, recall, F1-score, and ROC-AUC can provide additional insight.

## 📉 Advantages

- Simple and computationally efficient.
- Provides interpretable model coefficients.
- Produces probability estimates.
- Works well for many linearly separable classification problems.
- Supports regularization to help control overfitting.
- Serves as a strong baseline for classification tasks.

## ⚠️ Limitations

- A linear decision boundary may not capture complex relationships.
- Multicollinearity can make coefficients unstable.
- Outliers and influential observations can affect the fitted model.
- Feature scaling can be important for optimization and regularization.
- Performance may be limited when important nonlinear relationships are present.
- Probability estimates may require calibration for some applications.

## 🌍 Real-World Applications

- **Healthcare:** Disease risk classification.
- **Finance:** Credit default prediction.
- **Marketing:** Customer conversion prediction.
- **Cybersecurity:** Spam and suspicious activity detection.
- **E-commerce:** Purchase likelihood prediction.
- **Education:** Student dropout risk classification.

## 🔍 Important Concepts to Remember

- Logistic Regression is a classification algorithm despite its name.
- The sigmoid function converts linear outputs into probabilities.
- The decision threshold determines the predicted class.
- Log loss is the standard objective for binary Logistic Regression.
- Regularization helps control model complexity.
- Multiclass classification can use strategies such as one-vs-rest or multinomial Logistic Regression.
- Evaluation metrics should reflect the problem's requirements and class distribution.

## 📝 Practice Exercises

1. Implement Logistic Regression using Scikit-learn.
2. Implement binary Logistic Regression from scratch using NumPy.
3. Plot the sigmoid function for values of \(z\) from -10 to 10.
4. Compare model performance with and without feature scaling.
5. Experiment with different decision thresholds.
6. Calculate accuracy, precision, recall, and F1-score.
7. Visualize a confusion matrix.
8. Compare L1 and L2 regularization.
9. Train a model on an imbalanced dataset and evaluate its performance.
10. Compare Logistic Regression with K-Nearest Neighbors and Decision Trees.

## 📚 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 🚀 How to Run

Clone your repository and navigate to this directory.

```bash
cd 09-Classification/01-Logistic-Regression
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the implementation:

```bash
python logistic-regression.py
```

Run the from-scratch implementation:

```bash
python from-scratch.py
```

Run the visualization script:

```bash
python visualization.py
```

## 📖 References

- [Scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Scikit-learn: Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)

---

**Next Topic:** Classification using K-Nearest Neighbors (KNN).
```

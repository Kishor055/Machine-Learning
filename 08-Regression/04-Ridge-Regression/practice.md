# Ridge Regression — Practice Exercises

A structured practice guide for learning Ridge Regression, L2 regularization, coefficient shrinkage, model evaluation, and hyperparameter tuning using Python and Scikit-learn.

## 📌 Table of Contents

- [Learning Objectives](#-learning-objectives)
- [Prerequisites](#-prerequisites)
- [Level 1: Beginner](#-level-1-beginner)
- [Level 2: Intermediate](#-level-2-intermediate)
- [Level 3: Advanced](#-level-3-advanced)
- [Coding Challenges](#-coding-challenges)
- [Mini Projects](#-mini-projects)
- [Interview Questions](#-interview-questions)
- [Solutions and Hints](#-solutions-and-hints)
- [Completion Checklist](#-completion-checklist)

---

## 🎯 Learning Objectives

After completing these exercises, you should be able to:

- Understand Ridge Regression and L2 regularization.
- Explain the role of the `alpha` hyperparameter.
- Train Ridge Regression models using Scikit-learn.
- Apply feature scaling correctly.
- Compare Ridge Regression with ordinary Linear Regression.
- Evaluate models using MAE, MSE, RMSE, and \(R^2\).
- Tune hyperparameters using cross-validation.
- Analyze coefficient shrinkage and multicollinearity.
- Identify overfitting and underfitting.

## 🧰 Prerequisites

Install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Import the libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
```

---

## 🟢 Level 1: Beginner

### Exercise 1: Understand Ridge Regression

Answer the following questions in your own words.

1. What is Ridge Regression?
2. Why is regularization used in machine learning?
3. What is L2 regularization?
4. What is the purpose of the `alpha` parameter?
5. What happens when `alpha = 0`?
6. Does Ridge Regression eliminate features?
7. Why is feature scaling important?
8. How does Ridge Regression differ from ordinary Linear Regression?

**Expected outcome:** Explain the purpose of Ridge Regression without referring to code.

### Exercise 2: Train Your First Ridge Model

Create a synthetic regression dataset and train a Ridge Regression model.

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X, y = make_regression(
    n_samples=200,
    n_features=5,
    noise=15,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("First five predictions:", predictions[:5])
```

**Tasks:**

- Print the shapes of the training and testing sets.
- Display the intercept and coefficients.
- Predict values for at least five test samples.
- Explain what the coefficients represent.

### Exercise 3: Evaluate Model Performance

Use the model from Exercise 2 to calculate evaluation metrics.

```python
mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)
```

**Tasks:**

1. Explain the difference between MAE and MSE.
2. Why is RMSE expressed in the same units as the target?
3. What does a high \(R^2\) indicate?
4. Can \(R^2\) be negative on test data?
5. Which metric would you use if large prediction errors are particularly costly?

### Exercise 4: Experiment with `alpha`

Train separate Ridge Regression models using different regularization strengths.

```python
alphas = [0, 0.01, 0.1, 1, 10, 100]

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    print(
        f"Alpha: {alpha:>6} | "
        f"RMSE: {rmse:.4f}"
    )
```

**Tasks:**

- Compare the RMSE values.
- Observe how coefficients change as `alpha` increases.
- Identify the value that performs best on the test set for this experiment.
- Explain why the largest `alpha` is not necessarily the best.

*Note:* Use the test set only for this introductory exercise. For model selection in a real workflow, use cross-validation on the training set and evaluate the selected model on the test set once.

---

## 🟡 Level 2: Intermediate

### Exercise 5: Compare Linear Regression and Ridge Regression

Train both models on the same data split.

```python
linear_model = LinearRegression()
ridge_model = Ridge(alpha=10.0)

linear_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)
ridge_predictions = ridge_model.predict(X_test)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)

ridge_rmse = np.sqrt(
    mean_squared_error(y_test, ridge_predictions)
)

print("Linear Regression RMSE:", linear_rmse)
print("Ridge Regression RMSE:", ridge_rmse)

print("\nLinear Coefficients:")
print(linear_model.coef_)

print("\nRidge Coefficients:")
print(ridge_model.coef_)
```

**Tasks:**

- Compare the coefficients of both models.
- Compare their test RMSE.
- Explain why Ridge coefficients are often smaller.
- Determine whether Ridge Regression improves performance on this dataset.

### Exercise 6: Apply Feature Scaling

Compare Ridge Regression with and without standardization.

```python
unscaled_model = Ridge(alpha=10.0)
unscaled_model.fit(X_train, y_train)

scaled_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=10.0)),
])

scaled_pipeline.fit(X_train, y_train)

unscaled_predictions = unscaled_model.predict(X_test)
scaled_predictions = scaled_pipeline.predict(X_test)

print(
    "Unscaled RMSE:",
    np.sqrt(
        mean_squared_error(y_test, unscaled_predictions)
    ),
)

print(
    "Scaled RMSE:",
    np.sqrt(
        mean_squared_error(y_test, scaled_predictions)
    ),
)
```

**Tasks:**

1. Compare the two RMSE values.
2. Explain why feature scales matter for Ridge Regression.
3. Modify one feature so its values are much larger than the others.
4. Repeat the experiment and compare results.

**Challenge:** Use a pipeline whenever scaling is part of model training to prevent data leakage.

### Exercise 7: Visualize Coefficient Shrinkage

Study how coefficients change as regularization increases.

```python
alphas = np.logspace(-3, 3, 30)

coefficient_history = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    coefficient_history.append(model.coef_)

coefficient_history = np.asarray(coefficient_history)

plt.figure(figsize=(10, 6))

for feature_index in range(X_train.shape[1]):
    plt.plot(
        alphas,
        coefficient_history[:, feature_index],
        label=f"Feature {feature_index + 1}",
    )

plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Coefficient Value")
plt.title("Ridge Regression: Coefficient Shrinkage")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Tasks:**

- Describe the overall trend as `alpha` increases.
- Identify which coefficients change the most.
- Explain why Ridge coefficients generally approach zero rather than becoming exactly zero.
- Investigate the effect of feature scaling on coefficient comparisons.

### Exercise 8: Tune `alpha` with Cross-Validation

Use `GridSearchCV` to choose a suitable regularization strength.

```python
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge()),
])

param_grid = {
    "ridge__alpha": np.logspace(-3, 3, 13)
}

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
)

search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)
print("Best CV RMSE:", np.sqrt(-search.best_score_))

best_model = search.best_estimator_

test_predictions = best_model.predict(X_test)

print(
    "Test RMSE:",
    np.sqrt(
        mean_squared_error(y_test, test_predictions)
    ),
)
```

**Tasks:**

- Explain why the parameter is named `ridge__alpha`.
- Find the best `alpha`.
- Compare the cross-validation RMSE with the test RMSE.
- Try a wider range of regularization strengths.

---

## 🔴 Level 3: Advanced

### Exercise 9: Investigate Multicollinearity

Create a dataset with strongly correlated features.

```python
rng = np.random.default_rng(42)

n_samples = 300

x1 = rng.normal(size=n_samples)
x2 = x1 * 0.95 + rng.normal(
    scale=0.1,
    size=n_samples,
)

x3 = rng.normal(size=n_samples)

X_corr = np.column_stack([x1, x2, x3])

y_corr = (
    5 * x1
    + 5 * x2
    + 2 * x3
    + rng.normal(scale=0.5, size=n_samples)
)

print(np.corrcoef(X_corr, rowvar=False))
```

**Tasks:**

1. Calculate the feature correlation matrix.
2. Train Linear Regression and Ridge Regression.
3. Compare the coefficients across multiple training samples.
4. Test several `alpha` values.
5. Explain how regularization can stabilize coefficient estimates.

### Exercise 10: Compare Ridge, Lasso, and Elastic Net

```python
from sklearn.linear_model import Lasso, ElasticNet

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0, max_iter=10000),
    "Elastic Net": ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=10000,
    ),
}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    print(f"{name:20} RMSE: {rmse:.4f}")
```

**Tasks:**

- Compare the prediction errors.
- Examine the coefficients of the fitted models.
- Count how many coefficients are approximately zero for Lasso and Elastic Net.
- Explain why Ridge is often useful when many correlated features contribute to the target.

### Exercise 11: Investigate Underfitting and Overfitting

Train Ridge Regression with a broad range of `alpha` values.

**Tasks:**

1. Plot training RMSE against `alpha`.
2. Plot validation RMSE against `alpha`.
3. Identify where the model underfits.
4. Explain why training error alone is not enough to select `alpha`.
5. Repeat using cross-validation.

Use a logarithmic scale for `alpha` so small and large values are easy to compare.

### Exercise 12: Implement Ridge Regression from Scratch

Implement Ridge Regression using gradient descent.

Your implementation should:

- Initialize weights and an intercept.
- Calculate predictions.
- Calculate the gradient of mean squared error.
- Add the L2 penalty to the weight gradient.
- Update the weights and intercept.
- Predict values for new samples.

**Requirements:**

- Use NumPy.
- Do not regularize the intercept.
- Test your implementation on synthetic data.
- Compare its results with Scikit-learn after accounting for the objective's scaling convention.
- Experiment with the learning rate and number of iterations.

---

## 💻 Coding Challenges

### Challenge 1: House Price Prediction

Create a regression model using features such as:

- Area
- Number of bedrooms
- Number of bathrooms
- House age

**Requirements:**

- Perform a train/test split.
- Handle missing values if present.
- Scale features.
- Train Ridge Regression.
- Tune `alpha` with cross-validation.
- Report MAE, RMSE, and \(R^2\).

### Challenge 2: Regularization Experiment

Generate a dataset containing many features, including irrelevant features.

Compare Linear Regression and Ridge Regression using cross-validation.

**Goal:** Investigate whether regularization improves generalization.

### Challenge 3: Coefficient Stability

Generate several training samples from a dataset with correlated features.

Fit Linear Regression and Ridge Regression to each sample.

**Goal:** Compare the variation in the fitted coefficients across samples.

### Challenge 4: Build a Reusable Model Pipeline

Create a reusable function that accepts:

- `X` — input features
- `y` — target values
- A list of candidate `alpha` values

The function should:

1. Split the dataset.
2. Create a scaling and Ridge Regression pipeline.
3. Tune `alpha` with cross-validation.
4. Evaluate the selected model on the test set.
5. Return the fitted model and evaluation metrics.

---

## 🚀 Mini Projects

### Project 1: House Price Prediction

**Objective:** Predict house prices using property characteristics.

**Workflow:**

1. Load a real or synthetic dataset.
2. Explore the features and target.
3. Handle missing values.
4. Split the data.
5. Scale numerical features.
6. Train Linear Regression and Ridge Regression.
7. Tune `alpha`.
8. Compare model performance.
9. Analyze residuals and coefficients.

**Deliverables:**

- Python implementation
- Evaluation metrics
- Actual-versus-predicted plot
- Explanation of the selected model

### Project 2: Sales Forecasting

**Objective:** Predict product sales using marketing expenditure, pricing, and other relevant numerical features.

**Requirements:**

- Inspect feature correlations.
- Apply preprocessing using a pipeline.
- Compare regularized and unregularized regression.
- Tune the regularization strength.
- Explain the main limitations of the model.

### Project 3: Student Performance Prediction

**Objective:** Predict a continuous student outcome using relevant numerical variables.

**Requirements:**

- Inspect the dataset for missing values.
- Avoid using information unavailable at prediction time.
- Train Ridge Regression.
- Compare validation performance across several `alpha` values.
- Explain why correlation does not establish causation.

### Project 4: Regression Model Benchmark

**Objective:** Compare multiple regression algorithms on one dataset.

Compare:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Polynomial Regression with Ridge regularization

**Deliverables:**

- Consistent train/test split
- Appropriate preprocessing pipelines
- Cross-validation results
- Final test metrics
- A brief explanation of the trade-offs

---

## 🎤 Interview Questions

### Beginner

1. What is Ridge Regression?
2. What is regularization?
3. What is L2 regularization?
4. What is the meaning of `alpha` in Scikit-learn?
5. Does Ridge Regression eliminate features?
6. Why is standardization useful?

### Intermediate

7. How does Ridge Regression address multicollinearity?
8. What happens when `alpha` increases?
9. How does Ridge Regression differ from Lasso?
10. Why should feature scaling be fitted on training data only?
11. How do you select an appropriate `alpha`?
12. Why can training performance improve while test performance worsens?

### Advanced

13. Derive the Ridge Regression objective function.
14. Explain why Ridge Regression shrinks coefficients.
15. Why can Ridge Regression improve coefficient stability?
16. How can you compare coefficient stability across training samples?
17. What is the difference between regularization and feature selection?
18. Why should a final test set not be repeatedly used for hyperparameter tuning?

---

## 💡 Solutions and Hints

Use these hints to guide your work before consulting external references.

| Exercise | Hint |
|---|---|
| 1 | Focus on the L2 penalty and coefficient shrinkage. |
| 2 | Use `Ridge.fit()` and `Ridge.predict()`. |
| 3 | Calculate MAE, MSE, RMSE, and \(R^2\). |
| 4 | Compare coefficients and validation performance across `alpha` values. |
| 5 | Fit both models on identical data splits. |
| 6 | Place `StandardScaler` and `Ridge` in a pipeline. |
| 7 | Plot coefficients against a logarithmic `alpha` axis. |
| 8 | Use `GridSearchCV` with `ridge__alpha`. |
| 9 | Create correlated features and inspect coefficient changes. |
| 10 | Compare L1 and L2 regularization behavior. |
| 11 | Compare training and validation errors. |
| 12 | Add the L2 penalty to the gradient of the mean squared error objective. |

### Self-Assessment

Score each completed exercise:

- Beginner exercises: 1 point each
- Intermediate exercises: 2 points each
- Advanced exercises: 3 points each

Use the score to track progress rather than as a formal measure of mastery.

---

## ✅ Completion Checklist

- [ ] I can explain Ridge Regression and L2 regularization.
- [ ] I can train a Ridge model using Scikit-learn.
- [ ] I can evaluate regression predictions.
- [ ] I understand the effect of `alpha`.
- [ ] I can use feature scaling correctly.
- [ ] I can compare Ridge with ordinary Linear Regression.
- [ ] I can tune `alpha` using cross-validation.
- [ ] I understand multicollinearity and coefficient shrinkage.
- [ ] I can compare Ridge, Lasso, and Elastic Net.
- [ ] I can implement the Ridge objective using gradient descent.
- [ ] I have completed at least one mini-project.

---

## 📚 References

- [Scikit-learn: Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [Scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [Scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

**Next step:** Complete the beginner exercises first, then move to cross-validation, coefficient analysis, and the mini-projects.


# 01 - Regression Metrics

## 📌 Overview

Regression metrics are evaluation techniques used to measure how accurately a machine learning regression model predicts continuous numerical values. They compare actual target values with predicted values to help assess model performance and compare different regression algorithms.

Common regression algorithms include:
- Linear Regression
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression (SVR)

## 🎯 Learning Objectives

By completing this section, you will learn:

- What regression metrics are and why they are important.
- How to calculate Mean Absolute Error (MAE).
- How to calculate Mean Squared Error (MSE).
- How to calculate Root Mean Squared Error (RMSE).
- How to interpret the R² Score.
- How to evaluate regression models using Scikit-learn.
- How to compare regression models using evaluation metrics.

## 📚 Table of Contents

1. [Introduction to Regression Metrics](#introduction-to-regression-metrics)
2. [Mean Absolute Error (MAE)](#1-mean-absolute-error-mae)
3. [Mean Squared Error (MSE)](#2-mean-squared-error-mse)
4. [Root Mean Squared Error (RMSE)](#3-root-mean-squared-error-rmse)
5. [R² Score](#4-r-squared-score-r²)
6. [Other Regression Metrics](#other-regression-metrics)
7. [Practical Implementation](#practical-implementation)
8. [Comparing Regression Models](#comparing-regression-models)
9. [Choosing the Right Metric](#choosing-the-right-metric)
10. [Best Practices](#best-practices)
11. [Key Takeaways](#key-takeaways)
12. [References](#references)

## Introduction to Regression Metrics

A regression model predicts a numerical value, such as house prices, temperature, salary, or sales.

To evaluate a regression model, we compare its predictions with the actual values.

**Residual (Prediction Error):**

Residual = Actual Value − Predicted Value

For example:

| Actual Value | Predicted Value | Residual |
|---:|---:|---:|
| 100 | 90 | 10 |
| 200 | 210 | -10 |
| 300 | 280 | 20 |
| 400 | 410 | -10 |

Different evaluation metrics summarize these errors in different ways.

## 1. Mean Absolute Error (MAE)

### Definition

Mean Absolute Error measures the average absolute difference between actual and predicted values.

### Formula

MAE = (1/n) × Σ |yᵢ − ŷᵢ|

Where:

- `n` = Number of observations
- `yᵢ` = Actual value
- `ŷᵢ` = Predicted value

### Characteristics

- Lower MAE indicates smaller average prediction errors.
- MAE is always non-negative.
- MAE uses the same units as the target variable.
- MAE is less sensitive to large errors than squared-error metrics.

### Python Implementation

```python
from sklearn.metrics import mean_absolute_error

y_true = [100, 200, 300, 400]
y_pred = [90, 210, 280, 410]

mae = mean_absolute_error(y_true, y_pred)

print("Mean Absolute Error:", mae)
```

Output:

```text
Mean Absolute Error: 12.5
```

### Interpretation

The model's predictions differ from the actual values by an average absolute amount of 12.5 units.

## 2. Mean Squared Error (MSE)

### Definition

Mean Squared Error calculates the average of the squared differences between actual and predicted values.

### Formula

MSE = (1/n) × Σ (yᵢ − ŷᵢ)²

### Characteristics

- Lower MSE indicates better predictions.
- MSE penalizes large errors more heavily.
- MSE is always non-negative.
- Its units are the square of the target variable's units.
- It is sensitive to outliers.

### Python Implementation

```python
from sklearn.metrics import mean_squared_error

y_true = [100, 200, 300, 400]
y_pred = [90, 210, 280, 410]

mse = mean_squared_error(y_true, y_pred)

print("Mean Squared Error:", mse)
```

Output:

```text
Mean Squared Error: 175
```

### Interpretation

The average squared prediction error is 175 squared units.

## 3. Root Mean Squared Error (RMSE)

### Definition

Root Mean Squared Error is the square root of Mean Squared Error. It measures prediction error in the same units as the target variable.

### Formula

RMSE = √MSE

### Characteristics

- Lower RMSE indicates better predictions.
- Large errors receive greater influence.
- RMSE uses the same units as the target variable.
- RMSE is useful when large prediction errors are particularly costly.

### Python Implementation

```python
from sklearn.metrics import root_mean_squared_error

y_true = [100, 200, 300, 400]
y_pred = [90, 210, 280, 410]

rmse = root_mean_squared_error(y_true, y_pred)

print("Root Mean Squared Error:", rmse)
```

Output:

```text
Root Mean Squared Error: 13.2288
```

*Note: `root_mean_squared_error` is available in Scikit-learn 1.4 and later.*

For older versions, calculate RMSE using:

```python
import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)
```

## 4. R-Squared Score (R²)

### Definition

R², also called the coefficient of determination, measures how well a regression model explains the variation in the target variable relative to predicting the mean.

### Formula

R² = 1 − SS_res / SS_tot

Where:

- `SS_res` = Sum of squared prediction errors.
- `SS_tot` = Total sum of squared differences between actual values and their mean.

### Interpretation

| R² Value | Interpretation |
|---:|---|
| 1.0 | Perfect predictions |
| 0.8 | Strong explanatory performance in many contexts |
| 0.5 | Explains some variation |
| 0.0 | Equivalent to the mean-prediction baseline on the evaluated data |
| Below 0 | Worse than the mean-prediction baseline |

These interpretations are illustrative rather than universal thresholds.

### Python Implementation

```python
from sklearn.metrics import r2_score

y_true = [100, 200, 300, 400]
y_pred = [90, 210, 280, 410]

r2 = r2_score(y_true, y_pred)

print("R2 Score:", r2)
```

Output:

```text
R2 Score: 0.982
```

### Interpretation

The model explains approximately 98.2% of the variation in the actual target values on this evaluation dataset.

**Important:** A high R² score does not automatically mean that a model generalizes well to unseen data. Always evaluate performance on an appropriate test set.

## Other Regression Metrics

Beyond MAE, MSE, RMSE, and R², several other metrics are useful in specific scenarios.

| Metric | Purpose | Important consideration |
|---|---|---|
| Median Absolute Error | Measures the median absolute prediction error. | More resistant to outliers. |
| Mean Absolute Percentage Error (MAPE) | Expresses errors as percentages. | Can behave poorly when actual values are zero or close to zero. |
| Maximum Error | Measures the largest absolute prediction error. | Highlights worst-case predictions. |
| Mean Squared Logarithmic Error (MSLE) | Compares logarithmically transformed actual and predicted values. | Requires non-negative actual and predicted values. |
| Explained Variance Score | Measures how much variation in the target is explained by predictions. | Does not penalize systematic prediction offsets in the same way as R². |

### Example

```python
from sklearn.metrics import (
    median_absolute_error,
    mean_absolute_percentage_error,
    max_error,
    explained_variance_score
)

print("Median Absolute Error:",
      median_absolute_error(y_true, y_pred))

print("MAPE:",
      mean_absolute_percentage_error(y_true, y_pred))

print("Maximum Error:",
      max_error(y_true, y_pred))

print("Explained Variance:",
      explained_variance_score(y_true, y_pred))
```

## Practical Implementation

Let's evaluate a Linear Regression model using a sample dataset.

### Step 1: Import Libraries

```python
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)
```

### Step 2: Create a Dataset

```python
X, y = make_regression(
    n_samples=500,
    n_features=3,
    noise=20,
    random_state=42
)
```

### Step 3: Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Step 4: Train the Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Step 5: Make Predictions

```python
y_pred = model.predict(X_test)
```

### Step 6: Evaluate the Model

```python
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
```

The exact results are produced when the code runs. Because the dataset includes noise, predictions will generally not be perfect.

## Comparing Regression Models

Different algorithms can produce different prediction errors on the same dataset.

The following example compares Linear Regression, Decision Tree Regression, and Random Forest Regression.

```python
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE": mean_squared_error(
            y_test, predictions
        ) ** 0.5,
        "R2 Score": r2_score(y_test, predictions)
    })

results_df = pd.DataFrame(results)
print(results_df.round(4))
```

### How to Compare Results

- Prefer lower MAE when average absolute errors matter.
- Prefer lower RMSE when large errors should receive more weight.
- Use R² alongside error metrics to assess explanatory performance.
- Compare models using the same test dataset.
- Use cross-validation when a more reliable estimate of model performance is required.

The best model depends on the evaluation results and the requirements of the application.

## Choosing the Right Metric

| Requirement | Recommended Metric |
|---|---|
| Understand average prediction error | MAE |
| Penalize large errors more heavily | MSE or RMSE |
| Interpret error in original units | MAE or RMSE |
| Evaluate performance relative to a mean baseline | R² |
| Reduce sensitivity to extreme errors | Median Absolute Error |
| Evaluate percentage errors | MAPE, when actual values are safely away from zero |
| Monitor the worst prediction | Maximum Error |

## Best Practices

1. Evaluate models using data that was not used for training.
2. Avoid data leakage during preprocessing and feature engineering.
3. Compare multiple relevant metrics instead of relying on one score.
4. Investigate outliers and unusually large residuals.
5. Consider the business cost of underprediction and overprediction.
6. Use cross-validation for model selection where appropriate.
7. Compare models on the same data and target scale.
8. Do not interpret R² as a universal measure of model quality.
9. Check metric limitations, especially for percentage-based and logarithmic metrics.
10. Keep a simple baseline model for comparison.

## Key Takeaways

- Regression metrics measure the accuracy of continuous-value predictions.
- MAE summarizes average absolute errors.
- MSE penalizes large errors by squaring them.
- RMSE expresses squared-error-based performance in the target's original units.
- R² evaluates predictive performance relative to a mean baseline.
- Median Absolute Error, MAPE, and Maximum Error can provide additional insights.
- Metric selection should reflect the dataset and the real-world objective.
- Test-set evaluation and cross-validation help assess generalization.

## References

- [Scikit-learn: Regression Metrics and Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn: Metrics API Reference](https://scikit-learn.org/stable/api/sklearn.metrics.html)
- [Scikit-learn: Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

---

**Next Topic:** `02-Classification-Metrics`

**Part of:** Machine Learning — Model Evaluation

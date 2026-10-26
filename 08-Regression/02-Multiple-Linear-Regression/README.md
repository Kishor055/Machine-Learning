# Multiple Linear Regression

## 📌 Overview

**Multiple Linear Regression** is a supervised machine learning algorithm used to predict a continuous target variable using two or more independent variables (features).

Unlike Simple Linear Regression, which uses only one input feature, Multiple Linear Regression considers several features simultaneously to model their combined relationship with the target.

**Example:** Predicting house prices using area, number of bedrooms, house age, and location-related numerical features.

## 🎯 Learning Objectives

By completing this section, you will learn:

- Understand the mathematics behind Multiple Linear Regression.
- Differentiate Simple and Multiple Linear Regression.
- Train regression models using Scikit-learn.
- Interpret coefficients and the intercept.
- Evaluate models using regression metrics.
- Understand multicollinearity and feature selection.
- Analyze residuals and model assumptions.
- Build an end-to-end regression workflow.

## 🧠 Mathematical Foundation

The Multiple Linear Regression equation is:

\[
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_px_p
\]

Where:

- \(\hat{y}\) = predicted target value
- \(b_0\) = intercept
- \(b_1,b_2,\ldots,b_p\) = regression coefficients
- \(x_1,x_2,\ldots,x_p\) = input features
- \(p\) = number of input features

Each coefficient represents the estimated change in the prediction for a one-unit increase in that feature, **holding the other included features constant**.

### Example

Suppose a house price model is:

\[
\begin{aligned}
\text{Price} ={}& 50000 + 2000(\text{Area})\\
&+15000(\text{Bedrooms})\\
&-1000(\text{Age})
\end{aligned}
\]

For a house with an area of 1000 square feet, 3 bedrooms, and an age of 5 years:

\[
\begin{aligned}
\text{Price}={}&50000+2000(1000)\\
&+15000(3)-1000(5)
\end{aligned}
\]

\[
\text{Price}=2,090,000
\]

This is an illustrative equation, not a model trained on real housing data. The result uses the assumed target and feature units.

## 📂 Topics Covered

| No. | Topic | Description |
|---:|---|---|
| 1 | Introduction | Fundamentals of Multiple Linear Regression |
| 2 | Mathematical Foundation | Coefficients, intercept, and least squares |
| 3 | Dataset Preparation | Features, target, and train-test splitting |
| 4 | Model Training | Fit a regression model using Scikit-learn |
| 5 | Model Evaluation | MAE, MSE, RMSE, and R² |
| 6 | Coefficient Interpretation | Understand individual feature coefficients |
| 7 | Multicollinearity | Investigate correlated input features |
| 8 | Residual Analysis | Examine errors and regression assumptions |
| 9 | Feature Selection | Identify useful predictors |
| 10 | Practical Project | Build a house price prediction model |

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## ⚙️ Installation

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## 💻 Basic Implementation

The following example trains a Multiple Linear Regression model to predict house prices from area, bedrooms, and house age.

```python
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Features: area, bedrooms, house age
X = np.array([
    [1000, 2, 10],
    [1200, 2, 8],
    [1500, 3, 5],
    [1800, 3, 4],
    [2000, 4, 3],
    [2200, 4, 2],
    [2500, 4, 1],
    [2800, 5, 2],
    [3000, 5, 1],
    [1600, 3, 6],
    [2100, 4, 4],
    [2400, 4, 3],
])

# Target: house price
y = np.array([
    200000, 230000, 300000, 340000,
    390000, 420000, 470000, 510000,
    560000, 310000, 400000, 450000,
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Display learned parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Evaluate performance
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# Predict for a new house
new_house = np.array([[1900, 3, 4]])
predicted_price = model.predict(new_house)

print("Predicted house price:", predicted_price[0])
```

**Note:** This is a small synthetic dataset for demonstrating the workflow. Its test metrics are unstable and should not be interpreted as evidence of real-world predictive accuracy.

## 📊 Model Evaluation Metrics

### 1. Mean Absolute Error (MAE)

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

Measures the average absolute prediction error.

### 2. Mean Squared Error (MSE)

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Penalizes larger errors more strongly.

### 3. Root Mean Squared Error (RMSE)

\[
RMSE=\sqrt{MSE}
\]

Measures prediction error in the target's units.

### 4. Coefficient of Determination (R²)

\[
R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}
{\sum(y_i-\bar{y})^2}
\]

Measures performance relative to predicting the target mean. A negative test-set \(R^2\) is possible.

## ⚠️ Important Concepts

### Multicollinearity

Multicollinearity occurs when two or more input features are strongly linearly related. It can make coefficient estimates unstable and difficult to interpret.

Possible approaches include:

- Checking feature correlations.
- Calculating the Variance Inflation Factor (VIF).
- Removing redundant features when appropriate.
- Using regularization methods such as Ridge Regression.

### Feature Scaling

Ordinary least-squares Linear Regression generally does not require feature scaling to calculate its fitted predictions. However, scaling can be helpful when comparing coefficient magnitudes or using regularized models.

### Data Leakage

Fit preprocessing steps only on training data. Use a Scikit-learn `Pipeline` to keep preprocessing and model training together and avoid leakage from the test set.

### Assumptions

Check the linearity of the conditional relationship, error independence, zero conditional mean, constant error variance, and influential observations. Normality of errors is primarily relevant for conventional small-sample statistical inference, rather than being necessary to fit the model.

## 🚀 Practical Projects

1. **House Price Prediction** — predict house prices using multiple property features.
2. **Student Performance Prediction** — estimate scores from study time, attendance, and assignment performance.
3. **Sales Prediction** — predict sales from advertising expenditure across several channels.
4. **Energy Consumption Prediction** — estimate energy use from temperature, occupancy, and building characteristics.

Use suitable datasets, perform exploratory analysis, establish a baseline, evaluate on held-out data, and document the limitations of your model.

## 🎤 Interview Questions

1. What is Multiple Linear Regression?
2. How does it differ from Simple Linear Regression?
3. How do you interpret a regression coefficient?
4. What is the least-squares method?
5. What is multicollinearity, and why does it matter?
6. What is the difference between \(R^2\) and adjusted \(R^2\)?
7. Does Linear Regression require feature scaling?
8. How can outliers affect a regression model?
9. What is the purpose of residual analysis?
10. How can regularization help with correlated features?
11. What is data leakage?
12. Why should model performance be evaluated on unseen data?

## 📚 Recommended Learning Path

1. Learn [Simple Linear Regression](../01-Linear-Regression/README.md).
2. Understand the mathematics of Multiple Linear Regression.
3. Practice model training and evaluation.
4. Study residual analysis and multicollinearity.
5. Explore Polynomial Regression and regularized regression methods.

## ✅ Summary

Multiple Linear Regression extends Simple Linear Regression by using multiple input features to predict a continuous target. Understanding coefficient interpretation, evaluation metrics, data leakage, multicollinearity, and model assumptions helps you build more reliable regression workflows.

**Next step:** Implement the model from scratch, compare it with Scikit-learn, and complete a practical regression project.

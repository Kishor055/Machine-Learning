# File: `08-Regression/03-Polynomial-Regression/polynomial-regression.py`

```
"""
Polynomial Regression using Scikit-learn

Polynomial Regression models nonlinear relationships by adding
polynomial powers of the input features.

Example:
    y = b0 + b1*x + b2*x^2 + ... + bn*x^n

Install dependencies:
    pip install numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# 1. Create a Sample Dataset
# --------------------------------------------------

np.random.seed(42)

X = np.linspace(-5, 5, 100).reshape(-1, 1)

# Generate a nonlinear relationship with some noise.
y = (
    3
    + 2 * X.ravel()
    + 1.5 * X.ravel() ** 2
    + np.random.normal(0, 4, size=100)
)

df = pd.DataFrame({
    "Feature": X.ravel(),
    "Target": y,
})

print("Sample Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)


# --------------------------------------------------
# 2. Visualize the Original Data
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    X,
    y,
    color="steelblue",
    edgecolor="black",
    alpha=0.75,
    label="Observed Data",
)

plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.title("Original Dataset")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 3. Split the Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# --------------------------------------------------
# 4. Create and Train Polynomial Regression Model
# --------------------------------------------------

degree = 2

model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),
    LinearRegression(),
)

model.fit(X_train, y_train)

print(f"\nPolynomial Regression Model (Degree {degree})")
print("Model training completed!")


# --------------------------------------------------
# 5. Display the Learned Equation
# --------------------------------------------------

polynomial_transformer = model.named_steps["polynomialfeatures"]
linear_model = model.named_steps["linearregression"]

feature_names = polynomial_transformer.get_feature_names_out(["X"])

print("\nLearned Coefficients:")

for name, coefficient in zip(feature_names, linear_model.coef_):
    print(f"{name}: {coefficient:.4f}")

print(f"Intercept: {linear_model.intercept_:.4f}")


# --------------------------------------------------
# 6. Make Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
})

results["Residual"] = results["Actual"] - results["Predicted"]

print("\nActual vs Predicted:")
print(results.round(2).head(10).to_string(index=False))


# --------------------------------------------------
# 7. Evaluate Model Performance
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# --------------------------------------------------
# 8. Predict for New Data
# --------------------------------------------------

new_data = np.array([[-3], [0], [2], [4]])

new_predictions = model.predict(new_data)

prediction_results = pd.DataFrame({
    "Feature": new_data.ravel(),
    "Predicted Target": new_predictions,
})

print("\nPredictions for New Data:")
print(prediction_results.round(2).to_string(index=False))


# --------------------------------------------------
# 9. Visualize the Polynomial Regression Curve
# --------------------------------------------------

X_curve = np.linspace(
    X.min(),
    X.max(),
    300,
).reshape(-1, 1)

y_curve = model.predict(X_curve)

plt.figure(figsize=(9, 6))

plt.scatter(
    X_train,
    y_train,
    color="steelblue",
    edgecolor="black",
    alpha=0.7,
    label="Training Data",
)

plt.scatter(
    X_test,
    y_test,
    color="orange",
    edgecolor="black",
    alpha=0.8,
    label="Testing Data",
)

plt.plot(
    X_curve,
    y_curve,
    color="red",
    linewidth=2.5,
    label=f"Polynomial Curve (Degree {degree})",
)

plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.title("Polynomial Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. Visualize Residuals
# --------------------------------------------------

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    color="purple",
    edgecolor="black",
    alpha=0.8,
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
)

plt.xlabel("Predicted Target")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Polynomial Regression Residual Plot")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 11. Compare Different Polynomial Degrees
# --------------------------------------------------

degrees = range(1, 7)
comparison = []

for d in degrees:
    candidate_model = make_pipeline(
        PolynomialFeatures(degree=d, include_bias=False),
        LinearRegression(),
    )

    candidate_model.fit(X_train, y_train)

    train_predictions = candidate_model.predict(X_train)
    test_predictions = candidate_model.predict(X_test)

    comparison.append({
        "Degree": d,
        "Train R2": r2_score(y_train, train_predictions),
        "Test R2": r2_score(y_test, test_predictions),
        "Test RMSE": np.sqrt(
            mean_squared_error(y_test, test_predictions)
        ),
    })

comparison_df = pd.DataFrame(comparison)

print("\nPolynomial Degree Comparison:")
print(comparison_df.round(4).to_string(index=False))

print("\nTip: Choose polynomial degree using validation data or cross-validation,")
print("not just the highest training R².")
```



What is this?

## Run the code

```
pip install numpy pandas matplotlib scikit-learn
python polynomial-regression.py
```

Key concepts demonstrated:

- `PolynomialFeatures` transforms input features into polynomial terms.
- `make_pipeline()` combines feature transformation and linear regression.
- MAE, MSE, RMSE, and \\(R^2\\) evaluate model performance.
- The degree comparison demonstrates how model complexity can affect overfitting.

Important: A polynomial regression model is still linear in its learned coefficients, even though its fitted curve can be nonlinear in the original input feature.

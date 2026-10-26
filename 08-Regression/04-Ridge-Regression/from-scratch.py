# File: `08-Regression/04-Ridge-Regression/from-scratch.py`

```python
"""
Ridge Regression from Scratch using NumPy
=========================================

Ridge Regression is linear regression with L2 regularization.
The penalty discourages large coefficients and can help reduce
overfitting and improve coefficient stability.

Objective function:
    J(w, b) = (1/n) * sum((y_pred - y)^2) + alpha * sum(w^2)

The intercept (bias) is not regularized.

Features:
    1. Synthetic dataset generation
    2. Train-test split
    3. Feature standardization
    4. Ridge Regression using gradient descent
    5. MAE, MSE, RMSE, and R² metrics
    6. Comparison with ordinary least squares
    7. Visualization of predictions and training loss

Install dependencies:
    pip install numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================
# 1. Generate a Synthetic Dataset
# ==================================================

X, y = make_regression(
    n_samples=300,
    n_features=5,
    n_informative=5,
    noise=20,
    random_state=42,
)

print("=" * 55)
print("RIDGE REGRESSION FROM SCRATCH")
print("=" * 55)

print("\nDataset Shape:", X.shape)
print("Target Shape:", y.shape)

dataset = pd.DataFrame(
    X,
    columns=[f"Feature_{i + 1}" for i in range(X.shape[1])],
)
dataset["Target"] = y

print("\nFirst Five Rows:")
print(dataset.head())


# ==================================================
# 2. Split the Dataset
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# ==================================================
# 3. Standardize Features
# ==================================================

# Fit the scaler only on training data to avoid leakage.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==================================================
# 4. Implement Ridge Regression from Scratch
# ==================================================

class RidgeRegression:
    """
    Ridge Regression using batch gradient descent.

    Objective:
        J(w, b) = mean((X @ w + b - y)^2) + alpha * sum(w^2)

    The intercept b is not regularized.
    """

    def __init__(
        self,
        learning_rate=0.01,
        epochs=5000,
        alpha=1.0,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def fit(self, X, y):
        """Train the model using batch gradient descent."""

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if X.shape[0] == 0:
            raise ValueError("Training data cannot be empty.")

        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain finite values.")

        n_samples, n_features = X.shape

        # Initialize parameters.
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):

            # Forward pass.
            y_pred = X @ self.weights + self.bias

            # Prediction errors.
            errors = y_pred - y

            # Gradient of mean squared error + L2 penalty.
            dw = (
                (2.0 / n_samples) * (X.T @ errors)
                + 2.0 * self.alpha * self.weights
            )

            # Do not regularize the intercept.
            db = (2.0 / n_samples) * np.sum(errors)

            # Gradient descent update.
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Record the objective value after the update.
            updated_predictions = X @ self.weights + self.bias
            updated_errors = updated_predictions - y

            mse = np.mean(updated_errors ** 2)
            penalty = self.alpha * np.sum(self.weights ** 2)
            loss = mse + penalty

            self.loss_history.append(loss)

        return self

    def predict(self, X):
        """Predict continuous target values."""

        if self.weights is None:
            raise ValueError("Train the model before making predictions.")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[1] != len(self.weights):
            raise ValueError(
                "X must contain the same number of features used in training."
            )

        return X @ self.weights + self.bias


# ==================================================
# 5. Train the Ridge Regression Model
# ==================================================

model = RidgeRegression(
    learning_rate=0.01,
    epochs=5000,
    alpha=1.0,
)

model.fit(X_train_scaled, y_train)

print("\nModel Training Completed!")

print("\nIntercept:", round(model.bias, 4))

print("\nLearned Coefficients:")

for i, weight in enumerate(model.weights):
    print(f"Feature_{i + 1}: {weight:.4f}")


# ==================================================
# 6. Make Predictions
# ==================================================

y_pred = model.predict(X_test_scaled)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Residual": y_test - y_pred,
})

print("\nActual vs Predicted Values:")
print(results.head(10).round(2).to_string(index=False))


# ==================================================
# 7. Evaluate the Model
# ==================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ==================================================
# 8. Compare with Ordinary Linear Regression
# ==================================================

linear_model = LinearRegression()

linear_model.fit(X_train_scaled, y_train)

linear_predictions = linear_model.predict(X_test_scaled)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)

ridge_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

comparison = pd.DataFrame({
    "Model": [
        "Ordinary Linear Regression",
        "Ridge Regression from Scratch",
    ],
    "RMSE": [
        linear_rmse,
        ridge_rmse,
    ],
    "R2": [
        r2_score(y_test, linear_predictions),
        r2_score(y_test, y_pred),
    ],
})

print("\nModel Comparison:")
print(comparison.round(4).to_string(index=False))


# ==================================================
# 9. Visualize Training Loss
# ==================================================

plt.figure(figsize=(9, 5))

plt.plot(
    model.loss_history,
    label="Training Objective",
)

plt.xlabel("Epoch")
plt.ylabel("Objective Value")
plt.title("Ridge Regression: Gradient Descent")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# 10. Actual vs Predicted Visualization
# ==================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred,
    edgecolor="black",
    alpha=0.75,
    label="Ridge Predictions",
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    label="Perfect Prediction",
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted: Ridge Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# 11. Residual Plot
# ==================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    edgecolor="black",
    alpha=0.75,
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Ridge Regression Residual Plot")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# 12. Predict New Samples
# ==================================================

new_samples = np.array([
    [0.5, 1.2, -0.3, 0.8, 1.0],
    [-1.0, 0.4, 0.7, -0.2, 0.3],
])

# Apply the same scaler fitted on training data.
new_samples_scaled = scaler.transform(new_samples)

new_predictions = model.predict(new_samples_scaled)

print("\nPredictions for New Samples:")

for i, prediction in enumerate(new_predictions, start=1):
    print(f"Sample {i}: {prediction:.4f}")


# ==================================================
# 13. Experiment with Different Alpha Values
# ==================================================

alphas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]

print("\nEffect of Regularization Strength:")

for alpha in alphas:

    candidate = RidgeRegression(
        learning_rate=0.01,
        epochs=5000,
        alpha=alpha,
    )

    candidate.fit(X_train_scaled, y_train)

    candidate_predictions = candidate.predict(X_test_scaled)

    candidate_rmse = np.sqrt(
        mean_squared_error(y_test, candidate_predictions)
    )

    coefficient_norm = np.linalg.norm(candidate.weights)

    print(
        f"Alpha: {alpha:6.2f} | "
        f"RMSE: {candidate_rmse:9.4f} | "
        f"Coefficient Norm: {coefficient_norm:9.4f}"
    )


print("\nProgram Completed Successfully!")
```



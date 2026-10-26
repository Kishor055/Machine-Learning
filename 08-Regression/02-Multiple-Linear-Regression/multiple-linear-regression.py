# File: `08-Regression/02-Multiple-Linear-Regression/multiple-linear-regression.py`

```
"""
Multiple Linear Regression using Scikit-learn

Predict house prices using multiple independent variables.

Features:
    - Area (square feet)
    - Number of bedrooms
    - House age (years)

Target:
    - House price

Install dependencies:
    pip install numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# 1. Create a Sample Dataset
# --------------------------------------------------

data = {
    "Area": [
        1000, 1200, 1500, 1800, 2000,
        2200, 2500, 2700, 3000, 3200,
        1100, 1400, 1600, 1900, 2100,
        2400, 2600, 2800, 3100, 3300,
    ],
    "Bedrooms": [
        2, 2, 3, 3, 3,
        4, 4, 4, 5, 5,
        2, 3, 3, 3, 4,
        4, 4, 5, 5, 5,
    ],
    "Age": [
        10, 8, 5, 7, 4,
        6, 3, 5, 2, 1,
        12, 9, 6, 8, 5,
        4, 7, 3, 2, 1,
    ],
    "Price": [
        30, 35, 45, 50, 60,
        65, 75, 78, 90, 95,
        32, 42, 48, 53, 62,
        72, 76, 85, 92, 98,
    ],
}

df = pd.DataFrame(data)

print("Sample Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)


# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# --------------------------------------------------
# 3. Split Dataset into Training and Testing Sets
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
# 4. Create and Train the Model
# --------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed!")


# --------------------------------------------------
# 5. Display Model Parameters
# --------------------------------------------------

print("\nIntercept:", round(model.intercept_, 4))

print("\nFeature Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.4f}")

print("\nRegression Equation:")

equation = (
    f"Price = {model.intercept_:.4f}"
    f" + ({model.coef_[0]:.4f} * Area)"
    f" + ({model.coef_[1]:.4f} * Bedrooms)"
    f" + ({model.coef_[2]:.4f} * Age)"
)

print(equation)


# --------------------------------------------------
# 6. Make Predictions on Test Data
# --------------------------------------------------

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred,
})

results["Error"] = (
    results["Actual Price"] - results["Predicted Price"]
)

print("\nActual vs Predicted Prices:")
print(results.round(2).to_string(index=False))


# --------------------------------------------------
# 7. Evaluate Model Performance
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# --------------------------------------------------
# 8. Predict Price for a New House
# --------------------------------------------------

new_house = pd.DataFrame({
    "Area": [2300],
    "Bedrooms": [4],
    "Age": [3],
})

predicted_price = model.predict(new_house)

print("\nNew House Details:")
print(new_house.to_string(index=False))

print(
    f"\nPredicted House Price: "
    f"{predicted_price[0]:.2f} price units"
)


# --------------------------------------------------
# 9. Visualize Actual vs Predicted Values
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    color="blue",
    edgecolor="black",
    s=80,
    label="Predictions",
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linestyle="--",
    label="Perfect Prediction",
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. Visualize Residuals
# --------------------------------------------------

residuals = y_test.values - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    color="green",
    edgecolor="black",
    s=80,
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
)

plt.xlabel("Predicted House Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```



What is this?

## How to run the file

1. Save the code as `multiple-linear-regression.py` inside the `02-Multiple-Linear-Regression` folder.
2. Install the dependencies:

```
pip install numpy pandas matplotlib scikit-learn
```

3\. Run the script:

```
python multiple-linear-regression.py
```

Note: The dataset is synthetic, and the prices are expressed in arbitrary price units. The metrics show performance on a small test split, so they should not be interpreted as evidence of real-world predictive accuracy.

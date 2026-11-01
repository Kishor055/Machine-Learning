# 01 - Feature Creation

## 📌 Overview

Feature Creation is an important feature engineering technique in Machine Learning that involves creating new features (input variables) from existing features using mathematical operations, domain knowledge, transformations, and combinations.

The objective is to extract additional meaningful information from raw datasets, improve model performance, and help machine learning algorithms identify hidden patterns and relationships.

Instead of relying only on the original dataset features, feature creation generates more informative representations that can improve predictive accuracy and model generalization.

## 🎯 Learning Objectives

By completing this module, you will learn how to:

- Understand the fundamentals of feature creation.
- Create new features using existing numerical variables.
- Generate interaction features between multiple variables.
- Perform mathematical operations on existing features.
- Create domain-specific features using business knowledge.
- Extract meaningful information from date and time columns.
- Generate ratio, aggregate, and combination features.
- Understand polynomial and interaction features.
- Evaluate the impact of newly created features on model performance.
- Apply feature creation techniques using Pandas and Scikit-learn.

## 📚 Topics Covered

### 1. Introduction to Feature Creation

- What is Feature Creation?
- Importance of Feature Creation in Machine Learning.
- Feature Creation vs. Feature Selection.
- Role of Domain Knowledge.
- Benefits and Challenges of Feature Creation.

### 2. Mathematical Feature Creation

Creating new features through mathematical operations.

Examples:

- Addition and subtraction.
- Multiplication and division.
- Ratios and percentages.
- Logarithmic transformations.
- Square and square-root transformations.

```python
df["total_amount"] = df["price"] * df["quantity"]

df["price_per_unit"] = df["total_amount"] / df["quantity"]

df["income_expense_ratio"] = df["income"] / (df["expenses"] + 1)
```

### 3. Interaction Features

Interaction features combine two or more existing variables to capture relationships that individual features might not represent.

Examples:

- Age × Income.
- Price × Quantity.
- Experience × Education.
- Area × Number of Rooms.

```python
df["age_income_interaction"] = df["age"] * df["income"]

df["price_quantity_interaction"] = df["price"] * df["quantity"]
```

### 4. Date and Time Feature Creation

Extract useful information from date and time columns.

Commonly created features:

- Year.
- Month.
- Day.
- Day of the week.
- Quarter.
- Week of the year.
- Weekend indicator.
- Month start and month end.
- Hour and minute.

```python
df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
```

### 5. Aggregation-Based Features

Create additional features using statistical aggregations.

Examples:

- Total spending per customer.
- Average purchase amount.
- Maximum transaction value.
- Minimum product price.
- Number of transactions.
- Standard deviation of purchases.

```python
df["customer_avg_purchase"] = (
    df.groupby("customer_id")["purchase_amount"]
      .transform("mean")
)

df["customer_total_purchase"] = (
    df.groupby("customer_id")["purchase_amount"]
      .transform("sum")
)
```

### 6. Polynomial Features

Polynomial features introduce nonlinear relationships by generating powers and combinations of existing numerical features.

Examples:

- \(x^2\)
- \(x^3\)
- \(x_1 \times x_2\)

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_poly = poly.fit_transform(X)
```

### 7. Domain-Based Feature Creation

Use domain-specific knowledge to construct meaningful features.

Examples:

| Domain | Original Features | Created Feature |
|---|---|---|
| Banking | Income, Expenses | Savings |
| E-commerce | Price, Quantity | Total Amount |
| Healthcare | Weight, Height | BMI |
| Education | Marks, Total Marks | Percentage |
| Real Estate | Area, Price | Price per Sq. Ft. |
| Marketing | Clicks, Impressions | Click-Through Rate |

```python
df["savings"] = df["income"] - df["expenses"]

df["bmi"] = df["weight"] / (df["height"] ** 2)

df["percentage"] = (
    df["obtained_marks"] / df["total_marks"]
) * 100
```

## 🛠️ Technologies & Libraries

- **Python** - Programming language.
- **NumPy** - Mathematical operations and numerical computations.
- **Pandas** - Data manipulation and feature generation.
- **Scikit-learn** - Polynomial and automated feature generation.
- **Matplotlib** - Data visualization.
- **Seaborn** - Statistical visualization.

## 📂 Project Structure

```text
01-Feature-Creation/
│
├── README.md
├── feature-creation.py
├── mathematical-features.py
├── interaction-features.py
├── datetime-features.py
├── aggregation-features.py
├── polynomial-features.py
├── domain-based-features.py
├── visualization.py
└── practice.md
```

## 💻 Installation

Install the required Python libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

## 🚀 Practical Implementation

Follow these steps when implementing feature creation:

1. Load the dataset using Pandas.
2. Understand the existing features.
3. Identify relationships between variables.
4. Apply mathematical transformations.
5. Generate interaction features.
6. Extract date and time information.
7. Create aggregation-based features.
8. Apply polynomial feature generation when appropriate.
9. Check for missing values and infinite values.
10. Evaluate the usefulness of newly generated features.
11. Compare model performance before and after feature creation.

## ⚠️ Important Considerations

- Avoid creating redundant or irrelevant features.
- Use domain knowledge to generate meaningful features.
- Handle division by zero when creating ratio features.
- Prevent data leakage during feature generation.
- Calculate aggregation features using training data only when appropriate.
- Fit feature-generation transformations on training data and reuse them on validation and test data.
- Be careful with high-dimensional polynomial features.
- Consider multicollinearity when generating highly correlated features.
- Evaluate newly created features using cross-validation.

## 📊 Advantages

- Improves predictive performance.
- Helps models identify hidden relationships.
- Captures nonlinear patterns.
- Incorporates domain-specific knowledge.
- Provides more informative input variables.
- Can improve model interpretability.

## ❌ Challenges

- Can increase dataset dimensionality.
- May introduce irrelevant features.
- Can increase computational complexity.
- Requires domain expertise in some applications.
- May cause overfitting if too many features are generated.
- Requires careful handling to avoid data leakage.

## 📈 Model Evaluation

To understand whether feature creation improves model performance, compare the model before and after feature engineering.

Common evaluation metrics include:

**Regression:**
- Mean Absolute Error (MAE).
- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- R² Score.

**Classification:**
- Accuracy.
- Precision.
- Recall.
- F1-Score.
- ROC-AUC.

Use the same data splits and evaluation strategy for a fair comparison.

## 🧠 Key Takeaways

- Feature creation generates new meaningful variables from existing data.
- Mathematical operations can reveal hidden relationships.
- Interaction features capture relationships between multiple variables.
- Date and time features provide useful temporal information.
- Aggregation features summarize group-level behavior.
- Polynomial features help represent nonlinear relationships.
- Domain knowledge plays an important role in feature engineering.
- Always validate the impact of newly created features.

## 📖 References

1. [Scikit-learn - PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
2. [Pandas - DataFrame Documentation](https://pandas.pydata.org/docs/reference/frame.html)
3. [Feature Engineering for Machine Learning - Alice Zheng and Amanda Casari](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

---

**Next Topic:** Feature Transformation

**Module:** 11 - Feature Engineering

**Previous Topic:** Introduction to Feature Engineering

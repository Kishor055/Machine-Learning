# Introduction to Machine Learning

 ## 1\. What is Machine Learning?

 Machine Learning (ML) is a branch of Artificial Intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every situation.

 ### Simple Example

 Instead of writing rules for predicting house prices, we provide the model with historical house data.

```
House Data
    ↓
Machine Learning Model
    ↓
Learn Patterns
    ↓
Predict New House Price
```

---

 ## 2\. Traditional Programming

 In traditional programming, we provide:

```
Data + Rules
    ↓
Program
    ↓
Output
```

 Example:

```
temperature = 35

if temperature > 30:
    print("Hot")
else:
    print("Normal")
```

 The programmer creates the rules.

---

 ## 3\. Machine Learning

 In Machine Learning, we provide:

```
Data + Expected Output
    ↓
Machine Learning Algorithm
    ↓
Learned Model
```

 The model learns patterns from the provided data.

 Example:

```
House Area → House Price

1000 sq ft → ₹40 Lakhs
1500 sq ft → ₹60 Lakhs
2000 sq ft → ₹80 Lakhs
```

 The model can learn the relationship between area and price.

---

 ## 4\. Why Machine Learning?

 Machine Learning is useful when writing explicit rules is difficult.

 Common applications include:

 - Spam detection
- Recommendation systems
- Fraud detection
- House price prediction
- Image recognition
- Speech recognition
- Customer churn prediction
- Medical prediction
- Search engines
- Financial forecasting

---

 ## 5\. Basic Machine Learning Terminology

 ### Dataset

 A dataset is a collection of data used for analysis or Machine Learning.

 Example:

```
Area    Bedrooms    Price
1000    2           40
1500    3           60
2000    4           80
```

---

 ### Sample

 A sample is one individual observation or row in a dataset.

```
1000    2    40
```

 This is one sample.

---

 ### Feature

 A feature is an input variable used by the model.

 Example:

```
Area
Bedrooms
Bathrooms
Age
```

 These can be features for a house price model.

---

 ### Label

 A label is the value the model is expected to predict.

 Example:

```
Price
```

---

 ### Target

 Target is another common name for the output variable that the model predicts.

```
Features → Target
```

 Example:

```
Area + Bedrooms + Bathrooms → Price
```

---

 ## 6\. Features and Labels Example

 Consider:

```
Area    Bedrooms    Price
1000    2           40
1500    3           60
2000    4           80
```

 Features:

```
Area
Bedrooms
```

 Target:

```
Price
```

 In Python:

```
X = [
    [1000, 2],
    [1500, 3],
    [2000, 4]
]

y = [40, 60, 80]
```

 Usually:

```
X = Features
y = Target
```

---

 ## 7\. Training Data

 Training data is the data used to teach the Machine Learning model.

```
Training Data
      ↓
Machine Learning Algorithm
      ↓
Trained Model
```

 The model finds patterns in the training data.

---

 ## 8\. Testing Data

 Testing data is used to evaluate the trained model on data it has not seen during training.

```
Training Data → Train Model

Testing Data → Evaluate Model
```

 This helps determine whether the model can generalize to new data.

---

 ## 9\. Train-Test Split

 A dataset is commonly divided into training and testing sets.

 Example:

```
100% Dataset
     │
     ├── 80% Training Data
     │
     └── 20% Testing Data
```

 Python:

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

 ## 10\. Machine Learning Model

 A model is the learned representation of patterns in data.

 For example:

```
Input
  ↓
Model
  ↓
Prediction
```

 For house price prediction:

```
Area = 2000 sq ft
Bedrooms = 3
        ↓
     Model
        ↓
Predicted Price
```

---

 ## 11\. Training a Model

 Training means allowing the algorithm to learn patterns from the training data.

 Example:

```
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

 The `fit()` method trains the model.

---

 ## 12\. Prediction

 After training, the model can make predictions.

```
prediction = model.predict([[2000]])
```

 Example:

```
Input:
2000 sq ft

Output:
₹80 Lakhs
```

---

 ## 13\. Model Evaluation

 A model needs to be evaluated to determine how well it performs.

 For regression problems, common metrics include:

 - MAE
- MSE
- RMSE
- R²

 Example:

```
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print(mae)
```

---

 ## 14\. Generalization

 Generalization means the ability of a Machine Learning model to perform well on new, unseen data.

 A good model should learn general patterns instead of memorizing the training data.

```
Training Data
     ↓
    Model
     ↓
Unseen Data
     ↓
Good Predictions
```

---

 ## 15\. Overfitting

 Overfitting occurs when a model learns the training data too closely, including noise and unnecessary patterns.

 Example:

```
Training Performance → Very High
Testing Performance  → Low
```

 The model performs well on training data but poorly on new data.

---

 ## 16\. Underfitting

 Underfitting occurs when a model is too simple to learn the important patterns in the data.

 Example:

```
Training Performance → Low
Testing Performance  → Low
```

 The model performs poorly on both training and testing data.

---

 ## 17\. Machine Learning Workflow

 A basic Machine Learning workflow is:

```
1. Define Problem
       ↓
2. Collect Data
       ↓
3. Understand Data
       ↓
4. Clean Data
       ↓
5. Prepare Data
       ↓
6. Split Data
       ↓
7. Select Model
       ↓
8. Train Model
       ↓
9. Make Predictions
       ↓
10. Evaluate Model
       ↓
11. Improve Model
       ↓
12. Deploy Model
```

---

 ## 18\. Simple Example

 Suppose we want to predict house prices.

 ### Input Data

```
Area    Bedrooms    Price
1000    2           40
1200    2           48
1500    3           60
1800    3           72
2000    4           80
```

 ### Features

```
Area
Bedrooms
```

 ### Target

```
Price
```

 ### Model

```
Linear Regression
```

 ### Training

```
model.fit(X_train, y_train)
```

 ### Prediction

```
model.predict(X_test)
```

 ### Evaluation

```
model.score(X_test, y_test)
```

---

 ## 19\. Regression vs Classification

 Machine Learning problems can have different types of outputs.

 ### Regression

 Predicts a continuous numerical value.

 Examples:

```
House Price
Salary
Temperature
Sales
```

 Example:

```
House Price = ₹75.5 Lakhs
```

 ### Classification

 Predicts a category or class.

 Examples:

```
Spam / Not Spam
Pass / Fail
Disease / No Disease
Fraud / Not Fraud
```

 Example:

```
Email → Spam
```

---

 ## 20\. Parameters and Hyperparameters

 ### Parameters

 Parameters are values learned by the model during training.

 Example in Linear Regression:

```
Coefficient
Intercept
```

 ### Hyperparameters

 Hyperparameters are values configured before training.

 Examples:

```
Learning Rate
Number of Trees
Maximum Depth
Number of Neighbors
```

---

 ## 21\. Simple Mathematical Representation

 A simple Linear Regression model can be represented as:

```
y = mx + b
```

 Where:

```
y = prediction
m = coefficient
x = input
b = intercept
```

 Example:

```
Area → Price
```

 The model learns `m` and `b` from the training data.

---

 ## 22\. Important Machine Learning Concepts

 Before moving forward, understand these concepts:

```
Machine Learning
Dataset
Sample
Feature
Label
Target
Training Data
Testing Data
Model
Training
Prediction
Evaluation
Generalization
Overfitting
Underfitting
Parameter
Hyperparameter
Regression
Classification
```

---

 ## 23\. Key Takeaway

 Machine Learning allows computers to learn patterns from data and use those patterns to make predictions or decisions.

 The basic idea is:

```
Data
 ↓
Features + Target
 ↓
Train Model
 ↓
Learn Patterns
 ↓
Make Predictions
 ↓
Evaluate
 ↓
Improve
```

---

 ## Next Topic

 Continue with:

```
02-AI-vs-ML-vs-DL/
```

 **Author:** KISHOR KAKDE

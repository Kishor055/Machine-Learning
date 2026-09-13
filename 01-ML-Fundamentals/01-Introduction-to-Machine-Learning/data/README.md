# Data

 This folder contains datasets used for the **Introduction to Machine Learning** examples and exercises.

 ## Dataset Structure

```
data/
├── README.md
├── house_prices.csv
└── student_scores.csv
```

 ## Datasets

 ### `house_prices.csv`

 Used for basic regression examples.

 **Features:**

 - `area`
- `bedrooms`
- `bathrooms`
- `age`
- `parking`

 **Target:**

 - `price`

 ### `student_scores.csv`

 Used for basic prediction examples.

 **Features:**

 - `hours_studied`
- `attendance`
- `assignments`

 **Target:**

 - `exam_score`

 ## Example

 Load a dataset using Pandas:

```
import pandas as pd

data = pd.read_csv("data/house_prices.csv")

print(data.head())
print(data.shape)
print(data.info())
```

 ## Data Guidelines

 - Keep raw datasets unchanged.
- Use small datasets for beginner examples.
- Do not commit private or sensitive data.
- Document the source of external datasets.
- Keep datasets relevant to the topic being studied.

 **Author:** KISHOR KAKDE

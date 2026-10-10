# ============================================================
# One-Hot Encoding
# Machine Learning - Data Preprocessing
# ============================================================

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------
# 1. Create a Sample Dataset
# ------------------------------------------------------------

data = {
    "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Name": [
        "Amit", "Priya", "Rahul", "Sneha",
        "Karan", "Neha", "Arjun", "Pooja"
    ],
    "City": [
        "Pune", "Mumbai", "Delhi", "Pune",
        "Mumbai", "Delhi", "Nagpur", "Pune"
    ],
    "Department": [
        "Computer Science", "Mechanical", "Electrical",
        "Computer Science", "Civil", "Mechanical",
        "Electrical", "Computer Science"
    ],
    "Marks": [85, 92, 78, 88, 76, 90, 82, 95]
}

df = pd.DataFrame(data)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df)

# ------------------------------------------------------------
# 2. Identify Categorical Features
# ------------------------------------------------------------

categorical_columns = ["City", "Department"]

print("\nCategorical Columns:")
print(categorical_columns)

# ------------------------------------------------------------
# 3. Apply One-Hot Encoding Using Scikit-learn
# ------------------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=int
)

encoded_array = encoder.fit_transform(df[categorical_columns])

# Generate meaningful feature names
encoded_columns = encoder.get_feature_names_out(categorical_columns)

# Convert encoded array into a DataFrame
encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoded_columns,
    index=df.index
)

print("\n" + "=" * 60)
print("ONE-HOT ENCODED FEATURES")
print("=" * 60)
print(encoded_df)

# ------------------------------------------------------------
# 4. Combine Original and Encoded Features
# ------------------------------------------------------------

# Remove original categorical columns
df_without_categories = df.drop(columns=categorical_columns)

# Combine numerical, descriptive, and encoded features
final_df = pd.concat(
    [df_without_categories, encoded_df],
    axis=1
)

print("\n" + "=" * 60)
print("FINAL PROCESSED DATASET")
print("=" * 60)
print(final_df)

# ------------------------------------------------------------
# 5. Display Dataset Information
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Original Dataset Shape:", df.shape)
print("Encoded Features Shape:", encoded_df.shape)
print("Final Dataset Shape:", final_df.shape)

print("\nEncoded Feature Names:")
print(list(encoded_columns))

print("\nMissing Values:")
print(final_df.isnull().sum())

# ------------------------------------------------------------
# 6. Save the Processed Dataset
# ------------------------------------------------------------

output_file = "one_hot_encoded_students.csv"

final_df.to_csv(output_file, index=False)

print(f"\nProcessed dataset saved successfully: {output_file}")

# ------------------------------------------------------------
# 7. Transform New Data Using the Existing Encoder
# ------------------------------------------------------------

new_students = pd.DataFrame({
    "City": ["Pune", "Mumbai"],
    "Department": ["Computer Science", "Mechanical"]
})

new_encoded_array = encoder.transform(new_students[categorical_columns])

new_encoded_df = pd.DataFrame(
    new_encoded_array,
    columns=encoded_columns
)

print("\n" + "=" * 60)
print("ENCODING NEW STUDENT DATA")
print("=" * 60)
print(new_students)

print("\nEncoded New Student Data:")
print(new_encoded_df)

# ============================================================
# End of One-Hot Encoding Example
# ============================================================

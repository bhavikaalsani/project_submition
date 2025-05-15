import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("C:/Users/alsan/Customer360_AI/data/customer_360_data.csv")

# Display the first few rows
print("Sample data:")
print(df.head())

# Show column names and data types
print("\nColumns and Data Types:")
print(df.dtypes)

# Show summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Handle missing values
# For numerical columns, fill missing with median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# For categorical columns, fill missing with mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Encode categorical columns
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['marital_status'] = df['marital_status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})

# Scale selected numerical columns
scale_cols = ['age', 'income', 'credit_score', 'account_balance', 'loan_amount', 'num_transactions']
scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# Print the cleaned dataset
print("\nCleaned Data:")
print(df.head())

# Ensure output directory exists
output_dir = "C:/Users/alsan/Customer360_AI/data"
os.makedirs(output_dir, exist_ok=True)

# Save cleaned data to a new CSV file
df.to_csv(os.path.join(output_dir, "processed_customer_360.csv"), index=False)
print("\n Cleaned data saved to data/processed_customer_360.csv")

# -------------------------------
# Correlation Heatmap
# -------------------------------
corr_matrix = df.corr()

plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)

plt.title("Correlation Heatmap of Customer 360 Features", fontsize=16)
plt.tight_layout()
plt.show()

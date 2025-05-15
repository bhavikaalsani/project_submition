import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the trained model
try:
    model = joblib.load("C:/Users/alsan/Customer360_AI/models/random_forest_model.pkl")
    print("[✓] Loaded model successfully.")
except FileNotFoundError:
    print("[!] Model file not found. Please check the path.")
    exit(1)

# Check if sample input CSV exists
sample_input_path = "C:/Users/alsan/Customer360_AI/data/sample_input.csv"
if os.path.exists(sample_input_path):
    print(f"[✓] Found external input file: {sample_input_path}")
    sample_data = pd.read_csv(sample_input_path)
else:
    print("[!] sample_input.csv not found. Using fallback sample from processed dataset.")
    df = pd.read_csv("C:/Users/alsan/Customer360_AI/data/processed_customer_360.csv")
    sample_data = df.sample(5, random_state=42).drop("fraud_risk", axis=1)

# Label Encode categorical columns (if any exist in this sample)
categorical_cols = ['gender', 'marital_status']
for col in categorical_cols:
    if col in sample_data.columns:
        le = LabelEncoder()
        if col == 'gender':
            le.fit(['Female', 'Male'])
        elif col == 'marital_status':
            le.fit(['Single', 'Married', 'Divorced'])
        try:
            sample_data[col] = le.transform(sample_data[col])
        except Exception as e:
            print(f"[!] Encoding error in column {col}: {e}")
    else:
        print(f"[!] Column {col} not found in the sample data.")

# Ensure the input data has the same feature columns as the model was trained on
expected_columns = ['age', 'gender', 'marital_status', 'income', 'credit_score']
missing_cols = set(expected_columns) - set(sample_data.columns)
if missing_cols:
    print(f"[!] Missing columns in input data: {missing_cols}")
    exit(1)

# Check for missing data and handle it
if sample_data.isnull().sum().any():
    print("[!] Missing data found. Filling missing values with zero.")
    sample_data.fillna(0, inplace=True)

# Predict fraud probabilities
probabilities = model.predict_proba(sample_data)
predictions = (probabilities[:, 1] >= 0.3).astype(int)  # Adjust threshold as needed

# Display prediction results
print("\n[✓] Prediction Probabilities (Non-Fraud vs Fraud):")
print(probabilities)

print("\n[✓] Predicted Fraud Risk:")
print(predictions)

# Visualization: Bar and Pie charts
unique_preds, counts = np.unique(predictions, return_counts=True)
label_mapping = {0: 'No Fraud', 1: 'Fraud'}
labels = [label_mapping[pred] for pred in unique_preds]

# Bar Chart
plt.figure(figsize=(6, 4))
sns.barplot(x=labels, y=counts, palette='viridis')
plt.title('Fraud Prediction Count')
plt.ylabel('Number of Customers')
plt.xlabel('Prediction')
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(5, 5))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Fraud Prediction Distribution')
plt.tight_layout()
plt.show()

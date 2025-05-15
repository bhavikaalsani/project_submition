import pandas as pd
import os

# Sample data with raw categorical columns (no encoding)
data = {
    'age': [20, 60, 35, 48, 29],
    'gender': [1, 0, 1, 0, 1],
    'marital_status': [1, 1, 0, 1, 0],
    'income': [15000, 90000, 50000, 20000, 100000],
    'credit_score': [400, 820, 650, 420, 810],
    'account_balance': [100, 50000, 15000, 500, 80000],
    'loan_amount': [30000, 5000, 10000, 40000, 3000],
    'num_transactions': [1, 40, 10, 1, 50],
}


# Create DataFrame
df = pd.DataFrame(data)

# Path to save CSV
output_path = '../data/sample_input.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ sample_input.csv created at {output_path}")

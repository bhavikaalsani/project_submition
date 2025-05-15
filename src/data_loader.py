import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_config(config_path="config.yaml"):
    """
    Load the configuration file (YAML).
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(config):
    """
    Load dataset for training.
    """
    # Modify to match your dataset path or logic
    data_path = config["data"]["data_path"]
    df = pd.read_csv(data_path)

    # Encode categorical columns (gender, marital_status)
    label_encoders = {}
    for column in ['gender', 'marital_status']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Separate features and target variable
    X = df.drop(columns=config["data"]["target_column"])
    y = df[config["data"]["target_column"]]
    
    # Scale numerical columns
    scale_cols = ['age', 'income', 'credit_score', 'account_balance', 'loan_amount', 'num_transactions']
    scaler = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])

    return X, y

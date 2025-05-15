import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    logging.info(f"Dataset columns: {data.columns}")
    return data

# Preprocess data (split into features and target)
def preprocess_data(data):
    X = data.drop(columns=["fraud_risk", "customer_id"])
    y = data["fraud_risk"]
    return X, y

# Function for splitting data and balancing it
def split_and_balance_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply ADASYN to balance the dataset
    ada = ADASYN(sampling_strategy="minority", random_state=42)
    X_train_resampled, y_train_resampled = ada.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test

# Function for training the model and tuning hyperparameters
def train_model(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    
    return best_model

# Function to evaluate the model
def evaluate_model(best_model, X_test, y_test):
    # Model predictions
    y_pred = best_model.predict(X_test)
    
    # Accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy:.2f}")
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Main function
def main():
    file_path = 'C:/Users/alsan/Customer360_AI/data/processed_customer_360.csv'  # Provide the correct path to your data file
    data = load_data(file_path)
    
    X, y = preprocess_data(data)
    X_train_resampled, X_test, y_train_resampled, y_test = split_and_balance_data(X, y)
    
    # Train the model
    best_model = train_model(X_train_resampled, y_train_resampled)
    
    # Evaluate the model
    evaluate_model(best_model, X_test, y_test)

if __name__ == "__main__":
    main()


   

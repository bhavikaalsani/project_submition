
# 🧠 Customer 360 – Retail Banking Intelligence System

An AI-powered Customer 360 platform that delivers a unified view of retail banking customers, detects fraudulent loan activity, and offers personalized product recommendations. This project is designed to support risk analysts, marketers, and data scientists in gaining insights and enhancing decision-making.

---

## 🚀 Key Features

- ✅ **Customer 360 Profiling**  
  Consolidates demographic, behavioral, and transactional data into a comprehensive customer profile.

- 🔐 **Loan Fraud Detection**  
  Applies machine learning models to predict the probability of fraudulent loan applications.

- 🎯 **Personalized Product Recommendation**  
  Recommends banking products based on behavioral similarity and customer segmentation.

- 📊 **Modular Design**  
  Clean separation of data preprocessing, model training, and inference pipelines for scalability.

---

## 📁 Project Structure
customer_360_project/
│
├── data/ # Raw and processed data files
│ ├── sample_input.csv # Sample customer data
│ └── sample_output.csv # Prediction results
│
├── notebooks/ # Jupyter notebooks for EDA and experimentation
│ ├── eda.ipynb
│ ├── model_fraud.ipynb
│ └── model_recommendation.ipynb
│
├── src/ # Source code
│ ├── preprocessing.py # Data cleaning and feature engineering
│ ├── train_model.py # Model training script
│ ├── fraud_detection.py # Fraud prediction logic
│ └── recommender.py # Recommendation system module
│
├── app/ # Web-based interface (optional)
│ ├── app.py # Streamlit app (WIP)
│ └── components/ # UI components
│
├── outputs/ # Model outputs and reports
│ ├── model.pkl # Trained fraud detection model
│ └── fraud_metrics.png # Evaluation visualizations
│
├── config.yaml # Configuration file for paths and parameters
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # Open-source license

---

## 💡 Use Cases

| Stakeholder         | Use Case                                                                 |
|---------------------|-------------------------------------------------------------------------|
| Data Scientists     | Build and evaluate ML models for fraud detection and customer analytics |
| Risk Teams          | Identify and prioritize high-risk loan applications                     |
| Marketing Teams     | Target customers with personalized financial product recommendations    |
| Executives          | Monitor customer segments and product penetration                       |

---

## 📊 Dataset Overview

| Column Name        | Description                            |
|--------------------|----------------------------------------|
| customer_id        | Unique ID for each customer            |
| age                | Age of the customer                    |
| gender             | Gender (Male/Female/Other)             |
| marital_status     | Marital status                         |
| income             | Monthly income (numeric)               |
| credit_score       | Creditworthiness score                 |
| account_balance    | Current account balance                |
| loan_amount        | Amount applied for loan                |
| num_transactions   | Transaction count over last 90 days    |
| fraud_risk         | 0 = Not Fraud, 1 = Fraudulent (Target) |

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/customer_360_project.git
cd customer_360_project
Create and activate a virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate         # For Unix/macOS
# OR
venv\Scripts\activate            # For Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the model training pipeline
bash
Copy
Edit
python src/train_model.py
5. (Optional) Launch the web app
bash
Copy
Edit
streamlit run app/app.py
🧪 Model Performance (Fraud Detection)
Metric	Score
Accuracy	93.2%
Precision	89.4%
Recall	85.7%
ROC-AUC	0.94

🔮 Personalization Module
The recommender system uses customer similarity and clustering techniques to suggest relevant products such as:

Credit Cards

Insurance Plans

Investment Products

Loan Upgrades

Techniques used: Cosine Similarity, KMeans Clustering, Rule-based Matching.

📌 Configuration
The config.yaml file allows quick customization of:

File paths

Model hyperparameters

Preprocessing flags

Example:

yaml
Copy
Edit
data_path: data/sample_input.csv
model_path: outputs/model.pkl
random_state: 42
scoring_metric: roc_auc
🧰 Tools & Libraries
Data Processing: pandas, numpy

Modeling: scikit-learn, XGBoost, LightGBM

Visualization: matplotlib, seaborn

Deployment: Streamlit (optional)

Explainability (Upcoming): SHAP

🧠 Future Enhancements
 Streamlit dashboard with upload/predict functionality

 SHAP interpretability for fraud predictions

 Integration with CRM for live recommendations

 API endpoint for real-time fraud scoring

🙋‍♀️ Author
Bhavika

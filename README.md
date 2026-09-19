# 🏦 European Bank Customer Churn Prediction & Risk Scoring

## 🚀 Live Streamlit App

👉 **[Open the Live Streamlit Application]()**

---

## 📌 Project Overview

This project focuses on predicting customer churn for a European bank using Machine Learning and developing an interactive risk-scoring application with Streamlit.

The project combines exploratory data analysis, feature engineering, machine learning classification, customer risk categorization, model explainability, threshold analysis, and what-if scenario simulation.

The final Streamlit application allows users to enter customer characteristics and obtain an estimated churn probability and risk category.

---

## 🎯 Project Objectives

- Predict whether a bank customer is likely to churn.
- Calculate individual customer churn probability.
- Categorize customers into Low, Medium, and High Risk.
- Identify the features influencing customer churn.
- Analyze the trade-off between false positives and false negatives.
- Perform what-if scenario analysis.
- Provide an interactive customer risk calculator.
- Deploy the predictive model as a Streamlit web application.

---

## 📊 Dataset

The project uses the `European_Bank.csv` dataset containing customer-level banking information.

### Dataset characteristics

- **Total Customers:** 10,000
- **Features:** 14
- **Target Variable:** `Exited`
- **Training Customers:** 8,000
- **Testing Customers:** 2,000
- **Churned Customers:** 2,037
- **Overall Churn Rate:** 20.37%

### Important Features

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

Additional engineered features are also used during modelling.

---

## 🧠 Machine Learning Models

The project compares multiple classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Random Forest is used for the customer risk-scoring and explainability components of the application.

---

## 📈 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

The project also evaluates different probability thresholds to understand the relationship between false positives and false negatives.

---

## 🔍 Explainable AI

The project includes model explainability using:

### Feature Importance

Identifies the features that contribute most strongly to the Random Forest model's predictions.

### SHAP Analysis

SHAP is used to investigate how individual features influence model predictions.

### Partial Dependence

Partial dependence analysis is used to examine the model's predicted response to selected numerical features.

---

## ⚠️ Customer Risk Categories

Customers are categorized using their predicted churn probability:

| Churn Probability | Risk Category |
|-------------------|---------------|
| `< 0.30` | 🟢 Low Risk |
| `0.30 – < 0.60` | 🟡 Medium Risk |
| `>= 0.60` | 🔴 High Risk |

The application also uses a configurable decision threshold for converting churn probability into a binary churn prediction.

---

## 🖥️ Streamlit Application

The Streamlit application provides an interactive interface containing:

### 1. Customer Risk Calculator

Users can enter customer characteristics such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership

The application returns:

- Churn Probability
- Risk Category
- Predicted Churn
- Decision Threshold

---

### 2. Churn Probability Distribution

Displays the distribution of predicted churn probabilities across customers.

The visualization also shows the risk boundaries used to categorize customers.

---

### 3. Feature Importance Dashboard

Displays the most influential features used by the Random Forest model to predict customer churn.

---

### 4. What-If Scenario Simulator

Allows users to modify selected customer characteristics and observe how the predicted churn probability changes.

Example scenarios include changes to:

- Active Membership
- Number of Products

The application compares the original customer prediction with the simulated scenario.

---

### 5. Scenario Comparison

Provides a comparison between the original customer and the what-if scenario.

This helps demonstrate how model predictions respond to changes in customer inputs.

---

### 6. Customer Risk Categories

Displays the distribution of customers across:

- High Risk
- Medium Risk
- Low Risk

This provides an overview of the predicted customer-risk population.

---

## 🛠️ Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-learn
- XGBoost

### Explainable AI

- SHAP

### Application Development

- Streamlit

### Deployment

- GitHub
- Streamlit Community Cloud

---

## 📁 Project Structure

```text
European-Bank-Churn/
│
├── European-Bank.py
├── European_Bank.csv
├── requirements.txt
└── README.md

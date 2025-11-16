# Turning Consumer Credit Data Into Actionable Risk Intelligence

## Executive Summary
Consumer lending depends on accurately assessing borrower risk before funding a loan. LendingClub provides rich historical data on borrower profiles, loan terms, and repayment outcomes, but turning this raw data into reliable credit-risk predictions requires a structured machine learning pipeline.

In this project, I built an end-to-end credit risk modeling workflow that:
- Cleans and engineers LendingClub loan data  
- Explores and visualizes key risk drivers (FICO score, income, DTI, interest rate, term, etc.)  
- Trains and compares multiple machine learning models  
- Selects a regularized logistic regression model as the final, production-ready classifier

Using cross-validated AUROC as the main metric, I compared logistic regression (SGD), random forest, k-nearest neighbors (with LDA), and SVM, and evaluated their performance on a held-out test set of the most recent loans.

The final model provides a data-driven estimate of the probability that a loan will charge off, using only information available at listing time—supporting better underwriting and investor decision-making.

## Business Problem
Lenders and marketplace platforms face three central questions:

1. Which borrowers are most likely to default (charge off)?  
2. Which features (credit variables, loan features, income metrics) are most predictive of default?  
3. Can we build a scalable, interpretable model that scores new loans in real time?

This project aims to:
- Transform historic LendingClub data into a model-ready credit-risk dataset  
- Identify top risk drivers across borrower and loan characteristics  
- Build and evaluate machine learning models to predict charge-off probability  

## Methodology

### 1. Data Preparation
- Loaded historical LendingClub loan data (CSV)  
- Cleaned missing values and dropped high-missingness fields  
- Log-transformed skewed variables (income, revolving balance)  
- Encoded categorical variables  
- Constructed binary target: 0 = Fully Paid, 1 = Charged Off  

### 2. Exploratory Data Analysis (EDA)
Explored borrower behavior, loan structure, and financial stability using:
- FICO score  
- Annual income (log)  
- Debt-to-income ratio  
- Earliest credit line year  
- Grade/subgrade  
- Loan amount, interest rate, term  
- Employment length, home ownership  
- Public bankruptcies  

### 3. Feature Selection
Strongest predictors:
- Interest rate  
- Term  
- FICO score  
- Debt-to-income ratio  
- Subgrade  
- Revolving balance  
- Income  

Low-signal features:
- Certain states  
- Rare loan purposes  

### 4. Machine Learning Pipeline
All models implemented using scikit-learn Pipelines including:
- Mean imputation  
- Optional LDA reduction  
- StandardScaler  
- Estimator (SGD Logistic Regression, Random Forest, kNN, SVM)

Models evaluated with 5-fold cross-validation and AUROC scoring.
Final evaluation performed on a **time-based held-out test set**.

## Models Trained & Results

### Cross-Validated AUROC
- Logistic Regression (SGD): **0.7249**  
- Random Forest: **0.7254**  
- kNN (with LDA): **0.6693**  

### Test AUROC (Most Recent Loans)
- Logistic Regression (SGD): **0.7247**  
- Random Forest: **0.7265**  
- kNN: **0.6728**  
- SVM: **0.6345**  

Random Forest slightly outperformed Logistic Regression but Logistic was selected as the final model due to:
- Lower complexity  
- Faster training  
- High interpretability  
- Nearly identical AUROC  

### Final Logistic Regression – Test Classification Report
Class 0 (Fully Paid):  
- Precision: 0.83  
- Recall: 0.98  
- F1: 0.90  

Class 1 (Charged Off):  
- Precision: 0.53  
- Recall: 0.09  
- F1: 0.16  

Overall Accuracy: **0.82**  
Weighted F1: **0.76**  

## Key Insights
- Interest rate, term, FICO, DTI, and subgrade are the strongest predictors  
- Lower-income and higher revolving balance borrowers default more frequently  
- Subgrade provides a strong risk-tier hierarchy  
- AUROC ≈ 0.72 reflects real-world difficulty in credit risk separation  

## Business Impact
This project enables:
- Better lending decisions using predicted risk  
- Ability to reprice risky loans  
- Improved overall portfolio performance  
- More informed investor guidance  

## Next Steps
1. Cost-sensitive optimization  
2. Threshold tuning for better recall of default class  
3. Class-weighting or SMOTE  
4. Gradient boosting (XGBoost/LightGBM)  
5. Deploy scoring API (FastAPI or Lambda)  
6. Streamlit dashboard for real-time scoring  

## Tools & Technologies
- Python, Pandas, NumPy, SciPy  
- Scikit-Learn  
- Matplotlib, Seaborn  
- Jupyter Notebook

## Author
Aurel Sahiti  
Data Science Graduate | Machine Learning & Consumer Credit Risk Analytics  
[GitHub](https://github.com/aurelsahiti) | [LinkedIn](https://linkedin.com/in/aurelsahiti)

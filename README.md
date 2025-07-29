
# Fraud Detection System using XGBoost and Rule-Based Engineering

## Overview

This project presents a hybrid fraud detection system that combines domain-driven rules with machine learning (XGBoost) to identify fraudulent financial transactions in a highly imbalanced dataset of over 6 million entries. The system proactively identifies known and hidden frauds using engineered features and statistical learning.

## Project Objectives

- Clean and prepare a large-scale transaction dataset
- Design fraud-detection rules based on domain logic
- Engineer features that expose suspicious patterns
- Reduce dimensionality using SelectKBest
- Handle outliers and class imbalance
- Build a high-performing XGBoost model
- Evaluate the model using reliable metrics and cross-validation
- Interpret model decisions using feature importance
- Propose real-world fraud prevention strategies

## Dataset

The dataset consists of:
- 6.3 million transaction records
- Key features: transaction type, amount, sender/receiver balances, flags for fraud

Additional label:  
- `fraud_label` – a custom label generated using rule-based logic on top of the given `isFraud` column to capture hidden frauds.

## Key Steps

### 1. Data Cleaning & Preprocessing
- Identified and handled missing values using flag-based imputation
- Outliers in the `amount` column were treated using the IQR method
- Logical flags added for zero balances, missing data, and invalid transactions

### 2. Feature Engineering
New features added:
- `isMerchant`, `hasBalDestInfo`, `errorBalOrg`, `errorBalDes`, `BalChngeRatioOrgin`, `recipientGotMoney`, `isFullTrns`, etc.

### 3. Feature Selection
- Used `SelectKBest` with mutual information to select the most informative features

### 4. Class Imbalance Handling
- Leveraged XGBoost’s `scale_pos_weight` to deal with class imbalance

### 5. Model Building
- Trained an XGBoost Classifier using top features
- Achieved:
  - ROC AUC: 0.9997+ on both train/test
  - High precision, recall, and F1-score

### 6. Model Evaluation
- Confusion matrix, ROC curve, and classification reports used for validation
- 5-fold cross-validation confirmed stability

### 7. Feature Importance
- Visualized and interpreted top predictive features: `errorBalDes`, `errorBalOrg`, `isMerchant`, etc.

## Technologies Used

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost

## Results

- Identified both labeled and hidden fraud cases with near-perfect accuracy
- Balanced performance across precision, recall, and AUC
- Scalable system for real-time or batch fraud detection

## Deployment Suggestions

- Use real-time transaction scoring with risk thresholds
- Integrate feature logic into production APIs
- Schedule retraining and monitor model drift

## Questions Answered

The notebook addresses key analytical and business questions such as:
- What factors predict fraud?
- Are these factors logically sound?
- What prevention strategies can be adopted?
- How can the impact of these strategies be measured?

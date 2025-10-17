# 🎯 Customer Churn Prediction Model

A robust machine learning solution for predicting customer churn using Random Forest classifier with advanced feature selection and hyperparameter optimization.

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

## 📊 Project Overview

This project implements a comprehensive customer churn prediction system that:
- Identifies customers likely to churn using Random Forest algorithm
- Handles class imbalance with SMOTE oversampling
- Selects optimal features automatically
- Tunes hyperparameters for maximum performance
- Provides detailed analytics and visualizations

## 🚀 Key Features

- **Automated Feature Selection**: Selects top 10 most important features
- **Imbalance Handling**: Uses SMOTE for balanced training data
- **Hyperparameter Optimization**: Tests multiple parameter sets manually
- **Threshold Tuning**: Finds optimal classification threshold
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Model Persistence**: Saves trained model for deployment

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **AUC Score** | 0.84+ |
| **F1-Score (Churn)** | 0.75+ |
| **Recall (Churn)** | 0.80+ |
| **Precision (Churn)** | 0.70+ |

## 🛠 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt

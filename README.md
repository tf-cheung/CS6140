# Microsoft Malware Prediction

This repository contains the code for a machine learning project aimed at predicting malware infections on Windows machines using the Microsoft Malware Prediction dataset from Kaggle.

## Files Description

1. **DataPreprocessing.py**
   - This script is responsible for cleaning the dataset and performing feature engineering. It includes handling missing values, encoding categorical variables, and preparing the dataset for model training.

2. **LogisticRegression.py**
   - This script focuses on training a Logistic Regression model. It includes the implementation of the logistic regression algorithm, model training, performance evaluation, and metrics calculation.

3. **RandomForest.py**
   - This script is dedicated to training a RandomForest classifier. It covers the setup, training, and evaluation of the RandomForest model, including the calculation of various performance metrics.

## Dataset

The dataset used for this project is the Microsoft Malware Prediction dataset, which can be downloaded from Kaggle. It includes a wide range of features extracted from Windows machines, such as system configurations, software information, and security settings.

Link to the dataset: [Microsoft Malware Prediction - Kaggle](https://www.kaggle.com/c/microsoft-malware-prediction/data)

## Usage

To use these scripts, first ensure you have downloaded the dataset from Kaggle and have Python installed on your system. Each script can be run independently:

```bash
python DataPreprocessing.py
python LogisticRegression.py
python RandomForest.py

# Diabetes-DataScience-ML-2nd_Part
This project is Phase 2 of a diabetes analysis using a binary dataset from BRFSS 2015, created in Phase 1. It focuses on advanced modeling, predictions, and insights into diabetes and related health factors using machine learning techniques. Phase 1's dataset serves as the foundation for this continuation.

This repository focuses on diabetes prediction using health indicator datasets. It includes data preprocessing, exploratory data analysis (EDA), feature selection, handling imbalanced data, and building machine learning models such as Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN), and Random Forest. Techniques like SMOTE and ANOVA are used for feature optimization.

## Datasets
- 2015 Behavioral Risk Factor Dataset:https://github.com/juheemakwana/Diabetes-DataScience-ML/blob/main/Output_file_diabetes_012_health_indicators_BRFSS2015.csv
- Description: Contains behavioral risk factor data with health indicators for diabetes prediction from the 2015 CDC survey.

## Added Python Code for Part II: Modeling and Advanced Analysis
- The attached Python Notebook file includes advanced EDA, feature engineering, and model building.
- Developed workflows for handling data imbalances and optimizing predictive performance.

## Running Code from Google Colab in Jupyter Lab
- Initially developed in Google Colab, the code was later adapted for Jupyter Lab. The conversion process involved converting the Jupyter Notebook (.ipynb) to a Python script using the command: jupyter nbconvert --to script Diabetes_ML_DataScience_part_II-1.ipynb
- To execute the Python script: python Diabetes_ML_DataScience_part_II-1.py

## Uploaded Output Files
- Cleaned and Processed Datasets:

  - Diabetes Full Dataset (BRFSS 2015 health indicators).

  - Binary classification dataset.

  - Balanced binary classification dataset (50-50 split).

- Files are ready for ML model development and evaluation.

## Key Steps and Features:

### Exploratory Data Analysis (EDA):

- Identified trends, distributions, and correlations in the data.

- Generated visualizations to uncover insights about diabetes prevalence and risk factors.

### Feature Selection and Engineering:

- Conducted Variance Inflation Factor (VIF) analysis to identify multicollinearity.

- Used ANOVA for feature importance and selection.

- Selected optimal features for predictive modeling.

### Data Splitting and Imbalance Handling:

- Split data into training and testing sets.

- Addressed class imbalance using SMOTE and NearMiss techniques.

### Data Scaling:

- Standardized features to ensure consistent model performance.

### Modeling and Evaluation:

- Implemented and evaluated multiple machine learning models, including:

  - Logistic Regression
  
  - Decision Tree
  
  - K-Nearest Neighbors (KNN)
  
  - Random Forest

- Evaluation metrics used:

  - Accuracy
  
  - Precision
  
  - Recall
  
  - F1-score
  
  - ROC-AUC

### New Data Prediction

- The finalized model was used to predict diabetes outcomes on new, unseen data.

- Steps for prediction:

  - Preprocessing: The new data was cleaned and scaled using the same preprocessing pipeline as the training data.
  
  - Feature Selection: Ensured the selected features match those used in the trained model.
  
  - Prediction: Used the trained model to predict the outcomes for the new dataset.

### How to Use This Repository

1. Clone the repository and download the dataset.

2. Run the Python notebooks or scripts in sequence:

- Advanced Analysis and Modeling.

3. Use the processed datasets for machine learning model development and experimentation.

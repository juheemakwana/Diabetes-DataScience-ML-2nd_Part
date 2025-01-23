# app.py
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the models
rf_model = joblib.load('models/random_forest_model.pkl')
lgbm_model = joblib.load('models/lightgbm_model.pkl')
ensemble_model = joblib.load('models/ensemble_model.pkl')  # Load the ensemble model if it contains additional logic
scaler = joblib.load('models/scaler.pkl')  # Optional: Only if scaling is needed

# Define the prediction function
def predict_diabetes(input_data):
    # Preprocess input data
    input_scaled = scaler.transform(input_data) if scaler else input_data

    # Get predictions from both models
    rf_proba = rf_model.predict_proba(input_scaled)[:, 1]
    lgbm_proba = lgbm_model.predict_proba(input_scaled)[:, 1]

    # Combine predictions using the ensemble model if applicable
    if hasattr(ensemble_model, 'predict_proba'):
        ensemble_proba = ensemble_model.predict_proba(input_scaled)[:, 1]
    else:
        # Default ensemble logic using equal weights
        ensemble_proba = (0.5 * rf_proba) + (0.5 * lgbm_proba)

    # Return prediction based on the ensemble probability
    return "Diabetic" if ensemble_proba[0] >= 0.5 else "Non-Diabetic"

# Define the feature columns and prompts
feature_columns = [
    'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth',
    'DiffWalk', 'Age', 'Education', 'Income'
]
prompts = {
    'HighBP': "Enter HighBP (0 for No, 1 for Yes): ",
    'HighChol': "Enter HighChol (0 for No, 1 for Yes): ",
    'BMI': "Enter BMI (numeric): ",
    'Smoker': "Enter Smoker (0 for No, 1 for Yes): ",
    'Stroke': "Enter Stroke (0 for No, 1 for Yes): ",
    'HeartDiseaseorAttack': "Enter HeartDiseaseorAttack (0 for No, 1 for Yes): ",
    'PhysActivity': "Enter PhysActivity (0 for No, 1 for Yes): ",
    'HvyAlcoholConsump': "Enter HvyAlcoholConsump (0 for No, 1 for Yes): ",
    'GenHlth': "Enter GenHlth (1 for Excellent to 5 for Poor): ",
    'MentHlth': "Enter MentHlth (how many days during the past 30 days was your mental health not good?) (0-30 days in month): ",
    'PhysHlth': "Enter PhysHlth (how many days during the past 30 days was your physical health not good) (0-30 days in month): ",
    'DiffWalk': "Enter DiffWalk (serious difficulty walking or climbing stairs) (0 for No, 1 for Yes): ",
    'Age': "Enter Age (numeric): ",
    'Education': "Enter Education (1 for Less than High School to 4 for College): ",
    'Income': "Enter Income (1 for Lowest to 5 for Highest): "
}

@app.route('/')
def home():
    # Pass the feature columns and prompts to the HTML form
    return render_template('index.html', feature_columns=feature_columns, prompts=prompts)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    user_data = {feature: request.form[feature] for feature in feature_columns}

    # Convert input to DataFrame and ensure correct data types
    input_data = pd.DataFrame([user_data])
    input_data = input_data.astype(float)  # Ensure numeric inputs

    # Make prediction
    result = predict_diabetes(input_data)

    # Return the result page with the prediction
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)


import os
import json
import numpy as np
import pandas as pd
import time
from flask import Flask, render_template, request, redirect, url_for
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving and loading the model

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

# Define file paths
base_path = "/Users/abhishekjoshi/Documents/SelfProjects/My_Created_Projects/Clothing_Size_Recommendation/FitEz_Homepage/"
female_AHWBFit_input = os.path.join(base_path, 'Female_AHWBFit.json')
female_AHWBWHIFit_input = os.path.join(base_path, 'Female_AHWBWHIFit.json')
male_AHWWFit_input = os.path.join(base_path, 'Male_AHWWFit.json')
male_AHWCWHIFit_input = os.path.join(base_path, 'Male_AHWCWHIFit.json')

# Function to read JSON files
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            datafile = json.load(file)
        return datafile
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None

# Read data files
female_AHWBFit_jsondata = read_json_file(female_AHWBFit_input)
female_AHWBWHIFit_jsondata = read_json_file(female_AHWBWHIFit_input)
male_AHWWFit_jsondata = read_json_file(male_AHWWFit_input)
male_AHWCWHIFit_jsondata = read_json_file(male_AHWCWHIFit_input)

# Function to preprocess data and train the model
def preprocess_and_train(data, model_path):
    df = pd.DataFrame(data)
    le_Gender = LabelEncoder()
    le_FitPreference = LabelEncoder()
    le_ClothingSize = LabelEncoder()

    df['Gender'] = le_Gender.fit_transform(df['Gender'])
    df['FitPreference'] = le_FitPreference.fit_transform(df['FitPreference'])
    df['ClothingSize'] = le_ClothingSize.fit_transform(df['ClothingSize'])

    X = df.drop('ClothingSize', axis=1)
    y = df['ClothingSize']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and LabelEncoders along with feature columns
    joblib.dump((model, le_Gender, le_FitPreference, le_ClothingSize), model_path)


if not os.path.exists('female_AHWBFit_model.joblib'):
    preprocess_and_train(female_AHWBFit_jsondata,'female_AHWBFit_model.joblib')
if not os.path.exists('female_AHWBWHIFit_model.joblib'):
    preprocess_and_train(female_AHWBWHIFit_jsondata, 'female_AHWBWHIFit_model.joblib')
if not os.path.exists('male_AHWWFit_model.joblib'):    
    preprocess_and_train(male_AHWWFit_jsondata, 'male_AHWWFit_model.joblib')
if not os.path.exists('male_AHWCWHIFit_model.joblib'):
    preprocess_and_train(male_AHWCWHIFit_jsondata, 'male_AHWCWHIFit_model.joblib')

# Function to predict size
def predict_size(model_path, new_data):
    prog_start_time = time.time() # added on 6th June 2024 at 10 PM (EST)
    # Load the model and LabelEncoders along with feature columns
    model, le_Gender, le_FitPreference, le_ClothingSize = joblib.load(model_path)

    new_df = pd.DataFrame(new_data)
    new_df['Gender'] = le_Gender.transform(new_df['Gender'])
    new_df['FitPreference'] = le_FitPreference.transform(new_df['FitPreference'])

    predictions = model.predict(new_df)
    predicted_sizes = le_ClothingSize.inverse_transform(predictions)
    prog_end_time = time.time() # added on 6th June 2024 at 10 PM (EST)
    prog_time = prog_end_time - prog_start_time # added on 6th June 2024 at 10 PM (EST)
    predicted_sizes = [predicted_sizes, prog_time] # added on 6th June 2024 at 10 PM (EST)
    return predicted_sizes



@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            # Get user input
            gender = request.form['gender']
            height = float(request.form['height']) if request.form['height'] else None
            weight = float(request.form['weight']) if request.form['weight'] else None
            age = int(request.form['age']) if request.form['age'] else None
            fit_preference = request.form['fit_preference']
            
            # Process based on gender
            if gender == 'Female':
                bust_size = float(request.form['bust_size']) if request.form['bust_size'] else None
                input_continue = request.form['input_continue']
                if input_continue.lower() == 'yes': # if 'waist_size' in request.form:
                    waist_size = float(request.form['waist_size']) if request.form['waist_size'] else None
                    hip_circumference = float(request.form['hip_circumference']) if request.form['hip_circumference'] else None
                    inseam_length = float(request.form['inseam_length']) if request.form['inseam_length'] else None
                    new_data = {
                        'Age': [age],
                        'Gender': [gender],
                        'Height': [height],
                        'Weight': [weight],
                        'BustSize': [bust_size],
                        'WaistSize': [waist_size],
                        'HipCircumference': [hip_circumference],
                        'InseamLength': [inseam_length],
                        'FitPreference': [fit_preference]
                    }
                    model_path = 'female_AHWBWHIFit_model.joblib'
                else:
                    new_data = {
                        'Age': [age],
                        'Gender': [gender],
                        'Height': [height],
                        'Weight': [weight],
                        'BustSize': [bust_size],
                        'FitPreference': [fit_preference]
                    }
                    model_path = 'female_AHWBFit_model.joblib'
            else:  # Male
                waist_size = float(request.form['waist_size']) if request.form['waist_size'] else None
                input_continue = request.form['input_continue']
                if input_continue.lower() == 'yes': #if 'chest_circumference' in request.form:
                    chest_circumference = float(request.form['chest_circumference']) if request.form['chest_circumference'] else None
                    hip_circumference = float(request.form['hip_circumference']) if request.form['hip_circumference'] else None
                    inseam_length = float(request.form['inseam_length']) if request.form['inseam_length'] else None
                    new_data = {
                        'Age': [age],
                        'Gender': [gender],
                        'Height': [height],
                        'Weight': [weight],
                        'ChestCircumference': [chest_circumference],
                        'WaistSize': [waist_size],
                        'HipCircumference': [hip_circumference],
                        'InseamLength': [inseam_length],
                        'FitPreference': [fit_preference]
                    }
                    model_path = 'male_AHWCWHIFit_model.joblib'
                else:
                    new_data = {
                        'Age': [age],
                        'Gender': [gender],
                        'Height': [height],
                        'Weight': [weight],
                        'WaistSize': [waist_size],
                        'FitPreference': [fit_preference]
                    }
                    model_path = 'male_AHWWFit_model.joblib'

            # Predict Sizes
            predicted_sizes = predict_size(model_path, new_data)
            return render_template('size_predictor.html', predicted_size=predicted_sizes, **request.form) #return render_template('index.html', predicted_size=predicted_sizes[0], **request.form) //Replaced index.html with size-predictor.html
        return render_template('size_predictor.html')
    except Exception as e:
        app.logger.error(f'Error: {e}', exc_info=True)
        return render_template('size_predictor.html', error=str(e)), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
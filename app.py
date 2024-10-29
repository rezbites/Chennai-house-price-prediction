from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.impute import SimpleImputer
import os

app = Flask(__name__)

def load_model_and_data():
    try:
        # Load data and model
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chennai.csv")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
        
        data = pd.read_csv(data_path)
        with open(model_path, "rb") as file:
            model = pickle.load(file)
            
        print("Successfully loaded model and data")
        return model, data
    except Exception as e:
        print(f"Error loading files: {str(e)}")
        return None, None

model, data = load_model_and_data()

@app.route('/')
def index():
    if data is None:
        return render_template('error.html', 
                             error="Error: Could not load required files.")
    
    try:
        # Get unique areas from the dataset
        areas = data['AREA'].unique().tolist()
        # Get unique building types
        buildtypes = data['BUILDTYPE'].unique().tolist()
        
        return render_template('index.html', areas=areas, buildtypes=buildtypes)
    except Exception as e:
        error_message = f"Error processing data: {str(e)}"
        return render_template('error.html', error=error_message)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not loaded."
    
    try:
        # Get input values from form
        int_sqft = float(request.form['int_sqft'])
        n_bedroom = int(request.form['n_bedroom'])
        n_bathroom = int(request.form.get('n_bathroom', 1))
        area = request.form['area']
        buildtype = request.form.get('buildtype', data['BUILDTYPE'].iloc[0])  # Default to first buildtype if not provided
        
        # Calculate total rooms
        n_room = n_bedroom + 2  # Assuming a living room and kitchen
        total_rooms = n_bathroom + n_bedroom + n_room
        
        # Create input dataframe
        input_data = pd.DataFrame([[int_sqft, total_rooms]], columns=['INT_SQFT', 'total_rooms'])
        
        # Add building type and area dummy variables
        buildtype_dummies = pd.get_dummies([buildtype], prefix='BUILDTYPE')
        area_dummies = pd.get_dummies([area], prefix='AREA')
        
        # Combine all features
        input_data = pd.concat([input_data, buildtype_dummies, area_dummies], axis=1)
        
        # Make sure input data has all the columns model expects
        expected_columns = model.feature_names_in_
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0
                
        # Reorder columns to match training data
        input_data = input_data.reindex(columns=expected_columns, fill_value=0)
        
        # Predict price
        price = model.predict(input_data)[0]
        
        return render_template('result.html', 
                             prediction=f"₹ {price:,.2f}",
                             area=area,
                             sqft=int_sqft,
                             bedrooms=n_bedroom,
                             bathrooms=n_bathroom,
                             buildtype=buildtype)
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
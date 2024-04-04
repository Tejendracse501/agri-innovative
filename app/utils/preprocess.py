from flask import Flask, render_template, url_for,request
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def fertilizer_name():
    #FUNCTION to convert categorical values of input_data to numerical values
    def catergorical_to_num(input_data):
        numerical_values = input_data

        # Load the label encoders for Soil_Type and Crop_Type
        soil_type_encoder = LabelEncoder()
        soil_type_encoder.classes_ = np.array(['Black', 'Clayey', 'Loamy', 'Red', 'Sandy'])

        crop_type_encoder = LabelEncoder()
        crop_type_encoder.classes_ = np.array(['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat'])

        # Encode the categorical values
        soil_type_encoded = soil_type_encoder.transform([input_data[3]])[0]
        crop_type_encoded = crop_type_encoder.transform([input_data[4]])[0]

        # Combine numerical and encoded values
        encoded_data = input_data[:3] + [soil_type_encoded, crop_type_encoded] + list(map(int, input_data[-3:]))
        return encoded_data
    
    #loading pickles
    # Loading Standard_Scaler
    with open('models/standard_scalerFR.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Loading the Random Forest Model
    with open('models/fertilizer_recommendation.pkl', 'rb') as file:
        model_loaded = pickle.load(file)

    #get the input values from the form
    input_data=[]
    input_data.append(float(request.form['Temperature']))
    input_data.append(float(request.form['Humidity']))
    input_data.append(float(request.form['Moisture']))
    input_data.append(request.form['Soil Type'])
    input_data.append(request.form['Crop Type'])
    input_data.append(float(request.form['Nitrogen']))
    input_data.append(float(request.form['Potassium']))
    input_data.append(float(request.form['Phosphorous']))

    new_data = [catergorical_to_num(input_data)]
    # Applying Standar_Scaling to new data
    scaled_data = scaler.transform(new_data)  # Use transform, not fit_transform

    # Predicting for new value with the loaded model
    prediction = model_loaded.predict(scaled_data)

    # Print the prediction
    print("Predicted Crop:", prediction)

    # Reversing dictionary of label encodings

    reverse_fertilizer = {0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'}
    predicted_fertilizer = [reverse_fertilizer[prediction[0]]]

    # print("Predicted Fertilizer: "+predicted_fertilizer[0])
    return render_template('fertilizer_recommendation_result.html', recommended=predicted_fertilizer[0])


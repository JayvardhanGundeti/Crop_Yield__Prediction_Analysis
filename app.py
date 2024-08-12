from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

# Flask app

app = Flask(__name__)

# Load preprocessor and model
with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["GET",'POST'])
def predict():
    prediction=None
    if request.method== 'POST':
        # Retrieve input data from the form and convert to appropriate types
        rainfall = request.form['Rainfall']
        humidity = request.form['Humidity']
        state = request.form['states']
        temperature = request.form['temperature']
        phlevel = request.form['phLevel']  # Match this with HTML
        nitrogen = request.form['nitrogen']
        phosphorous = request.form['phosphorous']
        potassium = request.form['potassium']
        soiltype = request.form['soilType']  # Match this with HTML

        # Create a feature array with actual values
        features = np.array([[state, soiltype, nitrogen, phosphorous, potassium, temperature, humidity, phlevel, rainfall]])

        # Define column names
        column_names = ['STATE', 'SOIL_TYPE', 'N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']

        # Create DataFrame with input data
        input_data_df = pd.DataFrame(features, columns=column_names)

        # Transform the input data
        transformed_features = preprocessor.transform(input_data_df)

        # Make prediction
        prediction = model.predict(transformed_features)

        # Render the result on the prediction page
    return render_template('index2.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)

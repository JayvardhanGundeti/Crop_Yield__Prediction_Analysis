import pickle
import numpy as np
import pandas as pd

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
print(preprocessor)
print(model)
features = np.array([['Gujarat', 'Sandy soil', 3, 3, 3, 3, 3, 3, 3]])
        
        # Define column names
column_names = ['STATE', 'SOIL_TYPE', 'N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']
        
        # Create DataFrame with input data
input_data_df = pd.DataFrame(features, columns=column_names)
        
        # Transform the input data
transformed_features = preprocessor.transform(input_data_df)
        
        # Make prediction
prediction = model.predict(transformed_features)
print(prediction)
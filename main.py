# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
# to get the predicted value from the model
from model_files.ml_model import predict_mpg

# Load the Random Forest CLassifier model
with open('./model_files/model.bin', 'rb') as f_in:
        classifier = pickle.load(f_in)
        f_in.close()

app = Flask('mpg_prediction')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Cylinders = int(request.form['Cylinders'])
        Displacement = float(request.form['Displacement'])
        Horsepower = float(request.form['Horsepower'])
        Weight = float(request.form['Weight'])
        Acceleration = float(request.form['Acceleration'])
        Model_Year = int(request.form['Model_Year'])
        Origin = int(request.form['Origin'])

        if Origin == 1:
            country = [1, 2, 3]
        elif Origin == 2:
            country = [2, 1, 3]
        else: 
            country = [3, 1, 2]


        vehicle_config = {
            'Cylinders': [Cylinders, 2, 4],
            'Displacement': [Displacement, 100, 130],
            'Horsepower': [Horsepower, 93, 120],
            'Weight': [Weight, 2500, 3500],
            'Acceleration': [Acceleration, 13, 15],
            'Model Year': [Model_Year, 78, 82],
            'Origin': country
        }
        
        # data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = predict_mpg(vehicle_config, classifier)
        
    return render_template('result.html', prediction=my_prediction[0])

if __name__ == '__main__':
	app.run(debug=True)
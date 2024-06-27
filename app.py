from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))

def diabetes_prediction(input_data):
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    
    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Standardize the input data
    std_data = loaded_scaler.transform(input_data_reshaped)

    # Make the prediction
    prediction = loaded_model.predict(std_data)

    # Return the result
    if prediction[0] == 0:
        return 'The Person is Not Diabetic'
    else:
        return 'The Person is Diabetic'

@app.route('/', methods=['GET', 'POST'])
def index():
    diagnosis = ''
    if request.method == 'POST':
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
        
        if all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            diagnosis = diabetes_prediction(input_data)
        else:
            diagnosis = 'Please provide all the required input values.'

    return render_template('index.html', diagnosis=diagnosis)

if __name__ == '__main__':
    app.run(debug=True)

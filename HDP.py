from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Fetch values from form
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            thalch = float(request.form['thalch'])
            oldpeak = float(request.form['oldpeak'])
            ca = int(request.form['ca'])

            # Chest Pain Type (cp)
            cp_typical = 0
            cp_atypical = 0
            cp_nonanginal = 0
            cp_value = request.form['cp']
            if cp_value == 'typical angina':
                cp_typical = 1
            elif cp_value == 'atypical angina':
                cp_atypical = 1
            elif cp_value == 'non-anginal pain':
                cp_nonanginal = 1

            # Resting ECG (restecg)
            restecg_normal = 0
            restecg_stt = 0
            restecg_value = request.form['restecg']
            if restecg_value == 'normal':
                restecg_normal = 1
            elif restecg_value == 'st-t abnormality':
                restecg_stt = 1

            # Slope
            slope_flat = 0
            slope_up = 0
            slope_value = request.form['slope']
            if slope_value == 'flat':
                slope_flat = 1
            elif slope_value == 'upsloping':
                slope_up = 1

            # Thalassemia
            thal_normal = 0
            thal_reversible = 0
            thal_value = request.form['thal']
            if thal_value == 'normal':
                thal_normal = 1
            elif thal_value == 'reversable defect':
                thal_reversible = 1

            # Arrange features as per model
            input_data = np.array([[
                age, sex, trestbps, chol, thalch, oldpeak, ca,
                cp_atypical, cp_nonanginal, cp_typical,
                restecg_normal, restecg_stt,
                slope_flat, slope_up,
                thal_normal, thal_reversible
            ]])

            # Predict
            prediction = model.predict(input_data)[0]
            result = " Risk of Heart Disease" if prediction == 1 else "No Risk of Heart Disease"

            return render_template('form.html', prediction=result)

        except Exception as e:
            return render_template('form.html', prediction="Error: " + str(e))

if __name__ == '__main__':
    app.run(debug=True)
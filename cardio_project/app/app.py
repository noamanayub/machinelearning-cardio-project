from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import csv

app = Flask(__name__, template_folder='templates', static_folder='static')

# Dynamically load models from the parent directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(BASE_DIR, 'cardio_risk_model.pkl')
preprocessor_path = os.path.join(BASE_DIR, 'cardio_preprocessor.pkl')

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

import pandas as pd

def get_previous_records():
    save_path = os.path.join(BASE_DIR, 'prediction_records.csv')
    if os.path.isfile(save_path):
        df = pd.read_csv(save_path)
        # Show last 10 records, newest first
        return df.tail(10).iloc[::-1].to_dict(orient='records')
    return []

@app.route('/')
def home():
    previous_records = get_previous_records()
    return render_template('index.html', previous_records=previous_records)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_name = request.form.get('user_name', 'Anonymous')
        form_data = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['height']),
            float(request.form['weight']),
            float(request.form['ap_hi']),
            float(request.form['ap_lo']),
            float(request.form['cholesterol']),
            float(request.form['gluc']),
            float(request.form['smoke']),
            float(request.form['alco']),
            float(request.form['active']),
        ]

        # Create derived features
        age = form_data[0]
        height = form_data[2]
        weight = form_data[3]
        ap_hi = form_data[4]
        ap_lo = form_data[5]
        cholesterol = form_data[6]
        gluc = form_data[7]
        smoke = form_data[8]
        active = form_data[10]

        bmi = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo
        map_value = (2 * ap_lo + ap_hi) / 3
        hypertension = int(ap_hi >= 140 or ap_lo >= 90)
        bmi_hypertension = bmi * hypertension
        age_cholesterol = age * cholesterol
        active_smoker = active * smoke
        glucose_pressure = gluc * (ap_hi + ap_lo) / 2
        bmi_age = bmi * age / 100
        hdl_ratio = cholesterol / (ap_hi - 50) if ap_hi > 50 else 0
        arterial_stress = (ap_hi * cholesterol) / (age + 1)
        metabolic_risk = bmi * cholesterol * gluc / 1000
        age_group = '30-40' if age < 40 else '40-50' if age < 50 else '50-60' if age < 60 else '60-70'

        input_data = {
            'age': age,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'bmi': bmi,
            'pulse_pressure': pulse_pressure,
            'map': map_value,
            'hypertension': hypertension,
            'bmi_hypertension': bmi_hypertension,
            'age_cholesterol': age_cholesterol,
            'active_smoker': active_smoker,
            'glucose_pressure': glucose_pressure,
            'bmi_age': bmi_age,
            'hdl_ratio': hdl_ratio,
            'arterial_stress': arterial_stress,
            'metabolic_risk': metabolic_risk,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'age_group': age_group
        }

        df_input = pd.DataFrame([input_data])
        X_transformed = preprocessor.transform(df_input)

        prediction = model.predict(X_transformed)[0]
        is_healthy = prediction == 0
        result = '⚠️ High Risk of Cardiovascular Disease' if prediction == 1 else '✅ Low Risk - Healthy'
        plan = None
        if not is_healthy:
            plan = (
                "<ul>"
                "<li>Maintain a healthy weight and balanced diet (reduce salt, sugar, and saturated fats).</li>"
                "<li>Exercise regularly (at least 30 minutes most days).</li>"
                "<li>Quit smoking and limit alcohol consumption.</li>"
                "<li>Monitor and control blood pressure, cholesterol, and glucose levels.</li>"
                "<li>Manage stress and get regular medical checkups.</li>"
                "</ul>"
            )

        # Save record
        record = {
            'user_name': user_name,
            'age': age,
            'gender': request.form['gender'],
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': request.form['alco'],
            'active': active,
            'prediction': int(prediction),
            'result': result
        }
        save_path = os.path.join(BASE_DIR, 'prediction_records.csv')
        file_exists = os.path.isfile(save_path)
        with open(save_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

        previous_records = get_previous_records()
        return render_template('index.html', result=result, is_healthy=is_healthy, plan=plan, user_name=user_name, previous_records=previous_records)
    except Exception as e:
        previous_records = get_previous_records()
        return render_template('index.html', result=f"Error: {str(e)}", user_name=user_name, previous_records=previous_records)

@app.route('/clear_records')
def clear_records():
    save_path = os.path.join(BASE_DIR, 'prediction_records.csv')
    if os.path.isfile(save_path):
        os.remove(save_path)
    previous_records = []
    return render_template('index.html', previous_records=previous_records, result=None, user_name='', is_healthy=True, plan=None)

@app.route('/delete_selected', methods=['POST'])
def delete_selected():
    save_path = os.path.join(BASE_DIR, 'prediction_records.csv')
    if not os.path.isfile(save_path):
        previous_records = []
        return render_template('index.html', previous_records=previous_records, result=None, user_name='', is_healthy=True, plan=None)
    df = pd.read_csv(save_path)
    delete_indices = request.form.getlist('delete_ids')
    delete_indices = [int(i) for i in delete_indices]
    df = df.reset_index(drop=True)
    df = df.drop(df.index[delete_indices])
    df.to_csv(save_path, index=False)
    previous_records = get_previous_records()
    return render_template('index.html', previous_records=previous_records, result=None, user_name='', is_healthy=True, plan=None)

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS

app = Flask(__name__)
model = joblib.load("medical_cost_model.pkl")
CORS(app) 

@app.route('/')
def home():
    return "Welcome to the Medical Cost Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    age = data['age']
    weight = data['weight']  # in kg
    height = data['height']  # in meters
    bmi = weight / (height ** 2)

    children = data['children']
    sex = 1 if data['sex'] == 'male' else 0
    smoker = 1 if data['smoker'] == 'yes' else 0
    region = data['region']

    region_northwest = 1 if region == 'northwest' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0

    input_data = np.array([[age, bmi, children, sex, smoker,
                            region_northwest, region_southeast, region_southwest]])

    prediction = model.predict(input_data)[0]
    return jsonify({'predicted_charges': round(prediction, 2)})


from flask import Flask, request, jsonify
from flask import render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd


# Load model
model = joblib.load("xgboost_fraud_model.joblib")

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON data: {"features": [[...]]}
        data = request.get_json()
        features = data.get("features")

        if not features:
            return jsonify({"error": "No features provided"}), 400

        # Predict
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        return jsonify({
            "prediction": prediction.tolist(),
            "fraud_probability": prediction_proba[:, 1].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_model.pkl")
model = joblib.load("rf_classifier_3min_final.pkl")

# Expected features
selected_features = [
    'Delta_TP9', 'Beta_AF8', 'Gamma_AF8', 'Gyro_Y', 'Beta_AF7', 'Alpha_TP10',
    'Gamma_AF7', 'Beta_TP10', 'Theta_TP10', 'Theta_TP9', 'Alpha_TP9',
    'Delta_TP10', 'Delta_AF8', 'Gyro_X', 'Alpha_AF7', 'Alpha_AF8',
    'Delta_AF7', 'Beta_TP9', 'Accelerometer_Z', 'Gamma_TP10', 'Gamma_TP9',
    'Accelerometer_X', 'HSI_TP9', 'Theta_AF7', 'Second', 'Minute',
    'Theta_AF8', 'HSI_TP10', 'Gyro_Z', 'Accelerometer_Y', 'RAW_TP9',
    'RAW_TP10', 'RAW_AF8', 'RAW_AF7'
]

@app.route('/')
def home():
    return "EEG Classifier API (Distilled RF, 3-min ketamine)"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check file input
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        # Ensure all expected features are present
        missing_features = [feat for feat in selected_features if feat not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Extract relevant features
        X = df[selected_features]

        # Scale and apply PCA
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        # Predict
        predictions = model.predict(X_pca)
        probabilities = model.predict_proba(X_pca)

        # Format results
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append({
                "predicted_class": int(pred),
                "probabilities": [float(p) for p in prob]
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

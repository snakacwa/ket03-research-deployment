from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import shap
import os

# Initialize Flask app
app = Flask(__name__)

# Load already-trained models
scaler = joblib.load("scaler_ket3.pkl")
pca = joblib.load("pca_model_ket3.pkl")
model = joblib.load("rf_student_model_ket3.pkl")


# Load selected features (before PCA)
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
def index():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        df = pd.read_csv(file)

        # Ensure all expected features are present
        missing = [col for col in selected_features if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Extract features
        X = df[selected_features]
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        # Make predictions
        predictions = model.predict(X_pca)
        probabilities = model.predict_proba(X_pca)

        # SHAP explainability for Tree-based model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_pca)
        pca_components = pca.components_  # shape: (n_pca, n_original_features)

        # Construct results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            shap_vals = shap_values[pred][i]  # SHAP values for predicted class
            # Backproject to original features
            original_feature_shap = np.dot(shap_vals, pca_components)

            original_contrib = sorted(
                zip(selected_features, original_feature_shap),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            # Top 5 real EEG features
            top_5 = [{
                "feature": feat,
                "impact": round(val, 4)
            } for feat, val in original_contrib[:5]]

            results.append({
                "predicted_class": int(pred),
                "probabilities": [round(float(p), 4) for p in prob],
                "top_5_contributors": top_5
            })

        return render_template("form.html", results=results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

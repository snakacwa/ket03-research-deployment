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

def create_shap_plot(explainer, X_pca):
    shap_values = explainer.shap_values(X_pca)
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_pca, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probabilities = None
    shap_image = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                prediction = "Error: No file selected"
                return render_template('form.html', prediction=prediction)

            try:
                try:
                    df = pd.read_csv(file, encoding='utf-8')
                except UnicodeDecodeError:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1')

                # Clean column headers
                df.columns = df.columns.str.strip()
                df.columns = df.columns.str.replace('\n', '', regex=False)

                # Extract 'Minute' and 'Second' from TimeStamp
                if 'TimeStamp' in df.columns:
                    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
                    df['Minute'] = df['TimeStamp'].dt.minute
                    df['Second'] = df['TimeStamp'].dt.second
                else:
                    prediction = "Error: 'TimeStamp' column missing from uploaded file."
                    return render_template('form.html', prediction=prediction)

                # Check for missing EEG features
                missing_cols = [feat for feat in FEATURES if feat not in df.columns]
                if missing_cols:
                    prediction = f"Error: Missing columns: {missing_cols}"
                    return render_template('form.html', prediction=prediction)

                # Preprocess
                input_data = df[FEATURES]
                scaled = scaler.transform(input_data)
                reduced = pca.transform(scaled)

                # Prediction
                pred = model.predict(reduced)[0]
                proba = model.predict_proba(reduced)[0].tolist()

                prediction = "Relaxed" if pred == 0 else "Unconscious"
                probabilities = proba

                # SHAP XAI
                explainer = shap.Explainer(model)
                shap_image = create_shap_plot(explainer, reduced)

            except Exception as e:
                prediction = f"Error processing file: {e}"

        else:
            # Manual input (optional, commented out)
            """
            try:
                input_data = [float(request.form[feat]) for feat in FEATURES]
                df_manual = pd.DataFrame([dict(zip(FEATURES, input_data))])
                scaled = scaler.transform(df_manual)
                reduced = pca.transform(scaled)
                pred = model.predict(reduced)[0]
                proba = model.predict_proba(reduced)[0].tolist()
                prediction = "Relaxed" if pred == 0 else "Unconscious"
                probabilities = proba

                explainer = shap.Explainer(model)
                shap_image = create_shap_plot(explainer, reduced)

            except Exception as e:
                prediction = f"Error: {e}"
            """

    return render_template('form.html',
                           features=FEATURES,
                           prediction=prediction,
                           probabilities=probabilities,
                           shap_image=shap_image)

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)

    try:
        input_df = pd.DataFrame([data])
        missing_cols = [feat for feat in FEATURES if feat not in input_df.columns]
        if missing_cols:
            return jsonify({'error': f'Missing features: {missing_cols}'}), 400

        input_data = input_df[FEATURES]
        scaled = scaler.transform(input_data)
        reduced = pca.transform(scaled)

        prediction = model.predict(reduced)
        prediction_proba = model.predict_proba(reduced)

        return jsonify({
            'prediction': int(prediction[0]),
            'probabilities': prediction_proba[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

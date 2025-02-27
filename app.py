import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
from flask_cors import CORS  # Import CORS
from waitress import serve
import pandas as pd
from scipy.signal import find_peaks

app = Flask(__name__)
CORS(app)  # Enable CORS for the whole app

# Load model and kmeans
model = tf.keras.models.load_model("kmeans_model.keras")
kmeans = joblib.load("kmeans_model.pkl")

def load_and_preprocess(heartbeat_signal):
    """Convert input data to the required format"""
    df = pd.DataFrame({'MLII': heartbeat_signal})
    peaks, _ = find_peaks(df["MLII"], distance=150)
    heartbeats = []
    for peak in peaks:
        start = max(0, peak - 150)
        end = min(len(df), peak + 150)
        heartbeat = df["MLII"][start:end].values
        if len(heartbeat) == 300:
            heartbeats.append(heartbeat)
    heartbeats = np.array(heartbeats).reshape(-1, 300, 1)
    return heartbeats

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        heartbeat_signal = data.get("heartbeat_signal", None)

        if heartbeat_signal is None:
            return jsonify({"error": "No input data provided"}), 400

        heartbeats = load_and_preprocess(heartbeat_signal)

        if heartbeats.size == 0:
            return jsonify({"error": "No heartbeats detected"}), 400

        embeddings = model.predict(heartbeats)
        prediction = kmeans.predict(embeddings).tolist() #needs to be a list for json serialization.

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = os.environ.get("PORT", 80)  # Default to 80 for Render deployment
    try:
        port = int(port)
    except ValueError:
        port = 80  # Default to 80 for Render

    # Start server with Waitress
    serve(app, host="0.0.0.0", port=port)
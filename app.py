import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import joblib  # For loading the KMeans model

# Load the KMeans model
def load_kmeans_model(model_path):
    """
    Load the KMeans model from the specified path.
    """
    try:
        kmeans = joblib.load(model_path)
        print("KMeans model loaded successfully.")
        return kmeans
    except Exception as e:
        print(f"Error loading KMeans model: {e}")
        return None

# Load the TensorFlow/Keras model
def load_tf_model(model_path):
    """
    Load the TensorFlow/Keras model from the specified path.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print("TensorFlow/Keras model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading TensorFlow/Keras model: {e}")
        return None

# Preprocess input data (example function, adjust as needed)
def preprocess_input(data):
    """
    Preprocess the input data to match the model's expected input format.
    """
    # Example: Reshape and normalize the data
    data = np.array(data).reshape(-1, 300, 1)  # Reshape to (batch_size, 300, 1)
    data = data / np.max(data)  # Normalize (example)
    return data

# Run inference using the loaded models
def run_inference(input_data, tf_model, kmeans_model):
    """
    Run inference using the TensorFlow/Keras model and KMeans model.
    """
    # Preprocess the input data
    processed_data = preprocess_input(input_data)

    # Get embeddings from the TensorFlow/Keras model
    embeddings = tf_model.predict(processed_data)

    # Use the KMeans model to predict clusters
    clusters = kmeans_model.predict(embeddings)

    return clusters

# Main function
def main():
    # Paths to the saved models
    tf_model_path = "classification_head.keras"  # Replace with your TensorFlow model path
    kmeans_model_path = "kmeans_model.keras"  # Replace with your KMeans model path

    # Load the models
    tf_model = load_tf_model(tf_model_path)
    kmeans_model = load_kmeans_model(kmeans_model_path)

    if tf_model is None or kmeans_model is None:
        print("Failed to load models. Exiting.")
        return

    # Example input data (replace with your actual data)
    # This should be a list of heartbeats, each of shape (300,)
    input_data = [
        np.random.rand(300).tolist(),  # Example heartbeat 1
        np.random.rand(300).tolist(),  # Example heartbeat 2
    ]

    # Run inference
    clusters = run_inference(input_data, tf_model, kmeans_model)

    # Print the results
    print("Predicted Clusters:", clusters)

if __name__ == "__main__":
    main()
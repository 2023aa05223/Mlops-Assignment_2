import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import fashion_mnist
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
import datetime
import json
import os

# Set MLflow tracking URI - local for this example
mlflow.set_tracking_uri("file:./mlruns")

# Load and preprocess Fashion MNIST dataset
def load_fashion_mnist():
    # Load the Fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape images for CNN
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Split validation set from training data
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=42
    )
    
    # Class names for the Fashion MNIST dataset
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names

# Create a CNN model for Fashion MNIST
def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Calculate distribution statistics for drift detection
def calculate_distribution_stats(data):
    """Calculate statistics about the data distribution for drift detection"""
    # Calculate mean and std for each feature
    means = np.mean(data.reshape(data.shape[0], -1), axis=0)
    stds = np.std(data.reshape(data.shape[0], -1), axis=0)
    
    # Convert numpy types to native Python types for JSON serialization
    return {
        "mean": means.tolist(),
        "std": stds.tolist(),
        "mean_of_means": float(np.mean(means)),
        "mean_of_stds": float(np.mean(stds))
    }

# Detect drift between two datasets
def detect_drift(reference_stats, current_stats, threshold=0.1):
    """
    Detect if there's significant drift between reference and current data
    Returns drift magnitude and boolean indicating if drift exceeds threshold
    """
    # Calculate Wasserstein distance between means
    w_dist_means = wasserstein_distance(
        reference_stats["mean"], 
        current_stats["mean"]
    )
    
    # Calculate Wasserstein distance between standard deviations
    w_dist_stds = wasserstein_distance(
        reference_stats["std"],
        current_stats["std"]
    )
    
    # Overall drift magnitude (weighted sum)
    drift_magnitude = 0.7 * w_dist_means + 0.3 * w_dist_stds
    
    # Check if drift exceeds threshold
    significant_drift = drift_magnitude > threshold
    
    # Convert all values to native Python types for JSON compatibility
    return {
        "drift_magnitude": float(drift_magnitude),
        "significant_drift": bool(significant_drift),
        "w_dist_means": float(w_dist_means),
        "w_dist_stds": float(w_dist_stds),
        "threshold": float(threshold)
    }

# Train model and log with MLflow
def train_and_log_model(model_version="v1", epochs=5, batch_size=128, simulate_drift=False):
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names = load_fashion_mnist()
    
    # If simulating drift, apply transformation to test data
    if simulate_drift:
        # Add noise to simulate data drift
        noise_factor = 0.2
        x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
        x_test = np.clip(x_test, 0., 1.)
    
    # Calculate distribution statistics for reference data (training)
    reference_stats = calculate_distribution_stats(x_train)
    
    # Calculate distribution statistics for current data (test)
    current_stats = calculate_distribution_stats(x_test)
    
    # Detect drift
    drift_results = detect_drift(reference_stats, current_stats)
    
    # Create and train model
    model = create_model()
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"fashion-mnist-{model_version}"):
        # Train the model
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        
        # Log model parameters
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("simulate_drift", simulate_drift)
        
        # Log metrics from training
        for epoch in range(epochs):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Log test metrics
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        
        # Log drift metrics
        mlflow.log_metric("drift_magnitude", drift_results["drift_magnitude"])
        mlflow.log_metric("w_dist_means", drift_results["w_dist_means"])
        mlflow.log_metric("w_dist_stds", drift_results["w_dist_stds"])
        mlflow.log_param("drift_threshold", drift_results["threshold"])
        mlflow.log_param("significant_drift", drift_results["significant_drift"])
        
        # Create JSON-compatible dictionaries for statistics (excluding large arrays)
        stats_for_json = {
            "reference_stats": {
                "mean_of_means": reference_stats["mean_of_means"],
                "mean_of_stds": reference_stats["mean_of_stds"]
            },
            "current_stats": {
                "mean_of_means": current_stats["mean_of_means"],
                "mean_of_stds": current_stats["mean_of_stds"]
            }
        }
        
        # Save stats to JSON files
        with open("stats_summary.json", "w") as f:
            json.dump(stats_for_json, f)
        
        mlflow.log_artifact("stats_summary.json")
        
        # Generate and log confusion matrix
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Plot and save confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        
        # Log the model
        mlflow.tensorflow.log_model(model, "model")
        
        # Return information about the run
        return {
            "model_version": model_version,
            "test_accuracy": float(test_accuracy),
            "drift_detected": bool(drift_results["significant_drift"]),
            "drift_magnitude": float(drift_results["drift_magnitude"]),
            "run_id": mlflow.active_run().info.run_id
        }

# Function to simulate model performance monitoring over time
def simulate_monitoring_over_time(num_versions=3):
    results = []
    
    # Initial model training
    print("Training initial model (v1)...")
    initial_result = train_and_log_model(model_version="v1", simulate_drift=False)
    results.append(initial_result)
    
    # Simulate subsequent versions with increasing drift
    for i in range(2, num_versions + 1):
        print(f"Training model v{i} with simulated drift...")
        # Increase drift simulation for each version
        result = train_and_log_model(
            model_version=f"v{i}", 
            simulate_drift=True
        )
        results.append(result)
        
        # If significant drift detected, recommend retraining
        if result["drift_detected"]:
            print(f"⚠️ Significant drift detected in v{i}! Recommendation: Retrain the model.")
    
    return results

# Run the simulation
if __name__ == "__main__":
    print("Starting model monitoring simulation...")
    mlflow.set_experiment("fashion-mnist-monitoring")
    
    # Run monitoring simulation
    results = simulate_monitoring_over_time(num_versions=3)
    
    # Print summary of results
    print("\n=== MODEL MONITORING SUMMARY ===")
    for result in results:
        drift_status = "SIGNIFICANT DRIFT DETECTED ⚠️" if result["drift_detected"] else "No significant drift"
        print(f"Model {result['model_version']}: Accuracy = {result['test_accuracy']:.4f}, Drift = {result['drift_magnitude']:.4f} ({drift_status})")
        print(f"MLflow Run ID: {result['run_id']}")
        print("-" * 50)
    
    print("\nMonitoring complete! You can view detailed results in the MLflow UI:")
    print("Run 'mlflow ui' in this directory to view experiment tracking data.")
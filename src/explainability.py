import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List, Tuple
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
EPOCHS = 5
BATCH_SIZE = 32
OUTPUT_DIR = "explainability_reports"
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
FEATURE_IMPORTANCE_FILE = os.path.join(OUTPUT_DIR, "feature_importance.png")
PERMUTATION_IMPORTANCE_FILE = os.path.join(OUTPUT_DIR, "permutation_importance.png")

def load_and_preprocess_data():
    """
    Load and preprocess the Fashion MNIST dataset.
    Returns:
        x_train_scaled, y_train, x_test_scaled, y_test: Preprocessed training and testing data.
    """
    print("Loading the Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}")

    # Display class distribution
    print("Analyzing class distribution in the training data...")
    train_class_counts = Counter(y_train)
    for class_id, count in train_class_counts.items():
        print(f"Class {class_id} ({CLASS_NAMES[class_id]}): {count} samples")

    print("Analyzing class distribution in the testing data...")
    test_class_counts = Counter(y_test)
    for class_id, count in test_class_counts.items():
        print(f"Class {class_id} ({CLASS_NAMES[class_id]}): {count} samples")

    # Normalize the data
    print("Normalizing the data...")
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Flatten images
    print("Flattening the images...")
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    print(f"Flattened training data shape: {x_train_flat.shape}")
    print(f"Flattened testing data shape: {x_test_flat.shape}")

    # Standardize features
    print("Standardizing the features...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)
    print("Data preprocessing completed.")

    return x_train_scaled, y_train, x_test_scaled, y_test

def build_model(input_shape):
    """
    Build and compile a simple neural network classifier.
    Args:
        input_shape: Shape of the input data.
    Returns:
        Compiled Keras model.
    """
    print("Building the neural network model...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model built and compiled.")
    return model

def explain_instance_with_lime(explainer, instance, model, num_features=10):
    """
    Generate LIME explanation for a given instance.
    Args:
        explainer: LimeTabularExplainer object.
        instance: Instance to explain.
        model: Trained model.
        num_features: Number of features to include in the explanation.
    Returns:
        LIME explanation object.
    """
    print("Generating LIME explanation for the selected instance...")
    return explainer.explain_instance(instance.flatten(), model.predict, num_features=num_features)

def save_explanation(exp, output_dir, filename="lime_explanation.html"):
    """
    Save LIME explanation to an HTML file.
    Args:
        exp: LIME explanation object.
        output_dir: Directory to save the file.
        filename: Name of the HTML file.
    """
    print("Saving the LIME explanation to an HTML file...")
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, filename)
    exp.save_to_file(html_path)
    print(f"LIME explanation saved to {html_path}")

# Function to create output directory
def create_output_directory(directory: str) -> None:
    """Creates the output directory if it doesn't exist."""
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Output directory created: {directory}")
    except OSError as e:
        logging.error(f"Failed to create output directory: {e}")
        raise

# Function to plot feature importance
def plot_feature_importance(importances: np.ndarray, feature_names: List[str], output_path: str) -> None:
    """Plots and saves feature importance."""
    try:
        sorted_indices = np.argsort(importances)[::-1]
        sorted_importances = importances[sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, sorted_importances, color="skyblue")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Feature importance plot saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to plot feature importance: {e}")
        raise

# Function to calculate and log permutation importance
def calculate_permutation_importance(
    model, X: pd.DataFrame, y: pd.Series, output_path: str
) -> None:
    """Calculates and logs permutation importance."""
    try:
        logging.info("Calculating permutation importance...")
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        sorted_indices = result.importances_mean.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(X.columns[sorted_indices], result.importances_mean[sorted_indices], color="lightcoral")
        plt.xlabel("Mean Permutation Importance")
        plt.ylabel("Feature")
        plt.title("Permutation Importance")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Permutation importance plot saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to calculate permutation importance: {e}")
        raise

# Function to log explainability summary
def log_explainability_summary(feature_importance_path: str, permutation_importance_path: str) -> None:
    """Logs a summary of the explainability process."""
    logging.info("\n=== Explainability Summary ===")
    logging.info("1. Feature Importance:")
    logging.info(f"   - Feature Importance Plot: {feature_importance_path}")
    logging.info("     - Visualizes the importance of each feature in the model.")
    logging.info("2. Permutation Importance:")
    logging.info(f"   - Permutation Importance Plot: {permutation_importance_path}")
    logging.info("     - Shows the impact of each feature on model predictions.")

def main():
    # Load and preprocess data
    x_train_scaled, y_train, x_test_scaled, y_test = load_and_preprocess_data()

    # Build and train the model
    model = build_model(x_train_scaled.shape[1])
    print("Training the model...")
    model.fit(x_train_scaled, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test_scaled, y_test))
    print("Model training completed.")

    # Explainability using LIME
    print("Initializing the LIME explainer...")
    explainer = LimeTabularExplainer(
        x_train_scaled,
        feature_names=[f'pixel_{i}' for i in range(x_train_scaled.shape[1])],
        class_names=CLASS_NAMES,
        mode='classification'
    )
    print("LIME explainer initialized.")

    instance = x_test_scaled[0].reshape(1, -1)
    print(f"Explaining the first test instance: {instance.shape}")
    exp = explain_instance_with_lime(explainer, instance, model)

    # Save explanation
    save_explanation(exp, OUTPUT_DIR)

    # Load dataset (example: using a synthetic dataset for demonstration)
    logging.info("Loading dataset...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    target = pd.Series(y, name="Target")

    # Split dataset into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

    # Train a model (example: Random Forest)
    logging.info("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Create output directory
    create_output_directory(OUTPUT_DIR)

    # Plot feature importance
    logging.info("Generating feature importance plot...")
    feature_importances = model.feature_importances_
    plot_feature_importance(feature_importances, feature_names, FEATURE_IMPORTANCE_FILE)

    # Calculate and plot permutation importance
    logging.info("Generating permutation importance plot...")
    calculate_permutation_importance(model, X_test, y_test, PERMUTATION_IMPORTANCE_FILE)

    # Log explainability summary
    log_explainability_summary(FEATURE_IMPORTANCE_FILE, PERMUTATION_IMPORTANCE_FILE)

    logging.info("Explainability analysis completed successfully. All reports and visualizations are saved in the 'explainability_reports' directory.")

if __name__ == "__main__":
    main()

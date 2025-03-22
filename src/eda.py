import os
import logging
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import sweetviz as sv
from sklearn.decomposition import PCA
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
OUTPUT_DIR = "reports"
PCA_COMPONENTS = 100
CLASS_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
PANDAS_PROFILING_FILE = os.path.join(OUTPUT_DIR, "fashion_mnist_profiling.html")
SWEETVIZ_FILE = os.path.join(OUTPUT_DIR, "fashion_mnist_sweetviz.html")
CLASS_DIST_FILE = os.path.join(OUTPUT_DIR, "class_distribution.png")
HEATMAP_FILE = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")


def create_output_directory(directory: str) -> None:
    """Creates the output directory if it doesn't exist."""
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Output directory created: {directory}")
    except OSError as e:
        logging.error(f"Failed to create output directory: {e}")
        raise


def save_plot(output_path: str) -> None:
    """Saves the current plot to the specified path."""
    try:
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Plot saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
        raise


def plot_class_distribution(labels: np.ndarray, output_path: str) -> None:
    """Generates and saves a class distribution plot."""
    try:
        plt.figure(figsize=(10, 5))
        plt.hist(labels, bins=np.arange(11) - 0.5, edgecolor="black", alpha=0.7)
        plt.xticks(range(10), CLASS_LABELS, rotation=45, ha="right")
        plt.xlabel("Class Label")
        plt.ylabel("Frequency")
        plt.title("Class Distribution in Fashion MNIST")
        save_plot(output_path)
    except Exception as e:
        logging.error(f"Failed to plot class distribution: {e}")
        raise


def plot_correlation_heatmap(dataframe: pd.DataFrame, output_path: str) -> None:
    """Generates and saves a feature correlation heatmap."""
    try:
        correlation_matrix = dataframe.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, cmap="coolwarm", linewidths=0.5, annot=False)
        plt.title("Feature Correlation Heatmap")
        save_plot(output_path)
    except Exception as e:
        logging.error(f"Failed to plot correlation heatmap: {e}")
        raise


def generate_pandas_profiling_report(dataframe: pd.DataFrame, output_path: str) -> None:
    """Generates and saves a Pandas Profiling report."""
    try:
        profile = ProfileReport(dataframe, title="Fashion MNIST EDA Report", minimal=True)
        profile.to_file(output_path)
        logging.info(f"Pandas Profiling Report saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate Pandas Profiling Report: {e}")
        raise


def generate_sweetviz_report(dataframe: pd.DataFrame, output_path: str) -> None:
    """Generates and saves a Sweetviz report."""
    try:
        report = sv.analyze(dataframe)
        report.show_html(output_path)
        logging.info(f"Sweetviz Report saved: {output_path}")
    except Exception as e:
        logging.error(f"Failed to generate Sweetviz Report: {e}")
        raise


def log_dataset_details(x_train: np.ndarray, x_test: np.ndarray) -> None:
    """Logs details about the dataset."""
    logging.info("\n=== Dataset Details ===")
    logging.info("   - Dataset Type: Image Classification")
    logging.info("   - Dataset Name: Fashion MNIST")
    logging.info("   - Description: This dataset contains grayscale images of 10 different classes of clothing items.")
    logging.info(f"   - Classes: {', '.join(CLASS_LABELS)}")
    logging.info(f"1. Training Set Shape: {x_train.shape}")
    logging.info(f"   - Number of Samples: {x_train.shape[0]}")
    logging.info(f"   - Image Dimensions: {x_train.shape[1]}x{x_train.shape[2]} (Grayscale)")
    logging.info(f"2. Test Set Shape: {x_test.shape}")
    logging.info(f"   - Number of Samples: {x_test.shape[0]}")
    logging.info(f"   - Image Dimensions: {x_test.shape[1]}x{x_test.shape[2]} (Grayscale)")


def log_class_distribution(y_train: np.ndarray) -> None:
    """Logs the class distribution in the training set."""
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    logging.info("\n=== Class Distribution in Training Set ===")
    for cls, count in zip(unique_classes, class_counts):
        logging.info(f"   - Class {cls} ({CLASS_LABELS[cls]}): {count} samples ({(count / y_train.shape[0]) * 100:.2f}%)")


def log_eda_summary(
    x_train: np.ndarray, x_test: np.ndarray, reduced_dimensions: int, missing_values: int
) -> None:
    """Logs a summary of the EDA process."""
    logging.info("\n=== EDA Summary ===")
    logging.info("1. Dataset Details:")
    logging.info(f"   - Training Set Shape: {x_train.shape}")
    logging.info(f"   - Test Set Shape: {x_test.shape}")
    logging.info(f"   - Number of PCA Components: {PCA_COMPONENTS}")
    logging.info(f"   - Reduced Feature Dimensions: {reduced_dimensions}")
    logging.info("2. Reports and Visualizations:")
    logging.info(f"   - Pandas Profiling Report: {PANDAS_PROFILING_FILE}")
    logging.info(f"   - Sweetviz Report: {SWEETVIZ_FILE}")
    logging.info(f"   - Class Distribution Plot: {CLASS_DIST_FILE}")
    logging.info(f"   - Feature Correlation Heatmap: {HEATMAP_FILE}")
    logging.info("3. Missing Values:")
    logging.info(f"   - Total Missing Values: {missing_values}")
    if missing_values > 0:
        logging.warning("     - Warning: The dataset contains missing values.")
    else:
        logging.info("     - No missing values detected.")


def main() -> None:
    """Main function to perform EDA on the Fashion MNIST dataset."""
    # Load Fashion MNIST dataset
    logging.info("Loading Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Flatten images and apply PCA
    logging.info("Flattening images and applying PCA...")
    num_samples = x_train.shape[0]
    x_train_flattened = x_train.reshape(num_samples, -1)
    pca = PCA(n_components=PCA_COMPONENTS)
    x_train_reduced = pca.fit_transform(x_train_flattened)

    # Create a DataFrame with reduced features and labels
    df_reduced = pd.DataFrame(x_train_reduced)
    df_reduced["label"] = y_train

    # Create output directory
    create_output_directory(OUTPUT_DIR)

    # Generate Pandas Profiling Report
    logging.info("Generating Pandas Profiling Report...")
    generate_pandas_profiling_report(df_reduced, PANDAS_PROFILING_FILE)

    # Generate Sweetviz Report
    logging.info("Generating Sweetviz Report...")
    generate_sweetviz_report(df_reduced, SWEETVIZ_FILE)

    # Visualize Class Distribution
    logging.info("Visualizing class distribution...")
    plot_class_distribution(y_train, CLASS_DIST_FILE)

    # Check for missing values
    logging.info("Checking for missing values...")
    missing_values = df_reduced.isnull().sum().sum()
    if missing_values > 0:
        logging.warning(f"The dataset contains {missing_values} missing values.")
    else:
        logging.info("No missing values detected in the dataset.")

    # Generate Feature Correlation Heatmap
    logging.info("Generating feature correlation heatmap...")
    plot_correlation_heatmap(df_reduced, HEATMAP_FILE)

    # Log dataset details
    log_dataset_details(x_train, x_test)

    # Log class distribution
    log_class_distribution(y_train)

    # Log EDA summary
    log_eda_summary(x_train, x_test, x_train_reduced.shape[1], missing_values)

    logging.info("EDA process completed successfully. All reports and visualizations are saved in the 'reports' directory.")


if __name__ == "__main__":
    main()
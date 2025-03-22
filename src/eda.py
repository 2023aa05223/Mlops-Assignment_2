import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import sweetviz as sv
from sklearn.decomposition import PCA

# Constants
OUTPUT_DIR = "reports"
PCA_COMPONENTS = 100

# Function to create output directory
def create_output_directory(directory):
    os.makedirs(directory, exist_ok=True)

# Function to generate class distribution plot
def plot_class_distribution(labels, output_path):
    plt.figure(figsize=(10, 5))
    plt.hist(labels, bins=np.arange(11) - 0.5, edgecolor='black', alpha=0.7)
    plt.xticks(range(10))
    plt.xlabel("Class Label")
    plt.ylabel("Frequency")
    plt.title("Class Distribution in Fashion MNIST")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Function to generate feature correlation heatmap
def plot_correlation_heatmap(dataframe, output_path):
    correlation_matrix = dataframe.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Flatten images and apply PCA
num_samples = x_train.shape[0]
x_train_flattened = x_train.reshape(num_samples, -1)  # Flatten images
pca = PCA(n_components=PCA_COMPONENTS)
x_train_reduced = pca.fit_transform(x_train_flattened)

# Create a DataFrame with reduced features and labels
df_reduced = pd.DataFrame(x_train_reduced)
df_reduced['label'] = y_train

# Create output directory
create_output_directory(OUTPUT_DIR)

# Generate Pandas Profiling Report
profile = ProfileReport(df_reduced, title="Fashion MNIST EDA Report", minimal=True)
profile.to_file(f"{OUTPUT_DIR}/fashion_mnist_profiling.html")

# Generate Sweetviz Report
report = sv.analyze(df_reduced)
report.show_html(f"{OUTPUT_DIR}/fashion_mnist_sweetviz.html")

# Visualize Class Distribution
plot_class_distribution(y_train, f"{OUTPUT_DIR}/class_distribution.png")

# Check for missing values
missing_values = df_reduced.isnull().sum().sum()
print("\nMissing Values Check:")
print(f"   - Total Missing Values in the dataset: {missing_values}")
if missing_values > 0:
    print("   - Warning: The dataset contains missing values. Consider handling them before further analysis.")
else:
    print("   - No missing values detected in the dataset.")

# Generate Feature Correlation Heatmap
print("\nGenerating Feature Correlation Heatmap...")
plot_correlation_heatmap(df_reduced, f"{OUTPUT_DIR}/correlation_heatmap.png")
print(f"   - Heatmap saved at: {OUTPUT_DIR}/correlation_heatmap.png")

# Dataset details
print("\nDataset Details:")
print("   - Dataset Type: Image Classification")
print("   - Dataset Name: Fashion MNIST")
print("   - Description: This dataset contains grayscale images of 10 different classes of clothing items.")
print("   - Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot")
print(f"1. Training Set Shape: {x_train.shape}")
print(f"   - Number of Samples: {x_train.shape[0]}")
print(f"   - Image Dimensions: {x_train.shape[1]}x{x_train.shape[2]} (Grayscale)")
print(f"2. Test Set Shape: {x_test.shape}")
print(f"   - Number of Samples: {x_test.shape[0]}")
print(f"   - Image Dimensions: {x_test.shape[1]}x{x_test.shape[2]} (Grayscale)")
print(f"3. Number of PCA Components Used: {PCA_COMPONENTS}")
print(f"   - Reduced Feature Dimensions: {x_train_reduced.shape[1]}")

# Class distribution statistics
unique_classes, class_counts = np.unique(y_train, return_counts=True)
print("\nClass Distribution in Training Set:")
for cls, count in zip(unique_classes, class_counts):
    print(f"   - Class {cls}: {count} samples ({(count / x_train.shape[0]) * 100:.2f}%)")

# Summary
print("\nEDA Summary:")
print("1. Dataset Details:")
print(f"   - Training Set Shape: {x_train.shape}")
print(f"   - Test Set Shape: {x_test.shape}")
print(f"   - Number of PCA Components: {PCA_COMPONENTS}")
print(f"   - Reduced Feature Dimensions: {x_train_reduced.shape[1]}")
print("2. Class Distribution:")
for cls, count in zip(unique_classes, class_counts):
    print(f"   - Class {cls}: {count} samples ({(count / x_train.shape[0]) * 100:.2f}%)")
print("3. Reports and Visualizations:")
print("   - Pandas Profiling Report: 'fashion_mnist_profiling.html'")
print(f"     - Location: {OUTPUT_DIR}/fashion_mnist_profiling.html")
print("     - Contains detailed statistics, correlations, and warnings.")
print("   - Sweetviz Report: 'fashion_mnist_sweetviz.html'")
print(f"     - Location: {OUTPUT_DIR}/fashion_mnist_sweetviz.html")
print("     - Provides visualizations and comparisons of features.")
print("   - Class Distribution Plot: 'class_distribution.png'")
print(f"     - Location: {OUTPUT_DIR}/class_distribution.png")
print("     - Visualizes the frequency of each class in the dataset.")
print("   - Feature Correlation Heatmap: 'correlation_heatmap.png'")
print(f"     - Location: {OUTPUT_DIR}/correlation_heatmap.png")
print("     - Displays the correlation between features.")
print("4. Missing Values:")
print(f"   - Total Missing Values: {missing_values}")
if missing_values > 0:
    print("     - Warning: The dataset contains missing values.")
else:
    print("     - No missing values detected.")
print("\nEDA process completed successfully. All reports and visualizations are saved in the 'reports' directory.")
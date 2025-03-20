import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
import os

# Note: Using ydata-profiling instead of pandas-profiling
from ydata_profiling import ProfileReport
import sweetviz

# Load Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define class names for the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to convert image data to dataframe format for analysis
def prepare_fashion_mnist_for_eda(X, y, sample_size=None, feature_reduction=True):
    # Flatten the images and create feature names
    n_samples = X.shape[0]
    n_features = X.shape[1] * X.shape[2]
    
    if sample_size and sample_size < n_samples:
        # Take a random sample if specified
        indices = np.random.choice(n_samples, sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        n_samples = sample_size
    
    # Reshape images to (n_samples, n_features)
    X_flat = X.reshape(n_samples, n_features)
    
    # If feature reduction is enabled, reduce feature count
    if feature_reduction:
        # Calculate average pixel values by region (4x4 blocks)
        # This reduces 784 features to 49 features
        X_reduced = []
        for img in X:
            reduced_img = []
            for i in range(0, 28, 4):
                for j in range(0, 28, 4):
                    block = img[i:i+4, j:j+4]
                    reduced_img.append(np.mean(block))
            X_reduced.append(reduced_img)
        X_flat = np.array(X_reduced)
        feature_names = [f'region_{i}' for i in range(X_flat.shape[1])]
    else:
        feature_names = [f'pixel_{i}' for i in range(n_features)]
    
    # Create dataframe
    df = pd.DataFrame(X_flat, columns=feature_names)
    
    # Add class label
    df['class'] = [class_names[label] for label in y]
    
    return df

# Create output directory
output_dir = "fashion_mnist_eda_reports"
os.makedirs(output_dir, exist_ok=True)

# Create two versions of the dataset:
# 1. Full pixels for detailed visualizations
# 2. Reduced feature set for tools that have issues with high dimensionality
print("Preparing data for EDA...")
fashion_df_full = prepare_fashion_mnist_for_eda(X_train, y_train, sample_size=5000, feature_reduction=False)
fashion_df_reduced = prepare_fashion_mnist_for_eda(X_train, y_train, sample_size=5000, feature_reduction=True)

print(f"Full dataset shape: {fashion_df_full.shape}")
print(f"Reduced dataset shape: {fashion_df_reduced.shape}")

# 1. ydata-profiling (formerly pandas-profiling) - using reduced features
print("Generating ydata-profiling report...")
profile = ProfileReport(fashion_df_reduced, title="Fashion MNIST EDA Report", minimal=True)
profile.to_file(f"{output_dir}/fashion_mnist_profiling.html")

# 2. Sweetviz Analysis - explicitly turning off pairwise analysis to avoid the error
print("Generating Sweetviz report...")
sweet_report = sweetviz.analyze(fashion_df_reduced, pairwise_analysis='off')
sweet_report.show_html(f"{output_dir}/fashion_mnist_sweetviz.html", open_browser=False)

# 3. Custom class distribution visualization
plt.figure(figsize=(10, 6))
sns.countplot(y=fashion_df_full['class'], order=fashion_df_full['class'].value_counts().index)
plt.title('Class Distribution in Fashion MNIST Dataset')
plt.tight_layout()
plt.savefig(f"{output_dir}/class_distribution.png")

# 4. Generate mean images per class for visual reference
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(class_names):
    # Get indices for this class
    indices = np.where(y_train == i)[0]
    # Calculate mean image
    mean_image = X_train[indices].mean(axis=0)
    
    plt.subplot(2, 5, i+1)
    plt.imshow(mean_image, cmap='gray')
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}/mean_class_images.png")

# 5. Feature correlation heatmap (for the reduced features)
correlation_matrix = fashion_df_reduced.iloc[:, :-1].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
            square=True, linewidths=.5, annot=False)
plt.title('Correlation Heatmap (Region Features)')
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")

# 6. Class-wise pixel intensity distribution
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(class_names):
    class_data = fashion_df_reduced[fashion_df_reduced['class'] == class_name].iloc[:, :-1]
    plt.subplot(2, 5, i+1)
    sns.kdeplot(class_data.mean(axis=1), fill=True)
    plt.title(f"{class_name} - Pixel Intensity")
    plt.xlabel('Mean Pixel Value')
    plt.ylabel('Density')

plt.tight_layout()
plt.savefig(f"{output_dir}/class_pixel_intensity.png")

# 7. Pixel intensity histograms by class
plt.figure(figsize=(14, 10))
for i, class_name in enumerate(class_names):
    class_indices = np.where(y_train == i)[0][:100]  # Take first 100 samples of this class
    flat_pixels = X_train[class_indices].reshape(-1)
    
    plt.subplot(2, 5, i+1)
    plt.hist(flat_pixels, bins=50, alpha=0.7)
    plt.title(f"{class_name}")
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')

plt.tight_layout()
plt.savefig(f"{output_dir}/class_histograms.png")

# 8. Save class distribution statistics
class_counts = fashion_df_full['class'].value_counts()
class_counts.to_csv(f"{output_dir}/class_distribution.csv")

print(f"\nEDA reports generated successfully in '{output_dir}' directory.")
print("\nAutomated EDA Summary:")
print(f"- ydata-profiling report: {output_dir}/fashion_mnist_profiling.html")
print(f"- Sweetviz report: {output_dir}/fashion_mnist_sweetviz.html")
print(f"- Class distribution plot: {output_dir}/class_distribution.png")
print(f"- Mean class images: {output_dir}/mean_class_images.png")
print(f"- Correlation heatmap: {output_dir}/correlation_heatmap.png")
print(f"- Class-wise pixel intensity distribution: {output_dir}/class_pixel_intensity.png")
print(f"- Class-wise pixel histograms: {output_dir}/class_histograms.png")
print(f"- Class distribution statistics: {output_dir}/class_distribution.csv")
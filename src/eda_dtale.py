import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
import os
import dtale
import dtale.app as dtale_app
dtale_app.JUPYTER_SERVER_PROXY = False

# Load Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define class names for the Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create output directory
output_dir = "fashion_mnist_eda_reports"
os.makedirs(output_dir, exist_ok=True)

# Function to convert image data to dataframe with reduced features
def prepare_fashion_mnist_for_eda(X, y, sample_size=None):
    print("Preparing data for EDA...")
    # Sample the data if requested
    if sample_size and sample_size < len(X):
        indices = np.random.choice(len(X), sample_size, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Extract meaningful features from images instead of using all pixels
    num_samples = X.shape[0]
    
    # Create dataframe with meaningful features
    df = pd.DataFrame({
        'class': [class_names[label] for label in y],
        'class_id': y,
        'mean_pixel': np.mean(X, axis=(1, 2)),
        'std_pixel': np.std(X, axis=(1, 2)),
        'min_pixel': np.min(X, axis=(1, 2)),
        'max_pixel': np.max(X, axis=(1, 2))
    })
    
    # Add more features: quadrant means (dividing image into 4 parts)
    height, width = X.shape[1], X.shape[2]
    h_mid, w_mid = height // 2, width // 2
    
    # Add quadrant mean values as features
    df['top_left_mean'] = np.array([np.mean(img[:h_mid, :w_mid]) for img in X])
    df['top_right_mean'] = np.array([np.mean(img[:h_mid, w_mid:]) for img in X])
    df['bottom_left_mean'] = np.array([np.mean(img[h_mid:, :w_mid]) for img in X])
    df['bottom_right_mean'] = np.array([np.mean(img[h_mid:, w_mid:]) for img in X])
    
    print(f"Created dataframe with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

# Create a more manageable dataset for EDA
df = prepare_fashion_mnist_for_eda(X_train, y_train, sample_size=5000)

# Generate D-Tale report in non-interactive mode
print("Generating D-Tale report...")
# Get D-Tale instance
d = dtale.show(df, ignore_duplicate=True)
# Convert to static report
report_path = f"{output_dir}/dtale_report.html"
d.open_browser()

# Save correlations report to file
print("Generating correlation analysis...")
correlations = d.correlations()
corr_path = f"{output_dir}/dtale_correlations.html"
with open(corr_path, "w") as f:
    f.write(correlations.to_html())
print(f"Saved correlations to {corr_path}")

# Generate additional helpful visualizations
print("Generating additional visualizations...")

# 1. Class Distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(y=df['class'], order=df['class'].value_counts().index)
ax.set_title('Class Distribution in Fashion MNIST Dataset')
plt.tight_layout()
plt.savefig(f"{output_dir}/class_distribution.png")

# 2. Mean images by class (this doesn't rely on the dataframe)
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(class_names):
    indices = np.where(y_train == i)[0]
    mean_image = X_train[indices].mean(axis=0)
    
    plt.subplot(2, 5, i+1)
    plt.imshow(mean_image, cmap='gray')
    plt.title(class_name)
    plt.axis('off')

plt.tight_layout()
plt.savefig(f"{output_dir}/mean_class_images.png")

# 3. Feature correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.drop(columns=['class']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_correlations.png")

# 4. Box plots of pixel intensity by class
plt.figure(figsize=(12, 6))
sns.boxplot(x='class', y='mean_pixel', data=df)
plt.xticks(rotation=45)
plt.title('Mean Pixel Intensity by Class')
plt.tight_layout()
plt.savefig(f"{output_dir}/pixel_intensity_boxplot.png")

# Generate summary report
summary = f"""# Fashion MNIST Dataset Analysis with D-Tale

## Dataset Overview
- Total samples analyzed: {df.shape[0]}
- Features created: {df.shape[1]}
- Number of classes: {len(class_names)}

## Key Insights
- Class distribution is {df['class'].value_counts().min() == df['class'].value_counts().max() and 'balanced' or 'slightly imbalanced'}
- Feature correlations have been analyzed
- Mean pixel values vary significantly across classes

## Generated Files
- D-Tale correlations: {corr_path}
- Class distribution: {output_dir}/class_distribution.png
- Mean class images: {output_dir}/mean_class_images.png
- Feature correlations: {output_dir}/feature_correlations.png
- Pixel intensity by class: {output_dir}/pixel_intensity_boxplot.png
"""

with open(f"{output_dir}/summary_report.md", "w") as f:
    f.write(summary)

print(f"\nEDA analysis completed successfully!")
print(f"All reports and visualizations saved to: {output_dir}/")
print(f"\nSummary of generated files:")
for file in os.listdir(output_dir):
    print(f"- {file}")
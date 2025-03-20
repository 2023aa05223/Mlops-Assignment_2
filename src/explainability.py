import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.datasets import fashion_mnist

# Load the Fashion MNIST dataset
print("Loading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define the class names for better interpretability
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Dataset loaded. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Preprocessing and Feature Engineering
def preprocess_data(X):
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    # Reshape to 2D array for feature engineering
    X_reshaped = X.reshape(X.shape[0], -1)
    
    return X_reshaped

# Extract statistical features
def extract_statistical_features(X):
    features = []
    for img in X:
        # Calculate statistical features for each image
        mean = np.mean(img)
        std = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        median = np.median(img)
        q1 = np.percentile(img, 25)
        q3 = np.percentile(img, 75)
        
        # Calculate horizontal and vertical symmetry
        h_symmetry = np.mean(np.abs(img - np.fliplr(img)))
        v_symmetry = np.mean(np.abs(img - np.flipud(img)))
        
        # Calculate edge density using gradient magnitude
        gradient_y = np.gradient(img, axis=0)
        gradient_x = np.gradient(img, axis=1)
        edge_density = np.mean(np.sqrt(gradient_x**2 + gradient_y**2))
        
        features.append([mean, std, min_val, max_val, median, q1, q3, h_symmetry, v_symmetry, edge_density])
    
    return np.array(features)

print("Preprocessing data and extracting features...")
# Process the data
X_train_processed = preprocess_data(X_train)
X_test_processed = preprocess_data(X_test)

# Extract statistical features
X_train_stats = extract_statistical_features(X_train)
X_test_stats = extract_statistical_features(X_test)
print(f"Statistical features extracted. Shape: {X_train_stats.shape}")

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_stats)
X_test_scaled = scaler.transform(X_test_stats)

# Apply PCA for dimensionality reduction
print("Applying PCA...")
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)
print(f"PCA features extracted. Shape: {X_train_pca.shape}")

# Combine statistical features with PCA components
X_train_combined = np.hstack((X_train_scaled, X_train_pca))
X_test_combined = np.hstack((X_test_scaled, X_test_pca))
print(f"Combined features shape: {X_train_combined.shape}")

# Train a Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_combined, y_train)

# Predict and evaluate
print("Evaluating model...")
y_pred = rf_model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Feature importance
feature_names = ['Mean', 'Std', 'Min', 'Max', 'Median', 'Q1', 'Q3', 'H-Symmetry', 'V-Symmetry', 'Edge-Density'] + [f'PCA_{i+1}' for i in range(5)]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Creating feature importance visualization...")
plt.figure(figsize=(12, 6))
plt.title('Feature Importance')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Skip SHAP due to compatibility issues
print("Skipping SHAP analysis due to compatibility issues. Using direct feature importance instead.")

# Alternative feature importance visualization
print("Creating class-specific feature importance plots...")
plt.figure(figsize=(14, 10))
# Train a separate model for each class using one-vs-rest approach
for i in range(min(5, len(class_names))):
    # Create binary labels for this class
    y_binary = (y_train == i).astype(int)
    
    # Train a model for this class
    class_model = RandomForestClassifier(n_estimators=50, random_state=42)
    class_model.fit(X_train_combined, y_binary)
    
    # Get feature importance
    class_importance = class_model.feature_importances_
    class_indices = np.argsort(class_importance)[::-1]
    
    # Plot
    plt.subplot(2, 3, i+1)
    plt.barh(range(len(class_importance[:10])), 
             class_importance[class_indices[:10]], 
             align='center')
    plt.yticks(range(len(class_importance[:10])), 
               [feature_names[j] for j in class_indices[:10]])
    plt.title(f'Feature Importance for {class_names[i]}')

plt.tight_layout()
plt.savefig('class_feature_importance.png')
plt.close()

# LIME Explainability
print("Setting up LIME explainer...")
# Function to make predictions on images for LIME
def lime_predict_fn(images):
    # Convert images from LIME format to our feature format
    processed_images = []
    for img in images:
        # LIME provides images in float format 0-1, we need to reshape to 28x28
        img_reshaped = img.reshape(28, 28)
        
        # Extract statistical features
        # We need to add a dimension to make it compatible with our extraction function
        img_3d = np.expand_dims(img_reshaped, axis=0)
        stats = extract_statistical_features(img_3d)
        stats_scaled = scaler.transform(stats)
        
        # Extract PCA features
        # Reshape to 1D array for PCA
        img_1d = img_reshaped.reshape(1, -1)
        pca_feats = pca.transform(img_1d)
        
        # Combine features
        combined = np.hstack((stats_scaled, pca_feats))
        processed_images.append(combined[0])
    
    # Convert to numpy array and get predictions
    return rf_model.predict_proba(np.array(processed_images))

# Create a LIME explainer
explainer_lime = lime_image.LimeImageExplainer()

# Select a few images to explain (one from each class if possible)
print("Generating LIME explanations...")
classes_to_explain = min(5, len(class_names))  # Limit to 5 classes for brevity
images_to_explain = []
for i in range(classes_to_explain):
    # Find first image of this class in test set
    class_indices = np.where(y_test == i)[0]
    if len(class_indices) > 0:
        images_to_explain.append(class_indices[0])

# Create a figure for LIME explanations
plt.figure(figsize=(15, 3*len(images_to_explain)))

# Generate LIME explanations for selected images
for idx, img_idx in enumerate(images_to_explain):
    # Get the image
    image = X_test[img_idx]
    
    # Generate explanation
    try:
        explanation = explainer_lime.explain_instance(
            image.astype('double'), 
            lime_predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=500  # Reduced samples for faster computation
        )
        
        # Get the prediction
        pred_class = rf_model.predict([X_test_combined[img_idx]])[0]
        
        # Plot original image
        plt.subplot(len(images_to_explain), 3, idx*3 + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Original: {class_names[y_test[img_idx]]}")
        plt.axis('off')
        
        # Plot explanation mask
        plt.subplot(len(images_to_explain), 3, idx*3 + 2)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(f"LIME Explanation: {class_names[pred_class]}")
        plt.axis('off')
        
        # Plot heatmap
        plt.subplot(len(images_to_explain), 3, idx*3 + 3)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=False, 
            num_features=10, 
            hide_rest=False
        )
        plt.imshow(explanation.segments)
        plt.title("Segmentation Map")
        plt.axis('off')
    
    except Exception as e:
        print(f"Error generating LIME explanation for image {img_idx}: {str(e)}")
        continue

plt.tight_layout()
plt.savefig('lime_explanations.png')
plt.close()

# Feature engineering refinement based on explainability insights
print("Creating refined model based on top features...")
# Get top features from Random Forest importance
top_features_idx = indices[:7]  # Taking top 7 features

# Create a refined model with only top features
X_train_refined = X_train_combined[:, top_features_idx]
X_test_refined = X_test_combined[:, top_features_idx]

refined_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
refined_rf_model.fit(X_train_refined, y_train)

# Evaluate refined model
refined_y_pred = refined_rf_model.predict(X_test_refined)
refined_accuracy = accuracy_score(y_test, refined_y_pred)
print(f"\nRefined Model Accuracy: {refined_accuracy:.4f}")
print("\nRefined Classification Report:")
print(classification_report(y_test, refined_y_pred, target_names=class_names))

# Visualize top features across classes
print("Creating feature distribution visualizations...")
plt.figure(figsize=(12, 8))
for i, feature_idx in enumerate(top_features_idx[:3]):  # Show top 3 features
    plt.subplot(1, 3, i+1)
    for class_idx in range(10):
        class_data = X_train_combined[y_train == class_idx, feature_idx]
        sns.kdeplot(class_data, label=class_names[class_idx])
    plt.title(f'Distribution of {feature_names[feature_idx]}')
    if i == 2:  # Only show legend for the last plot to save space
        plt.legend(loc='upper right', fontsize='x-small')
plt.tight_layout()
plt.savefig('feature_distribution.png')
plt.close()

# Visualize feature correlation
print("Creating feature correlation matrix...")
correlation_matrix = np.corrcoef(X_train_combined.T)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', 
            xticklabels=feature_names, yticklabels=feature_names)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('feature_correlation.png')
plt.close()

# Feature engineering iterative improvement
print("Performing iterative feature engineering...")

# Analyze correlation to remove redundant features
correlation_threshold = 0.8
high_corr_pairs = []

for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        if abs(correlation_matrix[i, j]) > correlation_threshold:
            high_corr_pairs.append((feature_names[i], feature_names[j], correlation_matrix[i, j]))

if high_corr_pairs:
    print("\nHighly correlated feature pairs (correlation > 0.8):")
    for f1, f2, corr in high_corr_pairs:
        print(f"  {f1} and {f2}: {corr:.4f}")

# Identify features with poor class separation
print("\nAnalyzing class separation for each feature...")
separation_scores = []

for i, feature in enumerate(feature_names):
    class_means = []
    class_stds = []
    
    for class_idx in range(10):
        class_data = X_train_combined[y_train == class_idx, i]
        class_means.append(np.mean(class_data))
        class_stds.append(np.std(class_data))
    
    # Calculate coefficient of variation across class means (higher is better for separation)
    cv = np.std(class_means) / np.mean(class_means) if np.mean(class_means) != 0 else 0
    separation_scores.append((feature, cv))

# Sort by separation score
separation_scores.sort(key=lambda x: x[1], reverse=True)

print("\nFeatures ranked by class separation ability:")
for feature, score in separation_scores[:5]:
    print(f"  {feature}: {score:.4f}")

print("\n=== Analysis Summary ===")
print("Top 5 most important features:")
for i in range(5):
    print(f"{i+1}. {feature_names[indices[i]]} (importance: {importances[indices[i]]:.4f})")

print("\nAll visualizations have been saved as PNG files.")
print("1. feature_importance.png - Bar chart of feature importances")
print("2. class_feature_importance.png - Class-specific feature importance")
print("3. lime_explanations.png - LIME explanations for sample images")
print("4. feature_distribution.png - Distribution of top features across classes")
print("5. feature_correlation.png - Feature correlation matrix")

print("\nFinal model comparison:")
print(f"Original model accuracy: {accuracy:.4f}")
print(f"Refined model accuracy: {refined_accuracy:.4f}")
print(f"Change in accuracy: {refined_accuracy - accuracy:.4f}")
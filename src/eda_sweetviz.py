import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
import sweetviz as sv

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Convert to DataFrame
num_samples = x_train.shape[0]
x_train_flattened = x_train.reshape(num_samples, -1)  # Flatten images

df = pd.DataFrame(x_train_flattened)
df['label'] = y_train  # Add target labels

df_sample = df.sample(n=5000, random_state=42)  # Take a subset of 5000 rows

# Generate Pandas Profiling Report
profile = ProfileReport(df_sample, title="Fashion MNIST EDA Report", explorative=True)
profile.to_file("fashion_mnist_pandas_profiling.html")

# Generate Sweetviz Report
report = sv.analyze(df_sample)
report.show_html("fashion_mnist_sweetviz.html")

# Visualizing Class Distribution
plt.figure(figsize=(10,5))
plt.hist(y_train, bins=np.arange(11)-0.5, edgecolor='black', alpha=0.7)
plt.xticks(range(10))
plt.xlabel("Class Label")
plt.ylabel("Frequency")
plt.title("Class Distribution in Fashion MNIST")
plt.show()

# Check for missing values
missing_values = df_sample.isnull().sum().sum()
print(f"Total Missing Values: {missing_values}")

# Feature Correlations
correlation_matrix = df_sample.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

print("EDA reports generated: 'fashion_mnist_pandas_profiling.html' and 'fashion_mnist_sweetviz.html'")
print("Class distribution plot and feature correlation heatmap displayed.")
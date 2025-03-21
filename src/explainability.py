import os
import numpy as np
import pandas as pd
import tensorflow as tf
import lime
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images for simpler modeling
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# Build a simple neural network classifier
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_scaled, y_train, epochs=5, batch_size=32, validation_data=(x_test_scaled, y_test))

# Explainability using LIME
explainer = LimeTabularExplainer(x_train_scaled, feature_names=[f'pixel_{i}' for i in range(x_train_scaled.shape[1])], class_names=[str(i) for i in range(10)], mode='classification')
instance = x_test_scaled[0].reshape(1, -1)
prediction = model.predict(instance)
exp = explainer.explain_instance(instance.flatten(), model.predict, num_features=10)

# Create output directory
output_dir = "reports"
os.makedirs(output_dir, exist_ok=True)

# Visualizing LIME explanation
exp.show_in_notebook()
html_path = output_dir+"/lime_explanation.html"
exp.save_to_file(html_path)

# Print the HTML content to GitHub Actions logs
with open(html_path, "r", encoding="utf-8") as file:
    print(file.read())

# Justification:
# - Normalization ensures pixel values are within a consistent range.
# - Standardization improves convergence and performance of the model.
# - Explainability via LIME helps identify important pixels influencing predictions.
# - Insights can refine preprocessing by removing irrelevant features or focusing on key regions.

plt.savefig(f"{output_dir}/explanation.png")

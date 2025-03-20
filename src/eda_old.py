import os
import pandas as pd
import sweetviz as sv
import kagglehub
import dtale
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

current_path = os.getcwd()
path = kagglehub.dataset_download("zalando-research/fashionmnist")
print("Path to dataset files:", path)

# Load dataset
train_file = os.path.join(path, 'fashion-mnist_train.csv')
test_file = os.path.join(path, 'fashion-mnist_test.csv')
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

train_df.head()
train_df.info()

# Generate Automated EDA Report with Sweetviz
report = sv.analyze(train_df, pairwise_analysis='off')
report.show_html("fashion_mnist_sweetviz.html")

# Generate Automated EDA Report with Pandas Profiling
profile = train_df.profile_report(title="Fashion MNIST EDA Report")
profile.to_file("fashion_mnist_pandas_profiling.html")

# Launch D-Tale for Interactive EDA
dtale_app = dtale.show(train_df)
dtale_app.open_browser()

# Extract features and labels
x_train = train_df.drop(columns=['label']).values.reshape(-1, 28, 28) / 255.0
y_train = train_df['label'].values
x_test = test_df.drop(columns=['label']).values.reshape(-1, 28, 28) / 255.0
y_test = test_df['label'].values

# Visual Summaries
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train, palette='viridis')
plt.title("Class Distribution")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.savefig("class_distribution.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.savefig("missing_values.png")
plt.show()

corr_matrix = train_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.savefig("feature_correlation.png")
plt.show()

# Split data
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_val, y_val):
    model = create_model()
    model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)
    print(classification_report(y_test, y_pred_classes))

model = train_model(x_train.reshape(-1, 784), y_train, x_val.reshape(-1, 784), y_val)
evaluate_model(model, x_test.reshape(-1, 784), y_test)

# MLOps Pipeline Setup
mlflow.set_experiment("FashionMNIST_Experiment")
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

with mlflow.start_run(run_name=run_name) as mlflow_run:
    mlflow_run_id = mlflow_run.info.run_id
    print("MLFlow Run ID: ", mlflow_run_id)

    for epoch in range(10):
        history = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val), verbose=1)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        # Log Metrics
        mlflow.log_metric("epoch", epoch)
        mlflow.log_metric("test_accuracy", test_acc)

    mlflow.tensorflow.log_model(model, "model")

# Explainability with SHAP
explainer = shap.Explainer(model, x_test[:100])
shap_values = explainer(x_test[:10])
shap.summary_plot(shap_values, x_test[:10])

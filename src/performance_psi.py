import tensorflow as tf
import mlflow
import mlflow.tensorflow
import numpy as np
from sklearn.metrics import accuracy_score

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Define a simple CNN model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# MLflow Tracking
mlflow.set_experiment("FashionMNIST_Tracking")
with mlflow.start_run():
    model = create_model()
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    
    # Log metrics
    for epoch, acc in enumerate(history.history['accuracy']):
        mlflow.log_metric("train_accuracy", acc, step=epoch)
    for epoch, val_acc in enumerate(history.history['val_accuracy']):
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
    
    # Log model
    mlflow.tensorflow.log_model(model, "model")
    
    # Log parameters
    mlflow.log_param("epochs", 5)
    mlflow.log_param("optimizer", "adam")
    
    # Evaluate model
    y_pred = np.argmax(model.predict(x_test), axis=1)
    test_accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", test_accuracy)

# Drift Detection using PSI
def calculate_psi(expected, actual, buckets=10):
    def get_bucket_values(data, buckets):
        percentiles = np.linspace(0, 100, buckets + 1)
        bucket_edges = np.percentile(data, percentiles)
        return np.histogram(data, bins=bucket_edges)[0] / len(data)
    
    expected_dist = get_bucket_values(expected, buckets)
    actual_dist = get_bucket_values(actual, buckets)
    psi_values = (expected_dist - actual_dist) * np.log((expected_dist + 1e-8) / (actual_dist + 1e-8))
    return np.sum(psi_values)

# Compute PSI to detect drift
psi_value = calculate_psi(y_train, y_test)
print(f"PSI Value: {psi_value}")
mlflow.log_metric("PSI_value", psi_value)
